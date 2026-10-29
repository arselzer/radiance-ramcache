/* stager.cpp -- see stager.h. */
#include "place/stager.h"
#include "place/dspark_native_scope.h"
#include "place/router_predictor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>

namespace rad {

namespace {

/* The units `take` accepts, grouped by layer, in first-op order. Sets `*clash` to the first layer
 * whose span starts inside the one before it, or 0 when no spans interleave. */
template <class Take>
std::vector<StagedLayer> routed_layers(const Plan& plan, Take take, size_t* clash) {
    std::map<int32_t, StagedLayer> by_layer;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.layer < 0 || mu.first_use_op < 0 || mu.last_use_op < 0 || !take(mu)) continue;
        StagedLayer& L = by_layer[mu.layer];
        L.layer = mu.layer;
        L.units.push_back((int32_t)u);
        L.first_op = L.first_op < 0 ? mu.first_use_op : std::min(L.first_op, mu.first_use_op);
        L.last_op = std::max(L.last_op, mu.last_use_op);
    }
    std::vector<StagedLayer> out;
    for (auto& kv : by_layer) out.push_back(std::move(kv.second));
    std::sort(out.begin(), out.end(),
              [](const StagedLayer& a, const StagedLayer& b) { return a.first_op < b.first_op; });
    *clash = 0;
    for (size_t i = 1; i < out.size() && *clash == 0; ++i)
        if (out[i].first_op <= out[i - 1].last_op) *clash = i;
    return out;
}

/* Demand reads of missed units in flight, over every rank's stager (they share the drive), are
 * PieceReader::process()'s count: the read-ahead starts no read while any is (hx_pf_loop polls it,
 * the piece readers sleep on it). */

/* ..._PREFETCH=pieces: the stagers a finished prediction goes to, every rank's (hx_ra_ready). */
std::mutex g_ra_mu;
std::vector<FileStager*> g_ra_stagers;

}  /* namespace */

PrefillStager::~PrefillStager() {
    for (Place& p : place_) {
        if (p.done) rad_event_destroy(p.done);
        if (p.free) rad_event_destroy(p.free);
    }
    for (RadEvent& g : gate_)
        if (g) rad_event_destroy(g);
    if (moved_t_) rad_event_destroy(moved_t_);
    if (moved_c_) rad_event_destroy(moved_c_);
}

int PrefillStager::init(const Program& prog, const Plan& plan, Mover* mover, char* region,
                        char* low, int64_t bytes, int64_t min_rows) {
    layers_.clear();
    if (!mover || !region || bytes <= 0) return RAD_OK;

    /* A routed layer is its expert units; its op span is their declared uses. The first op must
     * find the copies; after the last (once the block's last pass of rows issued it) nothing reads
     * them. A draft head's layer, numbered past the trunk's, is excluded (see stager.h). */
    const int32_t trunk = prog.drafter.kind != RAD_DRAFT_NONE && prog.meta.n_layers > 0
                              ? (int32_t)prog.meta.n_layers : INT32_MAX;
    size_t c = 0;
    layers_ = routed_layers(
        plan, [&](const MoveUnit& mu) { return mu.expert >= 0 && mu.layer < trunk; }, &c);
    /* Two layers' spans must not overlap: a layer's copies are given back at the next layer's
     * first op, which must be past every op that reads them. */
    if (c > 0) {
        RAD_WARN("prefill staging: routed layers %d and %d are read by interleaved ops (%d..%d "
                 "and %d..%d); staging is off", layers_[c - 1].layer, layers_[c].layer,
                 layers_[c - 1].first_op, layers_[c - 1].last_op, layers_[c].first_op,
                 layers_[c].last_op);
        layers_.clear();
        return RAD_OK;
    }
    if (layers_.empty()) return RAD_OK;

    first_of_.assign(prog.ops.size(), -1);
    last_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i) {
        if ((size_t)layers_[i].first_op < first_of_.size()) first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
        if ((size_t)layers_[i].last_op < last_of_.size()) last_of_[(size_t)layers_[i].last_op] = (int32_t)i;
    }

    mover_ = mover;
    region_ = region;
    low_ = low ? low : region;
    base_ = region;
    want_ = nullptr;
    rebase_ = false;
    ring_ = bytes / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT * 2;
    head_ = 0;
    min_rows_ = min_rows;
    n_early_ = 0;
    ring_order_.clear();
    spare_.clear();
    for (int i = kPlaces - 1; i >= 0; --i) {
        Place& p = place_[i];
        if (rad_event_create_local(&p.done) < 0 || rad_event_create_local(&p.free) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
        spare_.push_back(i);
    }
    for (int i = 0; i < kGates; ++i) {
        if (rad_event_create_local(&gate_[i]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
        gate_li_[i] = -1;
    }
    /* The move's two waits, only where the ring has a second place to move to (rebase). */
    if (low_ != region_ &&
        ((!moved_t_ && rad_event_create_local(&moved_t_) < 0) ||
         (!moved_c_ && rad_event_create_local(&moved_c_) < 0))) {
        layers_.clear();
        return RAD_E_DEVICE;
    }
    /* An empty list makes the backend open the card's DMA queue now, at bring-up, rather than in
     * the first long prompt's first step. */
    return rad_memcpy_list_async(nullptr, 0, 0, mover_->transfer_stream());
}

/* Arms only for a trunk pass whose level has a ring and with at least min_rows tokens (see
 * stager.h). Layers the previous pass staged early are kept if it arms with the ring where they
 * are, and withdrawn if not. */
bool PrefillStager::begin_pass(const RadBatch* b) {
    if (!b || b->draft_pass != 0) return false;
    /* A pass the consumer skip cut (RadBatch::consumer_full) runs a prefix of the layers: declined,
     * its pooled experts read in place, so nothing is copied ahead for a layer it skips. */
    armed_ = want_ != nullptr && enabled() && b->n_tok >= min_rows_ && b->consumer_full == 0;
    staged_ = armed_;
    if (!armed_) {
        withdraw_early();
        return false;
    }
    if (want_ != base_) {
        withdraw_early();
        rebase_ = true;
    }
    first_op_ = true;
    next_ = 0;
    first_li_ = -1;
    for (int32_t& g : gate_li_) g = -1;
    return true;
}

/* Withdraw the early layers' copies. Nothing has read them, so there is no `free` to record. */
void PrefillStager::withdraw_early() {
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired) retire(p);
    }
    n_early_ = 0;
}

/* The host half of giving a place back: its units are published at their pool slots again, for
 * the next residency sync. The device half is `free`, which the copy that laps the place waits
 * on. */
void PrefillStager::retire(Place& p) {
    mover_->unstage_units(p.staged);
    p.staged.clear();
    p.retired = true;
}

/* The oldest place leaves the ring; the mover's stream waits on its last reader, so later copies
 * into it are ordered behind that read. */
int PrefillStager::recycle_front() {
    const int32_t i = ring_order_.front();
    Place& p = place_[i];
    if (p.free_rec) RAD_TRY(rad_event_wait(mover_->transfer_stream(), p.free));
    p.li = -1;
    p.free_rec = p.read = p.retired = false;
    ring_order_.pop_front();
    spare_.push_back(i);
    return RAD_OK;
}

/* Moves the ring (see stager.h) at the first op of a pass whose ring is not where the last armed
 * pass left it. Nothing in it is published any more (begin_pass withdrew it), so it starts over,
 * empty, at its new place. Either place is the other level's buffers, hence two waits: the compute
 * stream does not reach the new step before the copies still going into the old place land, and
 * the first copy into the new place waits for the compute work already queued, which used it. */
int PrefillStager::rebase(RadStream s) {
    rebase_ = false;
    const RadStream t = mover_->transfer_stream();
    RAD_TRY(rad_event_record(moved_t_, t));
    RAD_TRY(rad_event_wait(s, moved_t_));
    RAD_TRY(rad_event_record(moved_c_, s));
    RAD_TRY(rad_event_wait(t, moved_c_));
    while (!ring_order_.empty()) RAD_TRY(recycle_front());
    head_ = 0;
    base_ = want_;
    return RAD_OK;
}

/* A place of `want` bytes at the head of the ring or, with `partial`, all the room the head can
 * reach when `want` does not fit. The head may only overwrite retired places: it stays within one
 * ring of the oldest place still published. *out is -1 when nothing fits. */
int PrefillStager::take(int64_t want, bool partial, int32_t* out) {
    *out = -1;
    int64_t limit = head_ + ring_;
    for (int32_t i : ring_order_)
        if (!place_[i].retired) { limit = place_[i].beg + ring_; break; }
    int64_t len = std::min(want, limit - head_);
    if (len < want && !partial) return RAD_OK;
    if (len < RAD_ALIGN_UNIT) return RAD_OK;
    const int64_t end = head_ + len;
    /* Every place the new one laps leaves the ring first; then, if the table is full, the oldest
     * retired ones (their waits are on reads long finished). */
    while (!ring_order_.empty() && place_[ring_order_.front()].beg + ring_ < end)
        RAD_TRY(recycle_front());
    while (spare_.empty() && !ring_order_.empty() && place_[ring_order_.front()].retired)
        RAD_TRY(recycle_front());
    if (spare_.empty()) return RAD_OK;
    const int32_t i = spare_.back();
    spare_.pop_back();
    Place& p = place_[i];
    p.beg = head_;
    p.end = end;
    *out = i;
    return RAD_OK;
}

/* Stage one layer into the ring, behind `gate` when not null. The layer the run reaches next
 * (`now`) takes whatever room there is up to its need; any other goes only whole and within half
 * the ring, so the next layer always has at least half the ring. *got is kStaged, kNone when
 * nothing of the layer is pooled, or kLater when the ring has no room yet. */
int PrefillStager::stage_layer(int32_t li, bool now, RadEvent gate, int* got) {
    *got = kLater;
    const Layer& L = layers_[(size_t)li];
    int64_t unit = 0;
    const int64_t need = mover_->stage_need(L.units, &unit);
    if (need <= 0) { *got = kNone; return RAD_OK; }
    if (!now && need > ring_ / 2) return RAD_OK;
    /* A place that wraps past the region's end is two stretches, and the first may end with less
     * room than a unit, so it asks for one unit extra. */
    const int64_t want = head_ % ring_ + need > ring_ ? need + unit : need;
    int32_t i = -1;
    RAD_TRY(take(want, now, &i));
    if (i < 0) return RAD_OK;
    Place& p = place_[i];
    p.li = li;
    if (gate) RAD_TRY(rad_event_wait(mover_->transfer_stream(), gate));
    /* A wrapping place continues at the region's start. */
    const int64_t off = p.beg % ring_, len = p.end - p.beg;
    Mover::StageSpan sp[2] = { { base_ + off, std::min(len, ring_ - off) },
                               { base_, len - std::min(len, ring_ - off) } };
    RAD_TRY(mover_->stage_units(L.units, sp, sp[1].cap > 0 ? 2 : 1, nullptr, p.done, &p.staged));
    if (p.staged.empty()) {
        p.li = -1;
        spare_.push_back(i);
        *got = kNone;
        return RAD_OK;
    }
    head_ = p.end;
    ring_order_.push_back(i);
    *got = kStaged;
    return RAD_OK;
}

/* Stage the layers after `cur` as far as the pace and the ring allow. A layer with nothing pooled
 * is skipped; one without room yet, or whose pace layer the run has not reached, stops the walk
 * until a later op. */
int PrefillStager::stage_ahead(int32_t cur, bool* resync) {
    while ((size_t)next_ < layers_.size()) {
        const int32_t g = next_ - kLead;
        RadEvent gate = nullptr;
        if (g >= first_li_) {
            if (gate_li_[g % kGates] != g) break;
            gate = gate_[g % kGates];
        }
        int got = kLater;
        RAD_TRY(stage_layer(next_, next_ <= cur + 1, gate, &got));
        if (got == kLater) break;
        *resync = *resync || got == kStaged;
        ++next_;
    }
    return RAD_OK;
}

int PrefillStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;

    /* At the pass's first op of any kind, stage its first routed layer and the ones after it as
     * far as the pace and ring allow. Layers the previous pass staged early are adopted here. */
    if (first_op_) {
        first_op_ = false;
        if (rebase_) RAD_TRY(rebase(s));
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            next_ = (int32_t)(it - layers_.begin());
            first_li_ = next_;
            if (n_early_ > 0 && next_ != 0) {
                withdraw_early();
                *resync = true;
            }
            next_ = std::max(next_, n_early_);
            n_early_ = 0;
            RAD_TRY(stage_ahead(first_li_ - 1, resync));
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    /* The run's arrival at this layer paces the copies kLead layers on. */
    RAD_TRY(rad_event_record(gate_[li % kGates], s));
    gate_li_[li % kGates] = li;
    /* This layer's copies were published when issued; this one wait orders its ops behind them. */
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired && p.li == li) RAD_TRY(rad_event_wait(s, p.done));
    }
    /* Earlier layers are done: every pass of rows issued their last op and recorded `free`, and
     * layers do not interleave. They are retired here, not at their last op, which may repeat for
     * another pass of rows. */
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired && p.li < li && p.read) retire(p);
    }
    /* Stage further ahead into the freed room. */
    if (next_ <= li) next_ = li + 1;
    const bool was = *resync;
    RAD_TRY(stage_ahead(li, resync));
    *resync = *resync || was;
    return RAD_OK;
}

/* A layer's last op: record `free` behind it so a lapping copy can start as early as possible.
 * The op may repeat for another pass of rows, moving `free` again, which is why the copies are
 * retired only when the next layer starts. */
int PrefillStager::after_op(rad_op h, RadStream s) {
    if ((size_t)h >= last_of_.size()) return RAD_OK;
    const int32_t li = last_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired || p.li != li) continue;
        RAD_TRY(rad_event_record(p.free, s));
        p.free_rec = true;
        p.read = true;
    }
    return RAD_OK;
}

/* Retire everything this pass staged. Then, when the next step continues a prompt, stage its
 * trunk pass's first layers, layer k paced on this pass's layer n - kLead + k, as if the two
 * passes were one. */
int PrefillStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    int rc = RAD_OK;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired) continue;
        const int r = rad_event_record(p.free, s);
        if (r < 0 && rc >= 0) rc = r;
        p.free_rec = r >= 0 || p.free_rec;
        retire(p);
    }
    if (rc < 0 || !next_prefill_) return rc;
    const int32_t n = (int32_t)layers_.size();
    for (int32_t k = 0; k < kLead && k < n; ++k) {
        const int32_t g = n - kLead + k;
        if (g < first_li_ || gate_li_[g % kGates] != g) break;
        int got = kLater;
        RAD_TRY(stage_layer(k, /*now=*/false, gate_[g % kGates], &got));
        if (got == kLater) break;
        n_early_ = k + 1;
    }
    return RAD_OK;
}

/* ================================================================== the file stager */
struct FileStager::HeadOps {
    FileStager& f;
    bool epochs_current() const { return f.mover_ && f.mover_->source_epochs_current(); }
    int drain_copy() const { return rad_stream_sync(f.mover_->transfer_stream()); }
    int drain_compute() const { return f.head_compute_ ? rad_stream_sync(f.head_compute_) : RAD_E_STATE; }
    void withdraw() { f.mover_->unstage_units(f.head_staged_); f.head_staged_.clear(); }
    int stage(const DsparkHeadTailOwner::Budget& b) {
        if (f.mover_->stage_need(f.head_units_) != b.head_bytes) return RAD_E_STATE;
        const RadStream copy = f.mover_->transfer_stream();
        Mover::StageSpan spans[2];
        for (int i=0;i<2;++i) {
            //Valid native zero dummy on the first local-zero layer, plus both reader guards.
            RAD_TRY(rad_memset_async(f.buf_[i],0,b.serving_bytes,copy));
            RAD_TRY(rad_memset_async(f.buf_[i]+b.capacity[size_t(i)]-DsparkHeadTailOwner::guard,
                                    0,DsparkHeadTailOwner::guard,copy));
            spans[i]={reinterpret_cast<char*>(b.tail[size_t(i)]),b.tail_capacity[size_t(i)]};
        }
        RAD_TRY(f.mover_->stage_units(f.head_units_,spans,2,nullptr,f.head_done_,&f.head_staged_));
        auto actual=f.head_staged_,expected=f.head_units_;
        std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
        return actual==expected ? RAD_OK : RAD_E_STATE;
    }
    int wait_done() const { return rad_event_wait(f.head_compute_,f.head_done_); }
};

int FileStager::init_head_cache(const Program& prog,const Plan& plan) {
    if (!dspark_exact_one(std::getenv(kDsparkHeadTailEnv))) return RAD_OK;
    if (!mover_ || plan.dynamic || plan.world_size!=4 || plan.rank!=prog.ctx.rank ||
        prog.ctx.max_tok!=16 || !counts_ || !ep_counts_ || !dspark_native_program_scope(prog) ||
        layers_.empty() || !mover_->source_epochs_current()) return RAD_E_UNSUPPORTED;
    for (size_t i=0;i<layers_.size();++i)
        if (layers_[i].layer<0 || layers_[i].layer>=40 || !ep_layer_eligible_[i]) return RAD_E_UNSUPPORTED;
    int64_t actual_bytes=0;uint64_t generation=0;
    std::array<std::array<bool,32>,3> seen{};
    for (size_t u=0;u<plan.units.size();++u) {
        const auto& mu=plan.units[u];
        if (mu.layer<40 || mu.expert<0) continue;
        if (mu.layer>=43 || mu.expert>=32 || mu.weights.size()!=2 || mu.movable || !mu.host_forced ||
            mu.tier!=Tier::HostPinned || mu.site!=Site::DeviceZeroCopy ||
            mu.bytes!=DsparkHeadTailOwner::pair_bytes || mu.slot_bytes!=DsparkHeadTailOwner::pair_bytes)
            return RAD_E_UNSUPPORTED;
        if (seen[size_t(mu.layer-40)][size_t(mu.expert)]) return RAD_E_STATE;
        seen[size_t(mu.layer-40)][size_t(mu.expert)]=true;
        unsigned slots=0;
        for (auto h:mu.weights) {
            if (!h || size_t(h)>=prog.weights.size()) return RAD_E_STATE;
            const auto& w=prog.weights[h];
            if (!dspark_native_weight_scope(w) || w.file_id!=1 || !w.source_bank_generation ||
                w.decl.group.layer!=mu.layer || w.decl.group.expert!=mu.expert ||
                (slots&(1u<<w.decl.group.slot)))
                return RAD_E_UNSUPPORTED;
            slots|=1u<<w.decl.group.slot;
            if (!generation) generation=w.source_bank_generation;
            if (w.source_bank_generation!=generation) return RAD_E_STATE;
        }
        if (slots!=3) return RAD_E_UNSUPPORTED;
        head_units_.push_back(int32_t(u));actual_bytes+=mu.slot_bytes;
    }
    for (const auto& layer:seen) for (bool present:layer) if (!present) return RAD_E_STATE;
    int64_t largest=0;
    const int64_t available_need=mover_->stage_need(head_units_,&largest);
    if (actual_bytes!=available_need || !head_.configure(
        {reinterpret_cast<uintptr_t>(buf_[0]),reinterpret_cast<uintptr_t>(buf_[1])},
        {cap_,cap_},available_need,largest,int(head_units_.size()))) return RAD_E_NOMEM;
    RAD_TRY(rad_event_create_local(&head_done_));
    const auto& b=head_.budget();
    RAD_INFO("DSpark fixed head tail: rank=%d actual buffers=%lld/%lld bytes, serving=%lld bytes "
             "each, tail capacities=%lld/%lld bytes, staged need=%lld bytes for96 immutable "
             "pinned pairs; same-address copies, no added VRAM",plan.rank,
             (long long)b.capacity[0],(long long)b.capacity[1],(long long)b.serving_bytes,
             (long long)b.tail_capacity[0],(long long)b.tail_capacity[1],(long long)b.head_bytes);
    return RAD_OK;
}

int FileStager::head_revoke() {
    if (!head_.configured() || !head_.pending()) return RAD_OK;
    HeadOps ops{*this};return head_.revoke(ops);
}

int FileStager::head_prepare(RadStream s,bool* resync) {
    const bool before=head_.pending();
    HeadOps ops{*this};
    int rc=RAD_OK;
    if (head_pass_==HeadPass::Unknown) rc=head_.revoke(ops);
    else if (head_pass_==HeadPass::Query || head_.pending()) rc=head_.ensure(ops,reinterpret_cast<uintptr_t>(s));
    if (before!=head_.pending() || head_pass_==HeadPass::Query) *resync=true;
    return rc;
}

int FileStager::shutdown_head_cache() {
    if (!head_.configured() || head_closed_) return RAD_OK;
    RAD_TRY(head_revoke());
    if (head_done_) { rad_event_destroy(head_done_); head_done_=nullptr; }
    const auto& s=head_.stats();
    RAD_INFO("DSpark fixed head tail: %llu copy-ordered publications, %llu revoked generations, "
             "%llu warm pass keeps; checked copy+compute drain before tail reuse; no speed claim",
             (unsigned long long)s.copies,(unsigned long long)s.revokes,(unsigned long long)s.keeps);
    head_closed_=true;return RAD_OK;
}

FileStager::~FileStager() {
    if (hx_mw_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(hx_mw_mu_);
            hx_mw_stop_ = true;
        }
        hx_mw_cv_.notify_all();
        hx_mw_.join();
    }
    //Engine normally closes this before Ctx. Partial-init destruction has the same obligation.
    if (shutdown_head_cache() < 0) std::_Exit(88);
    for (int b = 0; b < 2; ++b) {
        if (done_[b]) rad_event_destroy(done_[b]);
        if (free_[b]) rad_event_destroy(free_[b]);
    }
    if (counts_ev_) rad_event_destroy(counts_ev_);
    if (counts_host_) rad_dev_free(counts_host_, RAD_MEM_HOST_PINNED);
    if (hx_pf_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(hx_pf_mu_);
            hx_pf_stop_ = true;
        }
        hx_pf_cv_.notify_all();
        hx_pf_.join();
        for (PfBatch& b : hx_pf_batch_) b.entries.clear();
        hx_pf_queue_.clear();
    }
    if (hx_ra_) {
        /* No prediction reaches this stager from here on, and no reader touches its units or their
         * buffers once forget returns; then the units give their buffers back to hx_planes_free_. */
        {
            std::lock_guard<std::mutex> lk(g_ra_mu);
            g_ra_stagers.erase(std::remove(g_ra_stagers.begin(), g_ra_stagers.end(), this),
                               g_ra_stagers.end());
        }
        hx_ra_->forget(this);
        std::lock_guard<std::mutex> lk(hx_ra_mu_);
        hx_ra_live_.clear();
        hx_ra_here_.clear();
        hx_miss_ra_.clear();
    }
    hx_drain();
    if (!hx_bg_.empty()) {
        {
            std::lock_guard<std::mutex> lk(hx_bg_mu_);
            hx_bg_stop_ = true;
        }
        hx_bg_cv_.notify_all();
        for (std::thread& t : hx_bg_) t.join();
        hx_bg_.clear();
    }
    hx_side_.clear();   /* its planes back to hx_planes_free_ while the list is there */
    if (trace_fp_) std::fclose(trace_fp_);
    /* The route trace's pinned staging, once its last copies have landed. */
    if (rt_ev_rec_ && rt_ev_) (void)rad_event_sync(rt_ev_);
    if (rt_ev_) rad_event_destroy(rt_ev_);
    for (void* p : { (void*)rt_ids_, (void*)rt_w_, (void*)rt_logits_, (void*)rt_bias_ })
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    if (hx_out_rec_ && hx_out_ev_) (void)rad_event_sync(hx_out_ev_);
    if (hx_in_ev_) rad_event_destroy(hx_in_ev_);
    if (hx_out_ev_) rad_event_destroy(hx_out_ev_);
    for (void* p : { (void*)hx_x_, (void*)hx_idx_, (void*)hx_out_, (void*)hx_chk_ })
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    if (hx_fl_rec_ && hx_fl_ev_) (void)rad_event_sync(hx_fl_ev_);
    if (hx_fl_ev_) rad_event_destroy(hx_fl_ev_);
    if (hx_fl_dev_) rad_dev_free(hx_fl_dev_, RAD_MEM_DEVICE);
    if (hx_fl_pin_) rad_dev_free(hx_fl_pin_, RAD_MEM_HOST_PINNED);
}

int FileStager::init(const Program& prog, const Plan& plan, Mover* mover, CountsRow counts,
                      EpCountsRow ep_counts) {
    //No owner/program switch while temporary addresses or old copy generations survive.
    RAD_TRY(shutdown_head_cache());
    if (head_.configured()) return RAD_E_STATE; //a head-cache FileStager is single-initialization
    layers_.clear();
    trace_requested_ = m1_trace_opt_in(std::getenv(kM1PrefetchTraceEnv));
    ep_counts_ = std::move(ep_counts);
    ep_program_ = &prog;
    ep_stats_ = {};
    if (!mover) return RAD_OK;

    /* A unit declared and never read has no op span and is left out. */
    size_t c = 0;
    layers_ = routed_layers(plan, [](const MoveUnit& mu) {
        return mu.tier == Tier::SSD && mu.access == RAD_ACCESS_CONDITIONAL;
    }, &c);
    if (layers_.empty()) return dspark_exact_one(std::getenv(kDsparkHeadTailEnv)) ? RAD_E_UNSUPPORTED : RAD_OK;
    ep_layer_eligible_.assign(layers_.size(), 0);
    if (ep_counts_ && prog.meta.arch_id && std::strcmp(prog.meta.arch_id, "deepseek41") == 0 &&
        prog.meta.n_layers == 40 && prog.meta.n_embd == 5120 && prog.meta.n_expert == 384 &&
        prog.meta.n_expert_used == 6 && plan.world_size == 4) {
        for (size_t li = 0; li < layers_.size(); ++li) {
            bool valid = true, seen[96] = {};
            for (int32_t u : layers_[li].units) {
                const MoveUnit& mu = plan.units[(size_t)u];
                if (mu.expert < 0 || mu.expert >= 96 || seen[mu.expert] ||
                    mu.weights.size() != 2 || mu.bytes != 18800640) { valid = false; break; }
                seen[mu.expert] = true;
                bool gu = false, dn = false;
                for (rad_weight h : mu.weights) {
                    const WeightInfo& w = prog.weights[h];
                    if (w.decl.shard != RAD_SHARD_NONE || w.layout_tag != "ggml_mxfp4") valid = false;
                    gu = gu || w.stored_bytes == 12533760;
                    dn = dn || w.stored_bytes == 6266880;
                }
                valid = valid && gu && dn;
            }
            ep_layer_eligible_[li] = valid;
        }
    }
    /* A layer's copies stay published until the next one starts, so interleaved layers would need
     * a third buffer for the read-ahead; with no fallback address, such a plan is refused. */
    if (c > 0) {
        RAD_ERR("file stager: routed layers %d and %d are read by interleaved ops (%d..%d and "
                "%d..%d), and a routed layer read from the container is published one layer at a "
                "time", layers_[c - 1].layer, layers_[c].layer, layers_[c - 1].first_op,
                layers_[c - 1].last_op, layers_[c].first_op, layers_[c].last_op);
        layers_.clear();
        return RAD_E_UNSUPPORTED;
    }

    buf_[0] = mover->file_buffer(0);
    buf_[1] = mover->file_buffer(1);
    cap_ = mover->file_buffer_bytes();
    if (!buf_[0] || !buf_[1] || cap_ < plan.file_layer_max) {
        RAD_ERR("file stager: the plan reads %zu routed layers from the container and the mover "
                "holds no buffers for them", layers_.size());
        layers_.clear();
        return RAD_E_STATE;
    }
    first_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i)
        if ((size_t)layers_[i].first_op < first_of_.size())
            first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
    mover_ = mover;
    plan_ = &plan;
    for (int b = 0; b < 2; ++b)
        if (rad_event_create_local(&done_[b]) < 0 || rad_event_create_local(&free_[b]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }

    /* Routed-only reads: one pinned row, as wide as the widest expert numbering a file-tier unit
     * names and never narrower than a router this engine runs (512). A failure here only loses
     * the routed-only path. */
    if (counts) {
        int64_t most = 512;
        for (const Layer& L : layers_)
            for (int32_t u : L.units)
                most = std::max<int64_t>(most, (int64_t)plan.units[(size_t)u].expert + 1);
        counts_host_ = (int32_t*)rad_dev_alloc(most * (int64_t)sizeof(int32_t),
                                               RAD_MEM_HOST_PINNED);
        /* The host waits on it (rad_event_sync), which a local event refuses. */
        if (counts_host_ && rad_event_create(&counts_ev_) >= 0) {
            counts_cap_ = most;
            counts_ = std::move(counts);
        } else {
            if (counts_host_) rad_dev_free(counts_host_, RAD_MEM_HOST_PINNED);
            counts_host_ = nullptr;
            counts_ev_ = nullptr;
        }
    }
    if (const char* td = std::getenv("RADIANCE_EXPERIMENTAL_DS41_ACCESS_TRACE"); td && *td && !trace_fp_) {
        const std::string path = std::string(td) + "/rank" + std::to_string(plan.rank) + ".bin";
        trace_fp_ = std::fopen(path.c_str(), "wb");
        if (!trace_fp_) RAD_WARN("access trace: cannot open %s", path.c_str());
        else if (plan.rank == 0) RAD_INFO("access trace: each routed layer's selected units to %s/rank<r>.bin", td);
    }
    RAD_TRY(tall_init(prog));
    RAD_TRY(init_head_cache(prog, plan));
    RAD_TRY(hx_init(prog));
    return rt_init(prog);
}

/* Arms for every pass: a file-tier unit has no other address, so an unarmed pass would reach its
 * first routed layer with null expert pointers. */
bool FileStager::begin_pass(const RadBatch* b) {
    trace_target_m1_ = b && m1_trace_target(b->phase==RAD_PHASE_DECODE,b->draft_pass!=0,
        b->n_tok,b->n_seq,b->n_spec,b->n_tok_decode,b->n_seq_decode);
    head_first_ = head_.configured();
    head_pass_ = head_.configured() ? dspark_head_tail_pass(b) : HeadPass::Unknown;
    //Unknown/prefill scope gives addresses back BEFORE Ctx's pass-wide residency refresh,
    //even when the architecture issues no op. A bool seam cannot return the error, so retain it.
    head_status_ = head_pass_==HeadPass::Unknown ? head_revoke() : RAD_OK;
    pass_cache_policy_ = native_file_cache_prefill_policy(
        std::getenv(kNativeCachePrefillReadOnlyEnv), b, true);
    armed_ = enabled() && b;
    if (!armed_) return false;
    holds_[0] = holds_[1] = -1;
    partial_[0] = partial_[1] = false;
    pending_[0] = pending_[1] = false;
    cur_ = -1;
    first_op_ = true;
    pass_kind_ = b->draft_pass;
    last_issued_ = -1;
    pass_tok_ = b->n_tok;
    pass_full_ = b->consumer_full > 0 ? b->consumer_full : INT32_MAX;
    hx_defer_li_ = -1;
    ++trace_pass_;
    if (rt_) rt_pass(b);
    routed_ = counts_host_ && b->n_tok > 0 && b->n_tok <= kRoutedRows;
    if (hx_ra_) {   /* predictions for this pass may come; units of earlier ones are stale */
        hx_ra_at_.store((uint64_t)trace_pass_ << 32, std::memory_order_release);
        hx_ra_drop_stale(trace_pass_, -1);
    }
    return true;
}

bool FileStager::will_issue(int32_t li) const {
    if (!moe_runs(layers_[(size_t)li].layer)) return false;   /* a layer the cut skips */
    const auto it = pass_last_.find(pass_kind_);
    return it == pass_last_.end() || layers_[(size_t)li].first_op <= it->second;
}

int FileStager::start(int32_t li) {
    //EVERY full-layer/read-ahead/repeated-slice path acquires the complete buffer here.
    RAD_TRY(head_revoke());
    if (hx_capped_) {
        RAD_ERR("host experts: layer %d needs the card's copy path (no live routing witness, or a "
                "sliced pass), and RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS left it no room; "
                "run without the cap", layers_[(size_t)li].layer);
        return RAD_E_STATE;
    }
    const int b = li & 1;
    if (holds_[b] >= 0) return RAD_E_STATE;     /* released before the next read, by construction */
    const int32_t layer = layers_[(size_t)li].layer;
    const auto cache_policy = layer >= 0 && layer < 40 && (size_t)li < ep_layer_eligible_.size() &&
        ep_layer_eligible_[(size_t)li] ? pass_cache_policy_ : NativeFileCacheReadPolicy::Normal;
    const int rc = mover_->stage_file_units(layers_[(size_t)li].units, buf_[b], cap_,
                                            free_rec_[b] ? free_[b] : nullptr, done_[b],
                                            &staged_[b], cache_policy);
    if (rc < 0) return rc;
    partial_[b] = false;
    pending_[b] = !staged_[b].empty();
    if (!staged_[b].empty()) holds_[b] = li;
    return RAD_OK;
}

/* THE ROUTED-ONLY READ. The layer's router has been issued on `s` ahead of this op and writes its
 * counts into the core's row (Ctx::route_counts_live), so a copy queued here sees this pass's
 * counts. Waiting for it stalls the issuing thread until the card reaches the router, which a
 * pass paced by the drive does anyway; what it buys is reading a pass's few routed experts
 * instead of the whole layer. A row that does not add up to the pass's tokens times top_k is not
 * trusted (a router that counts on another rank's share, or wrote another row), and the caller
 * reads the whole layer. */
int FileStager::start_ep_routed(int32_t li, RadStream s, bool* read) {
    *read = false;
    hx_defer_li_ = -1;   /* a deferral is for the gate_up right after its own layer's start only */
    if (!ep_counts_ || (size_t)li >= ep_layer_eligible_.size() || !ep_layer_eligible_[li])
        return RAD_OK;
    const Layer& L = layers_[(size_t)li];
    EpRouteCounts v;
    if (!ep_counts_(L.layer, &v)) return RAD_OK;
    if (!ep_route_current(v, ep_program_, s, v.generation, L.layer, pass_tok_) ||
        v.rank != plan_->rank) { RAD_TRY(head_revoke()); return RAD_E_STATE; }
    const int b = li & 1;
    if (holds_[b] >= 0 || counts_cap_ < 96) return RAD_E_STATE;
    /* An early-issued layer (..._EARLY_ISSUE) that no gather of it followed: its CPU share was never
     * decided, so whatever read its down rows read the card's stray rows for the CPU's units. */
    if (hx_early_li_ >= 0) {
        const int32_t was = layers_[(size_t)hx_early_li_].layer;
        hx_drain();
        RAD_ERR("host experts: layer %d's card ops went ahead of its counts and none of its gathers "
                "followed", was);
        return RAD_E_STATE;
    }
    /* A layer whose CPU rows were computed and never copied means its gather read the card's
     * stray rows for them. */
    if (hx_li_ >= 0) {
        const bool lost = hx_stage_ == 2;
        const int32_t was = layers_[(size_t)hx_li_].layer;
        hx_drain();
        if (lost) {
            RAD_ERR("host experts: layer %d's CPU rows reached no gather", was);
            return RAD_E_STATE;
        }
    }
    /* With the pool, the pass's token rows and sorted order ride the same wait (hx_fetch). */
    const bool hx = hx_pool_ && !head_.configured() && (size_t)li < hx_ops_.size() &&
                    hx_ops_[(size_t)li].gu >= 0 && hx_ops_[(size_t)li].x;
    /* Never seen at its gate_up (the first pass): nothing to fetch the rows from yet, so the CPU's
     * part waits for the gate_up's operands (hx_deferred) and the card gets aliases for all. */
    const bool defer = !hx && hx_defer_ok_ && hx_pool_ && !head_.configured() &&
                       (size_t)li < hx_ops_.size() && hx_ops_[(size_t)li].gu >= 0 &&
                       pass_tok_ <= hx_miss_rows_;
    /* Same actual producer stream; this host-wait event is the completion proof.
     * The Ctx witness means successfully issued, never completed before this wait.
     * With ..._FETCH_LIST the counts row and the pool's rows go as one ranged copy. */
    int listed = 0;
    if (hx && hx_fl_) RAD_TRY(listed = hx_fetch_list(li, v.row, s));
    if (!listed) {
        RAD_TRY(rad_memcpy_async(counts_host_, v.row, 96 * (int64_t)sizeof(int32_t), s));
        if (hx) RAD_TRY(hx_fetch(li, s));
    }
    RAD_TRY(rad_event_record(counts_ev_, s));
    /* ..._ROUTE_TRACE: rank 0's copies of the router's outputs, behind the counts' event (the
     * choice does not wait for them) and under their own, which rt_layer waits for. Best-effort:
     * copies that cannot be queued leave this layer's record without rank 0's router outputs
     * (counted, warned once); the pass runs on. */
    if (rt_) {
        if (const int st = rt_queue(li, s); st < 0) {
            rt_q_li_ = -1;
            rt_q_logits_ = false;
            if (rt_stats_.copy_failed++ == 0)
                RAD_WARN("route trace: rank 0's copies of routed layer %d's router outputs could not "
                         "be queued (%d); its record goes without them and the pass runs on (the "
                         "shutdown line counts such layers)", (int)li, st);
        }
    }
    /* ..._EARLY_ISSUE (stager.h): capped, every file-tier unit is the CPU's, so the card's aliases
     * are the same whichever units the router picked. Publish them for the whole layer and return
     * without waiting: the card runs the layer's resident experts, the shared expert and the ffn
     * mix while the copies above land and this thread waits for them at the gather. A requested
     * M1 trace keeps the ordinary path, which feeds it. */
    if (hx && hx_early_ && hx_capped_ && !trace_requested_) {
        staged_[b].clear();
        mover_->alias_file_units(L.units, buf_[b], &staged_[b]);
        partial_[b] = true;
        pending_[b] = false;
        if (!staged_[b].empty()) holds_[b] = li;
        hx_early_li_ = li;
        hx_early_seen_ = false;
        hx_early_v_ = v;
        *read = true;
        return RAD_OK;
    }
    const auto t_sync = std::chrono::steady_clock::now();
    RAD_TRY(rad_event_sync(counts_ev_));
    const auto t_counts = std::chrono::steady_clock::now();
    const uint64_t sync_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_counts - t_sync).count();
    ep_stats_.sync_ns += sync_ns;
    if (pass_tok_ == 1) { ++ep_stats_.d_reads; ep_stats_.d_sync_ns += sync_ns; }
    /* Every earlier layer's window copies fed a gather this sync followed on the card: their
     * cache leases go back now, not when a window next comes round (Mover::retire_file_windows). */
    if (hx) RAD_TRY(mover_->retire_file_windows());
    const auto t_retired = std::chrono::steady_clock::now();
    if (hx && hx_pred_ && pass_tok_ == 1 && plan_->rank == 0 && !hx_pred_->pieces() &&
        moe_runs(L.layer + 1))
        hx_pred_->post(trace_pass_, L.layer, hx_x_);
    int64_t total = 0;
    if (!ep_route_counts_valid(v, counts_host_, counts_cap_, &total)) {
        ++ep_stats_.corrupt;
        RAD_TRY(head_revoke());
        return RAD_E_STATE;
    }
    want_.clear(); skip_.clear();
    for (int32_t u : L.units) {
        bool selected = false;
        if (!ep_route_unit_selected(counts_host_, plan_->units[(size_t)u].expert, &selected))
            return RAD_E_STATE;
        (selected ? want_ : skip_).push_back(u);
    }
    /* The access trace takes want_ before the CPU split; its record is written after the stage
     * below, with the stage's drive reads. */
    if (trace_fp_) trace_units_ = want_;
    /* Deferred only if every selected unit can go to the CPU: the card gets aliases for all. */
    if (defer && std::all_of(want_.begin(), want_.end(),
                             [&](int32_t u) { return mover_->host_planes_eligible(u); })) {
        staged_[b].clear();
        mover_->alias_file_units(skip_, buf_[b], &staged_[b]);
        mover_->alias_file_units(want_, buf_[b], &staged_[b]);
        partial_[b] = true;
        pending_[b] = false;
        if (!staged_[b].empty()) holds_[b] = li;
        hx_defer_li_ = want_.empty() ? -1 : li;
        if (trace_fp_ && want_.empty()) trace_layer(L.layer, 0);   /* else hx_deferred writes it */
        if (rt_ && want_.empty()) {   /* the route trace's too: no file-tier pick to choose for */
            rt_units_.clear();
            rt_choice(li, true);
            rt_layer(li);
        }
        ++ep_stats_.reads;
        if (!total) ++ep_stats_.zero_local;
        *read = true;
        return RAD_OK;
    }
    /* The selected units the CPU computes leave want_ and start on the pool before the drive
     * reads below block this thread (host_expert_pool.h). */
    auto t_chosen = t_retired, t_submitted = t_retired, t_posted = t_retired;
    auto t_miss_posted = t_retired;
    /* ..._MISS_POST: misses posted to the readers below are joined on every way out of here;
     * hx_submit_misses_now joins them first on the way that computes them. */
    const HxMissPostJoin miss_post_join{ this };
    if (rt_) rt_units_ = want_;   /* hx_choose takes the CPU's units out of want_ */
    if (hx) {
        hx_choose(li);
        if (rt_) rt_choice(li, true);
        t_chosen = std::chrono::steady_clock::now();
        t_miss_posted = t_chosen;
        /* ..._MISS_POST: the misses' reads start now, ahead of the pool's wake in hx_submit and
         * of the copy path's staging (timed on its own: d_miss_post_ns). */
        if (hx_miss_post_on_ && hx_li_ == li) {
            hx_post_misses();
            t_miss_posted = std::chrono::steady_clock::now();
        }
        if (hx_li_ == li) RAD_TRY(hx_submit());
        t_submitted = std::chrono::steady_clock::now();
        t_posted = t_submitted;
        /* Pieces mode: the prediction of layer L+2 once this layer's CPU work is out, so the
         * predictor's thread does not compete with the pool's wake-up. */
        if (hx_ra_ && hx_pred_ && pass_tok_ == 1 && plan_->rank == 0 && moe_runs(L.layer + 2)) {
            hx_pred_->post(trace_pass_, L.layer, hx_x_);
            t_posted = std::chrono::steady_clock::now();
        }
    } else if (rt_) {
        rt_choice(li, false);   /* the card's copy path takes every selected file-tier unit */
    }
    /* This layer took its read-ahead entries: now the next layer's (predicted from this one). */
    if (hx && hx_pf_.joinable() && pass_tok_ == 1 &&
        (size_t)(li + hx_pf_ahead_) < layers_.size() &&
        moe_runs(layers_[(size_t)(li + hx_pf_ahead_)].layer))
        hx_pf_target(li + hx_pf_ahead_);
    const auto t_stage = std::chrono::steady_clock::now();
    ep_stats_.hx_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_stage - t_counts).count();
    if (hx && pass_tok_ == 1) {
        auto ns = [](auto a, auto b) { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
        ep_stats_.d_retire_ns += ns(t_counts, t_retired);
        ep_stats_.d_choose_ns += ns(t_retired, t_chosen);
        ep_stats_.d_miss_post_ns += ns(t_chosen, t_miss_posted);
        ep_stats_.d_submit_ns += ns(t_miss_posted, t_submitted);
        ep_stats_.d_post_ns += ns(t_submitted, t_posted);
        ep_stats_.d_decide_ns += ns(t_counts, t_stage);
    }
    const uint64_t relaid0 = mover_->stats().file_relaid;
    const int64_t need = int64_t(want_.size()) * DsparkHeadTailOwner::pair_bytes;
    if (head_.pending() && (head_pass_ != HeadPass::Verify ||
        !head_.selected_fits(int64_t(want_.size()), need))) RAD_TRY(head_revoke());
    const int64_t available = head_.pending() ? head_.budget().serving_bytes : cap_;
    const auto trace = m1_trace_scope(trace_requested_,trace_target_m1_,v);
    mover_->prefetch_trace_route(trace,counts_host_,want_.size());
    const int stage_status = mover_->stage_file_units(want_, buf_[b], available,
        free_rec_[b] ? free_[b] : nullptr, done_[b], &staged_[b], NativeFileCacheReadPolicy::Normal,trace);
    mover_->prefetch_trace_stage_done(trace,stage_status);
    if(stage_status<0)return stage_status;
    {
        const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t_stage).count();
        /* The CPU path's drive reads count too (verify reads them on the card path as well). */
        /* (Units read ahead whole were read before the layer.) */
        const uint64_t missed = mover_->stats().file_relaid - relaid0 +
            (hx_li_ == li && hx_policy_ != kHxVerify
                 ? 2 * (uint64_t)(hx_miss_.size() - (size_t)hx_ra_handed_ - (size_t)hx_side_handed_) : 0);
        ep_stats_.stage_ns += ns;
        if (missed) { ++ep_stats_.miss_layers; ep_stats_.miss_stage_ns += ns; }
        if (pass_tok_ == 1) {
            ep_stats_.d_stage_ns += ns;
            if (missed) { ++ep_stats_.d_miss_layers; ep_stats_.d_miss_stage_ns += ns; }
        }
        if (trace_fp_) trace_layer(L.layer, (uint32_t)missed);
    }
    pending_[b] = !staged_[b].empty();
    mover_->alias_file_units(skip_, buf_[b], &staged_[b]);
    bool cpu_aliased = false;
    if (hx_li_ == li && hx_policy_ != kHxVerify) {
        hx_keep_.clear();
        for (int k = 0; k < hx_n_; ++k)
            if (hx_units_[k].n) hx_keep_.push_back(hx_units_[k].unit);
        for (const auto& p : hx_miss_) hx_keep_.push_back(p->unit);
        mover_->alias_file_units(hx_keep_, buf_[b], &staged_[b]);
        cpu_aliased = !hx_keep_.empty();
    }
    partial_[b] = !skip_.empty() || cpu_aliased;
    if (hx_li_ == li) RAD_TRY(hx_submit_misses());
    /* The route trace's record, once the misses are in (or on hx_mw_): the pool has the layer's
     * work, and the next thing this thread does for it is wait at the gather. */
    if (rt_) rt_layer(li);
    if (!staged_[b].empty()) holds_[b] = li;
    ++ep_stats_.reads;
    ep_stats_.units += want_.size();
    if (!total) ++ep_stats_.zero_local;
    *read = true;
    return RAD_OK;
}

int FileStager::start_routed(int32_t li, RadStream s, bool* read) {
    //Generic/global rows cannot authorize the fixed local96/query6 reservation.
    RAD_TRY(head_revoke());
    *read = false;
    if (hx_capped_) {
        RAD_ERR("host experts: layer %d's pass of %lld tokens needs the card's copy path, and "
                "RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS left it no room; run without the cap",
                layers_[(size_t)li].layer, (long long)pass_tok_);
        return RAD_E_STATE;
    }
    const Layer& L = layers_[(size_t)li];
    int64_t ne = 0, top_k = 0;
    const int32_t* row = counts_ ? counts_(L.layer, &ne, &top_k) : nullptr;
    if (!row || ne <= 0 || ne > counts_cap_ || top_k <= 0) return RAD_OK;
    const int b = li & 1;
    if (holds_[b] >= 0) return RAD_E_STATE;

    RAD_TRY(rad_memcpy_async(counts_host_, row, ne * (int64_t)sizeof(int32_t), s));
    RAD_TRY(rad_event_record(counts_ev_, s));
    RAD_TRY(rad_event_sync(counts_ev_));
    int64_t sum = 0;
    for (int64_t e = 0; e < ne; ++e) sum += counts_host_[e];
    if (sum != pass_tok_ * top_k) {
        if (!warned_sum_) {
            RAD_WARN("file stager: layer %d's routing counts add up to %lld where a pass of %lld "
                     "token(s) at top_k %lld routes %lld; such passes read the whole layer",
                     L.layer, (long long)sum, (long long)pass_tok_, (long long)top_k,
                     (long long)(pass_tok_ * top_k));
            warned_sum_ = true;
        }
        return RAD_OK;
    }

    want_.clear();
    skip_.clear();
    for (int32_t u : L.units) {
        const int32_t e = plan_->units[(size_t)u].expert;
        if (e < 0 || e >= ne || counts_host_[e] != 0) want_.push_back(u);
        else skip_.push_back(u);
    }
    RAD_TRY(mover_->stage_file_units(want_, buf_[b], cap_, free_rec_[b] ? free_[b] : nullptr,
                                     done_[b], &staged_[b]));
    pending_[b] = !staged_[b].empty();
    mover_->alias_file_units(skip_, buf_[b], &staged_[b]);
    partial_[b] = !skip_.empty();
    if (!staged_[b].empty()) holds_[b] = li;
    *read = true;
    if (!told_) {
        RAD_INFO("file stager: a pass of %lld token(s) reads only the experts it routes to: %zu "
                 "of layer %d's %zu file-tier experts", (long long)pass_tok_, want_.size(),
                 L.layer, L.units.size());
        told_ = true;
    }
    return RAD_OK;
}

/* One access-trace record: u32 pass, u16 layer, u16 tokens, u16 units, u16 drive-read weights,
 * then each selected file-tier unit as an i32 (the plan's unit number). */
void FileStager::trace_layer(int32_t layer, uint32_t miss_weights) {
    const uint32_t pass = trace_pass_;
    const uint16_t hdr[4] = { (uint16_t)layer, (uint16_t)pass_tok_, (uint16_t)trace_units_.size(),
                              (uint16_t)std::min<uint32_t>(miss_weights, 65535) };
    std::fwrite(&pass, 4, 1, trace_fp_);
    std::fwrite(hdr, 2, 4, trace_fp_);
    std::fwrite(trace_units_.data(), 4, trace_units_.size(), trace_fp_);
}

/* ---- routed experts computed on the CPU (see stager.h and host_expert_pool.h) ------------------ */

namespace {
constexpr int64_t kHxEmbd = HostExpertPool::kEmbd;

int64_t hx_numel(const RadTensor& t) {
    int64_t n = 1;
    for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
    return t.rank > 0 ? n : 0;
}

uint64_t rt_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  /* namespace */

/* ---- RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE (route_trace.h; stager.h's rt_ members) ----------- */

/* The router select op of each routed layer, every layer's local tiers, the file-tier units and
 * rank 0's pinned staging; then the tiers record. Only for what the trace describes: DeepSeek-V4.1
 * EP4 layers with the counts witness (start_ep_routed); anything else writes no layer. */
int FileStager::rt_init(const Program& prog) {
    rt_ = nullptr;
    RouteTrace* rt = RouteTrace::get();
    if (!rt || !plan_ || layers_.empty()) return RAD_OK;
    const int rank = plan_->rank;
    bool eligible = false;
    for (uint8_t e : ep_layer_eligible_) eligible = eligible || e != 0;
    if (plan_->world_size != rtrace::kWorld || rank < 0 || rank >= rtrace::kWorld || !ep_counts_ ||
        !counts_host_ || counts_cap_ < rtrace::kLocal || !eligible) {
        if (rank == 0)
            RAD_WARN("route trace: it reads DeepSeek-V4.1's EP4 routed layers, and this plan has none "
                     "the file stager routes; no layer is traced");
        return RAD_OK;
    }
    /* A layer's select op: the router_scored_topk (router_sigmoid_topk has the same operands)
     * nearest before its first expert op, after the routed layer before it. */
    rt_li_of_op_.assign(prog.ops.size(), -1);
    rt_cap_.assign(layers_.size(), RtCap{});
    size_t found = 0;
    for (size_t li = 0; li < layers_.size(); ++li) {
        const int32_t lo = li ? layers_[li - 1].last_op + 1 : 0;
        for (int32_t h = layers_[li].first_op - 1; h >= lo && h >= 0; --h) {
            const std::string& n = prog.ops[(size_t)h].op;
            if (n == "router_scored_topk" || n == "router_sigmoid_topk") {
                rt_li_of_op_[(size_t)h] = (int32_t)li;
                ++found;
                break;
            }
        }
    }
    rt_tier_.assign((size_t)rtrace::kLayers, {});
    for (auto& t : rt_tier_) t.fill(255);
    for (const MoveUnit& mu : plan_->units)
        if (mu.access == RAD_ACCESS_CONDITIONAL && mu.layer >= 0 && mu.layer < rtrace::kLayers &&
            mu.expert >= 0 && mu.expert < rtrace::kLocal)
            rt_tier_[(size_t)mu.layer][(size_t)mu.expert] = (uint8_t)mu.tier;
    rt_unit_.assign(layers_.size(), {});
    rt_file_units_.assign(layers_.size(), {});
    for (size_t li = 0; li < layers_.size(); ++li) {
        rt_unit_[li].fill(-1);
        for (int32_t u : layers_[li].units) {
            const int32_t e = plan_->units[(size_t)u].expert;
            if (e < 0 || e >= rtrace::kLocal) continue;
            rt_unit_[li][(size_t)e] = u;
            rt_file_units_[li].push_back(u);
        }
    }
    if (rank == 0 && !rt_ids_) {
        const int64_t pick = kEpRouteMaxRows * rtrace::kTopK * 4;
        rt_ids_ = (int32_t*)rad_dev_alloc(pick, RAD_MEM_HOST_PINNED);
        rt_w_ = (float*)rad_dev_alloc(pick, RAD_MEM_HOST_PINNED);
        rt_logits_ = (float*)rad_dev_alloc(rtrace::kExperts * 4, RAD_MEM_HOST_PINNED);
        rt_bias_ = (float*)rad_dev_alloc((int64_t)layers_.size() * rtrace::kExperts * 4,
                                         RAD_MEM_HOST_PINNED);
        if (rad_event_create(&rt_ev_) < 0) rt_ev_ = nullptr;   /* host-waited: not a local event */
        if (!rt_ids_ || !rt_w_ || !rt_logits_ || !rt_bias_ || !rt_ev_) {
            RAD_WARN("route trace: pinned staging for the router's outputs failed; rank 0 writes "
                     "its residency only");
            for (void* p : { (void*)rt_ids_, (void*)rt_w_, (void*)rt_logits_, (void*)rt_bias_ })
                if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
            if (rt_ev_) rad_event_destroy(rt_ev_);
            rt_ids_ = nullptr;
            rt_w_ = rt_logits_ = rt_bias_ = nullptr;
            rt_ev_ = nullptr;
        }
    }
    rt_bias_state_.assign(layers_.size(), 0);
    rt_rec_.reserve(16 << 10);
    rt_ = rt;
    rt_rank_ = rank;
    rt_rec_.clear();
    rtrace::encode_tiers(rt_rec_, rt_tier_.data());
    rt_->put(rank, rt_rec_.data(), rt_rec_.size());
    if (rank == 0)
        RAD_INFO("route trace: %zu of %zu routed layers' router ops found; rank 0 copies each one's "
                 "picks and weights (and a one-token pass's 384 logits) behind its counts row, every "
                 "rank its picks' fate and its RAM cache's residency", found, layers_.size());
    return RAD_OK;
}

void FileStager::rt_pass(const RadBatch* b) {
    rt_step_ = b->step;
    rt_t0_ns_ = rt_now_ns();
    rt_q_li_ = -1;
    rtrace::PassRec p;
    p.step = b->step;
    p.pass = trace_pass_;
    p.phase = (int16_t)b->phase;
    p.draft_pass = (int16_t)b->draft_pass;
    p.n_tok = (int32_t)b->n_tok;
    p.n_seq = (int32_t)b->n_seq;
    p.n_spec = (int32_t)b->n_spec;
    p.n_tok_decode = (int32_t)b->n_tok_decode;
    p.max_ctx_len = (int32_t)b->max_ctx_len;
    p.t_ns = rt_t0_ns_;
    rt_rec_.clear();
    rtrace::encode_pass(rt_rec_, p);
    rt_->put(rt_rank_, rt_rec_.data(), rt_rec_.size());
    ++rt_stats_.passes;
}

/* The select op's operands as this pass issues them (logits, correction bias, ids, weights): where
 * rt_queue copies from, if the pass is the one routed and all its rows went in one issue. */
void FileStager::rt_capture(int32_t li, const RadTensor* t, int n_t) {
    if (n_t < 4 || (size_t)li >= rt_cap_.size()) return;
    const RadTensor &lg = t[0], &bias = t[1], &ids = t[2], &w = t[3];
    const int64_t rows = lg.rank == 2 ? lg.shape[0] : 0;
    auto rows_of = [&](const RadTensor& x, int64_t width, bool f32) {
        return x.data && x.rank == 2 && x.shape[0] == rows && x.shape[1] == width &&
               x.stride[1] == 1 && (rows == 1 || x.stride[0] >= width) &&
               (f32 ? x.dtype == RAD_F32 : (x.dtype == RAD_I32 || x.dtype == RAD_U32));
    };
    RtCap c;
    if (rows < 1 || rows > kEpRouteMaxRows || !rows_of(lg, rtrace::kExperts, true) ||
        !rows_of(ids, rtrace::kTopK, false) || !rows_of(w, rtrace::kTopK, true) || !bias.data ||
        bias.dtype != RAD_F32 || hx_numel(bias) != rtrace::kExperts ||
        bias.stride[bias.rank > 0 ? bias.rank - 1 : 0] != 1) {
        rt_cap_[(size_t)li] = c;
        return;
    }
    c.pass = trace_pass_;
    c.rows = rows;
    c.logits = (const char*)lg.data;
    c.bias = (const char*)bias.data;
    c.ids = (const char*)ids.data;
    c.w = (const char*)w.data;
    c.logits_ld = rows > 1 ? lg.stride[0] : rtrace::kExperts;
    c.ids_ld = rows > 1 ? ids.stride[0] : rtrace::kTopK;
    c.w_ld = rows > 1 ? w.stride[0] : rtrace::kTopK;
    rt_cap_[(size_t)li] = c;
}

/* Rank 0: the layer's ids and weights (and on a one-token pass its logits; the first time, its
 * bias) to pinned memory on s, behind the router that wrote them and the counts' event, then
 * rt_ev_ behind them (rt_layer waits for it). */
int FileStager::rt_queue(int32_t li, RadStream s) {
    rt_q_li_ = -1;
    rt_q_logits_ = false;
    if (rt_rank_ != 0 || !rt_ids_ || (size_t)li >= rt_cap_.size()) return RAD_OK;
    const RtCap& c = rt_cap_[(size_t)li];
    if (c.pass != trace_pass_ || c.rows != pass_tok_) {
        ++rt_stats_.uncaptured;
        return RAD_OK;
    }
    auto rows = [&](void* dst, const char* src, int64_t ld, int64_t width) -> int {
        if (c.rows == 1 || ld == width) return rad_memcpy_async(dst, src, c.rows * width * 4, s);
        for (int64_t r = 0; r < c.rows; ++r)
            RAD_TRY(rad_memcpy_async((char*)dst + r * width * 4, src + r * ld * 4, width * 4, s));
        return RAD_OK;
    };
    RAD_TRY(rows(rt_ids_, c.ids, c.ids_ld, rtrace::kTopK));
    RAD_TRY(rows(rt_w_, c.w, c.w_ld, rtrace::kTopK));
    if (c.rows == 1) {
        RAD_TRY(rad_memcpy_async(rt_logits_, c.logits, rtrace::kExperts * 4, s));
        rt_q_logits_ = true;
    }
    if (rt_bias_state_[(size_t)li] == 0) {
        RAD_TRY(rad_memcpy_async(rt_bias_ + (size_t)li * rtrace::kExperts, c.bias,
                                 rtrace::kExperts * 4, s));
        rt_bias_state_[(size_t)li] = 1;
    }
    if (rt_ev_) {
        RAD_TRY(rad_event_record(rt_ev_, s));
        rt_ev_rec_ = true;
    }
    rt_q_li_ = li;
    return RAD_OK;
}

/* Right after hx_choose (decided) or in its stead (the card's copy path takes every file-tier
 * pick): what this rank does with each of the layer's selected file-tier units, from hx_choose's
 * slots (>= 0 the cache's lease, -2 a miss for the CPU, -1 the copy path) and its misses' read-
 * ahead state, and on a one-token pass which of the layer's file-tier units the RAM cache holds
 * now (one lock for all, no lease or LRU touch: Mover::units_cached). */
void FileStager::rt_choice(int32_t li, bool decided) {
    rt_decided_ = decided;
    rt_us_ = (uint32_t)std::min<uint64_t>((rt_now_ns() - rt_t0_ns_) / 1000, UINT32_MAX);
    rt_status_.assign(rt_units_.size(), (uint8_t)rtrace::kCopyPath);
    if (decided) {
        size_t k = 0;   /* the misses in hx_choose's order: hx_miss_, hx_miss_ready_, hx_miss_ra_ */
        for (size_t i = 0; i < rt_units_.size(); ++i) {
            const int slot = i < hx_slot_.size() ? hx_slot_[i] : -1;
            uint8_t st = rtrace::kCopyPath;
            if (slot >= 0) {
                st = rtrace::kRamHit;
            } else if (slot == -2) {
                /* 1 read ahead whole, 2 the side store's: neither waited for a read */
                const bool whole = k < hx_miss_ready_.size() && hx_miss_ready_[k] != 0;
                const bool part = k < hx_miss_ra_.size() && hx_miss_ra_[k] != nullptr;
                st = whole ? rtrace::kAheadWhole : part ? rtrace::kAheadPart : rtrace::kMiss;
                ++k;
            }
            rt_status_[i] = st;
        }
    }
    rt_ram_ok_ = false;
    std::memset(rt_ram_, 0, sizeof rt_ram_);
    if (pass_tok_ == 1 && (size_t)li < rt_file_units_.size() && !rt_file_units_[(size_t)li].empty()) {
        const std::vector<int32_t>& us = rt_file_units_[(size_t)li];
        rt_hit_.resize(us.size());
        mover_->units_cached(us.data(), (int)us.size(), rt_hit_.data());
        for (size_t i = 0; i < us.size(); ++i)
            if (rt_hit_[i]) {
                const int32_t e = plan_->units[(size_t)us[i]].expert;
                rt_ram_[e >> 5] |= 1u << (e & 31);
            }
        rt_ram_ok_ = true;
    }
}

/* The layer's record (route_trace.h, type 2), and the first time rank 0's copy of the layer's bias
 * has landed, its bias record. Built from what rt_choice kept and the counts row, which the next
 * layer's copy has not yet overwritten; rank 0's router outputs landed with the counts. */
void FileStager::rt_layer(int32_t li) {
    const int32_t layer = layers_[(size_t)li].layer;
    if (layer < 0 || layer >= rtrace::kLayers) return;
    /* Rank 0's copies of this layer's router outputs: queued right behind the counts the choice
     * waited for, so in practice landed by now and this wait returns at once. */
    const bool route = rt_rank_ == 0 && rt_q_li_ == li &&
                       !(rt_ev_rec_ && rt_ev_ && rad_event_sync(rt_ev_) < 0);
    const bool logits = route && rt_q_logits_;
    const int64_t rows = pass_tok_;
    rtrace::Pick picks[rtrace::kLocal];
    int n = 0;
    const std::array<uint8_t, 96>& tier = rt_tier_[(size_t)layer];
    for (int e = 0; e < rtrace::kLocal; ++e) {
        if (counts_host_[e] <= 0) continue;
        uint8_t st = rtrace::kUndecided;
        if (tier[(size_t)e] == (uint8_t)Tier::VRAM) {
            st = rtrace::kOnCard;
        } else if (tier[(size_t)e] == (uint8_t)Tier::HostPinned ||
                   tier[(size_t)e] == (uint8_t)Tier::HostPageable) {
            st = rtrace::kHostPinned;
        } else {
            const int32_t u = rt_unit_[(size_t)li][(size_t)e];
            for (size_t i = 0; i < rt_units_.size(); ++i)
                if (rt_units_[i] == u) { st = rt_status_[i]; break; }
        }
        picks[n].expert = (uint16_t)(rt_rank_ * rtrace::kLocal + e);
        picks[n].rows = (uint8_t)std::min<int32_t>(counts_host_[e], 255);
        picks[n].status = st;
        ++n;
        if (rows == 1) ++rt_stats_.picks[st];
    }
    /* Rank 0's keys, from the logits that landed with the counts and the bias rt_queue copied
     * (this pass or an earlier one); NaN without logits. */
    float keys[kEpRouteMaxRows * rtrace::kTopK];
    if (route) {
        const float* bias = rt_bias_ + (size_t)li * rtrace::kExperts;
        for (int64_t i = 0; i < rows * rtrace::kTopK; ++i) {
            const int32_t id = rt_ids_[i];
            keys[i] = logits && id >= 0 && id < rtrace::kExperts
                          ? rtrace::sqrt_softplus(rt_logits_[id]) + bias[id]
                          : std::numeric_limits<float>::quiet_NaN();
        }
    }
    rtrace::LayerRec L;
    L.pass = trace_pass_;
    L.step = rt_step_;
    L.layer = (uint16_t)layer;
    L.rows = (uint16_t)rows;
    L.flags = (uint8_t)((rt_decided_ ? rtrace::kDecided : 0) | (route ? rtrace::kRoute : 0) |
                        (logits ? rtrace::kLogits : 0) | (rt_ram_ok_ ? rtrace::kRamMask : 0));
    L.us = rt_us_;
    std::memcpy(L.ram, rt_ram_, sizeof L.ram);
    L.picks = picks;
    L.n_picks = n;
    L.ids = rt_ids_;
    L.weights = rt_w_;
    L.keys = keys;
    L.logits = rt_logits_;
    rt_rec_.clear();
    rtrace::encode_layer(rt_rec_, L);
    rt_->put(rt_rank_, rt_rec_.data(), rt_rec_.size());
    ++rt_stats_.layers;
    rt_stats_.d_layers += rows == 1;
    rt_stats_.routes += route;
    rt_stats_.logits += logits;
    rt_stats_.masks += rt_ram_ok_;
    if (route && rt_bias_state_[(size_t)li] == 1) {   /* the layer's bias, the first time it landed */
        rt_rec_.clear();
        rtrace::encode_bias(rt_rec_, (uint16_t)layer, rt_bias_ + (size_t)li * rtrace::kExperts);
        rt_->put(rt_rank_, rt_rec_.data(), rt_rec_.size());
        rt_bias_state_[(size_t)li] = 2;
        ++rt_stats_.biases;
    }
    rt_q_li_ = -1;
}

/* ---- exact tall passes (RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL, ep_route_counts.h) ------------------
 *
 * A pass of 17 to 48 tokens may take the EP counts path and the CPU path only if every op gives it a
 * 16-token pass's bits, row for row, so that every reply is the 16-token chunking's. Four ops change
 * arithmetic with the pass height: the routed MoE GEMM above 16 tokens (moe_gemm_ggml_mxfp4_gemv's
 * band ends there; the tiled row sums in another order), the sparse attention and the indexer score
 * above 48 rows (libr4d's kWmRows), and every norm above 32 rows (r4d_act_threads halves its width,
 * which regroups the sum of squares). Under the switch libr4d adds the gemv's twin
 * moe_gemm_ggml_mxfp4_gemv_tall, bounded at the tallest pass its kernels keep the decode band's bits
 * at (libr4d/r4d_ds41_tall.h: 32 tokens where the norms' width holds by itself, 48 where libr4d pins
 * it), and keeps the sparse ops scalar at every height. The CPU's rows are the gemv's bits at any
 * number of rows.
 *
 * So the step (--max-num-batched-tokens) must be one every routed MXFP4 GEMM resolved on the gemv or
 * its twin in every band: a band on the tiled row means a libr4d without the twin, one loaded without
 * the switch, or a step taller than the twin reaches, and is refused here before a pass could mix the
 * sums. Every other op must run one kernel at every height it is declared for, or one of the pairs
 * known to give the same bits (gemm_bf16_nt_m16 forwards to gemm_bf16_nt_m64): a second row on any
 * other op is refused by name, since nothing here says its sums agree. A step above kEpRouteTallRows
 * is refused too: its passes would run with other arithmetic. */
namespace {
/* The bound a row places on M (RAD_C_LE), or 0. */
int64_t row_m_bound(const KernelRow* row) {
    if (!row || !row->info || !row->info->constraints) return 0;
    for (int i = 0; i < row->info->n_constraints; ++i) {
        const RadConstraint& c = row->info->constraints[i];
        if (c.key && !std::strcmp(c.key, "M") && c.op == RAD_C_LE) return c.ival;
    }
    return 0;
}
const char* row_name(const Band& b) {
    const KernelRow* r = b.dom[RAD_DOMAIN_DEVICE].row;
    return r && r->info && r->info->name ? r->info->name : nullptr;
}
}  /* namespace */

int FileStager::tall_init(const Program& prog) {
    ep_rows_ = kEpRouteMaxRows;
    const int st = ds41_exact_tall_state();
    if (st < 0) {
        const char* v = std::getenv(kExactTallEnv);
        RAD_ERR("%s=%s: 1 runs prefill passes of up to %lld tokens with a 16-token pass's bits (the routed "
                "GEMMs, the sparse attention, the indexer and the norms on their decode-band arithmetic), 0 or "
                "unset is off", kExactTallEnv, v ? v : "", (long long)kEpRouteTallRows);
        return RAD_E_INVAL;
    }
    if (st == 0) return RAD_OK;
    const int64_t T = prog.ctx.max_tok;
    if (T > kEpRouteTallRows) {
        RAD_ERR("%s=1 keeps a 16-token pass's bits in passes of up to %lld tokens; --max-num-batched-tokens "
                "%lld would run taller ones with other arithmetic (the norms' width, the sparse ops' WMMA "
                "forms): give at most %lld", kExactTallEnv, (long long)kEpRouteTallRows, (long long)T,
                (long long)kEpRouteTallRows);
        return RAD_E_INVAL;
    }
    int n_moe = 0, n_twin = 0, n_gemm = 0, n_ops = 0;
    int64_t reach = 0;   /* the twin's bound, as a band resolved to it states it */
    for (size_t h = 1; h < prog.ops.size(); ++h) {
        const OpInfo& op = prog.ops[h];
        if (op.bands.empty()) continue;
        ++n_ops;
        const char* ty = op.base.get_s("type");
        if (op.op == "moe_gemm_ggml" && ty && !std::strcmp(ty, "mxfp4")) {
            ++n_moe;
            int64_t lo = op.range_lo > 0 ? op.range_lo - 1 : 0;   /* the band's exclusive lower bound */
            int64_t ok_to = lo;   /* the gemv rows serve every band up to here */
            for (const Band& b : op.bands) {
                const char* name = row_name(b);
                const bool gemv = name && !std::strcmp(name, "moe_gemm_ggml_mxfp4_gemv");
                const bool twin = name && !std::strcmp(name, "moe_gemm_ggml_mxfp4_gemv_tall");
                if (!gemv && !twin) {
                    long long N = 0, K = 0;
                    (void)op.base.get_i("N", &N);
                    (void)op.base.get_i("K", &K);
                    RAD_ERR("%s=1: op %zu (moe_gemm_ggml, N %lld K %lld) resolves %s to %s, not to the decode "
                            "band's kernel (moe_gemm_ggml_mxfp4_gemv, or its twin moe_gemm_ggml_mxfp4_gemv_tall "
                            "above 16 tokens), so a %lld-token pass would not give a 16-token pass's bits. %s",
                            kExactTallEnv, h, N, K, b.span.empty() ? "a band" : b.span.c_str(),
                            name ? name : "nothing", (long long)b.hi,
                            ok_to > kEpRouteMaxRows
                                ? fmt("This libr4d keeps them in passes of up to %lld tokens: give "
                                      "--max-num-batched-tokens %lld or less", (long long)ok_to,
                                      (long long)ok_to).c_str()
                                : "Load a libr4d.so with the twin, with the switch in its environment");
                    return RAD_E_INVAL;
                }
                if (twin) {
                    ++n_twin;
                    reach = std::max(reach, row_m_bound(b.dom[RAD_DOMAIN_DEVICE].row));
                }
                ok_to = lo = b.hi;
            }
            continue;
        }
        const char* first = nullptr;
        bool m16 = false;
        for (const Band& b : op.bands) {
            const char* name = row_name(b);
            if (!name) continue;
            if (!first) { first = name; continue; }
            if (!std::strcmp(name, first)) continue;
            const bool pair = op.op == "gemm_nt" &&
                              ((!std::strcmp(first, "gemm_bf16_nt_m16") && !std::strcmp(name, "gemm_bf16_nt_m64")) ||
                               (!std::strcmp(first, "gemm_bf16_nt_m64") && !std::strcmp(name, "gemm_bf16_nt_m16")));
            if (pair) { m16 = true; continue; }
            RAD_ERR("%s=1: op %zu (%s) runs %s in one band and %s in %s: nothing here says the two give a "
                    "row the same bits, so a pass of another height could change it. Refused; the step "
                    "(--max-num-batched-tokens %lld) keeps it on one kernel at 16 or fewer tokens",
                    kExactTallEnv, h, op.op.c_str(), first, name, b.span.empty() ? "another" : b.span.c_str(),
                    (long long)T);
            return RAD_E_INVAL;
        }
        n_gemm += m16;
    }
    ep_rows_ = std::max<int64_t>(T, kEpRouteMaxRows);
    if (plan_ && plan_->rank == 0)
        RAD_INFO("exact tall passes: passes of up to %lld tokens (%s=1, --max-num-batched-tokens %lld): %d routed "
                 "MXFP4 GEMMs on moe_gemm_ggml_mxfp4_gemv to 16 tokens%s, %d gemm_nt on gemm_bf16_nt_m16 / _m64 "
                 "(one split law), the other %d of %d ops on one kernel at every height; libr4d keeps the sparse "
                 "attention, the indexer and the norms on their decode-band arithmetic; the routing counts and the "
                 "CPU path take passes of up to %lld tokens",
                 (long long)ep_rows_, kExactTallEnv, (long long)T, n_moe,
                 n_twin ? fmt(" and its twin moe_gemm_ggml_mxfp4_gemv_tall above (%d bands; this libr4d's "
                              "twin reaches %lld)", n_twin, (long long)reach).c_str() : "",
                 n_gemm, n_ops - n_moe - n_gemm, n_ops, (long long)ep_rows_);
    return RAD_OK;
}

namespace {
/* RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE: set and not 0. */
const char* side_store_asked() {
    const char* v = std::getenv("RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE");
    return v && *v && std::strcmp(v, "0") != 0 ? v : nullptr;
}
}  /* namespace */

/* RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=N (hx_side_store.h), from hx_init_impl once the misses'
 * rows are known and before the insert threads can see a refusal (the warm-up's). */
int FileStager::hx_side_init() {
    const char* v = side_store_asked();
    if (!v) return RAD_OK;
    const int n = std::atoi(v);
    if (n < 1 || n > 64 || std::to_string(n) != v || !mover_ || !mover_->packed_cache() ||
        hx_miss_rows_ <= 0) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=%s: N (1-64) units a rank whose inserts the packed "
                "RAM cache refuses keep their planes, which needs RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1 "
                "and the misses on the CPU (host experts, ..._MISSES)", v);
        return RAD_E_INVAL;
    }
    hx_side_.configure(n);
    if (plan_->rank == 0)
        RAD_INFO("host experts: side store: the planes of up to %d units a rank whose inserts the packed "
                 "RAM cache refuses are kept and handed to their later misses, which read nothing "
                 "(~18 MiB a unit; RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE)", n);
    return RAD_OK;
}

/* Which layers can hand units to the pool: the EP-routed ones whose first expert op is a decode
 * gate_up (moe_gemm_ggml mxfp4 [4608, 5120], token order), whose last is the down projection
 * ([5120, 2304], sorted order) with a swiglu_clamped at the pool's limit between, and whose
 * gathers (moe_gather, or ar_gather_hc_write folding it in, up to the next routed layer) read the
 * down projection's rows as operand 0. */
/* RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS caps the file buffers (planner.cpp), which then
 * hold only the aliases of units the CPU computes: refuse a configuration in which any routed pass
 * could need the card's copy path. The packed RAM cache (Mover::packed_cache) needs that cap: the
 * card cannot read its slots, so the CPU must compute every unit it holds. With
 * RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL the passes may be up to ep_rows_ tokens (the step, up to
 * 48), and the misses must be the CPU's in all of them. */
int FileStager::hx_init(const Program& prog) {
    RAD_TRY(hx_init_impl(prog));
    /* Asked for, the store must run (hx_side_init): host experts with misses on the CPU. */
    if (const char* v = side_store_asked(); v && !hx_side_.on()) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=%s: no store without host experts that compute "
                "the misses on the CPU (..._HOST_EXPERTS, ..._MISSES) from the packed RAM cache", v);
        return RAD_E_INVAL;
    }
    /* The misses the CPU must take: every pass's. Without the switch that is 16 tokens whatever
     * --max-num-batched-tokens says (as before); with it the step's (tall_init: ep_rows_ is the
     * step, at least 16). */
    const int64_t need = ep_rows_;
    hx_need_misses_ = need;
    auto packed_ok = [&]() -> int {
        if (!mover_ || !mover_->packed_cache() || hx_capped_) return hx_init_fetch();
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1: the RAM cache then holds a form only the CPU "
                "reads, so the CPU must compute every routed unit: host experts with ..._POLICY=all and "
                "..._MISSES=%lld, and RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS set",
                (long long)need);
        return RAD_E_INVAL;
    };
    const char* cap = std::getenv("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS");
    if (!cap || std::atoi(cap) <= 0 || layers_.empty()) return packed_ok();
    bool every = hx_pool_ != nullptr;
    for (size_t li = 0; every && li < layers_.size(); ++li) {
        every = li < ep_layer_eligible_.size() && ep_layer_eligible_[li] && li < hx_ops_.size() &&
                hx_ops_[li].gu >= 0;
        for (int32_t u : layers_[li].units)   /* a unit the CPU cannot read would need the card */
            every = every && mover_->host_planes_eligible(u);
    }
    if (!every || !hx_defer_ok_ || hx_miss_rows_ < need || prog.ctx.max_tok > ep_rows_) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=%s: the file buffers then hold no layer's "
                "units, so the CPU must compute every routed unit of every pass: host experts with "
                "..._POLICY=all and ..._MISSES=%lld on every routed layer, and passes of at most %lld "
                "tokens (--max-num-batched-tokens%s)", cap, (long long)need, (long long)ep_rows_,
                ds41_exact_tall_state() == 1
                    ? "" : "; taller with RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1, as far as libr4d keeps a "
                           "16-token pass's bits");
        return RAD_E_INVAL;
    }
    hx_capped_ = true;
    if (plan_->rank == 0)
        RAD_INFO("host experts: file buffers capped at %s units; every routed unit is the CPU's", cap);
    return packed_ok();
}

int FileStager::hx_init_impl(const Program& prog) {
    hx_drain();
    hx_pool_ = nullptr;
    hx_defer_ok_ = false;
    hx_defer_li_ = -1;
    hx_capped_ = false;
    hx_side_.clear();
    hx_side_.configure(0);
    hx_side_handed_ = 0;
    hx_ops_.assign(layers_.size(), HxOps{});
    /* The DSpark head tail owns the routed layers' file buffers and their copy budget. */
    if (head_.configured()) {
        if (plan_ && plan_->rank == 0 && std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS"))
            RAD_INFO("host experts: off; the DSpark head tail serves the routed layers");
        return RAD_OK;
    }
    HostExpertPool* pool = HostExpertPool::get();
    if (!pool || !ep_counts_ || !counts_host_ || !plan_) return RAD_OK;
    const char* pol = std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY");
    hx_policy_ = !pol || !*pol || !std::strcmp(pol, "greedy") ? kHxGreedy
               : !std::strcmp(pol, "all")                    ? kHxAll
               : !std::strcmp(pol, "verify")                 ? kHxVerify : -1;
    if (hx_policy_ < 0) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=%s: greedy, all or verify", pol);
        return RAD_E_INVAL;
    }
    auto gemm = [&](int32_t h, long long n, long long k, const char* order, HxOps* o) {
        if (h < 0 || (size_t)h >= prog.ops.size()) return false;
        const OpInfo& op = prog.ops[(size_t)h];
        long long N = 0, K = 0, ne = 0, tk = 0;
        const char* ord = op.base.get_s("a_order");
        const char* ty = op.base.get_s("type");
        if (op.op != "moe_gemm_ggml" || !op.base.get_i("N", &N) || !op.base.get_i("K", &K) ||
            !op.base.get_i("n_expert", &ne) || !op.base.get_i("top_k", &tk) || N != n || K != k ||
            ne != 96 || tk != 6 || !ord || std::strcmp(ord, order) || !ty || std::strcmp(ty, "mxfp4"))
            return false;
        if (o) { o->n_expert = ne; o->top_k = tk; }
        return true;
    };
    size_t n_ok = 0;
    for (size_t li = 0; li < layers_.size(); ++li) {
        if (li >= ep_layer_eligible_.size() || !ep_layer_eligible_[li]) continue;
        const Layer& L = layers_[li];
        HxOps o;
        if (!gemm(L.first_op, 2 * HostExpertPool::kFf, kHxEmbd, "token", &o) ||
            !gemm(L.last_op, kHxEmbd, HostExpertPool::kFf, "sorted", nullptr))
            continue;
        bool limit = false;
        for (int32_t h = L.first_op + 1; h < L.last_op; ++h) {
            const OpInfo& op = prog.ops[(size_t)h];
            double v = 0;
            if (op.op == "swiglu_clamped")
                limit = op.base.get_f("limit", &v) && (float)v == pool->limit();
        }
        /* Every reader of the down rows up to the next routed layer must be one of these: up to
         * the next layer staged here, or the next MoE block this stager does not stage (a DSpark
         * draft head's host-pinned experts after the last layer), whichever comes first. */
        const int32_t end = li + 1 < layers_.size() ? layers_[li + 1].first_op
                                                    : (int32_t)prog.ops.size();
        int n_gather = 0;
        bool shape = true;
        for (int32_t h = L.last_op + 1; h < end && shape; ++h) {
            const OpInfo& op = prog.ops[(size_t)h];
            if (op.op.compare(0, 8, "moe_gemm") == 0) break;
            if (op.op != "moe_gather" && op.op != "ar_gather_hc_write") continue;
            const RadOpSchema* sc = op.schema;
            shape = n_gather < 3 && sc && sc->n_operands >= 1 && sc->operands[0].name &&
                    !std::strcmp(sc->operands[0].name, "y_expert");
            if (shape) o.gather[n_gather++] = h;
        }
        if (!limit || !shape || !n_gather) continue;
        o.gu = L.first_op;
        hx_ops_[li] = o;
        ++n_ok;
    }
    if (!n_ok) {
        if (plan_->rank == 0)
            RAD_WARN("host experts: no routed layer has the DeepSeek-V4.1 decode shape the pool "
                     "computes; every expert stays on the card");
        return RAD_OK;
    }
    const int64_t rows = hx_sorted_max() * kHxEmbd * (int64_t)sizeof(uint16_t);
    /* Sized for passes of ep_rows_ tokens; made again if an earlier init sized them otherwise. */
    if (hx_alloc_rows_ != ep_rows_) {
        if (hx_out_rec_ && hx_out_ev_) (void)rad_event_sync(hx_out_ev_);
        hx_out_rec_ = false;
        if (hx_x_) rad_dev_free(hx_x_, RAD_MEM_HOST_PINNED);
        if (hx_idx_) rad_dev_free(hx_idx_, RAD_MEM_HOST_PINNED);
        if (hx_out_) rad_dev_free(hx_out_, RAD_MEM_HOST_PINNED);
        if (hx_chk_) rad_dev_free(hx_chk_, RAD_MEM_HOST_PINNED);
        hx_x_ = nullptr;
        hx_idx_ = nullptr;
        hx_out_ = nullptr;
        hx_chk_ = nullptr;
        hx_alloc_rows_ = ep_rows_;
    }
    if (!hx_x_) hx_x_ = (uint16_t*)rad_dev_alloc(ep_rows_ * kHxEmbd * 2, RAD_MEM_HOST_PINNED);
    if (!hx_idx_) hx_idx_ = (int32_t*)rad_dev_alloc((hx_sorted_max() + 128) * 4, RAD_MEM_HOST_PINNED);
    if (!hx_out_) hx_out_ = (uint16_t*)rad_dev_alloc(rows, RAD_MEM_HOST_PINNED);
    if (!hx_chk_ && hx_policy_ == kHxVerify)
        hx_chk_ = (uint16_t*)rad_dev_alloc(rows, RAD_MEM_HOST_PINNED);
    if (!hx_in_ev_ && rad_event_create(&hx_in_ev_) < 0) hx_in_ev_ = nullptr;
    if (!hx_out_ev_ && rad_event_create(&hx_out_ev_) < 0) hx_out_ev_ = nullptr;
    if (!hx_units_) hx_units_.reset(new (std::nothrow) Mover::HostUnit[kHxUnits]);
    if (!hx_x_ || !hx_idx_ || !hx_out_ || (hx_policy_ == kHxVerify && !hx_chk_) || !hx_in_ev_ ||
        !hx_out_ev_ || !hx_units_) {
        RAD_ERR("host experts: pinned staging or events failed; every expert stays on the card");
        return RAD_OK;
    }
    hx_pool_ = pool;
    if (const char* m = std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES"); m && *m)
        hx_miss_rows_ = std::clamp(std::atoi(m), 0, (int)ep_rows_);
    if (hx_miss_rows_ > 0 && !mover_->mapped_cache_on()) {
        if (plan_->rank == 0)
            RAD_WARN("host experts: ..._MISSES needs the mapped RAM cache "
                     "(RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1); misses stay on the copy link");
        hx_miss_rows_ = 0;
    }
    hx_unit_of_.assign(layers_.size(), {});
    for (size_t li = 0; li < layers_.size(); ++li) {
        hx_unit_of_[li].fill(-1);
        for (int32_t u : layers_[li].units) {
            const int32_t e = plan_->units[(size_t)u].expert;
            if (e >= 0 && e < 96) hx_unit_of_[li][(size_t)e] = u;
        }
    }
    hx_pred_ = RouterPredictor::get();
    if (hx_pred_ && !hx_pred_->load(prog, [this](const WeightInfo& w, void* dst) {
            return mover_->read_weight_file(w, dst);
        }))
        hx_pred_ = nullptr;
    if (hx_pred_ && !hx_pred_->shadow() && !hx_pred_->pieces() && hx_miss_rows_ > 0 &&
        !hx_pf_.joinable()) {
        hx_pf_ahead_ = hx_pred_->ahead();
        hx_pf_stop_ = false;
        hx_pf_ = std::thread(&FileStager::hx_pf_loop, this);
    }
    /* Read-ahead in pieces (stager.h, readahead.h): the process's readers, this rank's tables, and
     * the predictor's callback. */
    if (hx_pred_ && hx_pred_->pieces() && hx_miss_rows_ > 0 && !hx_ra_) {
        int qd = 4, live = 8;
        int64_t piece_kb = 512;
        if (const char* q = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_QD"); q && *q)
            qd = std::clamp(std::atoi(q), 1, 8);
        if (const char* b = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_PIECE_KB"); b && *b)
            piece_kb = std::clamp<int64_t>(std::atoll(b), 64, 16384) / 4 * 4;
        if (const char* l = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_LIVE"); l && *l)
            live = std::clamp(std::atoi(l), 1, 64);
        const char* fin = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_FINISH");
        hx_ra_finish_ = !fin || std::strcmp(fin, "0") != 0;
        PieceReader& r = PieceReader::process();
        if (!r.start(qd, piece_kb << 10)) {
            if (plan_->rank == 0)
                RAD_WARN("read-ahead: the piece readers did not start (%d readers, %lld KiB pieces); "
                         "no read-ahead", qd, (long long)piece_kb);
        } else {
            hx_ra_li_.assign(RouterPredictor::kLayers, -1);
            for (size_t li = 0; li < layers_.size(); ++li)
                if (layers_[li].layer >= 0 && layers_[li].layer < RouterPredictor::kLayers)
                    hx_ra_li_[(size_t)layers_[li].layer] = (int32_t)li;
            hx_ra_ok_.assign(plan_->units.size(), 0);
            for (const Layer& L : layers_)
                for (int32_t u : L.units) hx_ra_ok_[(size_t)u] = mover_->host_planes_eligible(u);
            hx_ra_cap_ = live;
            hx_ra_at_.store(0, std::memory_order_release);
            hx_ra_ = &r;
            {
                std::lock_guard<std::mutex> lk(g_ra_mu);
                g_ra_stagers.push_back(this);
            }
            hx_pred_->on_ready(&FileStager::hx_ra_ready, nullptr);
            if (plan_->rank == 0)
                RAD_INFO("read-ahead: each rank's uncached units among layer L+2's top %d predicted "
                         "experts are read at layer L in %lld KiB pieces, %d readers over the process, "
                         "yielding to demand reads; at most %d live units a rank; a partly read unit "
                         "its layer needs is %s", hx_pred_->k(), (long long)(r.piece_bytes() >> 10),
                         r.readers(), live, hx_ra_finish_ ? "finished by the readers" : "read afresh");
        }
    }
    /* ..._SIDE_STORE: before the insert threads can see a refusal (the warm-up's below). */
    RAD_TRY(hx_side_init());
    if (hx_miss_rows_ > 0 && hx_bg_.empty()) {
        hx_bg_stop_ = false;
        int n = 4;
        if (const char* t = std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_INSERT_THREADS"); t && *t)
            n = std::clamp(std::atoi(t), 1, 16);
        for (int i = 0; i < n; ++i) hx_bg_.emplace_back(&FileStager::hx_bg_loop, this);
        if (plan_->rank == 0)
            RAD_INFO("host experts: passes of at most %d token(s) compute missed experts on the CPU "
                     "from the container's planes; background threads relay them into the RAM cache",
                     hx_miss_rows_);
        /* A second copy of the planes on another drive, read beside the container (plane_mirror.h). */
        RAD_TRY(mover_->mirror_init());
        if (const char* a = std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES");
            a && std::atoi(a) > 0 && !hx_mw_.joinable()) {
            hx_async_rows_ = std::atoi(a);
            hx_mw_stop_ = false;
            hx_mw_ = std::thread(&FileStager::hx_mw_loop, this);
            hx_async_ = true;
            if (plan_->rank == 0)
                RAD_INFO("host experts: passes of %d+ tokens read their misses on a worker while the "
                         "card's ops are issued", hx_async_rows_);
        }
        /* RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1: a one-token pass's misses start on the kept
         * readers at the choice instead of after hx_submit (its pool wake, 28-32 us a layer) and
         * the copy path's staging; the same reads, so the same bytes. Units read ahead whole and
         * the ones the piece readers finish (..._PREFETCH=pieces) are not among them, as in
         * hx_submit_misses_now (hx_unread_misses). */
        if (const char* mp = std::getenv("RADIANCE_EXPERIMENTAL_DS41_MISS_POST"); mp && !std::strcmp(mp, "1")) {
            hx_miss_post_on_ = true;
            if (plan_->rank == 0)
                RAD_INFO("host experts: a one-token pass's missed planes start on the kept readers "
                         "as soon as the layer's choice is made (RADIANCE_EXPERIMENTAL_DS41_MISS_POST)");
        }
        /* RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=N: a cached unit the decode profile calls
         * hot (a share of its layer's selections of at least ..._CACHE_PROTECT_SHARE, default
         * 6 of the sealed profile's 4326) moves to the front up to N times between uses before
         * an insert may evict it. Replayed on the chat and held-out prompts' trace: critical
         * decode misses -7.9% (N=1) and -10.7% (N=2) against plain LRU. */
        if (const char* sc = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE");
            sc && std::atoi(sc) > 0) {
            double thr = 6.0 / 4326.0;
            if (const char* t = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CACHE_PROTECT_SHARE"); t && *t)
                thr = std::atof(t);
            std::vector<int32_t> hot;
            for (const Layer& L : layers_)
                for (int32_t u : L.units)
                    if (plan_->units[(size_t)u].profile_share >= thr) hot.push_back(u);
            const int n = mover_->mapped_cache_second_chances(hot, std::atoi(sc));
            RAD_INFO("host experts: rank %d protects %d of its file-tier units with %d second "
                     "chance(s) in the RAM cache", plan_->rank, n, std::atoi(sc));
        }
        if (const char* w = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM"); w && !std::strcmp(w, "1"))
            RAD_TRY(hx_warm());
    }
    if (plan_->rank == 0)
        RAD_INFO("host experts: %zu of %zu routed layers give the CPU %s cached experts%s", n_ok,
                 layers_.size(), hx_policy_ == kHxAll ? "every one of their" : "some of their",
                 hx_policy_ == kHxVerify ? " and still compute them on the card, comparing the rows"
                 : hx_policy_ == kHxGreedy ? " (whichever of the pool and the copy link frees sooner)"
                                           : "");
    hx_defer_ok_ = hx_policy_ == kHxAll && hx_miss_rows_ > 0;
    return RAD_OK;
}

/* Splits want_ (the layer's selected units, in container order) between the copy path and the
 * pool: every unit the RAM cache holds is borrowed; then each borrowed one goes to the pool when
 * the pool's backlog, including what other ranks reserved, would finish it no later than this
 * rank's copy link would finish its copies. Misses always take the copy path. */
void FileStager::hx_choose(int32_t li) {
    hx_n_ = 0;
    hx_reserved_ = 0;
    hx_miss_.clear();
    hx_miss_ready_.clear();
    hx_miss_ra_.clear();
    hx_ra_handed_ = 0;
    hx_side_handed_ = 0;
    hx_slot_.assign(want_.size(), -1);
    const bool ra = hx_ra_ && pass_tok_ == 1;   /* read-ahead in pieces: this layer's units */
    if (ra) hx_ra_begin(li);
    const double cb = hx_pool_->cpu_bytes_per_ns(), lb = hx_pool_->link_bytes_per_ns();
    const bool misses = hx_miss_rows_ > 0 && pass_tok_ <= hx_miss_rows_;
    int64_t link = 0;
    for (size_t i = 0; i < want_.size(); ++i) {
        const int32_t u = want_[i];
        Mover::HostUnit* hu = hx_n_ < kHxUnits ? &hx_units_[hx_n_] : nullptr;
        ++hx_stats_.probed;
        if (hu && mover_->borrow_host_unit(u, hu)) {
            bool gu = false, dn = false;
            for (int k = 0; k < hu->n; ++k) {
                gu = gu || hu->bytes[k] == HostExpertPool::kGuBytes;
                dn = dn || hu->bytes[k] == HostExpertPool::kDnBytes;
            }
            if (hu->n == 2 && gu && dn) {
                hx_slot_[i] = hx_n_++;
                if (ra)
                    for (RaEntry& r : hx_ra_here_)
                        if (r.u && r.unit == u) hx_ra_drop(r, true);
                continue;
            }
            mover_->release_host_unit(hu);
        }
        ++hx_stats_.missed;
        /* A miss for the CPU: its planes are read in hx_submit_misses; reserved now so the
         * schedule below and the other ranks see the pool's load. */
        const int32_t e = plan_->units[(size_t)u].expert;
        if (misses && (int)hx_miss_.size() < kHxUnits && mover_->host_planes_eligible(u)) {
            hx_slot_[i] = -2;
            /* Read ahead for this layer? Take it, waiting while its read is in flight; one whose
             * read has not begun is read below as any miss (the read-ahead then skips it). */
            std::shared_ptr<Mover::HostPlanes> ahead;
            std::shared_ptr<ReadAheadUnit> urgent;
            bool ready = false, side = false;
            /* ..._SIDE_STORE: a unit the packed cache refuses, its planes kept from an earlier miss
             * (hx_side_store.h): nothing to read. A read-ahead planned before they were kept goes. */
            if (hx_side_.on() && (ahead = hx_side_.find(u))) {
                side = true;
                if (ra)
                    for (RaEntry& r : hx_ra_here_)
                        if (r.u && r.unit == u) hx_ra_drop(r, true);
                ++hx_side_handed_;
                ++hx_stats_.side_hits;
                hx_stats_.side_d_hits += pass_tok_ == 1;
            } else if (hx_pf_.joinable()) {
                std::unique_lock<std::mutex> lk(hx_pf_mu_);
                PfBatch& pb = hx_pf_batch_[li & 1];
                if (pb.pass == trace_pass_ && pb.li == li)
                    for (PfEntry& pe : pb.entries)
                        if (pe.unit == u && pe.p) {
                            if (pe.state == 0) {
                                pe.state = 3;
                                ++hx_stats_.pf_taken;
                            } else {
                                if (pe.state == 1) {
                                    const auto w0 = std::chrono::steady_clock::now();
                                    ++hx_stats_.pf_waited;
                                    hx_pf_cv_.wait(lk, [&] { return pe.state == 2 || hx_pf_stop_; });
                                    hx_stats_.pf_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - w0).count();
                                }
                                if (pe.state == 2 && pe.ok) ahead = pe.p;
                            }
                            pe.p.reset();
                            break;
                        }
                if (ahead) ++hx_stats_.pf_used;
                ready = ahead != nullptr;
            } else if (ra) {
                ahead = hx_ra_take(u, &ready, &urgent);
            }
            hx_miss_ready_.push_back(side ? 2 : ready ? 1 : 0);
            if (hx_ra_) hx_miss_ra_.push_back(std::move(urgent));
            hx_miss_.push_back(ahead ? ahead : hx_planes_get(u));
            const int rows = e >= 0 && e < counts_cap_ ? std::max<int32_t>(counts_host_[e], 1) : 1;
            hx_pool_->reserve(HostExpertPool::cost_rows(rows));
            hx_reserved_ += HostExpertPool::cost_rows(rows);
            continue;
        }
        link += plan_->units[(size_t)u].bytes;
    }
    if (ra) {   /* the layer's units it did not select */
        for (RaEntry& r : hx_ra_here_) hx_ra_drop(r, false);
        hx_ra_here_.clear();
    }
    /* The predictor's guess for this layer, made one layer ago, against what the layer misses. */
    if (hx_pred_ && pass_tok_ == 1 && !hx_pred_->pieces()) {
        ++hx_stats_.pred_layers;
        int32_t g[RouterPredictor::kMaxK];
        if (hx_pred_->predicted(trace_pass_, layers_[(size_t)li].layer, g)) {
            ++hx_stats_.pred_ready;
            bool pred[96] = {};
            const int base = 96 * plan_->rank;
            for (int k = 0; k < hx_pred_->k(); ++k) {
                const int e = g[k] - base;
                if (e >= 0 && e < 96 && hx_unit_of_[(size_t)li][(size_t)e] >= 0) {
                    pred[e] = true;
                    ++hx_stats_.pred_file;
                }
            }
            bool sel[96] = {};
            for (size_t i = 0; i < want_.size(); ++i) {
                const int32_t e = plan_->units[(size_t)want_[i]].expert;
                const bool p = e >= 0 && e < 96 && pred[e];
                if (e >= 0 && e < 96) sel[e] = true;
                hx_stats_.pred_selected += p;
                if (hx_slot_[i] == -1 || hx_slot_[i] == -2) {   /* not borrowed: a miss */
                    ++hx_stats_.pred_misses;
                    hx_stats_.pred_hit += p;
                }
            }
            /* What a read-ahead of every uncached prediction would have read for nothing. */
            for (int e = 0; e < 96; ++e)
                if (pred[e] && !sel[e] && !mover_->unit_cached(hx_unit_of_[(size_t)li][(size_t)e]))
                    ++hx_stats_.pred_wasted;
            for (int k = 0; k < hx_pred_->k() && k < 16; ++k) {
                const int e = g[k] - base;
                if (e < 0 || e >= 96 || hx_unit_of_[(size_t)li][(size_t)e] < 0 ||
                    mover_->unit_cached(hx_unit_of_[(size_t)li][(size_t)e]))
                    continue;
                ++hx_stats_.pred_pos_unc[k];
                hx_stats_.pred_pos_hit[k] += sel[e];
            }
        }
        /* The prediction made two layers back, against the same misses (shadow only). */
        if (hx_pred_->shadow() && hx_pred_->predicted2(trace_pass_, layers_[(size_t)li].layer, g)) {
            ++hx_stats_.pred2_ready;
            bool pred[96] = {}, sel[96] = {};
            const int base = 96 * plan_->rank;
            for (int k = 0; k < hx_pred_->k(); ++k) {
                const int e = g[k] - base;
                if (e >= 0 && e < 96) pred[e] = true;
            }
            for (size_t i = 0; i < want_.size(); ++i) {
                const int32_t e = plan_->units[(size_t)want_[i]].expert;
                if (e >= 0 && e < 96) sel[e] = true;
                if (hx_slot_[i] == -1 || hx_slot_[i] == -2) {   /* not borrowed: a miss */
                    ++hx_stats_.pred2_misses;
                    hx_stats_.pred2_hit += e >= 0 && e < 96 && pred[e];
                }
            }
            for (int k = 0; k < hx_pred_->k() && k < 16; ++k) {
                const int e = g[k] - base;
                if (e < 0 || e >= 96 || hx_unit_of_[(size_t)li][(size_t)e] < 0 ||
                    mover_->unit_cached(hx_unit_of_[(size_t)li][(size_t)e]))
                    continue;
                ++hx_stats_.pred2_pos_unc[k];
                hx_stats_.pred2_pos_hit[k] += sel[e];
            }
        }
    }
    int n_cpu = 0;
    for (size_t i = 0; i < want_.size(); ++i) {
        if (hx_slot_[i] < 0) continue;
        Mover::HostUnit& hu = hx_units_[hx_slot_[i]];
        const int32_t e = plan_->units[(size_t)hu.unit].expert;
        const int rows = e >= 0 && e < counts_cap_ ? std::max<int32_t>(counts_host_[e], 1) : 1;
        const int64_t cost = HostExpertPool::cost_rows(rows), bytes = plan_->units[(size_t)hu.unit].bytes;
        const bool cpu = hx_policy_ != kHxGreedy ||
                         (double)(hx_pool_->backlog() + cost) / cb <= (double)(link + bytes) / lb;
        if (cpu) {
            hx_pool_->reserve(cost);
            hx_reserved_ += cost;
            ++n_cpu;
            continue;
        }
        mover_->release_host_unit(&hu);
        hx_slot_[i] = -1;
        ++hx_stats_.linked;
        link += bytes;
    }
    if (!n_cpu && hx_miss_.empty()) {
        hx_n_ = 0;
        return;
    }
    if (hx_policy_ != kHxVerify) {
        hx_keep_.clear();
        for (size_t i = 0; i < want_.size(); ++i)
            if (hx_slot_[i] == -1) hx_keep_.push_back(want_[i]);
        want_.swap(hx_keep_);
    }
    hx_li_ = li;
    hx_stage_ = 1;
}

/* The pass's token rows, sorted order and expert offsets to the host, queued behind the counts
 * row so the counts' wait covers them: the router and its scatter wrote them, and the router read
 * the rows. Their addresses are the gate_up operands hx_capture saw on an earlier pass, which
 * hx_capture checks again on this one. */
int FileStager::hx_fetch(int32_t li, RadStream s) {
    const HxOps& o = hx_ops_[(size_t)li];
    if (pass_tok_ < 1 || pass_tok_ > ep_rows_ || ep_rows_ > hx_alloc_rows_) return RAD_E_STATE;
    if (o.x_ld == kHxEmbd) {
        RAD_TRY(rad_memcpy_async(hx_x_, o.x, pass_tok_ * kHxEmbd * 2, s));
    } else {
        for (int64_t r = 0; r < pass_tok_; ++r)
            RAD_TRY(rad_memcpy_async(hx_x_ + r * kHxEmbd, (const char*)o.x + r * o.x_ld * 2,
                                     kHxEmbd * 2, s));
    }
    RAD_TRY(rad_memcpy_async(hx_idx_, o.sorted, pass_tok_ * o.top_k * 4, s));
    RAD_TRY(rad_memcpy_async(hx_idx_ + hx_sorted_max(), o.off, (o.n_expert + 1) * 4, s));
    return RAD_OK;
}

/* ..._FETCH_LIST: layer li's counts row (`row`, its witness's) and the copies hx_fetch makes, as
 * one ranged copy on s from the list armed for the layer at this pass's size. The list is written
 * first when it was armed for other addresses (the gate_up operands hx_capture keeps, the counts
 * row, the stream) or never. Each separate copy is a dispatch behind the one before with its own
 * system-scope release (the destination is host memory), on the path from the scatter to the
 * host's wake-up; the ranged copy is one, and with its list on the card it acquires at agent scope
 * as the separate copies did. Returns 1 when the copy is queued, 0 when the caller must make the
 * separate copies (an address the copy kernel cannot take; counted). */
int FileStager::hx_fetch_list(int32_t li, const int32_t* row, RadStream s) {
    const HxOps& o = hx_ops_[(size_t)li];
    const int64_t T = pass_tok_;
    const size_t at = (size_t)li * (size_t)ep_rows_ + (size_t)(T - 1);
    if (T < 1 || T > ep_rows_ || at >= hx_fl_key_.size() || at >= hx_fl_cap_) {
        ++hx_stats_.fl_plain;
        return 0;
    }
    HxFl& k = hx_fl_key_[at];
    RadCopyRange* const dev = hx_fl_dev_ + at * (size_t)hx_fl_slot_;
    if (k.n <= 0 || k.row != row || k.x != o.x || k.x_ld != o.x_ld || k.sorted != o.sorted ||
        k.off != o.off || k.s != s) {
        HxFetch f;
        f.counts = row;
        f.counts_host = counts_host_;
        f.n_counts = 96;
        f.x = o.x;
        f.x_ld = o.x_ld;
        f.x_host = hx_x_;
        f.embd = kHxEmbd;
        f.rows = T;
        f.sorted = o.sorted;
        f.sorted_host = hx_idx_;
        f.n_sorted = T * o.top_k;
        f.off = o.off;
        f.off_host = hx_idx_ + hx_sorted_max();
        f.n_off = o.n_expert + 1;
        RadCopyRange r[hx_fl_slot(kEpRouteTallRows)] = {};
        const int n = hx_fetch_ranges(f, r, std::min(hx_fetch_n_ranges(ep_rows_), kHxFlRangesMax));
        k = HxFl{};
        if (n <= 0) {
            ++hx_stats_.fl_plain;
            return 0;
        }
        /* The staging slots' last write may still be reading one (a list is written once a layer
         * and pass size, and again after an operand move, so this wait is rare and short). */
        if (hx_fl_rec_) {
            RAD_TRY(rad_event_sync(hx_fl_ev_));
            hx_fl_rec_ = false;
        }
        /* Whole pairs of ranges (48 bytes), so the write is a 16-byte copy. */
        const int64_t bytes = (int64_t)((n + 1) & ~1) * (int64_t)sizeof(RadCopyRange);
        RadCopyRange* const pin = hx_fl_pin_ + at * (size_t)hx_fl_slot_;
        std::memcpy(pin, r, (size_t)bytes);
        RAD_TRY(rad_memcpy_async(dev, pin, bytes, s));
        RAD_TRY(rad_event_record(hx_fl_ev_, s));
        hx_fl_rec_ = true;
        k = HxFl{ row, o.x, o.x_ld, o.sorted, o.off, s, n };
        ++hx_stats_.fl_armed;
    }
    RAD_TRY(rad_memcpy_ranges_async(counts_host_, row, 0, dev, k.n, s));
    ++hx_stats_.fl_copies;
    return 1;
}

/* The two switches of a routed layer's host round trip (stager.h), once hx_init knows whether the
 * file buffers are capped. A switch the configuration cannot honour is refused rather than ignored,
 * so an A/B arm never runs without what its name says. */
int FileStager::hx_init_fetch() {
    hx_early_ = false;
    hx_fl_ = false;
    auto asked = [](const char* name, const char** v) {
        *v = std::getenv(name);
        return *v && **v && std::strcmp(*v, "0") != 0;
    };
    const char* e = nullptr;
    if (asked("RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE", &e)) {
        if (std::strcmp(e, "1") != 0 || !hx_pool_ || !hx_capped_) {
            RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=%s: 1 issues a routed layer's card ops ahead "
                    "of its counts sync, which needs every file-tier unit to be the CPU's: host experts "
                    "with ..._POLICY=all and ..._MISSES=%lld, and RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS "
                    "set", e, (long long)hx_need_misses_);
            return RAD_E_INVAL;
        }
        hx_early_ = true;
        if (plan_->rank == 0)
            RAD_INFO("host experts: early issue: a routed layer's card ops go ahead of its counts sync, "
                     "and its CPU share is chosen at its gather");
    }
    hx_post_late_ = false;
    const char* pl = nullptr;
    if (asked("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE", &pl)) {
        if (std::strcmp(pl, "1") != 0 || !hx_early_ || !hx_pred_ || !hx_pred_->pieces() || !hx_ra_) {
            RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=%s: 1 posts the read-ahead's prediction "
                    "after an early-issued layer's CPU work is submitted, which needs "
                    "RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1 and the read-ahead in pieces "
                    "(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces, its readers running)", pl);
            return RAD_E_INVAL;
        }
        hx_post_late_ = true;
        if (plan_->rank == 0)
            RAD_INFO("host experts: early issue: rank 0 posts the read-ahead's router input once a layer's "
                     "CPU work is submitted (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE)");
    }
    const char* f = nullptr;
    if (asked("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST", &f)) {
        if (std::strcmp(f, "1") != 0 || !hx_pool_) {
            RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=%s: 1 copies a routed layer's counts and the "
                    "CPU's rows to the host in one ranged copy, which needs host experts", f);
            return RAD_E_INVAL;
        }
        const size_t n = layers_.size() * (size_t)ep_rows_;
        const int slot = hx_fl_slot(ep_rows_);
        const int64_t bytes = (int64_t)n * slot * (int64_t)sizeof(RadCopyRange);
        if (n != hx_fl_cap_ || slot != hx_fl_slot_) {
            if (hx_fl_rec_ && hx_fl_ev_) (void)rad_event_sync(hx_fl_ev_);
            hx_fl_rec_ = false;
            if (hx_fl_dev_) rad_dev_free(hx_fl_dev_, RAD_MEM_DEVICE);
            if (hx_fl_pin_) rad_dev_free(hx_fl_pin_, RAD_MEM_HOST_PINNED);
            hx_fl_dev_ = (RadCopyRange*)rad_dev_alloc(bytes, RAD_MEM_DEVICE);
            hx_fl_pin_ = (RadCopyRange*)rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
            hx_fl_cap_ = hx_fl_dev_ && hx_fl_pin_ ? n : 0;
            hx_fl_slot_ = slot;
            if (hx_fl_pin_) std::memset(hx_fl_pin_, 0, (size_t)bytes);
        }
        if (!hx_fl_ev_ && rad_event_create(&hx_fl_ev_) < 0) hx_fl_ev_ = nullptr;
        if (!hx_fl_cap_ || !hx_fl_ev_) {
            RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1: %s of range lists or their event failed: %s",
                    humanb(bytes).c_str(), rad_dev_last_error());
            return RAD_E_NOMEM;
        }
        hx_fl_key_.assign(n, HxFl{});
        hx_fl_ = true;
        if (plan_->rank == 0)
            RAD_INFO("host experts: fetch lists: a routed layer's counts and the CPU's rows reach the "
                     "host in one ranged copy (%zu lists a rank, %s on the card)", n,
                     humanb(bytes).c_str());
    }
    return RAD_OK;
}

/* ..._EARLY_ISSUE: layer hx_early_li_'s first gather, before its launch. The layer's card ops are
 * queued ahead of it (start_ep_routed returned without waiting), so now comes what start_ep_routed
 * does after its counts sync without the switch: wait for the counts and rows, choose the CPU's
 * share and submit it, read the misses. Every selected unit must go to the CPU, since the card
 * computed all of them from aliases (as on a first pass, hx_deferred). operands() then waits for
 * the CPU's rows and copies them over the card's (hx_finish). The wait here is shorter than the
 * one it replaces by the time the thread spent issuing the layer's card ops; what the card gains
 * is the time from its counts copy to the gate_up: the thread's wake-up, the choice and submission
 * (~35 us a layer), and in a decode pass the misses' drive reads (~2.4 ms a missed expert). */
int FileStager::hx_early_decide() {
    const int32_t li = hx_early_li_;
    hx_early_li_ = -1;
    const Layer& L = layers_[(size_t)li];
    if (!hx_early_seen_) {
        RAD_ERR("host experts: layer %d reached its gather with no gate_up seen to read the rows "
                "its counts sync fetched", L.layer);
        return RAD_E_STATE;
    }
    auto ns = [](auto a, auto b) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    /* Counts already in: this thread was behind the card, so the card ops it issued first held
     * back the CPU's start by their issue time (what the switch can lose). */
    if (rad_event_query(counts_ev_) == 1) ++hx_stats_.early_ready;
    const auto t_sync = std::chrono::steady_clock::now();
    RAD_TRY(rad_event_sync(counts_ev_));
    const auto t_counts = std::chrono::steady_clock::now();
    const uint64_t sync_ns = ns(t_sync, t_counts);
    ep_stats_.sync_ns += sync_ns;
    if (pass_tok_ == 1) { ++ep_stats_.d_reads; ep_stats_.d_sync_ns += sync_ns; }
    ++hx_stats_.early;
    hx_stats_.early_wait_ns += sync_ns;
    RAD_TRY(mover_->retire_file_windows());
    const auto t_retired = std::chrono::steady_clock::now();
    /* ..._PREFETCH_POST_LATE: after hx_submit below instead. */
    if (hx_pred_ && pass_tok_ == 1 && plan_->rank == 0 && !hx_post_late_ &&
        moe_runs(L.layer + (hx_pred_->pieces() ? 2 : 1)))
        hx_pred_->post(trace_pass_, L.layer, hx_x_);
    int64_t total = 0;
    if (!ep_route_counts_valid(hx_early_v_, counts_host_, counts_cap_, &total)) {
        ++ep_stats_.corrupt;
        return RAD_E_STATE;
    }
    want_.clear();
    skip_.clear();
    for (int32_t u : L.units) {
        bool selected = false;
        if (!ep_route_unit_selected(counts_host_, plan_->units[(size_t)u].expert, &selected))
            return RAD_E_STATE;
        (selected ? want_ : skip_).push_back(u);
    }
    if (trace_fp_) trace_units_ = want_;
    if (rt_) rt_units_ = want_;
    const size_t selected = want_.size();
    hx_choose(li);
    if (rt_) rt_choice(li, true);
    const auto t_chosen = std::chrono::steady_clock::now();
    if (!want_.empty() || (selected && hx_li_ != li)) {
        RAD_ERR("host experts: layer %d left %zu of %zu selected units off the CPU, and its card ops "
                "went ahead with aliases for all of them", L.layer, want_.size(), selected);
        return RAD_E_STATE;
    }
    /* ..._MISS_POST, as in start_ep_routed: the misses' reads start before the pool's wake, and are
     * joined on every way out of here (hx_submit_misses_now first). */
    const HxMissPostJoin miss_post_join{ this };
    auto t_miss_posted = t_chosen;
    if (hx_miss_post_on_ && hx_li_ == li) {
        hx_post_misses();
        t_miss_posted = std::chrono::steady_clock::now();
    }
    if (hx_li_ == li) RAD_TRY(hx_submit());
    const auto t_submitted = std::chrono::steady_clock::now();
    auto t_posted = t_submitted;
    /* ..._PREFETCH_POST_LATE: the prediction of layer L+2 once this layer's CPU work is out, as
     * start_ep_routed posts it, so the post does not hold back the pool's start. */
    if (hx_post_late_ && pass_tok_ == 1 && plan_->rank == 0 && moe_runs(L.layer + 2)) {
        hx_pred_->post(trace_pass_, L.layer, hx_x_);
        t_posted = std::chrono::steady_clock::now();
    }
    /* This layer took its read-ahead entries: now the next layer's (predicted from this one). */
    if (hx_pf_.joinable() && pass_tok_ == 1 && (size_t)(li + hx_pf_ahead_) < layers_.size() &&
        moe_runs(layers_[(size_t)(li + hx_pf_ahead_)].layer))
        hx_pf_target(li + hx_pf_ahead_);
    const auto t_stage = std::chrono::steady_clock::now();
    ep_stats_.hx_ns += ns(t_counts, t_stage);
    if (pass_tok_ == 1) {
        ep_stats_.d_retire_ns += ns(t_counts, t_retired);
        ep_stats_.d_choose_ns += ns(t_retired, t_chosen);
        ep_stats_.d_miss_post_ns += ns(t_chosen, t_miss_posted);
        ep_stats_.d_submit_ns += ns(t_miss_posted, t_submitted);
        ep_stats_.d_post_ns += ns(t_submitted, t_posted);
        ep_stats_.d_decide_ns += ns(t_counts, t_stage);
    }
    /* Nothing is staged (the card reads only aliases); the misses' drive reads count as the
     * ordinary path counts them, two weights a missed unit. */
    const uint64_t missed = hx_li_ == li ? 2 * (uint64_t)(hx_miss_.size() - (size_t)hx_side_handed_) : 0;
    if (missed) {
        ++ep_stats_.miss_layers;
        if (pass_tok_ == 1) ++ep_stats_.d_miss_layers;
    }
    if (trace_fp_) trace_layer(L.layer, (uint32_t)missed);
    if (hx_li_ == li) {
        RAD_TRY(hx_submit_misses());
        hx_confirmed_ = true;   /* hx_capture saw this pass's gate_up read the rows fetched */
    }
    /* The route trace's record, before this gather waits for the pool (as in start_ep_routed). */
    if (rt_) rt_layer(li);
    ++ep_stats_.reads;
    if (!total) ++ep_stats_.zero_local;
    return RAD_OK;
}

/* The jobs for local expert e: one a row it owns in the pass's sorted order. Expert e's rows are
 * moe_gemm_ggml's (mx_range): [off[e], off[e + 1]), the first expert from 0 and the last to T; a
 * slot whose sorted entry is negative or past the pass's tokens is a row of zeros on the card and
 * stays the card's. Returns the rows, or an error when they disagree with the expert's count. */
int FileStager::hx_jobs_for(int e, const HostExpertPool::Job& proto) {
    const HxOps& o = hx_ops_[(size_t)hx_li_];
    const int64_t AR = pass_tok_, T = pass_tok_ * o.top_k;
    const int32_t* const sorted = hx_idx_;
    const int32_t* const off = hx_idx_ + hx_sorted_max();
    const int ne = (int)o.n_expert;
    if (e < 0 || e >= ne) return RAD_E_STATE;
    int lo = e == 0 ? 0 : off[e], hi = e == ne - 1 ? (int)T : off[e + 1];
    lo = std::clamp(lo, 0, (int)T);
    hi = std::clamp(hi, lo, (int)T);
    int32_t n = 0;
    HostExpertPool::Job j = proto;
    int in_job = 0;
    const int per_job = HostExpertPool::max_rows();   /* 4, or up to 32 (..._HOST_EXPERTS_MAX_ROWS) */
    auto flush = [&] {
        if (!in_job) return;
        j.more = in_job - 1;
        j.cost = HostExpertPool::job_cost(in_job);
        if (in_job >= HostExpertPool::many_from()) {
            ++hx_stats_.many_jobs;
            hx_stats_.many_rows += (uint64_t)in_job;
        }
        hx_jobs_.push_back(j);
        j = proto;
        in_job = 0;
    };
    for (int i = lo; i < hi; ++i) {
        if (sorted[i] < 0 || sorted[i] / o.top_k >= AR) continue;
        const uint16_t* x = hx_x_ + (int64_t)(sorted[i] / o.top_k) * kHxEmbd;
        uint16_t* y = hx_out_ + (int64_t)i * kHxEmbd;
        if (!in_job) { j.x = x; j.y = y; }
        else { j.xs[in_job - 1] = x; j.ys[in_job - 1] = y; }
        if (++in_job == per_job) flush();
        hx_rows_.push_back(i);
        ++n;
    }
    flush();
    /* The scatter's own count for the expert, which chose it: a different number of rows means the
     * sorted order was not read as the card reads it. */
    if (e >= counts_cap_ || n != counts_host_[e]) {
        RAD_ERR("host experts: layer %d expert %d has %d rows in the sorted order and %d in its count",
                layers_[(size_t)hx_li_].layer, e, n, e < counts_cap_ ? counts_host_[e] : -1);
        return RAD_E_STATE;
    }
    return n;
}

/* The cached units' jobs, from their leases. The misses' come in hx_submit_misses, after their
 * reads. */
int FileStager::hx_submit() {
    /* The rows' last copies to the card are behind the counts this layer waited for. */
    if (hx_out_rec_) {
        RAD_TRY(rad_event_sync(hx_out_ev_));
        hx_out_rec_ = false;
    }
    hx_jobs_.clear();
    hx_rows_.clear();
    for (int k = 0; k < hx_n_; ++k) {
        const Mover::HostUnit& hu = hx_units_[k];
        if (!hu.n) continue;
        HostExpertPool::Job proto;
        const int g = hu.bytes[0] == HostExpertPool::kGuBytes ? 0 : 1;
        proto.gu = hu.host[g];
        proto.dn = hu.host[1 - g];
        if (hu.packed) {   /* the packed RAM cache: the slots' own sizes bound their views */
            proto.packed = true;
            proto.gu_bytes = hu.slot[g];
            proto.dn_bytes = hu.slot[1 - g];
        }
        const int n = hx_jobs_for(plan_->units[(size_t)hu.unit].expert, proto);
        if (n < 0) return n;
        ++hx_stats_.units;
    }
    /* What hx_choose reserved for the misses stays reserved for hx_submit_misses. */
    int64_t miss_res = 0;
    for (const auto& p : hx_miss_) {
        const int32_t e = plan_->units[(size_t)p->unit].expert;
        miss_res += HostExpertPool::cost_rows(e >= 0 && e < counts_cap_ ? std::max<int32_t>(counts_host_[e], 1) : 1);
    }
    int64_t submitted = 0;
    for (const HostExpertPool::Job& j : hx_jobs_) submitted += j.cost;
    if (hx_reserved_ - miss_res != submitted) hx_pool_->unreserve(hx_reserved_ - miss_res - submitted);
    hx_reserved_ = miss_res;
    RAD_TRY(hx_pool_->submit(hx_jobs_.data(), (int)hx_jobs_.size(), &hx_ticket_));
    hx_stage_ = 2;
    hx_confirmed_ = false;
    ++hx_stats_.layers;
    hx_stats_.rows += hx_rows_.size();
    return RAD_OK;
}

/* The missed units: their planes read from the container at once (blocking, as the card path's
 * reads do), urgent jobs from the planes, and the planes handed to the background thread, which
 * relays them into the RAM cache. The jobs and the inserts share the buffers (shared_ptr). */
int FileStager::hx_submit_misses() {
    if (hx_miss_.empty()) return RAD_OK;
    if (!hx_async_ || pass_tok_ < hx_async_rows_) return hx_submit_misses_now((int)pass_tok_);
    {
        std::lock_guard<std::mutex> lk(hx_mw_mu_);
        hx_mw_req_ = true;
        hx_mw_rc_ = RAD_OK;
        hx_mw_rows_ = (int)pass_tok_;
    }
    hx_mw_cv_.notify_all();
    return RAD_OK;
}

/* The request's result (RAD_OK when there was none), once hx_mw_ is idle. */
int FileStager::hx_mw_wait() {
    std::unique_lock<std::mutex> lk(hx_mw_mu_);
    hx_mw_cv_.wait(lk, [&] { return !hx_mw_req_; });
    const int rc = hx_mw_rc_;
    hx_mw_rc_ = RAD_OK;
    return rc;
}

void FileStager::hx_mw_loop() {
    std::unique_lock<std::mutex> lk(hx_mw_mu_);
    for (;;) {
        hx_mw_cv_.wait(lk, [&] { return hx_mw_stop_ || hx_mw_req_; });
        if (!hx_mw_req_) return;   /* stopping, nothing asked */
        const int rows = hx_mw_rows_;
        lk.unlock();
        const int rc = hx_submit_misses_now(rows);
        lk.lock();
        hx_mw_rc_ = rc;
        hx_mw_req_ = false;
        hx_mw_cv_.notify_all();
    }
}

/* The misses hx_submit_misses_now reads (stager.h): neither read ahead whole nor urgent. */
bool FileStager::hx_unread_misses(std::vector<Mover::HostPlanes*>* out) const {
    bool urgent = false;   /* units the piece readers are finishing for this layer */
    for (size_t i = 0; i < hx_miss_.size(); ++i) {
        if (i < hx_miss_ra_.size() && hx_miss_ra_[i]) urgent = true;
        else if (i >= hx_miss_ready_.size() || !hx_miss_ready_[i]) out->push_back(hx_miss_[i].get());
    }
    return urgent;
}

/* Reads the misses' planes (rows: the pass's tokens) and submits their jobs to the pool. */
int FileStager::hx_submit_misses_now(int rows) {
    std::vector<Mover::HostPlanes*> ptrs;
    const bool urgent = hx_unread_misses(&ptrs);
    const auto t0 = std::chrono::steady_clock::now();
    Mover::PlaneReadTiming tm;
    bool read = false;
    if (hx_miss_posted_ || !ptrs.empty() || urgent) {
        /* A demand read in flight: every rank's read-ahead holds back (hx_pf_loop, readahead.h).
         * The urgent units' last pieces are part of it. A read posted at the choice (..._MISS_POST,
         * hx_post_misses) entered the count there, with these same units. */
        PieceReader& demand = PieceReader::process();
        bool ok = true;
        if (hx_miss_posted_) {
            /* Finished before hx_ra_settle, which may read on the same readers (read_unit_planes):
             * the post holds their call until this finish. */
            hx_miss_posted_ = false;
            ok = mover_->finish_unit_planes(&tm);
            read = true;
            if (ptrs != hx_miss_post_ptrs_) {
                demand.demand_leave();
                RAD_ERR("host experts: layer %d's misses changed after their reads were posted",
                        layers_[(size_t)hx_li_].layer);
                return RAD_E_STATE;
            }
        } else {
            demand.demand_enter();
            if (!ptrs.empty()) {
                ok = mover_->read_unit_planes(ptrs.data(), (int)ptrs.size(), rows, &tm);
                read = true;
            }
        }
        if (ok && urgent) ok = hx_ra_settle(rows);
        demand.demand_leave();
        if (!ok) {
            RAD_ERR("host experts: reading layer %d's missed units for the CPU failed",
                    layers_[(size_t)hx_li_].layer);
            return RAD_E_IO;
        }
    }
    /* Units read ahead whole, or held by the side store: handed over only from the container the
     * program was loaded from. */
    if ((hx_ra_handed_ > 0 || hx_side_handed_ > 0) && !mover_->source_epochs_current()) {
        RAD_ERR("host experts: the container changed under layer %d's units read ahead or kept in the "
                "side store", layers_[(size_t)hx_li_].layer);
        return RAD_E_IO;
    }
    hx_stats_.miss_read_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t0).count();
    if (read && rows == 1) hx_note_reads(tm);
    const size_t first = hx_jobs_.size();
    for (auto& p : hx_miss_) {
        if (p->n != 2) return RAD_E_STATE;
        const int g = p->rows[0] == 2 * HostExpertPool::kFf ? 0 : 1, d = 1 - g;
        if (p->rows[g] != 2 * HostExpertPool::kFf || p->cols[g] != kHxEmbd ||
            p->rows[d] != kHxEmbd || p->cols[d] != HostExpertPool::kFf)
            return RAD_E_STATE;
        HostExpertPool::Job proto;
        proto.gu = p->codes[g];
        proto.gu_scales = p->scales[g];
        proto.dn = p->codes[d];
        proto.dn_scales = p->scales[d];
        proto.urgent = true;
        const int n = hx_jobs_for(plan_->units[(size_t)p->unit].expert, proto);
        if (n < 0) return n;
        ++hx_stats_.miss_units;
        hx_stats_.miss_rows += (uint64_t)n;
    }
    const int n = (int)(hx_jobs_.size() - first);
    int64_t submitted = 0;
    for (size_t k = first; k < hx_jobs_.size(); ++k) submitted += hx_jobs_[k].cost;
    if (hx_reserved_ != submitted) hx_pool_->unreserve(hx_reserved_ - submitted);
    hx_reserved_ = 0;
    RAD_TRY(hx_pool_->submit(hx_jobs_.data() + first, n, &hx_ticket_));
    std::sort(hx_rows_.begin(), hx_rows_.end());
    {   /* the side store's units are not inserted again: the packed cache refuses them */
        std::lock_guard<std::mutex> lk(hx_bg_mu_);
        for (size_t i = 0; i < hx_miss_.size(); ++i)
            if (i >= hx_miss_ready_.size() || hx_miss_ready_[i] != 2) hx_bg_q_.push_back(hx_miss_[i]);
    }
    hx_bg_cv_.notify_all();
    return RAD_OK;
}

/* RADIANCE_EXPERIMENTAL_DS41_MISS_POST: the layer's unread misses (hx_unread_misses: the set
 * hx_submit_misses_now reads) handed to the mover's kept readers right after hx_choose. A decode
 * miss layer used to start its reads only after hx_submit had woken the pool (28-32 us a layer) and
 * the copy path's staging and aliases had run; the readers' own wake-up now overlaps that work too.
 * Only where hx_submit_misses_now reads on this thread (one-token passes, not on hx_mw_). Never a
 * unit read ahead (..._PREFETCH): whole ones are not read again, and an urgent one is the piece
 * readers' until hx_ra_settle, which may give its slot fresh planes and the old ones back to the
 * free list. A demand read from the post on (PieceReader's count: no read-ahead piece starts) until
 * hx_submit_misses_now or hx_join_miss_post leaves it. Nothing posted (one in flight, no kept
 * readers, the readers busy with another thread's call): hx_submit_misses_now reads as before. */
void FileStager::hx_post_misses() {
    if (hx_miss_posted_ || pass_tok_ != 1 || (hx_async_ && pass_tok_ >= hx_async_rows_)) return;
    hx_miss_post_ptrs_.clear();
    (void)hx_unread_misses(&hx_miss_post_ptrs_);
    if (hx_miss_post_ptrs_.empty()) return;
    PieceReader& demand = PieceReader::process();
    demand.demand_enter();
    hx_miss_posted_ = mover_->start_unit_planes(hx_miss_post_ptrs_.data(),
                                                (int)hx_miss_post_ptrs_.size(), (int)pass_tok_);
    if (!hx_miss_posted_) demand.demand_leave();
}

/* A posted read not finished by hx_submit_misses_now (an error on the way): wait for it, so no
 * reader writes into planes the layer has let go, and end its demand read. */
void FileStager::hx_join_miss_post() {
    if (!hx_miss_posted_) return;
    hx_miss_posted_ = false;
    (void)mover_->finish_unit_planes();
    PieceReader::process().demand_leave();
}

/* One one-token miss read's timing into hx_stats_.mr (see HostStats::MissReads). */
void FileStager::hx_note_reads(const Mover::PlaneReadTiming& t) {
    HostStats::MissReads& s = hx_stats_.mr[t.units == 1 ? 0 : 1];
    ++s.calls;
    s.posted += t.posted;
    s.split += t.mirror_reads > t.units;
    if (t.mirror_reads && t.container_ns) {
        ++s.mirror_calls;
        if (t.mirror_ns > t.container_ns) {
            ++s.mirror_last;
            s.mirror_by_ns += (uint64_t)(t.mirror_ns - t.container_ns);
        } else {
            s.container_by_ns += (uint64_t)(t.container_ns - t.mirror_ns);
        }
        s.mirror_ns += (uint64_t)t.mirror_ns;
        s.copy_ns += (uint64_t)t.copy_ns;
        s.copy_sum_ns += (uint64_t)t.copy_sum_ns;
    }
    s.container_ns += (uint64_t)t.container_ns;
    s.first_ns += (uint64_t)t.first_ns;
    s.first_container_ns += (uint64_t)t.first_container_ns;
    s.last_start_ns += (uint64_t)t.last_start_ns;
    /* After every read landed, or after a posted call's caller came to wait if that was later. */
    s.back_ns += (uint64_t)std::max<int64_t>(
        t.return_ns - std::max({ t.mirror_ns, t.container_ns, t.join_ns }), 0);
    s.join_ns += (uint64_t)t.join_ns;
    s.total_ns += (uint64_t)t.return_ns;
}

/* A HostPlanes whose buffers are reused: the deleter returns it to the free list. */
std::shared_ptr<Mover::HostPlanes> FileStager::hx_planes_get(int32_t unit) {
    std::unique_ptr<Mover::HostPlanes> p;
    {
        std::lock_guard<std::mutex> lk(hx_planes_mu_);
        if (!hx_planes_free_.empty()) {
            p = std::move(hx_planes_free_.back());
            hx_planes_free_.pop_back();
        }
    }
    if (!p) p.reset(new Mover::HostPlanes());
    p->unit = unit;
    p->n = 0;
    return std::shared_ptr<Mover::HostPlanes>(p.release(), [this](Mover::HostPlanes* q) {
        std::lock_guard<std::mutex> lk(hx_planes_mu_);
        hx_planes_free_.emplace_back(q);
    });
}

/* The issuing thread has layer li - hx_pf_ahead_'s counts, whose router input predicts li: read
 * layer li's predicted uncached units ahead, into the batch of li's parity (the batch two layers
 * back is dropped; the one between is kept). */
void FileStager::hx_pf_target(int32_t li) {
    if (li < 0 || li >= (int32_t)layers_.size()) return;
    {
        std::lock_guard<std::mutex> lk(hx_pf_mu_);
        PfBatch& b = hx_pf_batch_[li & 1];
        b.entries.clear();
        b.pass = trace_pass_;
        b.li = li;
        b.seq = ++hx_pf_seq_;
        hx_pf_queue_.push_back(b.seq);
    }
    hx_pf_cv_.notify_all();
}

void FileStager::hx_pf_loop() {
    std::unique_lock<std::mutex> lk(hx_pf_mu_);
    for (;;) {
        hx_pf_cv_.wait(lk, [&] { return hx_pf_stop_ || !hx_pf_queue_.empty(); });
        if (hx_pf_stop_) return;
        const uint64_t seq = hx_pf_queue_.front();
        hx_pf_queue_.pop_front();
        PfBatch* b = nullptr;
        for (PfBatch& x : hx_pf_batch_) if (x.seq == seq) b = &x;
        if (!b) continue;   /* replaced before its turn */
        const uint32_t pass = b->pass;
        const int32_t li = b->li;
        lk.unlock();
        /* The prediction is a fraction of a millisecond behind the post; give up after 3 ms or when
         * the batch has been replaced. */
        int32_t g[RouterPredictor::kMaxK];
        bool got = false;
        const int layer = layers_[(size_t)li].layer;
        const auto t0 = std::chrono::steady_clock::now();
        while (!(got = hx_pf_ahead_ == 2 ? hx_pred_->predicted2(pass, layer, g)
                                         : hx_pred_->predicted(pass, layer, g))) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(3)) break;
            {
                std::lock_guard<std::mutex> chk(hx_pf_mu_);
                if (b->seq != seq || hx_pf_stop_) break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        lk.lock();
        if (!got || b->seq != seq || hx_pf_stop_) continue;
        const int base = 96 * plan_->rank;
        for (int k = 0; k < hx_pred_->k(); ++k) {
            const int e = g[k] - base;
            if (e < 0 || e >= 96) continue;
            const int32_t u = hx_unit_of_[(size_t)li][(size_t)e];
            if (u < 0 || !mover_->host_planes_eligible(u) || mover_->unit_cached(u) || hx_side_.holds(u))
                continue;
            PfEntry pe;
            pe.unit = u;
            pe.p = hx_planes_get(u);
            b->entries.push_back(std::move(pe));
        }
        /* One unit at a time, best prediction first, each started only while no demand read is in
         * flight (up to 2 ms) and while its layer has not taken it. */
        for (size_t i = 0; i < b->entries.size(); ++i) {
            if (b->seq != seq || hx_pf_stop_) break;
            lk.unlock();
            const auto y0 = std::chrono::steady_clock::now();
            while (PieceReader::process().demand_count() > 0 &&
                   std::chrono::steady_clock::now() - y0 < std::chrono::milliseconds(2))
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            const uint64_t yield_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - y0).count();
            lk.lock();
            hx_stats_.pf_yield_ns += yield_ns;
            if (b->seq != seq || hx_pf_stop_) break;
            PfEntry& pe = b->entries[i];
            if (pe.state != 0 || !pe.p) continue;   /* taken by its layer */
            pe.state = 1;
            std::shared_ptr<Mover::HostPlanes> p = pe.p;   /* held past a drop of the batch */
            Mover::HostPlanes* raw = p.get();
            lk.unlock();
            const bool ok = mover_->read_unit_planes(&raw, 1);
            lk.lock();
            ++hx_stats_.pf_read;
            if (b->seq == seq && i < b->entries.size()) {
                b->entries[i].state = 2;
                b->entries[i].ok = ok;
            }
            hx_pf_cv_.notify_all();
        }
    }
}

/* ---- read-ahead in pieces (stager.h, readahead.h) ---------------------------------------------- */

/* On the predictor's thread: layer `layer`'s prediction to every rank's stager. */
void FileStager::hx_ra_ready(void*, uint32_t pass, int layer, const int32_t* experts, int k) {
    std::lock_guard<std::mutex> lk(g_ra_mu);
    for (FileStager* s : g_ra_stagers) s->hx_ra_plan(pass, layer, experts, k);
}

int FileStager::hx_ra_read(void* ctx, const PlaneExtent& p) {
    return static_cast<Mover*>(ctx)->read_piece(p.file, p.off, p.len, p.dst, p.cap);
}

namespace {
/* hx_ra_at_'s pass and routed layer. */
inline bool ra_too_late(uint64_t at, uint32_t pass, int32_t li) {
    const uint32_t at_pass = (uint32_t)(at >> 32);
    const int32_t at_li = (int32_t)(uint32_t)at - 1;
    return (int32_t)(pass - at_pass) < 0 || (pass == at_pass && li <= at_li);
}
}  /* namespace */

/* On the predictor's thread: this rank's predicted experts of `layer` that are on its file tier,
 * readable by the CPU path and not in the RAM cache, laid out in planes buffers and queued for the
 * readers, best prediction first. Only the plan, the program, the cache's own lock and the free
 * list are touched here, never what the issuing thread writes. */
void FileStager::hx_ra_plan(uint32_t pass, int layer, const int32_t* g, int k) {
    if (!hx_ra_ || layer < 0 || layer >= (int)hx_ra_li_.size()) return;
    const int32_t li = hx_ra_li_[(size_t)layer];
    if (li < 0 || (size_t)li >= hx_unit_of_.size()) return;
    const int base = 96 * plan_->rank;
    for (int pos = 0; pos < k; ++pos) {
        const int e = g[pos] - base;
        if (e < 0 || e >= 96) continue;
        const int32_t u = hx_unit_of_[(size_t)li][(size_t)e];
        if (u < 0 || (size_t)u >= hx_ra_ok_.size() || !hx_ra_ok_[(size_t)u] || mover_->unit_cached(u) ||
            hx_side_.holds(u))   /* ..._SIDE_STORE: its planes are in */
            continue;
        {
            std::lock_guard<std::mutex> lk(hx_ra_mu_);
            if (ra_too_late(hx_ra_at_.load(std::memory_order_acquire), pass, li)) {
                ++hx_stats_.ra_late;
                return;
            }
            if ((int)hx_ra_live_.size() >= hx_ra_cap_) {
                ++hx_stats_.ra_capped;
                return;
            }
            bool dup = false;
            for (const RaEntry& r : hx_ra_live_) dup = dup || (r.pass == pass && r.li == li && r.unit == u);
            if (dup) continue;
        }
        std::shared_ptr<Mover::HostPlanes> p = hx_planes_get(u);
        PlaneExtent ext[2];
        if (!mover_->lay_out_unit_planes(p.get(), ext) || !mover_->source_epochs_current()) continue;
        auto ra = std::make_shared<ReadAheadUnit>(
            this, ((uint64_t)pass << 24) | ((uint64_t)layer << 8) | (uint64_t)pos, ext, 2,
            hx_ra_->piece_bytes(), &FileStager::hx_ra_read, mover_, p);
        {
            std::lock_guard<std::mutex> lk(hx_ra_mu_);
            /* The rank may have reached the layer while the unit was laid out. */
            if (ra_too_late(hx_ra_at_.load(std::memory_order_acquire), pass, li)) {
                ++hx_stats_.ra_late;
                return;
            }
            hx_ra_live_.push_back(RaEntry{ ra, p, pass, li, u });
            ++hx_stats_.ra_planned;
        }
        hx_ra_->submit(std::move(ra));
    }
}

void FileStager::hx_ra_drop_stale(uint32_t pass, int32_t li) {
    std::vector<RaEntry> stale;
    {
        std::lock_guard<std::mutex> lk(hx_ra_mu_);
        for (size_t i = 0; i < hx_ra_live_.size();) {
            const RaEntry& e = hx_ra_live_[i];
            if ((int32_t)(e.pass - pass) < 0 || (e.pass == pass && e.li <= li)) {
                stale.push_back(std::move(hx_ra_live_[i]));
                if (i + 1 < hx_ra_live_.size()) hx_ra_live_[i] = std::move(hx_ra_live_.back());
                hx_ra_live_.pop_back();
                continue;
            }
            ++i;
        }
    }
    for (RaEntry& e : stale) {
        hx_ra_->cancel(*e.u);
        ++hx_stats_.ra_stale;
        hx_stats_.ra_dropped_bytes += (uint64_t)e.u->bytes_in();
    }
}

void FileStager::hx_ra_begin(int32_t li) {
    hx_ra_at_.store(((uint64_t)trace_pass_ << 32) | (uint32_t)(li + 1), std::memory_order_release);
    hx_ra_here_.clear();
    {
        std::lock_guard<std::mutex> lk(hx_ra_mu_);
        for (size_t i = 0; i < hx_ra_live_.size();) {
            if (hx_ra_live_[i].pass == trace_pass_ && hx_ra_live_[i].li == li) {
                hx_ra_here_.push_back(std::move(hx_ra_live_[i]));
                if (i + 1 < hx_ra_live_.size()) hx_ra_live_[i] = std::move(hx_ra_live_.back());
                hx_ra_live_.pop_back();
                continue;
            }
            ++i;
        }
    }
    hx_ra_drop_stale(trace_pass_, li);   /* layers this rank passed without deciding them */
}

std::shared_ptr<Mover::HostPlanes> FileStager::hx_ra_take(int32_t u, bool* ready,
                                                          std::shared_ptr<ReadAheadUnit>* urgent) {
    *ready = false;
    ++hx_stats_.ra_misses;
    RaEntry* e = nullptr;
    for (RaEntry& r : hx_ra_here_)
        if (r.u && r.unit == u) { e = &r; break; }
    if (!e) return nullptr;
    ++hx_stats_.ra_miss_planned;
    std::shared_ptr<ReadAheadUnit> ra = std::move(e->u);
    std::shared_ptr<Mover::HostPlanes> p = std::move(e->p);
    if (ra->whole()) {   /* every piece in: the epoch is checked before the jobs go out */
        ++hx_stats_.ra_whole;
        hx_stats_.ra_used_bytes += (uint64_t)ra->bytes_in();
        ++hx_ra_handed_;
        *ready = true;
        return p;
    }
    /* No piece started: the layer reads it as any miss, into the buffers laid out for it. */
    if (hx_ra_->take_whole(*ra)) {
        ++hx_stats_.ra_fresh;
        return p;
    }
    /* Barely started: the demand read with the plane mirror's share beats the container alone for
     * the rest (or ..._PREFETCH_FINISH=0). A fresh buffer, since pieces in flight still land in
     * this one. */
    if (!hx_ra_finish_ || (int64_t)ra->started() * hx_ra_->piece_bytes() < mover_->decode_mirror_bytes()) {
        hx_ra_->cancel(*ra);
        ++hx_stats_.ra_fresh;
        hx_stats_.ra_dropped_bytes += (uint64_t)ra->bytes_in();
        return nullptr;
    }
    hx_ra_->make_urgent(*ra);
    ++hx_stats_.ra_finished;
    hx_stats_.ra_used_bytes += (uint64_t)ra->bytes_in();
    *urgent = std::move(ra);
    return p;
}

void FileStager::hx_ra_drop(RaEntry& e, bool hit) {
    if (!e.u) return;
    hx_ra_->cancel(*e.u);
    ++(hit ? hx_stats_.ra_cached : hx_stats_.ra_unselected);
    hx_stats_.ra_dropped_bytes += (uint64_t)e.u->bytes_in();
    e.u.reset();
    e.p.reset();
}

/* Inside the demand read: the urgent units' last pieces, then the units with a failed piece read
 * again whole into fresh buffers, and the container's epoch. */
bool FileStager::hx_ra_settle(int rows) {
    std::vector<Mover::HostPlanes*> again;
    for (size_t i = 0; i < hx_miss_ra_.size() && i < hx_miss_.size(); ++i) {
        std::shared_ptr<ReadAheadUnit>& ra = hx_miss_ra_[i];
        if (!ra) continue;
        hx_stats_.ra_wait_ns += hx_ra_->wait(*ra);
        if (!ra->whole()) {
            ++hx_stats_.ra_refetched;
            hx_miss_[i] = hx_planes_get(hx_miss_[i]->unit);
            again.push_back(hx_miss_[i].get());
        }
        ra.reset();
    }
    if (!again.empty() && !mover_->read_unit_planes(again.data(), (int)again.size(), rows)) return false;
    return mover_->source_epochs_current();
}

void FileStager::hx_bg_loop() {
    std::vector<uint8_t> scratch;
    bool told = false;
    std::unique_lock<std::mutex> lk(hx_bg_mu_);
    for (;;) {
        hx_bg_cv_.wait(lk, [&] { return hx_bg_stop_ || !hx_bg_q_.empty(); });
        if (hx_bg_q_.empty()) return;   /* stopping, nothing left */
        std::shared_ptr<Mover::HostPlanes> p = std::move(hx_bg_q_.front());
        hx_bg_q_.pop_front();
        ++hx_bg_busy_;
        lk.unlock();
        /* RADIANCE_EXPERIMENTAL_DS41_INSERT_YIELD_US=N: an insert streams ~37 MB through DRAM, which
         * the pool's workers are bound by; it waits (asleep) up to N us for the pool to drain. */
        static const int64_t yield_ns = [] {
            const char* y = std::getenv("RADIANCE_EXPERIMENTAL_DS41_INSERT_YIELD_US");
            return y && *y ? (int64_t)std::clamp(std::atoi(y), 0, 100000) * 1000 : (int64_t)0;
        }();
        if (yield_ns > 0 && hx_pool_) hx_pool_->wait_idle(yield_ns);
        const int rc = mover_->insert_unit_planes(*p, &scratch);
        if (rc < 0 && !told) {
            RAD_WARN("host experts: relaying a missed unit into the RAM cache failed (%s); it stays "
                     "a miss", rad_strerror(rc));
            told = true;
        }
        /* ..._SIDE_STORE: the packed cache refused it over its raw-round budget, as it always will:
         * its planes stay for its later misses (hx_side_store.h). */
        if (rc >= 0 && hx_side_.on() && mover_->packed_refused(p->unit)) (void)hx_side_.offer(p);
        p.reset();
        lk.lock();
        --hx_bg_busy_;
        hx_bg_cv_.notify_all();
    }
}

void FileStager::drain_host_inserts() {
    std::unique_lock<std::mutex> lk(hx_bg_mu_);
    hx_bg_cv_.wait(lk, [&] { return hx_bg_q_.empty() && hx_bg_busy_ == 0; });
}

/* A layer's gate_up op, before its launch, on every pass: the addresses of its token rows,
 * sorted order and expert offsets (moe_gemm_ggml: a, w, sorted_tok, expert_offset, y), which
 * the next pass's hx_fetch reads. With CPU rows in flight they must be the ones hx_fetch read on
 * this pass, or the pool computed from the wrong bytes. */
/* RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM=1: before the first request the RAM cache takes this
 * rank's file-tier experts in the expert profile's order, as many as fit, the hottest inserted
 * last so LRU keeps them longest: planes read as the CPU path reads a miss (the plane mirror
 * included), relaid by the background insert threads. A fresh server's first requests otherwise
 * miss on nearly every expert the card does not hold. */
int FileStager::hx_warm() {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> units;
    for (const Layer& L : layers_)
        for (int32_t u : L.units)
            if (plan_->units[(size_t)u].profile_share > 0.0f && mover_->host_planes_eligible(u))
                units.push_back(u);
    std::stable_sort(units.begin(), units.end(), [&](int32_t a, int32_t b) {
        return plan_->units[(size_t)a].profile_share > plan_->units[(size_t)b].profile_share;
    });
    const int64_t pair = mover_->mapped_pair_bytes();   /* a gate_up and a down slot, with guards */
    /* The slabs are pinned first (a refused one stops the growth), so the set is what the cache
     * really holds: an estimate that overshot would drop the hottest, inserted last. */
    const int64_t slots = mover_->mapped_cache_pregrow();
    const int64_t fit = slots > 0 ? slots : mover_->mapped_cache_bytes() / pair;
    if ((int64_t)units.size() > fit) units.resize((size_t)std::max<int64_t>(fit, 0));
    std::reverse(units.begin(), units.end());
    size_t done = 0;
    for (size_t i = 0; i < units.size(); i += 8) {
        {   /* at most a few batches' planes wait for the insert threads */
            std::unique_lock<std::mutex> lk(hx_bg_mu_);
            hx_bg_cv_.wait(lk, [&] { return hx_bg_q_.size() < 24; });
        }
        std::vector<std::shared_ptr<Mover::HostPlanes>> batch;
        std::vector<Mover::HostPlanes*> ptrs;
        for (size_t k = i; k < std::min(units.size(), i + 8); ++k) {
            batch.push_back(hx_planes_get(units[k]));
            ptrs.push_back(batch.back().get());
        }
        if (!mover_->read_unit_planes(ptrs.data(), (int)ptrs.size(), (int)kEpRouteMaxRows)) {
            RAD_WARN("host experts: warming rank %d's RAM cache stopped after %zu units: a read failed",
                     plan_->rank, done);
            break;
        }
        {
            std::lock_guard<std::mutex> lk(hx_bg_mu_);
            for (auto& p : batch) hx_bg_q_.push_back(p);
        }
        hx_bg_cv_.notify_all();
        done += batch.size();
    }
    drain_host_inserts();
    RAD_INFO("host experts: rank %d's RAM cache warmed with the profile's %zu hottest file-tier "
             "experts (%s) in %.1f s", plan_->rank, done, humanb((int64_t)done * pair).c_str(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return RAD_OK;
}

/* The first pass's CPU work for layer li, at its gate_up's operands: hx_capture has just taken
 * the rows' addresses, which the router and scatter wrote before this op on stream s. Every
 * selected unit was published as an alias, so all of them must go to the CPU. */
int FileStager::hx_deferred(int32_t li, RadStream s) {
    hx_defer_li_ = -1;
    if (!hx_ops_[(size_t)li].x) {
        RAD_ERR("host experts: layer %d's gate_up operands are not the shape the CPU path reads, "
                "and its first pass gave the card only aliases", layers_[(size_t)li].layer);
        return RAD_E_STATE;
    }
    RAD_TRY(hx_fetch(li, s));
    RAD_TRY(rad_event_record(counts_ev_, s));
    RAD_TRY(rad_event_sync(counts_ev_));
    const size_t selected = want_.size();
    if (rt_) rt_units_ = want_;
    hx_choose(li);
    if (rt_) rt_choice(li, true);
    if (!want_.empty() || (selected && hx_li_ != li)) {
        RAD_ERR("host experts: layer %d's first pass left %zu of %zu units off the CPU, and the card "
                "has only aliases for them", layers_[(size_t)li].layer, want_.size(), selected);
        return RAD_E_STATE;
    }
    if (trace_fp_) trace_layer(layers_[(size_t)li].layer, 2 * (uint32_t)(hx_miss_.size() - (size_t)hx_side_handed_));
    if (hx_li_ != li) {
        if (rt_) rt_layer(li);
        return RAD_OK;
    }
    RAD_TRY(hx_submit());
    RAD_TRY(hx_submit_misses());
    hx_confirmed_ = true;   /* the pool was given exactly the rows this gate_up reads */
    if (rt_) rt_layer(li);
    return RAD_OK;
}

int FileStager::hx_capture(int32_t li, RadStream s, const RadTensor* t, int n_t) {
    HxOps& o = hx_ops_[(size_t)li];
    if (n_t < 5) return RAD_E_STATE;
    const RadTensor &A = t[0], &so = t[2], &of = t[3], &Y = t[4];
    const int64_t AR = A.rank == 2 ? A.shape[0] : 0, T = Y.rank == 2 ? Y.shape[0] : 0;
    const int64_t x_ld = AR > 1 ? A.stride[0] : kHxEmbd;
    const bool shape = A.data && so.data && of.data && A.dtype == RAD_BF16 && A.rank == 2 &&
        A.shape[1] == kHxEmbd && A.stride[1] == 1 && x_ld >= kHxEmbd &&
        (so.dtype == RAD_I32 || so.dtype == RAD_U32) && (of.dtype == RAD_I32 || of.dtype == RAD_U32) &&
        hx_numel(of) >= o.n_expert + 1;
    const bool live = hx_li_ == li && hx_stage_ == 2;
    if (live) {
        if (!shape || A.data != o.x || x_ld != o.x_ld || so.data != o.sorted || of.data != o.off ||
            AR != pass_tok_ || T != pass_tok_ * o.top_k || hx_numel(so) < T) {
            RAD_ERR("host experts: layer %d's gate_up reads rows other than the ones the pool was "
                    "given (%lld rows of %lld sorted, a pass of %lld)", layers_[(size_t)li].layer,
                    (long long)AR, (long long)T, (long long)pass_tok_);
            return RAD_E_STATE;
        }
        hx_confirmed_ = true;
        return RAD_OK;
    }
    /* An early-issued layer (..._EARLY_ISSUE): its rows' copies are queued from the addresses an
     * earlier pass's gate_up read, and the pool has not started, so rows found at new addresses are
     * fetched again from there, behind the first copies on the same stream (the router and the
     * scatter wrote them ahead of this op), and the counts' event moves behind them. The layer's
     * range list (..._FETCH_LIST) is armed for the old addresses and is written again at its next
     * fetch, once the copy reading it has run. */
    if (hx_early_li_ == li) {
        if (!shape || AR != pass_tok_ || T != pass_tok_ * o.top_k || hx_numel(so) < T) {
            RAD_ERR("host experts: layer %d's gate_up reads rows the CPU path cannot (%lld rows of "
                    "%lld sorted, a pass of %lld), and its card ops went ahead of its counts",
                    layers_[(size_t)li].layer, (long long)AR, (long long)T, (long long)pass_tok_);
            return RAD_E_STATE;
        }
        if (A.data != o.x || x_ld != o.x_ld || so.data != o.sorted || of.data != o.off) {
            ++hx_stats_.moved;
            ++hx_stats_.early_refetch;
            o.x = A.data;
            o.x_ld = x_ld;
            o.sorted = so.data;
            o.off = of.data;
            RAD_TRY(hx_fetch(li, s));
            RAD_TRY(rad_event_record(counts_ev_, s));
        }
        hx_early_seen_ = true;
        return RAD_OK;
    }
    /* Only the addresses of a routed pass that reads all its rows at once (no slice, so from row
     * 0) are kept: what hx_fetch reads from on such a pass. */
    if (shape && routed_ && AR == pass_tok_ && AR >= 1 && AR <= ep_rows_ &&
        T == AR * o.top_k && hx_numel(so) >= T) {
        if (o.x && (A.data != o.x || x_ld != o.x_ld || so.data != o.sorted || of.data != o.off))
            ++hx_stats_.moved;
        o.x = A.data;
        o.x_ld = x_ld;
        o.sorted = so.data;
        o.off = of.data;
    }
    return RAD_OK;
}

/* The layer's gather, before its launch: the pool's rows over the card's in its y_expert, on
 * the gather's stream, or with ..._POLICY=verify the card's rows compared with the pool's. */
int FileStager::hx_finish(RadStream s, const RadTensor* t, int n_t) {
    if (hx_async_) {   /* the misses' jobs join the ticket first */
        const auto w0 = std::chrono::steady_clock::now();
        const int rc = hx_mw_wait();
        hx_stats_.async_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - w0).count();
        if (rc < 0) {
            hx_drain();   /* the layer is given up: no second report of rows that reached no gather */
            return rc;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    hx_pool_->wait(&hx_ticket_);
    hx_stats_.wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - t0).count();
    for (int k = 0; k < hx_n_; ++k) mover_->release_host_unit(&hx_units_[k]);
    hx_n_ = 0;
    hx_miss_.clear();
    hx_miss_ra_.clear();
    const int32_t li = hx_li_;
    const bool confirmed = hx_confirmed_;
    hx_li_ = -1;
    hx_stage_ = 0;
    hx_confirmed_ = false;
    if (!confirmed) {
        RAD_ERR("host experts: layer %d reached its gather with no gate_up seen to read the rows "
                "the pool was given", layers_[(size_t)li].layer);
        return RAD_E_STATE;
    }
    const RadTensor* Y = n_t >= 1 ? &t[0] : nullptr;
    const int64_t last = hx_rows_.empty() ? -1 : hx_rows_.back();
    if (!Y || !Y->data || Y->rank != 2 || Y->dtype != RAD_BF16 || Y->shape[1] != kHxEmbd ||
        Y->stride[1] != 1 || Y->shape[0] <= last || (Y->shape[0] > 1 && Y->stride[0] < kHxEmbd)) {
        RAD_ERR("host experts: layer %d's gather does not read the down projection's rows as the "
                "pool wrote them", layers_[(size_t)li].layer);
        return RAD_E_STATE;
    }
    const int64_t ld = Y->shape[0] > 1 ? Y->stride[0] : kHxEmbd;
    const int64_t row_bytes = kHxEmbd * (int64_t)sizeof(uint16_t);
    /* Runs of consecutive rows, one copy each when the rows are packed. */
    auto runs = [&](auto&& copy) -> int {
        for (size_t i = 0; i < hx_rows_.size();) {
            size_t j = i + 1;
            while (j < hx_rows_.size() && hx_rows_[j] == hx_rows_[j - 1] + 1) ++j;
            const int64_t r0 = hx_rows_[i], n = (int64_t)(j - i);
            if (ld == kHxEmbd) {
                RAD_TRY(copy(r0, n));
            } else {
                for (int64_t r = r0; r < r0 + n; ++r) RAD_TRY(copy(r, 1));
            }
            i = j;
        }
        return RAD_OK;
    };
    char* const ydev = (char*)Y->data;
    if (hx_policy_ == kHxVerify) {
        RAD_TRY(runs([&](int64_t r, int64_t n) {
            return rad_memcpy_async(hx_chk_ + r * kHxEmbd, ydev + r * ld * 2, n * row_bytes, s);
        }));
        RAD_TRY(rad_event_record(hx_in_ev_, s));
        RAD_TRY(rad_event_sync(hx_in_ev_));
        for (int32_t r : hx_rows_) {
            const uint16_t* a = hx_out_ + (int64_t)r * kHxEmbd;
            const uint16_t* c = hx_chk_ + (int64_t)r * kHxEmbd;
            int bad = 0, first = -1;
            for (int i = 0; i < kHxEmbd; ++i)
                if (a[i] != c[i]) { ++bad; if (first < 0) first = i; }
            ++hx_stats_.verified;
            if (!bad) continue;
            if (++hx_stats_.differ <= 8)
                RAD_WARN("host experts: layer %d sorted row %d: %d of %lld values differ from the "
                         "card's (first at %d: 0x%04x on the CPU, 0x%04x on the card)",
                         layers_[(size_t)li].layer, r, bad, (long long)kHxEmbd, first, a[first],
                         c[first]);
        }
        return RAD_OK;
    }
    rad_dev_host_wrote();
    RAD_TRY(runs([&](int64_t r, int64_t n) {
        return rad_memcpy_async(ydev + r * ld * 2, hx_out_ + r * kHxEmbd, n * row_bytes, s);
    }));
    RAD_TRY(rad_event_record(hx_out_ev_, s));
    hx_out_rec_ = true;
    if (hx_told_++ == 0 && plan_->rank == 0)
        RAD_INFO("host experts: layer %d's first CPU rows (%zu) copied over the card's before its "
                 "gather", layers_[(size_t)li].layer, hx_rows_.size());
    return RAD_OK;
}

/* Gives back whatever a layer still holds: waits for its jobs, returns its leases and its
 * unsubmitted reservation. */
void FileStager::hx_drain() {
    hx_join_miss_post();   /* before hx_miss_ lets the planes go */
    if (hx_async_) (void)hx_mw_wait();
    if (hx_stage_ == 2 && hx_pool_) hx_pool_->wait(&hx_ticket_);
    if (hx_reserved_ && hx_pool_) hx_pool_->unreserve(hx_reserved_);
    hx_reserved_ = 0;
    if (hx_units_)
        for (int k = 0; k < hx_n_; ++k) mover_->release_host_unit(&hx_units_[k]);
    hx_n_ = 0;
    hx_miss_.clear();
    /* Units a layer given up was waiting for: no further piece (the ones in flight keep their
     * buffers until they finish). */
    for (auto& r : hx_miss_ra_)
        if (r && hx_ra_) hx_ra_->cancel(*r);
    hx_miss_ra_.clear();
    hx_li_ = -1;
    hx_stage_ = 0;
    hx_confirmed_ = false;
    hx_early_li_ = -1;
    hx_early_seen_ = false;
}

int FileStager::operands(rad_op h, RadStream s, const RadTensor* t, int n_t) {
    /* ..._ROUTE_TRACE: a routed layer's router select op, before its launch (rt_capture). */
    if (rt_ && (size_t)h < rt_li_of_op_.size() && rt_li_of_op_[(size_t)h] >= 0)
        rt_capture(rt_li_of_op_[(size_t)h], t, n_t);
    if (!hx_pool_) return RAD_OK;
    if ((size_t)h < first_of_.size() && first_of_[(size_t)h] >= 0) {
        const int32_t li = first_of_[(size_t)h];
        if ((size_t)li < hx_ops_.size() && hx_ops_[(size_t)li].gu == (int32_t)h) {
            RAD_TRY(hx_capture(li, s, t, n_t));
            return hx_defer_li_ == li ? hx_deferred(li, s) : RAD_OK;
        }
        return RAD_OK;
    }
    /* An early-issued layer's CPU share is decided at the first of its gathers the pass issues,
     * behind its card ops; then that gather waits for the CPU rows below as any other. */
    if (hx_early_li_ >= 0)
        for (int32_t g : hx_ops_[(size_t)hx_early_li_].gather)
            if (g >= 0 && (int32_t)h == g) {
                RAD_TRY(hx_early_decide());
                break;
            }
    if (hx_li_ < 0 || hx_stage_ != 2) return RAD_OK;
    for (int32_t g : hx_ops_[(size_t)hx_li_].gather)
        if (g >= 0 && (int32_t)h == g) return hx_finish(s, t, n_t);
    return RAD_OK;
}

int FileStager::finish(int32_t b, RadStream s) {
    RAD_TRY(rad_event_record(free_[b], s));
    free_rec_[b] = true;
    mover_->unstage_units(staged_[b]);
    staged_[b].clear();
    holds_[b] = -1;
    partial_[b] = false;
    pending_[b] = false;
    return RAD_OK;
}

int FileStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;
    if (head_status_<0) return head_status_;
    if (head_.configured()) {
        //Preserve the OLD owner stream until its head readers are actually drained.
        if (head_.pending() && head_compute_ != s) { RAD_TRY(head_revoke()); *resync=true; }
        head_compute_ = s;
        if (head_first_) { head_first_ = false; RAD_TRY(head_prepare(s, resync)); }
    }
    if ((int32_t)h > last_issued_) last_issued_ = (int32_t)h;

    /* Read the pass's first routed layer at its first op of any kind -- unless the pass reads
     * routed experts only, which cannot be known before the layer's router has run. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (!routed_ && it != layers_.end()) {
            const int32_t li = (int32_t)(it - layers_.begin());
            if (will_issue(li)) {
                RAD_TRY(start(li));
                *resync = true;
            }
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    if (li == cur_) {
        /* The layer's first op again: a block that slices the pass routes each slice afresh, so
         * an expert aliased for the slice before may be wanted now. Read the whole layer behind
         * the ops already issued (finish records them; the copies wait on it). */
        const int b = li & 1;
        if (holds_[b] != li || !partial_[b]) return RAD_OK;
        /* CPU rows of the slice before not yet copied (or, early-issued, not yet chosen): a slice's
         * gate_up ahead of its gather. */
        if ((hx_li_ == li && hx_stage_ == 2) || hx_early_li_ == li) {
            RAD_ERR("host experts: layer %d's gate_up issued again before its gather",
                    layers_[(size_t)li].layer);
            return RAD_E_STATE;
        }
        hx_drain();
        RAD_TRY(finish(b, s));
        RAD_TRY(start(li));
        if (holds_[b] == li && pending_[b]) RAD_TRY(rad_event_wait(s, done_[b]));
        *resync = true;
        return RAD_OK;
    }

    /* Entering layer `li`: the layer before is behind every op issued so far on this stream. */
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0 && holds_[b] != li) { RAD_TRY(finish(b, s)); *resync = true; }
    /* A layer the read-ahead skipped (the learned bound said the pass ends before it), or every
     * layer of a routed-only pass, is read here rather than issued without addresses. */
    if (holds_[li & 1] != li) {
        bool read = false;
        const bool ep = ep_counts_ && pass_tok_ > 0 && pass_tok_ <= ep_rows_ &&
                        (size_t)li < ep_layer_eligible_.size() &&
                        ep_layer_eligible_[(size_t)li];
        if (routed_ && ep) RAD_TRY(start_ep_routed(li, s, &read));
        if (routed_ && !ep) RAD_TRY(start_routed(li, s, &read));
        if (!read) RAD_TRY(start(li));
        *resync = true;
    }
    if (holds_[li & 1] == li && pending_[li & 1]) RAD_TRY(rad_event_wait(s, done_[li & 1]));
    cur_ = li;

    const int32_t next = li + 1;
    if (!routed_ && (size_t)next < layers_.size() && will_issue(next)) {
        RAD_TRY(start(next));
        *resync = true;
    }
    return RAD_OK;
}

int FileStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    /* An early-issued layer whose gather never came is lost too: its card ops ran on aliases. */
    const bool lost = (hx_li_ >= 0 && hx_stage_ == 2) || hx_early_li_ >= 0;
    if (lost) RAD_ERR("host experts: layer %d's CPU rows reached no gather before the pass ended",
                      layers_[(size_t)(hx_early_li_ >= 0 ? hx_early_li_ : hx_li_)].layer);
    hx_drain();
    /* A cut pass ends early by design: its last op says nothing of where a whole pass ends. */
    if (pass_full_ == INT32_MAX) pass_last_[pass_kind_] = last_issued_;
    int rc = lost ? RAD_E_STATE : RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0) {
            const int r = finish(b, s);
            if (r < 0 && rc >= 0) rc = r;
        }
    cur_ = -1;
    return rc<0 ? rc : head_status_;
}

}  /* namespace rad */

/* stager.cpp -- see stager.h. */
#include "place/stager.h"
#include "place/dspark_native_scope.h"
#include "place/ds41_index_memory.h"
#include "../../arch/deepseek41/ds41_index_memory.h"
#include "place/router_predictor.h"
#include "ds41_host_experts.h"

#include <sys/prctl.h>

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
    /* One sequence: its passes are told apart as one sequence's (dspark_head_tail_pass). */
    if (!mover_ || plan.dynamic || plan.world_size!=4 || plan.rank!=prog.ctx.rank ||
        prog.ctx.max_tok!=16 || prog.ctx.max_seqs!=1 || !counts_ || !ep_counts_ || !dspark_native_program_scope(prog) ||
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
    /* The card path's threads use this stager's planes and insert queue: they stop first. */
    p1_.reset();
    if (p1_end_) rad_event_destroy(p1_end_);
    if (p1_lend_ev_) rad_event_destroy(p1_lend_ev_);
    if (p1_vev_) rad_event_destroy(p1_vev_);
    for (void* p : { (void*)p1_vidx_, (void*)p1_vx_, (void*)p1_vy_, (void*)p1_vh_ })
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
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
    if (hx_wake_ev_) rad_event_destroy(hx_wake_ev_);
    if (scratch_) rad_dev_free(scratch_, RAD_MEM_HOST_PINNED);
}

int FileStager::init(const Program& prog, const Plan& plan, Mover* mover,
                     RouteCounts routes, void* routes_ud, CountsRow counts,
                     EpCountsRow ep_counts) {
    //No owner/program switch while temporary addresses or old copy generations survive.
    RAD_TRY(shutdown_head_cache());
    if (head_.configured()) return RAD_E_STATE; //a head-cache FileStager is single-initialization
    layers_.clear();
    trace_requested_ = m1_trace_opt_in(std::getenv(kM1PrefetchTraceEnv));
    if (const char* pf = std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE"); pf && !std::strcmp(pf, "1"))
        prof_on_ = true;
    ep_counts_ = std::move(ep_counts);
    ep_program_ = &prog;
    ep_stats_ = {};
    if (!mover) return RAD_OK;

    /* Routing-aware reads. The source may still decline per layer (a layer that has not
     * reported this pass is read whole). */
    routes_ = routes;
    routes_ud_ = routes_ud;
    scratch_ = nullptr;
    scratch_n_ = 0;
    if (routes_ && prog.meta.n_expert > 0) {
        scratch_n_ = prog.meta.n_expert;
        scratch_ = (int32_t*)rad_dev_alloc((size_t)scratch_n_ * sizeof(int32_t),
                                           RAD_MEM_HOST_PINNED);
        if (!scratch_) { scratch_n_ = 0; routes_ = nullptr; }
    }

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
    n_model_layers_ = (int32_t)prog.meta.n_layers;
    RAD_TRY(tall_init(prog));
    RAD_TRY(init_head_cache(prog, plan));
    RAD_TRY(hx_init(prog));
    RAD_TRY(p1_init(prog));
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
    prof_last_counts_ = {};   /* ..._LCP_PROFILE: a pass's first layer has no span */
    pass_full_ = b->consumer_full > 0 ? b->consumer_full : INT32_MAX;
    lcp_admit_row_ = lcp_ ? std::max<int32_t>(b->lcp_admit_row, 0) : 0;
    if (b->consumer_row0 && n_model_layers_ > 0) pass_row0_.assign(b->consumer_row0, b->consumer_row0 + n_model_layers_);
    else pass_row0_.clear();
    /* Prompt-lookup speculation beside LCP (lcp_init took it; ep_route_counts.h): every pass with
     * speculative rows must be the verify step the admission covers, and no drafter pass may come. Counted,
     * and said once: the shutdown line's count outside the band must be 0 (ab_window.sh sets the arm aside
     * otherwise). */
    if (lcp_ && lcp_spec_ && (b->n_spec > 0 || b->draft_pass != 0)) {
        const bool cut = b->consumer_full != 0 || b->consumer_row0 != nullptr;
        if (b->n_spec > 0 && b->draft_pass == 0) {
            ++hx_stats_.spec_passes;
            hx_stats_.spec_rows += (uint64_t)std::max<int64_t>(b->n_tok, 0);
            hx_stats_.spec_max_rows = std::max<uint64_t>(hx_stats_.spec_max_rows, (uint64_t)std::max<int64_t>(b->n_tok, 0));
        }
        if (!ds41_verify_pass_covered(b->phase == RAD_PHASE_DECODE, b->draft_pass, b->n_spec, b->n_tok,
                                      b->n_tok_decode, cut, b->lcp_admit_row) ||
            b->n_tok > lcp_spec_rows_) {
            if (hx_stats_.spec_off++ == 0)
                RAD_ERR("large-chunk prefill: a pass outside the verify steps prompt-lookup speculation was "
                        "admitted for (phase %d, draft pass %d, %lld tokens, %lld of them decode rows, %d "
                        "speculative, %s, admission row %d; covered: a trunk decode pass of at most %lld rows, "
                        "uncut): its rows are not shown to be the engine's without speculation",
                        (int)b->phase, (int)b->draft_pass, (long long)b->n_tok, (long long)b->n_tok_decode,
                        (int)b->n_spec, cut ? "cut by the consumer skip" : "uncut", (int)b->lcp_admit_row,
                        (long long)lcp_spec_rows_);
        }
    }
    hx_defer_li_ = -1;
    hx_wake_li_ = -1;
    /* Prefill Phase 1: a pure prefill pass of the threshold's rows or more runs every routed expert row
     * of its layers of that many rows on the cards (p1_layer_on). Its first copy waits on the last
     * readers of the file buffers before it. */
    p1_armed_ = false;
    p1_li_ = -1;
    const bool p1_arms = p1_on() && lcp_ && p1_pass_arms(*b, p1_min_rows_, ep_rows_);
    /* The lent ring (card_expert_stager.h): a pass that arms the card path takes the residents' slots
     * back, and its first ring copy waits on their last reader too; any other pass, once the grace has
     * run out, reloads them and publishes those whose copies have passed. Here only, before the run
     * phase brings the weight tables up to date. With the lent arena a pass at the full plan takes them
     * back as well (its buffers reach the arena's top), its ops behind any reload copy still queued. */
    p1_lend_took_ = false;
    if (p1_on() && p1_->lending()) {
        p1_lend_took_ = p1_arms || (p1_arena_full_ && p1_arena_full_());
        /* ..._P1_LEND_AUDIT: a resident withdrawn at this pass's begin is read back behind the end of the
         * pass before, the residents' last reader. */
        if (p1_tun_.lend_audit) p1_->lend_audit_behind(p1_lend_rec_ ? p1_lend_ev_ : nullptr);
        (void)p1_->lender().begin_pass(p1_lend_took_, p1_now_ms());
        /* The lent slots' heat (..._P1_LEND_HEAT): the policy's turn, here only (residency changes before
         * the weight tables are brought up to date), with the heat counted through the pass before this
         * one; then this pass's rows age it, if it is one that counts. */
        if (p1_heat_) {
            ++p1_lend_pass_;
            p1_heat_policy();
            p1_heat_pass_ = p1_heat_counts(*b);
            if (p1_heat_pass_) p1_heat_->begin_pass(b->n_tok);
        }
    }
    if (p1_arms) {
        const RadEvent hand[3] = { free_rec_[0] ? free_[0] : nullptr, free_rec_[1] ? free_[1] : nullptr,
                                   p1_lend_rec_ ? p1_lend_ev_ : nullptr };
        if (p1_->begin_pass(b->n_tok, hand, 3) < 0) {
            RAD_ERR("ds41 p1: an armed pass could not begin; this pass goes to the CPU path");
        } else {
            p1_armed_ = true;
            p1_pass_rows_ = b->n_tok;
            p1_t0_ = std::chrono::steady_clock::now();
            p1_layers_ = 0;
            p1_cpu_layers_ = 0;
        }
    }
    ++trace_pass_;
    if (rt_) rt_pass(b);
    /* Under LCP every routed unit is the CPU's (hx_capped_), so a chunk of up to C tokens reads only
     * what it routes to as well: the card's copy path has no room for a whole layer. */
    routed_ = counts_host_ && b->n_tok > 0 &&
              (b->n_tok <= kRoutedRows || (lcp_ && hx_capped_ && b->n_tok <= ep_rows_));
    /* A one-token pass, or with ..._PREFETCH_MULTIROW a trunk decode pass of 2..N rows. */
    hx_ra_pass_ = ra_pass_reads_ahead(b->phase == RAD_PHASE_DECODE, b->draft_pass, b->n_tok, hx_ra_rows_);
    /* Two-batch overlap (stager.h, tbo_): the architecture declared it (rad_declare_two_batch) and
     * issues this pass as two micro-batches, so each routed layer arrives as slice A, then slice B. */
    tbo_ = ep_program_ && ep_program_->two_batch && tbo::step_splits(b);
    tbo_tok_ = b->n_tok;
    tbo_slice_ = 0;
    /* The aux expert worker takes layers of trunk passes only, never a sliced one. */
    aux_pass_ = aux_ && b->draft_pass == 0 && !tbo_;
    /* The dynamic worker (work/aux-dyn/SPEC.md): no lane is held across passes (a safety net: every layer leaves
     * it), and the online set counts the tokens of the passes it counts the hits of (B.2). */
    if (aux_in_lane_) aux_lane_leave();
    if (aux_online_ && aux_pass_ && b->n_tok <= ds41aux::kMaxRows) aux_hit_tok_ += (uint64_t)b->n_tok;
    if (tbo_) tbo_enter(0);
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

int FileStager::start(int32_t li, RadStream s) {
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
    /* Routing-aware reads: the trigger (this call) runs at the layer's first expert op,
     * after the layer's router on the same stream, so this pass's counts are ordered
     * before the copy. The stream sync costs far less than a gigabyte-scale layer read;
     * a source that declines (no report yet this pass) reads the whole layer. */
    const int32_t* ecnt = nullptr;
    int32_t n_expert = 0;
    if (routes_) {
        int64_t n = 0;
        const int32_t* dev = routes_(routes_ud_, layers_[(size_t)li].layer, &n);
        if (dev && n > 0 && n <= scratch_n_ &&
            rad_memcpy_async(scratch_, dev, (size_t)n * sizeof(int32_t), s) >= 0 &&
            rad_stream_sync(s) >= 0) {
            ecnt = scratch_;
            n_expert = (int32_t)n;
        }
    }
    const int rc = mover_->stage_file_units(layers_[(size_t)li].units, buf_[b], cap_,
                                            free_rec_[b] ? free_[b] : nullptr, done_[b],
                                            &staged_[b], &skipped_[b], ecnt, n_expert,
                                            &promoted_[b], cache_policy);
    if (rc < 0) return rc;
    partial_[b] = false;
    pending_[b] = !staged_[b].empty() || !promoted_[b].empty();
    if (!staged_[b].empty() || !promoted_[b].empty()) holds_[b] = li;
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
    if (!ep_route_current(v, ep_program_, s, v.generation, L.layer, layer_rows(li)) ||
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
    /* Prefill Phase 1 decides a layer at a time: in an armed pass a layer below the threshold (the
     * per-row cut's deep layers of a prompt's last chunk; at 16 rows or fewer the gemv rows, which
     * honour no skip mark) takes the CPU path below as a pass of its height does (p1_layer_on). */
    const bool p1 = p1_layer_on(li);
    if (p1_armed_ && !p1) ++p1_cpu_layers_;
    /* With the pool, the pass's token rows and sorted order ride the same wait (hx_fetch). */
    const bool hx = !p1 && hx_pool_ && !head_.configured() && (size_t)li < hx_ops_.size() &&
                    hx_ops_[(size_t)li].gu >= 0 && hx_ops_[(size_t)li].x;
    /* Never seen at its gate_up (the first pass): nothing to fetch the rows from yet, so the CPU's
     * part waits for the gate_up's operands (hx_deferred) and the card gets aliases for all. */
    const bool defer = !p1 && !hx && hx_defer_ok_ && hx_pool_ && !head_.configured() &&
                       (size_t)li < hx_ops_.size() && hx_ops_[(size_t)li].gu >= 0 &&
                       hx_miss_need(li) <= hx_miss_rows_;
    /* Same actual producer stream; this host-wait event is the completion proof.
     * The Ctx witness means successfully issued, never completed before this wait.
     * With ..._FETCH_LIST the counts row and the pool's rows go as one ranged copy. */
    int listed = 0;
    if (hx && hx_fl_) RAD_TRY(listed = hx_fetch_list(li, v.row, s));
    if (!listed) {
        RAD_TRY(rad_memcpy_async(counts_host_, v.row, 96 * (int64_t)sizeof(int32_t), s));
        if (hx) RAD_TRY(hx_fetch(li, s));
    }
    /* Prefill Phase 1's admission: in a pass whose rows reach the admission row, the sorted order and
     * the offsets ride the counts' wait (from the addresses the layer's gate_up last read, as hx_fetch
     * takes them; hx_capture counts it if this pass's gate_up reads others). */
    p1_order_ = false;
    if (p1 && lcp_admit_row_ > 0 && lcp_admit_row_ < pass_tok_ && hx_idx_ && (size_t)li < hx_ops_.size() &&
        hx_ops_[(size_t)li].sorted && hx_ops_[(size_t)li].off) {
        const HxOps& o = hx_ops_[(size_t)li];
        const int64_t T = layer_rows(li) * o.top_k;
        if (T > 0 && T <= hx_sorted_max()) {
            RAD_TRY(rad_memcpy_async(hx_idx_, o.sorted, T * (int64_t)sizeof(int32_t), s));
            RAD_TRY(rad_memcpy_async(hx_idx_ + hx_sorted_max(), o.off, (o.n_expert + 1) * (int64_t)sizeof(int32_t), s));
            p1_order_ = true;
        }
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
    /* ..._HOST_EXPERTS_PREWAKE: the router's event, the pool's pre-wake, then the counts (all in the
     * router wait). */
    std::chrono::steady_clock::time_point t_woke{};
    if (hx && hx_wake_) RAD_TRY(hx_wake_route(li, &t_woke));
    RAD_TRY(rad_event_sync(counts_ev_));
    const auto t_counts = std::chrono::steady_clock::now();
    if (t_woke.time_since_epoch().count())
        hx_stats_.wk_rt_gap_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_counts - t_woke).count();
    const uint64_t sync_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_counts - t_sync).count();
    ep_stats_.sync_ns += sync_ns;
    if (pass_tok_ == 1) { ++ep_stats_.d_reads; ep_stats_.d_sync_ns += sync_ns; }
    if (prof_on_) prof_counts(li, sync_ns, t_counts);
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
    if (p1_heat_pass_) p1_heat_count(li);   /* ..._P1_LEND_HEAT: the layer's rows by unit */
    want_.clear(); skip_.clear();
    for (int32_t u : L.units) {
        if (mover_->ring_resident(u)) {
            if (bank_track_) {
                bool selected=false;
                if (!ep_route_unit_selected(counts_host_,plan_->units[(size_t)u].expert,&selected)) return RAD_E_STATE;
                bank_selected_ += selected;
            }
            continue;   /* a lent-ring resident: the card's, as a VRAM one */
        }
        bool selected = false;
        if (!ep_route_unit_selected(counts_host_, plan_->units[(size_t)u].expert, &selected))
            return RAD_E_STATE;
        (selected ? want_ : skip_).push_back(u);
    }
    /* The access trace takes want_ before the CPU split; its record is written after the stage
     * below, with the stage's drive reads. */
    if (trace_fp_) trace_units_ = want_;
    /* Prefill Phase 1: every selected unit goes to the cards' ring (p1_layer). */
    if (p1) {
        RAD_TRY(p1_layer(li, s));
        if (trace_fp_) trace_layer(L.layer, 0);
        ++ep_stats_.reads;
        ep_stats_.units += want_.size();
        if (!total) ++ep_stats_.zero_local;
        *read = true;
        return RAD_OK;
    }
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
         * predictor's thread does not compete with the pool's wake-up. A multi-row decode pass
         * (..._PREFETCH_MULTIROW) posts every row; a one-token pass its one, as always. */
        if (hx_ra_ && hx_pred_ && hx_ra_pass_ && plan_->rank == 0 && moe_runs(L.layer + 2)) {
            hx_pred_->post(trace_pass_, L.layer, hx_x_, (int)pass_tok_);
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
    /* The lent ring's slots are the file buffers' idle ones (p1_lend_init), so with it nothing may take
     * the card's copy path (hx_init sends every miss to the CPU): staging would overwrite residents. */
    if (p1_ && p1_->lending() && !want_.empty()) {
        RAD_ERR("ds41 p1: lent ring: %zu routed units of layer %d reached the card's copy path", want_.size(), (int)li);
        return RAD_E_STATE;
    }
    const int stage_status = mover_->stage_file_units(want_, buf_[b], available,
        free_rec_[b] ? free_[b] : nullptr, done_[b], &staged_[b], nullptr, nullptr, 0, nullptr,
        NativeFileCacheReadPolicy::Normal,trace);
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
        hx_cpu_units();
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
    /* The lent ring's slots are the file buffers' idle ones (p1_lend_init), so with it nothing may take
     * the card's copy path (hx_init sends every miss to the CPU): staging would overwrite residents. */
    if (p1_ && p1_->lending() && !want_.empty()) {
        RAD_ERR("ds41 p1: lent ring: %zu routed units of layer %d reached the card's copy path", want_.size(), (int)li);
        return RAD_E_STATE;
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

/* ---- large-chunk prefill's pairs: kernels of one op that give a row the same bits ----------------
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_LCP an op may resolve to more than one kernel over [1, C] only
 * where the kernels are known to agree byte for byte, row by row:
 *   - gemm_nt: gemm_bf16_nt_m16 forwards to gemm_bf16_nt_m64, and gemm_bf16_nt_tiled is
 *     byte-identical to m64 by contract (the same WMMA sequence a tile, the same split, the slices
 *     summed in the same order from 0.f; r4d_gemm_bf16_nt_tiled.hip);
 *   - all_reduce: the exact one-shot and two-shot of one rank count, with or without their narrow
 *     (`_lane`) launch, fold every element in f32 from 0 in ascending rank order (the one-shot
 *     already hands messages from ts_min elements to the two-shot on the same instance; a band
 *     above 102 rows of 5120 resolves to the two-shot row itself).
 * The family a kernel belongs to, or null for any other. */
const char* lcp_family(const std::string& op, const char* name) {
    if (!name) return nullptr;
    if (op == "gemm_nt" && (!std::strcmp(name, "gemm_bf16_nt_m16") || !std::strcmp(name, "gemm_bf16_nt_m64") ||
                            !std::strcmp(name, "gemm_bf16_nt_tiled")))
        return "gemm_nt bf16 (m16 / m64 / tiled)";
    if (op == "all_reduce") {
        int ranks = 0, used = 0;
        char kind[16] = {};
        if (std::sscanf(name, "ar_%15[a-z]_%drank_exact%n", kind, &ranks, &used) == 2 && used > 0 &&
            (!std::strcmp(kind, "oneshot") || !std::strcmp(kind, "twoshot"))) {
            const char* rest = name + used;
            static const char* const fam[9] = { nullptr, nullptr, "all_reduce exact (2 ranks)",
                                                "all_reduce exact (3 ranks)", "all_reduce exact (4 ranks)",
                                                nullptr, nullptr, nullptr, "all_reduce exact (8 ranks)" };
            if ((!*rest || !std::strcmp(rest, "_lane")) && ranks > 0 && ranks < 9) return fam[ranks];
        }
    }
    return nullptr;
}
}  /* namespace */

/* RADIANCE_EXPERIMENTAL_DS41_LCP=<C> (ep_route_counts.h): chunks of C tokens with a 16-token pass's
 * bits. Everything EXACT_TALL checks, at C rows: the step is C, every band of every routed MXFP4 GEMM
 * resolved to the gemv or its twin, and the twin as this libr4d declares it reaches C (a libr4d that
 * does not know the switch has no twin, or one bounded at 48, and is refused here: the switch is
 * verified, not assumed); every other op runs one kernel at every height or one of lcp_family's
 * pairs. Refused with the reason, too: EXACT_TALL set to anything but 1 beside it, speculation other
 * than prompt lookup alone in steps of at most 16 decode rows (ds41_lcp_spec: the verify steps the
 * bands below cover), the
 * fetch lists (sized for 48 rows) and the packed VRAM residents (whose gemv serves 16 rows). What
 * the CPU path needs (every routed unit the CPU's: ..._POLICY=all, ..._MISSES >= C, the file buffers
 * capped) is hx_init's; the chunk geometry and the sequences a step holds are the scheduler's
 * (Scheduler::init, which also refuses LCP unless every rank's stager admitted it). */
int FileStager::lcp_init(const Program& prog, int64_t C) {
    const char* xt = std::getenv(kExactTallEnv);
    std::string why;
    const int64_t T = prog.ctx.max_tok;
    /* RADIANCE_EXPERIMENTAL_DS41_P1 (card_expert_stager.h): the twin above 16 rows is the P1 row, and
     * the packed residents' gemv_vp may serve 16 or fewer (the P1 row decodes the packed form). */
    p1_state_ = ds41p1::env_state(std::getenv(ds41p1::kEnv));
    const bool p1 = p1_state_ == 1;
    /* The threshold, for hx_init's CPU bound (p1_cpu_rows); a value out of range is p1_init's refusal. */
    p1_min_rows_ = ds41p1::kDefMinRows;
    if (p1) {
        p1::Tunables t;
        if (p1::parse_tunables([](const char* k) { return std::getenv(k); }, &t, nullptr)) {
            p1_min_rows_ = t.min_rows;
            p1_lend_asked_ = t.lend > 0 || t.lend_arena > 0 || t.lend_slab > 0;
        }
    }
    /* Speculation (ep_route_counts.h, "prompt-lookup speculation beside large-chunk prefill"): taken only
     * as prompt lookup alone (no draft head declared) whose steps of decode rows fit the decode band, where
     * the band checks below cover a verify step as they cover a 16-token pass; refused as before otherwise. */
    const LcpSpec spec = ds41_lcp_spec(prog.ctx.max_seqs, prog.ctx.max_spec, prog.drafter.kind != RAD_DRAFT_NONE);
    if (xt && *xt && std::strcmp(xt, "1") != 0)
        why = fmt("%s=%s beside it: large-chunk prefill keeps EXACT_TALL's pins at every height; leave it unset "
                  "(or empty) or 1", kExactTallEnv, xt);
    else if (T != C)
        why = fmt("--max-num-batched-tokens is %lld: give %lld, the chunk the switch was set for (a step of "
                  "another size would run passes the bands below were not checked at, or fewer rows a chunk)",
                  (long long)T, (long long)C);
    else if (prog.ctx.max_spec > 0 && spec == LcpSpec::Rows)
        why = fmt("speculation is on (a window of %d): %s", prog.ctx.max_spec,
                  ds41_lcp_spec_rows_why(prog.ctx.max_seqs, prog.ctx.max_spec).c_str());
    else if (prog.ctx.max_spec > 0 && spec != LcpSpec::Lookup)
        why = fmt("speculation is on (a window of %d): verify steps and DSpark's drafter are not covered",
                  prog.ctx.max_spec);
    else if (const char* fl = std::getenv("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST"); fl && !std::strcmp(fl, "1"))
        why = "RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1: its range lists are sized for passes of up to 48 tokens";
    else if (const char* vp = std::getenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED"); !p1 && vp && !std::strcmp(vp, "1"))
        why = "RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1: the packed residents' gemv_vp serves passes of up to 16 "
              "tokens";
    if (!why.empty()) {
        RAD_ERR("%s=%lld (large-chunk prefill) refused: %s", kLcpEnv, (long long)C, why.c_str());
        return RAD_E_INVAL;
    }
    int n_moe = 0, n_twin = 0, n_ops = 0, n_split = 0, n_runs = 0;
    int64_t reach = 0;
    /* The multi-kernel ops admitted, by family: ops and the kernels seen. */
    std::map<std::string, std::pair<int, std::vector<std::string>>> fams;
    for (size_t h = 1; h < prog.ops.size(); ++h) {
        const OpInfo& op = prog.ops[h];
        if (op.bands.empty()) continue;
        ++n_ops;
        long long N = 0, K = 0, part = 0;
        (void)op.base.get_i("N", &N);
        (void)op.base.get_i("K", &K);
        /* The plugin's own height switches, for the startup line: the router's column split (a
         * gemm_nt_f32 over a rank's 96 of 384 columns) and the indexer's runs (row_part > 0). */
        if ((op.op == "gemm_nt_f32" || op.op == "gemm_nt_f32bf16") && N == 96) ++n_split;
        if (op.op == "index_score" && op.base.get_i("row_part", &part) && part > 0) ++n_runs;
        const char* ty = op.base.get_s("type");
        if (op.op == "moe_gemm_ggml" && ty && !std::strcmp(ty, "mxfp4")) {
            ++n_moe;
            for (const Band& b : op.bands) {
                const char* name = row_name(b);
                if (p1) {
                    /* Under P1: the gemv (or gemv_vp) to 16 rows, the P1 twin above. */
                    const bool low = name && b.hi <= kEpRouteMaxRows &&
                                     (!std::strcmp(name, ds41p1::kGemvRow) || !std::strcmp(name, ds41p1::kGemvVpRow));
                    const bool high = name && !std::strcmp(name, ds41p1::kTwinRow);
                    if (!low && !high) {
                        RAD_ERR("%s=1 beside %s=%lld refused: op %zu (moe_gemm_ggml, N %lld K %lld) resolves %s to %s: "
                                "the card path needs %s (or %s) to 16 tokens and %s above, where every expert row is "
                                "computed on the card with the skip and packed marks. %s",
                                ds41p1::kEnv, kLcpEnv, (long long)C, h, N, K, b.span.empty() ? "a band" : b.span.c_str(),
                                name ? name : "nothing", ds41p1::kGemvRow, ds41p1::kGemvVpRow, ds41p1::kTwinRow,
                                name && !std::strcmp(name, ds41p1::kLcpTwinRow)
                                    ? "This libr4d.so has LCP's twin, not the P1 row: load one built with the P1 row, "
                                      "with RADIANCE_EXPERIMENTAL_DS41_P1=1 in its environment"
                                    : "Load a libr4d.so with the P1 row");
                        return RAD_E_INVAL;
                    }
                    if (high) {
                        ++n_twin;
                        const int64_t r = row_m_bound(b.dom[RAD_DOMAIN_DEVICE].row);
                        reach = reach ? std::min(reach, r) : r;
                    }
                    continue;
                }
                const bool gemv = name && !std::strcmp(name, "moe_gemm_ggml_mxfp4_gemv");
                const bool twin = name && !std::strcmp(name, "moe_gemm_ggml_mxfp4_gemv_tall");
                if (!gemv && !twin) {
                    RAD_ERR("%s=%lld (large-chunk prefill) refused: op %zu (moe_gemm_ggml, N %lld K %lld) "
                            "resolves %s to %s, not to the decode band's kernel (moe_gemm_ggml_mxfp4_gemv, or "
                            "its twin moe_gemm_ggml_mxfp4_gemv_tall above 16 tokens), so a %lld-token chunk "
                            "would not give 16-token passes' bits. Load a libr4d.so whose twin reaches %lld "
                            "tokens (one built with %s), with the switch in its environment",
                            kLcpEnv, (long long)C, h, N, K, b.span.empty() ? "a band" : b.span.c_str(),
                            name ? name : "nothing", (long long)b.hi, (long long)C, kLcpEnv);
                    return RAD_E_INVAL;
                }
                if (twin) {
                    ++n_twin;
                    const int64_t r = row_m_bound(b.dom[RAD_DOMAIN_DEVICE].row);
                    reach = reach ? std::min(reach, r) : r;
                }
            }
            continue;
        }
        const char* first = nullptr;
        const char* fam = nullptr;
        std::vector<std::string> kernels;
        for (const Band& b : op.bands) {
            const char* name = row_name(b);
            if (!name) continue;
            if (!first) {
                first = name;
                fam = lcp_family(op.op, name);
                kernels.emplace_back(name);
                continue;
            }
            if (std::find(kernels.begin(), kernels.end(), name) != kernels.end()) continue;
            const char* f2 = lcp_family(op.op, name);
            if (!fam || !f2 || std::strcmp(fam, f2)) {
                RAD_ERR("%s=%lld (large-chunk prefill) refused: op %zu (%s) runs %s in one band and %s in %s: "
                        "nothing here says the two give a row the same bits, so a chunk of another height "
                        "could change it. Admitted pairs: gemm_nt's gemm_bf16_nt_m16 / _m64 / _tiled and "
                        "all_reduce's exact one- and two-shot",
                        kLcpEnv, (long long)C, h, op.op.c_str(), first, name,
                        b.span.empty() ? "another" : b.span.c_str());
                return RAD_E_INVAL;
            }
            kernels.emplace_back(name);
        }
        if (kernels.size() > 1) {
            auto& e = fams[fam];
            ++e.first;
            for (const std::string& k : kernels)   /* every kernel the family showed, once */
                if (std::find(e.second.begin(), e.second.end(), k) == e.second.end()) e.second.push_back(k);
        }
    }
    if (n_moe == 0 || (C > kEpRouteMaxRows && n_twin == 0) || (n_twin && reach < C)) {
        RAD_ERR("%s=%lld (large-chunk prefill) refused: %s", kLcpEnv, (long long)C,
                n_moe == 0 ? "the program has no routed MXFP4 GEMM (not DeepSeek-V4.1's EP graph)"
                : n_twin == 0
                    ? (p1 ? "no band resolved to the P1 row moe_gemm_ggml_mxfp4_gemv_tall_p1: load a libr4d.so with "
                            "it, with RADIANCE_EXPERIMENTAL_DS41_P1=1 and the LCP switch in its environment"
                          : "no band resolved to moe_gemm_ggml_mxfp4_gemv_tall: load a libr4d.so with the twin, with the "
                            "switch in its environment")
                    : fmt("this libr4d's twin moe_gemm_ggml_mxfp4_gemv_tall reaches %lld tokens, short of the "
                          "chunk: a libr4d that knows %s declares it to %lld", (long long)reach, kLcpEnv,
                          (long long)kEpRouteLcpMax).c_str());
        return RAD_E_INVAL;
    }
    /* The compact floor may cross the sparse launcher's internal48-row threshold. LCP's env
     * pin is authoritative only for known untuned index rows from its actual admitted DSO. */
    const char* compact = std::getenv(deepseek41::kIndexCompactEnv);
    if (compact && !std::strcmp(compact, "1")) {
        std::string reason;
        if (!ds41_index_memory_admit(prog, &reason)) {
            RAD_ERR("%s=1 refused after LCP admission: %s", deepseek41::kIndexCompactEnv, reason.c_str());
            return RAD_E_UNSUPPORTED;
        }
        if (plan_ && plan_->rank == 0)
            RAD_INFO("compact index score admitted: known untuned scalar/index selection rows from the "
                     "same loaded LCP library (%s=1); target256MiB, floor64, guard column retained",
                     deepseek41::kIndexCompactEnv);
    }
    ep_rows_ = C;
    lcp_ = C;
    lcp_reach_ = reach;
    /* RADIANCE_EXPERIMENTAL_DS41_LCP_PLANES: missed units read at once a layer (hx_stream_misses). */
    hx_lcp_planes_ = 48;
    if (const char* pl = std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP_PLANES"); pl && *pl)
        hx_lcp_planes_ = std::clamp(std::atoi(pl), 1, 96);
    /* RADIANCE_EXPERIMENTAL_DS41_LCP_INSERTQ: admitted misses queued for the insert threads at once. */
    hx_lcp_insertq_ = 16;
    if (const char* iq = std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP_INSERTQ"); iq && *iq)
        hx_lcp_insertq_ = std::clamp(std::atoi(iq), 0, 1024);
    lcp_moe_ = n_moe;
    lcp_twin_ = n_twin;
    lcp_split_ = n_split;
    lcp_runs_ = n_runs;
    lcp_ops_ = n_ops;
    lcp_pairs_.clear();
    int n_pairs = 0;
    for (const auto& f : fams) {
        std::string ks;
        for (const std::string& k : f.second.second) ks += (ks.empty() ? "" : ", ") + k;
        lcp_pairs_ += fmt("%s%d %s on %s", lcp_pairs_.empty() ? "" : "; ", f.second.first, f.first.c_str(),
                          ks.c_str());
        n_pairs += f.second.first;
    }
    lcp_pair_ops_ = n_pairs;
    lcp_spec_ = spec == LcpSpec::Lookup ? prog.ctx.max_spec : 0;
    lcp_spec_rows_ = lcp_spec_ ? ds41_verify_rows(prog.ctx.max_seqs, prog.ctx.max_spec) : 0;
    return RAD_OK;
}

int FileStager::tall_init(const Program& prog) {
    ep_rows_ = kEpRouteMaxRows;
    lcp_ = 0;
    lcp_spec_ = 0;
    lcp_spec_rows_ = 0;
    const int64_t lcp = ds41_lcp_state();
    if (lcp < 0) {
        const char* v = std::getenv(kLcpEnv);
        RAD_ERR("%s=%s: large-chunk prefill takes a chunk C, a multiple of 16 from %lld to %lld (0 or unset is "
                "off)", kLcpEnv, v ? v : "", (long long)kEpRouteLcpMin, (long long)kEpRouteLcpMax);
        return RAD_E_INVAL;
    }
    if (lcp > 0) return lcp_init(prog, lcp);
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
    RAD_TRY(hx_wake_init(prog));
    /* Asked for, the store must run (hx_side_init): host experts with misses on the CPU. */
    if (const char* v = side_store_asked(); v && !hx_side_.on()) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=%s: no store without host experts that compute "
                "the misses on the CPU (..._HOST_EXPERTS, ..._MISSES) from the packed RAM cache", v);
        return RAD_E_INVAL;
    }
    RAD_TRY(hx_ra_rows_init());   /* ..._PREFETCH_MULTIROW, once the read-ahead is (or is not) up */
    /* The misses the CPU must take: every pass's. Without the switch that is 16 tokens whatever
     * --max-num-batched-tokens says (as before); with it the step's (tall_init: ep_rows_ is the
     * step, at least 16). */
    /* Under P1 every layer pass of the threshold's rows or more is the card path's (p1_init), so the
     * CPU takes the layer passes below it only (p1_cpu_rows: 16 at the threshold 17), whatever the
     * chunk. With the lent ring (..._P1_LEND) the CPU takes every other pass's misses too (hx_need_rows):
     * a pass that does not arm P1 (a decode step of more rows than the bound, a mixed step) would
     * otherwise stage them in the file buffers, whose idle slots then hold published residents. */
    const int64_t need = hx_need_rows();
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
    if (!cap || std::atoi(cap) <= 0 || layers_.empty()) {
        if (lcp_ && p1_state_ == 1) {
            RAD_ERR("%s=1 beside %s=%lld refused: the card path's ring is the file buffers, sized by "
                    "RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS (2 or more; the window measures 6), and the CPU path "
                    "takes the layer passes below %s=%lld (host experts with ..._POLICY=all and ..._MISSES=%lld or "
                    "more)", ds41p1::kEnv, kLcpEnv, (long long)lcp_, ds41p1::kMinRowsEnv, (long long)p1_min_rows_,
                    (long long)need);
            return RAD_E_INVAL;
        }
        if (lcp_) {
            RAD_ERR("%s=%lld (large-chunk prefill) refused: the CPU must compute every routed unit of every "
                    "chunk (the card's copy path has no room for a chunk's experts): host experts with "
                    "..._POLICY=all, ..._MISSES=%lld and RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS set",
                    kLcpEnv, (long long)lcp_, (long long)lcp_);
            return RAD_E_INVAL;
        }
        return packed_ok();
    }
    bool every = hx_pool_ != nullptr;
    for (size_t li = 0; every && li < layers_.size(); ++li) {
        every = li < ep_layer_eligible_.size() && ep_layer_eligible_[li] && li < hx_ops_.size() &&
                hx_ops_[li].gu >= 0;
        for (int32_t u : layers_[li].units)   /* a unit the CPU cannot read would need the card */
            every = every && mover_->host_planes_eligible(u);
    }
    if (!every || !hx_defer_ok_ || hx_miss_rows_ < need || prog.ctx.max_tok > ep_rows_) {
        RAD_ERR("%sRADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=%s: the file buffers then hold no layer's "
                "units, so the CPU must compute every routed unit of every pass: host experts with "
                "..._POLICY=all and ..._MISSES=%lld on every routed layer, and passes of at most %lld "
                "tokens (--max-num-batched-tokens%s)",
                lcp_ ? fmt("%s=%lld (large-chunk prefill) refused: ", kLcpEnv, (long long)lcp_).c_str() : "",
                cap, (long long)need, (long long)ep_rows_,
                lcp_ || ds41_exact_tall_state() == 1
                    ? "" : "; taller with RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1, as far as libr4d keeps a "
                           "16-token pass's bits");
        return RAD_E_INVAL;
    }
    hx_capped_ = true;
    if (plan_->rank == 0)
        RAD_INFO("host experts: file buffers capped at %s units; every routed unit is the CPU's", cap);
    const int rc = packed_ok();
    if (rc >= 0 && lcp_) {
        lcp_note_admitted();
        if (plan_->rank == 0) {
            lcp_report();
            lcp_spec_report();
        }
    }
    return rc;
}

/* Prompt-lookup speculation taken beside an admitted LCP (rank 0; ep_route_counts.h): ab_window.sh
 * measures an Sg arm with an Lc part only with this line. */
void FileStager::lcp_spec_report() const {
    if (!lcp_spec_) return;
    RAD_INFO("large-chunk prefill: prompt-lookup speculation admitted (--num-speculative-tokens %d, no draft "
             "head): a verify step is a decode pass of at most %lld rows, inside the decode band of %lld (%s=%lld's "
             "band checks hold the routed GEMMs, the head and every other op to the kernels of a 16-token pass at "
             "that height); the CPU path takes its routed units by rows like any decode pass of that height, the "
             "card path never arms for it, and it admits every unit it routes to; a drafter's passes and steps of "
             "more decode rows stay refused",
             lcp_spec_, (long long)lcp_spec_rows_, (long long)kEpRouteMaxRows, kLcpEnv, (long long)lcp_);
}

/* The startup line of an admitted LCP (rank 0): ab_window.sh measures an Lc arm only with it, and a
 * plain arm only without it. */
void FileStager::lcp_report() const {
    if (p1_state_ == 1) {
        RAD_INFO("large-chunk prefill: passes of up to %lld tokens (%s=%lld): %d routed MXFP4 GEMMs on "
                 "%s (or %s) to 16 tokens and the P1 row %s above (%d bands; this libr4d's row reaches %lld); "
                 "multi-kernel ops admitted: %s; the other %d of %d ops on one kernel at every height; the "
                 "plugin's own height switches: the router split by columns (%d ops)%s, the indexer in %d extra "
                 "runs; libr4d keeps the sparse attention, the indexer and the norms on their decode-band "
                 "arithmetic; every routed expert row of a layer pass of %lld rows or more is computed on the cards "
                 "(%s=1, %s=%lld), the CPU path takes every routed unit of the layer passes below it, up to %lld "
                 "rows, in jobs of up to %d rows over %d pool slots",
                 (long long)lcp_, kLcpEnv, (long long)lcp_, lcp_moe_, ds41p1::kGemvRow, ds41p1::kGemvVpRow,
                 ds41p1::kTwinRow, lcp_twin_, (long long)lcp_reach_, lcp_pairs_.empty() ? "none" : lcp_pairs_.c_str(),
                 lcp_ops_ - lcp_moe_ - lcp_pair_ops_, lcp_ops_, lcp_split_, lcp_split_ ? " from 256 rows" : "",
                 lcp_runs_, (long long)p1_min_rows_, ds41p1::kEnv, ds41p1::kMinRowsEnv, (long long)p1_min_rows_,
                 (long long)p1_cpu_rows(), HostExpertPool::max_rows(), HostExpertPool::slots());
        return;
    }
    RAD_INFO("large-chunk prefill: passes of up to %lld tokens (%s=%lld): %d routed MXFP4 GEMMs on "
             "moe_gemm_ggml_mxfp4_gemv to 16 tokens and its twin moe_gemm_ggml_mxfp4_gemv_tall above (%d bands; "
             "this libr4d's twin reaches %lld); multi-kernel ops admitted: %s; the other %d of %d ops on one "
             "kernel at every height; the plugin's own height switches: the router split by columns (%d "
             "ops)%s, the indexer in %d extra runs; libr4d keeps the sparse attention, the indexer and the norms "
             "on their decode-band arithmetic; the CPU path takes every routed unit of passes of up to %lld "
             "tokens in jobs of up to %d rows over %d pool slots, a chunk's misses streamed to the pool as "
             "they land, at most %d units read at once and %d admitted ones queued for the RAM cache%s",
             (long long)lcp_, kLcpEnv, (long long)lcp_, lcp_moe_, lcp_twin_, (long long)lcp_reach_,
             lcp_pairs_.empty() ? "none" : lcp_pairs_.c_str(), lcp_ops_ - lcp_moe_ - lcp_pair_ops_, lcp_ops_,
             lcp_split_, lcp_split_ ? " from 256 rows" : "", lcp_runs_, (long long)ep_rows_,
             HostExpertPool::max_rows(), HostExpertPool::slots(), hx_lcp_planes_, hx_lcp_insertq_,
             std::getenv("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE")
                 ? "; the route trace records passes over 16 tokens as uncaptured" : "");
    /* Not a matter of bits (gemv_avx2_few and gemv_avx2_many give a row the same ones), so a warning:
     * a chunk puts tens of rows on a hot expert, which 4-row jobs read the weights for again and again. */
    if (HostExpertPool::max_rows() <= HostExpertPool::kMaxRowsDefault)
        RAD_WARN("large-chunk prefill: pool jobs of at most %d rows (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS "
                 "unset or <= %d): a chunk's experts then run through gemv_avx2_few, about twice the CPU time a "
                 "row of the many-row kernel takes in jobs of 32 or 64 (the same bits); set it to 32 or 64",
                 HostExpertPool::max_rows(), HostExpertPool::kMaxRowsDefault);
}

int FileStager::hx_init_impl(const Program& prog) {
    /* A re-run of init drops the aux worker's binding: the engine binds after init (aux_bind). */
    if (aux_) aux_unbind();
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
            if (op.op == "swiglu_clamped") {
                limit = op.base.get_f("limit", &v) && (float)v == pool->limit();
                o.sw = h;   /* the aux expert worker relaunches it (aux_resolve) */
            }
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
        /* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT (stager.h, hx_ra_rest_): 0, 1 or 2. */
        hx_ra_rest_ = 0;
        if (const char* g = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT"); g && *g) {
            if (std::strcmp(g, "0") && std::strcmp(g, "1") && std::strcmp(g, "2")) {
                RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=%s: 0 (urgent pieces), 1 (the rest in the "
                        "layer's own read) or 2 (the same, with the plane mirror's share)", g);
                return RAD_E_INVAL;
            }
            hx_ra_rest_ = std::atoi(g);
            if (hx_ra_rest_ && !hx_ra_finish_) {
                if (plan_->rank == 0)
                    RAD_WARN("read-ahead: RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=%d does nothing with "
                             "..._PREFETCH_FINISH=0 (a partly read unit is read afresh)", hx_ra_rest_);
                hx_ra_rest_ = 0;
            }
        }
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
                         r.readers(), live, !hx_ra_finish_ ? "read afresh"
                         : hx_ra_rest_ ? "finished by its layer's own read (below)" : "finished by the readers");
        }
    }
    if (!hx_ra_) hx_ra_rest_ = 0;   /* no read-ahead in pieces: nothing to finish */
    /* ..._SIDE_STORE: before the insert threads can see a refusal (the warm-up's below). */
    RAD_TRY(hx_side_init());
    if (hx_miss_rows_ > 0 && hx_bg_.empty()) {
        hx_bg_stop_ = false;
        int n = 4;
        if (const char* t = std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_INSERT_THREADS"); t && *t)
            n = std::clamp(std::atoi(t), 1, 16);
        hx_ins_init();
        for (int i = 0; i < n; ++i) hx_bg_.emplace_back(&FileStager::hx_bg_loop, this);
        if (plan_->rank == 0)
            RAD_INFO("host experts: passes of at most %d token(s) compute missed experts on the CPU "
                     "from the container's planes; background threads relay them into the RAM cache",
                     hx_miss_rows_);
        /* A second copy of the planes on another drive, read beside the container (plane_mirror.h). */
        RAD_TRY(mover_->mirror_init());
        /* ..._PREFETCH_URGENT on every rank, once its decode mirror is known: what a partly read
         * unit's rest is read from. */
        if (hx_ra_ && hx_ra_rest_) {
            const int64_t mb = mover_->decode_mirror_bytes();
            RAD_INFO("read-ahead: rank %d reads the rest of a partly read unit its layer needs in the "
                     "layer's own call on the kept readers (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=%d), %s",
                     plan_->rank, hx_ra_rest_,
                     hx_ra_rest_ == 2 && mb > 0
                         ? fmt("with the plane mirror's decode share of it (%s of a whole unit)",
                               humanb(mb).c_str()).c_str()
                         : hx_ra_rest_ == 2 ? "from the container alone: this rank has no decode mirror"
                                            : "from the container alone");
        }
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
         * hx_submit_misses_now (hx_unread_misses); with ..._PREFETCH_URGENT a stopped unit's rest
         * is. */
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
    hx_miss_admit_.clear();
    hx_ra_handed_ = 0;
    hx_ra_fin_b_ = 0;
    hx_side_handed_ = 0;
    hx_slot_.assign(want_.size(), -1);
    /* The aux expert worker (stager_aux.cpp): whether this layer's selected resident units go to its lane.
     * Without a binding nothing below changes. */
    aux_take_ = false;
    aux_n_ = 0;
    aux_list_.clear();
    aux_job_ = kAuxNone;
    if (aux_) {
        /* The online set's count (work/aux-dyn/SPEC.md B.2): the rows each selected candidate unit meets in a pass
         * the worker could take, whatever the lane or the gate. */
        if (aux_online_ && aux_pass_ && layer_rows(li) <= ds41aux::kMaxRows && layers_[(size_t)li].layer < n_model_layers_ &&
            (size_t)li < aux_rows_.size() && aux_rows_[(size_t)li].ok)
            for (int32_t u : want_)
                if (u >= 0 && (size_t)u < aux_cand_.size() && aux_cand_[(size_t)u]) {
                    const int32_t e = plan_->units[(size_t)u].expert;
                    const int32_t c = e >= 0 && e < counts_cap_ ? counts_host_[e] : 1;
                    aux_hits_[(size_t)u] += (uint32_t)std::max<int32_t>(c, 1);
                }
        bool res = false;
        for (int32_t u : want_) res = res || aux_view_->is_resident(u);
        if (res) {   /* a decline counts only for a layer with a selected resident unit */
            if (!aux_pass_ || layers_[(size_t)li].layer >= n_model_layers_) ++aux_stats_.declined_draft;
            else if (layer_rows(li) > ds41aux::kMaxRows) ++aux_stats_.declined_tall;
            else if (aux_guard_bad_ || (size_t)li >= aux_rows_.size() || !aux_rows_[(size_t)li].ok)
                ++aux_stats_.declined_rows;
            else if (aux_gate_ && !aux_lane_enter()) ++aux_stats_.declined_released;   /* card sharing: released */
            else
                switch (aux_->state()) {
                    case aux::Lane::kReady: aux_take_ = true; break;
                    case aux::Lane::kBackoff: aux_->declined(); ++aux_stats_.declined_busy; break;
                    case aux::Lane::kBusy: ++aux_stats_.declined_busy; break;
                    default: ++aux_stats_.declined_dead; break;
                }
        }
        if (!aux_take_) aux_lane_leave();
    }
    /* Large-chunk prefill: whether the cache admits what this unit's rows select (lcp_admit_row_). */
    const bool lcp_chunk = lcp_ && pass_tok_ > kEpRouteMaxRows;
    /* Read-ahead in pieces: this layer's units (a one-token pass, or a multi-row decode pass with
     * ..._PREFETCH_MULTIROW). */
    const bool ra = hx_ra_ && hx_ra_pass_;
    if (ra) {
        hx_ra_begin(li);
        hx_ra_chose_ns_ = rt_now_ns();
    }
    const double cb = hx_pool_->cpu_bytes_per_ns(), lb = hx_pool_->link_bytes_per_ns();
    const bool misses = hx_miss_rows_ > 0 && hx_miss_need(li) <= hx_miss_rows_;
    int64_t link = 0;
    for (size_t i = 0; i < want_.size(); ++i) {
        const int32_t u = want_[i];
        /* The aux worker's resident unit: in serve mode its lease goes to aux_units_ (the base's admit and
         * touch); in verify mode the CPU computes it as always and the worker computes it too. */
        const bool aux_res = aux_take_ && aux_view_->is_resident(u);
        const bool to_aux = aux_res && !aux_verify_;
        if (aux_res && aux_verify_) aux_list_.push_back(u);
        Mover::HostUnit* hu = to_aux ? (aux_n_ < kHxUnits ? &aux_units_[aux_n_] : nullptr)
                                     : (hx_n_ < kHxUnits ? &hx_units_[hx_n_] : nullptr);
        ++hx_stats_.probed;
        const bool admit = hx_admits(li, plan_->units[(size_t)u].expert);
        if (hu && mover_->borrow_host_unit(u, hu, admit)) {
            if (lcp_chunk) ++(admit ? hx_stats_.lcp_hit_admit : hx_stats_.lcp_hit_ro);
            bool gu = false, dn = false;
            for (int k = 0; k < hu->n; ++k) {
                gu = gu || hu->bytes[k] == HostExpertPool::kGuBytes;
                dn = dn || hu->bytes[k] == HostExpertPool::kDnBytes;
            }
            if (hu->n == 2 && gu && dn) {
                if (to_aux) {   /* slot code -3: no pool job, no reservation; aliased as a CPU unit */
                    hx_slot_[i] = -3;
                    ++aux_n_;
                    aux_list_.push_back(u);
                } else {
                    hx_slot_[i] = hx_n_++;
                }
                if (ra)
                    for (RaEntry& r : hx_ra_here_)
                        if (r.u && r.unit == u) hx_ra_drop(r, true);
                continue;
            }
            mover_->release_host_unit(hu);
        }
        if (to_aux) ++aux_stats_.uncached;   /* the CPU's miss path reads it, as in the base */
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
            hx_miss_admit_.push_back(admit ? 1 : 0);
            if (hx_ra_) hx_miss_ra_.push_back(std::move(urgent));
            hx_miss_.push_back(ahead ? ahead : hx_planes_get(u));
            const int rows = e >= 0 && e < counts_cap_ ? std::max<int32_t>(counts_host_[e], 1) : 1;
            hx_pool_->reserve(HostExpertPool::cost_rows(rows));
            hx_reserved_ += HostExpertPool::cost_rows(rows);
            continue;
        }
        link += plan_->units[(size_t)u].bytes;
    }
    if (ra && !tbo_keeps_ra()) {   /* the layer's units it did not select */
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
    /* The aux worker took nothing (every resident unit uncached): no job. */
    if (aux_take_ && aux_list_.empty()) aux_take_ = false;
    if (!aux_take_) aux_lane_leave();
    if (!n_cpu && hx_miss_.empty() && !aux_n_) {
        hx_n_ = 0;
        aux_take_ = false;
        aux_lane_leave();
        aux_list_.clear();
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

void FileStager::hx_cpu_units() {
    hx_keep_.clear();
    for (int k = 0; k < hx_n_; ++k)
        if (hx_units_[k].n) hx_keep_.push_back(hx_units_[k].unit);
    for (int k = 0; k < aux_n_; ++k)   /* the aux worker's units: aliased as the CPU's */
        if (aux_units_[k].n) hx_keep_.push_back(aux_units_[k].unit);
    for (const auto& p : hx_miss_) hx_keep_.push_back(p->unit);
}

/* The pass's token rows, sorted order and expert offsets to the host, queued behind the counts
 * row so the counts' wait covers them: the router and its scatter wrote them, and the router read
 * the rows. Their addresses are the gate_up operands hx_capture saw on an earlier pass, which
 * hx_capture checks again on this one. */
int FileStager::hx_fetch(int32_t li, RadStream s) {
    const HxOps& o = hx_ops_[(size_t)li];
    /* The layer's rows: the pass's, or under the per-row cut its suffix from layer_r0 (o.x is row 0's
     * address, hx_capture). */
    const int64_t rows = layer_rows(li);
    const char* const x = (const char*)o.x + layer_r0(li) * o.x_ld * 2;
    if (rows < 1 || pass_tok_ > ep_rows_ || ep_rows_ > hx_alloc_rows_) return RAD_E_STATE;
    if (o.x_ld == kHxEmbd) {
        RAD_TRY(rad_memcpy_async(hx_x_, x, rows * kHxEmbd * 2, s));
    } else {
        for (int64_t r = 0; r < rows; ++r)
            RAD_TRY(rad_memcpy_async(hx_x_ + r * kHxEmbd, x + r * o.x_ld * 2, kHxEmbd * 2, s));
    }
    RAD_TRY(rad_memcpy_async(hx_idx_, o.sorted, rows * o.top_k * 4, s));
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
    /* A split pass's slice B starts past row 0, which the list's copies do not: separate copies. */
    if (tbo_ && layer_r0(li) != 0) {
        ++hx_stats_.fl_plain;
        return 0;
    }
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
    /* ..._HOST_EXPERTS_PREWAKE: the router's event, the pool's pre-wake, then the counts (all in the
     * counts wait, as the router wait it is). */
    std::chrono::steady_clock::time_point t_woke{};
    if (hx_wake_) RAD_TRY(hx_wake_route(li, &t_woke));
    RAD_TRY(rad_event_sync(counts_ev_));
    const auto t_counts = std::chrono::steady_clock::now();
    if (t_woke.time_since_epoch().count()) hx_stats_.wk_rt_gap_ns += ns(t_woke, t_counts);
    const uint64_t sync_ns = ns(t_sync, t_counts);
    ep_stats_.sync_ns += sync_ns;
    if (pass_tok_ == 1) { ++ep_stats_.d_reads; ep_stats_.d_sync_ns += sync_ns; }
    if (prof_on_) prof_counts(li, sync_ns, t_counts);
    ++hx_stats_.early;
    hx_stats_.early_wait_ns += sync_ns;
    RAD_TRY(mover_->retire_file_windows());
    const auto t_retired = std::chrono::steady_clock::now();
    /* The predictor's input: a one-token pass's (either mode), or every row of a multi-row decode
     * pass that reads ahead in pieces (..._PREFETCH_MULTIROW). ..._PREFETCH_POST_LATE: after
     * hx_submit below instead, for either. */
    const bool post_rows = pass_tok_ > 1 && hx_ra_ && hx_ra_pass_ && hx_pred_ && hx_pred_->pieces();
    if (hx_pred_ && (pass_tok_ == 1 || post_rows) && plan_->rank == 0 && !hx_post_late_ &&
        moe_runs(L.layer + (hx_pred_->pieces() ? 2 : 1)))
        hx_pred_->post(trace_pass_, L.layer, hx_x_, (int)pass_tok_);
    int64_t total = 0;
    if (!ep_route_counts_valid(hx_early_v_, counts_host_, counts_cap_, &total)) {
        ++ep_stats_.corrupt;
        return RAD_E_STATE;
    }
    if (p1_heat_pass_) p1_heat_count(li);   /* ..._P1_LEND_HEAT: the layer's rows by unit */
    want_.clear();
    skip_.clear();
    for (int32_t u : L.units) {
        if (mover_->ring_resident(u)) {
            if (bank_track_) {
                bool selected=false;
                if (!ep_route_unit_selected(counts_host_,plan_->units[(size_t)u].expert,&selected)) return RAD_E_STATE;
                bank_selected_ += selected;
            }
            continue;   /* a lent-ring resident: the card's, as a VRAM one */
        }
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
    if (hx_post_late_ && (pass_tok_ == 1 || post_rows) && plan_->rank == 0 && moe_runs(L.layer + 2)) {
        hx_pred_->post(trace_pass_, L.layer, hx_x_, (int)pass_tok_);
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
 * stays the card's. The rows go in HostExpertPool::jobs_for(n) jobs as even as the count allows
 * (job_rows: 40 rows at 32 a job are 20 and 20), so no job of a hot expert is a short tail that
 * reads the weights again for a few rows. Returns the rows, or an error when they disagree with the
 * expert's count. */
int FileStager::hx_jobs_for(int e, const HostExpertPool::Job& proto) {
    return hx_jobs_into(e, proto, &hx_jobs_, &hx_rows_);
}

int FileStager::hx_jobs_into(int e, const HostExpertPool::Job& proto, std::vector<HostExpertPool::Job>* jobs,
                             std::vector<int32_t>* rows) {
    const HxOps& o = hx_ops_[(size_t)hx_li_];
    const int64_t AR = layer_rows(hx_li_), T = AR * o.top_k;
    const int32_t* const sorted = hx_idx_;
    const int32_t* const off = hx_idx_ + hx_sorted_max();
    const int ne = (int)o.n_expert;
    if (e < 0 || e >= ne) return RAD_E_STATE;
    int lo = e == 0 ? 0 : off[e], hi = e == ne - 1 ? (int)T : off[e + 1];
    lo = std::clamp(lo, 0, (int)T);
    hi = std::clamp(hi, lo, (int)T);
    auto live = [&](int i) { return sorted[i] >= 0 && sorted[i] / o.top_k < AR; };
    int32_t n = 0;
    for (int i = lo; i < hi; ++i) n += live(i);
    HostExpertPool::Job j = proto;
    int in_job = 0, k = 0, want = HostExpertPool::job_rows(n, 0);
    auto flush = [&] {
        if (!in_job) return;
        j.more = in_job - 1;
        j.cost = HostExpertPool::job_cost(in_job);
        if (in_job >= HostExpertPool::many_from()) {
            ++hx_stats_.many_jobs;
            hx_stats_.many_rows += (uint64_t)in_job;
        }
        jobs->push_back(j);
        j = proto;
        in_job = 0;
        want = HostExpertPool::job_rows(n, ++k);
    };
    for (int i = lo; i < hi; ++i) {
        if (!live(i)) continue;
        const uint16_t* x = hx_x_ + (int64_t)(sorted[i] / o.top_k) * kHxEmbd;
        uint16_t* y = hx_out_ + (int64_t)i * kHxEmbd;
        if (!in_job) { j.x = x; j.y = y; }
        else { j.xs[in_job - 1] = x; j.ys[in_job - 1] = y; }
        if (++in_job == want) flush();
        if (rows) rows->push_back(i);
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
    /* Longest first (an LPT order): the pool takes the oldest job first, so an expert with hundreds of
     * a chunk's rows starts before the ones with a few, and the layer's last job to finish is short.
     * The same jobs and rows whatever the order. */
    hx_order_.clear();
    for (int k = 0; k < hx_n_; ++k) hx_order_.push_back(k);
    auto rows_of = [&](int k) {
        const int32_t e = plan_->units[(size_t)hx_units_[k].unit].expert;
        return e >= 0 && e < counts_cap_ ? counts_host_[e] : 0;
    };
    if (pass_tok_ > 1)
        std::stable_sort(hx_order_.begin(), hx_order_.end(), [&](int a, int b) { return rows_of(a) > rows_of(b); });
    for (int k : hx_order_) {
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
    /* The aux worker's slots, compacted beside the CPU's jobs and before the pool's submit: its count
     * check may still refuse the layer here, with no job running yet (hx_drain then waits for none). In
     * serve mode their sorted rows join hx_rows_ (`appended`), so the sort and the copy take them. */
    int appended = 0;
    if (aux_take_) RAD_TRY(aux_prepare(&appended));
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
    /* The aux worker's job, behind the CPU's. It never fails here: a step that fails marks the job failed,
     * and hx_finish has the CPU compute its rows. */
    if (aux_take_) aux_submit();
    /* hx_finish copies the rows in runs, from the lowest: the order above is the pool's only. */
    std::sort(hx_rows_.begin(), hx_rows_.end());
    hx_stage_ = 2;
    hx_confirmed_ = false;
    ++hx_stats_.layers;
    hx_stats_.rows += hx_rows_.size() - (size_t)appended;
    if (prof_on_) prof_[prof_bucket()].cpu_rows += hx_rows_.size() - (size_t)appended;
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

/* The misses hx_submit_misses_now reads (stager.h): neither read ahead whole nor finished by the
 * piece readers; with ..._PREFETCH_URGENT also the rest of the stopped units. */
bool FileStager::hx_unread_misses(std::vector<Mover::HostPlanes*>* out, std::vector<Mover::PlanesIn>* in,
                                  bool* alone) const {
    int fin = 0, other = 0;   /* partly read units finished for this layer; misses read whole */
    for (size_t i = 0; i < hx_miss_.size(); ++i) {
        const ReadAheadUnit* ra = i < hx_miss_ra_.size() ? hx_miss_ra_[i].get() : nullptr;
        if (ra) {
            ++fin;
            if (!hx_ra_rest_) continue;   /* the piece readers read the rest */
            /* Stopped (hx_ra_take): the pieces started are the readers', the rest is this call's;
             * every piece started leaves only the wait for them. */
            Mover::PlanesIn f;
            for (int w = 0; w < 2; ++w) f.bytes[w] = ra->front(w);
            if (f.bytes[0] >= ra->extent(0).len && f.bytes[1] >= ra->extent(1).len) continue;
            out->push_back(hx_miss_[i].get());
            in->push_back(f);
        } else if (i >= hx_miss_ready_.size() || !hx_miss_ready_[i]) {
            ++other;
            out->push_back(hx_miss_[i].get());
            in->push_back(Mover::PlanesIn{});
        }
    }
    if (alone) *alone = fin == 1 && other == 0;
    return fin > 0;
}

/* ---- large-chunk prefill's misses ------------------------------------------------------------------
 *
 * A chunk of C tokens misses tens of units a layer and rank. Read in one call and computed after the
 * last read landed, a layer's misses would serialise ~100 ms of pool work behind its reads. Here each
 * unit's jobs are built before the reads (its planes laid out where the read puts them), and the
 * reader that lands a unit's last extent hands its jobs to the pool (Mover::read_unit_planes'
 * `landed`; HostExpertPool::submit is thread-safe), so the pool computes a unit while the next ones
 * are read. Units go to the readers longest first, as their jobs go to the pool. At most
 * ..._LCP_PLANES units (48, ~0.9 GB) are read at once: more go in waves, each wave's jobs waited for
 * before its planes go back to the free list for the next wave. The same planes and the same jobs as
 * the call-then-submit path, so the same bytes. */
namespace {
struct LandedCtx {
    HostExpertPool* pool = nullptr;
    HostExpertPool::Ticket* ticket = nullptr;
    const HostExpertPool::Job* jobs = nullptr;
    const std::pair<size_t, int>* ranges = nullptr;   /* by unit of the call: first job, jobs */
    std::atomic<uint8_t>* done = nullptr;
    std::atomic<int> rc{ 0 };
};
void hx_landed(void* ctx, int i) {
    LandedCtx& c = *(LandedCtx*)ctx;
    const std::pair<size_t, int>& r = c.ranges[i];
    c.done[i].store(1, std::memory_order_relaxed);
    if (r.second <= 0) return;
    const int st = c.pool->submit(c.jobs + r.first, r.second, c.ticket);
    if (st < 0) c.rc.store(st, std::memory_order_relaxed);
}
}  /* namespace */

int FileStager::hx_stream_misses(int rows) {
    const auto t0 = std::chrono::steady_clock::now();
    /* Longest first. */
    std::vector<size_t> order(hx_miss_.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    auto rows_of = [&](size_t i) {
        const int32_t e = plan_->units[(size_t)hx_miss_[i]->unit].expert;
        return e >= 0 && e < counts_cap_ ? counts_host_[e] : 0;
    };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rows_of(a) > rows_of(b); });
    const size_t cap = (size_t)std::max(1, hx_lcp_planes_);
    /* The reservation hx_choose made for the misses becomes the jobs' own costs, wave by wave. */
    int64_t reserved = hx_reserved_;
    hx_reserved_ = 0;
    ++hx_stats_.lcp_layers;
    for (size_t w0 = 0; w0 < order.size(); w0 += cap) {
        const size_t w1 = std::min(order.size(), w0 + cap);
        const bool last = w1 == order.size();
        ++hx_stats_.lcp_waves;
        if (prof_on_) ++prof_[prof_bucket()].waves;
        std::vector<Mover::HostPlanes*> ptrs;
        std::vector<std::pair<size_t, int>> ranges;
        std::vector<int64_t> costs;
        int64_t cost = 0;
        for (size_t q = w0; q < w1; ++q) {
            std::shared_ptr<Mover::HostPlanes>& sp = hx_miss_[order[q]];
            /* Past the first wave, a unit without buffers takes planes an earlier wave gave back. */
            if (w0 > 0 && !sp->buf[0] && !sp->buf[1]) {
                std::shared_ptr<Mover::HostPlanes> b = hx_planes_backed(sp->unit);
                if (b) sp = std::move(b);
            }
            Mover::HostPlanes* p = sp.get();
            PlaneExtent ext[2];
            const int g = p && mover_->lay_out_unit_planes(p, ext) && p->n == 2 &&
                          p->rows[0] == 2 * HostExpertPool::kFf ? 0 : 1, d = 1 - g;
            if (!p || p->n != 2 || p->rows[g] != 2 * HostExpertPool::kFf || p->cols[g] != kHxEmbd ||
                p->rows[d] != kHxEmbd || p->cols[d] != HostExpertPool::kFf) {
                hx_pool_->unreserve(reserved);
                RAD_ERR("host experts: layer %d's missed unit %d cannot be laid out for the CPU",
                        layers_[(size_t)hx_li_].layer, p ? p->unit : -1);
                return RAD_E_STATE;
            }
            HostExpertPool::Job proto;
            proto.gu = p->codes[g];
            proto.gu_scales = p->scales[g];
            proto.dn = p->codes[d];
            proto.dn_scales = p->scales[d];
            proto.urgent = true;
            const size_t first = hx_jobs_.size();
            const int n = hx_jobs_for(plan_->units[(size_t)p->unit].expert, proto);
            if (n < 0) {
                hx_pool_->unreserve(reserved);
                return n;
            }
            ++hx_stats_.miss_units;
            if (prof_on_) ++prof_[prof_bucket()].miss_units;
            hx_stats_.miss_rows += (uint64_t)n;
            int64_t c = 0;
            for (size_t k = first; k < hx_jobs_.size(); ++k) c += hx_jobs_[k].cost;
            ranges.emplace_back(first, (int)(hx_jobs_.size() - first));
            costs.push_back(c);
            cost += c;
            ptrs.push_back(p);
        }
        /* The wave's jobs take over their part of the reservation; the last wave the rest. */
        if (last) {
            if (reserved != cost) hx_pool_->unreserve(reserved - cost);
            reserved = 0;
        } else {
            if (cost > reserved) hx_pool_->reserve(cost - reserved);
            reserved = std::max<int64_t>(reserved - cost, 0);
        }
        HostExpertPool::Ticket wave;
        HostExpertPool::Ticket* const t = last ? &hx_ticket_ : &wave;
        std::unique_ptr<std::atomic<uint8_t>[]> done(new std::atomic<uint8_t>[ptrs.size()]);
        for (size_t i = 0; i < ptrs.size(); ++i) done[i].store(0, std::memory_order_relaxed);
        LandedCtx ctx;
        ctx.pool = hx_pool_;
        ctx.ticket = t;
        ctx.jobs = hx_jobs_.data();   /* no job is added while the readers run */
        ctx.ranges = ranges.data();
        ctx.done = done.get();
        Mover::PlaneReadTiming tm;
        PieceReader& demand = PieceReader::process();
        demand.demand_enter();
        const bool ok = mover_->read_unit_planes(ptrs.data(), (int)ptrs.size(), rows, &tm, nullptr, false,
                                                 nullptr, hx_landed, &ctx);
        demand.demand_leave();
        int rc = ctx.rc.load();
        /* Units the readers did not hand over (a leg read again after them): theirs now. */
        for (size_t i = 0; i < ptrs.size() && ok && rc >= 0; ++i) {
            if (done[i].load(std::memory_order_relaxed)) {
                ++hx_stats_.lcp_landed;
                continue;
            }
            ++hx_stats_.lcp_late;
            if (ranges[i].second > 0) rc = hx_pool_->submit(hx_jobs_.data() + ranges[i].first, ranges[i].second, t);
            if (rc >= 0) done[i].store(1, std::memory_order_relaxed);
        }
        if (!ok || rc < 0) {
            /* The jobs never handed over give their reservation back, and the ones that were finish
             * before their planes may go (the last wave's are hx_ticket_'s: hx_drain waits). */
            int64_t left = reserved;
            for (size_t i = 0; i < ptrs.size(); ++i)
                if (!done[i].load(std::memory_order_relaxed)) left += costs[i];
            if (left) hx_pool_->unreserve(left);
            if (!last) hx_pool_->wait(&wave);
            RAD_ERR("host experts: reading layer %d's missed units for the CPU failed%s",
                    layers_[(size_t)hx_li_].layer, rc < 0 ? " (the pool refused a job)" : "");
            return rc < 0 ? rc : RAD_E_IO;
        }
        if (last) break;
        /* Before the next wave reads into planes again: this wave's jobs done, its planes to the
         * insert threads and, once inserted, to the free list (hx_planes_backed). */
        const auto w = std::chrono::steady_clock::now();
        hx_pool_->wait(&wave);
        hx_stats_.lcp_wave_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - w).count();
        hx_queue_inserts(order.data() + w0, w1 - w0, rows);
        for (size_t q = w0; q < w1; ++q) hx_miss_[order[q]].reset();   /* not admitted: to the free list now */
    }
    {
        const uint64_t rd = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t0).count();
        hx_stats_.miss_read_ns += rd;
        if (prof_on_) prof_[prof_bucket()].miss_read_ns += rd;
    }
    std::sort(hx_rows_.begin(), hx_rows_.end());
    /* The last wave's (the earlier ones went above). */
    const size_t from = order.empty() ? 0 : (order.size() - 1) / cap * cap;
    hx_queue_inserts(order.data() + from, order.size() - from, rows);
    return RAD_OK;
}

/* Reads the misses' planes (rows: the pass's tokens) and submits their jobs to the pool. */
int FileStager::hx_submit_misses_now(int rows) {
    /* A large-chunk prefill's chunk: streamed (above). Never a posted read or one with units read
     * ahead (both are one-token passes'). */
    if (lcp_ && rows > (int)kEpRouteMaxRows && !hx_miss_posted_ &&
        std::none_of(hx_miss_ready_.begin(), hx_miss_ready_.end(), [](uint8_t r) { return r != 0; }) &&
        std::none_of(hx_miss_ra_.begin(), hx_miss_ra_.end(), [](const auto& r) { return r != nullptr; }))
        return hx_stream_misses(rows);
    std::vector<Mover::HostPlanes*> ptrs;
    std::vector<Mover::PlanesIn> in;
    bool alone = false;
    const bool urgent = hx_unread_misses(&ptrs, &in, &alone);
    const auto t0 = std::chrono::steady_clock::now();
    Mover::PlaneReadTiming tm;
    bool read = false;
    /* ..._HOST_EXPERTS_PREWAKE (stager.h): a one-token pass's demand read pre-wakes the pool at
     * ..._PREWAKE_LEAD_US before its predicted landing, if the reads are still out by then. Not
     * with units being finished by the piece readers: their wait comes after the read. */
    Mover::ReadWaitHook wake_hook;
    if (hx_wake_ && rows == 1 && !urgent) {
        const int64_t at = hx_wake_at(hx_miss_posted_ ? hx_miss_post_ptrs_.size() : ptrs.size());
        if (at >= 0) {
            wake_hook.fn = &FileStager::hx_wake_miss;
            wake_hook.ctx = this;
            wake_hook.at_ns = at;
            hx_wake_fired_ns_ = 0;
            ++hx_stats_.wk_rd_armed;
            /* The wait's timer fires at its time, not up to the default 50 us slack later: 1 us for
             * that wait alone (the thread's other timed waits keep theirs). */
            wake_hook.slack_ns = 1000;
        }
    }
    const Mover::ReadWaitHook* wake = wake_hook.fn ? &wake_hook : nullptr;
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
            ok = mover_->finish_unit_planes(&tm, wake);
            read = true;
            bool same = ptrs == hx_miss_post_ptrs_ && in.size() == hx_miss_post_in_.size();
            for (size_t i = 0; same && i < in.size(); ++i)
                same = in[i].bytes[0] == hx_miss_post_in_[i].bytes[0] &&
                       in[i].bytes[1] == hx_miss_post_in_[i].bytes[1];
            if (!same) {
                demand.demand_leave();
                RAD_ERR("host experts: layer %d's misses changed after their reads were posted",
                        layers_[(size_t)hx_li_].layer);
                return RAD_E_STATE;
            }
        } else {
            demand.demand_enter();
            if (!ptrs.empty()) {
                /* With ..._PREFETCH_URGENT the stopped units' rest in the same call, the readers
                 * taking it beside the new misses while their pieces in flight land. */
                ok = mover_->read_unit_planes(ptrs.data(), (int)ptrs.size(), rows, &tm,
                                              hx_ra_rest_ ? in.data() : nullptr, hx_ra_rest_ == 2, wake);
                read = true;
            }
        }
        if (read) {
            hx_stats_.ra_rest += (uint64_t)tm.rest;
            hx_stats_.ra_rest_bytes += (uint64_t)tm.rest_bytes;
            hx_stats_.ra_rest_mirror_bytes += (uint64_t)tm.rest_mirror_bytes;
            hx_stats_.ra_rest_mirror += (uint64_t)tm.rest_mirror;
        }
        if (ok && urgent) ok = hx_ra_settle(rows);
        demand.demand_leave();
        if (!ok) {
            RAD_ERR("host experts: reading layer %d's missed units for the CPU failed",
                    layers_[(size_t)hx_li_].layer);
            return RAD_E_IO;
        }
        /* The layer's wait for its partly read units, either way they were finished: from its
         * choice, where urgent pieces start (a rest's read starts here, in this call). */
        if (urgent) {
            const uint64_t ns = rt_now_ns() - hx_ra_chose_ns_;
            ++hx_stats_.ra_fin_layers;
            hx_stats_.ra_fin_ns += ns;
            if (alone) {
                ++hx_stats_.ra_alone;
                hx_stats_.ra_alone_ns += ns;
                if (hx_ra_fin_b_ > 0) {
                    ++hx_stats_.ra_alone_b;
                    hx_stats_.ra_alone_b_ns += ns;
                }
            }
        }
    }
    /* Units read ahead whole, or held by the side store: handed over only from the container the
     * program was loaded from. */
    if ((hx_ra_handed_ > 0 || hx_side_handed_ > 0) && !mover_->source_epochs_current()) {
        RAD_ERR("host experts: the container changed under layer %d's units read ahead or kept in the "
                "side store", layers_[(size_t)hx_li_].layer);
        return RAD_E_IO;
    }
    {
        const uint64_t rd = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t0).count();
        hx_stats_.miss_read_ns += rd;
        if (prof_on_) prof_[prof_bucket()].miss_read_ns += rd;
    }
    if (read && rows == 1) hx_note_reads(tm);
    /* The armed pre-wake: how long after it the last read landed, or how long before its time (the
     * reads' landing from their dispatch, which tm counts, against the hook's time from it). */
    if (wake && read) {
        const int64_t land = std::max(tm.container_ns, tm.mirror_ns);
        if (hx_wake_fired_ns_) {
            ++hx_stats_.wk_rd_fired;
            hx_stats_.wk_rd_lead_ns += (uint64_t)std::max<int64_t>(tm.t0_ns + land - hx_wake_fired_ns_, 0);
        } else {
            hx_stats_.wk_rd_early_ns += (uint64_t)std::max<int64_t>(wake->at_ns - land, 0);
        }
        hx_wake_fired_ns_ = 0;
    }
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
        if (prof_on_) ++prof_[prof_bucket()].miss_units;
        hx_stats_.miss_rows += (uint64_t)n;
    }
    const int n = (int)(hx_jobs_.size() - first);
    int64_t submitted = 0;
    for (size_t k = first; k < hx_jobs_.size(); ++k) submitted += hx_jobs_[k].cost;
    if (hx_reserved_ != submitted) hx_pool_->unreserve(hx_reserved_ - submitted);
    hx_reserved_ = 0;
    RAD_TRY(hx_pool_->submit(hx_jobs_.data() + first, n, &hx_ticket_));
    std::sort(hx_rows_.begin(), hx_rows_.end());
    hx_queue_inserts(rows);
    return RAD_OK;
}

/* The admission row: the units whose rows include one at or past it are admitted (stager.h). */
bool FileStager::hx_admits(int32_t li, int32_t e) const {
    if (lcp_admit_row_ <= 0 || !lcp_ || li < 0 || (size_t)li >= hx_ops_.size()) return true;
    const HxOps& o = hx_ops_[(size_t)li];
    if (e < 0 || e >= (int32_t)o.n_expert || !hx_idx_ || o.top_k <= 0) return true;
    /* The layer's rows are the pass's from layer_r0 (the per-row cut); the admission row is the pass's. */
    const int64_t rows = layer_rows(li), T = rows * o.top_k, from = lcp_admit_row_ - layer_r0(li);
    const int32_t* const sorted = hx_idx_;
    const int32_t* const off = hx_idx_ + hx_sorted_max();
    int lo = e == 0 ? 0 : off[e], hi = e == (int32_t)o.n_expert - 1 ? (int)T : off[e + 1];
    lo = std::clamp(lo, 0, (int)T);
    hi = std::clamp(hi, lo, (int)T);
    for (int i = lo; i < hi; ++i)
        if (sorted[i] >= 0 && sorted[i] / o.top_k >= from && sorted[i] / o.top_k < rows) return true;
    return false;
}

/* The misses idx[0..n) to the insert threads (hx_bg_loop), with the pass's tokens: the admitted
 * ones only, and not the side store's units (hx_miss_ready_ 2: the packed cache refuses them, and
 * their planes stay held); in a large chunk at most hx_lcp_insertq_ queued or being inserted at
 * once (hx_bg_busy_ counts an insert for its whole run, slices and waits included). */
void FileStager::hx_queue_inserts(const size_t* idx, size_t n, int rows) {
    const bool chunk = lcp_ && pass_tok_ > kEpRouteMaxRows;
    bool any = false;
    {
        std::lock_guard<std::mutex> lk(hx_bg_mu_);
        for (size_t k = 0; k < n; ++k) {
            const size_t i = idx[k];
            if (i >= hx_miss_.size() || !hx_miss_[i]) continue;
            if (i < hx_miss_ready_.size() && hx_miss_ready_[i] == 2) continue;   /* the side store's */
            if (i < hx_miss_admit_.size() && !hx_miss_admit_[i]) {
                ++hx_stats_.lcp_miss_ro;   /* computed, and let go with the layer */
                continue;
            }
            if (chunk && (int64_t)hx_bg_q_.size() + hx_bg_busy_ >= hx_lcp_insertq_) {
                ++hx_stats_.lcp_miss_dropped;
                continue;
            }
            if (chunk) ++hx_stats_.lcp_miss_admit;
            hx_bg_q_.push_back(HxInsert{ hx_miss_[i], rows });
            any = true;
        }
    }
    if (any) hx_bg_cv_.notify_all();
}

/* RADIANCE_EXPERIMENTAL_DS41_MISS_POST: the layer's unread misses (hx_unread_misses: the set
 * hx_submit_misses_now reads) handed to the mover's kept readers right after hx_choose. A decode
 * miss layer used to start its reads only after hx_submit had woken the pool (28-32 us a layer) and
 * the copy path's staging and aliases had run; the readers' own wake-up now overlaps that work too.
 * Only where hx_submit_misses_now reads on this thread (one-token passes, not on hx_mw_). Never a
 * unit read ahead (..._PREFETCH): whole ones are not read again, and an urgent one is the piece
 * readers' until hx_ra_settle, which may give its slot fresh planes and the old ones back to the
 * free list. With ..._PREFETCH_URGENT a stopped unit's rest is posted too: no reader starts a piece
 * of it any more, its pieces in flight land before the rest's offsets, and hx_ra_settle runs only
 * after the post's finish. A demand read from the post on (PieceReader's count: no read-ahead piece
 * starts) until hx_submit_misses_now or hx_join_miss_post leaves it. Nothing posted (one in flight,
 * no kept readers, the readers busy with another thread's call): hx_submit_misses_now reads as
 * before. */
void FileStager::hx_post_misses() {
    if (hx_miss_posted_ || pass_tok_ != 1 || (hx_async_ && pass_tok_ >= hx_async_rows_)) return;
    hx_miss_post_ptrs_.clear();
    hx_miss_post_in_.clear();
    (void)hx_unread_misses(&hx_miss_post_ptrs_, &hx_miss_post_in_);
    if (hx_miss_post_ptrs_.empty()) return;
    PieceReader& demand = PieceReader::process();
    demand.demand_enter();
    hx_miss_posted_ = mover_->start_unit_planes(hx_miss_post_ptrs_.data(),
                                                (int)hx_miss_post_ptrs_.size(), (int)pass_tok_,
                                                hx_ra_rest_ ? hx_miss_post_in_.data() : nullptr,
                                                hx_ra_rest_ == 2);
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

/* One one-token miss read's timing into hx_stats_.mr (see HostStats::MissReads), and with
 * ..._HOST_EXPERTS_PREWAKE its landing into the ring the next reads' pre-wake time comes from. */
void FileStager::hx_note_reads(const Mover::PlaneReadTiming& t) {
    /* Not a call that carried a read-ahead unit's rest (..._PREFETCH_URGENT): the pre-wake is armed
     * only in layers with no unit being finished, so their landings predict fresh reads alone. */
    if (hx_wake_ && !t.rest) {
        const int k = t.units == 1 ? 0 : 1;
        const int64_t land = std::max(t.container_ns, t.mirror_ns);
        if (land > 0) hx_wake_ring_[k][hx_wake_seen_[k]++ % kWakeRing] = land;
    }
    HostStats::MissReads& s = hx_stats_.mr[t.rest ? 2 : t.units == 1 ? 0 : 1];
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

/* ..._HOST_EXPERTS_PREWAKE (stager.h): each routed layer's router GEMV -- the widest gemm reading
 * blk.L.glm_router.weight (the whole projection; a TP-split pass's part is a quarter of it, and an
 * on-die read is no gemm) -- and the event recorded before it in one-token passes. A layer whose
 * router is not found gets no router pre-wake (its misses still do). */
int FileStager::hx_wake_init(const Program& prog) {
    hx_wake_ = false;
    hx_wake_li_ = -1;
    hx_wake_op_.clear();
    if (!hx_pool_ || hx_pool_->prewake_n() <= 0) return RAD_OK;
    std::map<std::string, int32_t> want;   /* router weight name: routed layer */
    for (size_t li = 0; li < layers_.size(); ++li)
        if (li < hx_ops_.size() && hx_ops_[li].gu >= 0)
            want[fmt("blk.%d.glm_router.weight", layers_[li].layer)] = (int32_t)li;
    std::vector<int32_t> li_of_w(prog.weights.size(), -1);
    for (size_t w = 1; w < prog.weights.size(); ++w)
        if (const auto it = want.find(prog.weights[w].name); it != want.end()) li_of_w[w] = it->second;
    std::vector<int32_t> op(layers_.size(), -1);
    std::vector<long long> width(layers_.size(), 0);
    for (size_t h = 0; h < prog.ops.size(); ++h) {
        const OpInfo& o = prog.ops[h];
        long long N = 0;
        if (o.op.compare(0, 4, "gemm") != 0 || !o.base.get_i("N", &N)) continue;
        for (const rad_weight w : o.weights) {
            const int32_t li = w < li_of_w.size() ? li_of_w[w] : -1;
            if (li < 0 || N <= width[(size_t)li]) continue;
            op[(size_t)li] = (int32_t)h;
            width[(size_t)li] = N;
        }
    }
    hx_wake_op_.assign(prog.ops.size(), -1);
    size_t found = 0;
    for (size_t li = 0; li < op.size(); ++li)
        if (op[li] >= 0) {
            hx_wake_op_[(size_t)op[li]] = (int32_t)li;
            ++found;
        }
    if (!hx_wake_ev_ && rad_event_create_as(&hx_wake_ev_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT) < 0) {
        hx_wake_ev_ = nullptr;
        RAD_ERR("host experts: pre-wake: no event for the router (%s); no pre-wake", rad_dev_last_error());
        hx_wake_op_.clear();
        return RAD_OK;
    }
    hx_wake_ = true;
    if (plan_->rank == 0)
        RAD_INFO("host experts: pre-wake: an event before %zu of %zu routed layers' router GEMVs in "
                 "one-token passes%s", found, layers_.size(),
                 hx_pool_->prewake_lead_ns() > 0 ? "; miss reads pre-wake before their predicted landing"
                                                  : "; none in miss reads (..._PREWAKE_LEAD_US=0)");
    return RAD_OK;
}

/* Layer li's decision, before its counts sync: when this pass recorded li's router event and the
 * counts are not in yet, the event (the card has reached the router), then the pool's pre-wake --
 * unless the counts came meanwhile: the submit is then a few us away, and the pre-wake's wake loop
 * would only delay it. *woke: when the event was seen, for the router-to-counts counter;
 * untouched when there was no pre-wake. */
int FileStager::hx_wake_route(int32_t li, std::chrono::steady_clock::time_point* woke) {
    if (hx_wake_li_ != li) return RAD_OK;
    hx_wake_li_ = -1;
    if (rad_event_query(counts_ev_) == 1) {
        ++hx_stats_.wk_rt_ready;
        return RAD_OK;
    }
    RAD_TRY(rad_event_sync(hx_wake_ev_));
    if (rad_event_query(counts_ev_) == 1) {
        ++hx_stats_.wk_rt_ready;
        return RAD_OK;
    }
    const auto t0 = std::chrono::steady_clock::now();
    hx_pool_->prewake(HostExpertPool::kWakeRouter);
    ++hx_stats_.wk_rt;
    hx_stats_.wk_rt_call_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0).count();
    /* The counts landed while the call ran: it held the layer's decision back by up to its time. */
    if (rad_event_query(counts_ev_) == 1) ++hx_stats_.wk_rt_late;
    *woke = t0;
    return RAD_OK;
}

/* A one-token miss read of `units` units: its pre-wake's time from the dispatch, the median landing
 * of the last kWakeRing such reads less the lead (at least 0); -1 for none (fewer than 8 seen yet,
 * no units, or ..._PREWAKE_LEAD_US=0). */
int64_t FileStager::hx_wake_at(size_t units) const {
    if (!hx_wake_ || !units || !hx_pool_ || hx_pool_->prewake_lead_ns() <= 0) return -1;
    const int k = units == 1 ? 0 : 1;
    if (hx_wake_seen_[k] < 8) return -1;
    const size_t n = (size_t)std::min<uint64_t>(hx_wake_seen_[k], kWakeRing);
    int64_t v[kWakeRing];
    std::copy(hx_wake_ring_[k], hx_wake_ring_[k] + n, v);
    std::nth_element(v, v + n / 2, v + n);
    return std::max<int64_t>(v[n / 2] - hx_pool_->prewake_lead_ns(), 0);
}

/* A miss read's hook (Mover::ReadWaitHook), on the issuing thread: the reads are still out at
 * their predicted landing less the lead. */
void FileStager::hx_wake_miss(void* self) {
    FileStager* fs = static_cast<FileStager*>(self);
    fs->hx_wake_fired_ns_ = (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
    fs->hx_pool_->prewake(HostExpertPool::kWakeMiss);
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

/* hx_planes_get, but only a HostPlanes on the free list that still holds buffers (an earlier wave's,
 * given back once inserted), or null: a later wave of large-chunk misses reads into those rather
 * than growing new ones. */
std::shared_ptr<Mover::HostPlanes> FileStager::hx_planes_backed(int32_t unit) {
    std::unique_ptr<Mover::HostPlanes> p;
    {
        std::lock_guard<std::mutex> lk(hx_planes_mu_);
        for (size_t i = hx_planes_free_.size(); i-- > 0;)
            if (hx_planes_free_[i]->buf[0] || hx_planes_free_[i]->buf[1]) {
                p = std::move(hx_planes_free_[i]);
                hx_planes_free_.erase(hx_planes_free_.begin() + (ptrdiff_t)i);
                break;
            }
    }
    if (!p) return nullptr;
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
void FileStager::hx_ra_ready(void*, uint32_t pass, int layer, const int32_t* experts, int n,
                             int rows) {
    std::lock_guard<std::mutex> lk(g_ra_mu);
    for (FileStager* s : g_ra_stagers) s->hx_ra_plan(pass, layer, experts, n, rows);
}

/* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW (stager.h), once hx_init_impl has set up (or not)
 * the read-ahead in pieces: a value the configuration cannot honour is refused rather than
 * ignored, so an A/B arm never runs without what its name says. */
int FileStager::hx_ra_rows_init() {
    hx_ra_rows_ = 1;
    const char* v = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW");
    const int64_t limit = std::min<int64_t>(ep_rows_, RouterPredictor::kMaxRows);
    const int n = ra_multirow_rows(v, limit);
    if (n == 1) return RAD_OK;
    if (n < 2 || !hx_ra_ || !hx_pred_ || !hx_pred_->pieces() || n > hx_miss_rows_) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=%s: N (2-%lld) reads ahead in decode passes "
                "of 2..N rows too, which needs the read-ahead in pieces running "
                "(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces, host experts with ..._MISSES of at least N)%s",
                v, (long long)limit, hx_ra_ ? "" : "; it is not running on this rank");
        return RAD_E_INVAL;
    }
    hx_ra_rows_ = n;
    if (plan_ && plan_->rank == 0)
        RAD_INFO("read-ahead: decode passes of 2-%d rows read ahead too "
                 "(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=%d): rank 0 posts every row's router "
                 "input, each row predicts layer L+2's top %d, and each rank reads the union of the rows' "
                 "predictions under the same live cap; one-token passes as before", n, n, hx_pred_->k());
    return RAD_OK;
}

int64_t FileStager::hx_ra_mirror_bytes() const {
    if (!mover_) return 0;
    return pass_tok_ == 1 ? mover_->decode_mirror_bytes() : mover_->mirror_bytes((int)pass_tok_);
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
 * list are touched here, never what the issuing thread writes. A multi-row post's list is the
 * union of its rows' predictions (router_topk_union), taken the same way. */
void FileStager::hx_ra_plan(uint32_t pass, int layer, const int32_t* g, int n, int rows) {
    if (!hx_ra_ || layer < 0 || layer >= (int)hx_ra_li_.size()) return;
    const int32_t li = hx_ra_li_[(size_t)layer];
    if (li < 0 || (size_t)li >= hx_unit_of_.size()) return;
    const int base = 96 * plan_->rank;
    HostStats::RaRows& by = hx_stats_.ra_rows[HostStats::ra_slot(rows)];
    for (int pos = 0; pos < n; ++pos) {
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
                ++by.skipped;
                return;
            }
            if ((int)hx_ra_live_.size() >= hx_ra_cap_) {
                ++hx_stats_.ra_capped;
                ++by.skipped;
                return;
            }
            bool dup = false;
            for (const RaEntry& r : hx_ra_live_) dup = dup || (r.pass == pass && r.li == li && r.unit == u);
            if (dup) continue;
        }
        std::shared_ptr<Mover::HostPlanes> p = hx_planes_get(u);
        PlaneExtent ext[2];
        if (!mover_->lay_out_unit_planes(p.get(), ext) || !mover_->source_epochs_current()) continue;
        /* Readers take earlier passes, then earlier layers, then better predictions first (a union
         * is at most 16 rows of 16, so its position fits the byte). */
        auto ra = std::make_shared<ReadAheadUnit>(
            this, ((uint64_t)pass << 24) | ((uint64_t)layer << 8) | (uint64_t)std::min(pos, 255), ext, 2,
            hx_ra_->piece_bytes(), &FileStager::hx_ra_read, mover_, p);
        {
            std::lock_guard<std::mutex> lk(hx_ra_mu_);
            /* The rank may have reached the layer while the unit was laid out. */
            if (ra_too_late(hx_ra_at_.load(std::memory_order_acquire), pass, li)) {
                ++hx_stats_.ra_late;
                ++by.skipped;
                return;
            }
            hx_ra_live_.push_back(RaEntry{ ra, p, pass, li, u, rows });
            ++hx_stats_.ra_planned;
            ++by.planned;
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
        HostStats::RaRows& by = hx_stats_.ra_rows[HostStats::ra_slot(e.rows)];
        ++by.dropped;
        by.dropped_bytes += (uint64_t)e.u->bytes_in();
    }
}

void FileStager::hx_ra_begin(int32_t li) {
    hx_ra_at_.store(((uint64_t)trace_pass_ << 32) | (uint32_t)(li + 1), std::memory_order_release);
    /* A split pass's slice B keeps what slice A left of this layer (tbo_keeps_ra). */
    if (!(tbo_ && tbo_slice_ == 1)) hx_ra_here_.clear();
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
    HostStats::RaRows& by = hx_stats_.ra_rows[HostStats::ra_slot(pass_tok_)];
    ++hx_stats_.ra_misses;
    ++by.misses;
    RaEntry* e = nullptr;
    for (RaEntry& r : hx_ra_here_)
        if (r.u && r.unit == u) { e = &r; break; }
    if (!e) return nullptr;
    ++hx_stats_.ra_miss_planned;
    ++by.miss_planned;
    std::shared_ptr<ReadAheadUnit> ra = std::move(e->u);
    std::shared_ptr<Mover::HostPlanes> p = std::move(e->p);
    if (ra->whole()) {   /* every piece in: the epoch is checked before the jobs go out */
        ++hx_stats_.ra_whole;
        ++by.whole;
        hx_stats_.ra_used_bytes += (uint64_t)ra->bytes_in();
        by.used_bytes += (uint64_t)ra->bytes_in();
        ++hx_ra_handed_;
        *ready = true;
        return p;
    }
    /* No piece started: the layer reads it as any miss, into the buffers laid out for it. */
    if (hx_ra_->take_whole(*ra)) {
        ++hx_stats_.ra_fresh;
        ++by.fresh;
        return p;
    }
    /* Barely started: the demand read with the plane mirror's share beats the container alone for
     * the rest (or ..._PREFETCH_FINISH=0). A fresh buffer, since pieces in flight still land in
     * this one. With ..._PREFETCH_URGENT=2 the rest takes the mirror's share too: finished. The
     * mirror's share is the pass's (hx_ra_mirror_bytes: decode's for one token, its rows' else). */
    if (!hx_ra_finish_ || (hx_ra_rest_ != 2 && (int64_t)ra->started() * hx_ra_->piece_bytes() <
                                                   hx_ra_mirror_bytes())) {
        hx_ra_->cancel(*ra);
        ++hx_stats_.ra_fresh;
        ++by.fresh;
        hx_stats_.ra_dropped_bytes += (uint64_t)ra->bytes_in();
        by.dropped_bytes += (uint64_t)ra->bytes_in();
        return nullptr;
    }
    /* Finished: the readers read the rest as urgent pieces, or (..._PREFETCH_URGENT) they start no
     * further piece and the layer's own call reads the rest (hx_unread_misses). One that would be
     * finished without ..._PREFETCH_URGENT=2 too counts for the per-unit wait every arm compares
     * (HostStats::ra_alone_b; no mover in a test: no mirror). */
    if (hx_ra_rest_ != 2 || (int64_t)ra->started() * hx_ra_->piece_bytes() >=
                                (mover_ ? hx_ra_mirror_bytes() : 0))
        ++hx_ra_fin_b_;
    if (hx_ra_rest_) hx_ra_->stop(*ra);
    else hx_ra_->make_urgent(*ra);
    ++hx_stats_.ra_finished;
    ++by.finished;
    hx_stats_.ra_used_bytes += (uint64_t)ra->bytes_in();
    by.used_bytes += (uint64_t)ra->bytes_in();
    *urgent = std::move(ra);
    return p;
}

void FileStager::hx_ra_drop(RaEntry& e, bool hit) {
    if (!e.u) return;
    hx_ra_->cancel(*e.u);
    ++(hit ? hx_stats_.ra_cached : hx_stats_.ra_unselected);
    hx_stats_.ra_dropped_bytes += (uint64_t)e.u->bytes_in();
    HostStats::RaRows& by = hx_stats_.ra_rows[HostStats::ra_slot(e.rows)];
    ++by.dropped;
    by.dropped_bytes += (uint64_t)e.u->bytes_in();
    e.u.reset();
    e.p.reset();
}

/* Inside the demand read: the urgent units' last pieces (..._PREFETCH_URGENT: the stopped units'
 * pieces in flight, their rest read by the layer's call before this), then the units with a failed
 * piece read again whole into fresh buffers, and the container's epoch. `rows`: the pass's (on
 * hx_mw_ too, where pass_tok_ is the issuing thread's). */
bool FileStager::hx_ra_settle(int rows) {
    std::vector<Mover::HostPlanes*> again;
    HostStats::RaRows& by = hx_stats_.ra_rows[HostStats::ra_slot(rows)];
    for (size_t i = 0; i < hx_miss_ra_.size() && i < hx_miss_.size(); ++i) {
        std::shared_ptr<ReadAheadUnit>& ra = hx_miss_ra_[i];
        if (!ra) continue;
        const uint64_t waited = hx_ra_->wait(*ra);
        hx_stats_.ra_wait_ns += waited;
        by.wait_ns += waited;
        /* A stopped unit is never whole by its pieces: the rest was the call's. */
        if (hx_ra_rest_ ? ra->failed() : !ra->whole()) {
            ++hx_stats_.ra_refetched;
            ++by.refetched;
            hx_miss_[i] = hx_planes_get(hx_miss_[i]->unit);
            again.push_back(hx_miss_[i].get());
        }
        ra.reset();
    }
    if (!again.empty() && !mover_->read_unit_planes(again.data(), (int)again.size(), rows)) return false;
    return mover_->source_epochs_current();
}

namespace {

/* One background insert's ds41hx::PackCopy::before (..._INSERT_NT, ..._INSERT_SLICE_KIB): before
 * each piece of a weight's codes it sleeps while the pool has work queued or reserved, for what is
 * left of the insert's budget (0: never), and it tallies the pieces, their bytes, and the bytes
 * and time copied while the pool had work. */
struct InsertGate {
    using Clock = std::chrono::steady_clock;
    HostExpertPool* pool = nullptr;
    int64_t budget_ns = 0;
    uint64_t pieces = 0, bytes = 0, waited = 0, wait_ns = 0, busy_bytes = 0, busy_ns = 0;
    uint64_t reserved = 0; /* pieces begun with only reservations in the pool (a miss's read) */
    bool capped = false;   /* the budget ran out with the pool still busy */
    bool busy = false;     /* the piece being copied began while the pool had work, at t0 */
    Clock::time_point t0{};

    static void before(void* ctx, size_t n) { static_cast<InsertGate*>(ctx)->piece(n); }
    /* "Work": jobs submitted and not yet finished (HostExpertPool::work), not the reservations a
     * decode miss holds while its planes are read, when the pool has nothing to stream. */
    void piece(size_t n) {
        Clock::time_point now = end(Clock::now());
        if (budget_ns > 0 && pool->work() > 0) {
            pool->wait_work(budget_ns);
            const Clock::time_point t = Clock::now();
            const int64_t w = std::chrono::duration_cast<std::chrono::nanoseconds>(t - now).count();
            ++waited;
            wait_ns += (uint64_t)w;
            budget_ns -= w;
            capped = capped || (budget_ns <= 0 && pool->work() > 0);
            now = t;
        }
        ++pieces;
        bytes += n;
        busy = pool->work() > 0;
        if (busy) busy_bytes += n;
        else if (pool->backlog() > 0) ++reserved;
        t0 = now;
    }
    /* The piece being copied ended at `now`. */
    Clock::time_point end(Clock::time_point now) {
        if (busy) busy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
        busy = false;
        return now;
    }
};

}  // namespace

/* RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=1 and ..._INSERT_SLICE_KIB=N (stager.h, hx_ins_nt_). The
 * critical-path and DRAM reviews' sliced inserts: each of ~9.6 misses a decode token relays ~18 MB
 * into the RAM cache beside the pool's bursts, which are DRAM-bound. Cached stores read every line
 * of the slot first (glibc's memcpy streams only from 24 MiB: about 53 MB of traffic an insert,
 * 35 with streaming stores), and one insert of 3-5 ms spans several bursts, so a single wait before
 * it (..._INSERT_YIELD_US) could not keep it out of them. */
void FileStager::hx_ins_init() {
    hx_ins_nt_ = false;
    hx_ins_slice_ = 0;
    hx_ins_wait_ns_ = 0;
    const char* nt = std::getenv("RADIANCE_EXPERIMENTAL_DS41_INSERT_NT");
    const char* sl = std::getenv("RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB");
    const int64_t kib = sl && *sl ? std::clamp<int64_t>(std::atoll(sl), 0, (int64_t)1 << 20) : 0;
    if (!(nt && !std::strcmp(nt, "1")) && kib <= 0) return;
    if (!mover_->packed_cache()) {
        if (plan_->rank == 0)
            RAD_WARN("host experts: ..._INSERT_NT / ..._INSERT_SLICE_KIB write into the packed RAM cache only "
                     "(RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1); the background inserts copy as before");
        return;
    }
    hx_ins_nt_ = true;
    hx_ins_slice_ = (size_t)kib << 10;
    if (kib > 0) {
        int64_t us = 4000;
        if (const char* w = std::getenv("RADIANCE_EXPERIMENTAL_DS41_INSERT_WAIT_US"); w && *w)
            us = std::clamp<int64_t>(std::atoll(w), 0, 100000);
        hx_ins_wait_ns_ = us * 1000;
        /* The gate waits on the pool's submitted work (HostExpertPool::wait_work), whose sleepers
         * are woken after the draining job's ticket; wait_idle (..._INSERT_YIELD_US) is as before. */
    }
    if (plan_->rank == 0) {
        if (kib > 0)
            RAD_INFO("host experts: background inserts write a missed unit's codes by non-temporal stores in "
                     "pieces of %lld KiB; a one-token pass's insert sleeps before each piece while the pool has "
                     "submitted jobs unfinished (not for a miss's reservation alone), at most %lld us an insert "
                     "(RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB)",
                     (long long)kib, (long long)(hx_ins_wait_ns_ / 1000));
        else
            RAD_INFO("host experts: background inserts write a missed unit's codes by non-temporal stores "
                     "(RADIANCE_EXPERIMENTAL_DS41_INSERT_NT)");
    }
}

FileStager::InsertStats FileStager::insert_stats() const {
    InsertStats st;
    const auto get = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
    st.units = get(hx_ins_.units);
    st.d_units = get(hx_ins_.d_units);
    st.ns = get(hx_ins_.ns);
    st.d_ns = get(hx_ins_.d_ns);
    st.pieces = get(hx_ins_.pieces);
    st.nt_bytes = get(hx_ins_.nt_bytes);
    st.waited = get(hx_ins_.waited);
    st.wait_ns = get(hx_ins_.wait_ns);
    st.capped = get(hx_ins_.capped);
    st.busy_bytes = get(hx_ins_.busy_bytes);
    st.busy_ns = get(hx_ins_.busy_ns);
    st.reserved = get(hx_ins_.reserved);
    return st;
}

void FileStager::hx_bg_loop() {
    std::vector<uint8_t> scratch;
    bool told = false;
    std::unique_lock<std::mutex> lk(hx_bg_mu_);
    for (;;) {
        hx_bg_cv_.wait(lk, [&] { return hx_bg_stop_ || !hx_bg_q_.empty(); });
        if (hx_bg_q_.empty()) return;   /* stopping, nothing left */
        HxInsert job = std::move(hx_bg_q_.front());
        hx_bg_q_.pop_front();
        ++hx_bg_busy_;
        lk.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        /* RADIANCE_EXPERIMENTAL_DS41_INSERT_YIELD_US=N: an insert streams ~37 MB through DRAM, which
         * the pool's workers are bound by; it waits (asleep) up to N us for the pool to drain. */
        static const int64_t yield_ns = [] {
            const char* y = std::getenv("RADIANCE_EXPERIMENTAL_DS41_INSERT_YIELD_US");
            return y && *y ? (int64_t)std::clamp(std::atoi(y), 0, 100000) * 1000 : (int64_t)0;
        }();
        if (yield_ns > 0 && hx_pool_) hx_pool_->wait_idle(yield_ns);
        int rc;
        if (hx_ins_nt_ && hx_pool_) {
            /* Only a one-token pass's insert waits: prefill's misses come ~200 units a chunk and
             * would pile up behind the pool's bursts. */
            InsertGate g;
            g.pool = hx_pool_;
            g.budget_ns = hx_ins_slice_ > 0 && job.rows == 1 ? hx_ins_wait_ns_ : 0;
            ds41hx::PackCopy cp;
            cp.nt = true;
            cp.slice = hx_ins_slice_;
            cp.before = &InsertGate::before;
            cp.ctx = &g;
            rc = mover_->insert_unit_planes(*job.p, &scratch, &cp);
            g.end(std::chrono::steady_clock::now());
            if (job.rows > 0) {
                hx_ins_.pieces.fetch_add(g.pieces, std::memory_order_relaxed);
                hx_ins_.nt_bytes.fetch_add(g.bytes, std::memory_order_relaxed);
                hx_ins_.waited.fetch_add(g.waited, std::memory_order_relaxed);
                hx_ins_.wait_ns.fetch_add(g.wait_ns, std::memory_order_relaxed);
                hx_ins_.capped.fetch_add(g.capped ? 1 : 0, std::memory_order_relaxed);
                hx_ins_.busy_bytes.fetch_add(g.busy_bytes, std::memory_order_relaxed);
                hx_ins_.busy_ns.fetch_add(g.busy_ns, std::memory_order_relaxed);
                hx_ins_.reserved.fetch_add(g.reserved, std::memory_order_relaxed);
            }
        } else {
            rc = mover_->insert_unit_planes(*job.p, &scratch);
        }
        if (job.rows > 0) {
            const uint64_t dt = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - t0).count();
            hx_ins_.units.fetch_add(1, std::memory_order_relaxed);
            hx_ins_.ns.fetch_add(dt, std::memory_order_relaxed);
            if (job.rows == 1) {
                hx_ins_.d_units.fetch_add(1, std::memory_order_relaxed);
                hx_ins_.d_ns.fetch_add(dt, std::memory_order_relaxed);
            }
        }
        if (rc < 0 && !told) {
            RAD_WARN("host experts: relaying a missed unit into the RAM cache failed (%s); it stays "
                     "a miss", rad_strerror(rc));
            told = true;
        }
        /* ..._SIDE_STORE: the packed cache refused it over its raw-round budget, as it always will:
         * its planes stay for its later misses (hx_side_store.h). */
        if (rc >= 0 && hx_side_.on() && mover_->packed_refused(job.p->unit)) (void)hx_side_.offer(job.p);
        job.p.reset();
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
            for (auto& p : batch) hx_bg_q_.push_back(HxInsert{ p, 0 });
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
    const int64_t x_ld = AR > 1 ? A.stride[0] : (o.x ? o.x_ld : kHxEmbd);
    /* The layer's rows, and the address of the step's row 0 its operand starts r0 rows past (the
     * per-row cut slices it from layer_r0; every other pass reads from row 0). */
    const int64_t rows = layer_rows(li), r0 = layer_r0(li);
    const void* const x0 = A.data ? (const char*)A.data - r0 * x_ld * 2 : nullptr;
    const bool shape = A.data && so.data && of.data && A.dtype == RAD_BF16 && A.rank == 2 &&
        A.shape[1] == kHxEmbd && A.stride[1] == 1 && x_ld >= kHxEmbd &&
        (so.dtype == RAD_I32 || so.dtype == RAD_U32) && (of.dtype == RAD_I32 || of.dtype == RAD_U32) &&
        hx_numel(of) >= o.n_expert + 1;
    const bool live = hx_li_ == li && hx_stage_ == 2;
    if (live) {
        if (!shape || x0 != o.x || x_ld != o.x_ld || so.data != o.sorted || of.data != o.off ||
            AR != rows || T != rows * o.top_k || hx_numel(so) < T) {
            RAD_ERR("host experts: layer %d's gate_up reads rows other than the ones the pool was "
                    "given (%lld rows of %lld sorted from row %lld, a pass of %lld)", layers_[(size_t)li].layer,
                    (long long)AR, (long long)T, (long long)r0, (long long)pass_tok_);
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
        if (!shape || AR != rows || T != rows * o.top_k || hx_numel(so) < T) {
            RAD_ERR("host experts: layer %d's gate_up reads rows the CPU path cannot (%lld rows of "
                    "%lld sorted, a pass of %lld), and its card ops went ahead of its counts",
                    layers_[(size_t)li].layer, (long long)AR, (long long)T, (long long)pass_tok_);
            return RAD_E_STATE;
        }
        if (x0 != o.x || x_ld != o.x_ld || so.data != o.sorted || of.data != o.off) {
            ++hx_stats_.moved;
            ++hx_stats_.early_refetch;
            o.x = x0;
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
    if (shape && routed_ && AR == rows && AR >= 1 && AR <= ep_rows_ &&
        T == AR * o.top_k && hx_numel(so) >= T) {
        if (o.x && (x0 != o.x || x_ld != o.x_ld || so.data != o.sorted || of.data != o.off))
            ++hx_stats_.moved;
        o.x = x0;
        o.x_ld = x_ld;
        o.sorted = so.data;
        o.off = of.data;
    }
    return RAD_OK;
}

/* ..._LCP_PROFILE: a routed layer's counts reached (PassProfile, stager.h). */
void FileStager::prof_counts(int32_t li, uint64_t sync_ns, std::chrono::steady_clock::time_point t_counts) {
    PassProfile& p = prof_[prof_bucket()];
    ++p.layers;
    p.rows += (uint64_t)std::max<int64_t>(0, layer_rows(li));
    p.sync_ns += sync_ns;
    if (prof_last_counts_.time_since_epoch().count()) {
        ++p.spans;
        p.span_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_counts - prof_last_counts_).count();
    }
    prof_last_counts_ = t_counts;
}

/* The layer's gather, before its launch: the pool's rows over the card's in its y_expert, on
 * the gather's stream, or with ..._POLICY=verify the card's rows compared with the pool's. */
int FileStager::hx_finish(RadStream s, const RadTensor* t, int n_t) {
    if (hx_async_) {   /* the misses' jobs join the ticket first */
        const auto w0 = std::chrono::steady_clock::now();
        const int rc = hx_mw_wait();
        const uint64_t mw = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - w0).count();
        hx_stats_.async_wait_ns += mw;
        if (prof_on_) prof_[prof_bucket()].miss_wait_ns += mw;
        if (rc < 0) {
            hx_drain();   /* the layer is given up: no second report of rows that reached no gather */
            return rc;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    hx_pool_->wait(&hx_ticket_);
    const uint64_t waited = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t0).count();
    hx_stats_.wait_ns += waited;
    if (prof_on_) prof_[prof_bucket()].gather_ns += waited;
    if (pass_tok_ == 1) {   /* decode's own share, the misses' layers apart */
        const int m = hx_miss_.empty() ? 0 : 1;
        ++hx_stats_.d_waits[m];
        hx_stats_.d_wait_ns[m] += waited;
    }
    /* The aux worker: its rows placed (or compared, or computed by the CPU from the kept leases) before the
     * leases go back and before the copy below. */
    if (aux_ && layer_rows(hx_li_) <= ds41aux::kMaxRows) aux_stats_.eligible_rows += hx_rows_.size();
    if (aux_job_ != kAuxNone) RAD_TRY(aux_finish());
    for (int k = 0; k < hx_n_; ++k) mover_->release_host_unit(&hx_units_[k]);
    hx_n_ = 0;
    for (int k = 0; k < aux_n_; ++k) mover_->release_host_unit(&aux_units_[k]);
    aux_n_ = 0;
    aux_list_.clear();
    aux_take_ = false;
    aux_lane_leave();   /* a safety net: aux_finish left it */
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
    /* The aux worker's job is given up (its rows never read; the lane stays busy until it completes), its
     * fallback's jobs waited for, and its leases go back: nothing of it touches the stager's buffers. */
    aux_abandon();
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
            /* Prefill Phase 1's admission read the order from the addresses an earlier pass's gate_up
             * read: counted when this one reads others (the bits never depend on it). */
            if (p1_armed_ && p1_order_ && p1_li_ == li && n_t >= 4 &&
                (t[2].data != hx_ops_[(size_t)li].sorted || t[3].data != hx_ops_[(size_t)li].off))
                ++p1_order_moved_;
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
    if (!skipped_[b].empty()) {
        mover_->unstage_units(skipped_[b]);        /* the dummy publications go to null too */
        skipped_[b].clear();
    }
    /* Hot-cache units are NOT unstaged: their publications name their slots, which outlive
     * the buffer. The layer's copies into them landed before its ops ran (the done_ wait). */
    promoted_[b].clear();
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
    /* ..._HOST_EXPERTS_PREWAKE (stager.h): a one-token pass's router GEMV gets an event right before
     * it, which the layer's decision waits for ahead of its counts (hx_wake_route). */
    if (hx_wake_ && routed_ && pass_tok_ == 1 && (size_t)h < hx_wake_op_.size() && hx_wake_op_[(size_t)h] >= 0 &&
        moe_runs(layers_[(size_t)hx_wake_op_[(size_t)h]].layer)) {
        RAD_TRY(rad_event_record(hx_wake_ev_, s));
        hx_wake_li_ = hx_wake_op_[(size_t)h];
    }

    /* Read the pass's first routed layer at its first op of any kind -- unless the pass reads
     * routed experts only, which cannot be known before the layer's router has run. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (!routed_ && it != layers_.end()) {
            const int32_t li = (int32_t)(it - layers_.begin());
            if (will_issue(li)) {
                RAD_TRY(start(li, s));
                *resync = true;
            }
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    if (li == cur_ && tbo_) {
        /* A split pass: micro-batch B's slice of the layer, behind A's gather. */
        RAD_TRY(tbo_second(li, s));
        *resync = true;
        return RAD_OK;
    }
    if (li != cur_ && tbo_) tbo_enter(0);   /* a split pass enters a layer with micro-batch A */
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
        RAD_TRY(start(li, s));
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
        if (!read) RAD_TRY(start(li, s));
        *resync = true;
    }
    if (holds_[li & 1] == li && pending_[li & 1]) RAD_TRY(rad_event_wait(s, done_[li & 1]));
    cur_ = li;

    const int32_t next = li + 1;
    if (!routed_ && (size_t)next < layers_.size() && will_issue(next)) {
        RAD_TRY(start(next, s));
        *resync = true;
    }
    return RAD_OK;
}

/* RADIANCE_EXPERIMENTAL_SCHED_TBO (stager.h, tbo_): slice `slice` of the step's rows runs. */
void FileStager::tbo_enter(int slice) {
    tbo_slice_ = slice;
    pass_tok_ = tbo::rows_of(tbo_tok_, slice);
    hx_ra_pass_ = ra_pass_reads_ahead(true, 0, pass_tok_, hx_ra_rows_);
}

bool FileStager::tbo_keeps_ra() const {
    return tbo_ && tbo_slice_ == 0 &&
           ra_pass_reads_ahead(true, 0, tbo::rows_of(tbo_tok_, 1), hx_ra_rows_);
}

/* Layer li's second gate_up in a split pass: slice A has had its gather (its CPU rows were copied
 * there, or it had none), so its holds go and slice B's reads start as a layer's first op's do. */
int FileStager::tbo_second(int32_t li, RadStream s) {
    if (tbo_slice_ != 0) {
        RAD_ERR("two-batch overlap: layer %d's gate_up issued a third time in one pass",
                layers_[(size_t)li].layer);
        return RAD_E_STATE;
    }
    if ((hx_li_ == li && hx_stage_ == 2) || hx_early_li_ == li) {
        RAD_ERR("two-batch overlap: layer %d's slice B issued before slice A's gather",
                layers_[(size_t)li].layer);
        return RAD_E_STATE;
    }
    hx_drain();
    tbo_enter(1);
    const int b = li & 1;
    if (holds_[b] == li) RAD_TRY(finish(b, s));
    bool read = false;
    const bool ep = ep_counts_ && pass_tok_ > 0 && pass_tok_ <= ep_rows_ &&
                    (size_t)li < ep_layer_eligible_.size() && ep_layer_eligible_[(size_t)li];
    if (routed_ && ep) RAD_TRY(start_ep_routed(li, s, &read));
    if (routed_ && !ep) RAD_TRY(start_routed(li, s, &read));
    if (!read) RAD_TRY(start(li, s));
    if (holds_[b] == li && pending_[b]) RAD_TRY(rad_event_wait(s, done_[b]));
    ++tbo_slices_;
    return RAD_OK;
}

int FileStager::quiesce_for_draft_bank() {
    if (draft_bank_lend_conflict() || armed_ || p1_armed_ || hx_stage_ || hx_li_ >= 0 ||
        hx_early_li_ >= 0) return RAD_E_STATE;
    if (p1_) RAD_TRY(p1_->quiesce_for_draft_bank());
    drain_host_inserts();
    return RAD_OK;
}

int FileStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    /* Prefill Phase 1: every copy seen and every source let go before the buffers are released; the
     * mover's stream then waits behind this pass's last reader of the ring. */
    int p1_rc = RAD_OK;
    /* The lent ring: the grace runs from the last pass that took the slots back, and the residents' last
     * reader so far is behind this pass, which a later armed pass's first ring copy waits on. */
    if (p1_ && p1_->lending()) {
        if (p1_lend_took_) p1_->lender().end_p1(p1_now_ms());
        if (p1_->lender().residents() > 0) {
            const int r = s ? rad_event_record(p1_lend_ev_, s) : RAD_E_STATE;
            if (r >= 0) p1_lend_rec_ = true;
            else p1_rc = r;
            /* The heat policy's fence asks for the record of the pass right before its own. */
            if (r >= 0 && p1_heat_) p1_lend_rec_pass_ = p1_lend_pass_;
        }
        /* The lent arena: a pass at the full plan that the card path did not take (an armed pass's own
         * end_pass does this below) wrote the arena's top with nothing of it on the DMA lane, so the
         * mover's stream waits on its end: a later reload copy into an arena slot lands behind its last
         * reader of those bytes. */
        if (p1_lend_took_ && !p1_armed_ && p1_->lend_arena_slots() > 0) {
            int r = s ? rad_event_record(p1_end_, s) : RAD_E_STATE;
            if (r >= 0) r = mover_->mover_wait(p1_end_);
            if (r < 0 && p1_rc >= 0) p1_rc = r;
        }
        p1_lend_took_ = false;
    }
    if (p1_armed_) {
        p1_armed_ = false;
        if (p1_li_ >= 0)
            RAD_ERR("ds41 p1: routed layer %d's down projection was never issued this pass",
                    layers_[(size_t)p1_li_].layer);
        const int er = p1_->end_pass(s, p1_end_);
        if (p1_rc >= 0) p1_rc = er;
        if (p1_li_ >= 0 && p1_rc >= 0) p1_rc = RAD_E_STATE;
        p1_li_ = -1;
        if (plan_->rank == 0) {
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - p1_t0_).count();
            const auto& c = p1_->counts();
            static uint64_t units0 = 0, bytes0 = 0;
            const uint64_t du = c.drive_units - units0, db = c.drive_bytes - bytes0;
            units0 = c.drive_units;
            bytes0 = c.drive_bytes;
            long long avail_kb = -1;
            if (FILE* f = std::fopen("/proc/meminfo", "r")) {
                char line[256];
                while (std::fgets(line, sizeof line, f))
                    if (std::sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
                std::fclose(f);
            }
            RAD_INFO("%s rows %lld, layers %llu (and %llu cut below %lld rows, on the CPU path), wall %.3f s "
                     "(issue), rank 0 drive units %llu, %.2f GB/s, MemAvailable %lld MiB", ds41p1::kLogChunk,
                     (long long)p1_pass_rows_, (unsigned long long)p1_layers_, (unsigned long long)p1_cpu_layers_,
                     (long long)p1_min_rows_, wall, (unsigned long long)du,
                     wall > 0 ? (double)db * 1e-9 / wall : 0.0, avail_kb >= 0 ? avail_kb / 1024 : -1);
        }
    }
    /* An early-issued layer whose gather never came is lost too: its card ops ran on aliases. */
    const bool lost = (hx_li_ >= 0 && hx_stage_ == 2) || hx_early_li_ >= 0;
    if (lost) RAD_ERR("host experts: layer %d's CPU rows reached no gather before the pass ended",
                      layers_[(size_t)(hx_early_li_ >= 0 ? hx_early_li_ : hx_li_)].layer);
    hx_drain();
    tbo_ = false;
    /* A cut pass ends early by design: its last op says nothing of where a whole pass ends. */
    if (pass_full_ == INT32_MAX) pass_last_[pass_kind_] = last_issued_;
    int rc = lost ? RAD_E_STATE : RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0) {
            const int r = finish(b, s);
            if (r < 0 && rc >= 0) rc = r;
        }
    cur_ = -1;
    if (rc >= 0 && p1_rc < 0) rc = p1_rc;
    return rc<0 ? rc : head_status_;
}

int FileStager::quiesce_for_backing() {
    if (p1_on() || p1_lend_arena_asked()) return RAD_E_UNSUPPORTED;
    if (armed_ || hx_stage_ != 0 || hx_li_ >= 0 || hx_early_li_ >= 0 || p1_armed_)
        return RAD_E_STATE;
    /* end_pass/hx_drain already joined miss posts and expert jobs before the rank barrier. The
     * remaining insert threads touch immutable expert planes/RAM cache only, never KV. */
    drain_host_inserts();
    return RAD_OK;
}

/* ---- prefill Phase 1 (card_expert_stager.h) ---------------------------------------------------------- */

/* RADIANCE_EXPERIMENTAL_DS41_P1: the switch's checks beyond lcp_init's bands and hx_init's CPU path, and
 * the path's start. Off (unset, empty, 0): nothing. Refused, with the reason: an invalid value; P1
 * without an admitted LCP chunk or beyond the proven twin's bound; the tunables out of range;
 * FETCH_LIST; the route trace;
 * the DSpark head tail; a routed layer the counts path or the CPU path does not take, or whose gate_up,
 * SwiGLU and down cannot be named; a routed unit off the card neither in the container nor pinned in
 * the host pool of a static plan (a pooled unit is copied to the ring from its pool slot, p1_pool_; a
 * plan that moves units may relocate it between steps); a ring of fewer than kMinRingSlots slots. */
int FileStager::p1_init(const Program& prog) {
    p1_.reset();
    p1_armed_ = false;
    const char* v = std::getenv(ds41p1::kEnv);
    const int st = ds41p1::env_state(v);
    p1_state_ = st;
    if (st == 0) return RAD_OK;
    auto refuse = [&](const std::string& why) {
        RAD_ERR("%s=%s (prefill Phase 1, card experts) refused: %s", ds41p1::kEnv, v ? v : "", why.c_str());
        return RAD_E_INVAL;
    };
    if (st < 0) return refuse("1 is on, 0 / unset / empty off");
    if (!enabled() || !mover_ || !plan_) return refuse("no routed layer is read from the container on this rank");
    if (!lcp_) return refuse(fmt("it needs large-chunk prefill (%s=<C>, the chunk the cards take)", kLcpEnv));
    if (lcp_ > ds41p1::kTwinMaxRows || lcp_ > kEpRouteLcpMax)
        return refuse(fmt("%s=%lld: chunks must fit the admitted LCP and P1 twin bounds (up to %lld)",
                          kLcpEnv, (long long)lcp_,
                          (long long)std::min<int64_t>(ds41p1::kTwinMaxRows, kEpRouteLcpMax)));
    p1::Tunables t;
    std::string why;
    if (!p1::parse_tunables([](const char* k) { return std::getenv(k); }, &t, &why)) return refuse(why);
    if (t.lend_heat > 0 && t.lend <= 0 && t.lend_arena <= 0 && t.lend_slab <= 0)
        return refuse(fmt("%s=%d moves lent slots between units, and none is lent (%s, %s or %s)", ds41p1::kLendHeatEnv,
                          t.lend_heat, ds41p1::kLendEnv, ds41p1::kLendArenaEnv, ds41p1::kLendSlabEnv));
    if (t.lend_audit && t.lend <= 0 && t.lend_arena <= 0 && t.lend_slab <= 0)
        return refuse(fmt("%s=1 reads lent slots back, and none is lent (%s, %s or %s)", ds41p1::kLendAuditEnv,
                          ds41p1::kLendEnv, ds41p1::kLendArenaEnv, ds41p1::kLendSlabEnv));
    /* The lent slab: the plan gave up residents for exactly these slots (Planner::assign_expert_tiered, which
     * read the same switch), and the mover holds them. */
    if (t.lend_slab > 0) {
        int64_t slab_bytes = 0;
        const char* const slab = mover_->lend_slab_region(&slab_bytes);
        if ((int)plan_->lend_slab_units.size() != t.lend_slab || !slab ||
            slab_bytes < (int64_t)t.lend_slab * ds41p1::kSlotBytes)
            return refuse(fmt("%s=%d: this rank's plan set %zu slots aside for it (%s of VRAM); it does only in a "
                              "static plan (--deterministic) with the file tier and a weights budget, and with the "
                              "switch in the environment when the plan was made",
                              ds41p1::kLendSlabEnv, t.lend_slab, plan_->lend_slab_units.size(),
                              humanb(slab ? slab_bytes : 0).c_str()));
    } else if (!plan_->lend_slab_units.empty() || plan_->lend_slab_vram > 0) {
        return refuse(fmt("the plan set %zu slots aside for the lent slab and %s is not set now",
                          plan_->lend_slab_units.size(), ds41p1::kLendSlabEnv));
    }
    if (const char* fl = std::getenv("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST"); fl && !std::strcmp(fl, "1"))
        return refuse("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1 is not covered");
    /* Speculation: only what lcp_init took beside the chunk (lcp_spec_; ep_route_counts.h), prompt lookup
     * alone in steps of at most 16 decode rows. Such a step never arms the card path (p1_pass_arms: no
     * decode row, no speculation), so its routed layers run as without the switch: the residents, the
     * lent slots' among them, in the plugin's own ops on the decode band's rows, every other unit on the
     * CPU path; and its arena level is the small one (Engine::arena_step sizes a step by its sequences'
     * 1 + drafts rows), so it never takes the lent arena's slots back. A drafter's passes, and steps of
     * more decode rows, are refused as before. */
    if (prog.ctx.max_spec > 0 && !lcp_spec_) {
        const LcpSpec spec = ds41_lcp_spec(prog.ctx.max_seqs, prog.ctx.max_spec, prog.drafter.kind != RAD_DRAFT_NONE);
        return refuse(spec == LcpSpec::Rows
                          ? "speculation is on: " + ds41_lcp_spec_rows_why(prog.ctx.max_seqs, prog.ctx.max_spec)
                          : std::string("speculation is on"));
    }
    /* Set but empty is off (ab_window.sh states the switch on every arm without Rt). */
    if (const char* rte = std::getenv("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE"); rt_ || (rte && *rte))
        return refuse("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE is on: its records follow the CPU path's choice");
    if (head_.configured()) return refuse("the DSpark head tail holds the routed layers' file buffers");
    if (!hx_pool_ || !hx_capped_)
        return refuse("the CPU path must take the passes of up to 16 tokens with the file buffers capped "
                      "(host experts, ..._POLICY=all, ..._MISSES, ..._FILE_BUFFER_UNITS)");
    /* Every routed layer: on the counts path, with its gate_up (hx_ops_), one SwiGLU and its down. */
    p1_ops_.assign(layers_.size(), P1Ops{});
    for (size_t li = 0; li < layers_.size(); ++li) {
        const Layer& L = layers_[li];
        if (li >= ep_layer_eligible_.size() || !ep_layer_eligible_[li] || li >= hx_ops_.size() ||
            hx_ops_[li].gu != L.first_op)
            return refuse(fmt("routed layer %d is not on the EP counts path with its gate_up found", L.layer));
        int32_t sw = -1;
        std::string others;
        for (int32_t h = L.first_op + 1; h < L.last_op; ++h) {
            const OpInfo& op = prog.ops[(size_t)h];
            if (op.op == "swiglu_clamped" && sw < 0) sw = h;
            else others += (others.empty() ? "" : ", ") + op.op;
        }
        const OpInfo& dn = prog.ops[(size_t)L.last_op];
        const char* ord = dn.base.get_s("a_order");
        if (sw < 0 || !others.empty() || dn.op != "moe_gemm_ggml" || !ord || std::strcmp(ord, "sorted"))
            return refuse(fmt("routed layer %d's expert ops are not gate_up, one swiglu_clamped and down (ops %d..%d%s%s)",
                              L.layer, L.first_op, L.last_op, others.empty() ? "" : "; also ", others.c_str()));
        p1_ops_[li].sw = sw;
        p1_ops_[li].dn = L.last_op;
    }
    /* The routed units of these layers off the card: file-tier units (the ring takes them from the RAM
     * cache or the drive) and, in a static plan, units pinned in the host pool, which the ring takes
     * from their pool slots (p1_pool_) instead of the card reading them across the link once a row
     * pass. Anything else is refused. */
    {
        std::vector<int32_t> li_of((size_t)std::max<int32_t>(n_model_layers_, 1) + 64, -1);
        for (size_t li = 0; li < layers_.size(); ++li)
            if (layers_[li].layer >= 0 && (size_t)layers_[li].layer < li_of.size())
                li_of[(size_t)layers_[li].layer] = (int32_t)li;
        p1_pool_.assign(layers_.size(), {});
        p1_pool_units_ = 0;
        int other = 0, first = -1;
        const char* first_why = "";
        for (size_t u = 0; u < plan_->units.size(); ++u) {
            const MoveUnit& mu = plan_->units[u];
            if (mu.layer < 0 || (size_t)mu.layer >= li_of.size() || li_of[(size_t)mu.layer] < 0 ||
                mu.access != RAD_ACCESS_CONDITIONAL || mu.tier == Tier::VRAM || mu.tier == Tier::SSD)
                continue;
            std::vector<int32_t>& v = p1_pool_[(size_t)li_of[(size_t)mu.layer]];
            bool dup = false;
            for (int32_t w : v) dup = dup || plan_->units[(size_t)w].expert == mu.expert;
            p1::Source src;
            const char* no = mu.tier != Tier::HostPinned ? "neither in VRAM, the container nor the host pool"
                           : plan_->dynamic ? "pinned in the host pool of a plan that moves units between steps"
                           : mu.expert < 0 || mu.expert >= 96 || dup ? "not one unit of a local expert"
                           : !p1_pool_source((int32_t)u, &src) ? "pinned, but not a block_mxfp4 gate_up and down pair "
                                                                 "with a pool address"
                                                               : nullptr;
            if (!no) {
                v.push_back((int32_t)u);
                ++p1_pool_units_;
                continue;
            }
            if (!other++) {
                first = (int)u;
                first_why = no;
            }
        }
        if (other)
            return refuse(fmt("%d routed units of these layers are off the card in a form the card path cannot copy (the "
                              "first: unit %d, layer %d, %s); it takes file-tier units, and units pinned in the host "
                              "pool of a static plan", other, first, plan_->units[(size_t)first].layer, first_why));
    }
    if (!p1_end_ && rad_event_create_local(&p1_end_) < 0) return refuse("could not create its events");
    if (t.verify > 0) {
        const int64_t n = t.verify;
        if (!p1_vidx_) p1_vidx_ = (int32_t*)rad_dev_alloc((6 * lcp_ + 128) * 4, RAD_MEM_HOST_PINNED);
        if (!p1_vx_) p1_vx_ = (uint16_t*)rad_dev_alloc(n * kHxEmbd * 2, RAD_MEM_HOST_PINNED);
        if (!p1_vy_) p1_vy_ = (uint16_t*)rad_dev_alloc(n * kHxEmbd * 2, RAD_MEM_HOST_PINNED);
        if (!p1_vh_) p1_vh_ = (uint16_t*)rad_dev_alloc(n * kHxEmbd * 2, RAD_MEM_HOST_PINNED);
        if (!p1_vev_ && rad_event_create(&p1_vev_) < 0) p1_vev_ = nullptr;
        if (!p1_vidx_ || !p1_vx_ || !p1_vy_ || !p1_vh_ || !p1_vev_)
            return refuse("no pinned memory or event for the host verify (..._P1_VERIFY)");
    }
    auto path = std::make_unique<p1::CardExpertPath>();
    const int rc = path->init(
        mover_, t, plan_->rank, [this](int32_t u) { return hx_planes_get(u); },
        [this](std::shared_ptr<Mover::HostPlanes> p, int rows) {
            /* An admitted miss to the insert threads, at most ..._LCP_INSERTQ queued at once. */
            {
                std::lock_guard<std::mutex> lk(hx_bg_mu_);
                if (hx_bg_.empty() || (int64_t)hx_bg_q_.size() + hx_bg_busy_ >= hx_lcp_insertq_) {
                    ++hx_stats_.lcp_miss_dropped;
                    return;
                }
                ++hx_stats_.lcp_miss_admit;
                hx_bg_q_.push_back(HxInsert{ std::move(p), rows });
            }
            hx_bg_cv_.notify_all();
        },
        &why);
    if (rc < 0) return refuse(why);
    p1_ = std::move(path);
    p1_min_rows_ = t.min_rows;
    p1_tun_ = t;
    /* With the lent arena the lending waits for the arena's top (p1_lend_arena, once the engine has bound
     * its level), and the ring's slots join it there. */
    p1_arena_asked_ = t.lend_arena > 0;
    if ((t.lend > 0 || t.lend_slab > 0) && !p1_arena_asked_ && p1_lend_init(t, {}, &why) < 0) {
        p1_.reset();
        return refuse(why);
    }
    if (plan_->rank == 0)
        RAD_INFO("%s: every routed expert row of a pure prefill pass's layer of %lld to %lld rows on the cards "
                 "(%s=%lld; a shorter layer pass takes the CPU path, up to %lld rows); ring %d slots of %lld bytes "
                 "(%.0f MiB a card), waves of %d units, %d drive units in flight, %d pinned ready slots, %d units "
                 "read ahead, verify %d expert-rows a layer, chunks aligned to the %s; %d routed units pinned in the "
                 "host pool copied to the ring from their pool slots; the RAM cache read-only but for the units the "
                 "prompt's last rows select (..._LCP_ADMIT)",
                 ds41p1::kLogOn, (long long)p1_min_rows_, (long long)lcp_, ds41p1::kMinRowsEnv, (long long)p1_min_rows_,
                 (long long)p1_cpu_rows(), p1_->ring_slots(), (long long)ds41p1::kSlotBytes,
                 (double)p1_->ring_slots() * (double)ds41p1::kSlotBytes / 1048576.0, t.wave, t.inflight, t.ready,
                 t.ahead, t.verify, t.align_end ? "prompt's end" : "chunk grid (start)", p1_pool_units_);
    if (plan_->rank == 0 && lcp_spec_)
        RAD_INFO("ds41 p1: prompt-lookup speculation admitted (--num-speculative-tokens %d, no draft head): a verify "
                 "step of at most %lld decode rows never arms the card path; its residents run in the plugin's own ops "
                 "on the decode band's rows and every other routed unit on the CPU path, as in any decode pass",
                 lcp_spec_, (long long)lcp_spec_rows_);
    return RAD_OK;
}

bool FileStager::p1_pool_source(int32_t u, p1::Source* out) const {
    if (!mover_ || !plan_ || !ep_program_ || u < 0 || (size_t)u >= plan_->units.size() || !out) return false;
    const MoveUnit& mu = plan_->units[(size_t)u];
    const char* const base = (const char*)mover_->unit_base(u);
    if (mu.tier != Tier::HostPinned || !base || mover_->file_unit_in_slab(u) || mu.weights.size() != 2 ||
        mu.offset.size() != 2)
        return false;
    p1::Source src;
    for (size_t k = 0; k < 2; ++k) {
        const WeightInfo& w = ep_program_->weights[mu.weights[k]];
        const bool gu = w.stored_bytes == ds41p1::kGuBlockBytes;
        if ((!gu && w.stored_bytes != ds41p1::kDnBlockBytes) || w.layout_tag != "ggml_mxfp4" ||
            w.decl.shard != RAD_SHARD_NONE)
            return false;
        (gu ? src.gu : src.dn) = base + mu.offset[k];
        (gu ? src.gu_bytes : src.dn_bytes) = w.stored_bytes;
    }
    if (!src.gu || !src.dn) return false;
    *out = src;
    return true;
}

/* The lent ring (card_expert_stager.h): the ring's slots that neither file buffer's alias base names,
 * each lent to one of this rank's hottest file-tier units by the profile (a trunk layer's, one the CPU
 * path can read), at most ..._P1_LEND of them; then the lent arena's slots (`arena`, p1_lend_arena), up
 * to ..._P1_LEND_ARENA of them, to the next hottest. The ring's reloads begin at once and every pass
 * from the next publishes what has landed; with arena slots the reloads begin at the first pass that
 * does not take the slots back (a step below the full plan: the arena's top is not free before one).
 * Refused when a slot kind asked for has no slot or no unit left to lend. */
int FileStager::p1_lend_init(const p1::Tunables& t, const std::vector<char*>& arena, std::string* why) {
    const char* const ring = p1_->ring_base();
    const int n_slots = p1_->ring_slots();
    std::vector<int> slots;
    for (int r = 0; t.lend > 0 && r < n_slots; ++r) {
        const char* at = ring + (int64_t)r * ds41p1::kSlotBytes;
        bool alias = false;
        for (int b = 0; b < 2; ++b)
            alias = alias || (buf_[b] && at < buf_[b] + ds41p1::kSlotBytes && buf_[b] < at + ds41p1::kSlotBytes);
        if (!alias) slots.push_back(r);
    }
    std::vector<int32_t> cand;
    for (size_t li = 0; li < layers_.size(); ++li) {
        if (layers_[li].layer < 0 || layers_[li].layer >= n_model_layers_ || li >= ep_layer_eligible_.size() ||
            !ep_layer_eligible_[li])
            continue;
        for (int32_t u : layers_[li].units) {
            const MoveUnit& mu = plan_->units[(size_t)u];
            if (mu.bytes == ds41p1::kSlotBytes && mu.weights.size() == 2 && mu.offset.size() == 2 &&
                !mover_->file_unit_in_slab(u) && mover_->host_planes_eligible(u))
                cand.push_back(u);
        }
    }
    /* ..._P1_LEND_HEAT: every unit a slot may hold, as found (the policy's candidates). */
    std::vector<int32_t> heat_cand;
    if (t.lend_heat > 0) heat_cand = cand;
    /* ..._P1_LEND_SLAB: the slab's slots and their units, which the plan chose (Plan::lend_slab_units: its
     * hottest file-tier units once it had given up residents for the slots). Each must be a unit a slot can
     * hold; the ring's and the arena's slots then take the hottest of the others. */
    std::vector<int32_t> slab_units;
    std::vector<char*> slab_at;
    if (t.lend_slab > 0) {
        int64_t slab_bytes = 0;
        char* const slab = mover_->lend_slab_region(&slab_bytes);
        std::vector<uint8_t> is_cand(plan_->units.size(), 0), in_slab(plan_->units.size(), 0);
        for (int32_t u : cand) is_cand[(size_t)u] = 1;
        for (int32_t u : plan_->lend_slab_units) {
            if (u < 0 || (size_t)u >= plan_->units.size() || !is_cand[(size_t)u] || in_slab[(size_t)u]) {
                if (why)
                    *why = fmt("%s=%d: unit %d, which the plan gave the lent slab, is not a unit a slot can hold (a "
                               "trunk layer's file-tier unit of %lld bytes in two weights that the CPU path reads)",
                               ds41p1::kLendSlabEnv, t.lend_slab, (int)u, (long long)ds41p1::kSlotBytes);
                return RAD_E_INVAL;
            }
            in_slab[(size_t)u] = 1;
            slab_units.push_back(u);
        }
        if (!slab || (int)slab_units.size() != t.lend_slab ||
            slab_bytes < (int64_t)t.lend_slab * ds41p1::kSlotBytes) {
            if (why) *why = fmt("%s=%d: the plan set %zu slots aside", ds41p1::kLendSlabEnv, t.lend_slab, slab_units.size());
            return RAD_E_INVAL;
        }
        for (int i = 0; i < t.lend_slab; ++i) slab_at.push_back(slab + (int64_t)i * ds41p1::kSlotBytes);
        cand.erase(std::remove_if(cand.begin(), cand.end(), [&](int32_t u) { return in_slab[(size_t)u] != 0; }),
                   cand.end());
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t c) {
        return plan_->units[(size_t)a].profile_share > plan_->units[(size_t)c].profile_share;
    });
    const int alias_slots = n_slots - (int)slots.size();
    const int k = std::min<int>(t.lend, (int)std::min(slots.size(), cand.size()));
    if (t.lend > 0 && k <= 0) {
        if (why)
            *why = fmt("%s=%d: no ring slot (%d slots, %zu past the file buffers' alias bases) or file-tier unit "
                       "(%zu) to lend", ds41p1::kLendEnv, t.lend, n_slots, slots.size(), cand.size());
        return RAD_E_INVAL;
    }
    const int ka = std::min<int>((int)arena.size(), (int)cand.size() - std::max(k, 0));
    if (!arena.empty() && ka <= 0) {
        if (why)
            *why = fmt("%s=%d: no file-tier unit left to lend (%zu, %d of them to the ring)", ds41p1::kLendArenaEnv,
                       t.lend_arena, cand.size(), std::max(k, 0));
        return RAD_E_INVAL;
    }
    /* The lent slots' units in the lender's order: the ring's, the arena's, the slab's. */
    const int nk = std::max(k, 0), nra = nk + std::max(ka, 0), ns = (int)slab_units.size(), n = nra + ns;
    std::vector<int32_t> lent(cand.begin(), cand.begin() + nra);
    lent.insert(lent.end(), slab_units.begin(), slab_units.end());
    std::vector<p1::CardExpertPath::LendUnit> units((size_t)n);
    for (int i = 0; i < n; ++i) {
        const MoveUnit& mu = plan_->units[(size_t)lent[(size_t)i]];
        p1::CardExpertPath::LendUnit& lu = units[(size_t)i];
        lu.unit = lent[(size_t)i];
        for (int w = 0; w < 2; ++w) {
            lu.off[w] = mu.offset[(size_t)w];
            lu.bytes[w] = ep_program_->weights[mu.weights[(size_t)w]].stored_bytes;
        }
    }
    slots.resize((size_t)nk);
    const std::vector<char*> at(arena.begin(), arena.begin() + std::max(ka, 0));
    if (!p1_lend_ev_ && rad_event_create_local(&p1_lend_ev_) < 0) {
        p1_lend_ev_ = nullptr;
        if (why) *why = "could not create the lent ring's event";
        return RAD_E_DEVICE;
    }
    p1_lend_rec_ = false;
    /* The lent slots' heat (..._P1_LEND_HEAT=S): made only with the switch; without it none of it exists. */
    std::unique_ptr<p1::LendHeat> heat;
    if (t.lend_heat > 0) {
        heat = std::make_unique<p1::LendHeat>();
        if (heat->init(plan_->units.size(), t.lend_heat, t.lend_heat_half, t.lend_heat_margin) < 0) {
            if (why)
                *why = fmt("%s=%d with %s=%d and %s=%d: not a policy", ds41p1::kLendHeatEnv, t.lend_heat,
                           ds41p1::kLendHeatHalfEnv, t.lend_heat_half, ds41p1::kLendHeatMarginEnv, t.lend_heat_margin);
            return RAD_E_INVAL;
        }
    }
    RAD_TRY(p1_->lend_init(units, slots, at, slab_at, t.grace_ms, t.lend_audit, why));
    /* The rest after a long prefill (..._P1_LEND_REST_MS; by default with the heat or the slab only). */
    p1_->lend_set_rest(t.lend_rest_ms);
    if (plan_->rank == 0 && t.lend_rest_ms > 0)
        RAD_INFO("%s: %d ms (%s=%d%s): from a pass that takes the lent slots back (a card-path prefill pass, or a "
                 "step at the lent arena's full plan) until %d ms after the last such pass ends, the heat policy "
                 "moves no slot (declined: a long prefill or its rest) and no lent reload reads the drive (one the "
                 "RAM cache gives begins; one already queued waits); the slab's residents stay published",
                 ds41p1::kLogLendRest, t.lend_rest_ms, ds41p1::kLendRestEnv, t.lend_rest_ms,
                 std::getenv(ds41p1::kLendRestEnv) && *std::getenv(ds41p1::kLendRestEnv) ? "" : ", the default beside the heat or the slab",
                 t.lend_rest_ms);
    if (plan_->rank == 0 && t.lend_audit)
        RAD_INFO("%s: on (%s=1): every lent slot is copied back from the card and compared, once its copy has "
                 "passed and before its unit is published (byte for byte with the bytes copied) and again before "
                 "its resident is withdrawn (64-bit digests a 4 KiB page of what was published, behind the pass "
                 "before); a slot that differs is given up, its unit takes the CPU path and the rank's retargets "
                 "stop; each read-back is %.0f MiB off the card that the issuing thread waits for, so this is not a "
                 "timed run", ds41p1::kLogLendAudit, ds41p1::kLendAuditEnv, (double)ds41p1::kSlotBytes / 1048576.0);
    /* The units lent (the ring's, the arena's and the slab's): never the aux expert worker's (p1_lent). */
    p1_lent_.assign(plan_->units.size(), 0);
    for (int i = 0; i < n; ++i) p1_lent_[(size_t)lent[(size_t)i]] = 1;
    p1_heat_ = std::move(heat);
    p1_heat_pass_ = false;
    p1_heat_cand_ = std::move(heat_cand);
    p1_lend_pass_ = p1_lend_rec_pass_ = 0;
    if (at.empty()) (void)p1_->lender().begin_pass(false, p1_now_ms());   /* the first reloads begin now */
    /* The lent slab's units are in before the first pass: no pass writes its slots, so nothing waits for a
     * step below the full plan, and the card then holds them from the first token on. One that is not in
     * within the bound (or failed) takes the CPU path until a later pass's begin publishes it. */
    if (ns > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        const int in = p1_->lend_prime(kP1SlabPrimeMs);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double share = 0.0;
        for (int32_t u : slab_units) share += plan_->units[(size_t)u].profile_share;
        if (in < ns)
            RAD_WARN("%s: rank %d: %d of the %d slots hold their unit after %.1f s of bring-up; the others' units "
                     "take the CPU path until their reloads are in", ds41p1::kLogLendSlab, plan_->rank, in, ns, secs);
        if (plan_->rank == 0)
            RAD_INFO("%s: %d slots (%s=%d; %s of VRAM a card that no pass writes, for which the plan gave up %lld "
                     "static residents, coldest first) hold rank 0's %d hottest file-tier units of that plan (profile "
                     "share %.4f) as decode residents from bring-up on (%d loaded in %.1f s, from the RAM cache where it "
                     "holds the unit, relaid from the packed form where it is packed, else read from the drive); no "
                     "pass takes them back and they "
                     "know no grace; a unit not loaded, or whose load failed, takes the CPU path",
                     ds41p1::kLogLendSlab, ns, ds41p1::kLendSlabEnv, t.lend_slab, humanb(plan_->lend_slab_vram).c_str(),
                     (long long)plan_->lend_slab_given, ns, share, in, secs);
    }
    if (plan_->rank == 0 && nk > 0) {
        double share = 0.0;
        for (int i = 0; i < nk; ++i) share += plan_->units[(size_t)cand[(size_t)i]].profile_share;
        RAD_INFO("%s: %d of the ring's %d slots (%d of them past the file buffers' alias bases; %s=%d) hold "
                 "rank 0's %d hottest file-tier units (profile share %.4f) as decode residents outside the card "
                 "path's passes; a pass that arms it takes them back, and %d ms after its last pass (%s) they are "
                 "reloaded in the background (from the RAM cache where it holds the unit, its block form as it is or "
                 "the packed form relaid, else read from the drive and relaid, into two pinned buffers of %.0f MiB) "
                 "and published again; a unit not "
                 "yet reloaded, or whose reload failed, takes the CPU path",
                 ds41p1::kLogLend, nk, n_slots, n_slots - alias_slots, ds41p1::kLendEnv, t.lend, nk, share, t.grace_ms,
                 ds41p1::kGraceEnv,
                 (double)ds41p1::kSlotBytes / 1048576.0);
    }
    if (plan_->rank == 0 && !at.empty()) {
        double share = 0.0;
        for (int i = nk; i < nra; ++i) share += plan_->units[(size_t)cand[(size_t)i]].profile_share;
        RAD_INFO("%s: %d slots (%s=%d; %zu fit) in the activation arena's top, above the buffers of steps of up "
                 "to %lld tokens and the guard, hold rank 0's next %d hottest file-tier units (profile share "
                 "%.4f) as decode residents; a step at the full plan (above %lld tokens: every card-path pass "
                 "among them) takes them back, its ops behind any reload copy still queued, and %d ms after the "
                 "last such step (%s) they are reloaded in the background as the lent ring's are (from the RAM cache "
                 "where it holds the unit, relaid from the packed form where it is packed, else from the drive) and "
                 "published again; a unit not yet reloaded takes the CPU path",
                 ds41p1::kLogLendArena, ka, ds41p1::kLendArenaEnv, t.lend_arena, arena.size(),
                 (long long)p1_arena_rows_, ka, share, (long long)p1_arena_rows_, t.grace_ms, ds41p1::kGraceEnv);
    }
    if (plan_->rank == 0 && p1_heat_)
        RAD_INFO("%s: on (%s=%d; half-life %d decode rows, %s; margin %d%%, %s): each rank's lent slots (rank 0: "
                 "%d) follow live decode routing, at most %d handed to another unit a pass; a unit's heat is the "
                 "decode rows routed to it in pure trunk decode passes, halved every %d decode rows; between "
                 "passes, while every slot holds its resident, the coldest residents give their slots to the "
                 "hottest lendable units off the card (rank 0: %zu lendable in all) whose heat is above %.2f x the "
                 "resident's + %.1f rows; the DMA lane first waits behind the pass before, the withdrawn unit takes "
                 "the CPU path, and the new one is reloaded, copied and published as any lent unit",
                 ds41p1::kLogLendHeat, ds41p1::kLendHeatEnv, t.lend_heat, t.lend_heat_half, ds41p1::kLendHeatHalfEnv,
                 t.lend_heat_margin, ds41p1::kLendHeatMarginEnv, n, t.lend_heat, t.lend_heat_half,
                 p1_heat_cand_.size(), (double)t.lend_heat_margin / 100.0, p1::LendHeat::kMinRows);
    return RAD_OK;
}

/* ---- the lent slots follow live decode routing (..._P1_LEND_HEAT; card_expert_stager.h) -------------------
 *
 * Layer li's counts are on the host (counts_host_, checked): each of its file-tier units, a lent resident
 * or not, takes the rows routed to its expert. Trunk layers only; the pass is a pure decode pass
 * (p1_heat_pass_). */
void FileStager::p1_heat_count(int32_t li) {
    const Layer& L = layers_[(size_t)li];
    if (L.layer < 0 || L.layer >= n_model_layers_) return;
    for (int32_t u : L.units) {
        const int32_t e = plan_->units[(size_t)u].expert;
        if (e >= 0 && e < 96 && counts_host_[e] > 0) p1_heat_->add(u, counts_host_[e]);
    }
}

bool FileStager::p1_heat_eligible(int32_t u) const {
    if (u < 0 || (size_t)u >= plan_->units.size()) return false;
    const MoveUnit& mu = plan_->units[(size_t)u];
    return mu.bytes == ds41p1::kSlotBytes && mu.weights.size() == 2 && mu.offset.size() == 2 &&
           !mover_->file_unit_in_slab(u) && mover_->host_planes_eligible(u) &&
           !(aux_view_ && aux_view_->is_resident(u));
}

/* Between passes, right behind the lender's begin_pass: up to S of the coldest residents give their slots
 * to hotter units off the card. Nothing happens (counted by reason, RingLender::Decline) when this pass
 * takes the slots back or they are lent or in the grace, when a draft bank's loan holds ring residents on
 * this rank (its targets are published the same way and are not the lender's), when any slot is not its
 * resident's (a reload wanted or in flight, or failed until the next return), when p1_lend_ev_ was not
 * recorded at the end of the pass right before this one (the fence would be behind nothing), or when a
 * chosen unit can no longer hold a slot (it leaves the candidates). The order within a retarget, and why
 * it is enough, is card_expert_stager.h's (the lent slots' heat) and RingLender::retarget's. */
void FileStager::p1_heat_policy() {
    /* The engine's parts of p1::heat_turn: the lent slots' units, the test of a unit, and the retarget, which
     * lays the units out as p1_lend_init does and hands the card path the event the DMA lane waits on. */
    struct Owner {
        FileStager* fs;
        int32_t slot_unit(int k) const { return fs->p1_->lend_unit(k); }
        bool eligible(int32_t u) const { return fs->p1_heat_eligible(u); }
        int retarget(const int* ks, const int32_t* units, int n) {
            p1::CardExpertPath::LendUnit nu[ds41p1::kMaxLendHeat];
            if (n < 1 || n > ds41p1::kMaxLendHeat) return 0;
            for (int i = 0; i < n; ++i) {
                const MoveUnit& mu = fs->plan_->units[(size_t)units[i]];
                nu[i].unit = units[i];
                for (int w = 0; w < 2; ++w) {
                    nu[i].off[w] = mu.offset[(size_t)w];
                    nu[i].bytes[w] = fs->ep_program_->weights[mu.weights[(size_t)w]].stored_bytes;
                }
            }
            return fs->p1_->lend_retarget(ks, nu, n, fs->p1_lend_ev_);
        }
    } owner{ this };
    /* The fence's event is p1_lend_ev_, and it must be the record of the pass right before this one
     * (end_pass, behind lane 1's join): behind every kernel that could read a resident from its slot.
     * p1_lent_ is the policy's `taken` and follows the slots (the aux expert worker's candidates read it). */
    const bool fence_ready = p1_lend_ev_ && p1_lend_rec_ && p1_lend_rec_pass_ + 1 == p1_lend_pass_;
    (void)p1::heat_turn(p1_->lender(), *p1_heat_, p1_lend_took_, bank_track_, fence_ready, p1_heat_cand_, p1_lent_,
                        owner);
}

/* The lent arena: the engine bound the arena's smallest level (steps of up to `rows` tokens) and hands
 * over the top above its buffers and guard; the lending starts with as many whole slots as fit there
 * (at most ..._P1_LEND_ARENA), the ring's (..._P1_LEND) beside them. */
int FileStager::p1_lend_arena(char* lo, int64_t bytes, int64_t rows, std::function<bool()> full, std::string* why) {
    if (!p1_lend_arena_asked() || p1_->lending() || !full) {
        if (why) *why = "the card path does not wait for the arena, or lends already";
        return RAD_E_STATE;
    }
    /* The CPU path keeps a layer's gate_up operand addresses from an earlier pass (hx_capture), and a
     * change of arena level moves them: only early issue's refetch takes the move, so without it (or
     * with the M1 trace, which forces the ordinary path) the first pass after each level change would
     * stop at its gate_up. */
    if (!hx_early_ || trace_requested_) {
        if (why)
            *why = fmt("the lent arena needs RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1 and no %s: an arena level change "
                       "moves the gate_up operands the CPU path fetched from", kM1PrefetchTraceEnv);
        return RAD_E_INVAL;
    }
    const std::vector<char*> slots = p1::arena_slots(lo, bytes, p1_tun_.lend_arena);
    if (slots.empty()) {
        if (why)
            *why = fmt("the arena's top above the %lld-token level (%.1f MiB) holds no slot of %lld bytes", (long long)rows,
                       (double)std::max<int64_t>(bytes, 0) / 1048576.0, (long long)ds41p1::kSlotBytes);
        return RAD_E_INVAL;
    }
    p1_arena_full_ = std::move(full);
    p1_arena_rows_ = rows;
    const int rc = p1_lend_init(p1_tun_, slots, why);
    if (rc < 0) p1_arena_full_ = nullptr;
    return rc;
}

/* Layer li of an armed pass, its counts in counts_host_ and its selected file-tier units in want_
 * (the rest in skip_): the slab's residents stay where they are; every other selected unit goes to
 * the ring (a RAM-cache hit borrowed, a miss read), published with kMarkSkip; the rest aliased as
 * always. The layer's next one's likely misses are read after this one's. */
int FileStager::p1_layer(int32_t li, RadStream s) {
    (void)s;
    const Layer& L = layers_[(size_t)li];
    const int b = li & 1;
    if (p1_li_ >= 0) {
        RAD_ERR("ds41 p1: routed layer %d began while layer %d's waves had not run",
                L.layer, layers_[(size_t)p1_li_].layer);
        return RAD_E_STATE;
    }
    for (int e = 0; e < 96 && e < counts_cap_; ++e) p1_touched_ += counts_host_[e] > 0;
    /* Scatter's offsets are the exclusive prefix of these same synchronized ALL-expert counts,
     * including the residents. The optional wave SwiGLU view needs no extra device readback. */
    RAD_TRY(p1_->begin_layer(li, b, counts_host_, counts_cap_));
    /* The admission: none in a pass whose rows stop short of the admission row; every unit when the row
     * is 0 (a prompt within the last N tokens, or ..._LCP_ADMIT=all); else by the sorted order
     * fetched with the counts (none without it). */
    p1_nonres_.clear();
    p1_ram_.clear();
    p1_units_ = 0;
    for (int32_t u : want_) {
        if (mover_->file_unit_in_slab(u)) { ++p1_slab_; continue; }
        const int32_t e = plan_->units[(size_t)u].expert;
        const int rows = e >= 0 && e < counts_cap_ ? counts_host_[e] : 0;
        const bool admit = p1_admits(li, e);
        bool hit = false;
        RAD_TRY(p1_->add_ram(e, u, rows, admit, &hit));
        if (!hit) {
            if (!mover_->host_planes_eligible(u)) {
                RAD_ERR("ds41 p1: routed layer %d's unit %d cannot be read as planes", L.layer, u);
                return RAD_E_STATE;
            }
            RAD_TRY(p1_->add_drive(e, u, rows, admit));
        }
        p1_nonres_.push_back(u);
        p1_ram_.push_back(hit ? 1 : 0);
        ++p1_units_;
    }
    /* The layer's selected units pinned in the host pool: copied from their pool slots as a RAM-cache
     * unit is, and published skipped like the others until the layer is released. */
    p1_pool_sel_.clear();
    if ((size_t)li < p1_pool_.size())
        for (int32_t u : p1_pool_[(size_t)li]) {
            const int32_t e = plan_->units[(size_t)u].expert;
            const int rows = e >= 0 && e < counts_cap_ ? counts_host_[e] : 0;
            if (rows <= 0) continue;
            p1::Source src;
            if (!p1_pool_source(u, &src)) {
                RAD_ERR("ds41 p1: routed layer %d's unit %d has no pool slot to copy from", L.layer, u);
                return RAD_E_STATE;
            }
            RAD_TRY(p1_->add_pool(e, u, rows, src));
            p1_pool_sel_.push_back(u);
            ++p1_units_;
        }
    staged_[b].clear();
    mover_->alias_file_units(skip_, buf_[b], &staged_[b]);
    mover_->alias_file_units(p1_nonres_, (char*)p1_->alias_base(), &staged_[b], ds41p1::kMarkSkip);
    mover_->alias_pool_units(p1_pool_sel_, (char*)p1_->alias_base(), &staged_[b], ds41p1::kMarkSkip);
    partial_[b] = true;
    pending_[b] = false;
    if (!staged_[b].empty()) holds_[b] = li;
    RAD_TRY(p1_->start());
    /* The next routed layer's likely misses: its file-tier units neither in the slab nor in the RAM
     * cache, by the sealed profile's share (best first). */
    const int32_t nx = li + 1;
    if (p1_->tunables().ahead > 0 && (size_t)nx < layers_.size() && will_issue(nx) && p1_layer_on(nx)) {
        p1_ahead_.clear();
        for (int32_t u : layers_[(size_t)nx].units)
            if (!mover_->file_unit_in_slab(u) && mover_->host_planes_eligible(u) && !mover_->unit_cached(u))
                p1_ahead_.push_back(u);
        std::stable_sort(p1_ahead_.begin(), p1_ahead_.end(), [&](int32_t a, int32_t c) {
            return plan_->units[(size_t)a].profile_share > plan_->units[(size_t)c].profile_share;
        });
        RAD_TRY(p1_->read_ahead(nx, p1_ahead_));
    }
    p1_li_ = li;
    ++p1_layers_;
    return RAD_OK;
}

/* The admission: none in a pass whose rows stop short of the admission row; every unit when the row is
 * 0 (a prompt within the last N tokens, ..._LCP_ADMIT=all, or a step that is not a lone chunk); else
 * by the sorted order fetched with the counts (hx_admits), and none without it. */
bool FileStager::p1_admits(int32_t li, int32_t e) const {
    if (lcp_admit_row_ <= 0) return true;
    if (lcp_admit_row_ >= pass_tok_ || !p1_order_ || !hx_idx_ || li < 0 || (size_t)li >= hx_ops_.size()) return false;
    const HxOps& o = hx_ops_[(size_t)li];
    if (e < 0 || e >= (int32_t)o.n_expert || o.top_k <= 0) return false;
    /* Expert e's slots are [off[e], off[e + 1]) (the n_expert + 1 offsets fetched with the counts; the
     * slots past off[n_expert] are other ranks'), each a token of the layer's rows times top_k. */
    const int64_t rows = layer_rows(li), T = rows * o.top_k, from = lcp_admit_row_ - layer_r0(li);
    const int32_t* const sorted = hx_idx_;
    const int32_t* const off = hx_idx_ + hx_sorted_max();
    const int64_t lo = std::clamp<int64_t>(off[e], 0, T), hi = std::clamp<int64_t>(off[e + 1], lo, T);
    for (int64_t i = lo; i < hi; ++i)
        if (sorted[i] >= 0 && sorted[i] / o.top_k >= from && sorted[i] / o.top_k < rows) return true;
    return false;
}

int FileStager::launched(rad_op h, RadStream, const RadArgs* a, int (*launch)(const RadArgs*, RadStream)) {
    /* The aux worker relaunches the rows it resolved for these ops (aux_resolve): a launch of another row
     * at a height it takes stops its jobs on this rank (stager_aux.cpp aux_guard). */
    if (aux_ && (size_t)h < aux_op_.size() && aux_op_[(size_t)h] >= 0) aux_guard((int32_t)h, a, launch);
    if (!p1_armed_ || p1_li_ < 0) return RAD_OK;
    const size_t li = (size_t)p1_li_;
    const int32_t op = (int32_t)h;
    const int which = op == hx_ops_[li].gu ? 0 : op == p1_ops_[li].sw ? 1 : op == p1_ops_[li].dn ? 2 : -1;
    if (which < 0) return RAD_OK;
    bool span_ok = false;
    if (which == 1 && p1_->tunables().swiglu_span && ep_program_ && (size_t)h < ep_program_->ops.size() &&
        a && a->n_t == 2 && a->t && ep_program_->ops[(size_t)h].ranged_key == "M") {
        for (const Band& b : ep_program_->ops[(size_t)h].bands) {
            if (a->t[1].shape[0] > b.hi) continue;
            const Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
            const RadKernelInfo* ki = r && r.row ? r.row->info : nullptr;
            span_ok = p1::swiglu_span_launch_ok(ki, launch);
            break;
        }
    }
    return p1_->capture(which, a, launch, span_ok);
}

int FileStager::after_op(rad_op h, RadStream s) {
    if (!p1_armed_ || p1_li_ < 0 || (int32_t)h != p1_ops_[(size_t)p1_li_].dn) return RAD_OK;
    const int32_t li = p1_li_;
    p1_li_ = -1;
    /* Behind the down projection, before any later op: the waves (card_expert_stager.h). */
    const int rc = p1_->run(s);
    if (rc < 0) return rc;
    if (p1_->tunables().verify > 0) return p1_verify(li, s);
    return RAD_OK;
}

int FileStager::p1_verify(int32_t li, RadStream s) {
    auto& c = p1_->counts_mut();
    const int N = p1_->tunables().verify;
    int na = 0, nd = 0;
    const RadTensor* ga = p1_->captured_operands(0, &na);
    const RadTensor* da = p1_->captured_operands(2, &nd);
    const HxOps& o = hx_ops_[(size_t)li];
    if (!ga || !da || na < 5 || nd < 5 || !hx_pool_ || o.top_k <= 0) return RAD_E_STATE;
    const RadTensor &A = ga[0], &so = ga[2], &of = ga[3], &Y = da[4];
    const int64_t rows = A.rank == 2 ? A.shape[0] : 0, T = rows * o.top_k;
    const int64_t x_ld = rows > 1 ? A.stride[0] : kHxEmbd, y_ld = Y.rank == 2 && Y.shape[0] > 1 ? Y.stride[0] : kHxEmbd;
    if (rows <= 0 || T > 6 * lcp_ || A.dtype != RAD_BF16 || Y.dtype != RAD_BF16 || !so.data || !of.data) return RAD_E_STATE;
    /* The order and the offsets, as the card's gate_up read them. */
    int32_t* const sorted = p1_vidx_;
    int32_t* const off = p1_vidx_ + 6 * lcp_;
    RAD_TRY(rad_memcpy_async(sorted, so.data, T * 4, s));
    RAD_TRY(rad_memcpy_async(off, of.data, (o.n_expert + 1) * 4, s));
    RAD_TRY(rad_event_record(p1_vev_, s));
    RAD_TRY(rad_event_sync(p1_vev_));
    /* The samples: expert-rows of the layer's RAM-cache units, evenly over them. */
    std::vector<std::pair<int32_t, int32_t>> cand;   /* (index in p1_nonres_, sorted slot) */
    for (size_t k = 0; k < p1_nonres_.size(); ++k) {
        if (!p1_ram_[k]) continue;
        const int32_t e = plan_->units[(size_t)p1_nonres_[k]].expert;
        if (e < 0 || e >= (int32_t)o.n_expert) continue;
        const int64_t lo = std::clamp<int64_t>(off[e], 0, T), hi = std::clamp<int64_t>(off[e + 1], lo, T);
        for (int64_t i = lo; i < hi; ++i)
            if (sorted[i] >= 0 && sorted[i] / o.top_k < rows) cand.push_back({ (int32_t)k, (int32_t)i });
    }
    if (cand.empty()) return RAD_OK;
    const size_t step = std::max<size_t>(1, cand.size() / (size_t)N);
    std::vector<std::pair<int32_t, int32_t>> pick;
    for (size_t i = 0; i < cand.size() && (int)pick.size() < N; i += step) pick.push_back(cand[i]);
    for (size_t k = 0; k < pick.size(); ++k) {
        const int64_t i = pick[k].second, t = sorted[i] / o.top_k;
        RAD_TRY(rad_memcpy_async(p1_vx_ + k * kHxEmbd, (const char*)A.data + t * x_ld * 2, kHxEmbd * 2, s));
        RAD_TRY(rad_memcpy_async(p1_vy_ + k * kHxEmbd, (const char*)Y.data + i * y_ld * 2, kHxEmbd * 2, s));
    }
    RAD_TRY(rad_event_record(p1_vev_, s));
    RAD_TRY(rad_event_sync(p1_vev_));
    /* The same rows on the CPU from the cache's bytes, borrowed again (read-only). */
    std::vector<Mover::HostUnit> held(pick.size());
    std::vector<HostExpertPool::Job> jobs;
    std::vector<size_t> job_of;
    int64_t cost = 0;
    for (size_t k = 0; k < pick.size(); ++k) {
        Mover::HostUnit& hu = held[k];
        if (!mover_->borrow_host_unit(p1_nonres_[(size_t)pick[k].first], &hu, false) || hu.n != 2) {
            mover_->release_host_unit(&hu);
            ++c.verify_skipped;   /* evicted since its copy */
            continue;
        }
        HostExpertPool::Job j;
        const int g = hu.bytes[0] == HostExpertPool::kGuBytes ? 0 : 1;
        j.gu = hu.host[g];
        j.dn = hu.host[1 - g];
        if (hu.packed) {
            j.packed = true;
            j.gu_bytes = hu.slot[g];
            j.dn_bytes = hu.slot[1 - g];
        }
        j.x = p1_vx_ + k * kHxEmbd;
        j.y = p1_vh_ + k * kHxEmbd;
        j.cost = HostExpertPool::job_cost(1);
        cost += j.cost;
        jobs.push_back(j);
        job_of.push_back(k);
    }
    if (!jobs.empty()) {
        HostExpertPool::Ticket tk;
        hx_pool_->reserve(cost);
        const int rc = hx_pool_->submit(jobs.data(), (int)jobs.size(), &tk);
        if (rc < 0) {
            hx_pool_->unreserve(cost);
            for (Mover::HostUnit& hu : held) mover_->release_host_unit(&hu);
            return rc;
        }
        hx_pool_->wait(&tk);
    }
    for (Mover::HostUnit& hu : held) mover_->release_host_unit(&hu);
    for (size_t k : job_of) {
        const uint16_t* a = p1_vh_ + k * kHxEmbd;
        const uint16_t* b = p1_vy_ + k * kHxEmbd;
        int bad = 0, first = -1;
        for (int i = 0; i < kHxEmbd; ++i)
            if (a[i] != b[i]) { ++bad; if (first < 0) first = i; }
        ++c.verified;
        if (!bad) continue;
        if (++c.differ <= 8)
            RAD_WARN("ds41 p1 verify: layer %d sorted row %d (unit %d): %d of %d values differ from the CPU's "
                     "(first at %d: 0x%04x on the CPU, 0x%04x on the card)", layers_[(size_t)li].layer,
                     pick[k].second, p1_nonres_[(size_t)pick[k].first], bad, (int)kHxEmbd, first, a[first], b[first]);
    }
    return RAD_OK;
}

void FileStager::p1_lend_audit_final() {
    if (!p1_lend_audits()) return;
    p1_->lend_audit_behind(p1_lend_rec_ ? p1_lend_ev_ : nullptr);
    (void)p1_->lender().audit_residents();
}

std::string FileStager::p1_stats_line() const {
    if (!p1_) return std::string();
    std::string line = p1_->stats_line(plan_ ? plan_->rank : 0);
    const auto& sc = p1_->sched();
    const uint64_t ring = sc.ram + sc.drive + sc.pool;
    line += fmt("; touched experts %llu: VRAM residents %llu, slab %llu, ring %llu; admission orders moved %llu",
                (unsigned long long)p1_touched_,
                (unsigned long long)(p1_touched_ >= ring + p1_slab_ ? p1_touched_ - ring - p1_slab_ : 0),
                (unsigned long long)p1_slab_, (unsigned long long)ring, (unsigned long long)p1_order_moved_);
    /* ..._P1_LEND_HEAT: the half-life the policy ran with and the decode rows it counted (it acts only once
     * a half-life of them is in). */
    if (p1_heat_)
        line += fmt("; heat half-life %lld decode rows, %lld counted", (long long)p1_heat_->half_rows(),
                    (long long)p1_heat_->seen_rows());
    return line;
}

}  /* namespace rad */

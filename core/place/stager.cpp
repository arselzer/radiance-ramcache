/* stager.cpp -- see stager.h. */
#include "place/stager.h"
#include "place/dspark_native_scope.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
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
    armed_ = want_ != nullptr && enabled() && b->n_tok >= min_rows_;
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
    //Engine normally closes this before Ctx. Partial-init destruction has the same obligation.
    if (shutdown_head_cache() < 0) std::_Exit(88);
    for (int b = 0; b < 2; ++b) {
        if (done_[b]) rad_event_destroy(done_[b]);
        if (free_[b]) rad_event_destroy(free_[b]);
    }
    if (counts_ev_) rad_event_destroy(counts_ev_);
    if (counts_host_) rad_dev_free(counts_host_, RAD_MEM_HOST_PINNED);
    hx_drain();
    if (hx_bg_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(hx_bg_mu_);
            hx_bg_stop_ = true;
        }
        hx_bg_cv_.notify_all();
        hx_bg_.join();
    }
    if (trace_fp_) std::fclose(trace_fp_);
    if (hx_out_rec_ && hx_out_ev_) (void)rad_event_sync(hx_out_ev_);
    if (hx_in_ev_) rad_event_destroy(hx_in_ev_);
    if (hx_out_ev_) rad_event_destroy(hx_out_ev_);
    for (void* p : { (void*)hx_x_, (void*)hx_idx_, (void*)hx_out_, (void*)hx_chk_ })
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
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
    RAD_TRY(init_head_cache(prog, plan));
    return hx_init(prog);
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
    ++trace_pass_;
    routed_ = counts_host_ && b->n_tok > 0 && b->n_tok <= kRoutedRows;
    return true;
}

bool FileStager::will_issue(int32_t li) const {
    const auto it = pass_last_.find(pass_kind_);
    return it == pass_last_.end() || layers_[(size_t)li].first_op <= it->second;
}

int FileStager::start(int32_t li) {
    //EVERY full-layer/read-ahead/repeated-slice path acquires the complete buffer here.
    RAD_TRY(head_revoke());
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
    if (!ep_counts_ || (size_t)li >= ep_layer_eligible_.size() || !ep_layer_eligible_[li])
        return RAD_OK;
    const Layer& L = layers_[(size_t)li];
    EpRouteCounts v;
    if (!ep_counts_(L.layer, &v)) return RAD_OK;
    if (!ep_route_current(v, ep_program_, s, v.generation, L.layer, pass_tok_) ||
        v.rank != plan_->rank) { RAD_TRY(head_revoke()); return RAD_E_STATE; }
    const int b = li & 1;
    if (holds_[b] >= 0 || counts_cap_ < 96) return RAD_E_STATE;
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
    /* Same actual producer stream; this host-wait event is the completion proof.
     * The Ctx witness means successfully issued, never completed before this wait. */
    RAD_TRY(rad_memcpy_async(counts_host_, v.row, 96 * (int64_t)sizeof(int32_t), s));
    if (hx) RAD_TRY(hx_fetch(li, s));
    RAD_TRY(rad_event_record(counts_ev_, s));
    const auto t_sync = std::chrono::steady_clock::now();
    RAD_TRY(rad_event_sync(counts_ev_));
    const auto t_counts = std::chrono::steady_clock::now();
    const uint64_t sync_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_counts - t_sync).count();
    ep_stats_.sync_ns += sync_ns;
    if (pass_tok_ == 1) { ++ep_stats_.d_reads; ep_stats_.d_sync_ns += sync_ns; }
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
    /* The selected units the CPU computes leave want_ and start on the pool before the drive
     * reads below block this thread (host_expert_pool.h). */
    if (hx) {
        hx_choose(li);
        if (hx_li_ == li) RAD_TRY(hx_submit());
    }
    const auto t_stage = std::chrono::steady_clock::now();
    ep_stats_.hx_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t_stage - t_counts).count();
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
        const uint64_t missed = mover_->stats().file_relaid - relaid0 +
            (hx_li_ == li && hx_policy_ != kHxVerify ? 2 * (uint64_t)hx_miss_.size() : 0);
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
}  /* namespace */

/* Which layers can hand units to the pool: the EP-routed ones whose first expert op is a decode
 * gate_up (moe_gemm_ggml mxfp4 [4608, 5120], token order), whose last is the down projection
 * ([5120, 2304], sorted order) with a swiglu_clamped at the pool's limit between, and whose
 * gathers (moe_gather, or ar_gather_hc_write folding it in, up to the next routed layer) read the
 * down projection's rows as operand 0. */
int FileStager::hx_init(const Program& prog) {
    hx_drain();
    hx_pool_ = nullptr;
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
        /* Every reader of the down rows up to the next routed layer must be one of these. */
        const int32_t end = li + 1 < layers_.size() ? layers_[li + 1].first_op
                                                    : (int32_t)prog.ops.size();
        int n_gather = 0;
        bool shape = true;
        for (int32_t h = L.last_op + 1; h < end && shape; ++h) {
            const OpInfo& op = prog.ops[(size_t)h];
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
    const int64_t rows = (int64_t)kHxRows * kHxEmbd * (int64_t)sizeof(uint16_t);
    if (!hx_x_) hx_x_ = (uint16_t*)rad_dev_alloc(kEpRouteMaxRows * kHxEmbd * 2, RAD_MEM_HOST_PINNED);
    if (!hx_idx_) hx_idx_ = (int32_t*)rad_dev_alloc((kHxRows + 128) * 4, RAD_MEM_HOST_PINNED);
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
        hx_miss_rows_ = std::clamp(std::atoi(m), 0, (int)kEpRouteMaxRows);
    if (hx_miss_rows_ > 0 && !mover_->mapped_cache_on()) {
        if (plan_->rank == 0)
            RAD_WARN("host experts: ..._MISSES needs the mapped RAM cache "
                     "(RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1); misses stay on the copy link");
        hx_miss_rows_ = 0;
    }
    if (hx_miss_rows_ > 0 && !hx_bg_.joinable()) {
        hx_bg_stop_ = false;
        hx_bg_ = std::thread(&FileStager::hx_bg_loop, this);
        if (plan_->rank == 0)
            RAD_INFO("host experts: passes of at most %d token(s) compute missed experts on the CPU "
                     "from the container's planes; a background thread relays them into the RAM cache",
                     hx_miss_rows_);
    }
    if (plan_->rank == 0)
        RAD_INFO("host experts: %zu of %zu routed layers give the CPU %s cached experts%s", n_ok,
                 layers_.size(), hx_policy_ == kHxAll ? "every one of their" : "some of their",
                 hx_policy_ == kHxVerify ? " and still compute them on the card, comparing the rows"
                 : hx_policy_ == kHxGreedy ? " (whichever of the pool and the copy link frees sooner)"
                                           : "");
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
    hx_slot_.assign(want_.size(), -1);
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
            hx_miss_.push_back(hx_planes_get(u));
            const int rows = e >= 0 && e < counts_cap_ ? std::max<int32_t>(counts_host_[e], 1) : 1;
            hx_pool_->reserve(HostExpertPool::cost_rows(rows));
            hx_reserved_ += HostExpertPool::cost_rows(rows);
            continue;
        }
        link += plan_->units[(size_t)u].bytes;
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
    if (pass_tok_ < 1 || pass_tok_ > kEpRouteMaxRows) return RAD_E_STATE;
    if (o.x_ld == kHxEmbd) {
        RAD_TRY(rad_memcpy_async(hx_x_, o.x, pass_tok_ * kHxEmbd * 2, s));
    } else {
        for (int64_t r = 0; r < pass_tok_; ++r)
            RAD_TRY(rad_memcpy_async(hx_x_ + r * kHxEmbd, (const char*)o.x + r * o.x_ld * 2,
                                     kHxEmbd * 2, s));
    }
    RAD_TRY(rad_memcpy_async(hx_idx_, o.sorted, pass_tok_ * o.top_k * 4, s));
    RAD_TRY(rad_memcpy_async(hx_idx_ + kHxRows, o.off, (o.n_expert + 1) * 4, s));
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
    const int32_t* const off = hx_idx_ + kHxRows;
    const int ne = (int)o.n_expert;
    if (e < 0 || e >= ne) return RAD_E_STATE;
    int lo = e == 0 ? 0 : off[e], hi = e == ne - 1 ? (int)T : off[e + 1];
    lo = std::clamp(lo, 0, (int)T);
    hi = std::clamp(hi, lo, (int)T);
    int32_t n = 0;
    HostExpertPool::Job j = proto;
    int in_job = 0;
    auto flush = [&] {
        if (!in_job) return;
        j.more = in_job - 1;
        j.cost = HostExpertPool::job_cost(in_job);
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
        if (++in_job == HostExpertPool::kMaxRows) flush();
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
        proto.gu = hu.bytes[0] == HostExpertPool::kGuBytes ? hu.host[0] : hu.host[1];
        proto.dn = hu.bytes[0] == HostExpertPool::kGuBytes ? hu.host[1] : hu.host[0];
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
    std::vector<Mover::HostPlanes*> ptrs;
    for (auto& p : hx_miss_) ptrs.push_back(p.get());
    const auto t0 = std::chrono::steady_clock::now();
    if (!mover_->read_unit_planes(ptrs.data(), (int)ptrs.size())) {
        RAD_ERR("host experts: reading layer %d's missed units for the CPU failed",
                layers_[(size_t)hx_li_].layer);
        return RAD_E_IO;
    }
    hx_stats_.miss_read_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t0).count();
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
    {
        std::lock_guard<std::mutex> lk(hx_bg_mu_);
        for (auto& p : hx_miss_) hx_bg_q_.push_back(p);
    }
    hx_bg_cv_.notify_one();
    return RAD_OK;
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

void FileStager::hx_bg_loop() {
    std::vector<uint8_t> scratch;
    bool told = false;
    std::unique_lock<std::mutex> lk(hx_bg_mu_);
    for (;;) {
        hx_bg_cv_.wait(lk, [&] { return hx_bg_stop_ || !hx_bg_q_.empty(); });
        if (hx_bg_q_.empty()) return;   /* stopping, nothing left */
        std::shared_ptr<Mover::HostPlanes> p = std::move(hx_bg_q_.front());
        hx_bg_q_.pop_front();
        hx_bg_busy_ = true;
        lk.unlock();
        const int rc = mover_->insert_unit_planes(*p, &scratch);
        if (rc < 0 && !told) {
            RAD_WARN("host experts: relaying a missed unit into the RAM cache failed (%s); it stays "
                     "a miss", rad_strerror(rc));
            told = true;
        }
        p.reset();
        lk.lock();
        hx_bg_busy_ = false;
        hx_bg_cv_.notify_all();
    }
}

void FileStager::drain_host_inserts() {
    std::unique_lock<std::mutex> lk(hx_bg_mu_);
    hx_bg_cv_.wait(lk, [&] { return hx_bg_q_.empty() && !hx_bg_busy_; });
}

/* A layer's gate_up op, before its launch, on every pass: the addresses of its token rows,
 * sorted order and expert offsets (moe_gemm_ggml: a, w, sorted_tok, expert_offset, y), which
 * the next pass's hx_fetch reads. With CPU rows in flight they must be the ones hx_fetch read on
 * this pass, or the pool computed from the wrong bytes. */
int FileStager::hx_capture(int32_t li, const RadTensor* t, int n_t) {
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
    /* Only the addresses of a routed pass that reads all its rows at once (no slice, so from row
     * 0) are kept: what hx_fetch reads from on such a pass. */
    if (shape && routed_ && AR == pass_tok_ && AR >= 1 && AR <= kEpRouteMaxRows &&
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
    const auto t0 = std::chrono::steady_clock::now();
    hx_pool_->wait(&hx_ticket_);
    hx_stats_.wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - t0).count();
    for (int k = 0; k < hx_n_; ++k) mover_->release_host_unit(&hx_units_[k]);
    hx_n_ = 0;
    hx_miss_.clear();
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
    if (hx_stage_ == 2 && hx_pool_) hx_pool_->wait(&hx_ticket_);
    if (hx_reserved_ && hx_pool_) hx_pool_->unreserve(hx_reserved_);
    hx_reserved_ = 0;
    if (hx_units_)
        for (int k = 0; k < hx_n_; ++k) mover_->release_host_unit(&hx_units_[k]);
    hx_n_ = 0;
    hx_miss_.clear();
    hx_li_ = -1;
    hx_stage_ = 0;
    hx_confirmed_ = false;
}

int FileStager::operands(rad_op h, RadStream s, const RadTensor* t, int n_t) {
    if (!hx_pool_) return RAD_OK;
    if ((size_t)h < first_of_.size() && first_of_[(size_t)h] >= 0) {
        const int32_t li = first_of_[(size_t)h];
        if ((size_t)li < hx_ops_.size() && hx_ops_[(size_t)li].gu == (int32_t)h)
            return hx_capture(li, t, n_t);
        return RAD_OK;
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
        /* CPU rows of the slice before not yet copied: a slice's gate_up ahead of its gather. */
        if (hx_li_ == li && hx_stage_ == 2) {
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
        const bool ep = ep_counts_ && pass_tok_ > 0 && pass_tok_ <= kEpRouteMaxRows &&
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
    const bool lost = hx_li_ >= 0 && hx_stage_ == 2;
    if (lost) RAD_ERR("host experts: layer %d's CPU rows reached no gather before the pass ended",
                      layers_[(size_t)hx_li_].layer);
    hx_drain();
    pass_last_[pass_kind_] = last_issued_;
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

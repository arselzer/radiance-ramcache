/* stager_aux.cpp -- the file stager's part of the aux expert worker (aux_expert_worker.h): the unit set's
 * candidates, the rows a job relaunches, the load, the warm-up, and the per-layer hooks hx_choose /
 * hx_submit / hx_finish / hx_drain call (stager.h, "the aux expert worker's per-layer state"). With no
 * binding (aux_ null) none of this runs. */
#include "place/stager.h"
#include "ds41_host_experts.h"
#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace rad {

namespace {

constexpr int64_t kHxEmbd = HostExpertPool::kEmbd;   /* as stager.cpp's */

int64_t aux_now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* aux_op_name(int k) { return k == 0 ? "gate_up" : k == 1 ? "SwiGLU" : "down"; }

aux::Launch launch_of(const Resolved* r) {
    aux::Launch l;
    if (!r || !r->row || !r->row->info) return l;
    l.fn = r->row->info->launch;
    l.p = r->geom.params();
    l.n_p = r->geom.n_params();
    l.instance = r->instance;
    return l;
}

}  // namespace

/* ---- the unit set (work/aux/SPEC.md §5.2) -------------------------------------------------------- */

int FileStager::aux_candidates(std::vector<ds41aux::Cand>* out) const {
    out->clear();
    if (!plan_ || !mover_ || !ep_program_) return RAD_E_STATE;
    for (size_t li = 0; li < layers_.size(); ++li) {
        const bool layer_ok = li < ep_layer_eligible_.size() && ep_layer_eligible_[li] && li < hx_ops_.size() &&
                              hx_ops_[li].gu >= 0 && li < aux_rows_.size() && aux_rows_[li].ok;
        const bool trunk = layers_[li].layer >= 0 && layers_[li].layer < n_model_layers_;
        if (!layer_ok || !trunk) continue;
        for (int32_t u : layers_[li].units) {
            const MoveUnit& mu = plan_->units[(size_t)u];
            ds41aux::UnitFacts f;
            f.layer_ok = layer_ok;
            f.trunk = trunk;
            f.planes_eligible = mover_->host_planes_eligible(u);
            f.in_slab = mover_->file_unit_in_slab(u);
            f.lent = p1_lent(u);
            f.refused = mover_->packed_refused(u);
            f.n_weights = (int)mu.weights.size();
            for (size_t k = 0; k < mu.weights.size() && k < 2; ++k) f.bytes[k] = ep_program_->weights[mu.weights[k]].stored_bytes;
            f.expert = mu.expert;
            if (!ds41aux::candidate(f)) continue;
            ds41aux::Cand c;
            c.rank = plan_->rank;
            c.unit = u;
            c.share = mu.profile_share;
            c.expert = mu.expert;
            c.layer = mu.layer;
            out->push_back(c);
        }
    }
    return RAD_OK;
}

/* ---- the rows a job relaunches -------------------------------------------------------------------- */

int FileStager::aux_resolve(std::string* why) {
    aux_rows_.assign(layers_.size(), AuxRows{});
    aux_op_.assign(ep_program_ ? ep_program_->ops.size() : 0, -1);
    if (!ep_program_ || !plan_) {
        *why = "the stager has no program";
        return RAD_E_STATE;
    }
    int usable = 0;
    std::string first_bad;
    for (size_t li = 0; li < layers_.size(); ++li) {
        if (li >= ep_layer_eligible_.size() || !ep_layer_eligible_[li] || li >= hx_ops_.size() || hx_ops_[li].gu < 0)
            continue;
        if (layers_[li].layer < 0 || layers_[li].layer >= n_model_layers_) continue;
        const int32_t ops[3] = { hx_ops_[li].gu, hx_ops_[li].sw, layers_[li].last_op };
        AuxRows& R = aux_rows_[li];
        std::string bad;
        for (int k = 0; k < 3 && bad.empty(); ++k) {
            if (ops[k] < 0 || (size_t)ops[k] >= ep_program_->ops.size()) {
                bad = fmt("no %s op", aux_op_name(k));
                break;
            }
            const OpInfo& op = ep_program_->ops[(size_t)ops[k]];
            /* moe_gemm_ggml issues at M = the layer's rows, swiglu_clamped at M = its y operand's rows. */
            if (op.ranged_key != "M") {
                bad = fmt("its %s (%s) is not ranged by M", aux_op_name(k), op.op.c_str());
                break;
            }
            const int hi_m = k == 1 ? ds41aux::kMaxSlots : ds41aux::kMaxRows;
            if (op.range_lo > 1 || op.range_hi < hi_m) {
                bad = fmt("its %s issues at M in [%lld, %lld], not every height up to %d", aux_op_name(k),
                          (long long)op.range_lo, (long long)op.range_hi, hi_m);
                break;
            }
            const char* want = k == 1 ? "swiglu_clamped_bf16" : ds41p1::kGemvVpRow;
            std::vector<const Resolved*>& v = k == 0 ? R.gu : k == 1 ? R.sw : R.dn;
            v.assign((size_t)hi_m + 1, nullptr);
            for (int M = 1; M <= hi_m && bad.empty(); ++M) {
                const Band* b = nullptr;
                for (const Band& x : op.bands)
                    if (M <= x.hi) { b = &x; break; }
                const Resolved* r = b ? &b->dom[RAD_DOMAIN_DEVICE] : nullptr;
                const char* name = r && r->row && r->row->info ? r->row->info->name : nullptr;
                /* Only gemv_vp decodes the packed form every table entry of the worker names (T2 of ds41_p1.h:
                 * the plain gemv would read a packed slot as block_mxfp4). */
                if (!name || std::strcmp(name, want) || !r->row->info->launch)
                    bad = fmt("its %s's row at M %d is %s, not %s", aux_op_name(k), M, name ? name : "none", want);
                else if (r->scratch_bytes > 0 || r->instance)
                    bad = fmt("its %s's row %s takes scratch or an instance", aux_op_name(k), name);
                else
                    v[(size_t)M] = r;
            }
            if (!bad.empty() || k == 1) continue;
            /* The weight-table operand as the issue path builds it (Ctx::weight_table_tensor over entry 0's
             * template, Ctx::bind_weights): the declared shape behind one axis of n entries at stride 0. */
            rad_weight w0 = 0;
            int64_t n = 0;
            for (size_t i = 0; i < op.weights.size() && i < op.weight_opd.size(); ++i)
                if (op.weight_opd[i] == 1) {
                    if (!n) w0 = op.weights[i];
                    ++n;
                }
            if (n != ds41aux::kExperts || !w0 || (size_t)w0 >= ep_program_->weights.size()) {
                bad = fmt("its %s reads %lld table entries, not %d", aux_op_name(k), (long long)n, ds41aux::kExperts);
                break;
            }
            const WeightInfo& wi = ep_program_->weights[w0];
            RadTensor e{};
            e.dtype = wi.stored_dtype != RAD_DT_INVALID ? wi.stored_dtype : wi.decl.dtype;
            e.rank = wi.decl.rank;
            for (uint32_t d = 0; d < wi.decl.rank && d < RAD_MAX_RANK; ++d) e.shape[d] = wi.decl.shape[d];
            rad_tensor_pack(&e);
            if (e.rank + 1 > RAD_MAX_RANK) {
                bad = fmt("its %s's weights have rank %u", aux_op_name(k), e.rank);
                break;
            }
            RadTensor t = e;
            for (int d = (int)e.rank; d >= 1; --d) {
                t.shape[d] = e.shape[d - 1];
                t.stride[d] = e.stride[d - 1];
            }
            t.rank = e.rank + 1;
            t.shape[0] = n;
            t.stride[0] = 0;
            t.data = nullptr;
            (k == 0 ? R.gu_w : R.dn_w) = t;
        }
        if (!bad.empty()) {
            if (first_bad.empty()) first_bad = fmt("routed layer %d: %s", layers_[li].layer, bad.c_str());
            R = AuxRows{};
            continue;
        }
        R.ok = true;
        for (int k = 0; k < 3; ++k) aux_op_[(size_t)ops[k]] = (int32_t)(li * 3 + (size_t)k);
        ++usable;
    }
    if (!first_bad.empty() && plan_->rank == 0)
        RAD_WARN("aux experts: rank 0: %s; the worker takes no job of such a layer", first_bad.c_str());
    if (!usable) {
        *why = fmt("no routed layer's rows are the decode band's (%s): %s", ds41p1::kGemvVpRow,
                   first_bad.empty() ? "no eligible routed layer" : first_bad.c_str());
        return RAD_E_STATE;
    }
    return RAD_OK;
}

/* ---- loading (work/aux/SPEC.md §6) ---------------------------------------------------------------- */

int FileStager::aux_load(aux::Worker* w, int r, const std::vector<int32_t>& list, int64_t first, int64_t quota,
                         AuxLoadStats* st, std::string* why) {
    *st = AuxLoadStats{};
    aux::DeviceLane* lane = w->device_lane(r);
    if (!lane || !plan_ || !mover_) {
        *why = fmt("rank %d has no lane", r);
        return RAD_E_STATE;
    }
    const ds41aux::Layout& L = w->layout();
    aux::RankView& v = w->view(r);
    v.resident.assign(plan_->units.size(), 0);
    v.slot.assign(plan_->units.size(), -1);
    v.gu_tab_entries.assign(layers_.size() * ds41aux::kExperts, nullptr);
    v.dn_tab_entries.assign(layers_.size() * ds41aux::kExperts, nullptr);
    v.units = 0;
    v.share = 0;
    std::vector<int32_t> li_of(plan_->units.size(), -1);
    for (size_t li = 0; li < layers_.size(); ++li)
        for (int32_t u : layers_[li].units) li_of[(size_t)u] = (int32_t)li;
    /* The bounce and read-back buffers: pinned, mapped on the extra card only (the RAM cache's slabs are
     * not mapped there). A unit not cached is packed in a host scratch of a whole ring slot first
     * (pack_unit_for_ring's contract: gate_up at 0, down at ds41p1::kDnOff). */
    char* bounce = (char*)dev_alloc_pinned_on(L.stride, { w->device() });
    char* back = (char*)dev_alloc_pinned_on(L.stride, { w->device() });
    std::vector<uint8_t> scratch;
    /* Copies queued and not seen complete (a wait that timed out or failed): the queue may still read bounce
     * and write back, so both are left to the process exit, and the lane stays not drained (sync_bounded), so
     * the arena it writes stays too (work/aux/SPEC.md §7.5). */
    bool in_flight = false;
    auto done = [&](int rc) {
        if (!in_flight) {
            if (bounce) rad_dev_free(bounce, RAD_MEM_HOST_PINNED);
            if (back) rad_dev_free(back, RAD_MEM_HOST_PINNED);
        }
        return rc;
    };
    if (!bounce || !back) {
        *why = fmt("rank %d's pinned bounce buffers (device %d only): %s", r, w->device(), rad_dev_last_error());
        return done(RAD_E_NOMEM);
    }
    const RadStream s = lane->stream();
    int64_t k = 0;
    for (int32_t u : list) {
        if (k >= quota) break;
        const MoveUnit& mu = plan_->units[(size_t)u];
        const int32_t li = (size_t)u < li_of.size() ? li_of[(size_t)u] : -1;
        const int e = mu.expert;
        if (li < 0 || e < 0 || e >= ds41aux::kExperts) {
            ++st->skipped;
            continue;
        }
        Mover::HostUnit hu;
        const uint8_t* src[2] = { nullptr, nullptr };
        int64_t n[2] = { 0, 0 };
        const bool cached = mover_->borrow_host_unit(u, &hu, /*touch=*/false);
        if (cached) {
            const int g = hu.bytes[0] == HostExpertPool::kGuBytes ? 0 : 1;
            if (!hu.packed || hu.n != 2 || hu.bytes[g] != HostExpertPool::kGuBytes ||
                hu.bytes[1 - g] != HostExpertPool::kDnBytes) {
                mover_->release_host_unit(&hu);
                ++st->skipped;
                continue;
            }
            src[0] = hu.host[g];
            n[0] = hu.slot[g];
            src[1] = hu.host[1 - g];
            n[1] = hu.slot[1 - g];
        } else {
            Mover::HostPlanes p;
            p.unit = u;
            Mover::HostPlanes* pp = &p;
            bool packed = false;
            int64_t bytes[2] = { 0, 0 };
            if (scratch.empty()) scratch.resize((size_t)ds41p1::kSlotBytes);
            if (!mover_->read_unit_planes(&pp, 1) ||
                mover_->pack_unit_for_ring(p, scratch.data(), (int64_t)scratch.size(), &packed, bytes) < 0 || !packed) {
                ++st->skipped;   /* over the packed form's budget, or unreadable: the next candidate */
                continue;
            }
            src[0] = scratch.data();
            n[0] = bytes[0];
            src[1] = scratch.data() + ds41p1::kDnOff;
            n[1] = bytes[1];
        }
        ds41hx::Mx view;
        if (n[0] > L.gu_slot || n[1] > L.dn_slot ||
            !ds41hx::packed_view(src[0], (size_t)n[0], ds41aux::kGuN, ds41aux::kEmbd, &view) ||
            !ds41hx::packed_view(src[1], (size_t)n[1], ds41aux::kEmbd, ds41aux::kFf, &view)) {
            if (cached) mover_->release_host_unit(&hu);
            ++st->skipped;
            continue;
        }
        char* const slot = w->slot(first + k);
        std::memcpy(bounce, src[0], (size_t)n[0]);
        std::memcpy(bounce + L.dn_off, src[1], (size_t)n[1]);
        struct Copy { void* dst; const void* src; int64_t n; };
        const Copy copies[4] = { { slot, bounce, n[0] }, { slot + L.dn_off, bounce + L.dn_off, n[1] },
                                 { back, slot, n[0] }, { back + L.dn_off, slot + L.dn_off, n[1] } };
        int queued = 0, rc = RAD_OK;
        for (const Copy& c : copies) {
            if ((rc = rad_memcpy_async(c.dst, c.src, c.n, s)) < 0) break;
            ++queued;
        }
        const std::string qerr = rc < 0 ? rad_dev_last_error() : "";
        /* A bounded wait for whatever was queued: the four copies, or those before one that was refused. */
        const int synced = queued ? lane->sync_bounded(w->load_wait_ns()) : 1;
        if (rc < 0 || synced != 1) {
            in_flight = synced != 1;
            if (cached) mover_->release_host_unit(&hu);   /* the copies read bounce, never the lease */
            *why = fmt("rank %d unit %d: the copies to and from device %d %s%s", r, u, w->device(),
                       rc < 0 ? fmt("could not be queued (%s)", qerr.c_str()).c_str()
                       : synced == 0 ? fmt("did not complete in %g s", (double)w->load_wait_ns() * 1e-9).c_str()
                                     : "failed",
                       in_flight ? " (its pinned bounce and read-back buffers are left to the process exit)" : "");
            return done(RAD_E_DEVICE);
        }
        for (int i = 0; i < 2; ++i) {
            const char* got = back + (i ? L.dn_off : 0);
            if (std::memcmp(got, src[i], (size_t)n[i]) == 0) continue;
            int64_t at = 0;
            while (at < n[i] && (uint8_t)got[at] == src[i][at]) ++at;
            if (cached) mover_->release_host_unit(&hu);
            *why = fmt("rank %d unit %d (layer %d expert %d): its %s read back from device %d differs at byte %lld "
                       "(0x%02x, written 0x%02x)", r, u, mu.layer, e, i ? "down" : "gate_up", w->device(),
                       (long long)at, (unsigned)(uint8_t)got[at], (unsigned)src[i][at]);
            return done(RAD_E_STATE);
        }
        if (cached) mover_->release_host_unit(&hu);
        v.resident[(size_t)u] = 1;
        v.slot[(size_t)u] = (int32_t)(first + k);
        const size_t at = (size_t)li * ds41aux::kExperts + (size_t)e;
        v.gu_tab_entries[at] = ds41p1::mark(slot, ds41p1::kMarkPacked);
        v.dn_tab_entries[at] = ds41p1::mark(slot + L.dn_off, ds41p1::kMarkPacked);
        ++v.units;
        v.share += mu.profile_share;
        ++st->loaded;
        ++(cached ? st->cached : st->packed);
        st->share += mu.profile_share;
        ++k;
    }
    return done(RAD_OK);
}

/* ---- the warm-up job: libr4d's code object loaded on the extra card, and its rows the CPU pool's --- */

int FileStager::aux_warm(aux::Worker* w, int r, std::string* why) {
    aux::Lane* lane = w->lane(r);
    const aux::RankView& v = w->view(r);
    if (!lane || !hx_pool_ || !plan_) {
        *why = fmt("rank %d has no lane or pool", r);
        return RAD_E_STATE;
    }
    /* The first resident unit (layer order) the RAM cache holds now: the CPU computes it from its lease. */
    Mover::HostUnit hu;
    int32_t li = -1, u = -1;
    for (size_t l = 0; l < layers_.size() && u < 0; ++l) {
        if (l >= aux_rows_.size() || !aux_rows_[l].ok) continue;
        for (int32_t x : layers_[l].units) {
            if (!v.is_resident(x) || !mover_->borrow_host_unit(x, &hu, /*touch=*/false)) continue;
            if (hu.n == 2 && hu.packed) {
                li = (int32_t)l;
                u = x;
                break;
            }
            mover_->release_host_unit(&hu);
        }
    }
    if (u < 0) return RAD_OK;   /* nothing resident and cached on this rank: its lane takes no job anyway */
    const int e = plan_->units[(size_t)u].expert;
    std::vector<uint16_t> x((size_t)ds41aux::kEmbd), cpu((size_t)ds41aux::kEmbd);
    uint32_t s = 0x9e3779b9u ^ (uint32_t)u * 2654435761u;
    for (uint16_t& h : x) {   /* a fixed row of values in (-2, 2) */
        s = s * 1664525u + 1013904223u;
        h = ds41hx::f32_to_bf16((float)((int32_t)(s >> 8) - (1 << 23)) / (float)(1 << 22));
    }
    /* One token row with its six slots all on expert e: T' = AR * top_k, the whole-routing form (libr4d's
     * live tail takes it when the window sets LIVE_TAIL), and six copies of one row to compare. */
    int32_t sorted[ds41aux::kTopK], off[ds41aux::kExperts + 1];
    for (int i = 0; i < ds41aux::kTopK; ++i) sorted[i] = i;
    for (int i = 0; i <= ds41aux::kExperts; ++i) off[i] = i <= e ? 0 : ds41aux::kTopK;
    const AuxRows& R = aux_rows_[(size_t)li];
    aux::Job j;
    j.AR = 1;
    j.T = ds41aux::kTopK;
    j.x = x.data();
    j.sorted = sorted;
    j.off = off;
    j.gu = launch_of(R.gu[1]);
    j.sw = launch_of(R.sw[(size_t)ds41aux::kTopK]);
    j.dn = launch_of(R.dn[1]);
    j.gu_w = R.gu_w;
    j.dn_w = R.dn_w;
    j.gu_w.data = const_cast<void*>(v.gu_tab[(size_t)li]);
    j.dn_w.data = const_cast<void*>(v.dn_tab[(size_t)li]);
    j.rank = plan_->rank;
    j.world = plan_->world_size;
    int rc = lane->submit(j);
    /* The first launches load libr4d's code object and the blit kernels on the extra card: 30 s at most. */
    const int got = rc >= 0 ? lane->wait(aux_now_ns() + 30'000'000'000LL) : rc;
    if (got != 1) {
        if (got == 0) lane->abandon(false);
        mover_->release_host_unit(&hu);
        *why = fmt("rank %d's warm-up job (unit %d, layer %d expert %d) %s", r, u, layers_[(size_t)li].layer, e,
                   got == 0 ? "did not complete in 30 s" : ("failed: " + lane->why_dead()).c_str());
        return RAD_E_DEVICE;
    }
    HostExpertPool::Job cj;
    const int g = hu.bytes[0] == HostExpertPool::kGuBytes ? 0 : 1;
    cj.gu = hu.host[g];
    cj.dn = hu.host[1 - g];
    cj.packed = true;
    cj.gu_bytes = hu.slot[g];
    cj.dn_bytes = hu.slot[1 - g];
    cj.x = x.data();
    cj.y = cpu.data();
    cj.urgent = true;
    cj.cost = HostExpertPool::job_cost(1);
    HostExpertPool::Ticket t;
    hx_pool_->reserve(cj.cost);
    rc = hx_pool_->submit(&cj, 1, &t);
    if (rc >= 0) hx_pool_->wait(&t);
    else hx_pool_->unreserve(cj.cost);
    mover_->release_host_unit(&hu);
    if (rc < 0) {
        *why = fmt("rank %d's warm-up: the CPU pool refused the job (%d)", r, rc);
        return rc;
    }
    for (int k = 0; k < ds41aux::kTopK; ++k) {
        int first = -1;
        const int bad = ds41aux::row_diff(cpu.data(), lane->rows() + (int64_t)k * ds41aux::kEmbd, ds41aux::kEmbd, &first);
        if (!bad) continue;
        *why = fmt("rank %d's warm-up job (unit %d, layer %d expert %d): row %d has %d of %d values unlike the CPU "
                   "pool's (first at %d: 0x%04x on the CPU, 0x%04x on the extra card)", r, u, layers_[(size_t)li].layer,
                   e, k, bad, ds41aux::kEmbd, first, cpu[(size_t)first],
                   lane->rows()[(int64_t)k * ds41aux::kEmbd + first]);
        return RAD_E_STATE;
    }
    return RAD_OK;
}

/* ---- binding ---------------------------------------------------------------------------------------- */

void FileStager::aux_bind(aux::Lane* lane, const aux::RankView* view, const aux::Config& cfg) {
    aux_unbind();
    if (!lane || !view) return;
    if (!aux_units_) aux_units_.reset(new (std::nothrow) Mover::HostUnit[kHxUnits]);
    if (!aux_units_) return;
    aux_ = lane;
    aux_view_ = view;
    aux_verify_ = cfg.verify;
    aux_deadline_ns_ = cfg.deadline_us * 1000;
    aux_force_late_ = cfg.force_late;
    aux_seq_ = 0;
    aux_guard_bad_ = false;
    aux_stats_ = aux::Stats{};
}

void FileStager::aux_unbind() {
    aux_abandon();
    aux_ = nullptr;
    aux_view_ = nullptr;
    aux_pass_ = false;
    aux_take_ = false;
}

/* ---- the per-layer hooks ------------------------------------------------------------------------------ */

int FileStager::aux_prepare(int* appended) {
    *appended = 0;
    aux_cmp_.T = 0;
    if (hx_li_ < 0 || (size_t)hx_li_ >= hx_ops_.size()) return RAD_E_STATE;
    const HxOps& o = hx_ops_[(size_t)hx_li_];
    uint8_t worker[ds41aux::kExperts] = {};
    for (int32_t u : aux_list_) {
        const int32_t e = plan_->units[(size_t)u].expert;
        if (e >= 0 && e < ds41aux::kExperts) worker[e] = 1;
    }
    const int rc = ds41aux::compact(hx_idx_, hx_idx_ + hx_sorted_max(), (int)layer_rows(hx_li_), (int)o.top_k,
                                    (int)o.n_expert, worker, counts_host_, (int)counts_cap_, &aux_cmp_);
    if (rc == ds41aux::kCompactShape) {
        RAD_ERR("host experts: layer %d: the aux worker's compaction refused a pass of %lld rows",
                layers_[(size_t)hx_li_].layer, (long long)layer_rows(hx_li_));
        return RAD_E_STATE;
    }
    if (rc < 0) {
        const int e = -rc - 1;
        RAD_ERR("host experts: layer %d expert %d's rows in the sorted order disagree with its count %d (the aux "
                "worker's compaction)", layers_[(size_t)hx_li_].layer, e, e < counts_cap_ ? counts_host_[e] : -1);
        return RAD_E_STATE;
    }
    if (!aux_verify_) {
        for (int j = 0; j < aux_cmp_.T; ++j) hx_rows_.push_back(aux_cmp_.row_of[j]);
        *appended = aux_cmp_.T;
    }
    return RAD_OK;
}

void FileStager::aux_submit() {
    aux_job_ = kAuxNone;
    if (aux_cmp_.T <= 0 || hx_li_ < 0 || (size_t)hx_li_ >= aux_rows_.size()) return;
    const int32_t li = hx_li_;
    const AuxRows& R = aux_rows_[(size_t)li];
    const int AR = (int)layer_rows(li), T = aux_cmp_.T;
    aux::Job j;
    j.AR = AR;
    j.T = T;
    j.x = hx_x_;
    j.sorted = aux_cmp_.sorted;
    j.off = aux_cmp_.off;
    j.gu = launch_of(AR < (int)R.gu.size() ? R.gu[(size_t)AR] : nullptr);
    j.sw = launch_of(T < (int)R.sw.size() ? R.sw[(size_t)T] : nullptr);
    j.dn = launch_of(AR < (int)R.dn.size() ? R.dn[(size_t)AR] : nullptr);
    j.gu_w = R.gu_w;
    j.dn_w = R.dn_w;
    j.gu_w.data = (size_t)li < aux_view_->gu_tab.size() ? const_cast<void*>(aux_view_->gu_tab[(size_t)li]) : nullptr;
    j.dn_w.data = (size_t)li < aux_view_->dn_tab.size() ? const_cast<void*>(aux_view_->dn_tab[(size_t)li]) : nullptr;
    j.rank = plan_->rank;
    j.world = plan_->world_size;
    ++aux_seq_;
    ++aux_stats_.jobs;
    aux_stats_.units += aux_list_.size();
    aux_stats_.rows += (uint64_t)T;
    const int64_t t0 = aux_now_ns();
    const int rc = aux_->submit(j);
    aux_stats_.submit_ns += (uint64_t)(aux_now_ns() - t0);
    if (rc < 0) {   /* the lane said why, once; the CPU computes the rows at the gather */
        ++aux_stats_.failed;
        aux_job_ = kAuxFailed;
        return;
    }
    aux_t0_ns_ = t0;
    aux_job_ = aux_force_late_ > 0 && aux_seq_ % (uint64_t)aux_force_late_ == 0 ? kAuxForced : kAuxPending;
}

int FileStager::aux_finish() {
    const int job = aux_job_;
    aux_job_ = kAuxNone;
    if (job == kAuxNone) return RAD_OK;
    const int64_t t_pool = aux_now_ns();
    const int T = aux_cmp_.T;
    int got = -1;
    if (job == kAuxPending) {
        /* Late only if still pending at the later of the deadline from the submit and a short grace after
         * the pool's rows are in: a pool slower than the deadline must not abandon a finished job. The wait
         * queries at least once. */
        const int64_t deadline = std::max(aux_t0_ns_ + aux_deadline_ns_, t_pool + ds41aux::kGraceUs * 1000);
        got = aux_->wait(deadline);
        const int64_t t = aux_now_ns();
        aux_stats_.wait_ns += (uint64_t)(t - t_pool);
        if (got == 1) {
            const uint64_t rt = (uint64_t)(t - aux_t0_ns_);
            aux_stats_.rt_sum_ns += rt;
            aux_stats_.rt_max_ns = std::max(aux_stats_.rt_max_ns, rt);
            ++aux_stats_.rt_n;
        } else if (got < 0) {
            ++aux_stats_.failed;
        }
    } else if (job == kAuxForced) {
        got = 0;
        ++aux_stats_.forced_late;
    }
    if (got == 1) {
        const uint16_t* rows = aux_->rows();
        const int64_t rb = kHxEmbd * (int64_t)sizeof(uint16_t);
        if (!aux_verify_) {
            for (int j = 0; j < T; ++j)
                std::memcpy(hx_out_ + (int64_t)aux_cmp_.row_of[j] * kHxEmbd, rows + (int64_t)j * kHxEmbd, (size_t)rb);
            aux_stats_.placed += (uint64_t)T;
            return RAD_OK;
        }
        /* Verify: only rows the CPU wrote (a unit it borrowed or read as a miss); any other's hx_out_ row is
         * stale. */
        uint8_t have[ds41aux::kMaxSlots] = {};
        for (int32_t i : hx_rows_)
            if (i >= 0 && i < ds41aux::kMaxSlots) have[i] = 1;
        for (int j = 0; j < T; ++j) {
            const int32_t i = aux_cmp_.row_of[j];
            if (i < 0 || i >= ds41aux::kMaxSlots || !have[i]) {
                ++aux_stats_.not_compared;
                continue;
            }
            int first = -1;
            const uint16_t* a = hx_out_ + (int64_t)i * kHxEmbd;
            const uint16_t* c = rows + (int64_t)j * kHxEmbd;
            const int bad = ds41aux::row_diff(a, c, kHxEmbd, &first);
            ++aux_stats_.verified;
            if (!bad) continue;
            aux_stats_.differ_values += (uint64_t)bad;
            if (++aux_stats_.differ_rows <= 8) {
                int e = 0;
                while (e < ds41aux::kExperts - 1 && aux_cmp_.off[e + 1] <= j) ++e;
                const int32_t u = (size_t)hx_li_ < hx_unit_of_.size() ? hx_unit_of_[(size_t)hx_li_][(size_t)e] : -1;
                RAD_WARN("aux experts: rank %d layer %d unit %d expert %d sorted row %d: %d of %d values differ from "
                         "the CPU's (first at %d: 0x%04x on the CPU, 0x%04x on the extra card)", plan_->rank,
                         layers_[(size_t)hx_li_].layer, u, e, i, bad, (int)kHxEmbd, first, a[first], c[first]);
            }
        }
        return RAD_OK;
    }
    if (got == 0) {   /* late: the lane stays busy until it completes, then backs off */
        aux_->abandon(true);
        ++aux_stats_.late;
    }
    if (aux_verify_) {
        aux_stats_.not_compared += (uint64_t)T;
        return RAD_OK;
    }
    return aux_fallback();
}

/* The worker's units on the CPU, from the kept leases: hx_submit's jobs (urgent: the gather waits for them)
 * under a ticket of their own, which hx_drain also waits for. Their rows are in hx_rows_ already. */
int FileStager::aux_fallback() {
    const int64_t t0 = aux_now_ns();
    aux_fb_jobs_.clear();
    uint64_t units = 0;
    for (int k = 0; k < aux_n_; ++k) {
        const Mover::HostUnit& hu = aux_units_[k];
        if (!hu.n) continue;
        HostExpertPool::Job proto;
        const int g = hu.bytes[0] == HostExpertPool::kGuBytes ? 0 : 1;
        proto.gu = hu.host[g];
        proto.dn = hu.host[1 - g];
        if (hu.packed) {
            proto.packed = true;
            proto.gu_bytes = hu.slot[g];
            proto.dn_bytes = hu.slot[1 - g];
        }
        proto.urgent = true;
        const int n = hx_jobs_into(plan_->units[(size_t)hu.unit].expert, proto, &aux_fb_jobs_, nullptr);
        if (n < 0) return n;
        ++units;
    }
    if (!aux_fb_jobs_.empty()) {
        int64_t cost = 0;
        for (const HostExpertPool::Job& j : aux_fb_jobs_) cost += j.cost;
        hx_pool_->reserve(cost);   /* the submit takes the reservation over */
        const int rc = hx_pool_->submit(aux_fb_jobs_.data(), (int)aux_fb_jobs_.size(), &aux_ticket_);
        if (rc < 0) {
            hx_pool_->unreserve(cost);
            return rc;
        }
        aux_fb_ = true;
        hx_pool_->wait(&aux_ticket_);
        aux_fb_ = false;
    }
    aux_stats_.fallback_units += units;
    aux_stats_.fallback_rows += (uint64_t)aux_cmp_.T;
    aux_stats_.fallback_ns += (uint64_t)(aux_now_ns() - t0);
    return RAD_OK;
}

void FileStager::aux_abandon() {
    if (aux_fb_ && hx_pool_) {
        hx_pool_->wait(&aux_ticket_);
        aux_fb_ = false;
    }
    if (aux_ && (aux_job_ == kAuxPending || aux_job_ == kAuxForced)) aux_->abandon(false);
    aux_job_ = kAuxNone;
    if (aux_units_)
        for (int k = 0; k < aux_n_; ++k) mover_->release_host_unit(&aux_units_[k]);
    aux_n_ = 0;
    aux_list_.clear();
    aux_take_ = false;
}

void FileStager::aux_guard(int32_t op, const RadArgs* a, int (*launch)(const RadArgs*, RadStream)) {
    if (aux_guard_bad_ || !a) return;
    const int32_t code = aux_op_[(size_t)op];
    const size_t li = (size_t)(code / 3);
    const int k = code % 3;
    if (li >= aux_rows_.size() || !aux_rows_[li].ok) return;
    const int64_t rows = layer_rows((int32_t)li);
    if (rows < 1 || rows > ds41aux::kMaxRows) return;   /* a height the worker never takes */
    const AuxRows& R = aux_rows_[li];
    const std::vector<const Resolved*>& v = k == 0 ? R.gu : k == 1 ? R.sw : R.dn;
    const int64_t M = k == 1 ? (a->n_t >= 2 ? a->t[1].shape[0] : -1) : rows;
    const Resolved* r = M >= 1 && M < (int64_t)v.size() ? v[(size_t)M] : nullptr;
    bool ok = r && r->row && r->row->info && launch == r->row->info->launch && a->p == r->geom.params();
    if (ok && k != 1) {   /* the weight-table operand the worker builds is the one the rank issued */
        const RadTensor& m = k == 0 ? R.gu_w : R.dn_w;
        const RadTensor& t = a->t[1];
        ok = a->n_t >= 2 && t.dtype == m.dtype && t.rank == m.rank;
        for (uint32_t d = 0; ok && d < t.rank; ++d) ok = t.shape[d] == m.shape[d] && t.stride[d] == m.stride[d];
    }
    if (ok) return;
    aux_guard_bad_ = true;
    RAD_WARN("aux experts: rank %d: routed layer %d's %s launched another row or operand form than the worker "
             "resolved (M %lld); this rank's worker takes no more jobs", plan_ ? plan_->rank : -1,
             layers_[li].layer, aux_op_name(k), (long long)M);
}

}  // namespace rad

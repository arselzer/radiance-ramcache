/* stager_aux.cpp -- the file stager's part of the aux expert worker (aux_expert_worker.h): the unit set's
 * candidates, the rows a job relaunches, the load, the warm-up and the binding. With no binding (aux_ null)
 * none of this runs. */
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
    auto done = [&](int rc) {
        if (bounce) rad_dev_free(bounce, RAD_MEM_HOST_PINNED);
        if (back) rad_dev_free(back, RAD_MEM_HOST_PINNED);
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
        int rc = rad_memcpy_async(slot, bounce, n[0], s);
        if (rc >= 0) rc = rad_memcpy_async(slot + L.dn_off, bounce + L.dn_off, n[1], s);
        if (rc >= 0) rc = rad_memcpy_async(back, slot, n[0], s);
        if (rc >= 0) rc = rad_memcpy_async(back + L.dn_off, slot + L.dn_off, n[1], s);
        const int ok = rc >= 0 ? lane->sync_bounded(10'000'000'000LL) : rc;
        if (ok != 1) {
            if (cached) mover_->release_host_unit(&hu);
            *why = fmt("rank %d unit %d: the copies to and from device %d %s", r, u, w->device(),
                       ok == 0 ? "did not complete in 10 s" : "failed");
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

}  // namespace rad

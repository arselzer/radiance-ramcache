/* card_expert_stager.cpp -- see card_expert_stager.h. */
#include "place/card_expert_stager.h"
#include "rad_internal.h"
#include "device/device.h"
#include "place/readahead.h"

#include <algorithm>
#include <climits>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rad {
namespace p1 {

/* ---- tunables ------------------------------------------------------------------------------------ */
namespace {
bool whole(const char* v, int lo, int hi, int* out) {
    if (!v || !*v) return true;   /* unset: the default stays */
    long long n = 0;
    for (const char* p = v; *p; ++p) {
        if (*p < '0' || *p > '9' || n > 100000000) return false;
        n = n * 10 + (*p - '0');
    }
    if (n < lo || n > hi) return false;
    *out = (int)n;
    return true;
}
}  /* namespace */

bool parse_tunables(const std::function<const char*(const char*)>& get, Tunables* t, std::string* why) {
    Tunables r;
    struct K { const char* env; int lo, hi; int* at; } ks[] = {
        { ds41p1::kWaveEnv, 1, 16, &r.wave },
        { ds41p1::kInflightEnv, 1, 64, &r.inflight },
        { ds41p1::kReadyEnv, 1, 32, &r.ready },
        { ds41p1::kAheadEnv, 0, 64, &r.ahead },
        { ds41p1::kVerifyEnv, 0, 4096, &r.verify },
        { ds41p1::kMinRowsEnv, ds41p1::kMinArmRows, ds41p1::kTwinMaxRows, &r.min_rows },
        { ds41p1::kLendEnv, 0, ds41p1::kMaxLend, &r.lend },
        { ds41p1::kLendArenaEnv, 0, ds41p1::kMaxLend, &r.lend_arena },
        { ds41p1::kGraceEnv, 0, 60000, &r.grace_ms },
        { ds41p1::kLendHeatEnv, 0, ds41p1::kMaxLendHeat, &r.lend_heat },
        { ds41p1::kLendSlabEnv, 0, ds41p1::kMaxLendSlab, &r.lend_slab },
        /* Behind the slab: unset, the half-life's default follows it. */
        { ds41p1::kLendHeatHalfEnv, 1, ds41p1::kMaxLendHeatHalfRows, &r.lend_heat_half },
        { ds41p1::kLendHeatMarginEnv, 100, 1000, &r.lend_heat_margin },
    };
    for (const K& k : ks) {
        const char* v = get(k.env);
        if (k.at == &r.lend_heat_half && r.lend_slab > 0) r.lend_heat_half = ds41p1::kDefLendHeatHalfRowsSlab;
        if (!whole(v, k.lo, k.hi, k.at)) {
            if (why) *why = std::string(k.env) + "=" + v + ": a whole number from " + std::to_string(k.lo) +
                            " to " + std::to_string(k.hi);
            return false;
        }
    }
    if (const char* a = get(ds41p1::kAlignEnv); a && *a) {
        if (!std::strcmp(a, "end")) r.align_end = true;
        else if (!std::strcmp(a, "start")) r.align_end = false;
        else {
            if (why) *why = std::string(ds41p1::kAlignEnv) + "=" + a + ": end or start";
            return false;
        }
    }
    const char* span = get(kSwigluSpanEnv);
    const int ss = ds41p1::env_state(span);
    if (ss < 0) {
        if (why) *why = std::string(kSwigluSpanEnv) + "=" + span + ": 1 is on, 0 / unset / empty off";
        return false;
    }
    r.swiglu_span = ss == 1;
    *t = r;
    return true;
}

bool swiglu_span_launch_ok(const RadKernelInfo* row, RadLaunchFn launch) {
    return row && row->name && row->op && row->domain == RAD_DOMAIN_DEVICE && launch &&
           !std::strcmp(row->name, "swiglu_clamped_bf16") && !std::strcmp(row->op, "swiglu_clamped") &&
           row->launch == launch && !row->init && !row->scratch;
}

bool wave_swiglu_span(const int32_t* counts, int n, const void* const* gu, const void* const* dn,
                     int64_t rows, SwigluSpan* out) {
    if (!counts || n < 1 || n > 96 || !gu || !dn || rows < 0 || !out) return false;
    int64_t off = 0;
    SwigluSpan s{ rows, 0 };
    for (int e = 0; e < n; ++e) {
        if (counts[e] < 0 || counts[e] > rows - off || ds41p1::is_skip(gu[e]) != ds41p1::is_skip(dn[e]))
            return false;
        const int64_t hi = off + counts[e];
        if (!ds41p1::is_skip(gu[e]) && hi > off) {
            s.lo = std::min(s.lo, off);
            s.hi = hi;
        }
        off = hi;
    }
    *out = s.hi > s.lo ? s : SwigluSpan{};
    return true;
}

bool swiglu_span_view(const RadArgs* full, SwigluSpan span, std::vector<RadTensor>* tensors,
                      std::vector<RadParam>* params, RadArgs* out) {
    if (!full || !tensors || !params || !out || full->n_t != 2 || !full->t || full->n_p < 0 ||
        (full->n_p && !full->p) || full->instance || full->scratch_bytes != 0)
        return false;
    long long n = 0;
    if (!rad_args_geti(full, "n", &n) || n < 1 || n > INT_MAX / 2) return false;
    const RadTensor& gu = full->t[0];
    const RadTensor& y = full->t[1];
    const int64_t rows = y.shape[0];
    if (gu.dtype != RAD_BF16 || y.dtype != RAD_BF16 || gu.rank != 2 || y.rank != 2 ||
        rows < 1 || rows > 65535 || gu.shape[0] != rows || gu.shape[1] != 2 * n || y.shape[1] != n ||
        gu.stride[1] != 1 || y.stride[1] != 1 || gu.stride[0] < 2 * n || y.stride[0] < n ||
        !gu.data || !y.data || ((uintptr_t)gu.data & 1) || ((uintptr_t)y.data & 1) ||
        span.lo < 0 || span.hi < span.lo || span.hi > rows)
        return false;
    /* Bound byte arithmetic before rebasing the opaque device pointers. Disjoint operands are the
     * served graph's form; an overlapping view has no proven cross-row replay contract. */
    uintptr_t end[2];
    for (int k = 0; k < 2; ++k) {
        const RadTensor& t = full->t[k];
        if (t.stride[0] > INT64_MAX / 2 / rows) return false;
        const uint64_t bytes = (uint64_t)((rows - 1) * t.stride[0] + t.shape[1]) * 2;
        if (bytes > UINTPTR_MAX - (uintptr_t)t.data) return false;
        if ((uint64_t)span.lo * t.stride[0] * 2 > UINTPTR_MAX - (uintptr_t)t.data) return false;
        end[k] = (uintptr_t)t.data + bytes;
    }
    if ((uintptr_t)gu.data < end[1] && (uintptr_t)y.data < end[0]) return false;
    std::vector<RadTensor> ts(full->t, full->t + 2);
    std::vector<RadParam> ps;
    if (full->n_p) ps.assign(full->p, full->p + full->n_p);
    for (RadTensor& t : ts) {
        t.data = (void*)((uintptr_t)t.data + (uint64_t)span.lo * t.stride[0] * 2);
        t.shape[0] = span.hi - span.lo;
    }
    for (RadParam& p : ps)
        if (p.kind == RAD_P_INT && p.key && !std::strcmp(p.key, "M")) p.ival = span.hi - span.lo;
    RadArgs a = *full;
    *tensors = std::move(ts);
    *params = std::move(ps);
    a.t = tensors->data();
    a.p = params->empty() ? nullptr : params->data();
    *out = a;
    return true;
}

/* ---- the scheduling ------------------------------------------------------------------------------- */
int CardExpertStager::init(Device* dev, const char* ring_base, int n_slots, int wave, int n_local,
                           const void* alias_base, int ready_slots, std::string* why) {
    auto no = [&](const std::string& w) { if (why) *why = w; return -1; };
    if (!dev || !ring_base || !alias_base) return no("no device, ring or alias base");
    if (((uintptr_t)ring_base & 63) || ((uintptr_t)alias_base & 63)) return no("the ring is not 64-byte aligned");
    /* The engine's path asks for ds41p1::kMinRingSlots; the scheduling itself holds with two (the
     * ring stress test runs it there). */
    if (n_slots < 2) return no("the ring holds " + std::to_string(n_slots) + " slots, fewer than 2");
    if (wave < 1 || wave > 16 || wave > n_slots) return no("a wave of " + std::to_string(wave) + " units in a ring of " +
                                              std::to_string(n_slots));
    if (n_local < 1 || n_local > 96) return no("a layer of " + std::to_string(n_local) + " local experts");
    dev_ = dev;
    ring_ = ring_base;
    alias_ = alias_base;
    n_ = n_slots;
    wave_ = wave;
    n_local_ = n_local;
    ready_slots_ = std::max(1, ready_slots);
    head_ = 0;
    slot_.assign((size_t)n_, Slot{});
    for (Slot& s : slot_) {
        s.copy_ev = dev_->event_create(true);
        s.free_ev = dev_->event_create(false);
        if (s.copy_ev < 0 || s.free_ev < 0) return no("could not create the ring's events");
    }
    tab_gu_.assign((size_t)n_local_, nullptr);
    tab_dn_.assign((size_t)n_local_, nullptr);
    open_ = false;
    st_ = Stats{};
    return 0;
}

int CardExpertStager::begin_layer(int parity) {
    if (!dev_ || open_) return -1;
    u_.clear();
    ram_order_.clear();
    ram_next_ = 0;
    drive_q_.clear();
    queued_.clear();
    launched_ = 0;
    waiting_drive_ = 0;
    parity_ = parity & 1;
    wave_no_ = 0;
    open_ = true;
    started_ = false;
    return 0;
}

int CardExpertStager::add_ready(Kind k, int e, int32_t unit, int rows, const Source& s, int32_t handle) {
    if (!open_ || started_ || e < 0 || e >= n_local_ || !s.gu || !s.dn || s.gu_bytes <= 0 ||
        s.dn_bytes <= 0 || s.gu_bytes > ds41p1::kDnOff || s.dn_bytes > ds41p1::kSlotBytes - ds41p1::kDnOff)
        return -1;
    for (const U& x : u_)
        if (x.e == e) return -1;   /* one unit an expert */
    U u;
    u.e = e;
    u.unit = unit;
    u.rows = rows;
    u.handle = handle;
    u.kind = k;
    u.st = St::Ready;
    u.src = s;
    u_.push_back(u);
    ++(k == Kind::Pool ? st_.pool : st_.ram);
    return 0;
}

int CardExpertStager::add_ram(int e, int32_t unit, int rows, const Source& s, int32_t handle) {
    return add_ready(Kind::Ram, e, unit, rows, s, handle);
}

int CardExpertStager::add_pool(int e, int32_t unit, int rows, const Source& s) {
    if (s.packed) return -1;   /* the pool holds the card's block form */
    return add_ready(Kind::Pool, e, unit, rows, s, -1);
}

int CardExpertStager::add_drive(int e, int32_t unit, int rows) {
    if (!open_ || started_ || e < 0 || e >= n_local_) return -1;
    for (const U& x : u_)
        if (x.e == e) return -1;
    U u;
    u.e = e;
    u.unit = unit;
    u.rows = rows;
    u.kind = Kind::Drive;
    u.st = St::Waiting;
    u_.push_back(u);
    ++waiting_drive_;
    ++st_.drive;
    return 0;
}

int CardExpertStager::drive_ready(int32_t unit, const Source& s, int32_t handle) {
    if (!open_ || !s.gu || !s.dn || s.gu_bytes <= 0 || s.dn_bytes <= 0 || s.gu_bytes > ds41p1::kDnOff ||
        s.dn_bytes > ds41p1::kSlotBytes - ds41p1::kDnOff)
        return -1;
    for (size_t i = 0; i < u_.size(); ++i) {
        U& u = u_[i];
        if (u.kind != Kind::Drive || u.unit != unit || u.st != St::Waiting) continue;
        u.src = s;
        u.handle = handle;
        u.st = St::Ready;
        --waiting_drive_;
        drive_q_.push_back((int)i);
        return 0;
    }
    return -1;   /* not a unit this layer waits for */
}

int CardExpertStager::start() {
    if (!open_ || started_) return -1;
    started_ = true;
    for (size_t i = 0; i < u_.size(); ++i)
        if (u_[i].kind != Kind::Drive) ram_order_.push_back((int)i);
    std::stable_sort(ram_order_.begin(), ram_order_.end(),
                     [&](int a, int b) { return u_[(size_t)a].rows > u_[(size_t)b].rows; });
    ++st_.layers;
    return pump();
}

/* Lets go of slot s's source, once the host has seen its copy's event pass (wait: sync for it). */
int CardExpertStager::let_go(int s, bool wait) {
    Slot& sl = slot_[(size_t)s];
    if (!sl.held) return 0;
    int q = dev_->query(sl.copy_ev);
    if (q < 0) return q;
    if (!q) {
        if (!wait) return 0;
        ++st_.release_syncs;
        const int r = dev_->sync(sl.copy_ev);
        if (r < 0) return r;
    }
    sl.held = false;
    dev_->release(sl.kind, sl.handle);
    return 0;
}

void CardExpertStager::poll() {
    for (int s = 0; s < n_; ++s)
        if (slot_[(size_t)s].held) (void)let_go(s, false);
}

int CardExpertStager::held() const {
    int n = 0;
    for (const Slot& s : slot_) n += s.held;
    return n;
}

int CardExpertStager::queue_one(int i) {
    U& u = u_[(size_t)i];
    if (u.st != St::Ready || (int)queued_.size() >= n_) return -1;
    const int s = (int)(head_ % (uint64_t)n_);
    Slot& sl = slot_[(size_t)s];
    /* The slot's last source goes before its copy event is recorded again: that copy has run if the
     * stream reached the wave that read it, which the free event below orders before this copy. */
    if (const int r = let_go(s, true); r < 0) return r;
    const int rc = dev_->copy(s, u.src, sl.free_rec ? sl.free_ev : -1, sl.copy_ev);
    if (rc < 0) return rc;
    sl.held = u.kind != Kind::Pool;   /* a pool unit is read in place: nothing to let go */
    sl.kind = u.kind;
    sl.handle = u.handle;
    u.slot = s;
    u.st = St::Queued;
    ++head_;
    queued_.push_back(i);
    ++st_.copies;
    st_.copy_bytes += (uint64_t)(u.src.gu_bytes + u.src.dn_bytes);
    return 0;
}

/* Copies into the ring as far as it has room: a drive unit first while the pinned ready slots are
 * at least half full (they hold the drive's pipeline), else the RAM and pool units in descending rows,
 * then the drive units in the order they were packed. */
int CardExpertStager::pump() {
    while ((int)queued_.size() < n_) {
        const bool ram_left = ram_next_ < ram_order_.size();
        int pick = -1;
        if (!drive_q_.empty() && (!ram_left || (int)drive_q_.size() * 2 >= ready_slots_)) {
            pick = drive_q_.front();
            drive_q_.pop_front();
        } else if (ram_left) {
            pick = ram_order_[ram_next_++];
        } else {
            break;
        }
        if (const int r = queue_one(pick); r < 0) return r;
    }
    return 0;
}

int CardExpertStager::launch_wave(int n) {
    if (n <= 0 || n > (int)queued_.size()) return -1;
    if (wave_no_ >= 96) return -1;   /* more waves than the tables have rows */
    for (int e = 0; e < n_local_; ++e) {
        tab_gu_[(size_t)e] = ds41p1::mark(alias_, ds41p1::kMarkSkip);
        tab_dn_[(size_t)e] = ds41p1::mark(alias_, ds41p1::kMarkSkip);
    }
    int idx[16];
    for (int k = 0; k < n; ++k) {
        const int i = queued_.front();
        queued_.pop_front();
        idx[k] = i;
        U& u = u_[(size_t)i];
        if (u.st != St::Queued || u.slot < 0) return -1;
        const char* at = ring_ + (int64_t)u.slot * ds41p1::kSlotBytes;
        const uintptr_t m = u.src.packed ? ds41p1::kMarkPacked : 0;
        tab_gu_[(size_t)u.e] = ds41p1::mark(at, m);
        tab_dn_[(size_t)u.e] = ds41p1::mark(at + ds41p1::kDnOff, m);
        if (const int r = dev_->compute_wait(slot_[(size_t)u.slot].copy_ev); r < 0) return r;
    }
    if (const int r = dev_->wave(parity_, wave_no_, tab_gu_.data(), tab_dn_.data(), n_local_); r < 0) return r;
    for (int k = 0; k < n; ++k) {
        U& u = u_[(size_t)idx[k]];
        Slot& sl = slot_[(size_t)u.slot];
        if (const int r = dev_->compute_record(sl.free_ev); r < 0) return r;
        sl.free_rec = true;
        u.st = St::Launched;
        ++launched_;
    }
    ++wave_no_;
    ++st_.waves;
    return 0;
}

int CardExpertStager::run() {
    if (!open_ || !started_) return -1;
    const int total = (int)u_.size();
    while (launched_ < total) {
        if (const int r = pump(); r < 0) return r;
        const bool more = ram_next_ < ram_order_.size() || waiting_drive_ > 0 || !drive_q_.empty();
        const int q = (int)queued_.size();
        if (q >= wave_ || (!more && q > 0)) {
            if (q >= n_ && more) ++st_.ring_full_waits;
            if (const int r = launch_wave(std::min(q, wave_)); r < 0) return r;
            poll();
            continue;
        }
        if (waiting_drive_ > 0 && drive_q_.empty()) {
            /* The packer needs a free pinned ready slot for the next drive unit. When the ring holds
             * all of them, free one first: launch what is queued (its copies then pass), else wait for
             * the copies of the units already launched and let their slots go. */
            int held_drive = 0;
            for (const Slot& sl : slot_) held_drive += sl.held && sl.kind == Kind::Drive;
            if (held_drive >= ready_slots_) {
                if (q > 0) {
                    if (const int r = launch_wave(std::min(q, wave_)); r < 0) return r;
                    poll();
                    continue;
                }
                for (int s = 0; s < n_; ++s)
                    if (slot_[(size_t)s].held && slot_[(size_t)s].kind == Kind::Drive)
                        if (const int r = let_go(s, true); r < 0) return r;
                continue;
            }
            ++st_.drive_waits;
            const int r = dev_->wait_drive();
            if (r < 0) return r;
            if (r == 0) return -1;   /* nothing came although units are owed */
            continue;
        }
        if (!more && q == 0) break;
    }
    if (launched_ != total) return -1;
    st_.max_waves = std::max<uint64_t>(st_.max_waves, (uint64_t)wave_no_);
    open_ = false;
    return 0;
}

int CardExpertStager::end_pass() {
    int rc = open_ ? -1 : 0;
    open_ = false;
    for (int s = 0; s < n_; ++s) {
        const int r = let_go(s, true);
        if (r < 0 && rc >= 0) rc = r;
        if (r < 0) {   /* the source goes all the same: nothing will ever wait on it again */
            Slot& sl = slot_[(size_t)s];
            if (sl.held) {
                sl.held = false;
                dev_->release(sl.kind, sl.handle);
            }
        }
    }
    /* Units of an unfinished layer still holding their sources outside the ring. */
    for (U& u : u_)
        if (u.st == St::Ready && u.handle >= 0 && u.kind != Kind::Pool) {
            dev_->release(u.kind, u.handle);
            u.handle = -1;
        }
    u_.clear();
    queued_.clear();
    drive_q_.clear();
    return rc;
}

int CardExpertStager::slot_of(const void* entry) const {
    const char* p = (const char*)ds41p1::unmark(entry);
    if (!ring_ || p < ring_) return -1;
    const int64_t off = p - ring_;
    const int64_t s = off / ds41p1::kSlotBytes;
    if (s >= n_) return -1;
    return (int)s;
}

/* ---- the lent ring ------------------------------------------------------------------------------------- */
int RingLender::init(LendHost* host, int n, int64_t grace_ms, int keep) {
    if (!host || n < 1 || grace_ms < 0 || keep < 0 || keep > n || keep > ds41p1::kMaxLendSlab ||
        n - keep > 2 * ds41p1::kMaxLend)
        return -1;
    host_ = host;
    slot_.assign((size_t)n, Slot{});
    keep_ = keep;
    for (int k = n - keep; k < n; ++k) {
        slot_[(size_t)k].keep = true;
        slot_[(size_t)k].st = St::Want;
    }
    grace_ = grace_ms;
    last_p1_ = 0;
    lent_ = true;
    any_p1_ = false;
    st_ = Stats{};
    return 0;
}

int RingLender::residents() const {
    int n = 0;
    for (const Slot& s : slot_) n += s.st == St::Resident;
    return n;
}

int RingLender::busy() const {
    int n = 0;
    for (const Slot& s : slot_) n += s.st == St::Reading || s.st == St::Landed || s.st == St::Copying;
    return n;
}

/* Slot k's load or copy ended without a resident: Idle while lent; Want for one a card-path pass
 * overtook (whatever became of it: its ring slot may hold the card path's bytes now, so it reloads from
 * the start); else it failed, and its unit stays on the CPU path until the next return. A kept slot is
 * never lent or overtaken: its load or copy failed. */
void RingLender::settle(int k) {
    Slot& s = slot_[(size_t)k];
    const bool gone = away(s), overtaken = s.cancel || gone;
    ++(overtaken ? st_.cancelled : st_.failed);
    s.cancel = false;
    s.st = gone ? St::Idle : overtaken ? St::Want : St::Failed;
}

/* Landed loads copied (or let go, overtaken), passed copies published (or let go). */
void RingLender::poll() {
    bool ok = false;
    for (int k; (k = host_->landed(&ok)) >= 0;) {
        if (k >= n() || slot_[(size_t)k].st != St::Reading) continue;   /* not a load begun here */
        Slot& s = slot_[(size_t)k];
        if (!ok || s.cancel || away(s)) {
            host_->drop(k);
            settle(k);
            continue;
        }
        s.st = St::Landed;
    }
    for (int k = 0; k < n(); ++k) {
        Slot& s = slot_[(size_t)k];
        if (s.st == St::Landed) {
            if (away(s)) {
                host_->drop(k);
                settle(k);
                continue;
            }
            if (host_->copy(k) < 0) {
                host_->drop(k);
                settle(k);
                continue;
            }
            s.st = St::Copying;
        }
        if (s.st != St::Copying) continue;
        const int q = host_->copied(k);
        if (q == 0) continue;
        host_->drop(k);
        if (q < 0 || s.cancel || away(s)) {
            settle(k);
            continue;
        }
        host_->publish(k, true);
        s.st = St::Resident;
        ++st_.reloaded;
        if (s.moved) {   /* the unit the heat policy gave the slot is in */
            s.moved = false;
            ++st_.retargets;
        }
    }
}

/* Loads for the slots that want one, as far as the host has buffers. */
void RingLender::pump() {
    for (int k = 0; k < n(); ++k) {
        Slot& s = slot_[(size_t)k];
        if (s.st != St::Want) continue;
        const int r = host_->load(k);
        if (r == 0) continue;
        if (r < 0) {
            s.st = St::Failed;
            ++st_.failed;
            continue;
        }
        s.st = St::Reading;
        s.cancel = false;
    }
}

int RingLender::begin_pass(bool p1, int64_t now_ms) {
    if (!host_) return 0;
    if (p1) {
        int withdrawn = 0;
        for (int k = 0; k < n(); ++k) {
            Slot& s = slot_[(size_t)k];
            if (s.keep) continue;   /* the slab's: no pass writes there, its resident stays */
            switch (s.st) {
            case St::Resident:
                host_->publish(k, false);
                ++withdrawn;
                s.st = St::Idle;
                break;
            case St::Landed:
                host_->drop(k);
                ++st_.cancelled;
                s.st = St::Idle;
                break;
            case St::Copying:
                host_->fence(k);   /* the pass's writes into the slot wait for the copy */
                s.cancel = true;
                break;
            case St::Reading:
                s.cancel = true;
                break;
            default:
                s.st = St::Idle;
                break;
            }
        }
        if (!lent_) ++st_.lends;
        st_.withdrawn += (uint64_t)withdrawn;
        lent_ = true;
        any_p1_ = true;
        last_p1_ = now_ms;
        poll();   /* overtaken loads and copies that have ended give their buffers back */
        return withdrawn ? 1 : 0;
    }
    if (lent_) {
        if (any_p1_ && now_ms - last_p1_ < grace_) {
            ++st_.held_back;
            poll();
            return 0;
        }
        lent_ = false;
        ++st_.returns;
        for (Slot& s : slot_)
            if (s.st == St::Idle || s.st == St::Failed) s.st = St::Want;
    }
    poll();
    pump();
    return 0;
}

void RingLender::end_p1(int64_t now_ms) {
    any_p1_ = true;
    last_p1_ = now_ms;
}

int RingLender::prime() {
    if (!host_ || !keep_) return 0;
    poll();
    pump();   /* while the other slots are lent out only the kept ones want a load */
    int on_the_way = 0;
    for (const Slot& s : slot_)
        on_the_way += s.keep && (s.st == St::Want || s.st == St::Reading || s.st == St::Landed || s.st == St::Copying);
    return on_the_way;
}

/* ---- the lent slots' heat ---------------------------------------------------------------------------------- */
bool RingLender::settled() const {
    if (!host_ || lent_) return false;
    for (const Slot& s : slot_)
        if (s.st != St::Resident || s.cancel) return false;
    return true;
}

int RingLender::retarget(const int* ks, int n) {
    if (!host_ || !ks || n < 1) return 0;
    if (lent_) {
        decline(Decline::Lent);
        return 0;
    }
    if (!settled()) {
        decline(Decline::Busy);
        return 0;
    }
    int done = 0;
    for (int i = 0; i < n; ++i) {
        const int k = ks[i];
        if (k < 0 || k >= this->n() || slot_[(size_t)k].st != St::Resident) continue;   /* no slot, or named twice */
        /* The order is the fence's: the DMA lane waits behind the resident's last reader first, then the
         * resident is withdrawn, then the slot changes its unit; its reload comes last. */
        if (host_->retarget_fence(k) < 0) {
            decline(Decline::Fence);
            break;
        }
        host_->publish(k, false);
        host_->retarget(k);
        Slot& s = slot_[(size_t)k];
        s.st = St::Want;
        s.cancel = false;
        s.moved = true;
        ++st_.retarget_withdrawn;
        ++done;
    }
    if (done) pump();
    return done;
}

int LendHeat::init(size_t n_units, int max_pairs, int half_rows, int margin_pct) {
    if (max_pairs < 1 || max_pairs > ds41p1::kMaxLendHeat || half_rows < 1 || margin_pct < 100) return -1;
    h_.assign(n_units, 0.0);
    scale_ = 1.0;
    half_ = (double)half_rows;
    margin_ = (double)margin_pct / 100.0;
    seen_ = 0.0;
    max_ = max_pairs;
    dirty_ = false;
    return 0;
}

void LendHeat::begin_pass(int64_t rows) {
    if (rows <= 0) return;
    seen_ += (double)rows;
    /* A pass of very many half-lives leaves nothing of the older counts either way: bounded, so the scale
     * stays finite. */
    scale_ *= std::exp2(std::min((double)rows / half_, 400.0));
    if (scale_ > 0x1p+500) {
        const double inv = 1.0 / scale_;
        for (double& x : h_) x *= inv;
        scale_ = 1.0;
    }
}

int LendHeat::choose(const int32_t* slot_unit, int n_slots, const int32_t* cand, size_t n_cand, const uint8_t* taken,
                     Pair* out) const {
    if (!slot_unit || !cand || !out || n_slots < 1 || max_ < 1) return 0;
    const int want = std::min(max_, n_slots);
    /* The coldest slots, coldest first. */
    int    cold[ds41p1::kMaxLendHeat];
    double cold_h[ds41p1::kMaxLendHeat];
    int    nc = 0;
    for (int k = 0; k < n_slots; ++k) {
        const int32_t u = slot_unit[k];
        if (u < 0 || (size_t)u >= h_.size()) return 0;   /* a slot with no unit of this plan: nothing moves */
        const double h = h_[(size_t)u];
        if (nc == want && h >= cold_h[nc - 1]) continue;
        int at = nc < want ? nc++ : nc - 1;
        for (; at > 0 && h < cold_h[at - 1]; --at) {
            cold[at] = cold[at - 1];
            cold_h[at] = cold_h[at - 1];
        }
        cold[at] = k;
        cold_h[at] = h;
    }
    /* The hottest candidates off the card, hottest first; none at or below what the coldest slot asks. */
    const double floor0 = margin_ * cold_h[0] + kMinRows * scale_;
    int32_t hot[ds41p1::kMaxLendHeat];
    double  hot_h[ds41p1::kMaxLendHeat];
    int     nh = 0;
    for (size_t i = 0; i < n_cand; ++i) {
        const int32_t u = cand[i];
        if (u < 0 || (size_t)u >= h_.size()) continue;
        const double h = h_[(size_t)u];
        if (h <= floor0 || (nh == want && h <= hot_h[nh - 1]) || (taken && taken[(size_t)u])) continue;
        bool twice = false;
        for (int j = 0; j < nh; ++j) twice = twice || hot[j] == u;
        if (twice) continue;
        int at = nh < want ? nh++ : nh - 1;
        for (; at > 0 && h > hot_h[at - 1]; --at) {
            hot[at] = hot[at - 1];
            hot_h[at] = hot_h[at - 1];
        }
        hot[at] = u;
        hot_h[at] = h;
    }
    int n = 0;
    for (; n < nc && n < nh; ++n) {
        if (!(hot_h[n] > margin_ * cold_h[n] + kMinRows * scale_)) break;
        out[n].slot = cold[n];
        out[n].unit = hot[n];
    }
    return n;
}

/* ---- the engine's side ------------------------------------------------------------------------------ */
namespace {
uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
        .count();
}
constexpr int kWaveRows = 96;   /* wave table rows a parity: a wave a touched expert at most */
}  /* namespace */

CardExpertPath::~CardExpertPath() {
    lend_stop();
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_read_.notify_all();
    cv_pack_.notify_all();
    cv_done_.notify_all();
    cv_idle_.notify_all();
    if (reader_.joinable()) reader_.join();
    if (packer_.joinable()) packer_.join();
    for (auto& h : hu_)
        if (h && h->n && mover_) mover_->release_host_unit(h.get());
    for (size_t i = 0; i < ev_.size(); ++i)
        if (ev_[i]) {
            if (ev_host_[i]) (void)rad_event_sync(ev_[i]);
            rad_event_destroy(ev_[i]);
        }
    for (uint8_t* r : ready_)
        if (r) rad_dev_free(r, RAD_MEM_HOST_PINNED);
    if (tab_pin_) rad_dev_free(tab_pin_, RAD_MEM_HOST_PINNED);
    if (tab_dev_) rad_dev_free(tab_dev_, RAD_MEM_DEVICE);
}

int CardExpertPath::init(Mover* mover, const Tunables& t, int rank, PlanesGet planes, Insert insert,
                         std::string* why) {
    auto no = [&](const std::string& w) { if (why) *why = w; return RAD_E_INVAL; };
    if (on_) return no("initialised twice");
    if (!mover || !planes) return no("no mover");
    mover_ = mover;
    t_ = t;
    rank_ = rank;
    planes_ = std::move(planes);
    insert_ = std::move(insert);
    int64_t bytes = 0;
    ring_ = mover_->file_region(&bytes);
    const int n = ring_ ? (int)(bytes / ds41p1::kSlotBytes) : 0;
    if (!ring_ || n < ds41p1::kMinRingSlots)
        return no("the file buffers hold " + std::to_string(n) + " ring slots of " +
                  std::to_string(ds41p1::kSlotBytes) + " bytes, fewer than " +
                  std::to_string(ds41p1::kMinRingSlots) +
                  " (RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=2 or more: the arm part B2..B8; the window measures B6)");
    alias_ = ring_;
    tab_pin_ = (void**)rad_dev_alloc(2 * kWaveRows * 2 * 96 * (int64_t)sizeof(void*), RAD_MEM_HOST_PINNED);
    tab_dev_ = (void**)rad_dev_alloc(2 * kWaveRows * 2 * 96 * (int64_t)sizeof(void*), RAD_MEM_DEVICE);
    if (!tab_pin_ || !tab_dev_) return no("no memory for the wave tables");
    for (int k = 0; k < t_.ready; ++k) {
        uint8_t* r = (uint8_t*)rad_dev_alloc(ds41p1::kSlotBytes, RAD_MEM_HOST_PINNED);
        if (!r) return no("no pinned memory for " + std::to_string(t_.ready) + " ready slots of " +
                          std::to_string(ds41p1::kSlotBytes) + " bytes");
        ready_.push_back(r);
        ready_free_.push_back(k);
    }
    std::string w;
    if (st_.init(this, ring_, n, t_.wave, 96, alias_, t_.ready, &w) < 0) return no(w);
    stop_ = false;
    reader_ = std::thread(&CardExpertPath::reader_loop, this);
    packer_ = std::thread(&CardExpertPath::packer_loop, this);
    on_ = true;
    if (t_.swiglu_span)
        RAD_INFO("ds41 p1: rank %d's wave SwiGLU spans on (%s=1; known stateless row and synchronized counts, "
                 "full replay otherwise)", rank_, kSwigluSpanEnv);
    return RAD_OK;
}

int CardExpertPath::event_create(bool host) {
    RadEvent e = nullptr;
    const int rc = host ? rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT) : rad_event_create_local(&e);
    if (rc < 0 || !e) return -1;
    ev_.push_back(e);
    ev_host_.push_back(host ? 1 : 0);
    return (int)ev_.size() - 1;
}

int CardExpertPath::copy(int slot, const Source& src, int wait, int done) {
    if (slot < 0 || slot >= st_.n_slots() || done < 0 || (size_t)done >= ev_.size()) return RAD_E_INVAL;
    char* at = ring_ + (int64_t)slot * ds41p1::kSlotBytes;
    const RadCopySeg segs[2] = { { src.gu, at, src.gu_bytes }, { src.dn, at + ds41p1::kDnOff, src.dn_bytes } };
    const RadEvent w = wait >= 0 && (size_t)wait < ev_.size() ? ev_[(size_t)wait] : nullptr;
    return mover_->ring_copy(segs, 2, &w, w ? 1 : 0, ev_[(size_t)done]);
}

int CardExpertPath::compute_wait(int ev) {
    if (!s_ || ev < 0 || (size_t)ev >= ev_.size()) return RAD_E_INVAL;
    return rad_event_wait(s_, ev_[(size_t)ev]);
}
int CardExpertPath::compute_record(int ev) {
    if (!s_ || ev < 0 || (size_t)ev >= ev_.size()) return RAD_E_INVAL;
    return rad_event_record(ev_[(size_t)ev], s_);
}
int CardExpertPath::query(int ev) {
    if (ev < 0 || (size_t)ev >= ev_.size()) return RAD_E_INVAL;
    return rad_event_query(ev_[(size_t)ev]);
}
int CardExpertPath::sync(int ev) {
    if (ev < 0 || (size_t)ev >= ev_.size()) return RAD_E_INVAL;
    return rad_event_sync(ev_[(size_t)ev]);
}

int CardExpertPath::wave(int parity, int w, const void* const* gu_tab, const void* const* dn_tab, int n) {
    if (!s_ || n != 96 || w < 0 || w >= kWaveRows || !captured()) return RAD_E_STATE;
    const int64_t row = ((int64_t)(parity & 1) * kWaveRows + w) * 2 * 96;
    void** pin = tab_pin_ + row;
    void** dev = tab_dev_ + row;
    std::memcpy(pin, gu_tab, 96 * sizeof(void*));
    std::memcpy(pin + 96, dn_tab, 96 * sizeof(void*));
    rad_dev_host_wrote();
    RAD_TRY(rad_memcpy_async(dev, pin, 2 * 96 * (int64_t)sizeof(void*), s_));
    /* gate_up and down with the wave's table in their weight operand. With the span switch, only the
     * known stateless SwiGLU row may narrow: every gap is idempotent and the tail was done by the
     * original launch. The gate_up/down keep their full sorted operands and global offsets. */
    for (int k = 0; k < 3; ++k) {
        Cap& c = cap_[k];
        RadArgs a = c.a;
        std::vector<RadTensor> t = c.t;
        if (k != 1) t[1].data = (void*)(k == 0 ? dev : dev + 96);
        a.t = t.data();
        a.p = c.p.empty() ? nullptr : c.p.data();
        std::vector<RadParam> p;
        if (k == 1 && t_.swiglu_span) {
            SwigluSpan span;
            if (c.t.size() >= 2 && c.t[1].shape[0] > 0) c_.swiglu_full_rows += (uint64_t)c.t[1].shape[0];
            if (c.swiglu_span_ok && layer_counts_ok_ && c.t.size() == 2 &&
                wave_swiglu_span(layer_counts_.data(), n, gu_tab, dn_tab, c.t[1].shape[0], &span) &&
                swiglu_span_view(&a, span, &t, &p, &a)) {
                ++c_.swiglu_spans;
                c_.swiglu_rows += (uint64_t)(span.hi - span.lo);
                if (span.hi == span.lo) continue;   /* the wave has no local rows */
            } else {
                ++c_.swiglu_fallbacks;
                // Count executed rows too, so mixed full/span replay reports actual work.
                if (c.t.size() >= 2 && c.t[1].shape[0] > 0)
                    c_.swiglu_rows += (uint64_t)c.t[1].shape[0];
            }
        }
        const int rc = c.fn(&a, s_);
        if (rc < 0) {
            RAD_ERR("ds41 p1: rank %d's wave %d of routed layer %d: relaunching %s failed (%d)", rank_, w, li_,
                    k == 0 ? "gate_up" : k == 1 ? "the SwiGLU" : "down", rc);
            return rc;
        }
    }
    return RAD_OK;
}

void CardExpertPath::release(Kind k, int32_t handle) {
    if (k == Kind::Pool) return;   /* read in place */
    if (k == Kind::Ram) {
        if (handle < 0 || (size_t)handle >= hu_.size() || !hu_[(size_t)handle]) return;
        mover_->release_host_unit(hu_[(size_t)handle].get());
        hu_free_.push_back(handle);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (handle >= 0 && (size_t)handle < ready_.size()) ready_free_.push_back(handle);
    }
    cv_pack_.notify_all();
}

int CardExpertPath::capture(int which, const RadArgs* a, LaunchFn fn, bool swiglu_span_ok) {
    if (which < 0 || which > 2 || !a || !fn || !pass_) return RAD_E_STATE;
    Cap& c = cap_[which];
    if (c.ok) {
        RAD_ERR("ds41 p1: routed layer %d's %s was launched twice in one pass (a sliced block); the card "
                "path takes a layer's rows in one launch", li_, which == 0 ? "gate_up" : which == 1 ? "SwiGLU" : "down");
        return RAD_E_STATE;
    }
    if (a->scratch_bytes > 0) {
        RAD_ERR("ds41 p1: routed layer %d's %s takes %lld bytes of scratch, which a later relaunch could find "
                "reused; the card path relaunches only scratch-free rows", li_,
                which == 0 ? "gate_up" : which == 1 ? "SwiGLU" : "down", (long long)a->scratch_bytes);
        return RAD_E_STATE;
    }
    if (which != 1 && (a->n_t < 5 || a->t[1].rank < 1 || a->t[1].stride[0] != 0 || !a->t[1].data)) {
        RAD_ERR("ds41 p1: routed layer %d's %s does not read its weights as a table; the card path needs the "
                "table form", li_, which == 0 ? "gate_up" : "down");
        return RAD_E_STATE;
    }
    c.fn = fn;
    c.swiglu_span_ok = which == 1 && swiglu_span_ok && !a->instance;
    c.a = *a;
    c.t.assign(a->t, a->t + a->n_t);
    c.p.assign(a->p, a->p + a->n_p);
    c.a.t = nullptr;
    c.a.p = nullptr;
    c.ok = true;
    return RAD_OK;
}

int CardExpertPath::begin_pass(int64_t rows, const RadEvent* hand_over, int n_hand_over) {
    if (!on_ || pass_) return RAD_E_STATE;
    pass_ = true;
    rows_ = rows;
    li_ = -1;
    t_pass_ = std::chrono::steady_clock::now();
    ++c_.passes;
    {
        std::lock_guard<std::mutex> lk(mu_);
        error_ = 0;
        owed_ = 0;
    }
    for (int i = 0; i < n_hand_over; ++i)
        if (hand_over && hand_over[i]) RAD_TRY(mover_->mover_wait(hand_over[i]));
    return RAD_OK;
}

int CardExpertPath::begin_layer(int32_t li, int parity, const int32_t* counts, int n_counts) {
    if (!pass_) return RAD_E_STATE;
    if (st_.layer_open()) {
        RAD_ERR("ds41 p1: routed layer %d began before layer %d's waves ran", li, li_);
        return RAD_E_STATE;
    }
    li_ = li;
    layer_counts_ok_ = t_.swiglu_span && counts && n_counts >= (int)layer_counts_.size();
    if (layer_counts_ok_) std::copy_n(counts, layer_counts_.size(), layer_counts_.begin());
    for (Cap& c : cap_) c.ok = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        /* Units read ahead for an earlier layer that this one is not: dropped. */
        for (auto it = by_unit_.begin(); it != by_unit_.end();) {
            DUnit& d = *it->second;
            if (d.li != li && !d.demanded) {
                d.dropped = true;
                ++c_.ahead_dropped;
                if (d.state >= 2) d.planes.reset();
                it = by_unit_.erase(it);
            } else {
                ++it;
            }
        }
        for (const Done& d : done_q_) ready_free_.push_back(d.handle);
        done_q_.clear();
        owed_ = 0;
    }
    cv_pack_.notify_all();
    ++c_.layers;
    if (st_.begin_layer(parity) < 0) return RAD_E_STATE;
    return RAD_OK;
}

int CardExpertPath::add_ram(int e, int32_t unit, int rows, bool touch, bool* hit) {
    *hit = false;
    int32_t h;
    if (!hu_free_.empty()) {
        h = hu_free_.back();
        hu_free_.pop_back();
    } else {
        hu_.emplace_back(new Mover::HostUnit());
        h = (int32_t)hu_.size() - 1;
    }
    Mover::HostUnit& hu = *hu_[(size_t)h];
    if (!mover_->borrow_host_unit(unit, &hu, touch)) {
        hu_free_.push_back(h);
        return RAD_OK;
    }
    Source src;
    for (int k = 0; k < hu.n; ++k) {
        const bool gu = hu.bytes[k] == ds41p1::kGuBlockBytes;
        const int64_t n = hu.packed ? hu.slot[k] : hu.bytes[k];
        (gu ? src.gu : src.dn) = hu.dev[k];
        (gu ? src.gu_bytes : src.dn_bytes) = n;
    }
    src.packed = hu.packed;
    if (hu.n != 2 || st_.add_ram(e, unit, rows, src, h) < 0) {
        mover_->release_host_unit(&hu);
        hu_free_.push_back(h);
        RAD_ERR("ds41 p1: routed layer %d's cached unit %d (expert %d) does not fit a ring slot", li_, unit, e);
        return RAD_E_STATE;
    }
    *hit = true;
    if (touch) ++c_.admitted;
    return RAD_OK;
}

int CardExpertPath::add_pool(int e, int32_t unit, int rows, const Source& src) {
    if (st_.add_pool(e, unit, rows, src) < 0) {
        RAD_ERR("ds41 p1: routed layer %d's unit %d (expert %d) in the host pool does not fit a ring slot", li_,
                unit, e);
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int CardExpertPath::add_drive(int e, int32_t unit, int rows, bool admit) {
    if (st_.add_drive(e, unit, rows) < 0) return RAD_E_STATE;
    std::lock_guard<std::mutex> lk(mu_);
    ++owed_;
    auto it = by_unit_.find(unit);
    if (it != by_unit_.end()) {   /* read ahead: landed ones go to the packer now */
        DUnit& d = *it->second;
        d.li = li_;
        d.demanded = true;
        d.admit = admit;
        d.rows = rows;
        ++c_.ahead_used;
        if (d.state == 3) {   /* its read failed: read it again as a demand */
            d.state = 0;
            d.planes.reset();
            read_q_.push_front(it->second);
            cv_read_.notify_all();
        }
        if (d.state == 2 && !d.packing) {
            d.packing = true;
            pack_q_.push_back(it->second);
            cv_pack_.notify_all();
        }
        return RAD_OK;
    }
    auto d = std::make_shared<DUnit>();
    d->unit = unit;
    d->li = li_;
    d->rows = rows;
    d->demanded = true;
    d->admit = admit;
    by_unit_[unit] = d;
    return RAD_OK;
}

int CardExpertPath::start() {
    {
        /* The demand reads in descending rows, ahead of anything read ahead still queued. */
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::shared_ptr<DUnit>> want;
        for (auto& kv : by_unit_)
            if (kv.second->li == li_ && kv.second->demanded && kv.second->state == 0 &&
                !kv.second->planes)
                want.push_back(kv.second);
        std::stable_sort(want.begin(), want.end(),
                         [](const std::shared_ptr<DUnit>& a, const std::shared_ptr<DUnit>& b) { return a->rows > b->rows; });
        for (auto it = want.rbegin(); it != want.rend(); ++it) read_q_.push_front(*it);
        if (!want.empty()) cv_read_.notify_all();
    }
    const int rc = st_.start();
    if (rc < 0) {
        RAD_ERR("ds41 p1: rank %d could not queue routed layer %d's ring copies", rank_, li_);
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int CardExpertPath::read_ahead(int32_t next_li, const std::vector<int32_t>& units) {
    if (t_.ahead <= 0 || units.empty()) return RAD_OK;
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (int32_t u : units) {
        if (n >= t_.ahead) break;
        if (by_unit_.count(u)) continue;
        auto d = std::make_shared<DUnit>();
        d->unit = u;
        d->li = next_li;
        d->rows = (int)rows_;
        by_unit_[u] = d;
        read_q_.push_back(d);
        ++n;
    }
    c_.ahead_posted += (uint64_t)n;
    if (n) cv_read_.notify_all();
    return RAD_OK;
}

int CardExpertPath::run(RadStream s) {
    if (!captured()) {
        RAD_ERR("ds41 p1: routed layer %d's down was issued without its gate_up, SwiGLU and down launches seen",
                li_);
        return RAD_E_STATE;
    }
    s_ = s;
    const int rc = st_.run();
    if (rc < 0) {
        int err;
        {
            std::lock_guard<std::mutex> lk(mu_);
            err = error_;
        }
        RAD_ERR("ds41 p1: rank %d's routed layer %d: the waves did not run%s", rank_, li_,
                err < 0 ? " (a drive read or a pack failed)" : "");
        return err < 0 ? err : RAD_E_STATE;
    }
    return RAD_OK;
}

int CardExpertPath::wait_drive() {
    const auto t0 = std::chrono::steady_clock::now();
    std::deque<Done> got;
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&] { return !done_q_.empty() || error_ < 0 || stop_ || owed_ <= 0; });
        if (error_ < 0) return error_;
        if (done_q_.empty()) return 0;
        got.swap(done_q_);
        owed_ -= (int)got.size();
    }
    c_.wait_drive_ns += ns_since(t0);
    for (const Done& d : got)
        if (st_.drive_ready(d.unit, d.src, d.handle) < 0) {
            release(Kind::Drive, d.handle);
            return RAD_E_STATE;
        }
    return 1;
}

int CardExpertPath::drop_all() {
    std::unique_lock<std::mutex> lk(mu_);
    for (auto& kv : by_unit_) {
        kv.second->dropped = true;
        if (!kv.second->demanded) ++c_.ahead_dropped;
    }
    read_q_.clear();
    pack_q_.clear();
    cv_idle_.wait(lk, [&] { return (!reading_ && !packing_) || stop_; });
    for (auto& kv : by_unit_) kv.second->planes.reset();
    by_unit_.clear();
    for (const Done& d : done_q_) ready_free_.push_back(d.handle);
    done_q_.clear();
    owed_ = 0;
    return error_;
}

int CardExpertPath::quiesce_for_draft_bank() {
    std::lock_guard<std::mutex> lk(mu_);
    if (pass_ || lending() || reading_ || packing_ || owed_ ||
        !read_q_.empty() || !pack_q_.empty() || !done_q_.empty() || !by_unit_.empty())
        return RAD_E_STATE;
    return error_ < 0 ? error_ : RAD_OK;
}

int CardExpertPath::end_pass(RadStream s, RadEvent end) {
    if (!pass_) return RAD_OK;
    int rc = st_.end_pass();
    if (rc < 0) RAD_ERR("ds41 p1: rank %d's pass ended with routed layer %d's waves not run", rank_, li_);
    const int err = drop_all();
    if (err < 0 && rc >= 0) rc = err;
    for (auto& h : hu_)
        if (h && h->n) {
            ++c_.leases_at_end;
            mover_->release_host_unit(h.get());
        }
    hu_free_.clear();
    for (size_t i = 0; i < hu_.size(); ++i) hu_free_.push_back((int32_t)i);
    c_.leases_at_end += (uint64_t)st_.held();
    if (s && end) {
        const int r1 = rad_event_record(end, s);
        const int r2 = r1 < 0 ? r1 : mover_->mover_wait(end);
        if (r2 < 0 && rc >= 0) rc = r2;
    }
    c_.pass_ns += ns_since(t_pass_);
    pass_ = false;
    for (Cap& c : cap_) c.ok = false;
    return rc < 0 ? (rc == -1 ? RAD_E_STATE : rc) : RAD_OK;
}

/* ---- the lent ring's host ---------------------------------------------------------------------------- */
int CardExpertPath::lend_init(const std::vector<LendUnit>& units, const std::vector<int>& ring_slots,
                              const std::vector<char*>& arena_slots, const std::vector<char*>& slab_slots,
                              int64_t grace_ms, std::string* why) {
    auto no = [&](const std::string& w) { if (why) *why = w; return RAD_E_INVAL; };
    if (!on_ || lending()) return no("the card path is not on, or lends already");
    const int n = (int)units.size(), nr = (int)ring_slots.size(), na = (int)arena_slots.size();
    const int ns = (int)slab_slots.size();
    if (n < 1 || nr > ds41p1::kMaxLend || na > ds41p1::kMaxLend || ns > ds41p1::kMaxLendSlab || nr + na + ns != n)
        return no("a lent ring of " + std::to_string(nr) + " slots, " + std::to_string(na) + " arena slots and " +
                  std::to_string(ns) + " slab slots for " + std::to_string(n) + " units");
    std::vector<uint8_t> used((size_t)st_.n_slots(), 0);
    std::vector<char*> at((size_t)n, nullptr);
    for (int k = 0; k < n; ++k) {
        if (k < nr) {
            const int r = ring_slots[(size_t)k];
            if (r < 0 || r >= st_.n_slots() || used[(size_t)r]++) return no("lent ring slot " + std::to_string(r));
            at[(size_t)k] = ring_ + (int64_t)r * ds41p1::kSlotBytes;
        } else {
            /* An arena or slab slot: aligned, and clear of the ring and of every other lent slot. */
            const bool slab = k >= nr + na;
            char* const a = slab ? slab_slots[(size_t)(k - nr - na)] : arena_slots[(size_t)(k - nr)];
            const char* const ring_end = ring_ + (int64_t)st_.n_slots() * ds41p1::kSlotBytes;
            bool clear = a && ((uintptr_t)a & 4095) == 0 && (a + ds41p1::kSlotBytes <= ring_ || a >= ring_end);
            for (int j = nr; clear && j < k; ++j)
                clear = a + ds41p1::kSlotBytes <= at[(size_t)j] || at[(size_t)j] + ds41p1::kSlotBytes <= a;
            if (!clear)
                return no(std::string(slab ? "lent slab slot " : "lent arena slot ") +
                          std::to_string(slab ? k - nr - na : k - nr) + " overlaps the ring or another slot");
            at[(size_t)k] = a;
        }
        const LendUnit& u = units[(size_t)k];
        for (int i = 0; i < 2; ++i)
            if (u.off[i] < 0 || u.bytes[i] <= 0 || u.off[i] + u.bytes[i] > ds41p1::kSlotBytes)
                return no("lent unit " + std::to_string(u.unit) + " does not fit a ring slot");
    }
    lu_ = units;
    lat_ = std::move(at);
    lring_ = ring_slots;
    lev_.assign((size_t)n, -1);
    lsrc_.assign((size_t)n, -2);
    lhu_.clear();
    for (int k = 0; k < n; ++k) {
        lhu_.emplace_back(new Mover::HostUnit());
        lev_[(size_t)k] = event_create(true);
        if (lev_[(size_t)k] < 0) return no("could not create the lent ring's events");
    }
    for (int b = 0; b < 2; ++b) {
        uint8_t* buf = (uint8_t*)rad_dev_alloc(ds41p1::kSlotBytes, RAD_MEM_HOST_PINNED);
        if (!buf) return no("no pinned memory for the lent ring's two reload buffers");
        lbuf_.push_back(buf);
        lplanes_.emplace_back(new Mover::HostPlanes());
        lbuf_free_.push_back(b);
    }
    if (lend_.init(this, n, grace_ms, ns) < 0) return no("the lent ring's state");
    lstop_ = false;
    lth_ = std::thread(&CardExpertPath::lend_loop, this);
    return RAD_OK;
}

/* The slab's units before the first pass. Nothing reads or writes the slots yet, so each step is the
 * lender's own (a load, its copy on the DMA lane, the publish once the host has seen the copy pass); the
 * wait between steps is this thread's alone. */
int CardExpertPath::lend_prime(int64_t most_ms) {
    if (!lending() || lend_.kept() < 1) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (lend_.prime() > 0) {
        if ((int64_t)(ns_since(t0) / 1000000) >= most_ms) break;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    int in = 0;
    for (int k = lend_.n() - lend_.kept(); k < lend_.n(); ++k) in += lend_.resident(k);
    lprimed_ = in;
    return in;
}

/* The reload thread: a unit's two extents read whole from the container (Mover::read_piece, beside the
 * kept readers), relaid into the reload buffer in the unit's layout (Mover::relay_unit_block). */
void CardExpertPath::lend_loop() {
    for (;;) {
        std::pair<int, int> job;
        {
            std::unique_lock<std::mutex> lk(lmu_);
            lcv_.wait(lk, [&] { return lstop_ || !lq_.empty(); });
            if (lstop_) return;
            job = lq_.front();
            lq_.pop_front();
        }
        const int k = job.first, b = job.second;
        Mover::HostPlanes& p = *lplanes_[(size_t)b];
        p.unit = lu_[(size_t)k].unit;
        const auto t0 = std::chrono::steady_clock::now();
        PlaneExtent ext[2];
        bool ok = mover_->lay_out_unit_planes(&p, ext) && mover_->source_epochs_current();
        for (int i = 0; ok && i < 2; ++i) ok = mover_->read_piece(ext[i].file, ext[i].off, ext[i].len, ext[i].dst, ext[i].cap) >= 0;
        ok = ok && mover_->relay_unit_block(p, lbuf_[(size_t)b], ds41p1::kSlotBytes) >= 0;
        const uint64_t took = ns_since(t0);
        {
            std::lock_guard<std::mutex> lk(lmu_);
            ldone_.push_back({ k, ok });
            ++c_.lend_drive;
            c_.lend_drive_ns += took;
            c_.lend_read_failed += !ok;
        }
    }
}

void CardExpertPath::lend_stop() {
    {
        std::lock_guard<std::mutex> lk(lmu_);
        lstop_ = true;
    }
    lcv_.notify_all();
    if (lth_.joinable()) lth_.join();
    /* A copy still in flight reads its source until its event passes. */
    for (size_t k = 0; k < lsrc_.size(); ++k) {
        if (lsrc_[k] == -2) continue;
        if (lev_[k] >= 0 && (size_t)lev_[k] < ev_.size() && ev_[(size_t)lev_[k]]) (void)rad_event_sync(ev_[(size_t)lev_[k]]);
        if (lsrc_[k] == -1 && mover_) mover_->release_host_unit(lhu_[k].get());
        lsrc_[k] = -2;
    }
    for (uint8_t* b : lbuf_)
        if (b) rad_dev_free(b, RAD_MEM_HOST_PINNED);
    lbuf_.clear();
}

int CardExpertPath::load(int k) {
    if (k < 0 || k >= (int)lu_.size() || lsrc_[(size_t)k] != -2) return -1;
    /* The RAM cache in the card's block form: its lease is the source, landed at once. */
    if (!mover_->packed_cache()) {
        Mover::HostUnit& hu = *lhu_[(size_t)k];
        if (mover_->borrow_host_unit(lu_[(size_t)k].unit, &hu, false)) {
            bool fit = hu.n == 2 && !hu.packed;
            for (int i = 0; fit && i < 2; ++i) fit = hu.dev[i] && hu.bytes[i] == lu_[(size_t)k].bytes[i];
            if (fit) {
                lsrc_[(size_t)k] = -1;
                std::lock_guard<std::mutex> lk(lmu_);
                ldone_.push_back({ k, true });
                ++c_.lend_ram;
                return 1;
            }
            mover_->release_host_unit(&hu);
        }
    }
    std::lock_guard<std::mutex> lk(lmu_);
    if (lbuf_free_.empty()) return 0;
    const int b = lbuf_free_.back();
    lbuf_free_.pop_back();
    lsrc_[(size_t)k] = b;
    lq_.push_back({ k, b });
    lcv_.notify_all();
    return 1;
}

int CardExpertPath::landed(bool* ok) {
    std::lock_guard<std::mutex> lk(lmu_);
    if (ldone_.empty()) return -1;
    const std::pair<int, bool> d = ldone_.front();
    ldone_.pop_front();
    *ok = d.second;
    return d.first;
}

int CardExpertPath::copy(int k) {
    if (k < 0 || k >= (int)lu_.size()) return RAD_E_INVAL;
    char* const at = lat_[(size_t)k];
    const LendUnit& u = lu_[(size_t)k];
    RadCopySeg segs[2];
    int n = 0;
    if (lsrc_[(size_t)k] == -1) {   /* the lease, weight by weight in the unit's order */
        const Mover::HostUnit& hu = *lhu_[(size_t)k];
        for (int i = 0; i < 2; ++i) segs[n++] = { hu.dev[i], at + u.off[i], hu.bytes[i] };
    } else if (lsrc_[(size_t)k] >= 0) {   /* the reload buffer, laid out as the slot */
        segs[n++] = { lbuf_[(size_t)lsrc_[(size_t)k]], at, ds41p1::kSlotBytes };
    } else {
        return RAD_E_STATE;
    }
    return mover_->ring_copy(segs, n, nullptr, 0, ev_[(size_t)lev_[(size_t)k]]);
}

int CardExpertPath::copied(int k) {
    if (k < 0 || k >= (int)lu_.size()) return RAD_E_INVAL;
    return rad_event_query(ev_[(size_t)lev_[(size_t)k]]);
}

void CardExpertPath::drop(int k) {
    if (k < 0 || k >= (int)lu_.size()) return;
    const int src = lsrc_[(size_t)k];
    lsrc_[(size_t)k] = -2;
    if (src == -1) {
        mover_->release_host_unit(lhu_[(size_t)k].get());
    } else if (src >= 0) {
        std::lock_guard<std::mutex> lk(lmu_);
        lbuf_free_.push_back(src);
    }
}

void CardExpertPath::publish(int k, bool on) {
    if (k < 0 || k >= (int)lu_.size()) return;
    mover_->set_ring_resident(lu_[(size_t)k].unit, on ? lat_[(size_t)k] : nullptr);
}

/* A ring slot's next writer is the card path's own copy, behind this one on the DMA lane; an arena
 * slot's is the pass's ops, on the compute stream, which wait for the copy here (or, if the stream
 * cannot take the wait, the host waits for it before the pass issues anything). */
void CardExpertPath::fence(int k) {
    if (k < (int)lring_.size() || k >= (int)lu_.size()) return;
    const int e = lev_[(size_t)k];
    if (e < 0 || (size_t)e >= ev_.size()) return;
    const RadStream cs = mover_->compute_stream();
    if ((cs && rad_event_wait(cs, ev_[(size_t)e]) >= 0) || rad_event_sync(ev_[(size_t)e]) >= 0) return;
    RAD_ERR("ds41 p1: rank %d: neither the compute stream nor the host could wait on lent arena slot %d's reload copy",
            rank_, k);
}

/* ---- the lent slots' heat: the retargets ---------------------------------------------------------------- */
int CardExpertPath::lend_retarget(const int* ks, const LendUnit* units, int n, RadEvent behind) {
    if (!lending() || !ks || !units || n < 1) return 0;
    if (!lend_.settled()) return lend_.retarget(ks, n);   /* refused there, with the reason counted */
    bool ok = n <= ds41p1::kMaxLendHeat;
    for (int i = 0; ok && i < n; ++i) {
        const int k = ks[i];
        const LendUnit& u = units[i];
        ok = k >= 0 && k < (int)lu_.size() && lsrc_[(size_t)k] == -2 && u.unit >= 0;
        for (int j = 0; ok && j < i; ++j) ok = ks[j] != k && units[j].unit != u.unit;
        for (size_t j = 0; ok && j < lu_.size(); ++j) ok = lu_[j].unit != u.unit;   /* no unit in two slots */
        for (int w = 0; ok && w < 2; ++w)
            ok = u.off[w] >= 0 && u.bytes[w] > 0 && u.off[w] + u.bytes[w] <= ds41p1::kSlotBytes;
    }
    if (!ok) {
        lend_.decline(RingLender::Decline::Candidate);
        return 0;
    }
    if (!behind) {
        lend_.decline(RingLender::Decline::Fence);
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        lnext_k_[i] = ks[i];
        lnext_[i] = units[i];
    }
    lnext_n_ = n;
    lfence_ = behind;
    lfenced_ = false;
    const int done = lend_.retarget(ks, n);
    lnext_n_ = 0;
    lfence_ = nullptr;
    return done;
}

/* The same hand-over an armed pass's first ring copy gets (begin_pass's Mover::mover_wait on the residents'
 * last-reader event): the mover's stream is the one queue every slot copy goes through, so with this wait
 * queued now, a copy queued at any later pass runs behind the event, which the compute stream records at
 * the end of the pass before this one, after the last kernel that could read slot k's resident. */
int CardExpertPath::retarget_fence(int k) {
    if (k < 0 || k >= (int)lu_.size() || !lfence_) return RAD_E_STATE;
    if (lfenced_) return RAD_OK;
    RAD_TRY(mover_->mover_wait(lfence_));
    lfenced_ = true;
    return RAD_OK;
}

void CardExpertPath::retarget(int k) {
    for (int i = 0; i < lnext_n_; ++i)
        if (lnext_k_[i] == k) {
            lu_[(size_t)k] = lnext_[i];
            return;
        }
}

std::vector<char*> arena_slots(char* lo, int64_t bytes, int max) {
    std::vector<char*> out;
    if (!lo || bytes <= 0 || max <= 0) return out;
    const uintptr_t b = ((uintptr_t)lo + 4095) & ~(uintptr_t)4095;
    uintptr_t top = ((uintptr_t)lo + (uintptr_t)bytes) & ~(uintptr_t)4095;
    while ((int)out.size() < max && top >= b + (uintptr_t)ds41p1::kSlotBytes) {
        top -= (uintptr_t)ds41p1::kSlotBytes;
        out.push_back((char*)top);
    }
    return out;
}

/* ---- the drive's pipeline ---------------------------------------------------------------------------- */
void CardExpertPath::landed(void* ctx, int i) {
    LandCtx& c = *(LandCtx*)ctx;
    CardExpertPath& self = *c.path;
    const std::shared_ptr<DUnit>& d = c.units[(size_t)i];
    c.done[(size_t)i].store(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(self.mu_);
    d->state = 2;
    if (d->demanded && !d->dropped && !d->packing) {
        d->packing = true;
        self.pack_q_.push_back(d);
        self.cv_pack_.notify_all();
    }
}

void CardExpertPath::reader_loop() {
    for (;;) {
        std::vector<std::shared_ptr<DUnit>> batch;
        int64_t rows = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_read_.wait(lk, [&] { return stop_ || !read_q_.empty(); });
            if (stop_) return;
            while (!read_q_.empty() && (int)batch.size() < t_.inflight) {
                auto d = read_q_.front();
                read_q_.pop_front();
                if (d->dropped || d->state != 0) continue;
                d->state = 1;
                batch.push_back(d);
            }
            if (batch.empty()) continue;
            reading_ = true;
            rows = rows_;
        }
        /* Planes from the file stager's pool (outside the lock: it takes its own). */
        LandCtx ctx;
        ctx.path = this;
        ctx.done.reset(new std::atomic<uint8_t>[batch.size()]);
        std::vector<Mover::HostPlanes*> ptrs;
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i]->planes = planes_(batch[i]->unit);
            ptrs.push_back(batch[i]->planes.get());
            ctx.units.push_back(batch[i]);
            ctx.done[i].store(0, std::memory_order_relaxed);
        }
        const auto t0 = std::chrono::steady_clock::now();
        PieceReader& demand = PieceReader::process();
        demand.demand_enter();
        const bool ok = mover_->read_unit_planes(ptrs.data(), (int)ptrs.size(), (int)rows, nullptr, nullptr,
                                                 false, nullptr, &CardExpertPath::landed, &ctx);
        demand.demand_leave();
        const uint64_t busy = ns_since(t0);
        {
            std::lock_guard<std::mutex> lk(mu_);
            c_.drive_busy_ns += busy;
            for (size_t i = 0; i < batch.size(); ++i) {
                DUnit& d = *batch[i];
                ++c_.drive_units;
                c_.drive_bytes += (uint64_t)ds41p1::kSlotBytes;
                if (ctx.done[i].load(std::memory_order_relaxed)) continue;
                if (!ok) {
                    d.state = 3;
                    if (d.demanded && !d.dropped) error_ = RAD_E_IO;
                    d.planes.reset();
                    continue;
                }
                /* A leg read again after the readers: landed now. */
                d.state = 2;
                if (d.demanded && !d.dropped && !d.packing) {
                    d.packing = true;
                    pack_q_.push_back(batch[i]);
                }
            }
            for (auto& d : batch)
                if (d->dropped && d->state >= 2) d->planes.reset();
            reading_ = false;
        }
        cv_pack_.notify_all();
        cv_done_.notify_all();
        cv_idle_.notify_all();
    }
}

void CardExpertPath::packer_loop() {
    for (;;) {
        std::shared_ptr<DUnit> d;
        int32_t k = -1;
        {
            std::unique_lock<std::mutex> lk(mu_);
            const auto t0 = std::chrono::steady_clock::now();
            bool waited = false;
            for (;;) {
                if (stop_) return;
                while (!pack_q_.empty() && pack_q_.front()->dropped) {
                    pack_q_.front()->planes.reset();
                    pack_q_.pop_front();
                }
                if (!pack_q_.empty() && !ready_free_.empty()) break;
                if (!pack_q_.empty()) waited = true;
                cv_pack_.wait(lk);
            }
            if (waited) c_.ready_wait_ns += ns_since(t0);
            d = pack_q_.front();
            pack_q_.pop_front();
            k = ready_free_.back();
            ready_free_.pop_back();
            packing_ = true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool packed = false;
        int64_t bytes[2] = { 0, 0 };
        const int rc = d->planes ? mover_->pack_unit_for_ring(*d->planes, ready_[(size_t)k], ds41p1::kSlotBytes,
                                                              &packed, bytes)
                                 : RAD_E_STATE;
        const uint64_t took = ns_since(t0);
        std::shared_ptr<Mover::HostPlanes> planes;
        bool admit = false;
        int rows = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            c_.pack_ns += took;
            if (rc < 0 || d->dropped) {
                ready_free_.push_back(k);
                if (rc < 0 && !d->dropped) {
                    error_ = rc;
                    RAD_ERR("ds41 p1: rank %d could not pack missed unit %d for the ring (%d)", rank_, d->unit, rc);
                }
            } else {
                ++(packed ? c_.packed_units : c_.block_units);
                Done x;
                x.unit = d->unit;
                x.handle = k;
                x.src.gu = ready_[(size_t)k];
                x.src.gu_bytes = bytes[0];
                x.src.dn = ready_[(size_t)k] + ds41p1::kDnOff;
                x.src.dn_bytes = bytes[1];
                x.src.packed = packed;
                done_q_.push_back(x);
                admit = d->admit;
                rows = d->rows;
            }
            planes = std::move(d->planes);
            d->state = 4;
            auto it = by_unit_.find(d->unit);
            if (it != by_unit_.end() && it->second == d) by_unit_.erase(it);
            packing_ = false;
        }
        /* The planes to the RAM cache's insert threads when the prompt's last rows select the unit,
         * else back to the pool. */
        if (planes && admit && rc >= 0 && insert_) insert_(std::move(planes), rows);
        planes.reset();
        cv_done_.notify_all();
        cv_idle_.notify_all();
    }
}

std::string CardExpertPath::stats_line(int rank) const {
    const CardExpertStager::Stats& s = st_.stats();
    const double pass_s = (double)c_.pass_ns * 1e-9;
    char buf[1536];
    std::snprintf(buf, sizeof buf,
                  "%s %d: %" PRIu64 " chunks, %" PRIu64 " card-path layers; units RAM %" PRIu64 " / drive %" PRIu64
                  " / host pool %" PRIu64 " (read ahead: %" PRIu64 " posted, %" PRIu64 " used, %" PRIu64 " dropped); %" PRIu64
                  " waves (at most %" PRIu64 " a layer), %" PRIu64 " ring copies, %.2f GB over DMA; drive %" PRIu64
                  " units, %.2f GB, busy %.1f%% of %.2f s in chunks; issuing thread: %.3f s waiting for drive units (%" PRIu64
                  " waits), %" PRIu64 " ring-full waves, %" PRIu64 " release syncs; packer %.2f ms a unit (%" PRIu64
                  " packed, %" PRIu64 " block), %.3f s waiting for ready slots; admitted %" PRIu64 "; leases at pass end %" PRIu64
                  "; verify rows %" PRIu64 ", differ %" PRIu64 " (%" PRIu64 " samples skipped)",
                  ds41p1::kLogStats, rank, c_.passes, s.layers, s.ram, s.drive, s.pool, c_.ahead_posted, c_.ahead_used,
                  c_.ahead_dropped, s.waves, s.max_waves, s.copies, (double)s.copy_bytes * 1e-9, c_.drive_units,
                  (double)c_.drive_bytes * 1e-9, pass_s > 0 ? 100.0 * (double)c_.drive_busy_ns * 1e-9 / pass_s : 0.0,
                  pass_s, (double)c_.wait_drive_ns * 1e-9, s.drive_waits, s.ring_full_waits, s.release_syncs,
                  (c_.packed_units + c_.block_units) ? (double)c_.pack_ns * 1e-6 / (double)(c_.packed_units + c_.block_units) : 0.0,
                  c_.packed_units, c_.block_units, (double)c_.ready_wait_ns * 1e-9, c_.admitted, c_.leases_at_end,
                  c_.verified, c_.differ, c_.verify_skipped);
    std::string line = buf;
    if (t_.swiglu_span) {
        std::snprintf(buf, sizeof buf, "; wave SwiGLU spans %" PRIu64 ", rows %" PRIu64 " of %" PRIu64
                      ", full-replay fallbacks %" PRIu64, c_.swiglu_spans, c_.swiglu_rows,
                      c_.swiglu_full_rows, c_.swiglu_fallbacks);
        line += buf;
    }
    if (lending()) {
        const RingLender::Stats& l = lend_.stats();
        /* The ring's and the arena's slots here; the slab's (kept) in their own part below. */
        int slab_in = 0;
        for (int k = lend_.n() - lend_.kept(); k < lend_.n(); ++k) slab_in += lend_.resident(k);
        std::snprintf(buf, sizeof buf,
                      "; lent ring: %d slots (%d in the activation arena), %d residents now; %" PRIu64 " bursts took them back (%" PRIu64
                      " residents withdrawn), %" PRIu64 " returns, %" PRIu64 " reloaded (RAM cache %" PRIu64 ", drive %" PRIu64
                      " at %.1f ms a unit), %" PRIu64 " failed, %" PRIu64 " overtaken, %" PRIu64 " passes held back by the grace",
                      lend_.n() - lend_.kept(), lend_arena_slots(), lend_.residents() - slab_in, l.lends, l.withdrawn, l.returns,
                      l.reloaded, c_.lend_ram, c_.lend_drive,
                      c_.lend_drive ? (double)c_.lend_drive_ns * 1e-6 / (double)c_.lend_drive : 0.0, l.failed, l.cancelled,
                      l.held_back);
        line += buf;
        if (lend_.kept() > 0) {
            std::snprintf(buf, sizeof buf, "; lent slab: %d slots, %d residents now (%d at bring-up)", lend_.kept(),
                          slab_in, lprimed_);
            line += buf;
        }
        if (t_.lend_heat > 0) {
            uint64_t declined = 0;
            for (uint64_t d : l.declined) declined += d;
            std::snprintf(buf, sizeof buf,
                          "; lent heat: %" PRIu64 " retargets (units published at a slot another unit gave up), %" PRIu64
                          " residents withdrawn for them, %" PRIu64 " passes declined (%" PRIu64 " lent or in the grace, %" PRIu64
                          " a slot busy, %" PRIu64 " draft bank, %" PRIu64 " candidate, %" PRIu64 " no fence)",
                          l.retargets, l.retarget_withdrawn, declined, l.declined[(size_t)RingLender::Decline::Lent],
                          l.declined[(size_t)RingLender::Decline::Busy], l.declined[(size_t)RingLender::Decline::Bank],
                          l.declined[(size_t)RingLender::Decline::Candidate],
                          l.declined[(size_t)RingLender::Decline::Fence]);
            line += buf;
        }
    }
    return line;
}

}  /* namespace p1 */
}  /* namespace rad */

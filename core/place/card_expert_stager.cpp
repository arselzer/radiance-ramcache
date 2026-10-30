/* card_expert_stager.cpp -- see card_expert_stager.h. */
#include "place/card_expert_stager.h"
#include "rad_internal.h"
#include "device/device.h"
#include "place/readahead.h"

#include <algorithm>
#include <cinttypes>
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
        if (*p < '0' || *p > '9' || n > 1000000) return false;
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
    };
    for (const K& k : ks) {
        const char* v = get(k.env);
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
    *t = r;
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

int CardExpertStager::add_ram(int e, int32_t unit, int rows, const Source& s, int32_t handle) {
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
    u.kind = Kind::Ram;
    u.st = St::Ready;
    u.src = s;
    u_.push_back(u);
    ++st_.ram;
    return 0;
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
        if (u_[i].kind == Kind::Ram) ram_order_.push_back((int)i);
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
    sl.held = true;
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
 * at least half full (they hold the drive's pipeline), else the RAM units in descending rows, then the
 * drive units in the order they were packed. */
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
        if (u.st == St::Ready && u.handle >= 0) {
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

/* ---- the engine's side ------------------------------------------------------------------------------ */
namespace {
uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
        .count();
}
constexpr int kWaveRows = 96;   /* wave table rows a parity: a wave a touched expert at most */
}  /* namespace */

CardExpertPath::~CardExpertPath() {
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
    /* gate_up and down with the wave's table in their weight operand, the SwiGLU as captured. */
    for (int k = 0; k < 3; ++k) {
        Cap& c = cap_[k];
        RadArgs a = c.a;
        std::vector<RadTensor> t = c.t;
        if (k != 1) t[1].data = (void*)(k == 0 ? dev : dev + 96);
        a.t = t.data();
        a.p = c.p.empty() ? nullptr : c.p.data();
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

int CardExpertPath::capture(int which, const RadArgs* a, LaunchFn fn) {
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

int CardExpertPath::begin_layer(int32_t li, int parity) {
    if (!pass_) return RAD_E_STATE;
    if (st_.layer_open()) {
        RAD_ERR("ds41 p1: routed layer %d began before layer %d's waves ran", li, li_);
        return RAD_E_STATE;
    }
    li_ = li;
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
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "%s %d: %" PRIu64 " chunks, %" PRIu64 " card-path layers; units RAM %" PRIu64 " / drive %" PRIu64
                  " (read ahead: %" PRIu64 " posted, %" PRIu64 " used, %" PRIu64 " dropped); %" PRIu64
                  " waves (at most %" PRIu64 " a layer), %" PRIu64 " ring copies, %.2f GB over DMA; drive %" PRIu64
                  " units, %.2f GB, busy %.1f%% of %.2f s in chunks; issuing thread: %.3f s waiting for drive units (%" PRIu64
                  " waits), %" PRIu64 " ring-full waves, %" PRIu64 " release syncs; packer %.2f ms a unit (%" PRIu64
                  " packed, %" PRIu64 " block), %.3f s waiting for ready slots; admitted %" PRIu64 "; leases at pass end %" PRIu64
                  "; verify rows %" PRIu64 ", differ %" PRIu64 " (%" PRIu64 " samples skipped)",
                  ds41p1::kLogStats, rank, c_.passes, s.layers, s.ram, s.drive, c_.ahead_posted, c_.ahead_used,
                  c_.ahead_dropped, s.waves, s.max_waves, s.copies, (double)s.copy_bytes * 1e-9, c_.drive_units,
                  (double)c_.drive_bytes * 1e-9, pass_s > 0 ? 100.0 * (double)c_.drive_busy_ns * 1e-9 / pass_s : 0.0,
                  pass_s, (double)c_.wait_drive_ns * 1e-9, s.drive_waits, s.ring_full_waits, s.release_syncs,
                  (c_.packed_units + c_.block_units) ? (double)c_.pack_ns * 1e-6 / (double)(c_.packed_units + c_.block_units) : 0.0,
                  c_.packed_units, c_.block_units, (double)c_.ready_wait_ns * 1e-9, c_.admitted, c_.leases_at_end,
                  c_.verified, c_.differ, c_.verify_skipped);
    return buf;
}

}  /* namespace p1 */
}  /* namespace rad */

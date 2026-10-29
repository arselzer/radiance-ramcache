/* hx_side_store.h -- RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=N: the planes of the routed units the
 * packed RAM cache can never hold, kept so they stop being misses (FileStager's CPU path).
 *
 * ---- why --------------------------------------------------------------------------------------
 *
 * The packed RAM cache (RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE) gives each weight's raw rounds a
 * fixed room (512 / 320). 7 of the release's 15,360 routed experts have more, so the insert behind
 * each of their misses is refused (Mover::insert_unit_packed) and they stay misses for good: 11-26
 * refused inserts a rank over the eight held-out prompts (6 distinct units) and 0.107 a token over
 * the 256-token chat runs, each a ~2.7 ms drive read the layer waits for. The planes such a miss
 * was read into are exactly what the pool computes the unit from, so the store keeps them (at most
 * N units a rank, ~18 MiB each: a gate_up's and a down's codes and scales, 2 MiB-rounded) and later
 * passes hand them to the pool as a unit read ahead whole is handed: no drive read, the same bytes.
 *
 * ---- rules ------------------------------------------------------------------------------------
 *
 *   in       only the planes of an insert the packed cache refused over its raw-round budget
 *            (FileStager::hx_bg_loop, Mover::packed_refused), whole (both weights read): a refusal
 *            is the unit's for good (its scales decide it), so nothing ever leaves the store
 *   held     never written again: read_unit_planes writes only the planes a layer reads into, and
 *            the free list (FileStager::hx_planes_get) never hands out a HostPlanes still held
 *   out      to the layer's urgent jobs, read-only, after the container's epoch is checked, as for
 *            a unit read ahead whole; not queued for an insert again (it would be refused again)
 *   read-ahead  a held unit counts as cached: none is planned for it (hx_ra_plan, hx_pf_target)
 *
 * Thread-safe: the insert threads offer, the issuing thread finds, the predictor's thread asks
 * holds() while it plans a read-ahead. */
#pragma once
#include "mover.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rad {

class HxSideStore {
public:
    using Planes = std::shared_ptr<Mover::HostPlanes>;

    /* N units at most (0: off), set while no other thread uses the store. */
    void configure(int cap) { cap_.store(cap > 0 ? cap : 0, std::memory_order_relaxed); }
    bool on() const { return cap() > 0; }
    int  cap() const { return cap_.load(std::memory_order_relaxed); }

    /* Keeps p, a unit's two weights' planes, unless the store is full or holds the unit already;
     * true when kept. */
    bool offer(const Planes& p) {
        if (!p || !on() || p->n != 2 || p->unit < 0) return false;
        std::lock_guard<std::mutex> lk(mu_);
        for (const Planes& h : held_)
            if (h->unit == p->unit) return false;
        if ((int)held_.size() >= cap()) {
            ++full_;
            return false;
        }
        held_.push_back(p);
        bytes_ += p->cap[0] + p->cap[1];
        return true;
    }
    /* Unit u's planes, or null. */
    Planes find(int32_t u) const {
        if (!on()) return nullptr;
        std::lock_guard<std::mutex> lk(mu_);
        for (const Planes& h : held_)
            if (h->unit == u) return h;
        return nullptr;
    }
    bool holds(int32_t u) const { return find(u) != nullptr; }
    /* Lets every held unit go (their planes go back to their owner's free list once the layers
     * holding them are done). */
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        held_.clear();
        bytes_ = 0;
    }

    /* Units held, their buffers' bytes, refused units turned away for want of room, the room. */
    struct Stats { int units = 0; int64_t bytes = 0; uint64_t full = 0; int cap = 0; };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        return Stats{ (int)held_.size(), bytes_, full_, cap() };
    }

private:
    mutable std::mutex mu_;
    std::atomic<int> cap_{ 0 };
    std::vector<Planes> held_;
    int64_t  bytes_ = 0;
    uint64_t full_ = 0;
};

}  /* namespace rad */

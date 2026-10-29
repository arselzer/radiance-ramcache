/* hx_side_store_test.cpp -- the side store of RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE
 * (core/place/hx_side_store.h): it keeps a unit's planes up to its room and counts what it turns
 * away, keeps a unit once, refuses half a unit, finds what it holds and only that, gives the planes
 * back to their owner's free list only once nobody holds them, and stays consistent with insert
 * threads offering while the issuing and predictor threads look. */
#include "rad_test.h"

#include "place/hx_side_store.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using rad::HxSideStore;
using rad::Mover;

namespace {

/* Planes as FileStager::hx_planes_get hands them out: the deleter returns them to a free list. */
struct FreeList {
    std::mutex mu;
    std::vector<std::unique_ptr<Mover::HostPlanes>> free;
    std::atomic<int> returned{ 0 };
    HxSideStore::Planes get(int32_t unit, int n = 2, int64_t cap0 = 12 << 20, int64_t cap1 = 6 << 20) {
        auto* p = new Mover::HostPlanes();
        p->unit = unit;
        p->n = n;
        p->cap[0] = cap0;   /* no buffers: the sizes the store counts */
        p->cap[1] = cap1;
        return HxSideStore::Planes(p, [this](Mover::HostPlanes* q) {
            std::lock_guard<std::mutex> lk(mu);
            free.emplace_back(q);
            ++returned;
        });
    }
};

}  /* namespace */

TEST(side_store_keeps_up_to_its_room_once_a_unit) {
    FreeList fl;
    HxSideStore s;
    CHECK(!s.on());
    CHECK(!s.offer(fl.get(1)));   /* off: nothing kept */
    s.configure(2);
    CHECK(s.on());
    CHECK_EQ(s.cap(), 2);
    CHECK(s.offer(fl.get(7)));
    CHECK(!s.offer(fl.get(7)));   /* held already: the second copy is not kept */
    CHECK(!s.offer(fl.get(8, 1)));   /* half a unit */
    CHECK(!s.offer(fl.get(-1)));
    CHECK(!s.offer(nullptr));
    CHECK(s.offer(fl.get(9)));
    CHECK(!s.offer(fl.get(10)));  /* full */
    CHECK(!s.offer(fl.get(11)));
    const HxSideStore::Stats st = s.stats();
    CHECK_EQ(st.units, 2);
    CHECK_EQ(st.cap, 2);
    CHECK_EQ(st.full, (uint64_t)2);
    CHECK_EQ(st.bytes, (int64_t)2 * (18 << 20));
    CHECK(s.holds(7) && s.holds(9));
    CHECK(!s.holds(8) && !s.holds(10) && !s.holds(1));
    CHECK(s.find(7) && s.find(7)->unit == 7);
    CHECK(s.find(10) == nullptr);
    /* Everything not kept went back to the free list at once: 1, the second 7, 8, -1, 10, 11. */
    CHECK_EQ(fl.returned.load(), 6);
}

TEST(side_store_planes_go_back_only_once_nobody_holds_them) {
    FreeList fl;
    HxSideStore s;
    s.configure(4);
    {
        HxSideStore::Planes p = fl.get(3);
        CHECK(s.offer(p));
    }
    CHECK_EQ(fl.returned.load(), 0);   /* the store holds them */
    HxSideStore::Planes in_use = s.find(3);   /* a layer's jobs */
    s.clear();
    CHECK(!s.holds(3));
    CHECK_EQ(s.stats().units, 0);
    CHECK_EQ(s.stats().bytes, (int64_t)0);
    CHECK_EQ(fl.returned.load(), 0);   /* still the layer's */
    in_use.reset();
    CHECK_EQ(fl.returned.load(), 1);
    s.configure(0);
    CHECK(!s.on());
    CHECK(s.find(3) == nullptr);
}

TEST(side_store_offers_and_lookups_from_many_threads) {
    FreeList fl;
    HxSideStore s;
    s.configure(8);
    std::atomic<bool> go{ false };
    std::atomic<int> kept{ 0 }, bad{ 0 };
    std::vector<std::thread> t;
    for (int w = 0; w < 4; ++w)   /* insert threads: units 0..15, each offered by two threads */
        t.emplace_back([&, w] {
            while (!go.load()) std::this_thread::yield();
            for (int u = 0; u < 16; ++u)
                if ((u + w) % 2 == 0 && s.offer(fl.get(u))) ++kept;
        });
    for (int w = 0; w < 2; ++w)   /* the issuing and predictor threads */
        t.emplace_back([&] {
            while (!go.load()) std::this_thread::yield();
            for (int i = 0; i < 20000; ++i) {
                const int u = i % 16;
                HxSideStore::Planes p = s.find(u);
                if (p && p->unit != u) ++bad;
            }
        });
    go = true;
    for (std::thread& x : t) x.join();
    CHECK_EQ(bad.load(), 0);
    CHECK_EQ(kept.load(), 8);   /* the room, each unit once */
    const HxSideStore::Stats st = s.stats();
    CHECK_EQ(st.units, 8);
    int held = 0;
    for (int u = 0; u < 16; ++u) held += s.holds(u);
    CHECK_EQ(held, 8);
    /* 32 offers: 8 kept, the rest either duplicates or turned away for want of room. */
    CHECK_EQ(fl.returned.load(), 32 - 8);
    CHECK(st.full <= (uint64_t)24);
}

RAD_TEST_MAIN()

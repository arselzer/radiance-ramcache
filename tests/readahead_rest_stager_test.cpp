/* readahead_rest_stager_test.cpp -- the file stager's side of RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT
 * on the host backend, through the stager's test friend (stager.h's EpCountsHostProbe): a partly
 * read unit its layer takes is stopped -- not made urgent, not cancelled -- and handed to the layer
 * with the planes its pieces land in; the layer's read list (hx_unread_misses) carries its rest,
 * what each extent has past the pieces started, beside the plain misses and never a unit read ahead
 * whole; nothing for a unit whose every piece started (only the wait for them is left); and without
 * the switch none of the piece readers' units. "Alone" is one such unit with nothing else to read:
 * the layers whose wait is that unit's own (the per-unit wait the shutdown report compares). */
#include "rad_test.h"

#include "place/readahead.h"
#include "place/stager.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace rad {

/* FileStager's test friend. The piece readers outlive the stager (members in that order): its
 * destructor forgets its units on them. */
struct EpCountsHostProbe {
    PieceReader r;
    FileStager fs;

    /* A piece read: pieces from `hold` on wait in flight until `go`. */
    struct Held {
        std::atomic<int>  n{ 0 };
        int               hold = 1 << 30;
        std::atomic<bool> go{ false };
    };
    static int read(void* c, const PlaneExtent& p) {
        auto* h = static_cast<Held*>(c);
        if (h->n.fetch_add(1) >= h->hold)
            while (!h->go.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
        std::memset(p.dst, 0x5a, (size_t)p.len);
        return 0;
    }

    void rests_join_the_layers_read() {
        REQUIRE(r.start(1, 64 << 10));
        Held held;
        held.hold = 3;   /* pieces 0-2 land, piece 3 waits in flight */
        std::vector<uint8_t> a(1 << 20), b(1 << 20);
        PlaneExtent ext[2] = { { 0, 0, 640 << 10, a.data(), (int64_t)a.size() },
                               { 0, 4 << 20, 320 << 10, b.data(), (int64_t)b.size() } };
        auto planes = std::make_shared<Mover::HostPlanes>();
        planes->unit = 1;
        auto ra = std::make_shared<ReadAheadUnit>(&fs, 7, ext, 2, 64 << 10, &read, &held, nullptr);
        /* However the case ends, no reader is left in `held` or the buffers once they go. */
        struct Settle {
            PieceReader& r;
            Held& h;
            std::shared_ptr<ReadAheadUnit>& u;
            ~Settle() {
                h.go = true;
                if (u) {
                    r.cancel(*u);
                    r.wait(*u);
                }
            }
        } settle{ r, held, ra };
        r.submit(ra);
        for (int i = 0; i < 50000 && !(held.n.load() == 4 && ra->in() == 3); ++i)
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        REQUIRE_EQ(ra->in(), 3);
        REQUIRE_EQ(held.n.load(), 4);

        /* hx_ra_take with ..._PREFETCH_URGENT=2 (which never asks the mover about the mirror). */
        fs.hx_ra_ = &r;
        fs.hx_ra_finish_ = true;
        fs.hx_ra_rest_ = 2;
        fs.hx_ra_here_.push_back(FileStager::RaEntry{ ra, planes, 7, 0, 1 });
        bool ready = true;
        std::shared_ptr<ReadAheadUnit> part;
        CHECK(fs.hx_ra_take(1, &ready, &part) == planes);
        CHECK(!ready);
        CHECK(part == ra);
        CHECK_EQ(ra->started(), 4);
        CHECK(!ra->urgent());
        CHECK(!ra->cancelled());
        CHECK_EQ(r.stats().stopped, (uint64_t)1);
        CHECK_EQ(fs.hx_stats_.ra_finished, (uint64_t)1);
        CHECK_EQ(fs.hx_stats_.ra_fresh, (uint64_t)0);
        CHECK_EQ(fs.hx_stats_.ra_used_bytes, (uint64_t)(3 * (64 << 10)));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        CHECK_EQ(held.n.load(), 4);   /* no piece started after the stop */

        /* The layer's misses: the stopped unit, a plain miss, one read ahead whole. */
        auto plain = std::make_shared<Mover::HostPlanes>(), whole = std::make_shared<Mover::HostPlanes>();
        fs.hx_miss_ = { planes, plain, whole };
        fs.hx_miss_ready_ = { 0, 0, 1 };
        fs.hx_miss_ra_ = { part, nullptr, nullptr };
        std::vector<Mover::HostPlanes*> out;
        std::vector<Mover::PlanesIn> in;
        bool alone = true;
        CHECK(fs.hx_unread_misses(&out, &in, &alone));
        REQUIRE_EQ(out.size(), (size_t)2);
        REQUIRE_EQ(in.size(), (size_t)2);
        CHECK(out[0] == planes.get());
        CHECK_EQ(in[0].bytes[0], (int64_t)(4 * (64 << 10)));
        CHECK_EQ(in[0].bytes[1], (int64_t)0);
        CHECK(out[1] == plain.get());
        CHECK_EQ(in[1].bytes[0], (int64_t)0);
        CHECK_EQ(in[1].bytes[1], (int64_t)0);
        CHECK(!alone);
        /* Alone: the stopped unit and nothing else to read. */
        fs.hx_miss_ = { planes, whole };
        fs.hx_miss_ready_ = { 0, 1 };
        fs.hx_miss_ra_ = { part, nullptr };
        out.clear();
        in.clear();
        CHECK(fs.hx_unread_misses(&out, &in, &alone));
        CHECK(alone);
        REQUIRE_EQ(out.size(), (size_t)1);
        CHECK(out[0] == planes.get());
        /* Without the switch the piece readers read the rest: the unit is not on the list. */
        fs.hx_ra_rest_ = 0;
        out.clear();
        in.clear();
        CHECK(fs.hx_unread_misses(&out, &in, &alone));
        CHECK(out.empty());
        CHECK(alone);
        /* No partly read unit: false, and not alone. */
        fs.hx_miss_ = { plain };
        fs.hx_miss_ready_ = { 0 };
        fs.hx_miss_ra_ = { nullptr };
        out.clear();
        in.clear();
        CHECK(!fs.hx_unread_misses(&out, &in, &alone));
        CHECK(!alone);
        CHECK_EQ(out.size(), (size_t)1);

        /* The held piece lands: the unit is not whole by its pieces (its rest was the layer's),
         * and none failed -- what hx_ra_settle checks for a stopped unit. */
        held.go = true;
        r.wait(*ra);
        CHECK_EQ(ra->in(), 4);
        CHECK(!ra->whole());
        CHECK(!ra->failed());

        /* A unit whose every piece started: nothing left to read, only the wait. */
        Held all;
        std::vector<uint8_t> c(1 << 20), d(1 << 20);
        PlaneExtent ext2[2] = { { 0, 8 << 20, 128 << 10, c.data(), (int64_t)c.size() },
                                { 0, 9 << 20, 64 << 10, d.data(), (int64_t)d.size() } };
        auto planes2 = std::make_shared<Mover::HostPlanes>();
        auto ra2 = std::make_shared<ReadAheadUnit>(&fs, 8, ext2, 2, 64 << 10, &read, &all, nullptr);
        r.submit(ra2);
        r.wait(*ra2);
        REQUIRE(ra2->whole());
        CHECK_EQ(r.stop(*ra2), 3);
        fs.hx_ra_rest_ = 1;
        fs.hx_miss_ = { planes2 };
        fs.hx_miss_ready_ = { 0 };
        fs.hx_miss_ra_ = { ra2 };
        out.clear();
        in.clear();
        CHECK(fs.hx_unread_misses(&out, &in, &alone));
        CHECK(out.empty());
        CHECK(alone);
        fs.hx_miss_.clear();
        fs.hx_miss_ready_.clear();
        fs.hx_miss_ra_.clear();
        fs.hx_ra_here_.clear();
        fs.hx_ra_rest_ = 0;
    }
};

}  // namespace rad

TEST(a_partly_read_units_rest_joins_its_layers_read) {
    rad::EpCountsHostProbe p;
    p.rests_join_the_layers_read();
}

RAD_TEST_MAIN()

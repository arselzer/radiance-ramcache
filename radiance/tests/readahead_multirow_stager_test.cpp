/* readahead_multirow_stager_test.cpp -- the file stager's side of
 * RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW on the host backend, through the stager's test friend
 * (stager.h's EpCountsHostProbe), with no mover: begin_pass marks the passes that read ahead (every
 * one-token pass, as without the switch; with it trunk decode passes of 2..N rows, never a prefill or
 * mixed pass, a longer one, or the drafter's own passes); and a deciding layer's units, each tagged
 * with the rows of the post that planned it, are counted under their pass size -- a whole unit
 * handed over, one with no piece started read as any miss, a partly read one finished by the
 * readers, one not selected dropped, an earlier pass's dropped as stale -- beside the totals, which
 * keep their meaning; and the switch at init is off unset or 0, N when the read-ahead in pieces runs
 * and the CPU takes the misses of passes that size, and refused otherwise, never ignored.
 * (hx_ra_plan itself, with the actual mover, is native_mapped_copy_mover_test's
 * mapped_read_ahead_multirow_union_through_the_file_stager, a HOST-only binary.) */
#include "rad_test.h"

#include "place/readahead.h"
#include "place/router_predictor.h"
#include "place/stager.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rad {

/* FileStager's test friend. The piece readers outlive the stager (members in that order): its
 * destructor forgets its units on them. */
struct EpCountsHostProbe {
    PieceReader r;
    FileStager fs;

    /* A piece read: fills the piece; pieces from `hold` on wait in flight until `go`. */
    struct Held {
        std::atomic<int>  n{ 0 };
        int               hold = 1 << 30;
        std::atomic<bool> go{ false };
    };
    static int read(void* c, const PlaneExtent& p) {
        auto* h = static_cast<Held*>(c);
        if (h->n.fetch_add(1) >= h->hold)
            while (!h->go.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
        std::memset(p.dst, 0x3c, (size_t)p.len);
        return 0;
    }

    static RadBatch batch(int phase, int64_t n_tok, int32_t draft) {
        RadBatch b;
        std::memset(&b, 0, sizeof b);
        b.phase = phase;
        b.n_tok = n_tok;
        b.n_seq = 1;
        b.draft_pass = draft;
        return b;
    }

    bool reads_ahead(int phase, int64_t n_tok, int32_t draft) {
        RadBatch b = batch(phase, n_tok, draft);
        CHECK(fs.begin_pass(&b));
        fs.armed_ = false;
        return fs.hx_ra_pass_;
    }

    void passes_that_read_ahead() {
        StagedLayer L;
        L.layer = 12;
        L.units = { 0 };
        fs.layers_ = { L };   /* enabled(): begin_pass arms */
        /* Without the switch: one-token passes only, of any kind, as always. */
        fs.hx_ra_rows_ = 1;
        CHECK(reads_ahead(RAD_PHASE_DECODE, 1, 0));
        CHECK(reads_ahead(RAD_PHASE_PREFILL, 1, 0));
        CHECK(!reads_ahead(RAD_PHASE_DECODE, 2, 0));
        CHECK(!reads_ahead(RAD_PHASE_DECODE, 6, 0));
        /* PREFETCH_MULTIROW=6: trunk decode passes of 2..6 rows too. */
        fs.hx_ra_rows_ = 6;
        CHECK(reads_ahead(RAD_PHASE_DECODE, 1, 0));
        CHECK(reads_ahead(RAD_PHASE_DECODE, 2, 0));
        CHECK(reads_ahead(RAD_PHASE_DECODE, 6, 0));
        CHECK(!reads_ahead(RAD_PHASE_DECODE, 7, 0));
        CHECK(!reads_ahead(RAD_PHASE_PREFILL, 4, 0));    /* a short prompt's chunk */
        CHECK(!reads_ahead(RAD_PHASE_MIXED, 3, 0));
        CHECK(!reads_ahead(RAD_PHASE_DECODE, 5, 1));     /* the drafter's query pass */
        CHECK(!reads_ahead(RAD_PHASE_DECODE, 2, -1));    /* its context pass */
        CHECK(!reads_ahead(RAD_PHASE_DECODE, 16, 0));
        fs.layers_.clear();
    }

    void counts_by_pass_size() {
        REQUIRE(r.start(2, 64 << 10));
        fs.hx_ra_ = &r;
        fs.hx_ra_finish_ = true;
        Held plain, held;
        held.hold = 2;   /* that unit's pieces 0-1 land, piece 2 waits in flight */
        std::vector<std::vector<uint8_t>> bufs;
        bufs.reserve(16);
        /* A unit of 2 + 1 pieces of 64 KiB (192 KiB), read now (`read`) or never started. */
        auto unit = [&](uint32_t pass, int32_t li, int32_t u, int rows, bool read_now, Held* h) {
            bufs.emplace_back((size_t)(384 << 10));
            uint8_t* d = bufs.back().data();
            PlaneExtent ext[2] = { { 0, 0, 128 << 10, d, 256 << 10 },
                                   { 0, 1 << 20, 64 << 10, d + (256 << 10), 128 << 10 } };
            auto p = std::make_shared<Mover::HostPlanes>();
            p->unit = u;
            auto ra = std::make_shared<ReadAheadUnit>(&fs, ((uint64_t)pass << 24) | (uint64_t)u, ext, 2,
                                                      64 << 10, &read, h, p);
            if (read_now) {
                r.submit(ra);
                if (h == &plain) r.wait(*ra);
            }
            return FileStager::RaEntry{ ra, p, pass, li, u, rows };
        };
        const auto& by = fs.hx_stats_.ra_rows;
        const auto& hx = fs.hx_stats_;

        /* Pass 5, a 2-row verify pass at routed layer 0: units 1 (read whole), 2 (no piece started),
         * 3 (read, not selected), 4 (partly read); and pass 4's unit 9 from a 3-row post, stale. */
        fs.hx_ra_live_.push_back(unit(5, 0, 1, 2, true, &plain));
        fs.hx_ra_live_.push_back(unit(5, 0, 2, 2, false, &plain));
        fs.hx_ra_live_.push_back(unit(5, 0, 3, 2, true, &plain));
        fs.hx_ra_live_.push_back(unit(4, 0, 9, 3, false, &plain));
        auto part = unit(5, 0, 4, 2, true, &held);
        for (int i = 0; i < 50000 && !(held.n.load() == 3 && part.u->in() == 2); ++i)
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        REQUIRE_EQ(part.u->in(), 2);
        fs.hx_ra_live_.push_back(part);
        fs.trace_pass_ = 5;
        fs.pass_tok_ = 2;
        fs.hx_ra_begin(0);
        CHECK_EQ(fs.hx_ra_here_.size(), (size_t)4);
        CHECK(fs.hx_ra_live_.empty());
        CHECK_EQ(by[3].dropped, (uint64_t)1);   /* pass 4's, by its post's rows */
        CHECK_EQ(hx.ra_stale, (uint64_t)1);

        bool ready = false;
        std::shared_ptr<ReadAheadUnit> urgent;
        CHECK(fs.hx_ra_take(1, &ready, &urgent) != nullptr);
        CHECK(ready);
        CHECK(!urgent);
        ready = true;
        CHECK(fs.hx_ra_take(2, &ready, &urgent) != nullptr);   /* its buffers, read as any miss */
        CHECK(!ready);
        CHECK(!urgent);
        /* Partly read: no mover here, so no plane mirror share to beat -- the readers finish it. */
        CHECK(fs.hx_ra_take(4, &ready, &urgent) != nullptr);
        CHECK(!ready);
        REQUIRE(urgent != nullptr);
        CHECK(urgent->urgent());
        CHECK(fs.hx_ra_take(7, &ready, &urgent) == nullptr);   /* a miss nobody planned */
        for (auto& e : fs.hx_ra_here_) fs.hx_ra_drop(e, false);   /* unit 3: not selected */
        fs.hx_ra_here_.clear();
        held.go = true;
        r.wait(*part.u);
        CHECK(part.u->whole());

        CHECK_EQ(by[2].misses, (uint64_t)4);
        CHECK_EQ(by[2].miss_planned, (uint64_t)3);
        CHECK_EQ(by[2].whole, (uint64_t)1);
        CHECK_EQ(by[2].finished, (uint64_t)1);
        CHECK_EQ(by[2].fresh, (uint64_t)1);
        CHECK_EQ(by[2].dropped, (uint64_t)1);
        CHECK_EQ(by[2].used_bytes, (uint64_t)((192 + 128) << 10));   /* unit 1 whole, unit 4's two pieces */
        CHECK_EQ(by[2].dropped_bytes, (uint64_t)(192 << 10));        /* unit 3 */
        CHECK_EQ(by[1].misses, (uint64_t)0);
        CHECK_EQ(by[1].planned + by[1].whole + by[1].dropped, (uint64_t)0);
        /* The totals are the slots' sums. */
        CHECK_EQ(hx.ra_misses, (uint64_t)4);
        CHECK_EQ(hx.ra_miss_planned, (uint64_t)3);
        CHECK_EQ(hx.ra_whole + hx.ra_finished + hx.ra_fresh, (uint64_t)3);
        CHECK_EQ(hx.ra_unselected + hx.ra_cached + hx.ra_stale, by[2].dropped + by[3].dropped);
        CHECK_EQ(hx.ra_used_bytes, by[2].used_bytes);

        /* Pass 6, one token: its unit counts under 1 row, the 2-row slot unchanged. */
        fs.hx_ra_live_.push_back(unit(6, 0, 5, 1, true, &plain));
        fs.trace_pass_ = 6;
        fs.pass_tok_ = 1;
        fs.hx_ra_begin(0);
        CHECK(fs.hx_ra_take(5, &ready, &urgent) != nullptr);
        CHECK(ready);
        fs.hx_ra_here_.clear();
        CHECK_EQ(by[1].misses, (uint64_t)1);
        CHECK_EQ(by[1].whole, (uint64_t)1);
        CHECK_EQ(by[2].misses, (uint64_t)4);
        CHECK_EQ(fs.hx_ra_mirror_bytes(), (int64_t)0);   /* no mover, no mirror */
        fs.hx_ra_ = nullptr;
        r.forget(&fs);
    }

    /* The switch at init (hx_ra_rows_init): off when unset or 0; N when the read-ahead in pieces
     * runs and the CPU takes the misses of passes that size; refused otherwise, never ignored. */
    void switch_is_refused_rather_than_ignored() {
        const char* const kVar = "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW";
        struct Restore {
            const char* name;
            std::string old;
            bool had;
            ~Restore() { if (had) ::setenv(name, old.c_str(), 1); else ::unsetenv(name); }
        } restore{ kVar, std::getenv(kVar) ? std::getenv(kVar) : "", std::getenv(kVar) != nullptr };
        ::setenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH", "pieces", 1);   /* before the first get() */
        RouterPredictor* pred = RouterPredictor::get();
        REQUIRE(pred != nullptr);
        REQUIRE(pred->pieces());
        auto init = [&](const char* v) {
            if (v) ::setenv(kVar, v, 1); else ::unsetenv(kVar);
            return fs.hx_ra_rows_init();
        };
        fs.hx_miss_rows_ = 16;
        fs.ep_rows_ = 16;
        CHECK_OK(init(nullptr));
        CHECK_EQ(fs.hx_ra_rows_, 1);
        CHECK_OK(init("0"));
        CHECK_EQ(fs.hx_ra_rows_, 1);
        CHECK_EQ(init("6"), RAD_E_INVAL);   /* the read-ahead is not running */
        CHECK_EQ(fs.hx_ra_rows_, 1);
        REQUIRE(r.start(1, 64 << 10));
        fs.hx_ra_ = &r;
        CHECK_EQ(init("6"), RAD_E_INVAL);   /* no predictor */
        fs.hx_pred_ = pred;
        CHECK_OK(init("6"));
        CHECK_EQ(fs.hx_ra_rows_, 6);
        CHECK_OK(init("16"));
        CHECK_EQ(fs.hx_ra_rows_, 16);
        CHECK_EQ(init("17"), RAD_E_INVAL);  /* past the EP path's passes */
        CHECK_EQ(init("1"), RAD_E_INVAL);
        CHECK_EQ(init("yes"), RAD_E_INVAL);
        fs.hx_miss_rows_ = 4;               /* the CPU takes no misses of 6-row passes */
        CHECK_EQ(init("6"), RAD_E_INVAL);
        CHECK_OK(init("4"));
        CHECK_EQ(fs.hx_ra_rows_, 4);
        fs.hx_ra_ = nullptr;
        fs.hx_pred_ = nullptr;
    }
};

}  // namespace rad

TEST(multirow_passes_that_read_ahead_are_marked_at_begin_pass) {
    rad::EpCountsHostProbe p;
    p.passes_that_read_ahead();
}

TEST(multirow_units_are_counted_by_pass_size) {
    rad::EpCountsHostProbe p;
    p.counts_by_pass_size();
}

TEST(multirow_switch_is_refused_rather_than_ignored) {
    rad::EpCountsHostProbe p;
    p.switch_is_refused_rather_than_ignored();
}

RAD_TEST_MAIN()

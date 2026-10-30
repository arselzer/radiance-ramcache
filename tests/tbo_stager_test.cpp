/* tbo_stager_test.cpp -- the file stager's side of the two-batch overlap (rad_declare_two_batch,
 * RADIANCE_EXPERIMENTAL_SCHED_TBO, core/sched/tbo.h) on the host backend, through the stager's test
 * friend (stager.h's EpCountsHostProbe), with no mover:
 *   - begin_pass splits a pass only when the program declared the split and the step is one
 *     tbo::step_splits names (the switch alone, an undeclared program, one sequence, a prompt, a
 *     mixed or drafter pass: one pass of the step's rows, as before), and takes micro-batch A's slice
 *     first: its rows, from row 0;
 *   - slice B is B's rows from A's count on (the CPU path's rows, copies and jobs: layer_r0 and
 *     layer_rows), and each slice reads ahead as a decode pass of its own rows does;
 *   - a layer's second gate_up (slice B) is refused before slice A's CPU rows reached their gather
 *     (computed or early-issued), and a third one in the pass outright, never run on stale rows;
 *   - the read-ahead entries slice A left of a layer stay for slice B when B reads ahead, and a
 *     slice B finds the layer's entries planned since; slice B's own copies go one by one (the
 *     ranged copy list is armed for passes from row 0);
 *   - end_pass ends the split. */
#include "rad_test.h"

#include "place/readahead.h"
#include "place/stager.h"
#include "rad_core.h"
#include "sched/tbo.h"

#include <cstring>
#include <memory>
#include <vector>

namespace rad {

struct EpCountsHostProbe {
    FileStager fs;
    Program prog;

    static RadBatch decode(int64_t S) {
        RadBatch b;
        std::memset(&b, 0, sizeof b);
        b.phase = RAD_PHASE_DECODE;
        b.n_tok = S;
        b.n_seq = S;
        b.n_seq_decode = S;
        b.n_tok_decode = S;
        return b;
    }

    EpCountsHostProbe() {
        StagedLayer L;
        L.layer = 12;
        L.units = { 0 };
        fs.layers_ = { L };   /* enabled(): begin_pass arms */
        fs.ep_program_ = &prog;
    }

    void begin(const RadBatch& b) {
        CHECK(fs.begin_pass(&b));
        fs.armed_ = false;
    }

    void splits_only_declared_pure_decode_passes() {
        RadBatch b = decode(4);
        /* Not declared (the switch alone changes nothing in the core): one pass of 4 rows. */
        prog.two_batch = false;
        begin(b);
        CHECK(!fs.tbo_);
        CHECK_EQ(fs.pass_tok_, (int64_t)4);
        CHECK_EQ(fs.layer_r0(0), (int64_t)0);
        CHECK_EQ(fs.layer_rows(0), (int64_t)4);
        fs.ep_program_ = nullptr;
        begin(b);
        CHECK(!fs.tbo_);
        fs.ep_program_ = &prog;
        /* Declared: slice A first. */
        prog.two_batch = true;
        for (int64_t S : { 2, 4, 5, 8, 16 }) {
            RadBatch d = decode(S);
            begin(d);
            CHECK(fs.tbo_);
            CHECK_EQ(fs.tbo_slice_, 0);
            CHECK_EQ(fs.tbo_tok_, S);
            CHECK_EQ(fs.pass_tok_, (S + 1) / 2);
            CHECK_EQ(fs.layer_r0(0), (int64_t)0);
            CHECK_EQ(fs.layer_rows(0), (S + 1) / 2);
            fs.tbo_enter(1);
            CHECK_EQ(fs.pass_tok_, S / 2);
            CHECK_EQ(fs.layer_r0(0), (S + 1) / 2);
            CHECK_EQ(fs.layer_rows(0), S / 2);
        }
        /* Never split: one sequence, a prompt chunk, a mixed pass, the drafter's, a cut pass. */
        RadBatch one = decode(1);
        RadBatch chunk = decode(16);
        chunk.phase = RAD_PHASE_PREFILL;
        chunk.n_seq = 1;
        chunk.n_seq_decode = 0;
        chunk.n_tok_decode = 0;
        RadBatch mixed = decode(5);
        mixed.phase = RAD_PHASE_MIXED;
        mixed.n_seq_decode = 4;
        mixed.n_tok_decode = 4;
        mixed.n_seq = 5;
        RadBatch draft = decode(4);
        draft.draft_pass = 1;
        RadBatch verify = decode(6);
        verify.n_seq = 2;
        verify.n_seq_decode = 2;
        verify.n_spec = 2;
        RadBatch cut = decode(4);
        cut.consumer_full = 25;
        for (const RadBatch* x : { &one, &chunk, &mixed, &draft, &verify, &cut }) {
            begin(*x);
            CHECK(!fs.tbo_);
            CHECK_EQ(fs.pass_tok_, x->n_tok);
            CHECK_EQ(fs.layer_r0(0), (int64_t)0);
        }
        prog.two_batch = false;
    }

    /* Each slice reads ahead as a decode pass of its rows: one-token slices always, 2..N-row slices
     * with ..._PREFETCH_MULTIROW=N; slice A keeps the entries it did not select for B only when B
     * reads ahead. */
    void slices_read_ahead_by_their_rows() {
        prog.two_batch = true;
        fs.hx_ra_rows_ = 1;
        RadBatch two = decode(2), four = decode(4);
        begin(two);
        CHECK(fs.hx_ra_pass_);
        CHECK(fs.tbo_keeps_ra());
        fs.tbo_enter(1);
        CHECK(fs.hx_ra_pass_);
        CHECK(!fs.tbo_keeps_ra());   /* slice B: what it did not select goes */
        begin(four);
        CHECK(!fs.hx_ra_pass_);
        CHECK(!fs.tbo_keeps_ra());
        fs.hx_ra_rows_ = 6;
        begin(four);
        CHECK(fs.hx_ra_pass_);
        CHECK(fs.tbo_keeps_ra());
        RadBatch three = decode(3);   /* 2 + 1 */
        fs.hx_ra_rows_ = 1;
        begin(three);
        CHECK(!fs.hx_ra_pass_);
        CHECK(fs.tbo_keeps_ra());     /* B's one row reads ahead */
        fs.tbo_enter(1);
        CHECK(fs.hx_ra_pass_);
        fs.hx_ra_rows_ = 1;
        prog.two_batch = false;
    }

    /* Slice B's start (tbo_second) is refused before slice A's gather, and a third gate_up. */
    void slice_b_waits_for_slice_a_gather() {
        prog.two_batch = true;
        RadBatch four = decode(4);
        begin(four);
        fs.hx_li_ = 0;
        fs.hx_stage_ = 2;   /* A's CPU rows computed, not yet copied at its gather */
        CHECK_EQ(fs.tbo_second(0, nullptr), (int)RAD_E_STATE);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.hx_early_li_ = 0;   /* A's card ops early-issued, its CPU share not yet decided */
        CHECK_EQ(fs.tbo_second(0, nullptr), (int)RAD_E_STATE);
        fs.hx_early_li_ = -1;
        CHECK_EQ(fs.tbo_slice_, 0);
        fs.tbo_enter(1);
        CHECK_EQ(fs.tbo_second(0, nullptr), (int)RAD_E_STATE);   /* a third gate_up */
        prog.two_batch = false;
    }

    void read_ahead_entries_carry_to_slice_b() {
        prog.two_batch = true;
        RadBatch two = decode(2);
        begin(two);
        const uint32_t pass = fs.trace_pass_;
        auto entry = [&](int32_t u) { return FileStager::RaEntry{ nullptr, nullptr, pass, 0, u, 1 }; };
        /* Slice A at layer 0 takes the layer's planned entries. */
        fs.hx_ra_live_.push_back(entry(1));
        fs.hx_ra_live_.push_back(entry(2));
        fs.hx_ra_begin(0);
        CHECK_EQ(fs.hx_ra_here_.size(), (size_t)2);
        /* A used unit 1 (its entry's buffers moved out); unit 2 stays for B. One more was planned. */
        fs.hx_ra_here_.erase(fs.hx_ra_here_.begin());
        fs.hx_ra_live_.push_back(entry(3));
        fs.tbo_enter(1);
        fs.hx_ra_begin(0);
        REQUIRE_EQ(fs.hx_ra_here_.size(), (size_t)2);
        CHECK_EQ(fs.hx_ra_here_[0].unit, 2);
        CHECK_EQ(fs.hx_ra_here_[1].unit, 3);
        CHECK(fs.hx_ra_live_.empty());
        /* The next layer's slice A starts afresh. */
        fs.tbo_enter(0);
        fs.hx_ra_begin(0);
        CHECK(fs.hx_ra_here_.empty());
        /* Off the split, as before: a layer's start clears what the layer before left. */
        fs.tbo_ = false;
        fs.hx_ra_here_.push_back(entry(4));
        fs.hx_ra_begin(0);
        CHECK(fs.hx_ra_here_.empty());
        prog.two_batch = false;
    }

    void slice_b_copies_rows_one_by_one() {
        prog.two_batch = true;
        RadBatch four = decode(4);
        begin(four);
        fs.hx_ops_.resize(1);
        fs.tbo_enter(1);
        const uint64_t plain = fs.hx_stats_.fl_plain;
        CHECK_EQ(fs.hx_fetch_list(0, nullptr, nullptr), 0);
        CHECK_EQ(fs.hx_stats_.fl_plain, plain + 1);
        fs.hx_ops_.clear();
        prog.two_batch = false;
    }

    void end_pass_ends_the_split() {
        prog.two_batch = true;
        RadBatch four = decode(4);
        CHECK(fs.begin_pass(&four));
        CHECK(fs.tbo_);
        CHECK_OK(fs.end_pass(nullptr));
        CHECK(!fs.tbo_);
        prog.two_batch = false;
    }
};

}  // namespace rad

TEST(tbo_stager_splits_only_declared_pure_decode_passes) {
    rad::EpCountsHostProbe p;
    p.splits_only_declared_pure_decode_passes();
}

TEST(tbo_stager_slices_read_ahead_by_their_rows) {
    rad::EpCountsHostProbe p;
    p.slices_read_ahead_by_their_rows();
}

TEST(tbo_stager_slice_b_waits_for_slice_a_gather) {
    rad::EpCountsHostProbe p;
    p.slice_b_waits_for_slice_a_gather();
}

TEST(tbo_stager_read_ahead_entries_carry_to_slice_b) {
    rad::EpCountsHostProbe p;
    p.read_ahead_entries_carry_to_slice_b();
}

TEST(tbo_stager_slice_b_copies_rows_one_by_one) {
    rad::EpCountsHostProbe p;
    p.slice_b_copies_rows_one_by_one();
}

TEST(tbo_stager_end_pass_ends_the_split) {
    rad::EpCountsHostProbe p;
    p.end_pass_ends_the_split();
}

RAD_TEST_MAIN()

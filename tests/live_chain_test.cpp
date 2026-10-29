/* live_chain_test.cpp -- the live issue path's hazard bookkeeping (runtime/live_chain.h).
 *
 * LiveQueues is pure bookkeeping over the positions the backend reports, so these cases drive it
 * against a model of the AQL backend's side (aql.cpp, "the live window"): a queue with a write
 * index, the tag of the launch that wrote its last packet, and its run of packets without the
 * barrier bit; a permit honoured only for a window's first kernel at the checked index while the
 * run is under kTapeOverlapRun. The card is never touched.
 *
 * The main case is a plain DeepSeek-V4.1 decode layer in its issue order (deepseek41.cpp), with
 * every operand a disjoint byte range as the arena gives them: the check frees exactly the six
 * dispatches the offline model of haz_mark's rules found (RESEARCH-2-RANKED-PLAN.md #6), and keeps
 * the ordered points (the all-reduce's link hold and release) and every true dependence.
 */
#include "rad_test.h"

#include "device/device.h"
#include "runtime/live_chain.h"

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using rad::LiveOp;
using rad::LiveQueues;
using rad::LiveWhy;

namespace {

/* The backend's half, as aql.cpp implements it. */
struct FakeQueue {
    uint32_t id = 7;
    uint64_t wr = 100;
    uint64_t last_tag = 0;
    int      unord_run = 0;
    std::vector<bool> ordered;   /* per packet written */

    LiveQueues::Pos pos() const { return LiveQueues::Pos{ id, wr, last_tag }; }
    /* A packet the issue path did not write: an event, a wait, a copy (barrier bit set). */
    void foreign() {
        ordered.push_back(true);
        unord_run = 0;
        last_tag = 0;
        ++wr;
    }
    /* A launch of `kernels` kernels (then `barriers` barrier packets) under a window opened at
     * `at` with `permit`: returns the window as live_close reports it. */
    LiveQueues::Win launch(uint64_t tag, uint64_t at, bool permit, int kernels, int barriers = 0) {
        LiveQueues::Win w;
        for (int k = 0; k < kernels + barriers; ++k) {
            const bool barrier = k >= kernels;
            const bool unord = !barrier && permit && w.packets == 0 && wr == at &&
                               unord_run < rad::kTapeOverlapRun;
            if (w.packets == 0) {
                w.first = wr;
                if (unord) w.flags |= rad::LIVE_FIRST_UNORDERED;
            }
            if (barrier) w.flags |= rad::LIVE_BARRIER;
            ++w.packets;
            ordered.push_back(!unord);
            unord_run = unord ? unord_run + 1 : 0;
            last_tag = tag;
            ++wr;
        }
        return w;
    }
};

/* The issue path's half (Ctx::live_before / live_after) over the model. */
struct Issuer {
    LiveQueues q;
    FakeQueue  fq;
    uint64_t   tag = 0;
    bool       overlap = true;

    struct Out { LiveWhy why; bool unordered; uint64_t tag; };

    Out issue(const LiveOp& shape, int kernels = 1, int barriers = 0) {
        const int s = q.claim();
        LiveOp& o = q.op(s);
        o = shape;
        o.tag = ++tag;
        uint64_t clash = 0;
        int64_t foreign = 0;
        const LiveQueues::Pos before = fq.pos();
        const LiveWhy why = q.check(s, before, &clash, &foreign);
        const bool permit = overlap && why == rad::kLiveFree;
        const LiveQueues::Win w = fq.launch(o.tag, before.wr, permit, kernels, barriers);
        q.commit(s, before, w, fq.pos());
        return Out{ why, (w.flags & rad::LIVE_FIRST_UNORDERED) != 0, o.tag };
    }
};

/* Disjoint byte ranges, one a named buffer. */
struct Arena {
    std::map<std::string, std::pair<uintptr_t, uintptr_t>> b;
    uintptr_t next = 0x10000;
    std::pair<uintptr_t, uintptr_t> at(const std::string& n, uintptr_t bytes = 4096) {
        auto it = b.find(n);
        if (it != b.end()) return it->second;
        const auto r = std::make_pair(next, next + bytes);
        next += bytes + 256;
        b[n] = r;
        return r;
    }
};

LiveOp op(Arena& a, std::vector<std::string> rd, std::vector<std::string> wr) {
    LiveOp o;
    for (const auto& n : rd) {
        const auto r = a.at(n);
        o.read(r.first, r.second);
    }
    for (const auto& n : wr) {
        const auto r = a.at(n);
        o.write(r.first, r.second);
    }
    return o;
}

LiveOp opaque_op() {
    LiveOp o;
    o.opaque = true;
    return o;
}

}  // namespace

TEST(dependences_by_kind) {
    Arena a;
    const LiveOp w_x = op(a, {}, { "x" });
    const LiveOp r_x = op(a, { "x" }, { "y" });
    const LiveOp w_y = op(a, {}, { "y" });
    const LiveOp r_z = op(a, { "z" }, { "u" });
    CHECK_EQ(rad::live_dep(r_x, w_x), (int)rad::kLiveRaw);              /* reads what it wrote */
    CHECK_EQ(rad::live_dep(w_x, r_x), (int)rad::kLiveWar);              /* writes what it read */
    CHECK_EQ(rad::live_dep(w_y, r_x), (int)rad::kLiveWaw);              /* writes what it wrote */
    CHECK_EQ(rad::live_dep(r_z, w_x), 0);                               /* independent */
    CHECK_EQ(rad::live_dep(r_z, opaque_op()), (int)rad::kLiveUnknown);
    /* Two column slices of one row (o_a's groups) are disjoint ranges: independent. */
    LiveOp lo, hi;
    lo.read(0x1000, 0x1800);
    lo.write(0x9000, 0x9100);
    hi.read(0x1800, 0x2000);
    hi.write(0x9100, 0x9200);
    CHECK_EQ(rad::live_dep(hi, lo), 0);
    /* Half-open: touching ends do not meet. */
    LiveOp p, q;
    p.write(0x100, 0x200);
    q.read(0x200, 0x300);
    CHECK_EQ(rad::live_dep(q, p), 0);
    q.read(0x1ff, 0x200);
    CHECK_EQ(rad::live_dep(q, p), (int)rad::kLiveRaw);
}

TEST(a_plain_ds41_decode_layer_frees_six_dispatches) {
    Arena a;
    Issuer is;
    struct Step { const char* name; LiveOp o; int kernels; LiveWhy want; bool foreign_before; };
    LiveOp ar = opaque_op();
    std::vector<Step> layer = {
        { "pcn_a",  op(a, { "act", "h1", "post", "comb" }, { "h0", "pre_f", "xcol", "xn" }), 1, rad::kLiveForeign, true },
        { "q_a",    op(a, { "xn", "s1" }, { "qr", "s1" }),            1, rad::kLiveClash, false },
        { "kv_g",   op(a, { "xn", "s2" }, { "kv", "s2" }),            1, rad::kLiveFree, false },
        { "q_an",   op(a, { "qr" }, { "qrn" }),                       1, rad::kLiveClash, false },
        { "kv_f",   op(a, { "kv", "cs", "pos" }, { "kvn" }),          1, rad::kLiveFree, false },
        { "q_b",    op(a, { "qrn", "s3" }, { "q", "s3" }),            1, rad::kLiveClash, false },
        { "q_rope", op(a, { "q", "cs", "pos" }, { "q" }),             1, rad::kLiveClash, false },
        { "kv_st",  op(a, { "kvn", "slot" }, { "win" }),              1, rad::kLiveFree, false },
        { "attn",   op(a, { "q", "win", "pos" }, { "o" }),            1, rad::kLiveClash, false },
        { "o_f",    op(a, { "o", "cs", "pos" }, { "o" }),             1, rad::kLiveClash, false },
        { "o_a0",   LiveOp{},                                         1, rad::kLiveClash, false },
        { "o_a1",   LiveOp{},                                         1, rad::kLiveFree, false },
        { "mix_a",  op(a, { "h0", "s4" }, { "pre_a", "post", "comb", "s4" }), 2, rad::kLiveFree, false },
        { "o_b",    LiveOp{},                                         1, rad::kLiveFree, false },
        { "ar",     ar,                                               1, rad::kLiveForeign, true },
        { "pcn_f",  op(a, { "attn_out", "h0", "post", "comb" }, { "h1", "pre_a", "xcol", "act" }), 1, rad::kLiveForeign, true },
        { "router", op(a, { "act" }, { "logits" }),                   1, rad::kLiveClash, false },
        { "topk",   op(a, { "logits" }, { "ids", "ew" }),             1, rad::kLiveClash, false },
        { "scatter",op(a, { "ids" }, { "sorted", "eoff", "cnt" }),    1, rad::kLiveClash, false },
    };
    /* o_a's two column groups: slices of o and of oa; o_b reads all of oa. */
    const auto o = a.at("o"), oa = a.at("oa"), out = a.at("attn_out");
    const uintptr_t om = (o.first + o.second) / 2, oam = (oa.first + oa.second) / 2;
    layer[10].o.read(o.first, om);
    layer[10].o.write(oa.first, oam);
    layer[11].o.read(om, o.second);
    layer[11].o.write(oam, oa.second);
    layer[13].o.read(oa.first, oa.second);
    layer[13].o.write(out.first, out.second);

    int freed = 0, unordered = 0;
    std::string got;
    for (Step& st : layer) {
        if (st.foreign_before) is.fq.foreign();   /* the previous step's tail, a link hold or release */
        const Issuer::Out r = is.issue(st.o, st.kernels);
        CHECK_EQ(std::string(rad::live_why_name(r.why)), std::string(rad::live_why_name(st.want)));
        if (r.why != st.want) fprintf(stderr, "    at %s\n", st.name);
        freed += r.why == rad::kLiveFree;
        unordered += r.unordered;
        if (r.why == rad::kLiveFree) got += std::string(got.empty() ? "" : " ") + st.name;
    }
    CHECK_EQ(freed, 6);
    CHECK_EQ(unordered, 6);
    CHECK_EQ(got, std::string("kv_g kv_f kv_st o_a1 mix_a o_b"));
    /* mix_a's second dispatch kept its bit, so o_b ran beside it alone, never past it. */
    CHECK(is.fq.unord_run <= rad::kTapeOverlapRun);
}

TEST(the_dump_only_arm_changes_no_packet) {
    Arena a;
    Issuer is;
    is.overlap = false;
    is.fq.foreign();
    is.issue(op(a, { "x" }, { "y" }));
    const Issuer::Out r = is.issue(op(a, { "x" }, { "z" }));
    CHECK_EQ((int)r.why, (int)rad::kLiveFree);   /* counted as would-be free */
    CHECK(!r.unordered);
    for (bool o : is.fq.ordered) CHECK(o);
}

TEST(packets_the_issue_path_did_not_write_order_the_next_launch) {
    Arena a;
    Issuer is;
    is.issue(op(a, { "x" }, { "y" }));                 /* first on the queue: ordered */
    is.fq.foreign();                                   /* an event record, say */
    const Issuer::Out r = is.issue(op(a, { "p" }, { "q" }));
    CHECK_EQ((int)r.why, (int)rad::kLiveForeign);
    CHECK(!r.unordered);
    /* ...and once that launch is ordered, tracking resumes. */
    const Issuer::Out r2 = is.issue(op(a, { "u" }, { "v" }));
    CHECK_EQ((int)r2.why, (int)rad::kLiveFree);
    CHECK(r2.unordered);
}

TEST(a_packet_after_the_launch_inside_its_window_is_caught) {
    Arena a;
    LiveQueues q;
    FakeQueue fq;
    fq.foreign();
    const int s0 = q.claim();
    q.op(s0) = op(a, { "x" }, { "y" });
    q.op(s0).tag = 1;
    LiveQueues::Pos b0 = fq.pos();
    LiveQueues::Win w0 = fq.launch(1, b0.wr, false, 1);
    fq.foreign();                                      /* another thread's packet, mid-launch */
    q.commit(s0, b0, w0, fq.pos());
    const int s1 = q.claim();
    q.op(s1) = op(a, { "p" }, { "q" });
    q.op(s1).tag = 2;
    uint64_t clash = 0;
    int64_t foreign = 0;
    CHECK_EQ((int)q.check(s1, fq.pos(), &clash, &foreign), (int)rad::kLiveForeign);
    CHECK_EQ(foreign, (int64_t)-1);                    /* the queue is no longer known */
}

TEST(opaque_and_host_visible_launches_order_themselves_and_the_next) {
    Arena a;
    Issuer is;
    is.fq.foreign();
    is.issue(op(a, { "x" }, { "y" }));
    const Issuer::Out o = is.issue(opaque_op());
    CHECK_EQ((int)o.why, (int)rad::kLiveOpaque);
    CHECK(!o.unordered);
    const Issuer::Out n = is.issue(op(a, { "p" }, { "q" }));
    CHECK_EQ((int)n.why, (int)rad::kLiveAfter);
    CHECK(!n.unordered);
    LiveOp h = op(a, { "r" }, { "s" });
    h.host = true;
    const Issuer::Out hv = is.issue(h);
    CHECK_EQ((int)hv.why, (int)rad::kLiveHost);
    const Issuer::Out after = is.issue(op(a, { "t" }, { "u" }));
    CHECK_EQ((int)after.why, (int)rad::kLiveAfter);
    const Issuer::Out again = is.issue(op(a, { "v" }, { "w" }));
    CHECK_EQ((int)again.why, (int)rad::kLiveFree);
}

TEST(a_launch_never_joins_one_of_its_own_op) {
    /* The same op on disjoint slices (the stream expansion, an output projection's groups if they
     * were one op): independent by operands, but a row's per-instance state is shared. */
    Arena a;
    Issuer is;
    is.fq.foreign();
    LiveOp first = op(a, { "src" }, { "dst0" });
    first.op = 42;
    is.issue(first);
    LiveOp again = op(a, { "src" }, { "dst1" });
    again.op = 42;
    const Issuer::Out r = is.issue(again);
    CHECK_EQ((int)r.why, (int)rad::kLiveClash);
    CHECK(!r.unordered);
    LiveOp other = op(a, { "src" }, { "dst2" });
    other.op = 43;
    CHECK_EQ((int)is.issue(other).why, (int)rad::kLiveFree);
}

TEST(a_run_stops_at_the_overlap_cap) {
    Arena a;
    Issuer is;
    is.fq.foreign();
    is.issue(op(a, { "in0" }, { "out0" }));
    for (int i = 1; i <= rad::kTapeOverlapRun; ++i) {
        const Issuer::Out r = is.issue(op(a, { "in" + std::to_string(i) }, { "out" + std::to_string(i) }));
        CHECK_EQ((int)r.why, (int)rad::kLiveFree);
        CHECK(r.unordered);
    }
    const Issuer::Out capped = is.issue(op(a, { "inx" }, { "outx" }));
    CHECK_EQ((int)capped.why, (int)rad::kLiveCap);
    CHECK(!capped.unordered);
    /* Of any kTapeOverlapRun + 1 consecutive packets one kept the bit (reserve()'s margin). */
    int run = 0, worst = 0;
    for (bool o : is.fq.ordered) {
        run = o ? 0 : run + 1;
        worst = std::max(worst, run);
    }
    CHECK(worst <= rad::kTapeOverlapRun);
}

TEST(a_multi_dispatch_launch_leaves_only_itself_in_flight) {
    Arena a;
    Issuer is;
    is.fq.foreign();
    is.issue(op(a, { "a" }, { "b" }));
    /* Independent, two kernels: the first unordered, the second ordered behind everything. */
    const Issuer::Out m = is.issue(op(a, { "c" }, { "d" }), 2);
    CHECK(m.unordered);
    CHECK_EQ(is.q.in_flight(is.fq.id), 1);
    /* Reads what the first launch wrote: that one finished before m's second kernel started. */
    const Issuer::Out n = is.issue(op(a, { "b" }, { "e" }));
    CHECK_EQ((int)n.why, (int)rad::kLiveFree);
}

TEST(a_barrier_inside_a_launch_blinds_the_queue) {
    Arena a;
    Issuer is;
    is.fq.foreign();
    is.issue(op(a, { "a" }, { "b" }));
    is.issue(op(a, { "c" }, { "d" }), 1, 1);
    const Issuer::Out n = is.issue(op(a, { "e" }, { "f" }));
    CHECK_EQ((int)n.why, (int)rad::kLiveAfter);
}

TEST(a_launch_writing_nothing_on_its_queue_leaves_it_alone) {
    Arena a;
    Issuer is;
    is.fq.foreign();
    is.issue(op(a, { "a" }, { "b" }));
    is.issue(op(a, { "x" }, { "y" }), 0);              /* n = 0: no packet */
    const Issuer::Out n = is.issue(op(a, { "b" }, { "c" }));
    CHECK_EQ((int)n.why, (int)rad::kLiveClash);        /* still behind the first, which wrote b */
}

TEST(a_launch_whose_slot_was_reused_is_unknown) {
    Arena a;
    LiveQueues q;
    FakeQueue qa, qb;
    qa.id = 1;
    qb.id = 2;
    uint64_t tag = 0;
    auto run = [&](FakeQueue& fq, LiveOp shape) {
        const int s = q.claim();
        q.op(s) = shape;
        q.op(s).tag = ++tag;
        uint64_t clash = 0;
        int64_t foreign = 0;
        const LiveQueues::Pos b = fq.pos();
        const LiveWhy why = q.check(s, b, &clash, &foreign);
        const LiveQueues::Win w = fq.launch(q.op(s).tag, b.wr, why == rad::kLiveFree, 1);
        q.commit(s, b, w, fq.pos());
        return why;
    };
    qb.foreign();
    run(qb, op(a, { "lane1_in" }, { "lane1_out" }));
    qa.foreign();
    for (int i = 0; i < LiveQueues::kRing + 2; ++i)
        run(qa, op(a, { "a" + std::to_string(i) }, { "b" + std::to_string(i) }));
    /* lane 1's in-flight launch lost its slot to lane 0's: it could touch anything now. */
    CHECK_EQ((int)run(qb, op(a, { "z" }, { "zz" })), (int)rad::kLiveAfter);
}

TEST(the_switches_stay_off_where_the_backend_has_no_live_path) {
    /* The unit tests run on the host backend, which writes no packets. */
    if (rad::live_supported()) SKIP("the active backend has a live issue path");
    setenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP", "1", 1);
    setenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE", "4", 1);
    rad::LiveChain c;
    c.init(0, nullptr);
    unsetenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP");
    unsetenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE");
    CHECK(!c.on());
    CHECK(c.report().empty());
    RadBatch b{};
    b.phase = RAD_PHASE_DECODE;
    c.pass_begin(&b);
    CHECK(!c.checking());
    CHECK(!c.overlap());
    CHECK(!c.tracing());
    c.pass_end();
}

TEST(the_switches_off_mean_nothing_runs) {
    unsetenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP");
    unsetenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE");
    rad::LiveChain c;
    c.init(0, nullptr);
    CHECK(!c.on());
    RadBatch b{};
    b.phase = RAD_PHASE_DECODE;
    c.pass_begin(&b);
    CHECK(!c.checking());
    c.pass_end();
    CHECK(c.report().empty());
}

RAD_TEST_MAIN()

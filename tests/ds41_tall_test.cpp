/* ds41_tall_test.cpp -- RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL on the host: passes of 17 to 48 tokens
 * on DeepSeek-V4.1's EP counts path and CPU path (core/place/ep_route_counts.h, FileStager).
 *
 * The switch raises the counts contract's pass limit from 16 to 48 tokens, and the file stager's to
 * the step (--max-num-batched-tokens), with everything it sizes by that limit: the CPU path's pinned
 * token rows, its sorted order with the expert offsets behind it, its output rows, and the fetch
 * lists. tall_init admits a step only when its passes give a 16-token pass's bits: every band of
 * every routed MXFP4 GEMM on the decode band's kernel (moe_gemm_ggml_mxfp4_gemv, or its twin
 * moe_gemm_ggml_mxfp4_gemv_tall above 16 tokens, which libr4d bounds at the tallest pass its kernels
 * keep exact at), every other op on one kernel at every height or on gemm_nt's m16 / m64 pair, and a
 * step of at most 48. The kernels' side -- the twin's rows equal the gemv's over the same tokens in
 * passes of 16, the norms' width -- is tests/r4d_emu/test_ds41tall.cpp on the host and
 * ds41_mxfp4_device_check `tall` / `sweep` on a card. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/ep_route_counts.h"
#include "place/stager.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace rad {

/* FileStager's private side, for these cases only (stager.h names it a friend). */
struct EpCountsHostProbe {
    static int tall_init(FileStager& fs, const Program& p) { return fs.tall_init(p); }
    static int64_t ep_rows(const FileStager& fs) { return fs.ep_rows_; }
    static int64_t sorted_max(const FileStager& fs) { return fs.hx_sorted_max(); }

    /* The CPU path's tables as hx_init_impl makes them at the stager's limit, and one routed layer
     * with the decode shape (top_k 6 over 96 local experts) whose gate_up was seen. */
    static bool tables(FileStager& fs, int64_t pass_tok) {
        const int64_t lim = fs.ep_rows_;
        fs.hx_alloc_rows_ = lim;
        fs.hx_x_ = (uint16_t*)rad_dev_alloc(lim * HostExpertPool::kEmbd * 2, RAD_MEM_HOST_PINNED);
        fs.hx_idx_ = (int32_t*)rad_dev_alloc((fs.hx_sorted_max() + 128) * 4, RAD_MEM_HOST_PINNED);
        fs.hx_out_ = (uint16_t*)rad_dev_alloc(fs.hx_sorted_max() * HostExpertPool::kEmbd * 2, RAD_MEM_HOST_PINNED);
        fs.counts_host_ = (int32_t*)rad_dev_alloc(512 * 4, RAD_MEM_HOST_PINNED);
        fs.counts_cap_ = 512;
        StagedLayer L;
        L.layer = 7;
        fs.layers_.assign(1, L);
        FileStager::HxOps o;
        o.gu = 1;
        o.n_expert = 96;
        o.top_k = 6;
        fs.hx_ops_.assign(1, o);
        fs.hx_li_ = 0;
        fs.pass_tok_ = pass_tok;
        return fs.hx_x_ && fs.hx_idx_ && fs.hx_out_ && fs.counts_host_;
    }
    static int32_t* sorted(FileStager& fs) { return fs.hx_idx_; }
    static int32_t* off(FileStager& fs) { return fs.hx_idx_ + fs.hx_sorted_max(); }
    static int32_t* counts(FileStager& fs) { return fs.counts_host_; }
    static const uint16_t* x(const FileStager& fs) { return fs.hx_x_; }
    static uint16_t* out(const FileStager& fs) { return fs.hx_out_; }
    /* hx_jobs_for for local expert e: the rows, or a negative status; jobs and rows appended. */
    static int jobs_for(FileStager& fs, int e) { return fs.hx_jobs_for(e, HostExpertPool::Job{}); }
    static std::vector<HostExpertPool::Job>& jobs(FileStager& fs) { return fs.hx_jobs_; }
    static std::vector<int32_t>& rows(FileStager& fs) { return fs.hx_rows_; }
};

}  // namespace rad

using namespace rad;

namespace {

/* Forces the switch for one case and gives the environment's back at its end. */
struct Tall {
    explicit Tall(int state) { ds41_exact_tall_set_for_test(state); }
    ~Tall() { ds41_exact_tall_set_for_test(-2); }
};

/* A counts witness as Ctx::note_ep_route_producer records it for rank 1 of four. */
EpRouteCounts witness(const void* program, void* stream, int64_t rows) {
    EpRouteCounts v;
    v.program = program;
    static const int32_t row[96] = {};
    v.row = row;
    v.stream = stream;
    v.generation = 3;
    v.producer_serial = 9;
    v.layer = 12;
    v.rank = 1;
    v.world = 4;
    v.global_experts = 384;
    v.local_experts = 96;
    v.expert_base = 96;
    v.top_k = 6;
    v.rows = rows;
    v.successfully_issued = true;
    return v;
}

/* Rank-local counts of a pass of `rows` tokens: every token on expert 5 (a run as long as the
 * pass), the rest of its top-6 spread over the ranks. */
std::vector<int32_t> counts_of(int64_t rows) {
    std::vector<int32_t> c(96, 0);
    for (int64_t t = 0; t < rows; ++t) {
        ++c[5];
        ++c[(size_t)(10 + t % 80)];
    }
    return c;
}

/* Kernel rows by name, for the bands of a resolved op to point at; the twin as libr4d declares it
 * (its bound on M: 32, or 48). */
enum { kGemv, kTall32, kTall48, kTiled, kQGemv, kQTiled, kM16, kM64, kTiledNt, kNorm, kRows };
struct Rows {
    RadKernelInfo info[kRows] = {};
    KernelRow row[kRows];
    const RadConstraint le32[2] = { { "M", RAD_C_GE, 1, nullptr }, { "M", RAD_C_LE, 32, nullptr } };
    const RadConstraint le48[2] = { { "M", RAD_C_GE, 1, nullptr }, { "M", RAD_C_LE, 48, nullptr } };
    Rows() {
        const char* names[kRows] = { "moe_gemm_ggml_mxfp4_gemv", "moe_gemm_ggml_mxfp4_gemv_tall",
                                     "moe_gemm_ggml_mxfp4_gemv_tall", "moe_gemm_ggml_mxfp4_tiled",
                                     "moe_gemm_ggml_qk32_gemv", "moe_gemm_ggml_qk32_tiled",
                                     "gemm_bf16_nt_m16", "gemm_bf16_nt_m64", "gemm_nt_ggml_mxfp4_tiled",
                                     "rmsnorm_bf16" };
        for (int i = 0; i < kRows; ++i) {
            info[i].name = names[i];
            row[i].info = &info[i];
            row[i].plugin = "libr4d";
        }
        info[kTall32].constraints = le32;
        info[kTall32].n_constraints = 2;
        info[kTall48].constraints = le48;
        info[kTall48].n_constraints = 2;
    }
};

/* An op `name` with M ranged over [1, max_tok], each band (its upper bound, the row it resolved to
 * on the device) as given; moe_gemm_ggml at the routed experts' shape (`type`, 4608 x 5120). */
OpInfo an_op(const Rows& r, const char* name, const char* type, int64_t max_tok,
             std::vector<std::pair<int64_t, int>> bands) {
    OpInfo o;
    o.op = name;
    if (type) o.base.set_s("type", type);
    o.base.set_i("N", 4608);
    o.base.set_i("K", 5120);
    o.ranged_key = "M";
    o.range_lo = 1;
    o.range_hi = max_tok;
    int64_t lo = 0;
    for (const auto& b : bands) {
        Band band;
        band.hi = b.first;
        band.dom[RAD_DOMAIN_DEVICE].row = &r.row[b.second];
        band.span = lo ? "M in (" + std::to_string(lo) + ", " + std::to_string(b.first) + "]"
                       : "M <= " + std::to_string(b.first);
        lo = b.first;
        o.bands.push_back(band);
    }
    return o;
}
OpInfo moe_op(const Rows& r, const char* type, int64_t max_tok, std::vector<std::pair<int64_t, int>> bands) {
    return an_op(r, "moe_gemm_ggml", type, max_tok, std::move(bands));
}

/* A program of those ops, steps of at most max_tok tokens (made in place: a Program is not moved). */
void program(Program& p, int64_t max_tok, std::vector<OpInfo> ops) {
    p.ctx.max_tok = max_tok;
    p.ops.clear();
    p.ops.emplace_back();   /* handle 0 is no op */
    for (OpInfo& o : ops) p.ops.push_back(std::move(o));
}

/* tall_init's status on a program, and what it said. */
int init_of(FileStager& fs, const Program& p, std::string* err) {
    int rc = 0;
    *err = stderr_of([&] { rc = EpCountsHostProbe::tall_init(fs, p); });
    return rc;
}

}  // namespace

TEST(the_switch_raises_the_counts_contract_from_16_to_48_tokens) {
    int program_tag = 0, stream_tag = 0;
    const void* prog = &program_tag;
    void* stream = &stream_tag;
    {
        Tall off(0);
        CHECK_EQ(ep_route_max_rows(), (int64_t)16);
        CHECK(ep_route_current(witness(prog, stream, 16), prog, stream, 3, 12, 16));
        CHECK(!ep_route_current(witness(prog, stream, 17), prog, stream, 3, 12, 17));
        int64_t total = 0;
        auto c16 = counts_of(16), c17 = counts_of(17);
        CHECK(ep_route_counts_valid(witness(prog, stream, 16), c16.data(), 96, &total));
        CHECK_EQ(total, (int64_t)32);
        CHECK(!ep_route_counts_valid(witness(prog, stream, 17), c17.data(), 96, &total));
    }
    {
        Tall on(1);
        CHECK_EQ(ep_route_max_rows(), (int64_t)48);
        for (int64_t rows : { 1, 16, 17, 32, 33, 48 }) {
            CHECK(ep_route_current(witness(prog, stream, rows), prog, stream, 3, 12, rows));
            auto c = counts_of(rows);
            int64_t total = -1;
            CHECK(ep_route_counts_valid(witness(prog, stream, rows), c.data(), 96, &total));
            CHECK_EQ(total, 2 * rows);
        }
        CHECK(!ep_route_current(witness(prog, stream, 49), prog, stream, 3, 12, 49));
        auto c49 = counts_of(49);
        int64_t total = 0;
        CHECK(!ep_route_counts_valid(witness(prog, stream, 49), c49.data(), 96, &total));
        /* Still a pass's own count: an expert past the pass's rows, or more slots than top_k, is
         * not a counts row of this pass. */
        auto c = counts_of(48);
        c[5] = 49;
        CHECK(!ep_route_counts_valid(witness(prog, stream, 48), c.data(), 96, &total));
        c = counts_of(48);
        for (int e = 20; e < 80; ++e) c[(size_t)e] = 48;
        CHECK(!ep_route_counts_valid(witness(prog, stream, 48), c.data(), 96, &total));
    }
    {
        Tall bad(-1);   /* a value other than 0 or 1: off for the contract, refused at init */
        CHECK_EQ(ep_route_max_rows(), (int64_t)16);
    }
}

TEST(tall_init_takes_the_step_when_every_routed_band_is_on_the_gemv_rows) {
    Rows r;
    /* Off: the limit stays 16 whatever the bands and the step. */
    {
        Tall off(0);
        FileStager fs;
        Program p;
        program(p, 48, { moe_op(r, "mxfp4", 48, { { 16, kGemv }, { 48, kTiled } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)16);
        CHECK_EQ(EpCountsHostProbe::sorted_max(fs), (int64_t)96);
    }
    Tall on(1);
    /* The gemv to 16 and the twin (bounded at 32) to the step, on both projections: passes of up
     * to 32 tokens, the tables sized for 32. */
    {
        FileStager fs;
        Program p;
        program(p, 32, { moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }),
                         moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)32);
        CHECK_EQ(EpCountsHostProbe::sorted_max(fs), (int64_t)192);
    }
    /* A step between: the limit is the step. */
    {
        FileStager fs;
        Program p;
        program(p, 24, { moe_op(r, "mxfp4", 24, { { 16, kGemv }, { 24, kTall32 } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)24);
    }
    /* The twin bounded at 48 (a libr4d that pins the norms' width): passes of up to 48. */
    {
        FileStager fs;
        Program p;
        program(p, 48, { moe_op(r, "mxfp4", 48, { { 16, kGemv }, { 48, kTall48 } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)48);
        CHECK_EQ(EpCountsHostProbe::sorted_max(fs), (int64_t)288);
    }
    /* A step of at most 16 tokens never reaches a band above: admitted with or without the twin,
     * at the old limit. */
    {
        FileStager fs;
        Program p;
        program(p, 16, { moe_op(r, "mxfp4", 16, { { 16, kGemv } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)16);
        CHECK_EQ(EpCountsHostProbe::sorted_max(fs), (int64_t)96);
    }
}

TEST(tall_init_refuses_a_step_whose_passes_would_change_arithmetic) {
    Rows r;
    std::string err;
    Tall on(1);
    /* A step taller than the twin reaches: the band above resolves to the tiled row. Refused,
     * naming the row and the step this libr4d keeps exact. */
    {
        FileStager fs;
        Program p;
        program(p, 48, { moe_op(r, "mxfp4", 48, { { 16, kGemv }, { 32, kTall32 }, { 48, kTiled } }) });
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "moe_gemm_ggml_mxfp4_tiled"));
        CHECK(has(err, "M in (32, 48]"));
        CHECK(has(err, "up to 32 tokens"));
        CHECK(has(err, "--max-num-batched-tokens 32"));
    }
    /* No twin (a libr4d without it, or loaded without the switch): the tiled row above 16. */
    {
        FileStager fs;
        Program p;
        program(p, 32, { moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }),
                         moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTiled } }) });
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "op 2"));
        CHECK(has(err, "moe_gemm_ggml_mxfp4_tiled"));
        CHECK(has(err, "with the twin"));
    }
    /* Nothing resolved on the device in a band: refused. */
    {
        FileStager fs;
        Program p;
        program(p, 32, { moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }) });
        p.ops[1].bands[1].dom[RAD_DOMAIN_DEVICE].row = nullptr;
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "nothing"));
    }
    /* A step above 48: refused before any band is looked at. */
    {
        FileStager fs;
        Program p;
        program(p, 64, { moe_op(r, "mxfp4", 64, { { 16, kGemv }, { 64, kTall48 } }) });
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "--max-num-batched-tokens 64"));
    }
    /* Another op on two kernels: refused by name unless it is gemm_nt's m16 / m64 pair, which is
     * one split law (m16 forwards to m64). An op on one kernel in every band is fine. */
    {
        FileStager fs;
        Program p;
        program(p, 32, { moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }),
                         an_op(r, "gemm_nt", nullptr, 32, { { 16, kM16 }, { 32, kM64 } }),
                         an_op(r, "rmsnorm", nullptr, 32, { { 32, kNorm } }),
                         an_op(r, "rmsnorm", nullptr, 32, { { 16, kNorm }, { 32, kNorm } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)32);
        Program q;
        program(q, 32, { moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }),
                         an_op(r, "gemm_nt_ggml", "mxfp4", 32, { { 16, kGemv }, { 32, kTiledNt } }) });
        CHECK_EQ(init_of(fs, q, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "op 2 (gemm_nt_ggml)"));
        CHECK(has(err, "gemm_nt_ggml_mxfp4_tiled"));
        /* Another block type's grouped GEMM is no routed MXFP4 GEMM, but its gemv / tiled split is
         * a height change all the same. */
        Program s;
        program(s, 32, { moe_op(r, "q8_0", 32, { { 16, kQGemv }, { 32, kQTiled } }) });
        CHECK_EQ(init_of(fs, s, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "moe_gemm_ggml_qk32_tiled"));
        /* The pair named the other way round is the same pair; gemm_nt's m16 / m64 on another op
         * is not. */
        Program u;
        program(u, 32, { an_op(r, "gemm_nt", nullptr, 32, { { 16, kM64 }, { 32, kM16 } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, u));
        program(u, 32, { an_op(r, "gemm_nt_q", nullptr, 32, { { 16, kM16 }, { 32, kM64 } }) });
        CHECK_EQ(init_of(fs, u, &err), (int)RAD_E_INVAL);
    }
    /* A value other than 0 or 1 is refused rather than read as either. */
    {
        Tall bad(-1);
        FileStager fs;
        Program p;
        program(p, 32, { moe_op(r, "mxfp4", 32, { { 16, kGemv }, { 32, kTall32 } }) });
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
    }
}

/* The CPU path's jobs for a tall pass: hx_jobs_for reads the sorted order and the offsets where the
 * tables put them at the stager's limit (offsets after 6 sorted slots a token of the step), cuts
 * each expert's rows into jobs of at most HostExpertPool::max_rows(), points each job row at its
 * token's row and its sorted slot's output row, and refuses a run that disagrees with the expert's
 * count. */
TEST(a_tall_pass_cuts_its_jobs_from_the_step_sized_tables) {
    Tall on(1);
    for (int64_t step : { 32, 48 })
        for (int64_t T : { 17, 32, 33, 48 }) {
            if (T > step) continue;
            FileStager fs;
            {
                Rows r;
                Program p;
                program(p, step, { moe_op(r, "mxfp4", step, { { 16, kGemv }, { step, kTall48 } }) });
                CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
            }
            REQUIRE(EpCountsHostProbe::tables(fs, T));
            /* Each token on local expert 5 and on two others; its other three slots on other ranks. */
            std::mt19937 rng((uint32_t)(T * 100 + step));
            std::vector<std::vector<int32_t>> by(96);
            for (int64_t t = 0; t < T; ++t) {
                std::vector<int> e = { 5 };
                while (e.size() < 3) {
                    const int x = (int)(rng() % 96);
                    if (std::find(e.begin(), e.end(), x) == e.end()) e.push_back(x);
                }
                for (int k = 0; k < 3; ++k) by[(size_t)e[(size_t)k]].push_back((int32_t)(t * 6 + 2 * k));
            }
            int32_t* sorted = EpCountsHostProbe::sorted(fs);
            int32_t* off = EpCountsHostProbe::off(fs);
            int32_t* cnt = EpCountsHostProbe::counts(fs);
            int n = 0;
            for (int e = 0; e < 96; ++e) {
                off[e] = n;
                cnt[e] = (int32_t)by[(size_t)e].size();
                std::sort(by[(size_t)e].begin(), by[(size_t)e].end());
                for (int32_t f : by[(size_t)e]) sorted[n++] = f;
            }
            off[96] = n;
            for (int i = n; i < T * 6; ++i) sorted[i] = -1;   /* the other ranks' slots */
            /* The offsets sit after the step's sorted slots, not a 16-token table's 96. */
            CHECK_EQ(off - sorted, (ptrdiff_t)(step * 6));
            const uint16_t* x = EpCountsHostProbe::x(fs);
            uint16_t* y = EpCountsHostProbe::out(fs);
            int rows_seen = 0;
            for (int e = 0; e < 96; ++e) {
                if (!cnt[e]) continue;
                EpCountsHostProbe::jobs(fs).clear();
                EpCountsHostProbe::rows(fs).clear();
                const int got = EpCountsHostProbe::jobs_for(fs, e);
                CHECK_EQ(got, cnt[e]);
                int in_jobs = 0;
                for (const HostExpertPool::Job& j : EpCountsHostProbe::jobs(fs)) {
                    CHECK(j.more >= 0 && j.more < HostExpertPool::max_rows());
                    CHECK_EQ(j.cost, HostExpertPool::job_cost(j.more + 1));
                    for (int m = 0; m <= j.more; ++m) {
                        const uint16_t* jx = m ? j.xs[m - 1] : j.x;
                        uint16_t* jy = m ? j.ys[m - 1] : j.y;
                        const int i = EpCountsHostProbe::rows(fs)[(size_t)(in_jobs + m)];
                        CHECK(i >= off[e] && i < off[e + 1]);
                        CHECK(jx == x + (int64_t)(sorted[i] / 6) * HostExpertPool::kEmbd);
                        CHECK(jy == y + (int64_t)i * HostExpertPool::kEmbd);
                        CHECK(sorted[i] / 6 < T);
                    }
                    in_jobs += j.more + 1;
                }
                CHECK_EQ(in_jobs, cnt[e]);
                rows_seen += in_jobs;
            }
            CHECK_EQ(rows_seen, n);
            CHECK_EQ(cnt[5], (int32_t)T);   /* a run of every token of the pass */
            /* A count that disagrees with the run is refused. */
            ++cnt[5];
            EpCountsHostProbe::jobs(fs).clear();
            EpCountsHostProbe::rows(fs).clear();
            int rc = 0;
            (void)stderr_of([&] { rc = EpCountsHostProbe::jobs_for(fs, 5); });
            CHECK(rc < 0);
        }
}

RAD_TEST_MAIN()

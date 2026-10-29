/* ds41_tall_test.cpp -- RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL on the host: passes of 17 to 48 tokens
 * on DeepSeek-V4.1's EP counts path and CPU path (core/place/ep_route_counts.h, FileStager); and
 * RADIANCE_EXPERIMENTAL_DS41_LCP=<C> (large-chunk prefill): the switch, the counts contract at C,
 * FileStager::lcp_init's admitted kernels and refusals, the routed-only reads of a chunk and its jobs
 * cut from the C-sized tables.
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
    static int64_t lcp(const FileStager& fs) { return fs.lcp_; }
    static int64_t lcp_reach(const FileStager& fs) { return fs.lcp_reach_; }
    static const std::string& lcp_pairs(const FileStager& fs) { return fs.lcp_pairs_; }
    static bool routed(const FileStager& fs) { return fs.routed_; }
    static void set_capped(FileStager& fs, bool on) { fs.hx_capped_ = on; }
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

    /* Large-chunk prefill's per-pass state: the admission row, the per-row cut's rows by model layer. */
    static void set_admit(FileStager& fs, int32_t row) { fs.lcp_admit_row_ = row; }
    static void set_row0(FileStager& fs, std::vector<int32_t> r) { fs.pass_row0_ = std::move(r); }
    static int64_t layer_r0(const FileStager& fs, int32_t li) { return fs.layer_r0(li); }
    static int64_t layer_rows(const FileStager& fs, int32_t li) { return fs.layer_rows(li); }
    static bool admits(const FileStager& fs, int32_t li, int32_t e) { return fs.hx_admits(li, e); }
    /* Misses for units 0..n-1, `admit` saying which the cache admits; then hx_queue_inserts over all. */
    static void misses(FileStager& fs, const std::vector<uint8_t>& admit) {
        fs.hx_miss_.clear();
        fs.hx_miss_admit_ = admit;
        for (size_t i = 0; i < admit.size(); ++i) fs.hx_miss_.push_back(fs.hx_planes_get((int32_t)i));
    }
    static void queue_inserts(FileStager& fs) {
        std::vector<size_t> idx(fs.hx_miss_.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        fs.hx_queue_inserts(idx.data(), idx.size());
    }
    static size_t queued(FileStager& fs) {
        std::lock_guard<std::mutex> lk(fs.hx_bg_mu_);
        return fs.hx_bg_q_.size();
    }
    static void drop_queue(FileStager& fs) {
        std::lock_guard<std::mutex> lk(fs.hx_bg_mu_);
        fs.hx_bg_q_.clear();
    }
    static void set_insertq(FileStager& fs, int n) { fs.hx_lcp_insertq_ = n; }
    static const FileStager::HostStats& stats(const FileStager& fs) { return fs.hx_stats_; }
};

}  // namespace rad

using namespace rad;

namespace {

/* Forces the switch for one case and gives the environment's back at its end. */
struct Tall {
    explicit Tall(int state) { ds41_exact_tall_set_for_test(state); }
    ~Tall() { ds41_exact_tall_set_for_test(-2); }
};
/* LCP at `c` (C, 0 or -1) for one case. */
struct Lcp {
    explicit Lcp(int64_t c) { ds41_lcp_set_for_test(c); }
    ~Lcp() { ds41_lcp_set_for_test(-2); }
};
/* An environment variable set (or unset, null) for one case. */
struct Env {
    std::string k, old;
    bool had = false;
    Env(const char* key, const char* v) : k(key) {
        const char* o = std::getenv(key);
        had = o != nullptr;
        if (had) old = o;
        if (v) setenv(key, v, 1);
        else unsetenv(key);
    }
    ~Env() {
        if (had) setenv(k.c_str(), old.c_str(), 1);
        else unsetenv(k.c_str());
    }
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
enum { kGemv, kTall32, kTall48, kTiled, kQGemv, kQTiled, kM16, kM64, kTiledNt, kNorm,
       kTall2048, kBfTiled, kAr1, kAr2, kAr2Lane, kAr2Of3, kNorm2, kRows };
struct Rows {
    RadKernelInfo info[kRows] = {};
    KernelRow row[kRows];
    const RadConstraint le32[2] = { { "M", RAD_C_GE, 1, nullptr }, { "M", RAD_C_LE, 32, nullptr } };
    const RadConstraint le48[2] = { { "M", RAD_C_GE, 1, nullptr }, { "M", RAD_C_LE, 48, nullptr } };
    const RadConstraint le2048[2] = { { "M", RAD_C_GE, 1, nullptr }, { "M", RAD_C_LE, 2048, nullptr } };
    Rows() {
        const char* names[kRows] = { "moe_gemm_ggml_mxfp4_gemv", "moe_gemm_ggml_mxfp4_gemv_tall",
                                     "moe_gemm_ggml_mxfp4_gemv_tall", "moe_gemm_ggml_mxfp4_tiled",
                                     "moe_gemm_ggml_qk32_gemv", "moe_gemm_ggml_qk32_tiled",
                                     "gemm_bf16_nt_m16", "gemm_bf16_nt_m64", "gemm_nt_ggml_mxfp4_tiled",
                                     "rmsnorm_bf16", "moe_gemm_ggml_mxfp4_gemv_tall", "gemm_bf16_nt_tiled",
                                     "ar_oneshot_4rank_exact", "ar_twoshot_4rank_exact",
                                     "ar_twoshot_4rank_exact_lane", "ar_twoshot_3rank_exact", "rmsnorm_chain_bf16" };
        for (int i = 0; i < kRows; ++i) {
            info[i].name = names[i];
            row[i].info = &info[i];
            row[i].plugin = "libr4d";
        }
        info[kTall32].constraints = le32;
        info[kTall32].n_constraints = 2;
        info[kTall48].constraints = le48;
        info[kTall48].n_constraints = 2;
        info[kTall2048].constraints = le2048;
        info[kTall2048].n_constraints = 2;
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

/* ================================================================== large-chunk prefill (LCP) */

TEST(lcp_raises_the_counts_contract_to_the_chunk) {
    int program_tag = 0, stream_tag = 0;
    const void* prog = &program_tag;
    void* stream = &stream_tag;
    {
        Lcp on(1024);
        CHECK_EQ(ds41_lcp_rows(), (int64_t)1024);
        CHECK_EQ(ep_route_max_rows(), (int64_t)1024);
        for (int64_t rows : { 1, 16, 17, 49, 333, 1024 }) {
            CHECK(ep_route_current(witness(prog, stream, rows), prog, stream, 3, 12, rows));
            auto c = counts_of(rows);
            int64_t total = -1;
            CHECK(ep_route_counts_valid(witness(prog, stream, rows), c.data(), 96, &total));
            CHECK_EQ(total, 2 * rows);
        }
        CHECK(!ep_route_current(witness(prog, stream, 1025), prog, stream, 3, 12, 1025));
        /* EXACT_TALL beside it does not lower it. */
        Tall xt(1);
        CHECK_EQ(ep_route_max_rows(), (int64_t)1024);
    }
    {
        Lcp on(2048);
        CHECK_EQ(ep_route_max_rows(), (int64_t)2048);
    }
    {
        Lcp bad(-1);   /* a value outside the range: no limit raised, refused at init */
        CHECK_EQ(ds41_lcp_rows(), (int64_t)0);
        CHECK_EQ(ep_route_max_rows(), (int64_t)16);
    }
    {
        Lcp off(0);
        CHECK_EQ(ep_route_max_rows(), (int64_t)16);
    }
}

namespace {
/* DeepSeek-V4.1's shape at chunks of C: two routed GEMMs on the gemv and the twin, gemm_nt on m16 /
 * m64 / tiled, an exact all-reduce one-shot to 102 rows and two-shot above, a norm on one kernel. */
void lcp_program(Program& p, const Rows& r, int64_t C, int twin = kTall2048) {
    const int64_t m64_top = std::min<int64_t>(64, C), ar_top = std::min<int64_t>(102, C);
    std::vector<std::pair<int64_t, int>> nt = { { 16, kM16 }, { m64_top, kM64 } };
    if (C > 64) nt.push_back({ C, kBfTiled });
    std::vector<std::pair<int64_t, int>> ar = { { ar_top, kAr1 } };
    if (C > 102) ar.push_back({ C, kAr2 });
    program(p, C, { moe_op(r, "mxfp4", C, { { 16, kGemv }, { C, twin } }),
                    moe_op(r, "mxfp4", C, { { 16, kGemv }, { C, twin } }),
                    an_op(r, "gemm_nt", nullptr, C, nt),
                    an_op(r, "all_reduce", nullptr, C, ar),
                    an_op(r, "rmsnorm", nullptr, C, { { 32, kNorm }, { C, kNorm } }) });
}
}  // namespace

TEST(lcp_init_admits_chunks_on_the_proven_kernels) {
    Rows r;
    for (int64_t C : { 64, 256, 1024, 2048 }) {
        Lcp on(C);
        Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
        FileStager fs;
        Program p;
        lcp_program(p, r, C);
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), C);
        CHECK_EQ(EpCountsHostProbe::lcp(fs), C);
        CHECK_EQ(EpCountsHostProbe::sorted_max(fs), 6 * C);
        CHECK_EQ(EpCountsHostProbe::lcp_reach(fs), (int64_t)2048);
        const std::string& pairs = EpCountsHostProbe::lcp_pairs(fs);
        CHECK(has(pairs, "gemm_nt bf16"));
        if (C > 64) CHECK(has(pairs, "gemm_bf16_nt_tiled"));
        if (C > 102) CHECK(has(pairs, "all_reduce exact (4 ranks) on ar_oneshot_4rank_exact, ar_twoshot_4rank_exact"));
    }
    /* EXACT_TALL=1 beside it is fine; the narrow two-shot is the same family. */
    {
        Lcp on(1024);
        Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", "1");
        FileStager fs;
        Program p;
        program(p, 1024, { moe_op(r, "mxfp4", 1024, { { 16, kGemv }, { 1024, kTall2048 } }),
                           an_op(r, "all_reduce", nullptr, 1024, { { 102, kAr1 }, { 512, kAr2 }, { 1024, kAr2Lane } }) });
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        CHECK_EQ(EpCountsHostProbe::ep_rows(fs), (int64_t)1024);
        CHECK(has(EpCountsHostProbe::lcp_pairs(fs), "ar_twoshot_4rank_exact_lane"));
    }
}

TEST(lcp_init_refuses_what_was_not_proven_at_the_chunk) {
    Rows r;
    std::string err;
    Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
    Lcp on(1024);
    auto refused = [&](const Program& p, const char* what) {
        FileStager fs;
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "RADIANCE_EXPERIMENTAL_DS41_LCP=1024 (large-chunk prefill) refused"));
        CHECK(has(err, what));
        CHECK_EQ(EpCountsHostProbe::lcp(fs), (int64_t)0);
    };
    /* A libr4d whose twin stops at 48 (one that knows EXACT_TALL only): the band above is tiled. */
    {
        Program p;
        program(p, 1024, { moe_op(r, "mxfp4", 1024, { { 16, kGemv }, { 48, kTall48 }, { 1024, kTiled } }) });
        refused(p, "moe_gemm_ggml_mxfp4_tiled");
    }
    /* The twin's band reaching the step while its row says 48 (a table that disagrees): refused by
     * the row's own bound. */
    {
        Program p;
        program(p, 1024, { moe_op(r, "mxfp4", 1024, { { 16, kGemv }, { 1024, kTall48 } }) });
        refused(p, "reaches 48 tokens");
    }
    /* No twin at all. */
    {
        Program p;
        program(p, 1024, { moe_op(r, "mxfp4", 1024, { { 16, kGemv }, { 1024, kTiled } }) });
        refused(p, "not to the decode band's kernel");
    }
    /* Another op on two kernels outside the pairs; two all-reduce families; a pair on another op. */
    {
        Program p;
        lcp_program(p, r, 1024);
        p.ops.push_back(an_op(r, "rmsnorm", nullptr, 1024, { { 64, kNorm }, { 1024, kNorm2 } }));
        refused(p, "rmsnorm_chain_bf16");
    }
    {
        Program p;
        program(p, 1024, { moe_op(r, "mxfp4", 1024, { { 16, kGemv }, { 1024, kTall2048 } }),
                           an_op(r, "all_reduce", nullptr, 1024, { { 102, kAr1 }, { 1024, kAr2Of3 } }) });
        refused(p, "ar_twoshot_3rank_exact");
    }
    {
        Program p;
        program(p, 1024, { moe_op(r, "mxfp4", 1024, { { 16, kGemv }, { 1024, kTall2048 } }),
                           an_op(r, "gemm_nt_q", nullptr, 1024, { { 16, kM16 }, { 1024, kM64 } }) });
        refused(p, "op 2 (gemm_nt_q)");
    }
    /* The step is not C. */
    {
        Program p;
        lcp_program(p, r, 1024);
        p.ctx.max_tok = 512;
        refused(p, "--max-num-batched-tokens is 512");
    }
    /* Speculation. */
    {
        Program p;
        lcp_program(p, r, 1024);
        p.ctx.max_spec = 3;
        refused(p, "speculation");
    }
    /* EXACT_TALL set to anything but 1 beside it. */
    {
        Env x0("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", "0");
        Program p;
        lcp_program(p, r, 1024);
        refused(p, "EXACT_TALL=0 beside it");
    }
    /* The fetch lists and the packed VRAM residents. */
    {
        Env fl("RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST", "1");
        Program p;
        lcp_program(p, r, 1024);
        refused(p, "FETCH_LIST=1");
    }
    {
        Env vp("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED", "1");
        Program p;
        lcp_program(p, r, 1024);
        refused(p, "VRAM_PACKED=1");
    }
    /* No routed MXFP4 GEMM in the program. */
    {
        Program p;
        program(p, 1024, { an_op(r, "rmsnorm", nullptr, 1024, { { 1024, kNorm } }) });
        refused(p, "no routed MXFP4 GEMM");
    }
    /* A value outside the switch's range. */
    {
        Lcp bad(-1);
        FileStager fs;
        Program p;
        lcp_program(p, r, 1024);
        CHECK_EQ(init_of(fs, p, &err), (int)RAD_E_INVAL);
        CHECK(has(err, "a multiple of 16"));
    }
}

/* A chunk reads only the experts it routes to under LCP (the file buffers are capped, so the card's
 * copy path could not hold a whole layer); without LCP a pass over 64 tokens reads the layer. */
TEST(an_lcp_chunk_reads_only_its_routed_experts) {
    Rows r;
    Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
    for (int64_t C : { 256, 1024 }) {
        Lcp on(C);
        FileStager fs;
        Program p;
        lcp_program(p, r, C);
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
        REQUIRE(EpCountsHostProbe::tables(fs, C));
        EpCountsHostProbe::set_capped(fs, true);
        for (int64_t n : { (int64_t)1, (int64_t)16, (int64_t)65, C }) {
            RadBatch b{};
            b.n_tok = (int32_t)n;
            b.n_seq = 1;
            b.phase = RAD_PHASE_PREFILL;
            CHECK(fs.begin_pass(&b));
            CHECK(EpCountsHostProbe::routed(fs));
        }
        EpCountsHostProbe::set_capped(fs, false);   /* not capped: the old limit */
        RadBatch b{};
        b.n_tok = (int32_t)C;
        b.n_seq = 1;
        b.phase = RAD_PHASE_PREFILL;
        CHECK(fs.begin_pass(&b));
        CHECK(!EpCountsHostProbe::routed(fs));
    }
}

/* The CPU path's jobs for an LCP chunk, as for a tall pass: the sorted order and the offsets where
 * the tables put them at C rows, every job's rows inside the expert's run, its rows and outputs where
 * they belong. */
TEST(an_lcp_chunk_cuts_its_jobs_from_the_chunk_sized_tables) {
    Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
    for (int64_t C : { 256, 1024 })
        for (int64_t T : { (int64_t)17, (int64_t)100, C }) {
            Lcp on(C);
            FileStager fs;
            {
                Rows r;
                Program p;
                lcp_program(p, r, C);
                CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
            }
            REQUIRE(EpCountsHostProbe::tables(fs, T));
            std::mt19937 rng((uint32_t)(T * 7 + C));
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
            for (int i = n; i < T * 6; ++i) sorted[i] = -1;
            CHECK_EQ(off - sorted, (ptrdiff_t)(C * 6));
            const uint16_t* x = EpCountsHostProbe::x(fs);
            uint16_t* y = EpCountsHostProbe::out(fs);
            int rows_seen = 0;
            for (int e = 0; e < 96; ++e) {
                if (!cnt[e]) continue;
                EpCountsHostProbe::jobs(fs).clear();
                EpCountsHostProbe::rows(fs).clear();
                CHECK_EQ(EpCountsHostProbe::jobs_for(fs, e), cnt[e]);
                int in_jobs = 0;
                for (const HostExpertPool::Job& j : EpCountsHostProbe::jobs(fs)) {
                    CHECK(j.more >= 0 && j.more < HostExpertPool::max_rows());
                    for (int m = 0; m <= j.more; ++m) {
                        const uint16_t* jx = m ? j.xs[m - 1] : j.x;
                        uint16_t* jy = m ? j.ys[m - 1] : j.y;
                        const int i = EpCountsHostProbe::rows(fs)[(size_t)(in_jobs + m)];
                        CHECK(i >= off[e] && i < off[e + 1]);
                        CHECK(jx == x + (int64_t)(sorted[i] / 6) * HostExpertPool::kEmbd);
                        CHECK(jy == y + (int64_t)i * HostExpertPool::kEmbd);
                    }
                    in_jobs += j.more + 1;
                }
                CHECK_EQ(in_jobs, cnt[e]);
                rows_seen += in_jobs;
            }
            CHECK_EQ(rows_seen, n);
            CHECK_EQ(cnt[5], (int32_t)T);
        }
}

/* hx_jobs_for's even cuts: every expert's rows in HostExpertPool::jobs_for(n) jobs of job_rows(n, k)
 * rows, the larger first (a 1,024-row run at four rows a job is 256 jobs of four; six rows two of three). */
TEST(the_cpu_path_cuts_each_experts_rows_evenly) {
    Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
    Lcp on(1024);
    FileStager fs;
    {
        Rows r;
        Program p;
        lcp_program(p, r, 1024);
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
    }
    const int64_t T = 1024;
    REQUIRE(EpCountsHostProbe::tables(fs, T));
    /* Expert 5 every token, expert 6 the first six, expert 7 the first 37, expert 8 one. */
    std::vector<std::vector<int32_t>> by(96);
    for (int64_t t = 0; t < T; ++t) {
        by[5].push_back((int32_t)(t * 6));
        if (t < 6) by[6].push_back((int32_t)(t * 6 + 1));
        if (t < 37) by[7].push_back((int32_t)(t * 6 + 2));
        if (t == 900) by[8].push_back((int32_t)(t * 6 + 3));
    }
    int32_t* sorted = EpCountsHostProbe::sorted(fs);
    int32_t* off = EpCountsHostProbe::off(fs);
    int32_t* cnt = EpCountsHostProbe::counts(fs);
    int n = 0;
    for (int e = 0; e < 96; ++e) {
        off[e] = n;
        cnt[e] = (int32_t)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) sorted[n++] = f;
    }
    off[96] = n;
    for (int i = n; i < T * 6; ++i) sorted[i] = -1;
    for (int e : { 5, 6, 7, 8 }) {
        EpCountsHostProbe::jobs(fs).clear();
        EpCountsHostProbe::rows(fs).clear();
        CHECK_EQ(EpCountsHostProbe::jobs_for(fs, e), cnt[e]);
        const auto& jobs = EpCountsHostProbe::jobs(fs);
        REQUIRE_EQ((int)jobs.size(), HostExpertPool::jobs_for(cnt[e]));
        for (size_t k = 0; k < jobs.size(); ++k) {
            CHECK_EQ(jobs[k].more + 1, HostExpertPool::job_rows(cnt[e], (int)k));
            CHECK_EQ(jobs[k].cost, HostExpertPool::job_cost(jobs[k].more + 1));
        }
    }
}

/* The per-row cut on the CPU path: routed layer li runs on the pass's rows from its model layer's
 * row0, so its jobs count and index those rows only (the sorted order is the layer's own, from 0);
 * without a row0 every layer is the pass. */
TEST(the_cpu_path_takes_a_layers_rows_under_the_row_cut) {
    Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
    Lcp on(1024);
    FileStager fs;
    {
        Rows r;
        Program p;
        lcp_program(p, r, 1024);
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
    }
    REQUIRE(EpCountsHostProbe::tables(fs, 1024));   /* routed layer 0 is model layer 7 */
    CHECK_EQ(EpCountsHostProbe::layer_r0(fs, 0), (int64_t)0);
    CHECK_EQ(EpCountsHostProbe::layer_rows(fs, 0), (int64_t)1024);
    std::vector<int32_t> row0(40, 0);
    row0[7] = 640;
    EpCountsHostProbe::set_row0(fs, row0);
    CHECK_EQ(EpCountsHostProbe::layer_r0(fs, 0), (int64_t)640);
    CHECK_EQ(EpCountsHostProbe::layer_rows(fs, 0), (int64_t)384);
    /* The layer's 384 rows on expert 5, its sorted slots 0 .. 384 * 6 (pass-relative); a slot past the
     * layer's rows is not a row of it. */
    int32_t* sorted = EpCountsHostProbe::sorted(fs);
    int32_t* off = EpCountsHostProbe::off(fs);
    int32_t* cnt = EpCountsHostProbe::counts(fs);
    for (int e = 0; e <= 96; ++e) off[e] = e <= 5 ? 0 : 384;
    for (int i = 0; i < 384; ++i) sorted[i] = i * 6;
    for (int i = 384; i < 1024 * 6; ++i) sorted[i] = -1;
    std::fill(cnt, cnt + 96, 0);
    cnt[5] = 384;
    EpCountsHostProbe::jobs(fs).clear();
    EpCountsHostProbe::rows(fs).clear();
    CHECK_EQ(EpCountsHostProbe::jobs_for(fs, 5), 384);
    const uint16_t* x = EpCountsHostProbe::x(fs);
    int k = 0;
    for (const HostExpertPool::Job& j : EpCountsHostProbe::jobs(fs))
        for (int m = 0; m <= j.more; ++m, ++k) {
            const uint16_t* jx = m ? j.xs[m - 1] : j.x;
            CHECK(jx == x + (int64_t)k * HostExpertPool::kEmbd);   /* the layer's row k, at hx_x_'s row k */
        }
    CHECK_EQ(k, 384);
}

/* Causal admission (RadBatch::lcp_admit_row): a unit is admitted when one of its rows is at or past
 * the pass's admission row (counted from the layer's first row under the row cut), and every unit
 * with no admission row; and the insert queue takes the admitted misses only, at most ..._INSERTQ. */
TEST(lcp_admits_what_the_prompts_last_tokens_select_and_caps_the_inserts) {
    Env xt("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", nullptr);
    Lcp on(1024);
    FileStager fs;
    {
        Rows r;
        Program p;
        lcp_program(p, r, 1024);
        CHECK_OK(EpCountsHostProbe::tall_init(fs, p));
    }
    REQUIRE(EpCountsHostProbe::tables(fs, 1024));
    /* Expert 5: tokens 0 .. 99; expert 6: tokens 950 .. 999; expert 7: token 600 alone. */
    std::vector<std::vector<int32_t>> by(96);
    for (int t = 0; t < 100; ++t) by[5].push_back(t * 6);
    for (int t = 950; t < 1000; ++t) by[6].push_back(t * 6 + 1);
    by[7].push_back(600 * 6 + 2);
    int32_t* sorted = EpCountsHostProbe::sorted(fs);
    int32_t* off = EpCountsHostProbe::off(fs);
    int n = 0;
    for (int e = 0; e < 96; ++e) {
        off[e] = n;
        for (int32_t f : by[(size_t)e]) sorted[n++] = f;
    }
    off[96] = n;
    for (int i = n; i < 1024 * 6; ++i) sorted[i] = -1;
    EpCountsHostProbe::set_admit(fs, 0);   /* no admission row: everything */
    for (int e : { 5, 6, 7 }) CHECK(EpCountsHostProbe::admits(fs, 0, e));
    EpCountsHostProbe::set_admit(fs, 896);   /* the prompt's last 128 tokens from row 896 */
    CHECK(!EpCountsHostProbe::admits(fs, 0, 5));
    CHECK(EpCountsHostProbe::admits(fs, 0, 6));
    CHECK(!EpCountsHostProbe::admits(fs, 0, 7));
    EpCountsHostProbe::set_admit(fs, 600);
    CHECK(EpCountsHostProbe::admits(fs, 0, 7));
    /* Under the row cut the layer's tokens count from its first row: with row0 600 the layer's token t
     * is the pass's row 600 + t, and the layer has 424 rows. */
    std::vector<int32_t> row0(40, 0);
    row0[7] = 600;
    EpCountsHostProbe::set_row0(fs, row0);
    EpCountsHostProbe::set_admit(fs, 700);
    CHECK(!EpCountsHostProbe::admits(fs, 0, 5));   /* layer tokens 0..99: pass rows 600..699 */
    CHECK(!EpCountsHostProbe::admits(fs, 0, 6));   /* tokens 950..: past the layer's rows, none of them */
    EpCountsHostProbe::set_admit(fs, 650);
    CHECK(EpCountsHostProbe::admits(fs, 0, 5));    /* layer tokens 50..99 are pass rows 650..699 */
    EpCountsHostProbe::set_row0(fs, {});
    /* The insert queue: three of six misses admitted, a cap of two. */
    EpCountsHostProbe::set_insertq(fs, 2);
    EpCountsHostProbe::misses(fs, { 1, 0, 1, 0, 1, 0 });
    EpCountsHostProbe::queue_inserts(fs);
    CHECK_EQ(EpCountsHostProbe::queued(fs), (size_t)2);
    const FileStager::HostStats& st = EpCountsHostProbe::stats(fs);
    CHECK_EQ(st.lcp_miss_admit, (uint64_t)2);
    CHECK_EQ(st.lcp_miss_dropped, (uint64_t)1);
    CHECK_EQ(st.lcp_miss_ro, (uint64_t)3);
    EpCountsHostProbe::drop_queue(fs);
    EpCountsHostProbe::misses(fs, {});
}

RAD_TEST_MAIN()

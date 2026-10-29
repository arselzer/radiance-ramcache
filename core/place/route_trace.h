/* route_trace.h -- RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=<prefix>: DeepSeek-V4.1's routing as every
 * rank saw it, written for the lossy track's offline gates (tools/ds41-host-experts/rtrace.py).
 *
 * ---- why --------------------------------------------------------------------------------------
 *
 * The lossy options (RESEARCH-3-LOSSY.md: drop a missed expert, drop it below a weight, route around
 * it to a held near-tie) each need numbers the engine's counters do not give: how much routing
 * weight a missed expert carries, where it sits among the six picks, and whether an expert the
 * rank held anyway scored almost as high. So per routed layer of every pass of at most 16 tokens
 * the trace records the six picks with their weights and biased keys (score + correction bias),
 * rank 0's 384 router logits on a one-token pass, and for every pick whether the rank that owns it
 * had it on the card, in the RAM cache, read it ahead, or had to read it from the drive -- plus
 * which of each rank's experts the RAM cache held at that moment (a substitute's residency) and,
 * once, which are on the card. The engine's step loop adds which request and position every pass
 * carried and which passes ran after their request had already stopped (the post-stop pass).
 *
 * Nothing it does changes a byte the card or the CPU computes: rank 0 copies the router's outputs
 * to pinned memory right behind the counts row the file stager already copies (three more copies a
 * layer on the compute stream, ~1.6 KB, under an event of their own, so the CPU's choice does not
 * wait for them), each rank reads the RAM cache's residency without a lease, an LRU touch or a
 * statistic (Mover::units_cached), and a background thread writes the files. The issuing threads
 * only append to a buffer.
 *
 * ---- files ------------------------------------------------------------------------------------
 *
 * <prefix>.rank<r>.bin, one a rank (its file stager), and <prefix>.steps.bin (the step loop). Each
 * starts with a 64-byte header:
 *
 *   char magic[8] "RADRTRC1"; u32 version (1); u32 stream (the rank, or 4: steps); u32 world;
 *   u32 n_expert (384); u32 top_k (6); u32 local (96: a rank's experts); u32 n_layers (40);
 *   u32 header bytes (64); u64 t0 (steady clock, ns: the trace's start, the same in every file);
 *   zeros to 64
 *
 * then records, little-endian, each `u16 type, u16 flags, u32 bytes` (the record's whole size,
 * header included, a multiple of 4) and its fields:
 *
 *   1 pass   (rank)  i32 step; u32 pass; i16 phase; i16 draft_pass; i32 n_tok; i32 n_seq;
 *                    i32 n_spec; i32 n_tok_decode; i32 max_ctx_len; u32 0; u64 t_ns
 *                    -- every pass the file stager armed for (RadBatch; `pass` counts them).
 *   2 layer  (rank)  u32 pass; i32 step; u16 layer; u16 rows; u16 n_sel; u8 flags; u8 0; u32 us;
 *                    u32 ram[3];  then n_sel x { u16 expert; u8 rows; u8 status };
 *                    flags & 2 (route): rows x { u16 id[top_k]; f32 weight[top_k]; f32 key[top_k] };
 *                    flags & 4 (logits): rows x f32 logit[n_expert]
 *                    -- one EP-routed layer of a pass. `us`: since the pass began, at the choice.
 *                    The selected experts this rank owns (global ids), the rows routed to each and
 *                    what the rank did with it (Status). flags & 1: the CPU path chose (else file-
 *                    tier picks are kUndecided or kCopyPath); flags & 8: `ram` is valid, bit e the
 *                    rank's local expert e held by the RAM cache at the choice (one-token passes).
 *                    Rank 0 only: the router's picks in its round order (the card's ids, weights
 *                    = route_scale x normalised score), and keys recomputed on the host in f32
 *                    from the logits (sqrt(softplus(x)), x > 20: sqrt(x)) and the bias, NaN
 *                    without logits; logits only on one-token passes (a 16-token chunk's would be
 *                    25 KB a layer).
 *   3 bias   (rank 0) u16 layer; u16 n; f32 bias[n] -- the layer's correction bias, once.
 *   4 tiers  (rank)   u16 n_layers; u16 local; then n_layers x { u16 layer; u16 0; u8 tier[local] }
 *                    -- once at init: each local routed expert's tier in the plan (0 VRAM,
 *                    1 host pinned, 2 host pageable, 3 the file tier, 4 mapped, 255 none).
 *   5 step   (steps) i32 step; i16 phase; i16 draft_pass; i32 n_tok; u16 n; u8 ahead; u8 0;
 *                    u64 t_ns; n x { u64 seq; i32 ctx_len; i32 n_tokens; u8 flags; u8 n_spec; u16 0 }
 *                    -- every step the loop runs, when it becomes the open step: its entries in
 *                    row order (seq: the request id; ctx_len: the first row's position; flags 1
 *                    prefill, 2 produces a token, 4 a device row). ahead: issued before the
 *                    previous step's readback (Scheduler::step_next).
 *   6 end    (steps) i32 step; i32 ahead_step; u64 seq; u64 n_out
 *                    -- request `seq` ended at `step`'s commit with n_out tokens; ahead_step, if
 *                    not -1, was already issued behind it and still carries its row: the
 *                    post-stop pass, whose output is discarded.
 *
 * A layer record joins its pass by (`pass`, `step`), a pass its step record by `step`, and the
 * ranks' layer records one another by (`step`, `layer`): every rank runs every pass. */
#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {

namespace rtrace {
inline constexpr char     kMagic[8] = { 'R', 'A', 'D', 'R', 'T', 'R', 'C', '1' };
inline constexpr uint32_t kVersion = 1, kHeaderBytes = 64;
inline constexpr int      kExperts = 384, kTopK = 6, kLocal = 96, kLayers = 40, kWorld = 4;
enum Type : uint16_t { kPass = 1, kLayer = 2, kBias = 3, kTiers = 4, kStep = 5, kEnd = 6 };
/* What the owning rank did with a pick. */
enum Status : uint8_t {
    kOnCard = 0,       /* a VRAM resident: the card computed it */
    kRamHit = 1,       /* the RAM cache held it: the CPU computed it from the cache */
    kMiss = 2,         /* read from the drive while the layer waited, then computed by the CPU */
    kAheadWhole = 3,   /* read ahead whole before its layer (the read-ahead's unit): no wait */
    kAheadPart = 4,    /* read ahead in part, the rest read while the layer waited */
    kCopyPath = 5,     /* the card's copy path (a unit the CPU path could not take) */
    kUndecided = 6,    /* file tier, no CPU choice on this pass (none ran) */
    kHostPinned = 7,   /* in the pinned host pool, read by the card across the link */
    kStatusCount = 8
};
enum LayerFlags : uint8_t { kDecided = 1, kRoute = 2, kLogits = 4, kRamMask = 8 };
enum StepFlags : uint8_t { kPrefill = 1, kProduces = 2, kDevRow = 4 };

/* Appends one record to `b`: the constructor writes its header, finish() pads it to 4 bytes and
 * patches its size. Little-endian (x86) by construction. */
class Rec {
public:
    Rec(std::vector<uint8_t>& b, uint16_t type, uint16_t flags) : b_(b), at_(b.size()) {
        put(type);
        put(flags);
        put(uint32_t{ 0 });
    }
    template <class T> void put(const T& v) { raw(&v, sizeof v); }
    void raw(const void* p, size_t n) {
        const size_t o = b_.size();
        b_.resize(o + n);
        if (n) std::memcpy(b_.data() + o, p, n);
    }
    void finish() {
        while ((b_.size() - at_) % 4) b_.push_back(0);
        const uint32_t n = (uint32_t)(b_.size() - at_);
        std::memcpy(b_.data() + at_ + 4, &n, 4);
    }

private:
    std::vector<uint8_t>& b_;
    size_t at_;
};

/* The card's score of a router logit, in f32 (r4d_ds41_elem.hip's kSqrtSoftplus; the card's exp
 * and log1p are its own, so a key may differ from the card's in the last bit). */
float sqrt_softplus(float x);

/* The records, each appended to `b` in the layout above (the file stager writes them; tests and
 * rtrace.py read them). */
struct PassRec {
    int32_t  step = 0;
    uint32_t pass = 0;
    int16_t  phase = 0, draft_pass = 0;
    int32_t  n_tok = 0, n_seq = 0, n_spec = 0, n_tok_decode = 0, max_ctx_len = 0;
    uint64_t t_ns = 0;
};
void encode_pass(std::vector<uint8_t>& b, const PassRec& p);
struct Pick {
    uint16_t expert = 0;   /* global id */
    uint8_t  rows = 0, status = 0;
};
struct LayerRec {
    uint32_t pass = 0;
    int32_t  step = 0;
    uint16_t layer = 0, rows = 0;
    uint8_t  flags = 0;              /* LayerFlags */
    uint32_t us = 0;
    uint32_t ram[3] = {};
    const Pick* picks = nullptr;
    int n_picks = 0;
    /* flags & kRoute: rows x kTopK each (the ids as the card wrote them); flags & kLogits:
     * rows x kExperts. */
    const int32_t* ids = nullptr;
    const float *weights = nullptr, *keys = nullptr, *logits = nullptr;
};
void encode_layer(std::vector<uint8_t>& b, const LayerRec& L);
void encode_bias(std::vector<uint8_t>& b, uint16_t layer, const float* bias);   /* kExperts */
void encode_tiers(std::vector<uint8_t>& b, const std::array<uint8_t, kLocal>* tier);   /* kLayers */
}  /* namespace rtrace */

/* One step entry for RouteTrace::step (sched's StepEntry, without the dependency). */
struct RtStepEntry {
    uint64_t seq = 0;
    int32_t  ctx_len = 0, n_tokens = 0;
    uint8_t  flags = 0, n_spec = 0;
};

class RouteTrace {
public:
    static constexpr int kSteps = rtrace::kWorld;   /* the step loop's stream; ranks are 0..3 */
    static constexpr int kStreams = kSteps + 1;

    /* The process's trace, or null when RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE is unset or empty.
     * Created on the first call; its files open on their first record. */
    static RouteTrace* get();

    /* A trace writing <prefix>.rank<r>.bin and <prefix>.steps.bin (get() makes the process's from
     * the switch; tests make their own). */
    explicit RouteTrace(std::string prefix);
    ~RouteTrace();
    RouteTrace(const RouteTrace&) = delete;
    RouteTrace& operator=(const RouteTrace&) = delete;

    const std::string& prefix() const { return prefix_; }
    /* One record (its header included) for stream `s`; a full buffer goes to the writer. Dropped
     * and counted after close(), when the file cannot be opened, or past the writer's backlog cap. */
    void put(int s, const void* rec, size_t n);
    /* The step loop's records (format above). */
    void step(int32_t step, int phase, int draft_pass, int64_t n_tok, bool ahead,
              const RtStepEntry* e, int n);
    void end(int32_t step, int32_t ahead_step, uint64_t seq, uint64_t n_out);
    /* Every buffer written, the writer joined and the files closed. Idempotent; called by the
     * engine at shutdown, before its report, and by the destructor. */
    void close();

    struct Stats {
        uint64_t records[kStreams] = {}, bytes[kStreams] = {};
        uint64_t dropped = 0, writes = 0, write_ns = 0, max_backlog = 0;
        bool     open_failed = false;
    };
    Stats stats() const;
    /* One line for the shutdown report. */
    std::string report() const;

    /* Writer backlog past which buffers are dropped (counted), so a stalled disk cannot take the
     * host's memory: 512 MiB is minutes of rank 0's ~1.1 MB/s at 17 tok/s. */
    static constexpr size_t kBacklogCap = size_t(512) << 20;
    static constexpr size_t kHandoff = size_t(256) << 10;   /* a stream buffer's size at handoff */

private:
    struct Stream {
        mutable std::mutex mu;   /* its producer's appends, and close() */
        std::vector<uint8_t> cur;
        FILE* fp = nullptr;
        bool  tried = false;
        uint64_t records = 0, bytes = 0;
    };
    struct Chunk {
        int s = 0;
        std::vector<uint8_t> b;
    };
    bool open_locked(int s, Stream& st);   /* the file and its header, once */
    void handoff(int s, std::vector<uint8_t>& b);
    void loop();

    std::string prefix_;
    uint64_t t0_ns_ = 0;
    Stream st_[kStreams];
    std::atomic<bool> closed_{ false }, open_failed_{ false };
    std::atomic<uint64_t> dropped_{ 0 };
    /* The writer's queue; locked after a stream's mutex where both are held. */
    mutable std::mutex qmu_;
    std::condition_variable qcv_;
    std::deque<Chunk> q_;
    std::vector<std::vector<uint8_t>> free_;   /* written buffers, kept for reuse */
    size_t backlog_ = 0;
    uint64_t max_backlog_ = 0, writes_ = 0, write_ns_ = 0;
    bool stop_ = false;
    std::thread writer_;
};

}  /* namespace rad */

/* live_chain.h -- the card's serial dispatch chain on the live issue path: hazard-checked overlap
 * (RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP) and the instrument that shows the chain
 * (RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE).
 *
 * ---- why --------------------------------------------------------------------------------------
 *
 * A recorded pass leaves a dispatch without the AQL barrier bit where it depends on nothing still
 * running (Ctx::haz_mark). A pass a stager arms is issued live and never recorded
 * (runtime/stager.h), so haz_mark never sees it and every dispatch keeps the bit: the packet
 * processor waits for each kernel to end before it launches the next, ~2.5-4 us of idle card per
 * dependent pair, plus the whole of any kernel that could have run beside its predecessor. DS41
 * decode is such a pass at every step (the FileStager arms them all), and its issue order was
 * written for the overlap (deepseek41.cpp: the query and key chains alternate, the compressor and
 * the indexer sit between them, the attention mix between the output projection's groups). The
 * loop's audit counted played = recorded = unordered = 0. Before each layer's routing-counts sync
 * about 19 dispatches run strictly in series, ~290-320 us a plain layer and ~15 ms a token; an
 * offline model of haz_mark's rules on the issue order frees 6 / 9 / 17 of them in plain / indexer
 * / source layers (RESEARCH-2-RANKED-PLAN.md #6).
 *
 * ---- the overlap (RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1: decode passes; =2: every pass
 * issued live) -----------------------------------------------------------------------------------
 *
 * Per launch what haz_mark does per tape: a device op's first dispatch goes without the barrier
 * bit when it reads nothing the dispatches still in flight on its queue write, and writes nothing
 * they read or write. Exact: only the order in which independent kernels start changes, never what
 * a kernel computes.
 *
 * In flight means the packets on the queue from the last one with the barrier bit on. The issue
 * path knows them only if it wrote them all, so the backend tags every packet with the launch
 * that wrote it (DeviceBackend::live_pos). Anything else on the queue since the last tracked
 * launch -- an event record or wait, a weight-table patch, a stager copy, a CPU-row blit, a link
 * hold, another thread's packet -- is an ordered point: the next launch keeps its bit. Every rule
 * errs that way:
 *   - opaque ops (never unordered, and the op after one is ordered): a collective, a kernel without
 *     rad_kernel_concurrent (libr4d's hc_ mixes with stream-keyed counters, the tiled int GEMMs'
 *     split tails, LoRA), no schema, more operands than the schema, an unknown operand role, an
 *     operand without a dtype or with a negative stride;
 *   - host-visible ops (never unordered, and the next op is ordered): a host-arena operand, or a
 *     write through a raw pointer the core cannot place;
 *   - weights are read-only within a pass (what the stager and the mover write reaches the queue
 *     as an event), but a buffer passed in a weight's slot still counts as a read;
 *   - an op's later dispatches keep the bit (they depend on its first), and the backend refuses a
 *     permit it cannot honour: a ride or record on the packet, a system-scope fence, chaining, a
 *     tape capture, or a queue that already has kTapeOverlapRun unordered packets at its tail;
 *   - a launch never runs beside another launch of its own op, whatever their operands: a row may
 *     keep per-instance state (an arrival counter taken at init, one per op and band).
 * Scratch is a range like any other, so two split-K GEMVs on one lane overlap only while
 * scratch_take hands them disjoint slices; arena buffers that share bytes through the liveness
 * packing clash by address, whatever their names.
 *
 * ---- the instrument (RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE=N) --------------------------------
 *
 * Every Nth trunk decode pass on rank 0 is traced: one line per device op into <prefix>.ops.csv --
 * its layer, op, kernel, queue and packets, what the check found (`would`: nothing in flight to
 * wait for; `why` it kept the bit otherwise; the op it clashed with), the earlier ops of the pass
 * it depends on (r: reads what it wrote, a: writes what it read, w: writes what it wrote; from
 * the issued operands and their schema roles, the byte ranges the declared rad_op_reads /
 * rad_op_writes sets resolve to) and the buffers it reads and writes -- and one line per kernel
 * with its start and end on the card into <prefix>.stamps.csv (DeviceBackend::live_stamp, keyed by
 * queue and packet index; nothing waits for them). <prefix> is
 * RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT, default /tmp/radiance-chain-<pid>.
 * tools/ds41-host-experts/chain_report.py joins the two files: per layer the dispatches, their gaps
 * and the critical chain to the counts sync, and the fusion candidates. With the overlap off the
 * trace is the dump-only arm: `would` shows which dispatches the check frees, and no packet
 * changes (the host pays the bookkeeping and the stamps, so it is not a timing arm).
 *
 * Either switch prints a summary line per rank at shutdown (Engine's report, Ctx::live_report).
 */
#pragma once
#include "rad_abi.h"
#include "rad_runtime.h"
#include "device/device.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rad {

class Program;
struct KernelRow;
struct OpInfo;

/* One range an operand, plus the scratch; more operands than this make the op opaque. */
constexpr int kLiveRanges = 36;

struct LiveRange {
    uintptr_t lo = 0, hi = 0;   /* [lo, hi) */
};

/* One launch as the hazard check sees it. */
struct LiveOp {
    uint64_t  tag = 0;            /* the backend's packet tag; never 0 */
    /* Its op handle, 0 for none. A row's per-instance state (an arrival counter taken at init,
     * one per declared op and band) is shared by every launch of the op, so a launch never joins
     * one of its own op still in flight, whatever their operands. */
    uint32_t  op = 0;
    bool      opaque = false;     /* its operands do not bound what it touches */
    bool      host = false;       /* it touches host-visible memory */
    int       n_rd = 0, n_wr = 0;
    LiveRange rd[kLiveRanges];
    LiveRange wr[kLiveRanges];

    void reset(uint64_t t, uint32_t h = 0) {
        tag = t;
        op = h;
        opaque = host = false;
        n_rd = n_wr = 0;
    }
    void read(uintptr_t lo, uintptr_t hi) {
        if (n_rd < kLiveRanges) rd[n_rd++] = LiveRange{ lo, hi };
        else opaque = true;
    }
    void write(uintptr_t lo, uintptr_t hi) {
        if (n_wr < kLiveRanges) wr[n_wr++] = LiveRange{ lo, hi };
        else opaque = true;
    }
};

/* How launch `b`, issued after `a`, depends on it: kLiveRaw when it reads bytes `a` writes,
 * kLiveWar when it writes bytes `a` reads, kLiveWaw when it writes bytes `a` writes; kLiveUnknown
 * when either is opaque. 0: independent, `b` may run beside `a`. */
enum : int { kLiveRaw = 1, kLiveWar = 2, kLiveWaw = 4, kLiveUnknown = 8 };
int live_dep(const LiveOp& b, const LiveOp& a);

/* Why a launch's first dispatch keeps the barrier bit (or kLiveFree: it need not). */
enum LiveWhy : uint8_t {
    kLiveFree = 0,   /* nothing in flight it depends on */
    kLiveForeign,    /* the queue holds packets no tracked launch wrote since the last one */
    kLiveOpaque,     /* the op itself is opaque */
    kLiveHost,       /* the op touches host-visible memory */
    kLiveAfter,      /* an opaque or host-visible op, or one no longer known, is in flight */
    kLiveCap,        /* kTapeOverlapRun unordered launches are in flight already */
    kLiveClash,      /* it depends on an op in flight */
    kLiveWhys
};
const char* live_why_name(int why);

/* The queues' in-flight sets: pure bookkeeping over positions the backend reports, so the tests
 * drive it with made-up ones. A launch takes a slot (claim), fills its LiveOp, asks check() at the
 * queue's position before the launch, and reports what the launch wrote (commit). */
class LiveQueues {
public:
    struct Pos {
        uint32_t queue = 0;
        uint64_t wr = 0;          /* the index the queue's next packet goes to */
        uint64_t last_tag = 0;    /* the tag on the packet before it; 0: untracked */
    };
    struct Win {
        uint64_t first = 0;
        int32_t  packets = 0;     /* written on the queue the launch was checked on */
        int      flags = 0;       /* LIVE_FIRST_UNORDERED, LIVE_BARRIER */
    };

    static constexpr int kRing = 32;

    LiveQueues() : ring_(kRing) {}
    int     claim();                            /* a slot for the next launch */
    LiveOp& op(int slot) { return ring_[(size_t)slot]; }
    const LiveOp& op(int slot) const { return ring_[(size_t)slot]; }
    /* Whether slot `s`'s launch at `at` may leave its first dispatch unordered. `clash`: the tag of
     * the in-flight launch it depends on, for kLiveClash. `foreign`: packets on the queue since the
     * last tracked launch, -1 when the queue is not known. */
    LiveWhy check(int s, const Pos& at, uint64_t* clash, int64_t* foreign) const;
    /* What slot `s`'s launch wrote: the queue's position before and after it, and the window. */
    void    commit(int s, const Pos& before, const Win& w, const Pos& after);
    /* Launches in flight on `queue` that the check knows, for tests. */
    int     in_flight(uint32_t queue) const;
    void    reset() { qs_.clear(); }

private:
    struct Inflight { int32_t slot; uint64_t tag; };
    struct Q {
        uint32_t id = 0;
        bool     known = false;     /* end and last_tag describe the queue */
        uint64_t end = 0, last_tag = 0;
        bool     blind = true;      /* something unbounded may still run */
        int      run = 0;           /* unordered launches since the last ordered packet */
        std::vector<Inflight> infl; /* the launches since (and with) the last ordered packet */
    };
    const Q* find(uint32_t id) const;
    Q&       get(uint32_t id);
    std::vector<Q>      qs_;
    std::vector<LiveOp> ring_;
    int                 head_ = 0;
};

/* Counters for the shutdown line. One rank's thread writes them; read after it stops. */
struct LiveStats {
    uint64_t passes = 0, decode_passes = 0, overlap_passes = 0, traced = 0;
    uint64_t launches = 0;        /* device launches checked */
    uint64_t free = 0;            /* of them, nothing in flight to wait for */
    uint64_t unordered = 0;       /* first dispatches the backend wrote without the bit */
    uint64_t refused = 0;         /* permitted and kept by the backend */
    uint64_t why[kLiveWhys] = {};
    uint64_t multi = 0;           /* launches of more than one packet on their queue */
    uint64_t silent = 0;          /* launches that wrote nothing on their queue */
    uint64_t decode_launches = 0, decode_free = 0, decode_unordered = 0;
};

/* One operand of a traced launch, for naming. */
struct LiveOpd {
    uint8_t  kind = 0, role = 0;
    uint32_t handle = 0;
    int64_t  offset = 0;
};

/* One traced launch. */
struct LiveTraceRow {
    uint32_t         seq = 0;
    uint32_t         op = 0;
    const OpInfo*    info = nullptr;
    const KernelRow* row = nullptr;
    uint8_t          lane = 0;
    uint32_t         queue = 0;
    uint64_t         first = 0;
    int32_t          packets = 0;
    bool             unordered = false;
    bool             would = false;
    uint8_t          why = 0;
    int64_t          clash = -1;     /* the seq it clashed with */
    int64_t          foreign = -1;
    bool             scratch = false;
    int              n_opd = 0;
    LiveOpd          opd[40];
    LiveOp           ranges;
};

/* One launch between Ctx::live_before and live_after. */
struct LiveCall {
    bool            on = false;
    int             slot = 0;
    RadStream       s = nullptr;
    LiveQueues::Pos before;
    LiveWhy         why = kLiveFree;
    bool            permit = false;
    uint64_t        clash = 0;
    int64_t         foreign = -1;
    LiveTraceRow*   row = nullptr;   /* in a traced pass */
};

/* The run phase's side: configuration, the queues, the trace and the counters. Owned by a Ctx. */
class LiveChain {
public:
    LiveChain() = default;
    ~LiveChain();
    LiveChain(const LiveChain&) = delete;
    LiveChain& operator=(const LiveChain&) = delete;
    /* Reads the switches. Inert (on() false) when both are off or the backend has no live path. */
    void init(int rank, const Program* prog);
    bool on() const { return mode_ > 0 || every_ > 0; }

    /* Around one pass issued live. */
    void pass_begin(const RadBatch* b);
    void pass_end();
    /* Whether this pass's launches go through the check at all, and may be unordered. */
    bool checking() const { return in_pass_; }
    bool overlap() const { return overlap_pass_; }
    bool tracing() const { return traced_; }

    uint64_t next_tag() { return tag_base_ | ++tag_n_; }
    uint64_t pass_first_tag() const { return pass_tag0_; }

    LiveQueues& queues() { return q_; }
    LiveStats&  stats() { return st_; }
    const LiveStats& stats() const { return st_; }
    LiveTraceRow& trace_row();         /* the next row of the traced pass */

    /* The shutdown line; empty when both switches are off. */
    std::string report() const;

private:
    void write_trace();

    int            mode_ = 0;          /* CHAIN_OVERLAP: 0 off, 1 decode, 2 every live pass */
    int            every_ = 0;         /* CHAIN_TRACE: every Nth trunk decode pass on rank 0 */
    std::string    prefix_;
    int            rank_ = 0;
    const Program* prog_ = nullptr;
    bool           in_pass_ = false, overlap_pass_ = false, traced_ = false;
    uint64_t       decode_seen_ = 0, trace_id_ = 0;
    uint64_t       tag_base_ = 0, tag_n_ = 0, pass_tag0_ = 0;
    LiveQueues     q_;
    LiveStats      st_;
    std::vector<LiveTraceRow> rows_;
    size_t         n_rows_ = 0;
    FILE*          ops_out_ = nullptr;
    bool           ops_failed_ = false;
};

}  /* namespace rad */

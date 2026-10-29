/* host_expert_pool.h -- DeepSeek-V4.1 routed experts computed on the CPU from the RAM cache.
 *
 * The file stager hands this pool the RAM-resident experts it would otherwise copy to the card:
 * one job per (expert, token row), gate_up, swiglu_clamped and down over the cache's llama.cpp
 * block_mxfp4 bytes, each output bit-identical to libr4d's decode kernels (tools/ds41-host-experts,
 * proven against the measured plugin on the card). The stager copies the rows into the layer's
 * down-projection output before its gather, so the gather and everything after it are unchanged.
 *
 * One pool per process serves every rank. A job's output rows are cut into tiles of 64; workers
 * take tiles from the oldest job with work left, so the jobs of four ranks share the cores and
 * the first-submitted expert finishes first. The worker that ends a job's last gate_up tile runs
 * its swiglu and opens its down tiles.
 *
 * Off unless RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=<threads>. Also needs
 * RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=<file>: silu(g) for each of the 65536 bf16 gates as
 * the card computes it (f32, 256 KiB). The host's expf differs from the card's in the last bit, so
 * the table is captured on the card (ds41_mxfp4_device_check swiglu) rather than computed here.
 * Optional: ..._CPUS=<list> pins the workers ("4-23" or "4,5,6"), ..._GBPS and ..._LINK_GBPS set
 * the scheduling estimates (4.2 GB/s a thread up to 78, and 20), ..._SPIN_US how long an idle
 * worker spins before it sleeps (200), ..._NICE the workers' nice value (0), ..._MAX_ROWS the most
 * token rows of one expert a job carries (4; up to 64, see max_rows). */
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rad {

class HostExpertPool {
public:
    static constexpr int     kEmbd = 5120, kFf = 2304;
    static constexpr int64_t kGuBytes = 12533760, kDnBytes = 6266880;
    static constexpr int64_t kExpertBytes = kGuBytes + kDnBytes;

    /* One expert on one token row. gu and dn are block_mxfp4 rows (the RAM cache's form), or with
     * gu_scales and dn_scales set the release's planes (a fresh read from the container): codes
     * [N, K/2] and e8m0 scales [N, K/32], or with `packed` set packed slots of gu_bytes and
     * dn_bytes (the packed RAM cache: ds41_host_experts.h packed_view, checked at submit). An
     * urgent job (a missed expert, on the critical path) is claimed before every other. */
    /* Two of ds41hx's many-row calls (kGemvRows each): a job of 33 to 64 rows reads each 64-row
     * weight tile from DRAM once and again from L2 for its second call (a tile of gate_up is ~170 KB),
     * where two jobs of up to 32 rows each read it from DRAM. For large-chunk prefill
     * (RADIANCE_EXPERIMENTAL_DS41_LCP), whose chunks put hundreds of rows on a hot expert. */
    static constexpr int kMaxRowsCap = 64;
    static constexpr int kMaxRowsDefault = 4;  /* ds41hx::kFewRows */
    /* The most token rows a job carries: RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS, 1 ..
     * kMaxRowsCap (default 4), read once; an unusable value turns the pool off (get() says so) and
     * reads as the default here. With more than four allowed, a job of four or more rows runs the
     * many-row kernel (ds41hx::gemv_avx2_many, in calls of up to 32 rows), for prefill passes whose
     * rows meet in one expert; every other job runs ds41hx::gemv_avx2_few, as with the default. */
    static int max_rows();
    /* The fewest rows of a job that runs the many-row kernel: 4 when max_rows() > 4 (rbench, 40
     * threads: 10-18% under gemv_avx2_few at four rows, about even at three), else none. */
    static int many_from() { return max_rows() > kMaxRowsDefault ? kMaxRowsDefault : kMaxRowsCap + 1; }
    struct Job {
        const uint8_t*  gu = nullptr;   /* [2 * kFf, kEmbd] */
        const uint8_t*  dn = nullptr;   /* [kEmbd, kFf] */
        const uint8_t*  gu_scales = nullptr;
        const uint8_t*  dn_scales = nullptr;
        bool            packed = false;
        int64_t         gu_bytes = 0, dn_bytes = 0;   /* packed: the slots' sizes */
        const uint16_t* x = nullptr;    /* kEmbd bf16 */
        uint16_t*       y = nullptr;    /* kEmbd bf16 */
        /* Up to max_rows() - 1 more token rows of the same expert: each weight tile is read once for
         * all of them and each round of its codes decoded once (gemv_avx2_few, or gemv_avx2_many from
         * many_from() rows), each row's arithmetic unchanged. */
        const uint16_t* xs[kMaxRowsCap - 1] = {};
        uint16_t*       ys[kMaxRowsCap - 1] = {};
        int             more = 0;
        bool            urgent = false;
        int64_t         cost = 0;       /* backlog bytes it stands for; 0: cost_rows(1 + more) */
    };
    /* The backlog a job of `rows` rows stands for: the expert's bytes for the first row and for
     * each more half of them in a job of up to four rows (computed from L2), an eighth in a larger
     * one (the many-row kernel: ~0.05 ms a further row against ~0.38 ms for an expert's first, 40
     * threads; ds41_host_experts_test rbench). */
    static int64_t job_cost(int rows) {
        return kExpertBytes + (int64_t)(rows - 1) * (rows <= kMaxRowsDefault ? kExpertBytes / 2 : kExpertBytes / 8);
    }
    /* How an expert's `rows` rows are cut into jobs: ceil(rows / max_rows()) of them, as even as the
     * count allows (40 rows at 32 a job: 20 and 20, not 32 and 8), the larger ones first. Job k of
     * jobs_for(rows) holds job_rows(rows, k) rows. */
    static int jobs_for(int rows) {
        const int m = max_rows();
        return rows <= 0 ? 0 : (rows + m - 1) / m;
    }
    static int job_rows(int rows, int k) {
        const int n = jobs_for(rows);
        return n ? rows / n + (k < rows % n ? 1 : 0) : 0;
    }
    /* An expert's `rows` rows as jobs_for(rows) jobs. */
    static int64_t cost_rows(int rows) {
        int64_t c = 0;
        const int n = jobs_for(rows);
        for (int k = 0; k < n; ++k) c += job_cost(job_rows(rows, k));
        return c;
    }
    /* The jobs one caller waits for: submit adds them, the pool counts them down. */
    struct Ticket {
        std::atomic<int> left{ 0 };
    };

    /* The process's pool, started on the first call from the environment. Null when off, or when
     * the configuration is unusable (said once). */
    static HostExpertPool* get();

    /* Queues `n` jobs under `t`, each with its cost the caller reserved. Blocks while every job
     * slot is taken. */
    int  submit(const Job* jobs, int n, Ticket* t);
    /* Until every job of `t` has finished: spins briefly, then yields. */
    void wait(const Ticket* t) const;
    bool done(const Ticket* t) const { return t->left.load(std::memory_order_acquire) == 0; }

    /* Weight bytes queued and not yet read, over all callers, counting what callers have reserved
     * for jobs they are about to submit. A caller reserves a job's cost when it decides, so
     * ranks deciding at once see each other's choices; submit takes the reservation over (each
     * finished job gives back its cost) and unreserve returns what is never submitted. */
    int64_t backlog() const { return backlog_.load(std::memory_order_relaxed); }
    void    reserve(int64_t bytes) { backlog_.fetch_add(bytes, std::memory_order_relaxed); }
    void    unreserve(int64_t bytes) { drained(backlog_.fetch_sub(bytes, std::memory_order_relaxed) - bytes); }
    /* Sleeps until the pool has nothing queued or reserved, or `max_ns` has passed: for work that
     * should not share DRAM with the pool's streaming (the RAM cache's background inserts). */
    void    wait_idle(int64_t max_ns);
    double  cpu_bytes_per_ns() const { return cpu_bpns_; }
    double  link_bytes_per_ns() const { return link_bpns_; }
    float   limit() const { return limit_; }
    int     threads() const { return (int)workers_.size(); }
    /* Jobs queued at once over every caller; submit blocks for a free one past that. */
    static int slots() { return kSlots; }
    /* submit's waits for a free slot (a burst past kSlots): how many, and their time. Waited
     * outside the submit lock, so another caller's urgent jobs are not held behind one. */
    uint64_t submit_waits() const { return submit_waits_.load(std::memory_order_relaxed); }
    uint64_t submit_wait_ns() const { return submit_wait_ns_.load(std::memory_order_relaxed); }

    ~HostExpertPool();

private:
    /* 256 rather than 128: a large-chunk prefill layer queues ~100 jobs a rank, four ranks at once. */
    static constexpr int kSlots = 256;
    static constexpr int kTile = 64;                       /* output rows a tile */
    static constexpr int kGuTiles = 2 * kFf / kTile;       /* 72 */
    static constexpr int kDnTiles = kEmbd / kTile;         /* 80 */

    /* claim: the job's sequence number (low 32 bits) over the next tile to hand out, gate_up
     * tiles first. kDone, or any tile index past the last, hands out nothing. */
    static constexpr uint64_t kDone = (uint64_t)(kGuTiles + kDnTiles);
    /* Cache-line aligned, as when the row buffers lived inside: a slot's claim word, which workers
     * swap for every tile, shares no line with another slot's fields. */
    struct alignas(64) Slot {
        std::atomic<int>      state{ 0 };   /* 0 free, 1 taken (guarded by submit_mu_ for takers) */
        std::atomic<uint64_t> claim{ kDone };
        std::atomic<uint64_t> seq{ 0 };
        std::atomic<int>      gu_left{ 0 }, dn_left{ 0 };
        std::atomic<uint32_t> open{ 0 };    /* the sequence whose down tiles may start */
        std::atomic<int>      urgent{ 0 };
        Job      job;
        Ticket*  ticket = nullptr;
        uint16_t* gate_up = nullptr;   /* max_rows() rows of 2 * kFf, in rows_ */
        uint16_t* act = nullptr;       /* max_rows() rows of kFf */
    };

    HostExpertPool() = default;
    int  start(int threads, const std::vector<int>& cpus);
    void worker(int i, int cpu);
    void wake();                     /* new jobs published: wake sleeping workers */
    bool run_one();                  /* one tile of the oldest job with work; false when none */
    void tile(Slot& s, uint32_t gen, int k);
    void drained(int64_t left);      /* backlog_ now `left`: wake wait_idle at 0 */

    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<uint16_t, void (*)(void*)> rows_{ nullptr, nullptr };   /* the slots' row buffers */
    std::vector<float>      silu_;
    float  limit_ = 10.0f;
    double cpu_bpns_ = 67.0, link_bpns_ = 20.0;

    std::vector<std::thread> workers_;
    std::atomic<bool>     stop_{ false };
    std::atomic<int64_t>  backlog_{ 0 };
    std::atomic<uint64_t> work_gen_{ 0 };
    std::mutex              submit_mu_;
    uint64_t                seq_ = 0;
    std::atomic<uint64_t>   submit_waits_{ 0 }, submit_wait_ns_{ 0 };
    std::mutex              idle_mu_;
    std::condition_variable idle_cv_;
    std::atomic<int>        sleepers_{ 0 };
    std::mutex              slot_mu_;    /* submit waits on slot_cv_ while every slot is taken */
    std::condition_variable slot_cv_;
    std::mutex              quiet_mu_;   /* wait_idle */
    std::condition_variable quiet_cv_;
    std::atomic<int>        quiet_waiters_{ 0 };
};

}  /* namespace rad */

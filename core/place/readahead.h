/* readahead.h -- DeepSeek-V4.1 decode: routed units a predictor names two layers ahead, read from the
 * container in small pieces that give way to demand reads, and handed to their layer only whole
 * (RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces; FileStager::hx_ra_* and router_predictor.h).
 *
 * ---- why pieces ---------------------------------------------------------------------------------
 *
 * A routed unit the RAM cache misses is read once its layer's routing is known: ~18.8 MB, ~2.4 ms,
 * while the layer's gather waits (FileStager::hx_submit_misses_now, "the demand read"). Read ahead
 * whole (eee14f49), a predicted unit held the drive as long as a demand read does, and in 53.7% of
 * the miss layers a demand read waited behind one: the read-ahead paid nothing. Here a predicted
 * unit is read in pieces (512 KiB), at most 4 in flight over the process, and no piece starts while
 * any rank's demand read is in flight -- the readers sleep on the count, nothing polls -- so a demand
 * read meets at most the pieces already in flight. On the 990 Pro 4 x 512 KiB in flight read at
 * 6.77 GB/s, and a stream that yields so added ~0.075 ms to an 18.8 MB demand read (piecebench).
 *
 * ---- a unit's life --------------------------------------------------------------------------------
 *
 *   planned    laid out in its HostPlanes as read_unit_planes would lay it out and queued; readers
 *              take its pieces in order, and take units by their layer, then their prediction rank
 *   its layer  (routing known) not selected, or a RAM-cache hit: cancelled -- no further piece
 *              starts, the ones in flight finish, then its buffer goes back; never inserted
 *              selected and a miss:
 *                whole        handed to the layer as read
 *                not started  the layer reads it as any miss (take_whole)
 *                partly read  urgent: the readers read the rest at once, demand reads or not, and
 *                             the layer waits for its last piece (wait)
 *
 * A unit is handed over only when every piece of each of its weights is in (per-weight counters)
 * and none failed; the stager also checks the container's epoch then. One with a failed piece is
 * read again whole by its layer. The bytes are the container's either way: the outputs are the same
 * as without read-ahead.
 *
 * The demand count (demand_enter / demand_leave) is the process's: every rank's missed-unit reads
 * share the drive. A reader starts a non-urgent piece only at a count of zero, and the count is
 * checked again after the piece is announced, so a demand read either holds the piece back or sees
 * it in flight (and counts the wait). */
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rad {

/* One weight's container extent as read_unit_planes reads it: bytes [off, off + len) of container
 * `file`, landing at dst, which has cap bytes (the O_DIRECT tail rounding included). */
struct PlaneExtent {
    uint32_t file = 0;
    int64_t  off = 0, len = 0;
    uint8_t* dst = nullptr;
    int64_t  cap = 0;
};

class PieceReader;

/* A unit read ahead: up to two extents, read in that order in pieces of the reader's size. */
class ReadAheadUnit {
public:
    /* One piece's read: RAD_OK or an error. */
    using ReadFn = int (*)(void* ctx, const PlaneExtent& piece);
    /* `order`: readers take the smallest first. `keep` owns the destinations; the readers hold it
     * while a piece is in flight, so a unit dropped meanwhile keeps its buffer until then. */
    ReadAheadUnit(const void* owner, uint64_t order, const PlaneExtent* ext, int n_ext,
                  int64_t piece, ReadFn read, void* ctx, std::shared_ptr<void> keep);

    const void* owner() const { return owner_; }
    uint64_t    order() const { return order_; }
    int         pieces() const { return n_all_; }
    int         pieces(int w) const { return w == 0 || w == 1 ? n_[w] : 0; }
    /* Piece k: its extent's file, offset, length, destination and capacity from there. */
    PlaneExtent piece(int k) const;

    /* Pieces the readers have started (for a unit cancelled or taken whole: by then), and pieces
     * in. */
    int  started() const;
    int  in() const { return done_.load(std::memory_order_acquire); }
    int  in(int w) const { return w == 0 || w == 1 ? done_w_[w].load(std::memory_order_acquire) : 0; }
    int64_t bytes_in() const;
    /* Every piece of every extent in, none failed. */
    bool whole() const;
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    bool urgent() const { return urgent_.load(std::memory_order_acquire); }
    bool cancelled() const { return cancelled_.load(std::memory_order_acquire); }

private:
    friend class PieceReader;
    const void* owner_;
    uint64_t    order_;
    PlaneExtent ext_[2];
    int         n_ext_ = 0, n_[2] = { 0, 0 }, n_all_ = 0;
    int64_t     piece_;
    ReadFn      read_;
    void*       ctx_;
    std::shared_ptr<void> keep_;
    std::atomic<int>  next_{ 0 };        /* the next piece a reader starts; >= n_all_: none */
    std::atomic<int>  stopped_at_{ -1 }; /* cancelled or taken whole: the pieces started by then */
    std::atomic<int>  done_{ 0 };        /* pieces in, failed ones too */
    std::atomic<int>  done_w_[2] = {};   /* ...by extent */
    std::atomic<int64_t> bytes_in_{ 0 }; /* bytes of the pieces read */
    std::atomic<bool> failed_{ false }, urgent_{ false }, cancelled_{ false };
};

/* The readers and the demand count. PieceReader::process() is the engine's (every rank's units);
 * a test makes its own. */
class PieceReader {
public:
    PieceReader() = default;
    ~PieceReader();   /* stops and joins the readers; units still queued are dropped */
    PieceReader(const PieceReader&) = delete;
    PieceReader& operator=(const PieceReader&) = delete;

    /* The process's reader, never destroyed; its readers start with the first start(). */
    static PieceReader& process();

    /* Starts `readers` threads (1..8) reading pieces of `piece` bytes (a multiple of 4 KiB, 64 KiB
     * to 16 MiB). Once: later calls change nothing and return whether the first succeeded. */
    bool    start(int readers, int64_t piece);
    bool    started() const { return started_.load(std::memory_order_acquire); }
    int     readers() const { return n_threads_; }
    int64_t piece_bytes() const { return piece_; }

    /* ---- demand reads (every rank's missed-unit reads). Two atomic operations while no reader
     * runs; with readers, a lock when a demand read starts with pieces in flight, and when the
     * count returns to zero (to wake them). */
    void demand_enter();
    void demand_leave();
    int  demand_count() const { return demand_.load(std::memory_order_acquire); }

    /* ---- units */
    void submit(std::shared_ptr<ReadAheadUnit> u);
    /* No further piece starts; pieces in flight finish. */
    void cancel(ReadAheadUnit& u);
    /* The layer takes a unit no reader has started (true): it is the layer's to read, and no
     * reader ever will. False: a piece has started (see make_urgent). */
    bool take_whole(ReadAheadUnit& u);
    /* The unit's layer needs it now: its remaining pieces go before any other unit's, and start
     * whether demand reads are in flight or not. */
    void make_urgent(ReadAheadUnit& u);
    /* Blocks until the unit's pieces are all in, or, for a unit cancelled or taken whole, the
     * pieces started by then; a unit still queued is made urgent first. Returns the nanoseconds
     * waited, at once when the readers have stopped. */
    uint64_t wait(ReadAheadUnit& u);
    /* Cancels every unit of `owner` and waits for their pieces in flight: afterwards no reader
     * touches them or their buffers. */
    void forget(const void* owner);

    struct Stats {
        uint64_t pieces = 0, urgent_pieces = 0, failed = 0, bytes = 0, read_ns = 0;
        uint64_t held_ns = 0;          /* readers asleep with a piece to read, held by demand reads */
        uint64_t demand_reads = 0;
        uint64_t demand_met = 0;       /* demand reads that started with pieces in flight */
        uint64_t demand_wait_ns = 0;   /* ...summed from their start to the last such piece's end */
        uint64_t max_in_flight = 0;
    };
    Stats stats() const;

private:
    void loop(int id);
    /* Under mu_: the next unit to read from (urgent first, then the smallest order, the latter only
     * with no demand read in flight); prunes units with nothing left to start. */
    std::shared_ptr<ReadAheadUnit> pick(bool* urgent, bool* held);
    /* Under mu_: demand reads waiting on pieces settle once none is in flight. */
    void settle_locked(uint64_t now_ns);
    void cancel_locked(ReadAheadUnit& u);

    std::atomic<bool> started_{ false };   /* the readers run */
    bool    tried_ = false;
    int     n_threads_ = 0;
    uint64_t t0_ = 0;                      /* the clock's origin for the demand waits */
    int64_t piece_ = 512 << 10;
    std::vector<std::thread> threads_;

    mutable std::mutex mu_;
    std::condition_variable cv_;        /* readers: work, urgency, demand back to zero, stop */
    std::condition_variable done_cv_;   /* wait / forget: pieces in */
    bool stop_ = false;
    int  asleep_ = 0;                   /* readers waiting on cv_ */
    std::vector<std::shared_ptr<ReadAheadUnit>> queue_;
    std::vector<const ReadAheadUnit*> cur_;   /* by reader: the unit of its piece in flight */

    std::atomic<int> demand_{ 0 };
    std::atomic<uint64_t> demand_n_{ 0 };
    std::atomic<int> in_normal_{ 0 };   /* non-urgent pieces in flight */
    int      in_all_ = 0;               /* pieces in flight */
    int      waiting_ = 0;              /* demand reads that met pieces, until none is in flight */
    uint64_t waiting_since_ = 0;        /* the sum of their start times (ns) */
    Stats    st_;
};

}  /* namespace rad */

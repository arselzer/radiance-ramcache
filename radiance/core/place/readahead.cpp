/* readahead.cpp -- see readahead.h. */
#include "place/readahead.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <pthread.h>

namespace rad {

namespace {
uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  /* namespace */

ReadAheadUnit::ReadAheadUnit(const void* owner, uint64_t order, const PlaneExtent* ext, int n_ext,
                             int64_t piece, ReadFn read, void* ctx, std::shared_ptr<void> keep)
    : owner_(owner), order_(order), piece_(piece > 0 ? piece : 1), read_(read), ctx_(ctx),
      keep_(std::move(keep)) {
    n_ext_ = ext ? std::clamp(n_ext, 0, 2) : 0;
    for (int w = 0; w < n_ext_; ++w) {
        ext_[w] = ext[w];
        n_[w] = ext[w].len > 0 ? (int)((ext[w].len + piece_ - 1) / piece_) : 0;
        n_all_ += n_[w];
    }
}

PlaneExtent ReadAheadUnit::piece(int k) const {
    const int w = k < n_[0] ? 0 : 1;
    const int64_t at = (int64_t)(w ? k - n_[0] : k) * piece_;
    PlaneExtent p = ext_[w];
    p.off += at;
    p.len = std::min(piece_, ext_[w].len - at);
    p.dst += at;
    p.cap -= at;
    return p;
}

int ReadAheadUnit::started() const {
    const int stopped = stopped_at_.load(std::memory_order_acquire);
    return stopped >= 0 ? stopped : std::min(next_.load(std::memory_order_acquire), n_all_);
}

int64_t ReadAheadUnit::front(int w) const {
    if ((w != 0 && w != 1) || w >= n_ext_) return 0;
    const int s = started();
    const int k = w == 0 ? std::min(s, n_[0]) : std::max(s - n_[0], 0);
    return std::min((int64_t)k * piece_, ext_[w].len);
}

int64_t ReadAheadUnit::bytes_in() const { return bytes_in_.load(std::memory_order_acquire); }

bool ReadAheadUnit::whole() const {
    return done_.load(std::memory_order_acquire) == n_all_ && !failed_.load(std::memory_order_acquire);
}

PieceReader::~PieceReader() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    done_cv_.notify_all();
    for (std::thread& t : threads_) t.join();
}

PieceReader& PieceReader::process() {
    static PieceReader* p = new PieceReader();   /* never destroyed: ranks may read until exit */
    return *p;
}

bool PieceReader::start(int readers, int64_t piece) {
    std::unique_lock<std::mutex> lk(mu_);
    if (tried_) return started_.load(std::memory_order_acquire);
    tried_ = true;
    if (readers < 1 || readers > 8 || piece < (64 << 10) || piece > (16 << 20) || piece % 4096)
        return false;
    piece_ = piece;
    cur_.assign((size_t)readers, nullptr);
    t0_ = now_ns();
    try {
        for (int i = 0; i < readers; ++i) threads_.emplace_back(&PieceReader::loop, this, i);
    } catch (...) {
        /* The readers started wait on mu_, held here: stop them before they read anything. */
        stop_ = true;
        lk.unlock();
        cv_.notify_all();
        for (std::thread& t : threads_) t.join();
        threads_.clear();
        return false;
    }
    n_threads_ = readers;
    started_.store(true, std::memory_order_release);
    return true;
}

void PieceReader::demand_enter() {
    demand_.fetch_add(1, std::memory_order_seq_cst);
    if (!started_.load(std::memory_order_acquire)) return;
    demand_n_.fetch_add(1, std::memory_order_relaxed);
    /* A reader announces its piece (in_normal_) before its second look at the count: one of the
     * two sees the other. */
    if (in_normal_.load(std::memory_order_seq_cst) == 0) return;
    const uint64_t t = now_ns() - t0_;
    std::lock_guard<std::mutex> lk(mu_);
    if (in_normal_.load(std::memory_order_seq_cst) == 0) return;
    ++st_.demand_met;
    ++waiting_;
    waiting_since_ += t;
}

void PieceReader::demand_leave() {
    if (demand_.fetch_sub(1, std::memory_order_seq_cst) != 1) return;
    if (!started_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(mu_);
    /* Demand reads that met pieces and ended before those were in waited no longer than this. */
    if (waiting_) {
        st_.demand_wait_ns += (uint64_t)waiting_ * (now_ns() - t0_) - waiting_since_;
        waiting_ = 0;
        waiting_since_ = 0;
    }
    if (asleep_) cv_.notify_all();
}

void PieceReader::settle_locked(uint64_t now) {
    if (!waiting_ || in_normal_.load(std::memory_order_seq_cst) != 0) return;
    const uint64_t t = now - t0_;
    st_.demand_wait_ns += (uint64_t)waiting_ * t - waiting_since_;
    waiting_ = 0;
    waiting_since_ = 0;
}

void PieceReader::submit(std::shared_ptr<ReadAheadUnit> u) {
    if (!u) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_.load(std::memory_order_acquire) || stop_) return;
    queue_.push_back(std::move(u));
    if (asleep_) cv_.notify_all();
}

void PieceReader::cancel_locked(ReadAheadUnit& u) {
    if (u.cancelled_.exchange(true, std::memory_order_acq_rel)) return;
    const int was = u.next_.exchange(u.n_all_, std::memory_order_acq_rel);
    if (u.stopped_at_.load(std::memory_order_acquire) < 0)
        u.stopped_at_.store(std::min(was, u.n_all_), std::memory_order_release);
}

void PieceReader::cancel(ReadAheadUnit& u) {
    std::lock_guard<std::mutex> lk(mu_);
    cancel_locked(u);
    done_cv_.notify_all();   /* a wait() on it may now be satisfied */
}

bool PieceReader::take_whole(ReadAheadUnit& u) {
    int expected = 0;
    if (u.n_all_ == 0 ||
        !u.next_.compare_exchange_strong(expected, u.n_all_, std::memory_order_acq_rel))
        return false;
    u.stopped_at_.store(0, std::memory_order_release);
    return true;
}

void PieceReader::make_urgent(ReadAheadUnit& u) {
    if (u.urgent_.exchange(true, std::memory_order_acq_rel)) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (asleep_) cv_.notify_all();
}

/* As cancel_locked, but the unit is not cancelled: a reader that picked it finds nothing left to
 * start (its next_ is past the last piece) and pick() prunes it; wait() waits for the pieces
 * started by then. Under mu_, so no reader is between its pick and its piece's claim. */
int PieceReader::stop(ReadAheadUnit& u) {
    std::lock_guard<std::mutex> lk(mu_);
    const int was = u.next_.exchange(u.n_all_, std::memory_order_acq_rel);
    if (u.stopped_at_.load(std::memory_order_acquire) < 0) {
        u.stopped_at_.store(std::min(was, u.n_all_), std::memory_order_release);
        ++st_.stopped;
    }
    return u.stopped_at_.load(std::memory_order_acquire);
}

uint64_t PieceReader::wait(ReadAheadUnit& u) {
    /* A unit nobody hurries could wait behind this caller's own demand read. */
    if (u.stopped_at_.load(std::memory_order_acquire) < 0) make_urgent(u);
    std::unique_lock<std::mutex> lk(mu_);
    auto in = [&] {
        const int stopped = u.stopped_at_.load(std::memory_order_acquire);
        const int target = stopped >= 0 ? stopped : u.n_all_;
        return stop_ || u.done_.load(std::memory_order_acquire) >= target;
    };
    if (in()) return 0;
    const uint64_t t0 = now_ns();
    done_cv_.wait(lk, in);
    return now_ns() - t0;
}

void PieceReader::forget(const void* owner) {
    std::unique_lock<std::mutex> lk(mu_);
    for (const auto& q : queue_)
        if (q->owner_ == owner) cancel_locked(*q);
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(),
                                [&](const std::shared_ptr<ReadAheadUnit>& q) { return q->owner_ == owner; }),
                 queue_.end());
    done_cv_.wait(lk, [&] {
        for (const ReadAheadUnit* c : cur_)
            if (c && c->owner_ == owner) return false;
        return true;
    });
}

std::shared_ptr<ReadAheadUnit> PieceReader::pick(bool* urgent, bool* held) {
    *urgent = *held = false;
    /* Units with nothing left to start leave the queue (their pieces in flight still finish). */
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const std::shared_ptr<ReadAheadUnit>& q) {
                     return q->cancelled() || q->next_.load(std::memory_order_acquire) >= q->n_all_;
                 }),
                 queue_.end());
    const bool free = demand_.load(std::memory_order_seq_cst) == 0;
    const std::shared_ptr<ReadAheadUnit>* best = nullptr;
    bool best_urgent = false;
    for (const auto& q : queue_) {
        const bool u = q->urgent();
        if (!u && !free) {
            *held = true;
            continue;
        }
        if (!best || (u && !best_urgent) || (u == best_urgent && q->order_ < (*best)->order_)) {
            best = &q;
            best_urgent = u;
        }
    }
    if (!best) return nullptr;
    *urgent = best_urgent;
    return *best;
}

void PieceReader::loop(int id) {
    char name[16];
    std::snprintf(name, sizeof name, "rad-ra-%d", id);
    pthread_setname_np(pthread_self(), name);
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        if (stop_) return;
        bool urgent = false, held = false;
        std::shared_ptr<ReadAheadUnit> u = pick(&urgent, &held);
        int k = -1;
        if (u && !urgent) {
            in_normal_.fetch_add(1, std::memory_order_seq_cst);
            if (demand_.load(std::memory_order_seq_cst) != 0) {   /* a demand read came first */
                in_normal_.fetch_sub(1, std::memory_order_seq_cst);
                settle_locked(now_ns());
                u.reset();
                held = true;
            }
        }
        if (u) {
            k = u->next_.fetch_add(1, std::memory_order_acq_rel);
            if (k >= u->n_all_) {   /* taken, cancelled or exhausted meanwhile */
                if (!urgent) {
                    in_normal_.fetch_sub(1, std::memory_order_seq_cst);
                    settle_locked(now_ns());
                }
                continue;
            }
        }
        if (!u) {
            const uint64_t h0 = held ? now_ns() : 0;
            ++asleep_;
            cv_.wait(lk);
            --asleep_;
            if (held) st_.held_ns += now_ns() - h0;
            continue;
        }
        cur_[(size_t)id] = u.get();
        ++in_all_;
        st_.max_in_flight = std::max<uint64_t>(st_.max_in_flight, (uint64_t)in_all_);
        const PlaneExtent pc = u->piece(k);
        const int w = k < u->n_[0] ? 0 : 1;
        lk.unlock();
        const uint64_t t0 = now_ns();
        const int rc = u->read_ ? u->read_(u->ctx_, pc) : -1;
        const uint64_t t1 = now_ns();
        lk.lock();
        cur_[(size_t)id] = nullptr;
        --in_all_;
        if (!urgent) {
            in_normal_.fetch_sub(1, std::memory_order_seq_cst);
            settle_locked(t1);
        }
        ++st_.pieces;
        st_.urgent_pieces += urgent;
        st_.bytes += (uint64_t)pc.len;
        st_.read_ns += t1 - t0;
        if (rc < 0) {
            ++st_.failed;
            u->failed_.store(true, std::memory_order_release);
        } else {
            u->bytes_in_.fetch_add(pc.len, std::memory_order_acq_rel);
        }
        u->done_w_[w].fetch_add(1, std::memory_order_acq_rel);
        u->done_.fetch_add(1, std::memory_order_acq_rel);
        /* The last reference may go here (a unit its layer dropped): its buffer goes back before
         * forget() can see this reader idle. */
        u.reset();
        done_cv_.notify_all();
    }
}

PieceReader::Stats PieceReader::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = st_;
    s.demand_reads = demand_n_.load(std::memory_order_relaxed);
    return s;
}

}  /* namespace rad */

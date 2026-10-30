/* Native file-window readers: reusable CPU workers, with a blocking dispatch boundary.
 * No callback, context pointer or per-thread input survives run(). No GPU work is issued here.
 * Mover remains the sole dispatcher; the call mutex also serializes reset/reconfiguration.
 * Callbacks must not recursively call run/reset on this same object. Destruction requires the
 * ordinary C++ lifetime rule: no concurrent member callers. A reset may race a live run safely.
 *
 * post()/join() split a call in two for a caller with other work to do meanwhile: post() starts
 * the readers and returns at once, holding the call mutex; join() runs the caller's share, waits
 * and releases it. The same thread calls both; nothing between them may call run, post, reset or
 * stats on this object (another thread's call waits for the join, as it would for a run). post()
 * itself never waits: with the readers on another thread's call it posts nothing (that caller then
 * reads its own way, with no wait ahead of its other work).
 *
 * run() and join() take an optional WaitHook: a call made once on the caller's thread while it
 * waits for the readers, at a set time if they are not done by then (the CPU pool's pre-wake before
 * a miss read's jobs, RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE). Without one the wait is as
 * before.
 */
#pragma once
#include <sys/prctl.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace rad {

inline bool native_read_workers_opt_in(const char* value) noexcept {
    return value && value[0] == '1' && value[1] == '\0';
}

class NativeReadWorkers {
public:
    enum class Result { Complete, SerialFallback, Invalid, CallbackFailed };
    struct Stats {
        uint64_t thread_starts, pooled_dispatches, serial_dispatches, fallback_dispatches;
        uint64_t posted_dispatches = 0;   /* of the pooled ones, through post/join */
        uint64_t busy_posts = 0;          /* posts that found the readers on another call */
    };
    using Entry = void (*)(void*, int);
    using Launch = std::thread (*)(Entry, void*, int);
    /* fn(ctx) once, on the caller's thread and without the readers' state lock, when the caller
     * (its own share done) still waits for readers at `at_ns` (steady clock): at once if that has
     * passed by then; never when every reader is done first. `slack_ns` (0: the thread's own, 50 us
     * by default) is the caller's timer slack for that wait alone, restored after it. */
    struct WaitHook {
        void (*fn)(void*) = nullptr;
        void*   ctx = nullptr;
        int64_t at_ns = 0;
        unsigned long slack_ns = 0;
    };

    /* Launch injection lets the HOST checker exercise partial startup failure. The production
     * default is exactly std::thread; failed startup never publishes a read callback. */
    explicit NativeReadWorkers(Launch launch = launch_thread) noexcept : launch_(launch) {}
    NativeReadWorkers(const NativeReadWorkers&) = delete;
    NativeReadWorkers& operator=(const NativeReadWorkers&) = delete;
    ~NativeReadWorkers() { reset(); }

    template <class Work> Result run(int maximum, int active, Work& work, const WaitHook* hook = nullptr) {
        std::lock_guard<std::mutex> dispatch(call_mutex_);
        if (maximum < 1 || maximum > 8 || active < 1 || active > maximum)
            return Result::Invalid;
        if (maximum_ && maximum_ != maximum) stop_locked();
        if (active == 1) {
            ++serial_dispatches_;
            return invoke_serial(work, false);
        }
        if (unavailable_ || (!maximum_ && !start_locked(maximum))) {
            ++fallback_dispatches_;
            return invoke_serial(work, true);
        }

        ++pooled_dispatches_;
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            context_ = &work;
            invoke_ = [](void* context, int id) { (*static_cast<Work*>(context))(id); };
            active_ = active;
            pending_ = active - 1;
            callback_failed_ = false;
            ++generation_;
        }
        ready_.notify_all();
        bool caller_failed = false;
        try { work(0); } catch (...) { caller_failed = true; }
        {
            std::unique_lock<std::mutex> state(state_mutex_);
            wait_done(state, hook);
            const bool failed = callback_failed_ || caller_failed;
            /* All active readers have left the callback before its stack captures disappear. */
            context_ = nullptr;
            invoke_ = nullptr;
            active_ = 0;
            return failed ? Result::CallbackFailed : Result::Complete;
        }
    }

    /* run() without the caller's share up front: readers 1 .. `readers` (at most maximum - 1)
     * start on `work` now, and the caller returns at once still holding the call mutex. True:
     * posted, and join() must follow on this thread, with `work` alive until it returns. False:
     * nothing ran and nothing is held -- arguments out of range, no reader thread (run() would
     * read on the caller alone), or the readers busy with another thread's call (a run, or a
     * post not yet joined; counted in busy_posts) -- and the caller reads some other way. It never
     * waits for the readers: a caller that posts to get on with other work would otherwise wait
     * here, ahead of that work, for as long as the other call reads. */
    template <class Work> bool post(int maximum, int readers, Work& work) {
        if (maximum < 2 || maximum > 8 || readers < 1 || readers >= maximum) return false;
        std::unique_lock<std::mutex> call(call_mutex_, std::try_to_lock);
        if (!call.owns_lock()) {
            busy_posts_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (maximum_ && maximum_ != maximum) stop_locked();
        if (unavailable_ || (!maximum_ && !start_locked(maximum))) return false;
        ++pooled_dispatches_;
        ++posted_dispatches_;
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            context_ = &work;
            invoke_ = [](void* context, int id) { (*static_cast<Work*>(context))(id); };
            active_ = readers + 1;   /* ids 1 .. readers; id 0, the caller's, comes in join() */
            pending_ = readers;
            callback_failed_ = false;
            ++generation_;
        }
        ready_.notify_all();
        posted_ = true;
        call.release();   /* the call mutex stays held until join() */
        return true;
    }

    /* After a post() on this thread: work(0) on the caller -- whatever the readers have not
     * taken yet -- then the wait for them; the call mutex is released. Complete or
     * CallbackFailed as run(); Invalid when nothing was posted. */
    Result join(const WaitHook* hook = nullptr) {
        if (!posted_) return Result::Invalid;
        posted_ = false;
        /* This thread wrote context_ and invoke_ in post(); readers only read them. */
        bool caller_failed = false;
        try { invoke_(context_, 0); } catch (...) { caller_failed = true; }
        Result result;
        {
            std::unique_lock<std::mutex> state(state_mutex_);
            wait_done(state, hook);
            result = callback_failed_ || caller_failed ? Result::CallbackFailed : Result::Complete;
            context_ = nullptr;
            invoke_ = nullptr;
            active_ = 0;
        }
        call_mutex_.unlock();
        return result;
    }

    void reset() noexcept {
        std::lock_guard<std::mutex> dispatch(call_mutex_);
        stop_locked();
        unavailable_ = false;
    }

    Stats stats() const {
        std::lock_guard<std::mutex> dispatch(call_mutex_);
        return {thread_starts_, pooled_dispatches_, serial_dispatches_, fallback_dispatches_,
                posted_dispatches_, busy_posts_.load(std::memory_order_relaxed)};
    }
    uint64_t thread_starts() const { return stats().thread_starts; }

private:
    static std::thread launch_thread(Entry entry, void* context, int id) {
        return std::thread(entry, context, id);
    }
    /* The caller's wait for readers 1 .. active - 1, with the hook's call at its time. */
    void wait_done(std::unique_lock<std::mutex>& state, const WaitHook* hook) {
        if (hook && hook->fn && pending_ != 0) {
            const std::chrono::steady_clock::time_point at{ std::chrono::nanoseconds(hook->at_ns) };
            const int slack = hook->slack_ns ? prctl(PR_GET_TIMERSLACK, 0, 0, 0, 0) : -1;
            if (slack >= 0) prctl(PR_SET_TIMERSLACK, hook->slack_ns, 0UL, 0UL, 0UL);
            const bool done = done_.wait_until(state, at, [&] { return pending_ == 0; });
            if (slack >= 0) prctl(PR_SET_TIMERSLACK, (unsigned long)slack, 0UL, 0UL, 0UL);
            if (!done) {
                state.unlock();
                hook->fn(hook->ctx);
                state.lock();
            }
        }
        done_.wait(state, [&] { return pending_ == 0; });
    }
    template <class Work> static Result invoke_serial(Work& work, bool fallback) {
        try { work(0); } catch (...) { return Result::CallbackFailed; }
        return fallback ? Result::SerialFallback : Result::Complete;
    }
    static void enter(void* self, int id) {
        static_cast<NativeReadWorkers*>(self)->loop(id);
    }
    bool start_locked(int maximum) {
        try {
            threads_.reserve(static_cast<std::size_t>(maximum - 1));
            for (int id = 1; id < maximum; ++id) {
                threads_.emplace_back(launch_(&enter, this, id));
                ++thread_starts_;
            }
            maximum_ = maximum;
            return true;
        } catch (...) {
            /* Every partial worker is waiting for publication; stop/join before serial work. */
            stop_locked();
            unavailable_ = true;
            return false;
        }
    }
    void stop_locked() noexcept {
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        /* A join failure terminates rather than freeing buffers beneath an unretired reader. */
        for (std::thread& thread : threads_) if (thread.joinable()) thread.join();
        threads_.clear();
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            stopping_ = false;
            generation_ = 0;
            context_ = nullptr;
            invoke_ = nullptr;
            active_ = pending_ = maximum_ = 0;
            callback_failed_ = false;
        }
    }
    void loop(int id) {
        uint64_t observed = 0;
        std::unique_lock<std::mutex> state(state_mutex_);
        for (;;) {
            ready_.wait(state, [&] { return stopping_ || generation_ != observed; });
            if (stopping_) return;
            observed = generation_;
            if (id >= active_) continue;
            void* context = context_;
            auto invoke = invoke_;
            state.unlock();
            bool failed = false;
            try { invoke(context, id); } catch (...) { failed = true; }
            state.lock();
            callback_failed_ = callback_failed_ || failed;
            --pending_;
            if (!pending_) done_.notify_one();
        }
    }

    Launch launch_;
    mutable std::mutex call_mutex_;
    std::mutex state_mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> threads_;
    void* context_ = nullptr;
    void (*invoke_)(void*, int) = nullptr;
    uint64_t generation_ = 0, thread_starts_ = 0;
    uint64_t pooled_dispatches_ = 0, serial_dispatches_ = 0, fallback_dispatches_ = 0;
    uint64_t posted_dispatches_ = 0;
    std::atomic<uint64_t> busy_posts_{ 0 };   /* bumped without the call mutex (post's try_lock) */
    int maximum_ = 0, active_ = 0, pending_ = 0;
    bool stopping_ = false, unavailable_ = false, callback_failed_ = false;
    bool posted_ = false;   /* a post() awaits its join(); only the posting thread touches it */
};

}  // namespace rad

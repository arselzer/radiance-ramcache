/* native_read_workers_test.cpp -- core/place/native_read_workers.h: run() and the post()/join()
 * split the mover's plane reads use (RADIANCE_EXPERIMENTAL_DS41_MISS_POST). Every item is taken
 * exactly once, by readers 1 .. N after a post and by the caller in the join; arguments out of range
 * post nothing and hold nothing; a reader's exception comes back from the join; a run() on another
 * thread waits for the join (one call at a time), while a post() never waits for another thread's
 * call (it posts nothing); post and run share the threads (no restart); a failed thread start posts
 * nothing. A wait hook (the CPU pool's pre-wake in a miss read) runs once on the caller at its time
 * while readers are still out, at once when its time has passed, and never when they finish first.
 * Host threads only, at most 8. */
#include "rad_test.h"

#include "place/native_read_workers.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

using rad::NativeReadWorkers;

namespace {

/* Items handed out by an atomic cursor, as Mover::PlaneCall does; who took each one. */
struct Items {
    std::vector<std::atomic<int>> taken;   /* takes an item had; 1 each when done */
    std::vector<int> by;
    std::atomic<size_t> next{ 0 };
    std::atomic<int> caller_items{ 0 };
    int block_us = 0;                      /* each item's "read" */
    explicit Items(size_t n) : taken(n), by(n, -1) {}
    void operator()(int id) {
        for (size_t j; (j = next.fetch_add(1)) < taken.size();) {
            taken[j].fetch_add(1);
            by[j] = id;
            if (id == 0) caller_items.fetch_add(1);
            if (block_us) std::this_thread::sleep_for(std::chrono::microseconds(block_us));
        }
    }
    bool each_once() const {
        for (const auto& t : taken) if (t.load() != 1) return false;
        return true;
    }
};

}  // namespace

TEST(native_read_workers_post_join_takes_every_item_once) {
    NativeReadWorkers w;
    for (int round = 0; round < 50; ++round) {
        Items it(6);
        it.block_us = 200;
        REQUIRE(w.post(8, 6, it));
        /* Meanwhile the caller is free: the readers read without it. */
        const auto t0 = std::chrono::steady_clock::now();
        while (it.next.load() < 6 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
            std::this_thread::yield();
        CHECK_EQ(w.join(), NativeReadWorkers::Result::Complete);
        CHECK(it.each_once());
        for (int b : it.by) CHECK(b >= 1 && b <= 6);   /* six readers, none of them the caller */
    }
    const auto st = w.stats();
    CHECK_EQ(st.posted_dispatches, (uint64_t)50);
    CHECK_EQ(st.pooled_dispatches, (uint64_t)50);
    CHECK_EQ(st.thread_starts, (uint64_t)7);
}

TEST(native_read_workers_join_takes_what_the_readers_left) {
    NativeReadWorkers w;
    /* One reader for twelve slow items: the caller's join takes the rest. */
    Items it(12);
    it.block_us = 3000;
    REQUIRE(w.post(8, 1, it));
    CHECK_EQ(w.join(), NativeReadWorkers::Result::Complete);
    CHECK(it.each_once());
    CHECK(it.caller_items.load() >= 1);
    for (int b : it.by) CHECK(b == 0 || b == 1);
    /* A join with nothing posted is refused, and holds nothing. */
    CHECK_EQ(w.join(), NativeReadWorkers::Result::Invalid);
    Items again(3);
    CHECK_EQ(w.run(8, 3, again), NativeReadWorkers::Result::Complete);
    CHECK(again.each_once());
}

TEST(native_read_workers_post_out_of_range_posts_nothing) {
    NativeReadWorkers w;
    Items it(4);
    CHECK(!w.post(8, 0, it));
    CHECK(!w.post(8, 8, it));    /* readers are ids 1 .. maximum - 1 */
    CHECK(!w.post(9, 2, it));
    CHECK(!w.post(1, 1, it));
    CHECK_EQ(it.next.load(), (size_t)0);
    CHECK_EQ(w.join(), NativeReadWorkers::Result::Invalid);
    /* Nothing is held: run and stats proceed. */
    CHECK_EQ(w.run(8, 4, it), NativeReadWorkers::Result::Complete);
    CHECK(it.each_once());
    CHECK_EQ(w.stats().posted_dispatches, (uint64_t)0);
}

TEST(native_read_workers_post_and_run_share_the_threads) {
    NativeReadWorkers w;
    for (int round = 0; round < 20; ++round) {
        Items a(5), b(5);
        CHECK_EQ(w.run(8, 5, a), NativeReadWorkers::Result::Complete);
        REQUIRE(w.post(8, 5, b));
        CHECK_EQ(w.join(), NativeReadWorkers::Result::Complete);
        CHECK(a.each_once());
        CHECK(b.each_once());
    }
    CHECK_EQ(w.stats().thread_starts, (uint64_t)7);
}

TEST(native_read_workers_a_reader_exception_comes_back_from_the_join) {
    NativeReadWorkers w;
    struct Throws {
        std::atomic<int> calls{ 0 };
        void operator()(int id) {
            calls.fetch_add(1);
            if (id == 2) throw std::runtime_error("reader 2");
        }
    } t;
    REQUIRE(w.post(8, 3, t));
    CHECK_EQ(w.join(), NativeReadWorkers::Result::CallbackFailed);
    CHECK_EQ(t.calls.load(), 4);   /* readers 1-3 and the caller */
    Items it(3);
    CHECK_EQ(w.run(8, 3, it), NativeReadWorkers::Result::Complete);
    CHECK(it.each_once());
}

TEST(native_read_workers_a_run_on_another_thread_waits_for_the_join) {
    NativeReadWorkers w;
    Items posted(4);
    REQUIRE(w.post(8, 4, posted));
    std::atomic<bool> joining{ false };
    std::atomic<int> saw_joining{ -1 };
    std::thread other([&] {
        struct Probe {
            std::atomic<bool>* joining;
            std::atomic<int>* saw;
            void operator()(int id) { if (id == 0) saw->store(joining->load() ? 1 : 0); }
        } probe{ &joining, &saw_joining };
        (void)w.run(8, 2, probe);
    });
    /* The other thread's run cannot start before the join releases the call mutex. */
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK_EQ(saw_joining.load(), -1);
    joining.store(true);
    CHECK_EQ(w.join(), NativeReadWorkers::Result::Complete);
    other.join();
    CHECK_EQ(saw_joining.load(), 1);
    CHECK(posted.each_once());
}

TEST(native_read_workers_a_post_never_waits_for_another_threads_call) {
    NativeReadWorkers w;
    /* Another thread's run, held in its caller's share until released (after 2 s at the latest, so
     * a post that waited would fail the check below instead of hanging). */
    std::atomic<bool> inside{ false }, release{ false };
    struct Hold {
        std::atomic<bool>* inside;
        std::atomic<bool>* release;
        void operator()(int id) {
            if (id != 0) return;
            inside->store(true);
            while (!release->load()) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    } hold{ &inside, &release };
    std::thread other([&] { (void)w.run(8, 2, hold); });
    std::thread watchdog([&] {
        const auto t0 = std::chrono::steady_clock::now();
        while (!release.load() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        release.store(true);
    });
    const auto t0 = std::chrono::steady_clock::now();
    while (!inside.load() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
        std::this_thread::yield();
    if (!inside.load()) {   /* no thread left behind a REQUIRE */
        release.store(true);
        watchdog.join();
        other.join();
    }
    REQUIRE(inside.load());
    Items it(4);
    CHECK(!w.post(8, 3, it));
    CHECK(!release.load());   /* it came back while the run still held the readers */
    CHECK_EQ(it.next.load(), (size_t)0);
    CHECK_EQ(w.join(), NativeReadWorkers::Result::Invalid);   /* nothing posted, nothing held */
    release.store(true);
    watchdog.join();
    other.join();
    CHECK_EQ(w.stats().busy_posts, (uint64_t)1);
    /* Free again: the post goes through. */
    REQUIRE(w.post(8, 3, it));
    CHECK_EQ(w.join(), NativeReadWorkers::Result::Complete);
    CHECK(it.each_once());

    /* A post on another thread holds the readers the same way until its join. */
    std::atomic<bool> posted{ false }, go{ false }, joined{ false };
    Items theirs(3);
    std::thread poster([&] {
        if (!w.post(8, 2, theirs)) return;
        posted.store(true);
        const auto p0 = std::chrono::steady_clock::now();
        while (!go.load() && std::chrono::steady_clock::now() - p0 < std::chrono::seconds(2))
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        (void)w.join();
        joined.store(true);
    });
    const auto t1 = std::chrono::steady_clock::now();
    while (!posted.load() && std::chrono::steady_clock::now() - t1 < std::chrono::seconds(5))
        std::this_thread::yield();
    if (!posted.load()) {
        go.store(true);
        poster.join();
    }
    REQUIRE(posted.load());
    Items mine(2);
    CHECK(!w.post(8, 1, mine));
    CHECK(!joined.load());   /* it came back while the other post still held the readers */
    CHECK_EQ(mine.next.load(), (size_t)0);
    go.store(true);
    poster.join();
    CHECK(theirs.each_once());
    const auto st = w.stats();
    CHECK_EQ(st.busy_posts, (uint64_t)2);
    CHECK_EQ(st.posted_dispatches, (uint64_t)2);
}

namespace {

/* The caller's share is short (id 0), the readers' long: the caller waits for them. */
struct Uneven {
    int caller_ms = 1, reader_ms = 100;
    std::atomic<int> done{ 0 };
    void operator()(int id) {
        std::this_thread::sleep_for(std::chrono::milliseconds(id == 0 ? caller_ms : reader_ms));
        done.fetch_add(1);
    }
};

/* A hook that notes its calls, when, and how many readers were done then. */
struct HookProbe {
    std::atomic<int> calls{ 0 }, done_then{ -1 };
    std::atomic<int64_t> at_ns{ 0 };
    std::atomic<int>* done = nullptr;
    std::thread::id thread;
    static void fn(void* self) {
        auto* h = static_cast<HookProbe*>(self);
        h->calls.fetch_add(1);
        h->done_then.store(h->done ? h->done->load() : -1);
        h->at_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()).count());
        h->thread = std::this_thread::get_id();
    }
};

int64_t now_plus_ns(int64_t ns) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() + ns;
}

}  // namespace

TEST(native_read_workers_a_wait_hook_runs_once_at_its_time_while_readers_read) {
    NativeReadWorkers w;
    Uneven u;
    HookProbe h;
    h.done = &u.done;
    NativeReadWorkers::WaitHook hook{ &HookProbe::fn, &h, now_plus_ns(15'000'000) };
    const int64_t t0 = now_plus_ns(0);
    CHECK_EQ(w.run(8, 3, u, &hook), NativeReadWorkers::Result::Complete);
    CHECK_EQ(h.calls.load(), 1);
    CHECK_EQ(u.done.load(), 3);
    CHECK_EQ(h.done_then.load(), 1);   /* the caller's share only: both readers still out */
    CHECK(h.thread == std::this_thread::get_id());
    CHECK(h.at_ns.load() >= hook.at_ns);   /* not before its time ... */
    CHECK(h.at_ns.load() - t0 < 90'000'000);   /* ... and before the readers' 100 ms */
}

TEST(native_read_workers_a_wait_hook_past_its_time_runs_as_the_caller_comes_to_wait) {
    NativeReadWorkers w;
    Uneven u;
    HookProbe h;
    h.done = &u.done;
    NativeReadWorkers::WaitHook hook{ &HookProbe::fn, &h, now_plus_ns(-1'000'000'000) };
    CHECK_EQ(w.run(8, 4, u, &hook), NativeReadWorkers::Result::Complete);
    CHECK_EQ(h.calls.load(), 1);
    CHECK_EQ(h.done_then.load(), 1);
    /* The same through post and join. */
    Uneven v;
    HookProbe g;
    g.done = &v.done;
    REQUIRE(w.post(8, 2, v));
    NativeReadWorkers::WaitHook later{ &HookProbe::fn, &g, now_plus_ns(10'000'000) };
    CHECK_EQ(w.join(&later), NativeReadWorkers::Result::Complete);
    CHECK_EQ(g.calls.load(), 1);
    CHECK_EQ(v.done.load(), 3);
    CHECK(g.done_then.load() <= 1);   /* at most the caller's share in by then */
}

TEST(native_read_workers_a_wait_hook_never_runs_when_the_readers_finish_first) {
    NativeReadWorkers w;
    /* Readers as quick as the caller, the hook's time far off: the wait ends without it, and
     * without waiting for its time. */
    Uneven u;
    u.reader_ms = 1;
    HookProbe h;
    NativeReadWorkers::WaitHook hook{ &HookProbe::fn, &h, now_plus_ns(5'000'000'000) };
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_EQ(w.run(8, 3, u, &hook), NativeReadWorkers::Result::Complete);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2));
    CHECK_EQ(h.calls.load(), 0);
    /* Readers done while the caller's own share runs: no hook, though its time has passed. */
    Uneven v;
    v.caller_ms = 300;
    v.reader_ms = 1;
    HookProbe g;
    NativeReadWorkers::WaitHook past{ &HookProbe::fn, &g, now_plus_ns(-1'000'000'000) };
    CHECK_EQ(w.run(8, 3, v, &past), NativeReadWorkers::Result::Complete);
    CHECK_EQ(g.calls.load(), 0);
    /* A null hook, or one without a function, is the plain wait. */
    Uneven x;
    NativeReadWorkers::WaitHook none{};
    CHECK_EQ(w.run(8, 3, x, &none), NativeReadWorkers::Result::Complete);
    CHECK_EQ(w.run(8, 3, x, nullptr), NativeReadWorkers::Result::Complete);
}

TEST(native_read_workers_a_failed_start_posts_nothing) {
    /* Every thread start fails: the post is refused, nothing is held, run() reads on the caller. */
    NativeReadWorkers w([](NativeReadWorkers::Entry, void*, int) -> std::thread {
        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
    });
    Items it(3);
    CHECK(!w.post(8, 3, it));
    CHECK_EQ(it.next.load(), (size_t)0);
    CHECK_EQ(w.run(8, 3, it), NativeReadWorkers::Result::SerialFallback);
    CHECK(it.each_once());
    for (int b : it.by) CHECK_EQ(b, 0);
}

RAD_TEST_MAIN()

/* native_read_workers_test.cpp -- core/place/native_read_workers.h: run() and the post()/join()
 * split the mover's plane reads use (RADIANCE_EXPERIMENTAL_DS41_MISS_POST). Every item is taken
 * exactly once, by readers 1 .. N after a post and by the caller in the join; arguments out of range
 * post nothing and hold nothing; a reader's exception comes back from the join; a run() on another
 * thread waits for the join (one call at a time); post and run share the threads (no restart); a
 * failed thread start posts nothing. Host threads only, at most 8. */
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

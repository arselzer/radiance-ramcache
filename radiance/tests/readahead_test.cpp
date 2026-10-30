/* readahead_test.cpp -- the read-ahead's piece readers and hand-off (core/place/readahead.h): pieces
 * cover each extent once and land where read_unit_planes would put the bytes (through O_DIRECT
 * too); no non-urgent piece starts while a demand read is in flight, and at most the readers' count
 * is ever in flight; urgent units finish behind demand reads; a cancelled unit starts no piece and
 * keeps its buffer until the pieces in flight are in; a stopped unit starts no piece and leaves the
 * rest past front() to its layer; a unit is whole only with every piece in and none failed;
 * take_whole only before any piece; earlier orders first; forget waits for an owner's pieces; demand
 * reads that meet pieces are counted. */
#include "rad_test.h"

#include "place/readahead.h"
#include "device/directio.h"
#include "rad_core.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using rad::PieceReader;
using rad::PlaneExtent;
using rad::ReadAheadUnit;

namespace {

constexpr int64_t kA = 4096, kPiece = 64 << 10;
int64_t align_up(int64_t v) { return (v + kA - 1) / kA * kA; }
void sleep_us(int us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }

/* A container in memory, read the way FileTier::read reads: a 4 KiB-aligned offset, `len` rounded
 * up to the alignment and no more than the destination holds. Records every piece. */
struct Source {
    std::vector<uint8_t> bytes;
    std::atomic<int> in_flight{ 0 }, max_in_flight{ 0 }, reads{ 0 }, bad{ 0 };
    std::atomic<int> fail_at{ -1 };             /* the read with this index fails */
    std::atomic<int> delay_us{ 0 };
    std::atomic<int64_t> last_start_ns{ 0 };
    std::mutex mu;
    std::vector<std::pair<int64_t, int64_t>> log;   /* (offset, length) in read order */
    std::vector<uint64_t> owner_log;                /* the ctx tag of each read, in order */

    explicit Source(int64_t n) : bytes((size_t)n) {
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (auto& b : bytes) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b = (uint8_t)(s >> 32); }
    }
};

struct Ctx {
    Source*  src;
    uint64_t tag;
    std::atomic<bool>* gone = nullptr;   /* set when the destination is freed */
};

int64_t now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int fake_read(void* c, const PlaneExtent& p) {
    Ctx* ctx = static_cast<Ctx*>(c);
    Source& s = *ctx->src;
    const int idx = s.reads.fetch_add(1);
    s.last_start_ns.store(now_ns());
    const int n = s.in_flight.fetch_add(1) + 1;
    for (int m = s.max_in_flight.load(); n > m && !s.max_in_flight.compare_exchange_weak(m, n);) {}
    if (p.off % kA || p.len <= 0 || align_up(p.len) > p.cap || p.off + p.len > (int64_t)s.bytes.size())
        s.bad.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(s.mu);
        s.log.emplace_back(p.off, p.len);
        s.owner_log.push_back(ctx->tag);
    }
    if (const int d = s.delay_us.load()) sleep_us(d);
    if (ctx->gone && ctx->gone->load()) s.bad.fetch_add(1);   /* a piece landing in freed memory */
    std::memcpy(p.dst, s.bytes.data() + p.off, (size_t)p.len);
    s.in_flight.fetch_sub(1);
    return idx == s.fail_at.load() ? -5 : 0;
}

/* A unit's destination: two buffers with the O_DIRECT tail, and a flag for its lifetime. */
struct Dest {
    std::vector<uint8_t> a, b;
    std::atomic<bool>* gone = nullptr;
    ~Dest() { if (gone) gone->store(true); }
};

struct Unit {
    std::shared_ptr<Dest> dest;
    std::shared_ptr<ReadAheadUnit> u;
    PlaneExtent ext[2];
};

/* Extents [off0, off0 + len0) and [off1, off1 + len1) of the source as one unit. */
Unit make_unit(const void* owner, uint64_t order, Ctx* ctx, int64_t off0, int64_t len0, int64_t off1,
               int64_t len1, int64_t piece = kPiece, std::atomic<bool>* gone = nullptr) {
    Unit x;
    x.dest = std::make_shared<Dest>();
    x.dest->a.assign((size_t)align_up(len0), 0);
    x.dest->b.assign((size_t)align_up(len1), 0);
    x.dest->gone = gone;
    x.ext[0] = PlaneExtent{ 0, off0, len0, x.dest->a.data(), (int64_t)x.dest->a.size() };
    x.ext[1] = PlaneExtent{ 0, off1, len1, x.dest->b.data(), (int64_t)x.dest->b.size() };
    x.u = std::make_shared<ReadAheadUnit>(owner, order, x.ext, 2, piece, &fake_read, ctx, x.dest);
    return x;
}

bool same(const Unit& x, const Source& s) {
    return std::memcmp(x.dest->a.data(), s.bytes.data() + x.ext[0].off, (size_t)x.ext[0].len) == 0 &&
           std::memcmp(x.dest->b.data(), s.bytes.data() + x.ext[1].off, (size_t)x.ext[1].len) == 0;
}

/* Waits up to `ms` for cond. */
template <class F> bool eventually(F cond, int ms = 2000) {
    for (int i = 0; i < ms * 10; ++i) {
        if (cond()) return true;
        sleep_us(100);
    }
    return cond();
}

}  // namespace

TEST(readahead_pieces_cover_each_extent_once_and_land_in_place) {
    Source src(8 << 20);
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(4, kPiece));
    CHECK(!r.start(2, 2 * kPiece) || r.readers() == 4);   /* once: the second changes nothing */
    CHECK_EQ(r.piece_bytes(), kPiece);
    /* Lengths off the piece size and off the alignment; extents in either order in the file. */
    Unit x = make_unit(&ctx, 1, &ctx, 3 * kA, 1300001, 5 << 20, 700003);
    CHECK_EQ(x.u->pieces(0), (int)((1300001 + kPiece - 1) / kPiece));
    CHECK_EQ(x.u->pieces(1), (int)((700003 + kPiece - 1) / kPiece));
    CHECK_EQ(x.u->pieces(), x.u->pieces(0) + x.u->pieces(1));
    const PlaneExtent last = x.u->piece(x.u->pieces(0) - 1);
    CHECK_EQ(last.off + last.len, (int64_t)(3 * kA + 1300001));
    CHECK(last.dst + last.cap == x.dest->a.data() + x.dest->a.size());
    r.submit(x.u);
    CHECK(eventually([&] { return x.u->in() == x.u->pieces(); }));
    CHECK_EQ(r.wait(*x.u) < (uint64_t)1000000000, true);
    CHECK(x.u->whole());
    CHECK_EQ(x.u->in(0), x.u->pieces(0));
    CHECK_EQ(x.u->in(1), x.u->pieces(1));
    CHECK_EQ(x.u->bytes_in(), (int64_t)(1300001 + 700003));
    CHECK(same(x, src));
    CHECK_EQ(src.bad.load(), 0);
    /* Every byte read once: the pieces tile both extents exactly. */
    std::vector<uint8_t> hit(src.bytes.size(), 0);
    for (const auto& [off, len] : src.log)
        for (int64_t i = off; i < off + len; ++i) ++hit[(size_t)i];
    int64_t once = 0, more = 0;
    for (uint8_t h : hit) { once += h == 1; more += h > 1; }
    CHECK_EQ(once, (int64_t)(1300001 + 700003));
    CHECK_EQ(more, (int64_t)0);
    const PieceReader::Stats st = r.stats();
    CHECK_EQ(st.pieces, (uint64_t)x.u->pieces());
    CHECK_EQ(st.bytes, (uint64_t)(1300001 + 700003));
    CHECK(st.max_in_flight <= 4);
}

/* The same through O_DIRECT reads of a file, as Mover::read_piece reads the container. */
TEST(readahead_pieces_read_a_file_through_odirect) {
    const std::string path = radtest::tmpdir() + "/readahead_container.bin";
    Source src(3 << 20);
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        REQUIRE(f);
        REQUIRE_EQ(std::fwrite(src.bytes.data(), 1, src.bytes.size(), f), src.bytes.size());
        std::fclose(f);
    }
    rad::DirectFile file;
    if (file.open(path.c_str(), rad::RAD_DIO_READ) < 0) SKIP("no O_DIRECT in %s", path.c_str());
    struct FileCtx { rad::DirectFile* f; std::atomic<int> bad{ 0 }; } fc{ &file };
    auto read = [](void* c, const PlaneExtent& p) -> int {
        auto* x = static_cast<FileCtx*>(c);
        const int64_t want = align_up(p.len);   /* FileTier::read */
        if (want > p.cap || (uintptr_t)p.dst % kA) { x->bad.fetch_add(1); return -1; }
        return x->f->read_covering(p.dst, p.off, want, p.len);
    };
    const int64_t len0 = 1000003, len1 = (int64_t)src.bytes.size() - (1 << 20) - 5 * kA;   /* to the file's end */
    const int64_t cap0 = align_up(len0) + 2 * kA, cap1 = align_up(len1);
    uint8_t* b0 = (uint8_t*)std::aligned_alloc(kA, (size_t)cap0);
    uint8_t* b1 = (uint8_t*)std::aligned_alloc(kA, (size_t)cap1);
    REQUIRE(b0 && b1);
    std::shared_ptr<void> keep(nullptr, [b0, b1](void*) { std::free(b0); std::free(b1); });
    PlaneExtent ext[2] = { { 0, 2 * kA, len0, b0, cap0 }, { 0, (1 << 20) + 5 * kA, len1, b1, cap1 } };
    PieceReader r;
    REQUIRE(r.start(3, kPiece));
    auto u = std::make_shared<ReadAheadUnit>(&fc, 1, ext, 2, kPiece, +read, &fc, keep);
    r.submit(u);
    r.wait(*u);
    CHECK(u->whole());
    CHECK_EQ(fc.bad.load(), 0);
    CHECK(std::memcmp(b0, src.bytes.data() + 2 * kA, (size_t)len0) == 0);
    CHECK(std::memcmp(b1, src.bytes.data() + (1 << 20) + 5 * kA, (size_t)len1) == 0);
    u.reset();
    keep.reset();
}

TEST(readahead_no_piece_starts_while_a_demand_read_is_in_flight) {
    Source src(16 << 20);
    src.delay_us = 300;
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(4, kPiece));
    Unit x = make_unit(&ctx, 1, &ctx, 0, 6 << 20, 8 << 20, 3 << 20);
    r.submit(x.u);
    CHECK(eventually([&] { return x.u->in() >= 8; }));
    r.demand_enter();
    CHECK_EQ(r.demand_count(), 1);
    /* Pieces already in flight finish; none starts after them while the demand read lasts. */
    sleep_us(5000);
    const int frozen = src.reads.load();
    sleep_us(40000);
    CHECK_EQ(src.reads.load(), frozen);
    CHECK_EQ(x.u->started(), x.u->in());
    r.demand_enter();   /* two demand reads (two ranks): still held until both are out */
    r.demand_leave();
    sleep_us(10000);
    CHECK_EQ(src.reads.load(), frozen);
    r.demand_leave();
    CHECK_EQ(r.demand_count(), 0);
    r.wait(*x.u);
    CHECK(x.u->whole());
    CHECK(same(x, src));
    CHECK(src.max_in_flight.load() <= 4);
    CHECK(src.max_in_flight.load() >= 2);
    const PieceReader::Stats st = r.stats();
    CHECK_EQ(st.demand_reads, (uint64_t)2);
    CHECK(st.held_ns > 0);
}

TEST(readahead_urgent_units_finish_behind_demand_reads_and_go_first) {
    Source src(16 << 20);
    src.delay_us = 200;
    Ctx ctx{ &src, 1 };
    Ctx ctx2{ &src, 2 };
    PieceReader r;
    REQUIRE(r.start(2, kPiece));
    Unit normal = make_unit(&ctx2, 1, &ctx2, 0, 2 << 20, 4 << 20, 1 << 20);   /* the earlier order */
    Unit needed = make_unit(&ctx, 9, &ctx, 8 << 20, 2 << 20, 12 << 20, 1 << 20);
    r.submit(normal.u);
    r.submit(needed.u);
    CHECK(eventually([&] { return normal.u->in() >= 2; }));
    /* The needed unit's layer: its own demand read, and the unit made urgent. */
    r.demand_enter();
    r.make_urgent(*needed.u);
    const uint64_t waited = r.wait(*needed.u);
    CHECK(needed.u->whole());
    CHECK(same(needed, src));
    CHECK(waited > 0);
    CHECK(!normal.u->whole());   /* held back the whole time */
    r.demand_leave();
    r.wait(*normal.u);
    CHECK(normal.u->whole());
    CHECK(same(normal, src));
    CHECK(r.stats().urgent_pieces >= (uint64_t)(needed.u->pieces() - 2));
}

TEST(readahead_cancel_starts_no_piece_and_keeps_the_buffer_while_one_is_in_flight) {
    Source src(16 << 20);
    src.delay_us = 2000;
    std::atomic<bool> gone{ false };
    Ctx ctx{ &src, 1, &gone };
    PieceReader r;
    REQUIRE(r.start(2, kPiece));
    Unit x = make_unit(&ctx, 1, &ctx, 0, 4 << 20, 8 << 20, 2 << 20, kPiece, &gone);
    r.submit(x.u);
    CHECK(eventually([&] { return x.u->in() >= 2 && src.in_flight.load() > 0; }));
    r.cancel(*x.u);
    CHECK(x.u->cancelled());
    const int started = x.u->started();
    CHECK(started < x.u->pieces());
    /* The unit and its buffer dropped by everyone but the reader with a piece in flight. */
    std::weak_ptr<ReadAheadUnit> weak = x.u;
    const bool in_flight = src.in_flight.load() > 0;
    x.u.reset();
    x.dest.reset();
    CHECK(!in_flight || !gone.load());   /* the reader's piece keeps it */
    CHECK(eventually([&] { return src.in_flight.load() == 0; }));
    sleep_us(5000);
    CHECK_EQ(src.reads.load(), started);   /* nothing started after the cancel */
    CHECK(eventually([&] { return gone.load(); }));
    CHECK(eventually([&] { return weak.expired(); }));
    CHECK_EQ(src.bad.load(), 0);
}

TEST(readahead_wait_on_a_cancelled_unit_returns_once_its_started_pieces_are_in) {
    Source src(16 << 20);
    src.delay_us = 1500;
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(4, kPiece));
    Unit x = make_unit(&ctx, 1, &ctx, 0, 4 << 20, 8 << 20, 2 << 20);
    r.submit(x.u);
    CHECK(eventually([&] { return x.u->in() >= 1 && src.in_flight.load() > 0; }));
    r.cancel(*x.u);
    r.wait(*x.u);
    CHECK_EQ(x.u->in(), x.u->started());
    CHECK(!x.u->whole());
    CHECK(x.u->bytes_in() > 0);
}

/* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT: a stopped unit starts no further piece, its pieces in
 * flight finish and wait() waits for them; front(w) covers exactly the pieces started, so the layer
 * reading [front(w), len) of each extent itself -- the same bytes at the same offsets, here while a
 * piece is still in flight -- leaves the unit as the readers would have. Not a cancel; stopped once,
 * and a later stop or cancel changes nothing. */
TEST(readahead_stop_leaves_the_rest_to_the_layer) {
    Source src(16 << 20);
    src.delay_us = 1500;
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(4, kPiece));
    /* Extent 0 of 21 pieces (its last one short), extent 1 of 9. */
    Unit x = make_unit(&ctx, 1, &ctx, 3 * kA, 20 * kPiece + 12345, 8 << 20, 9 * kPiece - 777);
    r.submit(x.u);
    CHECK(eventually([&] { return x.u->in() >= 5 && src.in_flight.load() > 0; }));
    const int started = r.stop(*x.u);
    CHECK_EQ(started, x.u->started());
    CHECK(started >= 5);
    REQUIRE(started < x.u->pieces(0));
    CHECK(!x.u->cancelled());
    CHECK(!x.u->urgent());
    CHECK_EQ(r.stop(*x.u), started);
    CHECK_EQ(r.stats().stopped, (uint64_t)1);
    const int64_t f0 = x.u->front(0), f1 = x.u->front(1);
    CHECK_EQ(f0, (int64_t)started * kPiece);
    CHECK_EQ(f1, (int64_t)0);
    CHECK_EQ(x.u->extent(0).len, x.ext[0].len);
    CHECK_EQ(x.u->extent(2).len, (int64_t)0);
    /* The layer's own read of the rest, with the unit's pieces in flight beside it. */
    std::memcpy(x.dest->a.data() + f0, src.bytes.data() + x.ext[0].off + f0, (size_t)(x.ext[0].len - f0));
    std::memcpy(x.dest->b.data() + f1, src.bytes.data() + x.ext[1].off + f1, (size_t)(x.ext[1].len - f1));
    r.wait(*x.u);
    CHECK_EQ(x.u->in(), started);
    CHECK(!x.u->whole());   /* the rest was not the readers' */
    CHECK(!x.u->failed());
    CHECK(same(x, src));
    sleep_us(5000);
    CHECK_EQ(src.reads.load(), started);   /* nothing started after the stop */
    {
        /* Every byte the readers read lies in front of the stop. */
        std::lock_guard<std::mutex> lk(src.mu);
        for (const auto& [off, len] : src.log) {
            const bool e0 = off >= x.ext[0].off && off + len <= x.ext[0].off + f0;
            const bool e1 = off >= x.ext[1].off && off + len <= x.ext[1].off + f1;
            CHECK(e0 || e1);
        }
    }
    r.cancel(*x.u);
    CHECK_EQ(x.u->started(), started);
    CHECK_EQ(x.u->front(0), f0);
    CHECK_EQ(r.stats().stopped, (uint64_t)1);
    CHECK_EQ(src.bad.load(), 0);
}

/* front() of a unit stopped before any piece (nothing the readers': the layer reads it all) and
 * after its every piece (the extents' whole lengths: nothing left to read but the wait). */
TEST(readahead_front_before_any_piece_and_after_every_piece) {
    Source src(16 << 20);
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(2, kPiece));
    r.demand_enter();   /* nothing starts */
    Unit a = make_unit(&ctx, 1, &ctx, 0, 3 * kPiece, 4 << 20, kPiece + 5);
    r.submit(a.u);
    sleep_us(3000);
    CHECK_EQ(r.stop(*a.u), 0);
    CHECK_EQ(a.u->front(0), (int64_t)0);
    CHECK_EQ(a.u->front(1), (int64_t)0);
    CHECK_EQ(r.wait(*a.u), (uint64_t)0);
    r.demand_leave();
    sleep_us(3000);
    CHECK_EQ(src.reads.load(), 0);
    Unit b = make_unit(&ctx, 2, &ctx, 8 << 20, 3 * kPiece, 12 << 20, kPiece + 5);
    r.submit(b.u);
    r.wait(*b.u);
    REQUIRE(b.u->whole());
    CHECK_EQ(r.stop(*b.u), b.u->pieces());
    CHECK_EQ(b.u->front(0), b.ext[0].len);
    CHECK_EQ(b.u->front(1), b.ext[1].len);
    CHECK(same(b, src));
    CHECK_EQ(r.stats().stopped, (uint64_t)2);
}

TEST(readahead_take_whole_only_before_any_piece) {
    Source src(16 << 20);
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(2, kPiece));
    r.demand_enter();   /* nothing starts */
    Unit a = make_unit(&ctx, 1, &ctx, 0, 1 << 20, 2 << 20, 1 << 20);
    r.submit(a.u);
    sleep_us(5000);
    CHECK_EQ(src.reads.load(), 0);
    CHECK(r.take_whole(*a.u));
    CHECK(!r.take_whole(*a.u));
    CHECK_EQ(r.wait(*a.u), (uint64_t)0);   /* taken whole: nothing of the readers' to wait for */
    r.demand_leave();
    Unit b = make_unit(&ctx, 2, &ctx, 4 << 20, 1 << 20, 6 << 20, 1 << 20);
    r.submit(b.u);
    r.wait(*b.u);
    CHECK(b.u->whole());
    CHECK(!r.take_whole(*b.u));
    sleep_us(2000);
    /* a was never read: every read was b's. */
    std::lock_guard<std::mutex> lk(src.mu);
    for (const auto& [off, len] : src.log) CHECK(off >= (4 << 20));
}

TEST(readahead_a_failed_piece_is_never_whole) {
    Source src(16 << 20);
    Ctx ctx{ &src, 1 };
    src.fail_at = 3;
    PieceReader r;
    REQUIRE(r.start(4, kPiece));
    Unit x = make_unit(&ctx, 1, &ctx, 0, 1 << 20, 2 << 20, 1 << 20);
    r.submit(x.u);
    r.wait(*x.u);
    CHECK_EQ(x.u->in(), x.u->pieces());
    CHECK(x.u->failed());
    CHECK(!x.u->whole());
    CHECK_EQ(r.stats().failed, (uint64_t)1);
}

TEST(readahead_earlier_orders_first) {
    Source src(32 << 20);
    PieceReader r;
    REQUIRE(r.start(1, kPiece));
    r.demand_enter();
    std::vector<Ctx> ctx(3);
    std::vector<Unit> u;
    const uint64_t order[3] = { 5, 1, 3 };
    for (int i = 0; i < 3; ++i) {
        ctx[(size_t)i] = Ctx{ &src, order[i] };
        u.push_back(make_unit(&ctx[(size_t)i], order[i], &ctx[(size_t)i], (int64_t)i * (8 << 20),
                              256 << 10, (int64_t)i * (8 << 20) + (4 << 20), 128 << 10));
        r.submit(u.back().u);
    }
    r.demand_leave();
    /* (wait() would make a unit urgent, which goes first.) */
    for (auto& x : u) CHECK(eventually([&] { return x.u->whole(); }));
    std::lock_guard<std::mutex> lk(src.mu);
    REQUIRE_EQ(src.owner_log.size(), (size_t)(3 * 6));
    CHECK(std::is_sorted(src.owner_log.begin(), src.owner_log.end()));
    CHECK_EQ(src.owner_log.front(), (uint64_t)1);
    CHECK_EQ(src.owner_log.back(), (uint64_t)5);
}

TEST(readahead_forget_waits_for_the_owners_pieces) {
    Source src(32 << 20);
    src.delay_us = 3000;
    std::atomic<bool> gone{ false };
    Ctx a{ &src, 1, &gone }, b{ &src, 2 };
    PieceReader r;
    REQUIRE(r.start(2, kPiece));
    Unit x = make_unit(&a, 1, &a, 0, 2 << 20, 4 << 20, 1 << 20, kPiece, &gone);
    Unit y = make_unit(&b, 2, &b, 8 << 20, 1 << 20, 12 << 20, 1 << 20);
    r.submit(x.u);
    r.submit(y.u);
    CHECK(eventually([&] { return src.in_flight.load() > 0; }));
    std::shared_ptr<ReadAheadUnit> keep_x = x.u;
    x.u.reset();
    x.dest.reset();
    r.forget(&a);
    /* Afterwards no reader holds a's unit: the last reference is the test's. */
    CHECK(keep_x->cancelled());
    CHECK_EQ(keep_x.use_count(), 1L);
    CHECK_EQ(keep_x->in(), keep_x->started());
    keep_x.reset();
    CHECK(gone.load());
    r.wait(*y.u);
    CHECK(y.u->whole());
    CHECK(same(y, src));
}

TEST(readahead_demand_reads_that_meet_pieces_are_counted) {
    Source src(16 << 20);
    src.delay_us = 20000;
    Ctx ctx{ &src, 1 };
    PieceReader r;
    REQUIRE(r.start(1, kPiece));
    Unit x = make_unit(&ctx, 1, &ctx, 0, 1 << 20, 2 << 20, 1 << 20);
    r.submit(x.u);
    CHECK(eventually([&] { return src.in_flight.load() == 1; }));
    r.demand_enter();   /* meets the 20 ms piece in flight */
    CHECK(eventually([&] { return src.in_flight.load() == 0; }, 200));
    r.cancel(*x.u);     /* while the demand read holds the reader back: nothing more starts */
    r.wait(*x.u);
    r.demand_leave();
    const PieceReader::Stats st = r.stats();
    CHECK_EQ(st.demand_reads, (uint64_t)1);
    CHECK_EQ(st.demand_met, (uint64_t)1);
    CHECK(st.demand_wait_ns > 1000000);     /* the rest of the piece: well over 1 ms */
    CHECK(st.demand_wait_ns < 200000000);
    /* A demand read with nothing in flight meets nothing. */
    r.demand_enter();
    r.demand_leave();
    CHECK_EQ(r.stats().demand_met, (uint64_t)1);
}

/* A reader that never started counts demand reads as plain atomics, as the old hx_demand_reads. */
TEST(readahead_demand_count_without_readers) {
    PieceReader r;
    r.demand_enter();
    r.demand_enter();
    CHECK_EQ(r.demand_count(), 2);
    r.demand_leave();
    r.demand_leave();
    CHECK_EQ(r.demand_count(), 0);
    CHECK(!r.started());
    CHECK_EQ(r.stats().demand_reads, (uint64_t)0);
    CHECK(!r.start(0, kPiece));      /* refused: no readers */
    CHECK(!r.start(4, kPiece));      /* once */
}

RAD_TEST_MAIN()

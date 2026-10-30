/* plane_mirror_test.cpp -- core/place/plane_mirror.h on synthetic checkpoint shards: every expert's w3
 * is indexed from the shards' headers; reads of any extent (unaligned starts and lengths, a tensor
 * ending the shard) return its bytes through the O_DIRECT bounce; an expert in two shards makes the
 * directory unusable; a shard whose times change fails its reads; the fault knob fails exactly one
 * read; turn_off fails every read after it; split() cuts a tail into whole pages, and a tail read
 * in its pieces lands the same bytes as one read. Skipped where the temporary directory refuses
 * O_DIRECT. */
#include "rad_test.h"

#include "place/plane_mirror.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

struct PlaneMirrorTest {
    static PlaneMirror* make(const std::string& dir) {
        auto* m = new PlaneMirror();
        if (!m->open(dir)) { delete m; return nullptr; }
        return m;
    }
    static void fault(PlaneMirror* m, int64_t n) { m->fault_ = n; }
    static void destroy(PlaneMirror* m) { delete m; }
};

}  // namespace rad

using rad::PlaneMirror;
using rad::PlaneMirrorTest;

namespace {

uint8_t byte_of(int layer, int expert, size_t i) { return (uint8_t)(i * 131 + layer * 7 + expert * 13 + (i >> 9)); }

/* A shard of `tensors` (layer, expert, bytes): each expert's w1 (a few bytes, to offset what follows)
 * then its w3, so w3 data starts at unaligned offsets and the last w3 ends the file. */
struct T { int layer, expert; size_t bytes; };
bool write_shard(const std::string& path, const std::vector<T>& ts) {
    std::string h = "{";
    std::vector<uint8_t> data;
    for (const T& t : ts) {
        const std::string pre = "layers." + std::to_string(t.layer) + ".ffn.experts." + std::to_string(t.expert);
        const size_t w1 = 37, a = data.size();
        data.resize(a + w1, 0x5a);
        h += "\"" + pre + ".w1.weight\":{\"dtype\":\"I8\",\"shape\":[1," + std::to_string(w1) +
             "],\"data_offsets\":[" + std::to_string(a) + "," + std::to_string(a + w1) + "]},";
        const size_t b = data.size();
        for (size_t i = 0; i < t.bytes; ++i) data.push_back(byte_of(t.layer, t.expert, i));
        h += "\"" + pre + ".w3.weight\":{\"dtype\":\"I8\",\"shape\":[1," + std::to_string(t.bytes) +
             "],\"data_offsets\":[" + std::to_string(b) + "," + std::to_string(b + t.bytes) + "]},";
    }
    h.back() = '}';
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint64_t n = h.size();
    bool ok = std::fwrite(&n, 8, 1, f) == 1 && std::fwrite(h.data(), 1, h.size(), f) == h.size() &&
              std::fwrite(data.data(), 1, data.size(), f) == data.size();
    return std::fclose(f) == 0 && ok;
}

/* A subdirectory of the binary's temporary directory, one a case. */
std::string subdir(const char* name) {
    const std::string d = radtest::tmpdir() + "/" + name;
    ::mkdir(d.c_str(), 0777);
    return d;
}

bool direct_ok(const std::string& dir) {
    const std::string p = dir + "/probe";
    { FILE* f = std::fopen(p.c_str(), "wb"); if (!f) return false; std::fputs("x", f); std::fclose(f); }
    const int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) ::close(fd);
    std::remove(p.c_str());
    return fd >= 0;
}

bool same(PlaneMirror* m, int layer, int expert, uint64_t from, uint64_t n) {
    PlaneMirror::Ref r;
    if (!m->up_codes(layer, expert, &r)) return false;
    std::vector<uint8_t> got(n);
    if (m->read(r, from, n, got.data()) != 0) return false;
    for (uint64_t i = 0; i < n; ++i) if (got[i] != byte_of(layer, expert, from + i)) return false;
    return true;
}

}  // namespace

TEST(plane_mirror_indexes_and_reads_exact_bytes) {
    const std::string dir = subdir("plane_mirror");
    if (!direct_ok(dir)) { std::printf("  skip: %s refuses O_DIRECT\n", dir.c_str()); return; }
    CHECK(write_shard(dir + "/model-00001-of-00002.safetensors", { { 1, 0, 300000 }, { 1, 1, 70001 } }));
    CHECK(write_shard(dir + "/model-00002-of-00002.safetensors", { { 2, 5, 123457 } }));
    PlaneMirror* m = PlaneMirrorTest::make(dir);
    CHECK(m != nullptr);
    if (!m) return;
    PlaneMirror::Ref r;
    CHECK(m->up_codes(1, 0, &r)); CHECK_EQ(r.bytes, (uint64_t)300000);
    CHECK(!m->up_codes(1, 2, &r));
    CHECK(same(m, 1, 0, 0, 300000));                  /* whole */
    CHECK(same(m, 1, 0, 300000 - 54321, 54321));      /* a tail, unaligned */
    CHECK(same(m, 1, 1, 4095, 4097));                 /* across a 4 KiB boundary */
    CHECK(same(m, 2, 5, 123457 - 1000, 1000));        /* the shard's last bytes */
    CHECK(same(m, 2, 5, 0, 1));
    std::vector<uint8_t> buf(16);
    CHECK(m->up_codes(1, 1, &r));
    CHECK(m->read(r, r.bytes - 8, 16, buf.data()) < 0);   /* past the tensor */
    const auto st = m->stats();
    CHECK_EQ(st.reads, (uint64_t)5);
    PlaneMirrorTest::destroy(m);
}

/* PlaneMirror::split (RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT): the pieces cover the tail in order,
 * every boundary on the 4 KiB grid from the tail's start, at most the pieces asked for (4 at most),
 * none under 64 KiB unless the tail is. */
TEST(plane_mirror_split_cuts_whole_pages) {
    int64_t out[4];
    CHECK_EQ(PlaneMirror::split(0, 3, out), 0);
    const int64_t decode_tail = 2441216;   /* decode share 0.13 of an 18,800,640 B unit, 4 KiB-aligned */
    CHECK_EQ(PlaneMirror::split(decode_tail, 3, out), 3);
    CHECK_EQ(out[0] + out[1] + out[2], decode_tail);
    for (int i = 0; i < 3; ++i) CHECK_EQ(out[i] % 4096, (int64_t)0);
    CHECK(out[0] >= 4096 * 198 && out[2] <= 4096 * 199);
    CHECK_EQ(PlaneMirror::split(decode_tail, 1, out), 1);
    CHECK_EQ(out[0], decode_tail);
    CHECK_EQ(PlaneMirror::split(decode_tail, 9, out), 4);   /* at most 4 */
    CHECK_EQ(PlaneMirror::split(decode_tail, -2, out), 1);
    CHECK_EQ(PlaneMirror::split(160 << 10, 3, out), 2);     /* 53 KiB pieces would be too small */
    CHECK_EQ(out[0] + out[1], (int64_t)(160 << 10));
    CHECK_EQ(PlaneMirror::split(100 << 10, 3, out), 1);
    CHECK_EQ(PlaneMirror::split(300001, 3, out), 3);        /* not whole pages: the last takes the rest */
    CHECK_EQ(out[0] + out[1] + out[2], (int64_t)300001);
    CHECK_EQ(out[0] % 4096, (int64_t)0);
    CHECK_EQ((out[0] + out[1]) % 4096, (int64_t)0);
}

/* The tail read as the split's pieces, each its own read as the mover's readers do, lands the same
 * bytes as one read; copy_ns comes back and the mirror's stats count the copies. */
TEST(plane_mirror_tail_in_pieces_reads_the_same_bytes) {
    const std::string dir = subdir("plane_mirror_pieces");
    if (!direct_ok(dir)) { std::printf("  skip: %s refuses O_DIRECT\n", dir.c_str()); return; }
    CHECK(write_shard(dir + "/model-00001-of-00001.safetensors", { { 4, 2, 1 << 20 }, { 4, 3, 9000 } }));
    PlaneMirror* m = PlaneMirrorTest::make(dir);
    CHECK(m != nullptr);
    if (!m) return;
    PlaneMirror::Ref r;
    REQUIRE(m->up_codes(4, 2, &r));
    const uint64_t tail = 380 << 10, from = r.bytes - tail;
    std::vector<uint8_t> one(tail), pieces(tail, 0);
    uint64_t copy = 0;
    CHECK_EQ(m->read(r, from, tail, one.data(), &copy), 0);
    int64_t piece[4];
    const int np = PlaneMirror::split((int64_t)tail, 3, piece);
    CHECK_EQ(np, 3);
    uint64_t at = 0;
    for (int q = 0; q < np; at += (uint64_t)piece[q], ++q) {
        uint64_t c = 0;
        CHECK_EQ(m->read(r, from + at, (uint64_t)piece[q], pieces.data() + at, &c), 0);
    }
    CHECK_EQ(at, tail);
    CHECK(one == pieces);
    CHECK(same(m, 4, 2, from, tail));
    const auto st = m->stats();
    CHECK_EQ(st.reads, (uint64_t)5);
    CHECK(st.copy_ns <= st.ns);
    PlaneMirrorTest::destroy(m);
}

TEST(plane_mirror_refuses_an_expert_twice) {
    const std::string dir = subdir("plane_mirror_dup");
    if (!direct_ok(dir)) { std::printf("  skip: %s refuses O_DIRECT\n", dir.c_str()); return; }
    CHECK(write_shard(dir + "/model-00001-of-00002.safetensors", { { 1, 0, 5000 } }));
    CHECK(write_shard(dir + "/model-00002-of-00002.safetensors", { { 1, 0, 5000 } }));
    PlaneMirror* m = PlaneMirrorTest::make(dir);
    CHECK(m == nullptr);
    PlaneMirrorTest::destroy(m);
}

TEST(plane_mirror_fails_reads_after_a_change_a_fault_or_turn_off) {
    const std::string dir = subdir("plane_mirror_change");
    if (!direct_ok(dir)) { std::printf("  skip: %s refuses O_DIRECT\n", dir.c_str()); return; }
    const std::string shard = dir + "/model-00001-of-00001.safetensors";
    CHECK(write_shard(shard, { { 3, 7, 20000 }, { 3, 8, 20000 } }));
    PlaneMirror* m = PlaneMirrorTest::make(dir);
    CHECK(m != nullptr);
    if (!m) return;
    CHECK(same(m, 3, 7, 100, 9000));
    PlaneMirrorTest::fault(m, 1);                     /* the second read from now fails */
    CHECK(same(m, 3, 7, 0, 10));
    CHECK(!same(m, 3, 7, 0, 10));
    CHECK(same(m, 3, 8, 0, 10));
    struct timespec ts[2] = { { 0, UTIME_OMIT }, { 12345, 0 } };   /* new mtime (and ctime) */
    CHECK(::utimensat(AT_FDCWD, shard.c_str(), ts, 0) == 0);
    CHECK(!same(m, 3, 8, 0, 10));
    CHECK(m->on());
    m->turn_off(-1);
    CHECK(!m->on());
    PlaneMirrorTest::destroy(m);
}

RAD_TEST_MAIN()

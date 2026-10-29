/* route_trace_test.cpp -- the routing trace's writer and record layout (core/place/route_trace.h,
 * RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE), read back as tools/ds41-host-experts/rtrace.py reads it.
 *
 * A file is its 64-byte header (the same trace start in every file) and then each stream's records
 * in the order they were put, every one 4-byte padded with its size patched in; buffers handed to
 * the background writer from five threads at once land whole and in order; nothing put after
 * close() or into a file that could not be opened is written, and both are counted; and the host's
 * key is the card's score function. */
#include "rad_test.h"

#include "place/route_trace.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using rad::RouteTrace;
using rad::RtStepEntry;
namespace rt = rad::rtrace;

namespace {

std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> b;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return b;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f);
    return b;
}

template <class T> T at(const std::vector<uint8_t>& b, size_t o) {
    T v{};
    if (o + sizeof v <= b.size()) std::memcpy(&v, b.data() + o, sizeof v);
    return v;
}

struct Record {
    uint16_t type = 0, flags = 0;
    size_t off = 0, bytes = 0;   /* where its fields start, and its whole size */
};

/* The header's fields checked, then every record's header; false when a size does not tile the
 * file (a torn or unpadded record). */
bool records(const std::vector<uint8_t>& b, uint32_t stream, uint64_t* t0, std::vector<Record>* out) {
    if (b.size() < rt::kHeaderBytes || std::memcmp(b.data(), rt::kMagic, 8) != 0) return false;
    if (at<uint32_t>(b, 8) != rt::kVersion || at<uint32_t>(b, 12) != stream ||
        at<uint32_t>(b, 16) != (uint32_t)rt::kWorld || at<uint32_t>(b, 20) != (uint32_t)rt::kExperts ||
        at<uint32_t>(b, 24) != (uint32_t)rt::kTopK || at<uint32_t>(b, 28) != (uint32_t)rt::kLocal ||
        at<uint32_t>(b, 32) != (uint32_t)rt::kLayers || at<uint32_t>(b, 36) != rt::kHeaderBytes)
        return false;
    *t0 = at<uint64_t>(b, 40);
    for (size_t o = rt::kHeaderBytes; o < b.size();) {
        Record r;
        r.type = at<uint16_t>(b, o);
        r.flags = at<uint16_t>(b, o + 2);
        r.bytes = at<uint32_t>(b, o + 4);
        r.off = o + 8;
        if (r.bytes < 8 || r.bytes % 4 || o + r.bytes > b.size()) return false;
        out->push_back(r);
        o += r.bytes;
    }
    return true;
}

/* A test record of type 99: u32 stream, u32 seq, then `extra` bytes of a pattern from both. */
std::vector<uint8_t> numbered(uint32_t stream, uint32_t seq, size_t extra) {
    std::vector<uint8_t> b;
    rt::Rec r(b, 99, 7);
    r.put(stream);
    r.put(seq);
    for (size_t i = 0; i < extra; ++i) r.put((uint8_t)(stream * 31 + seq * 7 + i));
    r.finish();
    return b;
}

bool numbered_ok(const std::vector<uint8_t>& b, const Record& r, uint32_t stream, uint32_t seq,
                 size_t extra) {
    if (r.type != 99 || r.flags != 7 || at<uint32_t>(b, r.off) != stream ||
        at<uint32_t>(b, r.off + 4) != seq || r.bytes != (8 + 8 + extra + 3) / 4 * 4)
        return false;
    for (size_t i = 0; i < extra; ++i)
        if (b[r.off + 8 + i] != (uint8_t)(stream * 31 + seq * 7 + i)) return false;
    return true;
}

}  // namespace

TEST(the_trace_is_off_without_its_switch) {
    if (std::getenv("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE")) SKIP("the switch is set here");
    CHECK(RouteTrace::get() == nullptr);
}

TEST(a_record_is_padded_to_four_bytes_with_its_size_patched) {
    std::vector<uint8_t> b(3, 0xee);   /* a record after other bytes keeps its own offsets */
    rt::Rec r(b, rt::kBias, 5);
    r.put(uint8_t{ 1 });
    r.put(uint32_t{ 0x01020304 });
    r.finish();
    REQUIRE_EQ(b.size(), (size_t)(3 + 16));   /* 8 + 5 -> 16 */
    CHECK_EQ(at<uint16_t>(b, 3), (uint16_t)rt::kBias);
    CHECK_EQ(at<uint16_t>(b, 5), (uint16_t)5);
    CHECK_EQ(at<uint32_t>(b, 7), (uint32_t)16);
    CHECK_EQ(b[11], (uint8_t)1);
    CHECK_EQ(at<uint32_t>(b, 12), (uint32_t)0x01020304);
    CHECK_EQ(b[16] | b[17] | b[18], 0);
}

TEST(a_trace_file_is_its_header_then_its_records) {
    const std::string prefix = radtest::tmp_path("rt1");
    {
        RouteTrace t(prefix);
        std::vector<uint8_t> b;
        rt::Rec r(b, rt::kPass, 0);
        r.put(int32_t{ 41 });
        r.put(uint32_t{ 7 });
        r.finish();
        t.put(1, b.data(), b.size());
        RtStepEntry e[2];
        e[0].seq = 12345678901ull; e[0].ctx_len = 63; e[0].n_tokens = 1;
        e[0].flags = rt::kProduces | rt::kDevRow;
        e[1].seq = 9; e[1].ctx_len = 0; e[1].n_tokens = 16; e[1].flags = rt::kPrefill; e[1].n_spec = 3;
        t.step(41, 2, 0, 17, true, e, 2);
        t.end(41, 42, 12345678901ull, 64);
        t.close();
        const RouteTrace::Stats s = t.stats();
        CHECK_EQ(s.records[1], (uint64_t)1);
        CHECK_EQ(s.records[RouteTrace::kSteps], (uint64_t)2);
        CHECK_EQ(s.dropped, (uint64_t)0);
        CHECK(!s.open_failed);
    }
    const std::vector<uint8_t> r1 = slurp(prefix + ".rank1.bin"), st = slurp(prefix + ".steps.bin");
    std::vector<Record> a, b;
    uint64_t t0a = 0, t0b = 1;
    REQUIRE(records(r1, 1, &t0a, &a));
    REQUIRE(records(st, RouteTrace::kSteps, &t0b, &b));
    CHECK_EQ(t0a, t0b);   /* one start for every file */
    CHECK(t0a > 0);
    CHECK(slurp(prefix + ".rank0.bin").empty());   /* a stream with no record has no file */
    REQUIRE_EQ(a.size(), (size_t)1);
    CHECK_EQ(a[0].type, (uint16_t)rt::kPass);
    CHECK_EQ(a[0].bytes, (size_t)16);
    CHECK_EQ(at<int32_t>(r1, a[0].off), 41);
    CHECK_EQ(at<uint32_t>(r1, a[0].off + 4), (uint32_t)7);
    REQUIRE_EQ(b.size(), (size_t)2);
    /* step: i32 step; i16 phase; i16 draft; i32 n_tok; u16 n; u8 ahead; u8 0; u64 t; n x 20 bytes */
    const size_t s0 = b[0].off;
    CHECK_EQ(b[0].type, (uint16_t)rt::kStep);
    CHECK_EQ(b[0].bytes, (size_t)(8 + 24 + 2 * 20));
    CHECK_EQ(at<int32_t>(st, s0), 41);
    CHECK_EQ(at<int16_t>(st, s0 + 4), (int16_t)2);
    CHECK_EQ(at<int16_t>(st, s0 + 6), (int16_t)0);
    CHECK_EQ(at<int32_t>(st, s0 + 8), 17);
    CHECK_EQ(at<uint16_t>(st, s0 + 12), (uint16_t)2);
    CHECK_EQ(st[s0 + 14], (uint8_t)1);
    CHECK(at<uint64_t>(st, s0 + 16) >= t0b);
    const size_t e0 = s0 + 24, e1 = e0 + 20;
    CHECK_EQ(at<uint64_t>(st, e0), (uint64_t)12345678901ull);
    CHECK_EQ(at<int32_t>(st, e0 + 8), 63);
    CHECK_EQ(at<int32_t>(st, e0 + 12), 1);
    CHECK_EQ(st[e0 + 16], (uint8_t)(rt::kProduces | rt::kDevRow));
    CHECK_EQ(at<uint64_t>(st, e1), (uint64_t)9);
    CHECK_EQ(at<int32_t>(st, e1 + 12), 16);
    CHECK_EQ(st[e1 + 16], (uint8_t)rt::kPrefill);
    CHECK_EQ(st[e1 + 17], (uint8_t)3);
    /* end: i32 step; i32 ahead_step; u64 seq; u64 n_out */
    CHECK_EQ(b[1].type, (uint16_t)rt::kEnd);
    CHECK_EQ(b[1].bytes, (size_t)32);
    CHECK_EQ(at<int32_t>(st, b[1].off), 41);
    CHECK_EQ(at<int32_t>(st, b[1].off + 4), 42);
    CHECK_EQ(at<uint64_t>(st, b[1].off + 8), (uint64_t)12345678901ull);
    CHECK_EQ(at<uint64_t>(st, b[1].off + 16), (uint64_t)64);
}

/* Each stream's producer is its own thread, as each rank's issuing thread and the step loop are;
 * rank 0's records are its decode layer records' size (1660 bytes), so its buffer goes to the
 * writer many times over. */
TEST(full_buffers_reach_their_files_whole_and_in_order) {
    const std::string prefix = radtest::tmp_path("rt2");
    const uint32_t n = 1500;
    const size_t extra[RouteTrace::kStreams] = { 1660 - 16, 52, 61, 3, 40 };
    {
        RouteTrace t(prefix);
        std::vector<std::thread> th;
        for (uint32_t s = 0; s < (uint32_t)RouteTrace::kStreams; ++s)
            th.emplace_back([&t, s, n, &extra] {
                for (uint32_t i = 0; i < n; ++i) {
                    const std::vector<uint8_t> b = numbered(s, i, extra[s]);
                    t.put((int)s, b.data(), b.size());
                }
            });
        for (std::thread& x : th) x.join();
        t.close();
        const RouteTrace::Stats st = t.stats();
        CHECK_EQ(st.dropped, (uint64_t)0);
        CHECK(st.writes > (uint64_t)RouteTrace::kStreams);   /* rank 0's alone: ~2.5 MB / 256 KiB */
        for (int s = 0; s < RouteTrace::kStreams; ++s) CHECK_EQ(st.records[s], (uint64_t)n);
        CHECK(t.report().find("records rank0..3 1500/1500/1500/1500, steps 1500") != std::string::npos);
    }
    for (uint32_t s = 0; s < (uint32_t)RouteTrace::kStreams; ++s) {
        const std::string path = prefix + (s == (uint32_t)RouteTrace::kSteps
                                               ? std::string(".steps.bin")
                                               : ".rank" + std::to_string(s) + ".bin");
        const std::vector<uint8_t> b = slurp(path);
        std::vector<Record> rs;
        uint64_t t0 = 0;
        REQUIRE(records(b, s, &t0, &rs));
        REQUIRE_EQ(rs.size(), (size_t)n);
        bool ok = true;
        for (uint32_t i = 0; i < n && ok; ++i) ok = numbered_ok(b, rs[i], s, i, extra[s]);
        CHECK(ok);
    }
}

TEST(records_after_close_are_dropped_and_counted) {
    const std::string prefix = radtest::tmp_path("rt3");
    RouteTrace t(prefix);
    const std::vector<uint8_t> a = numbered(0, 0, 4), b = numbered(0, 1, 4);
    t.put(0, a.data(), a.size());
    t.close();
    t.put(0, b.data(), b.size());
    t.step(1, 0, 0, 1, false, nullptr, 0);
    t.close();   /* idempotent */
    const RouteTrace::Stats s = t.stats();
    CHECK_EQ(s.records[0], (uint64_t)1);
    CHECK_EQ(s.dropped, (uint64_t)2);
    const std::vector<uint8_t> f = slurp(prefix + ".rank0.bin");
    std::vector<Record> rs;
    uint64_t t0 = 0;
    REQUIRE(records(f, 0, &t0, &rs));
    REQUIRE_EQ(rs.size(), (size_t)1);
    CHECK(numbered_ok(f, rs[0], 0, 0, 4));
    CHECK(slurp(prefix + ".steps.bin").empty());
}

TEST(a_file_that_cannot_be_opened_drops_its_records) {
    RouteTrace t(radtest::tmp_path("no-such-dir") + "/rt");
    const std::vector<uint8_t> a = numbered(2, 0, 8);
    t.put(2, a.data(), a.size());
    t.put(2, a.data(), a.size());
    t.close();
    const RouteTrace::Stats s = t.stats();
    CHECK(s.open_failed);
    CHECK_EQ(s.dropped, (uint64_t)2);
    CHECK_EQ(s.records[2], (uint64_t)0);
    CHECK(t.report().find("could not be opened") != std::string::npos);
}

/* Every record kind the file stager and the step loop write, through the encoders they use, with
 * values tools/ds41-host-experts/rtrace_test.py's CppTrace case reads back (RTRACE_DUMP=<dir>
 * writes the trace to <dir>/rt and keeps it, for `RTRACE_CPP_PREFIX=<dir>/rt rtrace_test.py`).
 * Here the layer record's fields are checked at their documented offsets. */
TEST(the_encoders_write_the_documented_layout) {
    const char* dump = std::getenv("RTRACE_DUMP");
    const std::string prefix = dump && *dump ? std::string(dump) + "/rt" : radtest::tmp_path("rt5");
    {
        RouteTrace t(prefix);
        std::vector<uint8_t> b;
        std::array<uint8_t, rt::kLocal> tiers[rt::kLayers];
        for (auto& x : tiers) x.fill(3);
        tiers[12][40] = 0;
        rt::encode_tiers(b, tiers);
        rt::PassRec p;
        p.step = 41; p.pass = 7; p.phase = 2; p.n_tok = 1; p.n_seq = 1; p.n_tok_decode = 1;
        p.max_ctx_len = 63; p.t_ns = 123456789;
        rt::encode_pass(b, p);
        const int32_t ids[6] = { 3, 140, 40, 250, 99, 383 };
        const float w[6] = { 0.5f, 0.2f, 0.1f, 0.1f, 0.05f, 0.05f };
        const float keys[6] = { 3.0f, 2.9f, 2.8f, 2.7f, 2.6f, 2.5f };
        std::vector<float> logits(rt::kExperts, -1.0f), bias(rt::kExperts, 0.0f);
        logits[383] = 3.25f;
        bias[7] = -0.125f;
        const rt::Pick picks[2] = { { 3, 1, rt::kMiss }, { 40, 1, rt::kOnCard } };
        rt::LayerRec L;
        L.pass = 7; L.step = 41; L.layer = 12; L.rows = 1;
        L.flags = rt::kDecided | rt::kRoute | rt::kLogits | rt::kRamMask;
        L.us = 900; L.ram[0] = 1u << 7; L.ram[2] = 1u << 6;
        L.picks = picks; L.n_picks = 2; L.ids = ids; L.weights = w; L.keys = keys;
        L.logits = logits.data();
        const size_t at_layer = b.size();
        rt::encode_layer(b, L);
        /* 8 + 32 + 2 picks x 4 + 60 (one row's route) + 1536 (its logits) */
        REQUIRE_EQ(at<uint32_t>(b, at_layer + 4), (uint32_t)(8 + 32 + 8 + 60 + 4 * rt::kExperts));
        CHECK_EQ(at<uint32_t>(b, at_layer + 8), (uint32_t)7);          /* pass */
        CHECK_EQ(at<int32_t>(b, at_layer + 12), 41);                   /* step */
        CHECK_EQ(at<uint16_t>(b, at_layer + 16), (uint16_t)12);        /* layer */
        CHECK_EQ(at<uint16_t>(b, at_layer + 18), (uint16_t)1);         /* rows */
        CHECK_EQ(at<uint16_t>(b, at_layer + 20), (uint16_t)2);         /* n_sel */
        CHECK_EQ(b[at_layer + 22], (uint8_t)15);                       /* flags */
        CHECK_EQ(at<uint32_t>(b, at_layer + 24), (uint32_t)900);       /* us */
        CHECK_EQ(at<uint32_t>(b, at_layer + 28), 1u << 7);             /* ram[0] */
        CHECK_EQ(at<uint32_t>(b, at_layer + 36), 1u << 6);             /* ram[2] */
        CHECK_EQ(at<uint16_t>(b, at_layer + 40), (uint16_t)3);         /* pick 0 */
        CHECK_EQ(b[at_layer + 43], (uint8_t)rt::kMiss);
        CHECK_EQ(at<uint16_t>(b, at_layer + 48 + 10), (uint16_t)383);  /* id 5 */
        CHECK_EQ(at<float>(b, at_layer + 48 + 12), 0.5f);              /* weight 0 */
        CHECK_EQ(at<float>(b, at_layer + 48 + 36 + 20), 2.5f);         /* key 5 */
        CHECK_EQ(at<float>(b, at_layer + 108 + 4 * 383), 3.25f);       /* logit 383 */
        rt::encode_bias(b, 12, bias.data());
        t.put(0, b.data(), b.size());
        b.clear();
        rt::encode_tiers(b, tiers);
        rt::encode_pass(b, p);
        const rt::Pick mine[1] = { { 140, 1, rt::kRamHit } };
        rt::LayerRec L1 = L;
        L1.flags = rt::kDecided | rt::kRamMask;
        L1.picks = mine;
        L1.n_picks = 1;
        rt::encode_layer(b, L1);
        CHECK_EQ(at<uint32_t>(b, b.size() - (8 + 32 + 4) + 4), (uint32_t)(8 + 32 + 4));   /* no route */
        t.put(1, b.data(), b.size());
        RtStepEntry e;
        e.seq = 77; e.ctx_len = 63; e.n_tokens = 1; e.flags = rt::kProduces | rt::kDevRow;
        t.step(41, 2, 0, 1, true, &e, 1);
        t.end(41, 42, 77, 64);
        t.close();
        CHECK_EQ(t.stats().dropped, (uint64_t)0);
    }
    const std::vector<uint8_t> f0 = slurp(prefix + ".rank0.bin");
    std::vector<Record> rs;
    uint64_t t0 = 0;
    REQUIRE(records(f0, 0, &t0, &rs));
    REQUIRE_EQ(rs.size(), (size_t)4);
    CHECK_EQ(rs[0].type, (uint16_t)rt::kTiers);
    CHECK_EQ(rs[0].bytes, (size_t)(8 + 4 + rt::kLayers * (4 + rt::kLocal)));
    CHECK_EQ(f0[rs[0].off + 4 + 12 * (4 + rt::kLocal) + 4 + 40], (uint8_t)0);   /* layer 12, local 40 */
    CHECK_EQ(rs[1].type, (uint16_t)rt::kPass);
    CHECK_EQ(rs[1].bytes, (size_t)52);
    CHECK_EQ(at<uint64_t>(f0, rs[1].off + 36), (uint64_t)123456789);
    CHECK_EQ(rs[2].type, (uint16_t)rt::kLayer);
    CHECK_EQ(rs[3].type, (uint16_t)rt::kBias);
    CHECK_EQ(at<float>(f0, rs[3].off + 4 + 4 * 7), -0.125f);
}

/* sqrt(softplus(x)) in f32, as router_scored_topk scores (x > 20: sqrt(x), where softplus is x to
 * f32 precision); within a few f32 ulps of the exact value everywhere else. */
TEST(the_host_key_is_the_cards_score_function) {
    for (float x : { 20.5f, 33.0f, 1000.0f }) CHECK_EQ(rt::sqrt_softplus(x), std::sqrt(x));
    double worst = 0;
    for (int i = -4000; i <= 2000; ++i) {
        const float x = (float)i / 100.0f;
        const double ref = std::sqrt(std::log1p(std::exp((double)x)));
        const double got = rt::sqrt_softplus(x);
        worst = std::max(worst, std::fabs(got - ref) / ref);
    }
    CHECK(worst < 4e-7);
}

RAD_TEST_MAIN()

/* route_trace_stager_test.cpp -- the file stager's part of the routing trace (FileStager::rt_*,
 * RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE) on the host backend, through the stager's test friend.
 *
 * No fixture here builds a DeepSeek-V4.1 EP4 program, so start_ep_routed itself runs only on the
 * cards; these cases drive the pieces it calls with the state it leaves: a router select op's
 * operands captured (rt_capture) and copied behind the counts on a stream (rt_queue) -- this pass's
 * only, as many rows as the pass; what each selected file-tier unit's CPU-path slot says the rank
 * did with it (rt_choice: a lease is a RAM hit, -2 a miss read whole, read ahead whole or in part,
 * -1 the card's copy path); and the layer record and the bias record (rt_layer), read back from the
 * file: the picks this rank owns in expert order with their status (the card's for a VRAM
 * resident), the router's ids and weights as copied, keys from the logits and bias, the logits
 * themselves, and the bias once. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/route_trace.h"
#include "place/stager.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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

/* Every record after the header: (type, offset of its fields, bytes). */
std::vector<std::array<size_t, 3>> records(const std::vector<uint8_t>& b) {
    std::vector<std::array<size_t, 3>> out;
    for (size_t o = rt::kHeaderBytes; o + 8 <= b.size();) {
        const size_t n = at<uint32_t>(b, o + 4);
        if (n < 8 || o + n > b.size()) break;
        out.push_back({ at<uint16_t>(b, o), o + 8, n });
        o += n;
    }
    return out;
}

RadTensor tensor(void* p, int dtype, std::initializer_list<int64_t> shape) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t d : shape) t.shape[i++] = d;
    rad_tensor_pack(&t);
    return t;
}

}  // namespace

namespace rad {

/* FileStager's test friend (stager.h). Layer 12 is the stager's routed layer 0; rank 0 owns
 * experts 0..95: 3, 20 and 50 on the file tier (units 0, 1, 2), 40 a VRAM resident. */
struct EpCountsHostProbe {
    Plan plan;
    FileStager fs;
    RouteTrace* trace;
    std::vector<float> logits, bias, w;
    std::vector<int32_t> ids;

    explicit EpCountsHostProbe(RouteTrace* t) : trace(t) {
        plan.rank = 0;
        plan.world_size = 4;
        for (int e : { 3, 20, 50, 40 }) {
            MoveUnit mu;
            mu.layer = 12;
            mu.expert = e;
            mu.access = RAD_ACCESS_CONDITIONAL;
            mu.tier = e == 40 ? Tier::VRAM : Tier::SSD;
            plan.units.push_back(mu);
        }
        fs.plan_ = &plan;
        StagedLayer L;
        L.layer = 12;
        L.units = { 0, 1, 2 };
        fs.layers_ = { L };
        fs.counts_host_ = (int32_t*)rad_dev_alloc(96 * 4, RAD_MEM_HOST_PINNED);
        std::memset(fs.counts_host_, 0, 96 * 4);
        fs.counts_cap_ = 96;
        fs.rt_ = trace;
        fs.rt_rank_ = 0;
        fs.rt_tier_.assign(rt::kLayers, {});
        for (auto& x : fs.rt_tier_) x.fill((uint8_t)Tier::SSD);
        fs.rt_tier_[12][40] = (uint8_t)Tier::VRAM;
        fs.rt_unit_.assign(1, {});
        fs.rt_unit_[0].fill(-1);
        fs.rt_unit_[0][3] = 0;
        fs.rt_unit_[0][20] = 1;
        fs.rt_unit_[0][50] = 2;
        fs.rt_file_units_.assign(1, {});   /* no RAM snapshot: there is no mover here */
        fs.rt_cap_.assign(1, FileStager::RtCap{});
        fs.rt_bias_state_.assign(1, 0);
        fs.rt_ids_ = (int32_t*)rad_dev_alloc(16 * 6 * 4, RAD_MEM_HOST_PINNED);
        fs.rt_w_ = (float*)rad_dev_alloc(16 * 6 * 4, RAD_MEM_HOST_PINNED);
        fs.rt_logits_ = (float*)rad_dev_alloc(384 * 4, RAD_MEM_HOST_PINNED);
        fs.rt_bias_ = (float*)rad_dev_alloc(384 * 4, RAD_MEM_HOST_PINNED);
        if (rad_event_create(&fs.rt_ev_) < 0) fs.rt_ev_ = nullptr;
        logits.resize(384);
        bias.resize(384);
        for (int e = 0; e < 384; ++e) {
            logits[(size_t)e] = std::sin(0.37f * (float)e) * 3.0f;
            bias[(size_t)e] = 0.01f * (float)(e % 7) - 0.03f;
        }
        ids = { 3, 140, 20, 40, 300, 50 };
        w = { 0.40f, 0.20f, 0.15f, 0.10f, 0.10f, 0.05f };
    }
    void pass(uint32_t p, int64_t tok) {
        fs.trace_pass_ = p;
        fs.pass_tok_ = tok;
        fs.rt_step_ = 40 + (int32_t)p;
        fs.rt_t0_ns_ = 0;
        fs.rt_q_li_ = -1;
    }
    /* The select op's operands at its issue: logits, bias, ids, weights. */
    void capture(int64_t rows) {
        RadTensor t[4] = { tensor(logits.data(), RAD_F32, { rows, 384 }),
                           tensor(bias.data(), RAD_F32, { 384 }),
                           tensor(ids.data(), RAD_I32, { rows, 6 }),
                           tensor(w.data(), RAD_F32, { rows, 6 }) };
        fs.rt_capture(0, t, 4);
    }

    void copies_behind_the_counts() {
        RadStream s = nullptr;
        REQUIRE(rad_stream_create(&s, 0) >= 0);
        pass(7, 1);
        capture(1);
        REQUIRE_EQ(fs.rt_cap_[0].pass, (uint32_t)7);
        REQUIRE(fs.rt_ev_ != nullptr);
        CHECK_OK(fs.rt_queue(0, s));
        CHECK(fs.rt_ev_rec_);   /* recorded behind the copies; rt_layer waits for it */
        CHECK_OK(rad_event_sync(fs.rt_ev_));
        CHECK_EQ(fs.rt_q_li_, 0);
        CHECK(fs.rt_q_logits_);
        CHECK_EQ(fs.rt_bias_state_[0], (uint8_t)1);
        CHECK(std::memcmp(fs.rt_ids_, ids.data(), 24) == 0);
        CHECK(std::memcmp(fs.rt_w_, w.data(), 24) == 0);
        CHECK(std::memcmp(fs.rt_logits_, logits.data(), 384 * 4) == 0);
        CHECK(std::memcmp(fs.rt_bias_, bias.data(), 384 * 4) == 0);
        /* The next pass has not shown its select op: nothing is queued, and it is counted. */
        pass(8, 1);
        CHECK_OK(fs.rt_queue(0, s));
        CHECK_EQ(fs.rt_q_li_, -1);
        CHECK_EQ(fs.rt_stats_.uncaptured, (uint64_t)1);
        /* A two-token pass: ids and weights for both rows, no logits, the bias not again. */
        pass(9, 2);
        ids.resize(12, 7);
        w.resize(12, 0.5f);
        logits.resize(768, 1.0f);
        capture(2);
        CHECK_OK(fs.rt_queue(0, s));
        CHECK_OK(rad_stream_sync(s));
        CHECK_EQ(fs.rt_q_li_, 0);
        CHECK(!fs.rt_q_logits_);
        CHECK(std::memcmp(fs.rt_ids_, ids.data(), 48) == 0);
        CHECK(std::memcmp(fs.rt_w_, w.data(), 48) == 0);
        CHECK_EQ(fs.rt_bias_state_[0], (uint8_t)1);
        /* Operands of another shape (a bias of 383) are not captured. */
        RadTensor bad[4] = { tensor(logits.data(), RAD_F32, { 1, 384 }), tensor(bias.data(), RAD_F32, { 383 }),
                             tensor(ids.data(), RAD_I32, { 1, 6 }), tensor(w.data(), RAD_F32, { 1, 6 }) };
        pass(10, 1);
        fs.rt_capture(0, bad, 4);
        CHECK_EQ(fs.rt_cap_[0].pass, (uint32_t)0);
        rad_stream_destroy(s);
    }

    void slots_say_what_the_rank_did() {
        pass(7, 1);
        /* Units 0, 1, 2 selected: a lease, a miss, the copy path. */
        fs.rt_units_ = { 0, 1, 2 };
        fs.hx_slot_ = { 0, -2, -1 };
        fs.hx_miss_ready_ = { 0 };
        fs.hx_miss_ra_.assign(1, nullptr);
        fs.rt_choice(0, true);
        REQUIRE_EQ(fs.rt_status_.size(), (size_t)3);
        CHECK_EQ(fs.rt_status_[0], (uint8_t)rt::kRamHit);
        CHECK_EQ(fs.rt_status_[1], (uint8_t)rt::kMiss);
        CHECK_EQ(fs.rt_status_[2], (uint8_t)rt::kCopyPath);
        CHECK(!fs.rt_ram_ok_);   /* no file-tier units listed: no snapshot */
        /* Three misses: read ahead whole, read ahead in part (the readers finish it), read now. */
        fs.hx_slot_ = { -2, -2, -2 };
        fs.hx_miss_ready_ = { 1, 0, 0 };
        fs.hx_miss_ra_.assign(3, nullptr);
        fs.hx_miss_ra_[1] = std::make_shared<ReadAheadUnit>(nullptr, 0, nullptr, 0, 512 << 10,
                                                            nullptr, nullptr, nullptr);
        fs.rt_choice(0, true);
        CHECK_EQ(fs.rt_status_[0], (uint8_t)rt::kAheadWhole);
        CHECK_EQ(fs.rt_status_[1], (uint8_t)rt::kAheadPart);
        CHECK_EQ(fs.rt_status_[2], (uint8_t)rt::kMiss);
        /* No CPU choice: the card's copy path takes every one. */
        fs.rt_choice(0, false);
        for (uint8_t st : fs.rt_status_) CHECK_EQ(st, (uint8_t)rt::kCopyPath);
        fs.hx_miss_ra_.clear();
        fs.hx_miss_ready_.clear();
        fs.hx_slot_.clear();
    }

    /* Two passes' layer records (and the bias once). */
    void two_layer_records() {
        RadStream s = nullptr;
        REQUIRE(rad_stream_create(&s, 0) >= 0);
        pass(7, 1);
        capture(1);
        CHECK_OK(fs.rt_queue(0, s));
        CHECK_OK(rad_stream_sync(s));
        /* Rank 0's picks: 3 (a lease), 20 (a miss), 40 (on the card), 50 (the copy path). */
        for (int e : { 3, 20, 40, 50 }) fs.counts_host_[e] = 1;
        fs.rt_units_ = { 0, 1, 2 };
        fs.hx_slot_ = { 0, -2, -1 };
        fs.hx_miss_ready_ = { 0 };
        fs.hx_miss_ra_.assign(1, nullptr);
        fs.rt_choice(0, true);
        fs.rt_layer(0);
        CHECK_EQ(fs.rt_bias_state_[0], (uint8_t)2);
        CHECK_EQ(fs.rt_q_li_, -1);
        /* The next pass's layer: a record again, the bias not again. */
        pass(8, 1);
        capture(1);
        CHECK_OK(fs.rt_queue(0, s));
        CHECK_OK(rad_stream_sync(s));
        fs.rt_choice(0, true);
        fs.rt_layer(0);
        const auto& st = fs.rt_stats_;
        CHECK_EQ(st.layers, (uint64_t)2);
        CHECK_EQ(st.d_layers, (uint64_t)2);
        CHECK_EQ(st.routes, (uint64_t)2);
        CHECK_EQ(st.logits, (uint64_t)2);
        CHECK_EQ(st.biases, (uint64_t)1);
        CHECK_EQ(st.picks[rt::kRamHit], (uint64_t)2);
        CHECK_EQ(st.picks[rt::kMiss], (uint64_t)2);
        CHECK_EQ(st.picks[rt::kOnCard], (uint64_t)2);
        CHECK_EQ(st.picks[rt::kCopyPath], (uint64_t)2);
        fs.hx_miss_ra_.clear();
        fs.hx_miss_ready_.clear();
        fs.hx_slot_.clear();
        rad_stream_destroy(s);
    }
};

}  // namespace rad

using rad::EpCountsHostProbe;
using rad::RouteTrace;

TEST(the_select_ops_operands_are_copied_behind_the_counts_for_this_pass_only) {
    RouteTrace trace(radtest::tmp_path("rts1"));
    {
        EpCountsHostProbe p(&trace);
        p.copies_behind_the_counts();
    }
    trace.close();
}

TEST(each_selected_units_slot_says_what_the_rank_did_with_it) {
    RouteTrace trace(radtest::tmp_path("rts2"));
    {
        EpCountsHostProbe p(&trace);
        p.slots_say_what_the_rank_did();
    }
    trace.close();
}

TEST(the_layer_record_has_the_ranks_picks_and_rank_0s_route) {
    const std::string prefix = radtest::tmp_path("rts3");
    {
        RouteTrace trace(prefix);
        {
            EpCountsHostProbe p(&trace);
            p.two_layer_records();
        }
        trace.close();
    }
    const std::vector<uint8_t> f = slurp(prefix + ".rank0.bin");
    const auto rs = records(f);
    REQUIRE_EQ(rs.size(), (size_t)3);   /* layer, bias, layer */
    CHECK_EQ(rs[0][0], (size_t)rt::kLayer);
    CHECK_EQ(rs[1][0], (size_t)rt::kBias);
    CHECK_EQ(rs[2][0], (size_t)rt::kLayer);
    const size_t o = rs[0][1];
    CHECK_EQ(rs[0][2], (size_t)(8 + 32 + 4 * 4 + 60 + 1536));
    CHECK_EQ(at<uint32_t>(f, o), (uint32_t)7);         /* pass */
    CHECK_EQ(at<int32_t>(f, o + 4), 47);               /* step */
    CHECK_EQ(at<uint16_t>(f, o + 8), (uint16_t)12);    /* the model layer */
    CHECK_EQ(at<uint16_t>(f, o + 10), (uint16_t)1);
    CHECK_EQ(at<uint16_t>(f, o + 12), (uint16_t)4);    /* picks this rank owns */
    CHECK_EQ(f[o + 14], (uint8_t)(rt::kDecided | rt::kRoute | rt::kLogits));
    const uint16_t want_e[4] = { 3, 20, 40, 50 };
    const uint8_t want_s[4] = { rt::kRamHit, rt::kMiss, rt::kOnCard, rt::kCopyPath };
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(at<uint16_t>(f, o + 32 + 4 * i), want_e[i]);
        CHECK_EQ(f[o + 32 + 4 * i + 2], (uint8_t)1);
        CHECK_EQ(f[o + 32 + 4 * i + 3], want_s[i]);
    }
    const size_t r = o + 32 + 16;
    const int32_t ids[6] = { 3, 140, 20, 40, 300, 50 };
    for (int k = 0; k < 6; ++k) {
        CHECK_EQ(at<uint16_t>(f, r + 2 * k), (uint16_t)ids[k]);
        const float lg = std::sin(0.37f * (float)ids[k]) * 3.0f;
        const float key = rt::sqrt_softplus(lg) + (0.01f * (float)(ids[k] % 7) - 0.03f);
        CHECK_EQ(at<float>(f, r + 36 + 4 * k), key);
    }
    CHECK_EQ(at<float>(f, r + 12), 0.40f);
    CHECK_EQ(at<float>(f, r + 60 + 4 * 300), std::sin(0.37f * 300.0f) * 3.0f);
    CHECK_EQ(at<uint16_t>(f, rs[1][1]), (uint16_t)12);
    CHECK_EQ(at<float>(f, rs[1][1] + 4 + 4 * 6), 0.01f * 6.0f - 0.03f);
}

RAD_TEST_MAIN()

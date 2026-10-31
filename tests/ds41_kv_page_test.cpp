/* Physical packed pages, independently of device quantization: real manager/prefix/COW/binding
 * paths over malloc pools. Numerical producer closure is the codec lane's separate card gate. */
#include "rad_test.h"
#include "mem/kv.h"
#include "mem/kv_storage_decode.h"
#include "mem/pools.h"
#include "mem/prefix.h"
#include "kvstream/plan.h"
#include "runtime/kvbind.h"
#include "runtime/tierexec.h"
#include "device/device.h"
#include "rad_internal.h"

#include <cstring>
#include <numeric>

using namespace rad;
namespace codec = rad_ds41_kv_codec;

namespace {
constexpr int32_t Latent = 1, Index = 2, Tail = 3, Window = 4;

Config config(int64_t mib = 1) {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = mib;
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 2;
    c.max_tok = 16;
    c.max_ctx = 4096;
    c.prefix_cache = true;
    c.checkpoint_interval = 4;
    c.checkpoint_slots = 2;
    return c;
}

KVGroupInfo packed(uint32_t format, int64_t ratio = 2) {
    KVGroupInfo g;
    g.name = format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16 ? "latent" : "index";
    g.decl.kind = RAD_KV_FULL;
    g.decl.dtype = RAD_BF16;
    g.decl.n_head_kv = 1;
    const int64_t width = format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16 ? 512 : 128;
    g.decl.head_dim = width / (2 * ratio);
    g.decl.storage_format = format;
    g.decl.storage_block_bytes = codec::row_bytes(format, width);
    g.block_size = ratio;
    g.layers = { 2, 8 };             // nonconsecutive absolute layers; physical stripes are dense
    g.replicated = true;
    return g;
}

std::vector<KVGroupInfo> groups(bool packed_storage = true, bool state_window = false) {
    std::vector<KVGroupInfo> gs(1);
    gs.push_back(packed(RAD_KV_STORAGE_DS41_FP4_E4M3_G16));
    gs.push_back(packed(RAD_KV_STORAGE_DS41_FP4_POW2_G32));
    if (!packed_storage)
        for (size_t i = 1; i < gs.size(); ++i) {
            gs[i].decl.storage_format = RAD_KV_STORAGE_PLAIN;
            gs[i].decl.storage_block_bytes = 0;
        }
    if (state_window) {
        KVGroupInfo tail;
        tail.name = "raw_f32_tail";
        tail.decl.kind = RAD_KV_LINEAR;
        tail.decl.dtype = RAD_F32;
        tail.decl.n_head_kv = 1;
        tail.decl.state_dim[0] = 2;
        tail.decl.state_dim[1] = 16;
        tail.layers = { 2, 8 };
        gs.push_back(tail);
        KVGroupInfo win;
        win.name = "unchanged_bf16_window";
        win.decl.kind = RAD_KV_WINDOW;
        win.decl.dtype = RAD_BF16;
        win.decl.n_head_kv = 1;
        win.decl.head_dim = 8;
        win.decl.window = 8;
        win.block_size = 4;
        win.layers = { 2, 8 };
        gs.push_back(win);
    }
    return gs;
}

struct Fixture {
    Pools pools;
    KVManager kv;
    PrefixCache pc;
    Config cfg = config();
    int build(const std::vector<KVGroupInfo>& gs, bool prefix = false) {
        RAD_TRY(pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
        RAD_TRY(kv.configure(gs, cfg, pools.kv()));
        return prefix ? pc.configure(cfg, &kv, cfg.checkpoint_interval) : RAD_OK;
    }
};

uint8_t* page(const KVGroupPlan& p, int64_t layer, int32_t block) {
    return static_cast<uint8_t*>(p.base) + layer * p.layer_stride() +
           block * p.layer_bytes_per_block();
}

/* Self-contained escapes deliberately include signed zero and distinct NaN payloads. These are
 * literal captured outputs, so page copying cannot recreate/quiet/reorder their floating bits. */
void fill_pair_page(uint32_t format, uint8_t* dst, int seed) {
    const int64_t width = format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16 ? 512 : 128;
    std::memset(dst, 0, (size_t)codec::row_bytes(format, width));
    const int G = codec::group_size(format);
    for (int64_t g = 0; g < width / G; ++g) {
        uint8_t* payload = dst + codec::payload_offset(format, g);
        codec::store_u16_le(payload, seed & 1 ? 0x8000 : 0);
        codec::store_u16_le(payload + 2, (uint16_t)(0x7FC1 + ((seed + g) & 31)));
        for (int i = 0; i < G; ++i)
            if ((i + seed + g) % 3) payload[4 + i / 8] |= (uint8_t)(1u << (i & 7));
    }
}
}  // namespace

TEST(ds41_packed_page_geometry_refuses_implicit_or_wrong_layouts) {
    for (uint32_t format : { uint32_t(RAD_KV_STORAGE_DS41_FP4_E4M3_G16),
                             uint32_t(RAD_KV_STORAGE_DS41_FP4_POW2_G32) }) {
        for (int64_t ratio : { 1, 2 }) {
            auto g = packed(format, ratio);
            CHECK(kv_storage_decl_error(g.decl) == nullptr);
            CHECK(kv_storage_page_error(g.decl, ratio) == nullptr);
            ++g.decl.head_dim;
            CHECK(kv_storage_page_error(g.decl, ratio) != nullptr);
        }
        const auto good = packed(format);
        for (int invalid = 0; invalid < 6; ++invalid) {
            auto bad = good;
            if (invalid == 0) bad.decl.storage_format = 99;
            if (invalid == 1) ++bad.decl.storage_block_bytes;
            if (invalid == 2) bad.decl.kind = RAD_KV_WINDOW;
            if (invalid == 3) bad.decl.dtype = RAD_F32;
            if (invalid == 4) bad.decl.n_head_kv = 2;
            if (invalid == 5) bad.decl.head_dim = 0;
            KVManager kv;
            CHECK_EQ(kv.plan_groups({ KVGroupInfo{}, bad }, config()), RAD_E_INVAL);
        }
        auto wrong_size = good;
        wrong_size.bytes_per_block = 2 * good.decl.storage_block_bytes + 4;
        KVManager kv;
        CHECK_EQ(kv.plan_groups({ KVGroupInfo{}, wrong_size }, config()), RAD_E_SHAPE);
        auto bad_width = good;
        ++bad_width.decl.head_dim;
        CHECK_EQ(kv.plan_groups({ KVGroupInfo{}, bad_width }, config()), RAD_E_SHAPE);
    }
    auto plain = packed(RAD_KV_STORAGE_DS41_FP4_POW2_G32);
    plain.decl.storage_format = RAD_KV_STORAGE_PLAIN;
    CHECK(kv_storage_decl_error(plain.decl) != nullptr); // no hidden pitch with plain dtype
    plain.decl.storage_block_bytes = 0;
    CHECK(kv_storage_decl_error(plain.decl) == nullptr);
}

TEST(ds41_packed_pages_use_physical_bytes_and_honest_u8_bindings_at_small_budgets) {
    if (!device_is_host()) SKIP("pool byte/accounting fixture requires the host device backend");
    const auto gs = groups();
    Fixture small, large, plain;
    large.cfg.vram_kv_mib = 2;
    REQUIRE_EQ(small.build(gs), RAD_OK);
    REQUIRE_EQ(large.build(gs), RAD_OK);
    REQUIRE_EQ(plain.build(groups(false)), RAD_OK);
    CHECK(small.kv.paged_tokens(1ll << 20) > plain.kv.paged_tokens(1ll << 20));
    CHECK(large.kv.total_blocks(Latent) > small.kv.total_blocks(Latent));
    for (const Fixture* f : { &small, &large }) {
        Program program;
        program.kv_groups = gs;
        std::vector<KVPoolBinding> binding;
        bind_kv_pools(program, [&](int32_t g) { return f->kv.plan(g); }, &binding);
        for (int32_t g : { Latent, Index }) {
            const KVGroupPlan* p = f->kv.plan(g);
            REQUIRE(p);
            const int64_t frag = g == Latent ? 288 : 68;
            CHECK_EQ(p->layer_bytes_per_block(), frag);
            CHECK_EQ(p->bytes_per_block, 2 * frag);
            CHECK_EQ(p->pool_bytes, p->n_blocks * p->bytes_per_block);
            CHECK_EQ(p->layer_stride(), p->n_blocks * frag);
            CHECK_EQ(binding[g].dtype, uint32_t(RAD_U8));
            CHECK_EQ(binding[g].rank, 2u);
            CHECK_EQ(binding[g].shape[0], p->n_blocks);
            CHECK_EQ(binding[g].shape[1], frag);
            CHECK_EQ(binding[g].shape[2], 0ll);
            CHECK_EQ(binding[g].layer_slot[2], 0);
            CHECK_EQ(binding[g].layer_slot[8], 1);
        }
        CHECK(f->pools.kv().used() <= f->cfg.vram_kv_mib * (1ll << 20));
    }
    Program old;
    old.kv_groups = groups(false);
    std::vector<KVPoolBinding> binding;
    bind_kv_pools(old, [&](int32_t g) { return plain.kv.plan(g); }, &binding);
    CHECK_EQ(binding[Latent].dtype, uint32_t(RAD_BF16));
    CHECK_EQ(binding[Latent].rank, 4u);
    CHECK_EQ(plain.kv.plan(Latent)->layer_bytes_per_block(), 1024ll);
    CHECK_EQ(plain.kv.plan(Index)->layer_bytes_per_block(), 256ll);
}

TEST(ds41_packed_logical_hash_runs_keep_each_ratio_half_and_literal_bit_pattern) {
    for (uint32_t format : { uint32_t(RAD_KV_STORAGE_DS41_FP4_E4M3_G16),
                             uint32_t(RAD_KV_STORAGE_DS41_FP4_POW2_G32) })
        for (int64_t ratio : { 1, 2 }) {
            const auto g = packed(format, ratio);
            const int64_t width = 2 * ratio * g.decl.head_dim;
            const int64_t pitch = g.decl.storage_block_bytes;
            std::vector<uint8_t> physical((size_t)(3 * pitch), 0);
            fill_pair_page(format, physical.data() + pitch, 1);
            fill_pair_page(format, physical.data() + 2 * pitch, 2);
            /* Exercise both descriptor kinds through the actual logical-run adapter. Index254
             * records the original minimum-scale half-product FTZ; it is never source254. */
            const bool index = format == RAD_KV_STORAGE_DS41_FP4_POW2_G32;
            std::memset(physical.data() + 2 * pitch, index ? 0x91 : 0x21,
                        (size_t)(codec::group_size(format) / 2));
            physical[(size_t)(2 * pitch + width / 2)] = index ? 254 : 56;
            std::vector<uint16_t> expected((size_t)(3 * width)), scratch((size_t)width);
            for (int64_t b = 0; b < 3; ++b)
                REQUIRE(codec::decode_row(format, physical.data() + b * pitch, width,
                                          expected.data() + b * width));
            CHECK_EQ(expected[(size_t)(2 * width)], uint16_t(index ? 0 : 0x3F00));
            CHECK_EQ(expected[(size_t)(2 * width + 1)], uint16_t(index ? 0x8000 : 0x3F80));
            for (int64_t b : { 2, 0, 1 })
                for (int64_t pos = 0; pos < ratio; ++pos) {
                    const int64_t bytes = 4 * g.decl.head_dim;
                    const int64_t off = b * width * 2 + pos * bytes;
                    const uint8_t* logical = nullptr;
                    REQUIRE(kv_storage_decode_run(g.decl, ratio, physical.data(),
                        (int64_t)physical.size(), off, bytes, scratch.data(), width, &logical));
                    CHECK_EQ(std::memcmp(logical, reinterpret_cast<const uint8_t*>(expected.data()) +
                                                    off, (size_t)bytes), 0);
                }
            const uint8_t* logical = reinterpret_cast<const uint8_t*>(1);
            CHECK(!kv_storage_decode_run(g.decl, ratio, physical.data(), physical.size(),
                0, width * 2 + 2, scratch.data(), width, &logical));
            CHECK(logical == nullptr);
            CHECK(!kv_storage_decode_run(g.decl, ratio, physical.data(), physical.size(),
                6 * width, 2, scratch.data(), width, &logical));
            CHECK(!kv_storage_decode_run(g.decl, ratio, physical.data(), physical.size(),
                1, 2, scratch.data(), width, &logical));
            physical[(size_t)(width / 2)] = 255; // invalid stored metadata, including index255
            CHECK(!kv_storage_decode_run(g.decl, ratio, physical.data(), physical.size(),
                0, 2, scratch.data(), width, &logical));
            CHECK(logical == nullptr);
        }
}

TEST(ds41_packed_cow_copies_only_self_contained_physical_pages_and_releases_on_cancel) {
    if (!device_is_host()) SKIP("pool byte/COW fixture requires the host device backend");
    Fixture f;
    REQUIRE_EQ(f.build(groups()), RAD_OK);
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK);
    REQUIRE_EQ(f.kv.ensure(1, 3), RAD_OK); // one whole page plus a partial page
    for (int32_t g : { Latent, Index }) {
        const auto* p = f.kv.plan(g);
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b : f.kv.block_table(1, g))
                fill_pair_page(p->storage_format, page(*p, l, b), (int)(b + 7 * l));
    }
    REQUIRE_EQ(f.kv.fork(1, 2), RAD_OK);
    REQUIRE_EQ(f.kv.ensure(2, 4), RAD_OK);
    const auto copies = f.kv.take_pending_copies();
    REQUIRE_EQ(copies.size(), 2u);
    for (const BlockCopy& c : copies) {
        const auto* p = f.kv.plan(c.group);
        REQUIRE(p);
        const int64_t frag = p->layer_bytes_per_block();
        for (int64_t l = 0; l < p->n_layers; ++l) {
            std::vector<uint8_t> before(page(*p, l, c.src_block), page(*p, l, c.src_block) + frag);
            std::memcpy(page(*p, l, c.dst_block), before.data(), (size_t)frag);
            CHECK_EQ(std::memcmp(page(*p, l, c.dst_block), before.data(), (size_t)frag), 0);
            page(*p, l, c.dst_block)[4] ^= 1; // a changed literal selector cannot affect the parent
            CHECK_EQ(std::memcmp(page(*p, l, c.src_block), before.data(), (size_t)frag), 0);
        }
    }
    f.kv.free_sequence(2); // cancellation after COW
    f.kv.free_sequence(1);
    for (int32_t g : { Latent, Index }) CHECK_EQ(f.kv.free_blocks(g), f.kv.total_blocks(g));
}

TEST(ds41_packed_prefix_pages_survive_owner_release_adoption_and_eviction) {
    if (!device_is_host()) SKIP("pool byte/prefix fixture requires the host device backend");
    Fixture f;
    REQUIRE_EQ(f.build(groups(), true), RAD_OK);
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK);
    REQUIRE_EQ(f.kv.ensure(1, 8), RAD_OK);
    std::vector<int32_t> tokens(8);
    std::iota(tokens.begin(), tokens.end(), 100);
    std::vector<std::vector<int32_t>> tables;
    for (int32_t g : f.pc.cached_groups()) {
        tables.push_back(f.kv.block_table(1, g));
        const auto* p = f.kv.plan(g);
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b : tables.back())
                fill_pair_page(p->storage_format, page(*p, l, b), (int)(b + 7 * l));
    }
    REQUIRE_EQ(f.pc.insert(1, tokens, {}, tables, 8), RAD_OK);
    f.kv.free_sequence(1);
    tokens.push_back(999); // leave a suffix to compute; the full cached eight tokens can be reused
    std::vector<std::vector<int32_t>> hit_tables;
    const auto hit = f.pc.lookup(tokens, {}, &hit_tables);
    REQUIRE_EQ(hit.n_attn_tokens, 8);
    REQUIRE_EQ(hit_tables.size(), f.pc.cached_groups().size());
    REQUIRE_EQ(f.kv.add_sequence(2), RAD_OK);
    for (size_t i = 0; i < hit_tables.size(); ++i) {
        const int32_t g = f.pc.cached_groups()[i];
        const auto* p = f.kv.plan(g);
        REQUIRE_EQ(f.kv.adopt(2, g, hit_tables[i], hit.n_attn_tokens), RAD_OK);
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b : hit_tables[i]) {
                std::vector<uint8_t> expected((size_t)p->layer_bytes_per_block());
                fill_pair_page(p->storage_format, expected.data(), (int)(b + 7 * l));
                CHECK_EQ(std::memcmp(page(*p, l, b), expected.data(), expected.size()), 0);
            }
    }
    CHECK_EQ(f.pc.evict_lru(100), 0ll); // live adopted pages remain held
    f.kv.free_sequence(2);
    CHECK_EQ(f.pc.evict_lru(100), 4ll);
    for (int32_t g : { Latent, Index }) CHECK_EQ(f.kv.free_blocks(g), f.kv.total_blocks(g));
}

TEST(ds41_packed_pages_do_not_stream_or_alias_plain_replicated_tier_fragments) {
    if (!device_is_host()) SKIP("pool byte/tier fixture requires the host device backend");
    Fixture f;
    const auto gs = groups();
    REQUIRE_EQ(f.build(gs), RAD_OK);
    Program program;
    program.kv_groups = gs;
    CHECK_EQ(kv_stream_group(gs, &program), -1);
    CHECK_EQ(kv_stream_group(f.kv, &program), -1);
    CHECK(!kvstream_ops(program, &gs[Latent]).streams());
    Fixture capped;
    REQUIRE_EQ(capped.pools.configure(capped.cfg, 0, &rad_malloc_allocator(),
                                      &rad_malloc_allocator()), RAD_OK);
    REQUIRE_EQ(capped.kv.plan_groups(gs, capped.cfg), RAD_OK);
    capped.kv.cap_resident_tokens(Latent, 2);
    CHECK_EQ(capped.kv.carve(capped.pools.kv()), RAD_E_UNSUPPORTED);

    auto packed_index = packed(RAD_KV_STORAGE_DS41_FP4_POW2_G32, 1);
    auto plain = packed_index;
    plain.decl.storage_format = RAD_KV_STORAGE_PLAIN;
    plain.decl.storage_block_bytes = 0;
    plain.decl.head_dim = 17; // also68B physically: equality of bytes alone is insufficient
    Fixture a, b;
    REQUIRE_EQ(a.build({ KVGroupInfo{}, packed_index }), RAD_OK);
    REQUIRE_EQ(b.build({ KVGroupInfo{}, plain }), RAD_OK);
    REQUIRE_EQ(a.kv.plan(1)->bytes_per_block, b.kv.plan(1)->bytes_per_block);
    TierExec::SlotLayout layout;
    CHECK_EQ(TierExec::slot_layout({ &a.kv, &b.kv }, { 1 }, &layout), RAD_E_STATE);
    CHECK_EQ(TierExec::slot_layout({ &a.kv, &a.kv }, { 1 }, &layout), RAD_OK);
    CHECK_EQ(layout.rep_bytes, 136ll);
}

TEST(ds41_packed_full_pages_leave_raw_f32_and_window_snapshot_bytes_unchanged) {
    if (!device_is_host()) SKIP("queued snapshot copies require the host device backend");
    Fixture f, old;
    REQUIRE_EQ(f.build(groups(true, true)), RAD_OK);
    REQUIRE_EQ(old.build(groups(false, true)), RAD_OK);
    const auto* tail = f.kv.plan(Tail);
    const auto* win = f.kv.plan(Window);
    REQUIRE(tail && win);
    CHECK_EQ(f.kv.checkpoint_bytes(), old.kv.checkpoint_bytes());
    CHECK_EQ(f.kv.checkpoint_bytes(), tail->bytes_per_state +
                                        win->window_blocks * win->bytes_per_block);
    HostArena store;
    REQUIRE_EQ(store.open(f.kv.checkpoint_bytes(), 2 * f.kv.checkpoint_bytes()), RAD_OK);
    f.kv.set_checkpoint_store(&store, 0);
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK);
    REQUIRE_EQ(f.kv.add_sequence(2), RAD_OK);
    REQUIRE_EQ(f.kv.ensure(1, 8), RAD_OK);
    std::vector<int32_t> source((size_t)f.kv.n_groups(), -1), target = source;
    source[Tail] = f.kv.state_slot(1, Tail);
    target[Tail] = f.kv.state_slot(2, Tail);
    for (int64_t l = 0; l < tail->n_layers; ++l)
        std::memset(f.kv.state_ptr(Tail, (int32_t)l, source[Tail]), (int)(0x41 + l),
                    (size_t)tail->layer_bytes_per_state());
    std::vector<int32_t> rows;
    REQUIRE(f.kv.window_rows(1, 8, &rows));
    for (int64_t l = 0; l < win->n_layers; ++l)
        for (size_t j = 0; j < rows.size(); ++j)
            if (rows[j] >= 0) std::memset(page(*win, l, rows[j]), (int)(0x70 + l + j),
                                         (size_t)win->layer_bytes_per_block());
    int32_t ck = -1;
    REQUIRE_EQ(f.kv.alloc_checkpoint(&ck), RAD_OK);
    RadStream stream = nullptr;
    REQUIRE_EQ(rad_stream_create(&stream, 0), RAD_OK);
    CHECK_OK(f.kv.checkpoint_save(source.data(), (int)source.size(), ck, stream, &rows));
    CHECK_OK(f.kv.seed_window(2, 8));
    std::vector<int32_t> restored;
    REQUIRE(f.kv.window_rows(2, 8, &restored));
    CHECK_OK(f.kv.checkpoint_restore(target.data(), (int)target.size(), ck, stream, &restored));
    CHECK_OK(rad_stream_sync(stream));
    for (int64_t l = 0; l < tail->n_layers; ++l)
        CHECK_EQ(std::memcmp(f.kv.state_ptr(Tail, (int32_t)l, source[Tail]),
                             f.kv.state_ptr(Tail, (int32_t)l, target[Tail]),
                             (size_t)tail->layer_bytes_per_state()), 0);
    REQUIRE_EQ(rows.size(), restored.size());
    for (int64_t l = 0; l < win->n_layers; ++l)
        for (size_t j = 0; j < rows.size(); ++j)
            if (rows[j] >= 0) {
                REQUIRE(restored[j] >= 0);
                CHECK_EQ(std::memcmp(page(*win, l, rows[j]), page(*win, l, restored[j]),
                                     (size_t)win->layer_bytes_per_block()), 0);
            }
    rad_stream_destroy(stream);
    f.kv.free_checkpoint(ck);
}

RAD_TEST_MAIN()

/* mapped_slab_pregrow_test.cpp -- MappedExpertSlabCache::pregrow (core/place/mapped_expert_slab_cache.h):
 * a pregrown cache pins the slabs inserts would grow, in the same order and sizes, and then admits,
 * evicts and hits exactly as one grown on demand; a second pregrow adds nothing; a host that refuses a
 * slab stops the growth and keeps what was pinned. */
#include "rad_test.h"

#include "place/mapped_expert_slab_cache.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using ds41_mapped::SlabCache;
using S = SlabCache::Status;

namespace {

/* Pinned memory stood in by aligned host memory; `refuse_after` slabs are granted, then none. */
struct Host {
    int granted = 0, refuse_after = 1 << 30;
};
size_t quote(void*, size_t request) noexcept { return request + 4096; }
bool allocate(void* ctx, size_t request, size_t q, SlabCache::Allocation* a) noexcept {
    Host* h = static_cast<Host*>(ctx);
    if (h->granted >= h->refuse_after) return false;
    void* p = std::aligned_alloc(64, (request + 63) / 64 * 64);
    if (!p) return false;
    ++h->granted;
    a->token = p; a->host = static_cast<uint8_t*>(p); a->device = p; a->bytes = request; a->charged_bytes = q;
    return true;
}
bool release(void*, const SlabCache::Allocation& a) noexcept { std::free(a.token); return true; }

constexpr size_t kBudget = (size_t)1 << 30, kKeys = 1000;   /* ~53 expert pairs */

}  // namespace

TEST(mapped_slab_pregrow_matches_growth_on_demand) {
    std::vector<uint8_t> gu(SlabCache::GU, 7), dn(SlabCache::DN, 9);
    Host ha, hb;
    SlabCache a, b;
    CHECK(a.configure(1, kBudget, kKeys, { &ha, quote, allocate, release }) == S::Ok);
    CHECK(b.configure(1, kBudget, kKeys, { &hb, quote, allocate, release }) == S::Ok);
    size_t added = 0;
    CHECK(b.pregrow(1, &added) == S::Ok);
    CHECK(added > 0);
    const auto grown = b.stats();
    for (size_t k = 0; k < 150; ++k) {   /* more pairs than fit: the LRU tail is evicted */
        CHECK(a.insert(1, 2 * k, gu.data(), gu.size()) == b.insert(1, 2 * k, gu.data(), gu.size()));
        CHECK(a.insert(1, 2 * k + 1, dn.data(), dn.size()) == b.insert(1, 2 * k + 1, dn.data(), dn.size()));
    }
    const auto sa = a.stats(), sb = b.stats();
    CHECK_EQ(sa.charge(), sb.charge());
    CHECK_EQ(sb.charge(), grown.charge());   /* nothing grew after the pregrow */
    CHECK_EQ(sa.growth_attempts, sb.growth_attempts);
    CHECK_EQ(sa.cache.inserts, sb.cache.inserts);
    CHECK_EQ(sa.cache.evictions, sb.cache.evictions);
    CHECK_EQ(sa.cache.resident_entries, sb.cache.resident_entries);
    for (int c = 0; c < 2; ++c) CHECK_EQ(sa.classes[c].capacity_slots, sb.classes[c].capacity_slots);
    for (size_t k = 0; k < 300; ++k) {   /* the same residents, with the bytes they were given */
        SlabCache::Lease la, lb;
        const size_t n = k % 2 ? dn.size() : gu.size();
        const S x = a.borrow(1, k, n, &la), y = b.borrow(1, k, n, &lb);
        CHECK(x == y);
        if (y == S::Hit) CHECK(std::memcmp(lb.host(), k % 2 ? dn.data() : gu.data(), n) == 0);
        (void)la.release();
        (void)lb.release();
    }
    size_t again = 1;
    CHECK(b.pregrow(1, &again) == S::Ok);
    CHECK_EQ(again, (size_t)0);
}

TEST(mapped_slab_pregrow_keeps_what_a_refusing_host_pinned) {
    Host h;
    h.refuse_after = 2;
    SlabCache c;
    CHECK(c.configure(1, kBudget, kKeys, { &h, quote, allocate, release }) == S::Ok);
    size_t added = 0;
    CHECK(c.pregrow(1, &added) == S::Allocation);
    CHECK_EQ(added, (size_t)2);
    const auto st = c.stats();
    CHECK_EQ(st.growth_failures, (uint64_t)1);
    CHECK(st.classes[0].capacity_slots > 0);
    CHECK_EQ(st.classes[0].capacity_slots, st.classes[1].capacity_slots);
    std::vector<uint8_t> gu(SlabCache::GU, 3);
    CHECK(c.insert(1, 0, gu.data(), gu.size()) == S::Ok);   /* the pinned slots still admit */
}

/* Other class sizes (the packed RAM cache's slots, RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE): the
 * slabs are laid out with them -- more pairs in the same budget, every slot 64-byte aligned behind
 * intact guards -- the cache admits, reserves and lends exactly those sizes and refuses block_mxfp4's,
 * and it evicts in the same LRU order as a block cache. Sizes that are equal, zero or not 64-byte
 * multiples are refused. */
TEST(mapped_slab_runtime_class_sizes) {
    const SlabCache::Sizes packed{ 11999296, 6001984 };   /* the engine's default packed slots */
    CHECK(SlabCache::sizes_valid(packed));
    CHECK(SlabCache::sizes_valid(SlabCache::block_sizes));
    Host hb, hp;
    SlabCache blk, pk;
    CHECK(blk.configure(1, kBudget, kKeys, { &hb, quote, allocate, release }) == S::Ok);
    CHECK(pk.configure(1, kBudget, kKeys, { &hp, quote, allocate, release }, packed) == S::Ok);
    CHECK(blk.class_bytes() == SlabCache::block_sizes);
    CHECK(pk.class_bytes() == packed);
    size_t a = 0, b = 0;
    CHECK(blk.pregrow(1, &a) == S::Ok);
    CHECK(pk.pregrow(1, &b) == S::Ok);
    const auto sb = blk.stats(), sp = pk.stats();
    for (int c = 0; c < 2; ++c) {
        CHECK_EQ(sp.classes[c].capacity_slots, sp.classes[0].capacity_slots);
        CHECK_EQ(sp.classes[c].capacity_logical_bytes, sp.classes[c].capacity_slots * packed[(size_t)c]);
        CHECK_EQ(sb.classes[c].capacity_logical_bytes, sb.classes[c].capacity_slots * SlabCache::block_sizes[(size_t)c]);
    }
    /* The same budget holds more pairs, as many as the pair sizes' ratio allows (each slab's quote
     * here is its request plus 4 KiB). */
    CHECK(sp.classes[0].capacity_slots > sb.classes[0].capacity_slots);
    CHECK(sp.charge() <= kBudget);
    /* Block sizes are not this cache's: refused before any admission. */
    std::vector<uint8_t> G(SlabCache::GU, 7), D(SlabCache::DN, 8);
    CHECK(pk.insert(1, 0, G.data(), G.size()) == S::Invalid);
    SlabCache::Lease none;
    CHECK(pk.borrow(1, 0, D.size(), &none) == S::Invalid);
    SlabCache::Write nw;
    CHECK(pk.reserve(1, 0, G.size(), &nw) == S::Invalid);
    std::vector<uint8_t> pg(packed[0], 0), pd(packed[1], 0);
    CHECK(blk.insert(1, 0, pg.data(), pg.size()) == S::Invalid);
    /* Pairs past the capacity in both caches: the same admissions and evictions, slot by slot. */
    const size_t pairs = sp.classes[0].capacity_slots + 20;
    for (size_t k = 0; k < pairs; ++k) {
        std::memset(pg.data(), (int)(k & 255), 4096);
        std::memset(pd.data(), (int)((k + 1) & 255), 4096);
        pg[pg.size() - 1] = (uint8_t)k;
        SlabCache::Write w;
        CHECK(pk.reserve(1, 2 * k, pg.size(), &w) == S::Ok);   /* a reserved slot, as the engine fills it */
        REQUIRE(w.host() != nullptr);
        CHECK_EQ(w.bytes(), packed[0]);
        CHECK_EQ(reinterpret_cast<uintptr_t>(w.host()) % 64, (uintptr_t)0);
        std::memcpy(w.host(), pg.data(), pg.size());
        SlabCache::Lease l;
        CHECK(w.publish(&l) == S::Ok);
        CHECK(l.release() == S::Ok);
        CHECK(pk.insert(1, 2 * k + 1, pd.data(), pd.size()) == S::Ok);
    }
    const auto after = pk.stats();
    CHECK_EQ(after.cache.evictions, (uint64_t)40);   /* 20 pairs past the capacity, two classes */
    CHECK_EQ(after.cache.resident_entries, 2 * sp.classes[0].capacity_slots);
    CHECK_EQ(after.growth_attempts, sp.growth_attempts);   /* nothing grew past the pregrow */
    for (size_t k = pairs - 3; k < pairs; ++k) {   /* the newest are resident with their bytes */
        SlabCache::Lease lg, ld;
        CHECK(pk.borrow(1, 2 * k, packed[0], &lg) == S::Hit);
        CHECK(pk.borrow(1, 2 * k + 1, packed[1], &ld) == S::Hit);
        CHECK_EQ(lg.bytes(), packed[0]);
        CHECK_EQ(ld.bytes(), packed[1]);
        if (lg.host()) { CHECK_EQ(lg.host()[packed[0] - 1], (uint8_t)k); CHECK_EQ(lg.host()[0], (uint8_t)(k & 255)); }
        if (ld.host()) CHECK_EQ(ld.host()[0], (uint8_t)((k + 1) & 255));
        CHECK(lg.release() == S::Ok);
        CHECK(ld.release() == S::Ok);
    }
    SlabCache::Lease oldest;
    CHECK(pk.borrow(1, 0, packed[0], &oldest) == S::Miss);   /* the first pair went first */
    CHECK(pk.clear() == S::Ok);   /* every slot's guards intact, or clear reports Layout */
    /* Refused sizes. */
    Host hx;
    SlabCache x;
    for (const SlabCache::Sizes& bad : { SlabCache::Sizes{ packed[0], packed[0] }, SlabCache::Sizes{ 0, packed[1] },
                                         SlabCache::Sizes{ packed[0] + 1, packed[1] }, SlabCache::Sizes{ packed[0], packed[1] + 32 } })
        CHECK(x.configure(1, kBudget, kKeys, { &hx, quote, allocate, release }, bad) == S::Invalid);
    CHECK_EQ(hx.granted, 0);
}

/* present_many (the routing trace's residency snapshot, RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE)
 * answers present() for every key under one lock: residents, evicted keys, keys never inserted,
 * keys past the cache's count, another namespace, and an unconfigured cache. It touches nothing:
 * the LRU order after it is the one before (the same victim next). */
TEST(mapped_slab_present_many_is_present_for_each_key) {
    std::vector<uint8_t> gu(SlabCache::GU, 5), dn(SlabCache::DN, 6);
    Host h;
    SlabCache c;
    CHECK(c.configure(1, kBudget, kKeys, { &h, quote, allocate, release }) == S::Ok);
    size_t added = 0;
    CHECK(c.pregrow(1, &added) == S::Ok);
    const size_t cap = c.stats().classes[0].capacity_slots, pairs = cap + 5;
    for (size_t k = 0; k < pairs; ++k) {   /* the first five pairs are evicted */
        CHECK(c.insert(1, 2 * k, gu.data(), gu.size()) == S::Ok);
        CHECK(c.insert(1, 2 * k + 1, dn.data(), dn.size()) == S::Ok);
    }
    std::vector<size_t> keys;
    for (size_t k = 0; k < 2 * pairs + 7; ++k) keys.push_back(k);   /* resident, evicted, never */
    keys.push_back(kKeys + 3);                                        /* past the count */
    std::vector<uint8_t> got(keys.size(), 9);
    c.present_many(1, keys.data(), keys.size(), got.data());
    size_t resident = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        CHECK_EQ(got[i], (uint8_t)(c.present(1, keys[i]) ? 1 : 0));
        resident += got[i];
    }
    CHECK_EQ(resident, 2 * cap);
    CHECK_EQ(got[0], (uint8_t)0);
    CHECK_EQ(got[2 * pairs - 1], (uint8_t)1);
    std::fill(got.begin(), got.end(), 9);
    c.present_many(2, keys.data(), keys.size(), got.data());   /* another namespace: nothing */
    CHECK(std::all_of(got.begin(), got.end(), [](uint8_t v) { return v == 0; }));
    /* Untouched: the next insert evicts the oldest resident pair's gate_up, as without the call. */
    CHECK(c.insert(1, 2 * pairs, gu.data(), gu.size()) == S::Ok);
    CHECK(!c.present(1, 10));
    CHECK(c.present(1, 11));
    SlabCache none;
    std::fill(got.begin(), got.end(), 9);
    none.present_many(1, keys.data(), 3, got.data());
    CHECK(got[0] == 0 && got[1] == 0 && got[2] == 0);
}

RAD_TEST_MAIN()

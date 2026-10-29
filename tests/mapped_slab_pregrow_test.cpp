/* mapped_slab_pregrow_test.cpp -- MappedExpertSlabCache::pregrow (core/place/mapped_expert_slab_cache.h):
 * a pregrown cache pins the slabs inserts would grow, in the same order and sizes, and then admits,
 * evicts and hits exactly as one grown on demand; a second pregrow adds nothing; a host that refuses a
 * slab stops the growth and keeps what was pinned. */
#include "rad_test.h"

#include "place/mapped_expert_slab_cache.h"

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

RAD_TEST_MAIN()

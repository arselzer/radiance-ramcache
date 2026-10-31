#include "rad_test.h"
#include "mem/kv_hash_runs.h"

using namespace rad;
TEST(kv_hash_fetch_preserves_logical_order_but_copies_no_unreferenced_plain_holes) {
    RadKVGroupDecl plain{};
    KVHashRuns logical = { { 1024, 128 }, { 128, 128 }, { 256, 128 }, { 1024, 128 } };
    const auto original = logical;
    KVHashRuns physical;
    REQUIRE(kv_hash_physical_runs(plain, 2, 2048, logical, &physical));
    CHECK(logical == original); // caller still hashes the original per-run order/length
    CHECK(physical == (KVHashRuns{ { 128, 256 }, { 1024, 128 } }));
    const auto before = physical;
    CHECK(!kv_hash_physical_runs(plain, 2, 2048, { { 2047, 2 } }, &physical));
    CHECK(physical == before);
}

TEST(kv_hash_packed_ratio_halves_fetch_each_complete_physical_page_once_without_holes) {
    for (uint32_t format : { uint32_t(RAD_KV_STORAGE_DS41_FP4_E4M3_G16),
                             uint32_t(RAD_KV_STORAGE_DS41_FP4_POW2_G32) }) {
        RadKVGroupDecl d{};
        d.kind = RAD_KV_FULL; d.dtype = RAD_BF16; d.n_head_kv = 1;
        const int64_t width = format == 1 ? 512 : 128;
        d.head_dim = width / 4; d.storage_format = format;
        d.storage_block_bytes = format == 1 ? 288 : 68;
        const int64_t page = width * 2, half = page / 2, pitch = d.storage_block_bytes;
        KVHashRuns logical = { { 5 * page, half }, { 5 * page + half, half },
                               { page, half }, { page + half, half } };
        KVHashRuns physical;
        REQUIRE(kv_hash_physical_runs(d, 2, 8 * pitch, logical, &physical));
        CHECK(physical == (KVHashRuns{ { pitch, pitch }, { 5 * pitch, pitch } }));
        const auto old = physical;
        CHECK(!kv_hash_physical_runs(d, 2, 8 * pitch, { { 5 * page + half, page } }, &physical));
        CHECK(physical == old);
        CHECK(!kv_hash_physical_runs(d, 2, 8 * pitch, { { 8 * page, half } }, &physical));
        CHECK(physical == old);
    }
}
RAD_TEST_MAIN()

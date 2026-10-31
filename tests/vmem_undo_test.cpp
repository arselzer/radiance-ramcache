#include "rad_test.h"
#include "device/vmem_undo.h"

TEST(vmem_protect_failure_undo_keeps_failed_unmaps_in_the_actual_ledger) {
    const int64_t G = 65536;
    for (int failed : { -1, 0, 1, 2 }) {
        std::vector<uint8_t> actual(4, 1), ledger = actual;
        const std::vector<int64_t> newly_mapped = { G, 2 * G, 3 * G };
        int calls = 0, forgotten = 0;
        const int rc = rad::vmem_undo_new(newly_mapped, newly_mapped.size(), G,
            [&](int64_t off, int64_t bytes) {
                CHECK_EQ(bytes, G);
                const int ordinal = calls++;
                if (ordinal == failed) return int(RAD_E_DEVICE);
                actual[(size_t)(off / G)] = 0;
                return int(RAD_OK);
            }, [&](int64_t off, int64_t bytes) {
                CHECK_EQ(bytes, G);
                CHECK_EQ(actual[(size_t)(off / G)], uint8_t(0));
                ledger[(size_t)(off / G)] = 0;
                ++forgotten;
            });
        CHECK_EQ(rc, failed < 0 ? RAD_OK : RAD_E_DEVICE);
        CHECK_EQ(calls, 3);
        CHECK_EQ(forgotten, failed < 0 ? 3 : 2);
        CHECK(actual == ledger);
        CHECK_EQ(ledger[0], uint8_t(1)); // preexisting accessible extent is never touched
        if (failed >= 0) CHECK_EQ(ledger[(size_t)failed + 1], uint8_t(1));
    }
}

TEST(vmem_protect_failure_undo_uses_only_the_committed_prefix_and_retains_first_error) {
    const std::vector<int64_t> want = { 0, 65536, 131072, 196608 };
    int calls = 0, forgotten = 0;
    CHECK_EQ(rad::vmem_undo_new(want, 2, 65536,
        [&](int64_t off, int64_t) {
            ++calls;
            CHECK(off < 131072);
            return off ? int(RAD_E_IO) : int(RAD_E_DEVICE);
        }, [&](int64_t, int64_t) { ++forgotten; }), RAD_E_DEVICE);
    CHECK_EQ(calls, 2);
    CHECK_EQ(forgotten, 0);
    CHECK_EQ(rad::vmem_undo_new(want, 0, 65536,
        [&](int64_t, int64_t) { ++calls; return int(RAD_OK); },
        [&](int64_t, int64_t) { ++forgotten; }), RAD_OK);
    CHECK_EQ(calls, 2);
}

RAD_TEST_MAIN()

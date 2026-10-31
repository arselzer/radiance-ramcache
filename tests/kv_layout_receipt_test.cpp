#include "rad_test.h"
#include "mem/kv_layout_receipt.h"
using namespace rad;
TEST(kv_layout_receipt_flag_is_explicit_and_off_by_default) {
    CHECK_EQ(kv_layout_flag(nullptr), 0); CHECK_EQ(kv_layout_flag("0"), 0);
    CHECK_EQ(kv_layout_flag("1"), 1);
    for (const char* bad : {"", "on", "2", "01", "1 ", "-1"}) CHECK_EQ(kv_layout_flag(bad), -1);
}
TEST(kv_layout_receipt_64bit_arithmetic_refuses_overflow_without_mutation) {
    const int64_t top = std::numeric_limits<int64_t>::max();
    int64_t out = 71;
    CHECK(!kv_layout_add(top, 1, &out)); CHECK_EQ(out, 71);
    CHECK(!kv_layout_mul(top, 2, &out)); CHECK_EQ(out, 71);
    CHECK(!kv_layout_mul(-1, 0, &out)); CHECK_EQ(out, 71);
    CHECK(kv_layout_mul(top, 0, &out)); CHECK_EQ(out, 0);
    CHECK(kv_layout_add(top-1, 1, &out)); CHECK_EQ(out, top);
}
TEST(kv_layout_receipt_physical_tail_and_buffer_extent_are_distinct) {
    const int64_t logical = 187ll << 20, physical = 188ll << 20;
    CHECK(kv_layout_extent(0, logical, physical));
    CHECK(!kv_layout_extent(logical, physical-logical, logical));
    CHECK(kv_layout_extent(logical, physical-logical, physical));
    CHECK(!kv_layout_extent(std::numeric_limits<int64_t>::max(), 1, physical));
}
RAD_TEST_MAIN()

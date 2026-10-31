#include "rad_test.h"
#include "runtime/kv_storage_guard.h"

TEST(packed_kv_every_operand_requires_the_exact_op_row_role_and_both_format_declarations) {
    struct Case { uint32_t format; const char* op; const char* row; int operand; } cases[] = {
        { RAD_KV_STORAGE_DS41_FP4_E4M3_G16, "entry_store_kvpack", "ds41_entry_store_kvpack", 3 },
        { RAD_KV_STORAGE_DS41_FP4_POW2_G32, "entry_store_kvpack", "ds41_entry_store_kvpack", 3 },
        { RAD_KV_STORAGE_DS41_FP4_POW2_G32, "index_score_kvpack", "ds41_index_score_d128_kvpack", 2 },
        { RAD_KV_STORAGE_DS41_FP4_E4M3_G16, "attn_sparse_sink_kvpack", "ds41_attn_sparse_sink_d512_kvpack", 4 },
    };
    for (const Case& c : cases) {
        const auto check = [&](const char* op, const char* row, const char* row_op, int operand,
                               int64_t decl, int64_t resolved) {
            return rad::kv_storage_operand_error(c.format, op, row, row_op, operand, decl, resolved);
        };
        CHECK(check(c.op, c.row, c.op, c.operand, c.format, c.format) == nullptr);
        CHECK(check(c.op, c.row, c.op, c.operand, -1, c.format) != nullptr);
        CHECK(check(c.op, c.row, c.op, c.operand, c.format, -1) != nullptr);
        CHECK(check(c.op, c.row, c.op, c.operand, 3 - c.format, c.format) != nullptr);
        CHECK(check(c.op, c.row, c.op, c.operand, c.format, 3 - c.format) != nullptr);
        CHECK(check("entry_store", "entry_store", "entry_store", c.operand, c.format, c.format) != nullptr);
        CHECK(check(c.op, "plain_fallback", c.op, c.operand, c.format, c.format) != nullptr);
        CHECK(check(c.op, c.row, "plain_fallback", c.operand, c.format, c.format) != nullptr);
        for (int operand = 0; operand < 6; ++operand)
            if (operand != c.operand) CHECK(check(c.op, c.row, c.op, operand, c.format, c.format) != nullptr);
    }
    CHECK(rad::kv_storage_operand_error(RAD_KV_STORAGE_DS41_FP4_E4M3_G16,
        "index_score_kvpack", "ds41_index_score_d128_kvpack", "index_score_kvpack", 2, 1, 1) != nullptr);
    CHECK(rad::kv_storage_operand_error(RAD_KV_STORAGE_DS41_FP4_POW2_G32,
        "attn_sparse_sink_kvpack", "ds41_attn_sparse_sink_d512_kvpack", "attn_sparse_sink_kvpack", 4, 2, 2) != nullptr);
    CHECK(rad::kv_storage_operand_error(99, "entry_store_kvpack", "ds41_entry_store_kvpack",
                                         "entry_store_kvpack", 3, 99, 99) != nullptr);
    CHECK(rad::kv_storage_operand_error(0, "entry_store", "ordinary", "entry_store", 3, -1, -1) == nullptr);
}

RAD_TEST_MAIN()

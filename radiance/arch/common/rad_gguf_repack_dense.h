/* Lossless canonical Q5_K/Q8_0 -> native GGUF block layout.
 * These independently written inverse maps follow the standard GGML layouts at
 * llama.cpp 06938ac12 (the source of libquant/lq_ggml.cpp and lq_ggml_port.h).
 * No scale search, activation conversion, or floating-point roundtrip occurs.
 * This host helper is not a GPU kernel and does not alias Q8_0's group32 to group128.
 */
#pragma once
#include "rad_gguf_repack_base.h"

namespace rad::gguf::detail {
inline int dense_validate(Format format, const Reader&, const Region&, std::string* why) {
    if (format == Format::q5_k || format == Format::q8_0) return RAD_OK;
    if (why) *why = "not a supported dense GGUF block format";
    return RAD_E_UNSUPPORTED;
}

inline void dense_write(Format format, const Reader& reader, const Region& region, uint8_t* dst) {
    auto half = [](uint8_t* out, uint32_t bits) {
        out[0] = uint8_t(bits);
        out[1] = uint8_t(bits >> 8);
    };
    if (format == Format::q8_0) {
        for (int64_t r = 0; r < region.rows; ++r) {
            const int64_t row = region.row + r;
            for (int64_t c = 0; c < region.cols; c += 32) {
                const int64_t first = region.col + c;
                half(dst, reader.bits(1, row, first / 32));
                for (int64_t value = 0; value < 32; ++value)
                    dst[2 + value] = uint8_t(reader.bits(0, row, first + value));
                dst += 34;
            }
        }
    } else if (format == Format::q5_k) {
        for (int64_t r = 0; r < region.rows; ++r) {
            const int64_t row = region.row + r;
            for (int64_t c = 0; c < region.cols; c += 256) {
                const int64_t first = region.col + c;
                // All bits in the 176-byte block are assigned. Zero only the two
                // OR-assembled code fields; scale/min fields are written directly.
                half(dst, reader.bits(2, row, first / 256));
                half(dst + 2, reader.bits(4, row, first / 256));
                uint8_t* scales = dst + 4;
                for (int64_t group = 0; group < 4; ++group) {
                    const uint32_t s0 = reader.bits(1, row, first / 32 + group);
                    const uint32_t s1 = reader.bits(1, row, first / 32 + group + 4);
                    const uint32_t m0 = reader.bits(3, row, first / 32 + group);
                    const uint32_t m1 = reader.bits(3, row, first / 32 + group + 4);
                    scales[group] = uint8_t((s0 & 63U) | ((s1 >> 4) << 6));
                    scales[group + 4] = uint8_t((m0 & 63U) | ((m1 >> 4) << 6));
                    scales[group + 8] = uint8_t((s1 & 15U) | ((m1 & 15U) << 4));
                }
                uint8_t* high = dst + 16;
                uint8_t* low = dst + 48;
                for (int i = 0; i < 32; ++i) high[i] = 0;
                for (int i = 0; i < 128; ++i) low[i] = 0;
                for (int64_t value = 0; value < 256; ++value) {
                    const uint32_t code = reader.bits(0, row, first + value);
                    const int64_t low_byte = 32 * (value / 64) + value % 32;
                    const unsigned shift = value % 64 < 32 ? 0U : 4U;
                    low[low_byte] |= uint8_t((code & 15U) << shift);
                    high[value % 32] |= uint8_t(((code >> 4) & 1U) << (value / 32));
                }
                dst += 176;
            }
        }
    }
}
} // namespace rad::gguf::detail

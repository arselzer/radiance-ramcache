/* Lossless storage of an ALREADY BF16-rounded f32 score. No float conversion or
 * re-rounding here, including signed zeros and NaN payloads. */
#pragma once
#include <cstdint>
namespace rad_ds41_score_b16 {
inline constexpr uint16_t high_bits(uint32_t already_rounded) noexcept { return uint16_t(already_rounded>>16); }
inline constexpr uint32_t expand_bits(uint16_t stored) noexcept { return uint32_t(stored)<<16; }
inline constexpr bool representable(uint32_t bits) noexcept { return (bits&0xffffu)==0; }
}

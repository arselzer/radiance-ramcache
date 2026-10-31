/* Exact byte-complement stores for the immutable bank readback poison.
 * No allocation, ISA requirement, float arithmetic or readback authority. */
#pragma once
#include <cstdint>
#include <cstring>

namespace rad { namespace draft_bank_backup_detail {
// Conservative bound for transient word/index/address locals, not a payload
// buffer. The enabled helper charges it before any pinned allocation.
inline constexpr int64_t kComplementWordWorkspaceBytes = 128;

inline bool complement_words(void* output, const void* input, int64_t bytes) noexcept {
    if (bytes == 0) return true; // null pointers are legal only for the empty span
    if (bytes < 0 || !output || !input) return false;
    const uintptr_t dst = uintptr_t(output), src = uintptr_t(input);
    if (uint64_t(bytes) > UINTPTR_MAX - dst || uint64_t(bytes) > UINTPTR_MAX - src) return false;
    auto* back = static_cast<uint8_t*>(output);
    const auto* saved = static_cast<const uint8_t*>(input);
    // Preserve the original forward byte-loop behavior for all overlaps,
    // including output==input. A word read must not change its dependencies.
    const uintptr_t distance = dst >= src ? dst - src : src - dst;
    if (uint64_t(distance) < uint64_t(bytes)) {
        for (int64_t j = 0; j < bytes; ++j) back[j] = uint8_t(~saved[j]);
        return true;
    }
    int64_t j = 0;
    for (; bytes - j >= int64_t(sizeof(uint64_t)); j += int64_t(sizeof(uint64_t))) {
        uint64_t word;
        std::memcpy(&word, saved + j, sizeof(word)); // no alignment or aliasing assumption
        word = ~word;
        std::memcpy(back + j, &word, sizeof(word));
    }
    for (; j < bytes; ++j) back[j] = uint8_t(~saved[j]);
    return true;
}
}} // namespace rad::draft_bank_backup_detail

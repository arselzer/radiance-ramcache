/* Exact byte-complement stores for the immutable bank readback poison.
 * No allocation, ISA requirement, float arithmetic or readback authority. */
#pragma once
#include <cstdint>
#include <cstring>

namespace rad { namespace draft_bank_backup_detail {
// Includes bounded word/address locals and the existing logger's4096B stack
// buffer plus varargs reserve. Enabled-only native receipts fit that buffer.
inline constexpr int64_t kComplementWordWorkspaceBytes = 8192;

struct ComplementWork {
    uint64_t word_calls=0,word_bytes=0,byte_calls=0,byte_bytes=0;
    bool add(const ComplementWork& v) noexcept {
        if(word_calls>UINT64_MAX-v.word_calls||word_bytes>UINT64_MAX-v.word_bytes||
           byte_calls>UINT64_MAX-v.byte_calls||byte_bytes>UINT64_MAX-v.byte_bytes)return false;
        word_calls+=v.word_calls;word_bytes+=v.word_bytes;byte_calls+=v.byte_calls;byte_bytes+=v.byte_bytes;
        return true;
    }
};

inline bool complement_words(void* output, const void* input, int64_t bytes, ComplementWork* work=nullptr) noexcept {
    if(work)*work={};
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
        if(work){work->byte_calls=1;work->byte_bytes=uint64_t(bytes);}
        return true;
    }
    int64_t j = 0;
    for (; bytes - j >= int64_t(sizeof(uint64_t)); j += int64_t(sizeof(uint64_t))) {
        uint64_t word;
        std::memcpy(&word, saved + j, sizeof(word)); // no alignment or aliasing assumption
        word = ~word;
        std::memcpy(back + j, &word, sizeof(word));
    }
    const int64_t word_bytes=j;
    for (; j < bytes; ++j) back[j] = uint8_t(~saved[j]);
    if(work){work->word_calls=uint64_t(word_bytes>0);work->word_bytes=uint64_t(word_bytes);
        work->byte_calls=uint64_t(bytes>word_bytes);work->byte_bytes=uint64_t(bytes-word_bytes);}
    return true;
}
}} // namespace rad::draft_bank_backup_detail

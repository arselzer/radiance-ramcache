#include "ds41_packed_inverse.h"
#include "../../tools/ds41-host-experts/ds41_host_experts.h"
#include "../../libr4d/r4d_ds41_mxfp4_pack_sse2.h"
#include "../../libquant/lq_ggml_planes.h"
#include <cstring>
#include <limits>

namespace rad::ds41_packed_inverse {
namespace {
bool range(const void* p, size_t n) noexcept {
    return p && n && n <= std::numeric_limits<uintptr_t>::max() - uintptr_t(p);
}
bool apart(const void* a, size_t an, const void* b, size_t bn) noexcept {
    return uintptr_t(a) + an <= uintptr_t(b) || uintptr_t(b) + bn <= uintptr_t(a);
}
uint32_t index(const ds41hx::Mx& v, int i) noexcept {
    uint32_t out;
    // packed_view permits an unaligned byte source; never dereference its u32
    // exception view as an aligned object.
    std::memcpy(&out, reinterpret_cast<const uint8_t*>(v.exc_idx) + size_t(i) * 4, 4);
    return out;
}
} // namespace
size_t block_bytes(int N, int K) noexcept {
    if (N <= 0 || K <= 0 || K % 32) return 0;
    const uint64_t blocks = uint64_t(N) * uint64_t(K / 32);
    const uint64_t rounds = uint64_t(N) * uint64_t((K / 32 + 31) / 32);
    if (rounds > uint64_t(INT32_MAX) || blocks > uint64_t(SIZE_MAX) / 17) return 0;
    return size_t(blocks * 17);
}
int to_block(const uint8_t* source, size_t source_bytes, int N, int K,
             uint8_t* dst, size_t capacity, Stats* stats) noexcept {
    const size_t need = block_bytes(N, K);
    if (!need || !range(source, source_bytes) || !dst) return RAD_E_INVAL;
    if (capacity < need) return RAD_E_FULL;
    if (!range(dst, capacity) || !apart(source, source_bytes, dst, capacity) ||
        (stats && (!range(stats, sizeof(*stats)) || !apart(stats, sizeof(*stats), source, source_bytes) ||
                   !apart(stats, sizeof(*stats), dst, capacity))))
        return RAD_E_INVAL;
    const int blocks = K / 32, rounds = (blocks + 31) / 32;
    const uint64_t total_rounds = uint64_t(N) * uint64_t(rounds);
    // packed_view uses size_t section arithmetic. Bound even its largest cap
    // before calling it, including the header plus four alignment roundups.
    const uint64_t worst_packed = uint64_t(N) * uint64_t(K / 2) +
        uint64_t(N) * uint64_t(ds41hx::pk_row_bytes(K)) + total_rounds * 36 + 5 * 64;
    if (worst_packed > uint64_t(SIZE_MAX)) return RAD_E_INVAL;
    ds41hx::Mx v;
    if (!ds41hx::packed_view(source, source_bytes, N, K, &v)) return RAD_E_STATE;
    uint32_t previous = 0;
    for (int i = 0; i < v.n_exc; ++i) {
        const uint32_t at = index(v, i);
        if (uint64_t(at) >= total_rounds || (i && at <= previous)) return RAD_E_STATE;
        previous = at;
    }
#if !DS41_MXFP4_PACK_HAS_SSE2
    const auto pack = lq::ggml::pack_of(lq::ggml::GGML_TYPE_MXFP4);
    if (!pack) return RAD_E_UNSUPPORTED;
#endif
    const size_t packed_pitch = ds41hx::pk_row_bytes(K);
    int exception = 0;
    for (int row = 0; row < N; ++row) {
        for (int r = 0; r < rounds; ++r) {
            const int count = r < blocks / 32 ? 32 : blocks % 32;
            const uint32_t ordinal = uint32_t(row) * uint32_t(rounds) + uint32_t(r);
            const uint8_t* scales;
            uint8_t expanded[32];
            if (exception < v.n_exc && index(v, exception) == ordinal) {
                scales = v.exc_raw + size_t(exception++) * 32;
            } else {
                const int bytes = (count + 3) / 4;
                const uint8_t* p = v.pk + size_t(row) * packed_pitch + size_t(r) * 8;
                // Exactly pk_round's full/partial round ordering, scalar integer
                // expansion of e0+c. e0<=252 was checked by packed_view.
                for (int j = 0; j < count; ++j)
                    expanded[j] = uint8_t(v.e0 + ((p[j % bytes] >> (2 * (j / bytes))) & 3));
                scales = expanded;
            }
            const size_t first_block = size_t(row) * size_t(blocks) + size_t(r) * 32;
            const uint8_t* planes[2] = {v.codes + first_block * 16, scales};
#if DS41_MXFP4_PACK_HAS_SSE2
            // Same integer byte relayout selected by the existing MXFP4 loader.
            ds41_mxfp4_pack_sse2::pack(planes, count, dst + first_block * 17);
#else
            pack(planes, count, dst + first_block * 17);
#endif
        }
    }
    if (stats) *stats = {uint64_t(N) * uint64_t(blocks), total_rounds, uint64_t(v.n_exc),
                         uint64_t(N) * uint64_t(K / 2), uint64_t(need)};
    return RAD_OK;
}
} // namespace rad::ds41_packed_inverse

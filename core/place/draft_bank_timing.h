/* Private optional host diagnostics. Counters describe requested bytes and
 * reported success, not physical DMA traffic. No clock, allocator or GPU here. */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rad {
enum class DraftBankTimingPart : size_t {
    Allocation, Zero, CaptureD2H, CaptureFNV, CaptureComplement, CaptureVerifyD2H, CaptureMemcmp,
    RestoreH2D, RestoreComplement, RestoreD2H, RestoreMemcmp,
    InvariantComplement, InvariantD2H, InvariantMemcmp, Count
};
struct DraftBankTimingCounter {
    uint64_t ns = 0, attempts = 0, ok = 0, requested_bytes = 0, ok_bytes = 0;
    /* Transactional: on overflow EVERY field remains unchanged. A zero-duration
     * measurement is valid; the clock's resolution is not an operation oracle. */
    bool record(uint64_t elapsed, uint64_t bytes, bool success) noexcept {
        constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
        if (attempts == max || ns > max-elapsed || requested_bytes > max-bytes ||
            (success && (ok == max || ok_bytes > max-bytes))) return false;
        ns += elapsed; ++attempts; requested_bytes += bytes;
        if (success) { ++ok; ok_bytes += bytes; }
        return true;
    }
};
struct DraftBankTimings {
    bool enabled = false, overflow = false;
    std::array<DraftBankTimingCounter, size_t(DraftBankTimingPart::Count)> part{};
    bool record(DraftBankTimingPart p, uint64_t ns, uint64_t bytes, bool success) noexcept {
        if (!enabled) return true;
        if (overflow || size_t(p) >= part.size() || !part[size_t(p)].record(ns,bytes,success)) {
            overflow = true; return false;
        }
        return true;
    }
    /* Deliberately no clear/reset operation. A new Image starts all-zero; lend,
     * restore, release, failed/repeated capture and poison retain cumulative
     * counters and the overflow latch. Only destroying it starts a new lifetime. */
};
/* Each of14 fixed names is <=32 bytes; five u64 columns need <=20 digits each.
 * Header allowance256 includes status/hash/fixed schema. Charge all formatting
 * storage and scalar clock/record locals through the existing explicit budget. */
inline constexpr size_t kDraftBankTimingLogBytes = 256 + size_t(DraftBankTimingPart::Count)*(32+1+5*20+4+1);
// log_emit uses its existing4096-byte stack buffer simultaneously. Our bounded
// line fits it, so its long-line std::string allocation is never needed. Charge
// both buffers plus scalar/varargs reserve, rounded to an explicit8KiB bound.
inline constexpr int64_t kDraftBankTimingWorkspaceBytes = 8192;
static_assert(kDraftBankTimingLogBytes + 4096 + 1024 <= size_t(kDraftBankTimingWorkspaceBytes));
} // namespace rad

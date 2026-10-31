/* Allocation-free preparation of one leased native trunk GU/DN pair. Nothing
 * here copies to a card, publishes residency, releases a lease or reads a file. */
#pragma once
#include "mover.h"
#include "ds41_packed_inverse.h"
#include "draft_vram.h"

namespace rad::draft_bank_packed_fill {
enum class Refusal : uint8_t { None, Epoch, Unit, Geometry, Lease, Alias, Inverse };
struct Workspace {
    ds41_packed_inverse::Stats inverse[2]{};
    uint64_t source_bytes = 0, raw_bytes = 0;
    Refusal refusal = Refusal::None;
};
/* No payload scratch is added: dst is the backup's existing one-pair relay.
 * Workspace is retained in (and charged through) sizeof(Engine::DraftBankLoan).
 * Charge this additional fixed call-metadata allowance while the option is on:
 * two parsed views/headers, scale expansion <=32B, integer/SSE relayout locals
 * and the adapter's scalar temporaries; no recursion, VLA or model-sized stack. */
inline constexpr int64_t kCallMetadataBound = 4096;
using EpochCheck = bool (*)(void*) noexcept;
using Inverse = int (*)(const uint8_t*, size_t, int, int, uint8_t*, size_t,
                       ds41_packed_inverse::Stats*) noexcept;
/* A refused inverse may have written part of dst. The caller must clear the
 * whole pair before the original-file fallback. Neither a partial pair nor a
 * stale-source result may be copied/published. The actual active leases remain
 * owned by the caller THROUGH synchronous H2D completion, also on fallback. */
int prepare(const Program& program, const Plan& plan, int32_t unit,
            const Mover::HostUnit& leased, uint8_t* dst, size_t bytes,
            Workspace* work, EpochCheck epochs, void* epoch_context,
            Inverse inverse = ds41_packed_inverse::to_block) noexcept;
} // namespace rad::draft_bank_packed_fill

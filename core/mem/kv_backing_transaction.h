/* All-rank backing transaction, deliberately without an engine caller/default-on switch.
 *
 * Backend calls change real physical mappings. Publication and quiescence are mandatory seams:
 * withdrawing allocatable IDs BEFORE unmap, and publishing only after every rank succeeds, are
 * not optional policy callbacks. A failed rollback poisons the participants instead of resuming.
 * The fixed-expert first integration may free physical KV but cannot grow expert residency. */
#pragma once
#include "mem/kv_growth.h"
#include "rad_types.h"
#include <functional>

namespace rad {
class Pool;

struct KVBackingSnapshot {
    int64_t reserved_bytes = 0;
    int64_t granule_bytes = 0;
    std::vector<uint8_t> backed; // PHYSICAL presence; failed commit does not establish accessibility
};

class KVBackingBackend {
public:
    virtual ~KVBackingBackend() = default;
    /* Reuse the supplied bitmap capacity after first read. A negative mapping call can have
     * changed mappings; read must report those physical extents truthfully even on that path. */
    virtual int read(KVBackingSnapshot* out) = 0;
    virtual int commit(int64_t offset, int64_t bytes) = 0;
    virtual int release(int64_t offset, int64_t bytes) = 0;
};

/* Actual Pool/VMem adapter. It binds its rank's device before each operation. No fixed-pool
 * fallback claims dynamic saving: unsupported/nonelastic pools are refused by read(). */
class PoolKVBackingBackend final : public KVBackingBackend {
public:
    PoolKVBackingBackend(Pool* pool, int device) : pool_(pool), device_(device) {}
    int read(KVBackingSnapshot* out) override;
    int commit(int64_t offset, int64_t bytes) override;
    int release(int64_t offset, int64_t bytes) override;
private:
    Pool* pool_;
    int device_;
};

class KVBackingPublisher {
public:
    virtual ~KVBackingPublisher() = default;
    /* Allocate/validate BOTH old and target publications now. No visible mutation on refusal.
     * Every byte a held ID can read must remain mapped. A target ID is eligible only when its
     * entire physical page in EVERY bound layer is mapped, including cross-granule pages.
     * Future KVManager adapter must cover grab/reserve_specific/adopt/follow and free counts. */
    virtual int stage(const KVBackingSnapshot& old_map, const KVBackingSnapshot& target_map) = 0;
    virtual void discard() noexcept = 0;
    /* Infallible, allocation-free swaps while all allocators/readers are quiescent. withdraw
     * exposes NO allocatable IDs; existing held tables remain pinned by required ranges. */
    virtual void withdraw() noexcept = 0;
    virtual void publish_target() noexcept = 0;
    virtual void publish_old() noexcept = 0;
    virtual void poison() noexcept = 0; // no IDs may be handed out after unrepairable mapping error
};

class KVBackingQuiescence {
public:
    virtual ~KVBackingQuiescence() = default;
    /* Freeze scheduler admission/cancel/prefix/tier publication and drain all rank compute,
     * second-lane, draft, mover, stager, tier and copy streams. Recall all KV expert loans.
     * enter failure must leave the old engine state unchanged and unlocked. */
    virtual int enter() = 0;
    virtual void leave() noexcept = 0;
    virtual void poison() noexcept = 0; // latch fatal engine state; never resume inference
};

class KVBackingFunding {
public:
    virtual ~KVBackingFunding() = default;
    /* Reserve the per-rank peak AND rollback-to-old physical costs against a joint ledger BEFORE
     * the first map. A reduced maximum is accepted only at commit: keep old funding until then.
     * For fixed experts, these ceilings come from the existing configured KV split. A future
     * expanding expert pool must really release physical expert pages here, not merely loans.
     * Failure leaves funding unchanged. Final callbacks must be infallible/allocation-free. */
    virtual int reserve(const std::vector<KVBackingSnapshot>& old_maps,
                         const std::vector<KVBackingPlan>& plans) = 0;
    virtual void committed() noexcept = 0;
    virtual void rolled_back() noexcept = 0;
    virtual void poison() noexcept = 0;
};

struct KVBackingParticipant {
    KVBackingBackend* backend = nullptr;
    KVBackingPublisher* publisher = nullptr;
    KVBackingLimits limits;
    /* Called AFTER enter(), under its ownership lock. Include every current holder plus fixed
     * ranges and next-step writes. Returning a stale pre-enter inventory is not permitted. */
    std::function<int(std::vector<KVBackingRange>*)> inventory;
};

enum class KVBackingTransactionResult { Committed, Refused, RolledBack, Poisoned };
struct KVBackingTransactionReport {
    KVBackingTransactionResult result = KVBackingTransactionResult::Refused;
    int status = RAD_E_INVAL;
    std::vector<KVBackingPlan> plans;
    // Last verified PHYSICAL mapping. validity0 means a backend read failed; do not infer backing.
    std::vector<KVBackingSnapshot> actual;
    std::vector<uint8_t> actual_valid;
};

/* The mandatory holder inventory must be complete and old held mappings accessible on entry.
 * On commit failure, physical presence is NEVER taken as proof of access: rollback unmaps every
 * new extent even if the failed call left it recorded, and only then restores old publication.
 * No caller yet wires this to the live engine or KVManager free lists. */
KVBackingTransactionReport kv_backing_transact(const std::vector<KVBackingParticipant>& ranks,
                                              KVBackingQuiescence* quiet,
                                              KVBackingFunding* funding);
}  // namespace rad

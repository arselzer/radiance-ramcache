/* Immutable native DSpark hot-bank bytes. This helper never authorizes a bank
 * query: the lender must also be in its restored cohort phase, with all bank
 * work drained. No allocation/copy occurs on a steady-ready query. */
#pragma once
#include "draft_vram.h"
#include "draft_bank_timing.h"
#include "rad_device.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace rad {
struct Program;
struct Plan;
class Mover;
class DraftBankBackup;

struct DraftBankBackupRank {
    int rank = -1, device = -1;
    const Program* program = nullptr;
    const Plan* plan = nullptr;
    Mover* mover = nullptr;
    RadStream stream = nullptr; // already drained, owned by Engine
};
using DraftBankBackupRanks = std::array<DraftBankBackupRank, kDraftRanks>;

struct DraftBankBackupBudget {
    int64_t backup_mib = 0; // REQUIRED explicit cap; never inferred from RAM
    int64_t expert_cache_budget_bytes = 0;
    int64_t expert_cache_backed_bytes = 0;
    int64_t other_future_bytes = 0;
    int64_t system_headroom_bytes = 0;
    // Retained lender metadata outside this helper, e.g. sizeof(Engine::Loan).
    // Count it even if its fixed storage already exists at capture time.
    int64_t controller_metadata_bytes = 0;
    bool timings = false; // explicit caller opt-in; no helper getenv/steady-ready clock
    bool complement_words = false; // exact uint64 stores; literal byte loop remains the default
};

struct DraftBankBackupStats {
    int64_t raw_bytes = 0, mask_bytes = 0;
    int64_t relay_bytes = 0, readback_bytes = 0, shared_scratch_bytes = 0;
    int64_t pinned_requested_bytes = 0, pinned_charge_bytes = 0;
    int64_t metadata_bytes = 0, controller_metadata_bytes = 0, total_charge_bytes = 0, cap_bytes = 0;
    int64_t future_cache_bytes = 0, future_other_bytes = 0, system_headroom_bytes = 0;
    int64_t capture_copy_bytes = 0, restore_copy_bytes = 0, verify_copy_bytes = 0;
    uint64_t image_hash = 0; // diagnostic only; strict memcmp decides readiness
    uint64_t completed_restores = 0;
    bool complement_words = false;
    DraftBankTimings timings{}; // included in sizeof(Image)/Loan accounting, even when off
};

/* The injected adapter must outlive the backup. All copies are synchronous and
 * byte-preserving. A success with a short copy is detected by readback. */
class DraftBankBackupIO {
public:
    virtual ~DraftBankBackupIO() = default;
    virtual int64_t allocation_charge(int64_t requested) const noexcept = 0;
    virtual bool room(int64_t need, int64_t still, int64_t margin, std::string* why) = 0;
    virtual void* alloc_pinned(int64_t bytes) = 0;
    virtual void free_pinned(void*) noexcept = 0;
    virtual int copy_sync(int rank, void* dst, const void* src, int64_t bytes) = 0;
};

/* Real adapter. bind/copy_sync are public so the controller can fill a borrowed
 * slot through this same bounded scratch and the rank's existing copy stream.
 * Each operation returns to rank0's selected device. */
class DraftBankBackupDeviceIO final : public DraftBankBackupIO {
public:
    DraftBankBackupDeviceIO() = default;
    explicit DraftBankBackupDeviceIO(const DraftBankBackupRanks& ranks) { bind(ranks); }
    void bind(const DraftBankBackupRanks& ranks) noexcept { ranks_ = ranks; }
    int64_t allocation_charge(int64_t requested) const noexcept override;
    bool room(int64_t need, int64_t still, int64_t margin, std::string* why) override;
    void* alloc_pinned(int64_t bytes) override;
    void free_pinned(void*) noexcept override;
    int copy_sync(int rank, void* dst, const void* src, int64_t bytes) override;
private:
    DraftBankBackupRanks ranks_{};
};

struct DraftBankBackupSlot {
    void* base = nullptr;
    int64_t bytes = 0;
    int32_t hot_unit = -1;
    int stage = -1, expert = -1;
};

namespace draft_bank_backup_detail {
constexpr int kSlots = kDraftStages * kDraftRankExperts;
constexpr int kBindings = 2 * kSlots;
constexpr int kSpans = kSlots + kDraftStages;

struct Binding { uint32_t weight = 0; void* ptr = nullptr; bool hot = false; };
struct Plane { uint32_t weight = 0; void* ptr = nullptr; int64_t bytes = 0; };
struct Layout {
    std::array<DraftBankBackupSlot, kSlots> slots{};
    std::array<Binding, kBindings> bindings{};
    std::array<Plane, kDraftStages> planes{};
    int nslots = 0, nbindings = 0;
};

/* Pure metadata seam. Native runtime uses Mover::unit_base/residency; source
 * tests use synthetic addresses without allocating/reading model payload. */
struct Addresses {
    void* context = nullptr;
    void* (*unit)(void*, int32_t) = nullptr;
    void* (*weight)(void*, uint32_t) = nullptr;
};
int derive_layout(const Program&, const Plan&, int n, const Addresses&, Layout*, std::string* why);
int check_bindings(const Program&, const Layout&, const Addresses&, std::string* why);

struct Span { void* base = nullptr; int64_t bytes = 0; bool restore = true; };
struct ImageRank {
    std::array<Span, kSpans> spans{};
    int nspans = 0;
    int64_t scratch_bytes = 0;
};

/* Bounded byte-copy mechanism shared by native wrapper and small fault tests.
 * Runtime must use DraftBankBackup::capture, which first proves native scope.
 * No public environment switch or engine bypass calls this generic mechanism. */
class Image {
    friend class ::rad::DraftBankBackup;
public:
    Image() = default;
    ~Image();
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    int capture(const std::array<ImageRank, kDraftRanks>&, const DraftBankBackupBudget&,
                int64_t metadata_bytes, DraftBankBackupIO*, std::string* why);
    int restore_verify_all(std::string* why);
    int verify_unchanged(std::string* why);
    void mark_borrowed() noexcept { ready_ = false; }
    int poison(const char* why, std::string* out);
    bool ready() const noexcept { return ready_ && !failed_; }
    bool failed() const noexcept { return failed_; }
    void* scratch(int rank) const noexcept;
    int64_t scratch_bytes(int rank) const noexcept;
    const DraftBankBackupStats& stats() const noexcept { return stats_; }
private:
    enum class VerifyPhase : uint8_t { Capture, Restore, Invariant };
    int verify(int rank, int span, std::string* why, VerifyPhase phase);
    int64_t timing_begin() const noexcept;
    int timing_end(DraftBankTimingPart, int64_t start, int64_t bytes, bool success, std::string* why);
    void timing_log(const char* operation, int status) const noexcept;
    void release() noexcept;
    std::array<ImageRank, kDraftRanks> ranks_{};
    std::array<void*, kDraftRanks> allocation_{};
    std::array<int64_t, kDraftRanks> snapshot_bytes_{};
    std::array<std::array<int64_t, kSpans>, kDraftRanks> offsets_{};
    DraftBankBackupIO* io_ = nullptr;
    DraftBankBackupStats stats_{};
    bool attempted_ = false, captured_ = false, ready_ = false, failed_ = false;
};
} // namespace draft_bank_backup_detail

class DraftBankBackup {
public:
    DraftBankBackup() = default;
    DraftBankBackup(const DraftBankBackup&) = delete;
    DraftBankBackup& operator=(const DraftBankBackup&) = delete;
    /* One-shot startup capture AFTER normal load, alias publication and router
     * masking. Caller holds the drained all-rank cohort; no query/context can run.
     * A null adapter uses owned real IO, which outlives the immutable image. */
    int capture(const DraftBankBackupRanks&, int n, const DraftBankBackupBudget&,
                DraftBankBackupIO* io = nullptr, std::string* why = nullptr);
    int restore_verify_all(std::string* why = nullptr);
    /* Expensive pointer AND mask-byte check, for transition boundaries only. */
    int validate_invariants(std::string* why = nullptr);
    void mark_borrowed() noexcept { image_.mark_borrowed(); }
    bool ready() const noexcept { return image_.ready(); }
    bool failed() const noexcept { return image_.failed(); }
    int slots(int rank) const noexcept;
    const DraftBankBackupSlot& slot(int rank, int i) const;
    void* scratch(int rank) const noexcept { return image_.scratch(rank); }
    int64_t scratch_bytes() const noexcept { return kDraftPairBytes; }
    const DraftBankBackupStats& stats() const noexcept { return image_.stats(); }
    DraftBankBackupDeviceIO& device_io() noexcept { return device_io_; }
private:
    int check_pointers(std::string* why);
    DraftBankBackupDeviceIO device_io_; // declared before image_: freed after it
    DraftBankBackupRanks ranks_{};
    std::array<draft_bank_backup_detail::Layout, kDraftRanks> layout_{};
    draft_bank_backup_detail::Image image_;
    bool attempted_ = false;
};
} // namespace rad

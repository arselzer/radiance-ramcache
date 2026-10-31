/* Successful query-owned rank0 D2H enqueues, bound after the existing real query/echo check.
 * Pure accounting: no clocks, allocation, device work, target-copy accounting or fake epoch. */
#pragma once
#include <cstdint>
namespace rad {
inline constexpr char kDraftQueryCopyEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_QUERY_COPY_STATS";
struct DraftQueryCopyStats {
    enum Kind { Observer, Slots };
    uint64_t observer_calls = 0, observer_bytes = 0, slot_calls = 0, slot_bytes = 0;
    uint64_t started = 0, bound_epochs = 0, owner_mismatches = 0, last_epoch = 0, last_seq = 0;
    uint64_t observer_row_bytes = 0, slot_copy_bytes = 0;
    bool configure(uint64_t observer_width, uint64_t slot_width) noexcept {
        if (!observer_width || !slot_width || started || observer_calls || slot_calls || pending_) return false;
        observer_row_bytes = observer_width; slot_copy_bytes = slot_width;
        return true;
    }
    bool begin(int step, uint64_t seq, int slot, int card) noexcept {
        if (pending_ || step < 0 || !seq || slot < 0 || card != 0 ||
            !observer_row_bytes || !slot_copy_bytes || started == UINT64_MAX) return mismatch();
        pending_ = true; step_ = step; seq_ = seq; slot_ = slot; card_ = card;
        observer_ = slots_ = false; ++started;
        return true;
    }
    bool copied(int status, Kind kind, uint64_t bytes) noexcept {
        if (status < 0) return false; // a failed API call is never a successful enqueue
        uint64_t& calls = kind == Observer ? observer_calls : slot_calls;
        uint64_t& total = kind == Observer ? observer_bytes : slot_bytes;
        if (calls == UINT64_MAX || bytes > UINT64_MAX-total) return mismatch();
        ++calls; total += bytes; // actual successes remain counted even if their binding is bad
        bool& seen = kind == Observer ? observer_ : slots_;
        const uint64_t expected = kind == Observer ? observer_row_bytes : slot_copy_bytes;
        if (!pending_ || seen || bytes != expected) return mismatch();
        seen = true;
        return true;
    }
    bool bind(int step, uint64_t seq, int slot, int card, uint64_t epoch) noexcept {
        if (!pending_ || !observer_ || !slots_ || owner_mismatches || step != step_ ||
            seq != seq_ || slot != slot_ || card != card_ || !epoch ||
            bound_epochs == UINT64_MAX || epoch != bound_epochs+1) return mismatch();
        ++bound_epochs; last_epoch = epoch; last_seq = seq; pending_ = false;
        return true;
    }
    uint64_t unbound() const noexcept { return started-bound_epochs; }
    bool pending() const noexcept { return pending_; }
private:
    bool pending_ = false, observer_ = false, slots_ = false;
    int step_ = -1, slot_ = -1, card_ = -1;
    uint64_t seq_ = 0;
    bool mismatch() noexcept { if (owner_mismatches != UINT64_MAX) ++owner_mismatches; return false; }
};
} // namespace rad

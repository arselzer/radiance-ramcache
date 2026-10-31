/* Optional host-only observation of exact P1 arena ownership. No allocator,
 * device operation, clock or authority to publish/reuse a slot. */
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <cstring>

namespace rad { namespace p1 {
inline bool arena_audit_config(const char* text, int arena, int ring, bool* enabled) noexcept {
    if (!enabled) return false;
    *enabled = false;
    if (!text || !*text || !std::strcmp(text, "0")) return true;
    if (std::strcmp(text, "1") || arena < 1 || arena > 128 || ring != 0) return false;
    *enabled = true;
    return true;
}
class ArenaUseAudit {
public:
    static constexpr int kSlots = 128;
    void init(bool enabled, int n, int64_t slot_bytes) noexcept {
        *this = ArenaUseAudit{};
        enabled_ = enabled;
        if (!enabled) return;
        valid_ = n > 0 && n <= kSlots && slot_bytes > 0 && slot_bytes <= INT64_MAX / n;
        if (valid_) { n_ = n; bytes_ = n * slot_bytes; }
    }
    bool enabled() const noexcept { return enabled_; }
    bool valid() const noexcept { return enabled_ && valid_; }
    void refuse() noexcept { if (enabled_) valid_ = false; }
    void bind(int k, int32_t unit, uintptr_t base) noexcept {
        if (!enabled_) return;
        if (!valid_ || k < 0 || k >= n_ || unit < 0 || !base || slots_[k].bound) { refuse(); return; }
        for (int i = 0; i < n_; ++i)
            if (slots_[i].bound && (slots_[i].unit == unit || slots_[i].base == base)) { refuse(); return; }
        slots_[k].unit = unit; slots_[k].base = base; slots_[k].bound = true;
    }
    void publish(int k, bool on) noexcept {
        if (!enabled_) return;
        if (!valid_ || k < 0 || k >= n_ || !slots_[k].bound ||
            slots_[k].epoch == std::numeric_limits<uint64_t>::max()) { refuse(); return; }
        auto& s = slots_[k];
        ++s.epoch; s.live = on;
        if (on) published_[k / 64] |= uint64_t{1} << (k % 64);
    }
    uint64_t epoch(int k) const noexcept { return k >= 0 && k < n_ ? slots_[k].epoch : 0; }
    void select(int k, int32_t unit, uintptr_t base, uint64_t epoch,
                int64_t selected_rows, bool current_owner) noexcept {
        if (!enabled_ || selected_rows == 0) return;
        if (!valid_ || k < 0 || k >= n_ || selected_rows < 0 || !current_owner ||
            !slots_[k].bound || !slots_[k].live || slots_[k].unit != unit ||
            slots_[k].base != base || slots_[k].epoch != epoch ||
            selections_ == std::numeric_limits<uint64_t>::max()) { refuse(); return; }
        selected_[k / 64] |= uint64_t{1} << (k % 64);
        ++selections_;
    }
    int granted() const noexcept { return n_; }
    int64_t granted_bytes() const noexcept { return bytes_; }
    int published() const noexcept { return count(published_); }
    int published_upper() const noexcept { return __builtin_popcountll(published_[1]); }
    int selected() const noexcept { return count(selected_); }
    int selected_upper() const noexcept { return __builtin_popcountll(selected_[1]); }
    uint64_t selections() const noexcept { return selections_; }
private:
    struct Slot { int32_t unit = -1; uintptr_t base = 0; uint64_t epoch = 0; bool bound = false, live = false; };
    static int count(const std::array<uint64_t, 2>& a) noexcept {
        return __builtin_popcountll(a[0]) + __builtin_popcountll(a[1]);
    }
    std::array<Slot, kSlots> slots_{};
    std::array<uint64_t, 2> published_{}, selected_{};
    int n_ = 0;
    int64_t bytes_ = 0;
    uint64_t selections_ = 0;
    bool enabled_ = false, valid_ = false;
};
} } // namespace rad::p1

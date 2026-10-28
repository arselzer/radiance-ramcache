/* Native file-weight cache. No cache pointer escapes copy(): hits complete an ordinary CPU
 * copy into the existing pinned Mover window while locked, before any subsequent DMA. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rad {

class NativeFileCache {
public:
    struct Stats {
        uint64_t hits = 0, misses = 0, hit_bytes = 0, miss_bytes = 0;
        uint64_t inserts = 0, evictions = 0, bypasses = 0, allocation_failures = 0;
        int64_t budget_bytes = 0, metadata_bytes = 0, payload_charge = 0, peak_charge = 0;
        size_t entries = 0;
        uint64_t payload_maps = 0, payload_unmaps = 0, reused_payloads = 0;
        uint64_t map_failures = 0, unmap_failures = 0;
        int64_t mapped_bytes = 0, peak_mapped_bytes = 0, page_bytes = 0;
        bool mmap_reuse_enabled = false;
    };

    ~NativeFileCache();
    /* Quiescent one-Mover/one-Program namespace. Exact env value "1" opts in; default OFF.
     * Budget includes this candidate's larger metadata and page-rounded mappings. */
    int configure(int64_t budget_bytes, size_t weight_count);
    /* An unmap failure is an error: its owner and charge remain, allowing a later retry. */
    int clear();
    int copy(size_t weight, int64_t bytes, void* dst, int64_t dst_cap, bool* hit);
    int insert(size_t weight, const void* src, int64_t bytes);
    Stats stats() const;

private:
    friend struct NativeFileCacheMmapTest; // private class-only allocation-failure seam
    using Map = uint8_t* (*)(void*, size_t);
    using Unmap = int (*)(void*, uint8_t*, size_t);
    static uint8_t* os_map(void*, size_t);
    static int os_unmap(void*, uint8_t*, size_t);
    static constexpr size_t none = SIZE_MAX;
    struct Mapping {
        uint8_t* ptr = nullptr;
        size_t bytes = 0;
        Unmap unmap = nullptr;
        void* context = nullptr;
        Mapping() = default;
        Mapping(const Mapping&) = delete;
        Mapping& operator=(const Mapping&) = delete;
        ~Mapping();
        int close();
        void swap(Mapping&) noexcept;
    };
    struct Entry {
        std::unique_ptr<uint8_t[]> data;
        Mapping mapping;
        int64_t bytes = 0, charge = 0;
        size_t previous = none, next = none;
        uint8_t* payload() const { return mapping.ptr ? mapping.ptr : data.get(); }
    };
    int reset_locked();
    void unlink_locked(size_t weight);
    void touch_locked(size_t weight);
    int evict_locked(size_t weight);
    void account_eviction_locked(size_t weight);
    int insert_mapped_locked(size_t weight, const void* src, int64_t bytes);

    mutable std::mutex mutex_;
    std::unique_ptr<Entry[]> entries_;
    size_t count_ = 0, first_ = none, last_ = none;
    Stats stats_;
    Map map_ = os_map;
    Unmap unmap_ = os_unmap;
    void* mapping_context_ = nullptr;
    int test_mode_ = -1; // private fixture only; -1 is normal environment selection
    int64_t test_page_bytes_ = 0;
};

} // namespace rad

/* Native file-weight cache. No cache pointer escapes copy(): hits complete an ordinary CPU
 * copy into the existing pinned Mover window before any subsequent DMA. Optional concurrent
 * hits retain a shared payload lifetime lock while metadata locking stays short. */
#pragma once
#include "native_file_cache_policy.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>

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
        bool concurrent_copy_enabled = false;
        uint64_t concurrent_copies = 0;
        size_t active_copy_leases = 0, peak_copy_leases = 0;
    };

    ~NativeFileCache();
    /* Quiescent one-Mover/one-Program namespace. Exact env value "1" opts in; default OFF.
     * Budget includes this candidate's larger metadata and page-rounded mappings. */
    int configure(int64_t budget_bytes, size_t weight_count);
    /* An unmap failure is an error: its owner and charge remain, allowing a later retry. */
    int clear();
    int copy(size_t weight, int64_t bytes, void* dst, int64_t dst_cap, bool* hit, NativeFileCacheReadPolicy policy = NativeFileCacheReadPolicy::Normal);
    int insert(size_t weight, const void* src, int64_t bytes);
    Stats stats() const;
    /* The payload charge inserts may grow to (RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET); INT64_MAX,
     * the default, is none. An insert past it evicts in LRU order as a full budget does, so the
     * cache stops growing and keeps replacing. An atomic: any thread, any time. */
    void set_growth_limit(int64_t bytes) { growth_limit_.store(bytes, std::memory_order_relaxed); }
    int64_t growth_limit() const { return growth_limit_.load(std::memory_order_relaxed); }

private:
    friend struct NativeFileCacheConcurrentTest; // private HOST lifetime/overlap seam
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
    int copy_shared(size_t weight, int64_t bytes, void* dst, bool* hit, NativeFileCacheReadPolicy policy);
    int reset_locked();
    void unlink_locked(size_t weight);
    void touch_locked(size_t weight);
    int evict_locked(size_t weight);
    void account_eviction_locked(size_t weight);
    int insert_mapped_locked(size_t weight, const void* src, int64_t bytes);

    // Writers always acquire payload lifetime before metadata. stats() only takes metadata.
    mutable std::shared_mutex payload_mutex_;
    mutable std::mutex mutex_;
    std::atomic<bool> concurrent_copy_{false};
    std::atomic<int64_t> growth_limit_{INT64_MAX};
    std::unique_ptr<Entry[]> entries_;
    size_t count_ = 0, first_ = none, last_ = none;
    Stats stats_;
    Map map_ = os_map;
    Unmap unmap_ = os_unmap;
    void* mapping_context_ = nullptr;
    int test_mode_ = -1; // private fixture only; -1 is normal environment selection
    int64_t test_page_bytes_ = 0;
    int test_concurrent_mode_ = -1; // private fixture only; -1 selects exact environment value
    void (*test_copy_lease_)(void*) noexcept = nullptr;
    void* test_copy_context_ = nullptr;
};

} // namespace rad

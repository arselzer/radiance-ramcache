#include "native_file_cache.h"
#include "rad_types.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <sys/mman.h>
#include <unistd.h>

namespace rad {

uint8_t* NativeFileCache::os_map(void*, size_t bytes) {
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
}
int NativeFileCache::os_unmap(void*, uint8_t* p, size_t bytes) { return ::munmap(p, bytes); }

int NativeFileCache::Mapping::close() {
    if (!ptr) return RAD_OK;
    if (!unmap || unmap(context, ptr, bytes) != 0) return RAD_E_IO;
    ptr = nullptr; bytes = 0; unmap = nullptr; context = nullptr;
    return RAD_OK;
}
NativeFileCache::Mapping::~Mapping() {
    if (close() != RAD_OK)
        std::fprintf(stderr, "native cache: mapping teardown failed; %zu bytes NOT confirmed unmapped\n", bytes);
}
void NativeFileCache::Mapping::swap(Mapping& other) noexcept {
    std::swap(ptr, other.ptr); std::swap(bytes, other.bytes);
    std::swap(unmap, other.unmap); std::swap(context, other.context);
}
NativeFileCache::~NativeFileCache() {
    if (clear() != RAD_OK)
        std::fprintf(stderr, "native cache: clear failed during destruction; remaining owners retry teardown\n");
}

int NativeFileCache::reset_locked() {
    while (last_ != none) {
        const int rc = evict_locked(last_);
        if (rc != RAD_OK) return rc; // keep failed victim/namespace/charge truthful
    }
    entries_.reset();
    count_ = 0;
    first_ = last_ = none;
    stats_ = {};
    concurrent_copy_.store(false, std::memory_order_release);
    return RAD_OK;
}

int NativeFileCache::clear() {
    std::unique_lock<std::shared_mutex> lifetime(payload_mutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    return reset_locked();
}

int NativeFileCache::configure(int64_t budget_bytes, size_t weight_count) {
    std::unique_lock<std::shared_mutex> lifetime(payload_mutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    const int cleared = reset_locked();
    if (cleared != RAD_OK) return cleared;
    if (budget_bytes < 0) return RAD_E_INVAL;
    if (!budget_bytes) return RAD_OK;
    constexpr int64_t overhead = sizeof(NativeFileCache) + 128;
    if (!weight_count || weight_count > (uint64_t)(INT64_MAX - overhead) / sizeof(Entry))
        return RAD_E_INVAL;
    const int64_t metadata = overhead + (int64_t)weight_count * sizeof(Entry);
    if (metadata >= budget_bytes) return RAD_E_SCRATCH;
    const char* opt = std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MMAP_REUSE");
    const bool mapped = test_mode_ >= 0 ? test_mode_ == 1 : opt && std::strcmp(opt, "1") == 0;
    const int64_t page = mapped ? (test_page_bytes_ ? test_page_bytes_ : ::sysconf(_SC_PAGESIZE)) : 0;
    if (mapped && (page < 64 || (page & (page - 1)) || uint64_t(page) > SIZE_MAX))
        return RAD_E_STATE;
    std::unique_ptr<Entry[]> entries(new (std::nothrow) Entry[weight_count]);
    if (!entries) return RAD_E_NOMEM;
    entries_ = std::move(entries);
    count_ = weight_count;
    stats_.budget_bytes = budget_bytes;
    stats_.metadata_bytes = metadata;
    stats_.peak_charge = metadata;
    stats_.mmap_reuse_enabled = mapped;
    stats_.page_bytes = page;
    const char* parallel = std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_CONCURRENT_COPY");
    const bool concurrent = test_concurrent_mode_ >= 0 ? test_concurrent_mode_ == 1
        : parallel && std::strcmp(parallel, "1") == 0;
    stats_.concurrent_copy_enabled = concurrent;
    concurrent_copy_.store(concurrent, std::memory_order_release);
    return RAD_OK;
}

void NativeFileCache::unlink_locked(size_t weight) {
    Entry& e = entries_[weight];
    if (e.previous != none) entries_[e.previous].next = e.next;
    else first_ = e.next;
    if (e.next != none) entries_[e.next].previous = e.previous;
    else last_ = e.previous;
    e.previous = e.next = none;
}

void NativeFileCache::touch_locked(size_t weight) {
    Entry& e = entries_[weight];
    if (first_ == weight) return;
    if (e.previous != none || e.next != none || last_ == weight) unlink_locked(weight);
    e.next = first_;
    e.previous = none;
    if (first_ != none) entries_[first_].previous = weight;
    else last_ = weight;
    first_ = weight;
}

void NativeFileCache::account_eviction_locked(size_t weight) {
    Entry& e = entries_[weight];
    unlink_locked(weight);
    stats_.payload_charge -= e.charge;
    --stats_.entries;
    ++stats_.evictions;
    e.data.reset();
    e.bytes = e.charge = 0;
}

int NativeFileCache::evict_locked(size_t weight) {
    Entry& e = entries_[weight];
    if (e.mapping.ptr) {
        const int64_t mapped = int64_t(e.mapping.bytes);
        if (e.mapping.close() != RAD_OK) { ++stats_.unmap_failures; return RAD_E_IO; }
        ++stats_.payload_unmaps;
        stats_.mapped_bytes -= mapped;
    }
    account_eviction_locked(weight);
    return RAD_OK;
}

int NativeFileCache::copy(size_t weight, int64_t bytes, void* dst, int64_t dst_cap, bool* hit,
                          NativeFileCacheReadPolicy policy) {
    if (!hit || !dst || bytes <= 0 || dst_cap < bytes || !native_file_cache_policy_valid(policy))
        return RAD_E_INVAL;
    *hit = false;
    if (concurrent_copy_.load(std::memory_order_acquire))
        return copy_shared(weight, bytes, dst, hit, policy);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entries_) return RAD_OK;
    if (weight >= count_) return RAD_E_INVAL;
    Entry& e = entries_[weight];
    if (!e.payload()) {
        ++stats_.misses;
        stats_.miss_bytes += (uint64_t)bytes;
        return RAD_OK;
    }
    if (e.bytes != bytes) return RAD_E_STATE; // Program/native layout must not change in place.
    std::memcpy(dst, e.payload(), (size_t)bytes);
    if (policy == NativeFileCacheReadPolicy::Normal) touch_locked(weight);
    ++stats_.hits;
    stats_.hit_bytes += (uint64_t)bytes;
    *hit = true;
    return RAD_OK;
}


int NativeFileCache::copy_shared(size_t weight, int64_t bytes, void* dst, bool* hit,
                                 NativeFileCacheReadPolicy policy) {
    // A writer cannot evict, reuse, replace or clear any payload until this lease ends.
    // Metadata is released during memcpy, allowing disjoint destination copies to overlap.
    std::shared_lock<std::shared_mutex> lifetime(payload_mutex_);
    std::unique_lock<std::mutex> lock(mutex_);
    if (!entries_) return RAD_OK;
    if (weight >= count_) return RAD_E_INVAL;
    Entry& e = entries_[weight];
    if (!e.payload()) {
        ++stats_.misses;
        stats_.miss_bytes += uint64_t(bytes);
        return RAD_OK;
    }
    if (e.bytes != bytes) return RAD_E_STATE;
    const uint8_t* source = e.payload();
    ++stats_.active_copy_leases;
    stats_.peak_copy_leases = std::max(stats_.peak_copy_leases, stats_.active_copy_leases);
    lock.unlock();
    if (test_copy_lease_) test_copy_lease_(test_copy_context_);
    std::memcpy(dst, source, size_t(bytes));
    lock.lock();
    --stats_.active_copy_leases;
    ++stats_.concurrent_copies;
    // Completion order defines LRU ordering among concurrent hits. No pointer escapes.
    if (policy == NativeFileCacheReadPolicy::Normal) touch_locked(weight);
    ++stats_.hits;
    stats_.hit_bytes += uint64_t(bytes);
    *hit = true;
    return RAD_OK;
}

int NativeFileCache::insert_mapped_locked(size_t weight, const void* src, int64_t bytes) {
    const int64_t page = stats_.page_bytes;
    if (bytes > INT64_MAX - (page - 1) - 64) return RAD_E_INVAL;
    const int64_t mapped = ((bytes + page - 1) / page) * page;
    if (uint64_t(mapped) > SIZE_MAX) return RAD_E_INVAL;
    const int64_t charge = mapped + 64;
    const int64_t room = std::min(stats_.budget_bytes - stats_.metadata_bytes,
                                  growth_limit_.load(std::memory_order_relaxed));
    if (charge > room) { ++stats_.bypasses; return RAD_OK; }
    Mapping retained;
    while (stats_.payload_charge > room - charge) {
        Entry& victim = entries_[last_];
        /* Reuse ONLY the final tail victim already required for room. All earlier required
         * victims are normally released in exact LRU order. No search, extra eviction or
         * persistent spare mapping. After retention only memcpy/nonthrow ownership remain. */
        if (victim.bytes == bytes && victim.mapping.bytes == size_t(mapped) &&
            stats_.payload_charge - victim.charge <= room - charge) {
            const size_t key = last_;
            retained.swap(victim.mapping);
            account_eviction_locked(key);
            ++stats_.reused_payloads;
            break;
        }
        const int rc = evict_locked(last_);
        if (rc != RAD_OK) return rc;
    }
    if (!retained.ptr) {
        uint8_t* p = map_(mapping_context_, size_t(mapped));
        if (!p) {
            ++stats_.map_failures; ++stats_.allocation_failures; ++stats_.bypasses;
            return RAD_OK;
        }
        retained.ptr = p; retained.bytes = size_t(mapped);
        retained.unmap = unmap_; retained.context = mapping_context_;
        ++stats_.payload_maps;
        stats_.mapped_bytes += mapped;
        stats_.peak_mapped_bytes = std::max(stats_.peak_mapped_bytes, stats_.mapped_bytes);
    }
    std::memcpy(retained.ptr, src, size_t(bytes));
    Entry& e = entries_[weight];
    e.mapping.swap(retained);
    e.bytes = bytes; e.charge = charge;
    stats_.payload_charge += charge;
    ++stats_.entries; ++stats_.inserts;
    stats_.peak_charge = std::max(stats_.peak_charge, stats_.metadata_bytes + stats_.payload_charge);
    touch_locked(weight);
    return RAD_OK;
}

int NativeFileCache::insert(size_t weight, const void* src, int64_t bytes) {
    if (!src || bytes <= 0 || bytes > INT64_MAX - 127) return RAD_E_INVAL;
    std::unique_lock<std::shared_mutex> lifetime(payload_mutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entries_) return RAD_OK;
    if (weight >= count_) return RAD_E_INVAL;
    Entry& e = entries_[weight];
    if (e.payload()) {
        if (e.bytes != bytes) return RAD_E_STATE;
        touch_locked(weight);
        return RAD_OK;
    }
    if (stats_.mmap_reuse_enabled) return insert_mapped_locked(weight, src, bytes);
    // Legacy default-OFF allocation and charge policy; same victim order and bypass semantics.
    const int64_t charge = ((bytes + 63) / 64) * 64 + 64;
    const int64_t room = std::min(stats_.budget_bytes - stats_.metadata_bytes,
                                  growth_limit_.load(std::memory_order_relaxed));
    if (charge > room) { ++stats_.bypasses; return RAD_OK; }
    while (stats_.payload_charge > room - charge) {
        const int rc = evict_locked(last_);
        if (rc != RAD_OK) return rc;
    }
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[(size_t)bytes]);
    if (!data) {
        ++stats_.allocation_failures;
        ++stats_.bypasses;
        return RAD_OK;
    }
    std::memcpy(data.get(), src, (size_t)bytes);
    e.data = std::move(data);
    e.bytes = bytes;
    e.charge = charge;
    stats_.payload_charge += charge;
    ++stats_.entries;
    ++stats_.inserts;
    const int64_t total = stats_.metadata_bytes + stats_.payload_charge;
    if (total > stats_.peak_charge) stats_.peak_charge = total;
    touch_locked(weight);
    return RAD_OK;
}

NativeFileCache::Stats NativeFileCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace rad

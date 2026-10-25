/* Bounded, immutable HOST RoPE table sharing. This is a planning/lifetime helper,
 * not an architecture graph, GPU allocation, or a persistent table file format. */
#pragma once
#include "ds41_transform_table.h"
#include <bit>
#include <cfenv>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>

namespace rad::ds41::transform {
inline constexpr uint32_t rope_table_algorithm_version=1;
static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);

struct RopeTableLimits {uint64_t payload_bytes=0,max_entries=0;};
struct RopeTableStats {uint64_t payload_bytes=0,entries=0;};
struct RopeTableKey {
    int64_t rotary=0,original_context=0,positions=0;
    uint32_t theta_bits=0,factor_bits=0,beta_fast_bits=0,beta_slow_bits=0,algorithm_version=0;
    auto operator<=>(const RopeTableKey&) const=default;
};

/* Run before BOTH a hit and a construction. Floating-point control settings
 * are thread-local caller state; they must remain unchanged during this call.
 * Arithmetic probes may set exception status flags, but do not change controls. */
inline bool rope_table_environment_supported() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    return false;
#else
    if(std::fegetround()!=FE_TONEAREST)return false;
    volatile float normal=std::bit_cast<float>(uint32_t(0x00800000u)),half=.5f;
    volatile float tiny=std::bit_cast<float>(uint32_t(0x00400000u)),two=2.f;
    return std::bit_cast<uint32_t>(normal*half)==0x00400000u&&
           std::bit_cast<uint32_t>(tiny*two)==0x00800000u;
#endif
}

/* This ledger never owns the map or a table. Keeping a read-only accounting
 * handle does not retain payload, and cannot form a table/map ownership cycle. */
class RopeTableAccounting final {
public:
    explicit RopeTableAccounting(RopeTableLimits limits):limits_(limits) {}
    RopeTableStats stats() const {std::lock_guard lock(mutex_);return live_;}
    RopeTableLimits limits() const noexcept {return limits_;}
private:
    friend class RopeTableStore;
    friend class RopeTable;
    const RopeTableLimits limits_;
    mutable std::mutex mutex_;
    RopeTableStats live_;
    bool reserve(uint64_t bytes) {
        std::lock_guard lock(mutex_);
        if(live_.entries>=limits_.max_entries||live_.payload_bytes>limits_.payload_bytes||
           bytes>limits_.payload_bytes-live_.payload_bytes)return false;
        ++live_.entries;live_.payload_bytes+=bytes;return true;
    }
    void release(uint64_t bytes) noexcept {
        std::lock_guard lock(mutex_);
        --live_.entries;live_.payload_bytes-=bytes;
    }
};

class RopeTable final {
public:
    RopeTable(const RopeTable&)=delete;
    RopeTable& operator=(const RopeTable&)=delete;
    ~RopeTable() {
        // Physical vector release precedes the charge refund, including when the
        // last external lease outlives the store and its cache map.
        std::vector<float>().swap(pairs_);
        accounting_->release(uint64_t(receipt_.bytes));
    }
    const RopeTableKey& key() const noexcept {return key_;}
    const RopeSpec& spec() const noexcept {return spec_;}
    const TableReceipt& receipt() const noexcept {return receipt_;}
    std::span<const float> pairs() const noexcept {return pairs_;}
private:
    friend class RopeTableStore;
    RopeTable(RopeTableKey key,RopeSpec spec,TableReceipt receipt,std::vector<float>&& pairs,
              std::shared_ptr<RopeTableAccounting> accounting) noexcept:
        key_(key),spec_(spec),receipt_(receipt),pairs_(std::move(pairs)),accounting_(std::move(accounting)) {}
    const RopeTableKey key_;
    const RopeSpec spec_;
    const TableReceipt receipt_;
    std::vector<float> pairs_;
    std::shared_ptr<RopeTableAccounting> accounting_;
};

class RopeTableStore final {
public:
    using Lease=std::shared_ptr<const RopeTable>;
    explicit RopeTableStore(RopeTableLimits limits):accounting_(std::make_shared<RopeTableAccounting>(limits)) {}
    RopeTableStore(const RopeTableStore&)=delete;
    RopeTableStore& operator=(const RopeTableStore&)=delete;
    std::shared_ptr<const RopeTableAccounting> accounting() const noexcept {return accounting_;}
    RopeTableStats stats() const {return accounting_->stats();}

    /* A cold construction holds this store's mutex. This intentionally serial
     * baseline guarantees same-key dedup and bounded reservations without an
     * async job layer. Successful replacement is the only write to *out. */
    int get(const RopeSpec& spec,int64_t positions,Lease* out,
            uint32_t algorithm_version=rope_table_algorithm_version) {
        if(!out)return RAD_E_INVAL;
        if(algorithm_version!=rope_table_algorithm_version)return RAD_E_UNSUPPORTED;
        const int valid=validate_spec(spec);if(valid!=RAD_OK)return valid;
        if(positions<=0||positions>max_positions)return RAD_E_SHAPE;
        int64_t elements=0,bytes=0;
        if(!math::multiply(positions,spec.rotary,&elements)||!math::multiply(elements,4,&bytes)||
           uint64_t(bytes)>SIZE_MAX||uint64_t(elements)>std::vector<float>().max_size())return RAD_E_SHAPE;
        if(!rope_table_environment_supported())return RAD_E_UNSUPPORTED;
        const RopeTableKey key{spec.rotary,spec.original_context,positions,
            std::bit_cast<uint32_t>(spec.theta),std::bit_cast<uint32_t>(spec.factor),
            std::bit_cast<uint32_t>(spec.beta_fast),std::bit_cast<uint32_t>(spec.beta_slow),algorithm_version};
        std::lock_guard lock(mutex_);
        if(const auto it=tables_.find(key);it!=tables_.end()) {
            Lease table=it->second.strong?it->second.strong:it->second.weak.lock();
            if(table) {it->second.strong=table;*out=std::move(table);return RAD_OK;}
            tables_.erase(it);
        }
        // An entry whose sole owner was concurrently promoted from a weak lease
        // during clear retains its weak identity. Remove only truly expired keys
        // before admitting new metadata, so both entries and payload stay bounded.
        for(auto it=tables_.begin();it!=tables_.end();) {
            if(!it->second.strong&&it->second.weak.expired())it=tables_.erase(it);
            else ++it;
        }
        if(!accounting_->reserve(uint64_t(bytes)))return RAD_E_NOMEM;
        struct Reservation {
            std::shared_ptr<RopeTableAccounting> accounting;uint64_t bytes;bool active=true;
            ~Reservation() {if(active)accounting->release(bytes);}
        } reservation{accounting_,uint64_t(bytes)};
        try {
            std::vector<float> pairs;TableReceipt receipt;
            const int status=build_pairs(spec,positions,uint64_t(bytes),&pairs,&receipt);
            if(status!=RAD_OK)return status;
            auto owned=std::unique_ptr<RopeTable>(new RopeTable(key,spec,receipt,std::move(pairs),accounting_));
            reservation.active=false; // The table now refunds after freeing its vector.
            Lease table(std::move(owned));
            tables_.emplace(key,Entry{table,table});
            *out=std::move(table);return RAD_OK;
        } catch(const std::bad_alloc&) {return RAD_E_NOMEM;}
          catch(const std::length_error&) {return RAD_E_SHAPE;}
    }

    /* There is no automatic eviction. A live external lease is never removed or
     * reused; release it and explicitly clear to recover that table's budget.
     * Weak promotion racing the sole-owner check keeps the key and object alive. */
    size_t clear_unleased() {
        std::lock_guard lock(mutex_);size_t count=0;
        for(auto it=tables_.begin();it!=tables_.end();) {
            if(it->second.strong&&it->second.strong.use_count()==1)it->second.strong.reset();
            if(!it->second.strong&&it->second.weak.expired()) {it=tables_.erase(it);++count;}
            else ++it;
        }
        return count;
    }
private:
    struct Entry {Lease strong;std::weak_ptr<const RopeTable> weak;};
    std::shared_ptr<RopeTableAccounting> accounting_;
    std::mutex mutex_;
    std::map<RopeTableKey,Entry> tables_;
};
} // namespace rad::ds41::transform

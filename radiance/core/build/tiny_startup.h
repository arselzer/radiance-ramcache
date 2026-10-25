/* Isolated research transport admission. No Engine/CLI activation. */
#pragma once
#include "rad_abi.h"
#include "rad_device.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rad {
struct TinyBackendQuote {
    uint32_t version = 1, host = 0;
    uint64_t stream_size = 0, event_size = 0, task_size = 0, queue_depth = 0;
    uint64_t stream_heap_bytes = 0, stream_control_allowance = 0;
    uint64_t stream_packet_bytes = 0, stream_karg_bytes = 0, stream_bounce_bytes = 0;
    uint64_t karg_host = 0, backend_threads_per_stream = 0;
    uint64_t stack_bytes = 0, guard_bytes = 0;
    // Real private process-global/TLS bookkeeping facts. These caches may be
    // retained by the backend beyond stream destruction. Control reservation
    // is explicit; this POD does not claim to bound driver/allocator RSS.
    uint64_t signal_size = 0, signal_pool_size = 0, allocation_record_size = 0;
    uint64_t allocation_history_limit = 0, host_wait_tls_element_size = 0;
    uint64_t stream_registry_element_size = 0, global_control_allowance = 262144;
};
/* Actual selected backend private types; absent appended hook refuses. */
int tiny_backend_quote(TinyBackendQuote* out);
int tiny_backend_stream_create(RadStream* out);
int tiny_thread_defaults(uint64_t* stack, uint64_t* guard);

enum class TinyCategory : uint32_t {
    Source, Factory, Provider, Cache, Adapter, NativeMetadata, EngineMetadata,
    Pools, Arena, Runtime, Loader, AuxStacks, BackendStreams, Controls, Count
};
constexpr size_t kTinyCategoryCount = static_cast<size_t>(TinyCategory::Count);
struct TinyBudgetSnapshot {
    // Admission/ticket requests; these counters are not malloc/backend traces.
    std::array<uint64_t,kTinyCategoryCount> live{}, peak{}, requests{};
    uint64_t streams = 0, aux_threads = 0, whole_reserved = 0;
    // Non-ticketed credit owned by this admission until Budget destruction.
    // Generic Controls tickets and stream controls cannot spend it again.
    uint64_t admission_controls = 0;
};
class TinyBudget;
class TinyTicket {
public:
    TinyTicket() = default;
    ~TinyTicket();
    TinyTicket(const TinyTicket&) = delete;
    TinyTicket& operator=(const TinyTicket&) = delete;
    TinyTicket(TinyTicket&&) noexcept;
    TinyTicket& operator=(TinyTicket&&) noexcept;
    void reset() noexcept;
    bool live() const noexcept { return budget_ != nullptr; }
private:
    friend class TinyBudget;
    TinyBudget* budget_ = nullptr;
    TinyCategory category_ = TinyCategory::Source;
    uint64_t bytes_ = 0;
    uint32_t role_ = 0; // 1 stream, 2 auxiliary thread
};
class TinyBudget {
public:
    static constexpr uint64_t kWholeCap = 201326592;
    static constexpr uint64_t kCategorySum = 135244800;
    static constexpr std::array<uint64_t,kTinyCategoryCount> caps{{
        12582912,8388608,25165824,163840,65536,4194304,8388608,
        6291456,4194304,4194304,2097152,16785408,38538240,4194304}};
    TinyBudget() = default;
    ~TinyBudget(); // tickets must retire before this owner
    TinyBudget(const TinyBudget&) = delete;
    TinyBudget& operator=(const TinyBudget&) = delete;
    int admit(); // once only, verifies actual backend and all five HOST stack defaults
    int take(TinyCategory category, uint64_t bytes, TinyTicket* out);
    int take_aux_thread(TinyTicket* out);
    int take_stream(TinyTicket* backing, TinyTicket* control);
    TinyBudgetSnapshot snapshot() const;
    TinyBackendQuote backend() const;
    bool admitted() const;
private:
    friend class TinyTicket;
    void release(TinyCategory, uint64_t, uint32_t) noexcept;
    int take_locked(TinyCategory,uint64_t,uint32_t,TinyTicket*);
    mutable std::mutex mu_;
    TinyBudgetSnapshot state_{};
    TinyBackendQuote backend_{};
    uint64_t aux_bytes_ = 0;
    bool admitted_ = false;
};
/* Negative real drain is not evidence of retirement. Only this explicit research
 * path fails stop before releasing any possibly referenced source/backing. */
[[noreturn]] void tiny_unretired_fail_stop() noexcept;
void tiny_drain_or_fail(RadStream,int device) noexcept;
class TinyStream {
public:
    TinyStream() = default;
    ~TinyStream();
    TinyStream(const TinyStream&) = delete;
    TinyStream& operator=(const TinyStream&) = delete;
    int open(TinyBudget& budget, int device);
    void close(); // real successful drain before destruction/ticket release
    RadStream get() const noexcept { return stream_; }
private:
    TinyTicket backing_, control_;
    RadStream stream_ = nullptr;
    int device_ = 0;
};
}

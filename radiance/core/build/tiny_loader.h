#pragma once
#include "tiny_startup.h"
#include <cstddef>
#include <cstdint>
namespace rad {
class Program;
/* Manual research seam to the actual RelayoutPool, not an Engine/CLI switch.
 * Borrowed Program, jobs, fill contexts and destinations must outlive the call.
 * No callback is retained after all workers and actual uplinks retire. */
struct TinyLoadJob {
    const Program* program = nullptr;
    size_t weight = 0;
    uint8_t* destination = nullptr;
    int device = 0;
    bool device_destination = false;
    int64_t input_bytes = 0;
    int64_t plane_offset[RAD_ENC_MAX_PLANES] = {};
    int (*fill)(void*,uint8_t*,int64_t) = nullptr;
    void* fill_context = nullptr;
};
enum class TinyLoaderFault : uint32_t {
    None, BeforeSecondWorker, BeforePinnedAllocation, AfterCopyBeforeRecord,
    AfterRecord, BeforeSlotWait
};
struct TinyLoaderQuote {
    uint64_t jobs=0, input_pages=0, input_arena=1048576, pinned_slots=524288;
    uint64_t staged_size=0, pool_size=0, bounce_size=0, uplink_size=0, slot_size=0;
    uint64_t control_allowance=262144, requested_bytes=0, page_upper=0;
    uint64_t event_size=0, events=4, worker_threads=2;
};
struct TinyLoaderReceipt {
    uint64_t workers_started=0, workers_joined=0, copies_enqueued=0, records_enqueued=0;
    uint64_t slot_waits=0, real_uplink_drains=0, live_bounces_at_final_drain=0;
    uint64_t faults_triggered=0, pinned_allocations=0, pinned_releases=0;
    uint64_t input_pages=0, category_bytes=0;
    bool submissions_stopped=false, all_retired=false;
};
int tiny_loader_quote(const TinyLoadJob*,size_t,TinyLoaderQuote*);
/* Success publishes receipt only at final retirement. Failure also publishes a
 * diagnostic receipt after cleanup, never a ready/result authority. If an
 * allocation fails before the Pool exists, no work has started and all_retired
 * remains false; after a constructed Pool it records proven retirement.
 * Destination rollback is intentionally not promised after a terminal error. */
int tiny_loader_run(const TinyLoadJob*,size_t,TinyBudget&,TinyLoaderFault,
                    TinyLoaderReceipt* diagnostic);
}

/* Engine-owned first live physical-KV cohort. It changes backing only inside the Engine's
 * pre-admission drain/ownership lease; expert resident addresses and budgets remain fixed. */
#pragma once
#include "mem/kv_live_budget.h"
#include "mem/kv_backing_allocator.h"

#include <functional>
#include <memory>
#include <vector>

namespace rad {
class Pool;
class KVManager;
struct Program;

class KVLiveController {
public:
    struct Rank { int device = 0; Pool* pool = nullptr; KVManager* manager = nullptr;
                  const Program* program = nullptr; };
    using FreeBytes = std::function<int(int, int64_t*)>;
    KVLiveController();
    ~KVLiveController();
    KVLiveController(const KVLiveController&) = delete;
    KVLiveController& operator=(const KVLiveController&) = delete;
    /* All ranks already carved in backing mode, no requests/rank issuers have started. The
     * caller drained eager copies. Validate/fund the FULL rounded maximum, then commit the
     * physical minimum plus raw-state/window floors and install the leader grow callback.
     * `free_bytes` is a narrow host-test seam; production reads RadDeviceProps::vram_free. */
    int start(const KVLiveOptions& options, const std::vector<Rank>& ranks,
               int64_t startup_headroom, int64_t late_startup, int64_t late_runtime,
               FreeBytes free_bytes = {});
    /* Called ONLY by Engine::kv_quiescent_plan after its all-rank drains while scheduler
     * ownership is held. Remains active through step_locked; finish_plan runs after it returns.
     * The grow callback cannot map outside this lease. */
    int begin_plan();
    void finish_plan() noexcept;
    int status() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace rad

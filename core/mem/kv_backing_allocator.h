/* The actual KVManager publication adapter; no environment switch or engine enablement here. */
#pragma once
#include "mem/kv_backing_transaction.h"
namespace rad {
class KVManager;
struct KVBackingBlock { int32_t group = -1, block = -1; };
struct KVBackingGrowResult { int status = RAD_OK; bool fatal = false; };
class KVManagerBackingPublisher final : public KVBackingPublisher {
public:
    explicit KVManagerBackingPublisher(KVManager* manager) : manager_(manager) {}
    int stage(const KVBackingSnapshot&, const KVBackingSnapshot&) override;
    void discard() noexcept override;
    void withdraw() noexcept override;
    void publish_target() noexcept override;
    void publish_old() noexcept override;
    void poison() noexcept override;
private:
    KVManager* manager_;
};
} // namespace rad

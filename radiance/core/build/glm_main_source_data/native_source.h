#pragma once
// Complete-main DATA adapter only. No Session/Engine admission or TP execution.
#include "main_bundle_reader.h"
#include "storage/rank_native_store.h"
#include "place/record_owner.h"
#include <mutex>
namespace rad::glm_main_data {
namespace sc=rad::glm_main_provider;
struct AdapterUsage {
    size_t live_holders=0,peak_holders=0,holder_limit=64;
    uint64_t metadata_reservation=0;
};
class GlmMainNativeSource final:public rad::NativeRecordSource {
public:
    // The verified authority and its borrowed Program/Plan/Checkpoint owners must
    // remain immutable/alive while open. Strong record leases survive close and
    // never inspect those borrowed owners afterward. Limits are explicit: the
    // small synthetic provider defaults do NOT fit actual 3,342,336-byte records.
    static int open(std::shared_ptr<sc::ProcessBudget>,
                    std::shared_ptr<const VerifiedRankMain>,const sc::Limits&,
                    std::shared_ptr<GlmMainNativeSource>* out,std::string* why=nullptr)noexcept;
    ~GlmMainNativeSource()override;
    std::span<const rad::NativeRecordDescriptor> catalog()const noexcept override;
    int acquire(std::span<const uint32_t>,rad::NativeRecordBatch*)override;
    // This initial no-resident profile has no bootstrap payload path.
    int bootstrap(int32_t,std::span<std::byte>)noexcept;
    // Exactly once, before acquire: setup-only rank-store stamps/Program audit.
    // Reader/source startup audits belong to their separately bounded owners.
    int finish_bootstrap()noexcept;
    int close()noexcept;
    sc::Stats stats()const noexcept;
    sc::ProcessUsage process_usage()const noexcept;
    sc::Accounting accounting()const noexcept;
    AdapterUsage adapter_usage()const noexcept;
private:
    struct Quota;struct Holder;
    GlmMainNativeSource();
    std::shared_ptr<sc::ProcessBudget> process_;
    std::shared_ptr<const VerifiedRankMain> authority_;
    std::unique_ptr<sc::RankNativeStore> store_;
    std::vector<sc::BoundRecord> handles_;
    std::shared_ptr<Quota> quota_;
    mutable std::mutex operation_;
    size_t request_limit_=0;
    bool bootstrap_done_=false,closed_=false;
};
int native_status(sc::Error)noexcept;
} // namespace rad::glm_main_data

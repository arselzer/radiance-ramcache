#pragma once
#include "rank_native_store.h"
#include "place/record_owner.h"
#include <mutex>
namespace glm53::engine_core_p3 {
namespace sc=glm53::engine_core_sharded;
struct AdapterUsage {size_t live_holders=0,peak_holders=0;uint64_t reserved_bytes=65536;};
class ShardedNativeSource final:public rad::NativeRecordSource {
public:
    static int open(const rad::Program&,const rad::Plan&,const std::string& root,
                    std::span<const sc::ShardInput>,uint64_t cold,uint64_t hot,
                    std::shared_ptr<ShardedNativeSource>* out,std::string* why=nullptr);
    ~ShardedNativeSource()override;
    std::span<const rad::NativeRecordDescriptor> catalog()const noexcept override{return descriptors_;}
    int acquire(std::span<const uint32_t>,rad::NativeRecordBatch*)override;
    // Setup only, before the owner/forward. This is not the routed NativeRecordSource catalog.
    int bootstrap(int32_t unit,std::span<std::byte> target);
    int finish_bootstrap();
    int close();
    sc::Stats stats()const noexcept{return store_->stats();}
    sc::ProcessUsage process_usage()const noexcept{return process_->usage();}
    sc::Accounting accounting()const noexcept{return store_->accounting();}
    AdapterUsage adapter_usage()const noexcept;
    sc::RankNativeStore& store_for_test(){return *store_;}
    sc::BoundRecord handle_for_test(size_t n)const{return handles_.at(n);}
    std::shared_ptr<sc::ProcessBudget> process_for_test()const{return process_;}
    size_t bootstrap_count()const noexcept{return bootstrap_.size();}
    std::span<const uint32_t> last_request()const noexcept{return last_;}
private:
    struct Quota;struct Holder;
    std::shared_ptr<sc::ProcessBudget> process_;
    std::unique_ptr<sc::RankNativeStore> store_;
    std::vector<rad::NativeRecordDescriptor> descriptors_;
    std::vector<sc::BoundRecord> handles_,request_;
    std::vector<uint32_t> ids_,last_;
    std::vector<std::pair<int32_t,sc::BoundRecord>> bootstrap_;
    std::shared_ptr<Quota> quota_;
    std::mutex operation_;
    bool bootstrap_done_=false,closed_=false;
    ShardedNativeSource();
};
int status(sc::Error)noexcept;
} // namespace glm53::engine_core_p3

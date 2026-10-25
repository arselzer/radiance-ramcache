#pragma once
#include "rad_core.h"
#include "cache_prepare4_coordinator.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
namespace rad {
// Device-free structural admission; backend/verified source capabilities remain
// separate gates. This does not assert loaded weights or permit a Native plan.
int glm_main_tp4_cache_program_profile(const Program&) noexcept;
class CachePrepare4Session;
class GlmMainTp4VerifiedSourceFactory; // no implementation/admission in this slice
class CachePrepare4RankBinding {
public:
    int rank()const noexcept{return rank_;}
    bool runtime_admitted()const noexcept;
    uint64_t model_generation()const noexcept;
    uint64_t source_generation()const noexcept;
    bool active()const noexcept;
    int check_batch(const RadBatch&)const noexcept;
    int enter(int layer,int64_t first,int64_t rows,std::span<const int32_t> ids);
    int ready(int status);
    int finish(const RadBatch&,int status);
    [[noreturn]] void fail_stop(int status)const noexcept;
private:
    friend class CachePrepare4Session;
    CachePrepare4Session* session_=nullptr;int rank_=-1;
    glm53::cache_prepare4::Key key_{};bool entered_=false;
};
class CachePrepare4ProtocolHarness;
class CachePrepare4Session final {
public:
    CachePrepare4Session(const CachePrepare4Session&)=delete;
    ~CachePrepare4Session();
    int begin(const RadBatch&); // scheduler-only while all rank workers are parked
    int complete_frame(const RadBatch&); // once, AFTER successful model+sampler/final collect
    void stop_admission()noexcept; // serialize idle close against begin BEFORE Engine worker abort
    bool admission_closed()const noexcept{return closed_.load(std::memory_order_acquire);}
    CachePrepare4RankBinding* binding(int rank)noexcept;
    bool runtime_admitted()const noexcept{return source_admitted_;}
    bool active()const noexcept{return active_.load(std::memory_order_acquire);}
    uint64_t model_generation()const noexcept{return model_;}
    uint64_t source_generation()const noexcept{return source_;}
    glm53::cache_prepare4::Snapshot observe()const{return coordinator_.observe();}
    [[noreturn]] void fail_stop(int rank,int status)noexcept;
    void stop_if_active()noexcept;
private:
    friend class CachePrepare4RankBinding;
    friend class CachePrepare4ProtocolHarness;
    friend class GlmMainTp4VerifiedSourceFactory;
    // No public numbers/boolean can confer source admission. The real factory
    // is deliberately absent until full four-rank catalogs/budgets are verified.
    CachePrepare4Session(uint64_t model,uint64_t source);
    glm53::cache_prepare4::Pass pass()const;
    const uint64_t model_,source_;
    bool source_admitted_=false;
    glm53::cache_prepare4::Coordinator coordinator_;
    mutable std::mutex pass_mutex_;
    glm53::cache_prepare4::Pass pass_{};
    uint64_t epoch_=0;
    std::atomic<bool> active_{false};
    std::atomic<bool> closed_{false};
    std::array<CachePrepare4RankBinding,4> bindings_{};
};
// HOST protocol/shape/error tests only. This object CANNOT grant runtime source
// admission; Ctx and Engine reject its bindings even if its CPU gates succeed.
class CachePrepare4ProtocolHarness {
public:
    static int create(uint64_t model,uint64_t source,std::unique_ptr<CachePrepare4Session>*)noexcept;
};
} // namespace rad

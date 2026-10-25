#pragma once
#include "rad_expert_prepare.h"
#include "device/device.h"
#include <cstdint>
#include <span>
#include <vector>
namespace rad {
struct ExpertRouteMap;
class Program;
class NativeInventory;
class Ctx;
struct ExpertPrepareLimits {
    int64_t max_rows=2048,max_experts=512,max_top_k=32;
    size_t max_maps=256,max_entries=65536;
    uint64_t max_readback_bytes=1u<<20,max_metadata_bytes=8u<<20;
};
struct ExpertPrepareView {
    const Program* program;
    const Ctx* context;
    const RadExpertRouteDecl& declaration;
    std::span<const RadExpertRouteEntry> map;
    int64_t step,total_rows,first_row,rows;
    uint64_t generation;
    RadStream producer_stream;
    RadTensor counts_source,offsets_source,sorted_source,ids_source;
    // Borrowed immutable HOST spans, valid only until prepare returns. Sorted values are
    // pass-relative token*top_k+slot, grouped by runtime expert, -1 after live count.
    std::span<const int32_t> counts,offsets,sorted,ids;
};
class ExpertPrepareOwner {
public:
    virtual ~ExpertPrepareOwner()=default;
    virtual int prepare(const ExpertPrepareView&,bool* residency_changed)=0;
    virtual int check_program(const Program&) const { return RAD_OK; }
    // Experimental TP4 only: actual checked upload completion, not enqueue or WTAB invalidation.
    virtual int complete_prepared(const ExpertPrepareView&) { return RAD_E_UNSUPPORTED; }
    virtual bool accepts_native_inventory(const NativeInventory*) const { return false; }
    virtual int bind_native_context(const NativeInventory*,const Ctx*,RadStream) { return RAD_E_UNSUPPORTED; }
    // Called after all compute lanes join, including preparation/issue failure. Future owners
    // must drain their own transfers before releasing leases. No cross-op rollback is supplied.
    virtual int end_pass(RadStream,int status){(void)status;return RAD_OK;}
};
struct ExpertPrepareState {
    ExpertPrepareOwner* owner=nullptr;
    ExpertPrepareLimits limits;
    std::vector<ExpertRouteMap> maps;
    std::vector<uint8_t> seen;
    void* allocation=nullptr;
    int32_t *host=nullptr,*device=nullptr;
    RadEvent ready=nullptr;
    uint64_t bytes=0,generation=0,copies=0,copied_bytes=0,callbacks=0;
    int fatal=RAD_OK;
    bool active_pass=false;
    ~ExpertPrepareState();
};
} // namespace rad

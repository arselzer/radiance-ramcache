/* Shared HOST/manual-AQL integration fixture. Actual Builder/Ctx/public buffer operands,
 * unmodified registered launch, guarded tiny arenas, independent raw-bit head scatter. */
#pragma once
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "device/device.h"
#include "../libds41/ds41_reorder_heads_fixture.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
namespace ds41_reorder_core {
using namespace rad;
namespace r=rad::ds41::reorder_heads;
namespace f=r::testing;
inline constexpr int64_t max_rows=5,fixed_rows=3,guard_bytes=256,arena_cap=65536;
inline constexpr uint8_t poison=0xBD;
inline void need(bool okay,const std::string& why){if(!okay)throw std::runtime_error(why);}
inline void okay(int status,const char* why){need(status>=0,std::string(why)+": "+rad_strerror(status));}
struct Stream {
    RadStream value=nullptr;
    Stream(){okay(rad_stream_create(&value,0),"create tiny core stream");need(value!=nullptr,"null core stream");}
    ~Stream(){if(value){static_cast<void>(rad_stream_sync(value));rad_stream_destroy(value);}}
};
struct Binding {
    rad_buf handle=0;int64_t bytes=0,offset=0,offset_elements=0,unit_elements=0;uint32_t dtype=0;void* base=nullptr;
};
struct Harness;
inline thread_local Harness* active=nullptr;
inline void step(RadCtx*,const RadBatch*);
struct Harness {
    bool host,flat;uint32_t dtype;int64_t tp,heads=2,dim=7,rows=0,arena_bytes=0;
    std::unique_ptr<Builder> builder;rad_op op=0;std::array<Binding,r::operand_count> bindings;
    std::array<RadOperand,r::operand_count> operands{};std::vector<RadLaunchFn> original_launches;Ctx ctx;
    Harness(Registry& registry,RadStream stream,uint32_t d,int64_t ranks,bool flattened,bool tape)
        :host(device_is_host()),flat(flattened),dtype(d),tp(ranks) {
        f::Fixture seed(dtype,tp,fixed_rows,heads,dim,flat);auto args=seed.args();
        RadModelMeta meta{};meta.arch_id="ds41-local-reorder-core-fixture";meta.name="tiny-synthetic";meta.quant="none";
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=max_rows;bc.max_seqs=1;bc.max_ctx=max_rows;bc.scope="";
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        for(int i=0;i<r::operand_count;++i) {
            auto& b=bindings[size_t(i)];const auto& source=args.t[i];RadBufDecl decl{};
            b.dtype=source.dtype;decl.dtype=source.dtype;decl.rank=source.rank;decl.kind=RAD_BUF_PERSIST;decl.domain=host?RAD_DOMAIN_HOST:RAD_DOMAIN_DEVICE;
            for(unsigned axis=0;axis<decl.rank;++axis)decl.shape[axis]=source.shape[axis];
            if(i<r::query_out&&flat)decl.shape[0]=tp*max_rows;
            if(i>=r::query_out)decl.shape[0]=max_rows;
            b.unit_elements=1;for(unsigned axis=1;axis<decl.rank;++axis)b.unit_elements*=decl.shape[axis];
            const int64_t width=rad_dtype_bits(decl.dtype)/8,row_bytes=b.unit_elements*width,
                          guard_rows=(guard_bytes+row_bytes-1)/row_bytes;
            b.offset_elements=guard_rows*b.unit_elements;b.offset=b.offset_elements*width;
            decl.shape[0]+=2*guard_rows;b.bytes=decl.shape[0]*row_bytes;
            need(b.bytes>0&&b.bytes<=16384,"core fixture tensor is not tiny");
            b.handle=rad_decl_buffer(builder.get(),("guarded.reorder."+std::to_string(i)).c_str(),&decl);need(b.handle!=0,"Builder refused real reorder buffer");
        }
        auto params=seed.params;params[0]=RAD_RANGE("M",1,max_rows);
        op=rad_decl_op(builder.get(),"ds41_reorder_heads",params.data(),params.size(),nullptr,0);need(op!=0,"actual reorder row did not resolve");
        const rad_buf inputs[]{bindings[r::query].handle,bindings[r::weights].handle},outputs[]{bindings[r::query_out].handle,bindings[r::weights_out].handle};
        okay(rad_op_reads(builder.get(),op,inputs,2),"declare ordinary input hazards");okay(rad_op_writes(builder.get(),op,outputs,2),"declare ordinary output hazards");
        okay(builder->finish(),"finish real reorder Builder");auto& program=builder->program();
        need(program.scratch_bytes==0,"scratchless reorder reserved scratch");
        const int domain=host?RAD_DOMAIN_HOST:RAD_DOMAIN_DEVICE;const char* plugin=host?"libds41_ref":"libds41";
        for(const auto& band:program.ops[op].bands) {
            const auto& resolved=band.dom[domain];need(resolved.row&&resolved.row->info&&resolved.row->plugin==plugin&&resolved.row->info->launch,
                "actual unmodified reorder row missing");
            need(!resolved.row->deferred_status&&!resolved.row->collective&&!resolved.row->releases_link&&!resolved.row->info->scratch,
                "local reorder unexpectedly has status/scratch/collective hooks");original_launches.push_back(resolved.row->info->launch);
        }
        arena_bytes=arena_plan_bytes(program,nullptr,nullptr)+align_up(arena_host_bytes(program),RAD_ALIGN_UNIT);
        need(arena_bytes>0&&arena_bytes<=arena_cap,"core reorder arena exceeded tiny cap");
        CtxDesc cd{};cd.program=&program;cd.rank=0;cd.world_size=1;cd.device=0;cd.stream=stream;cd.arch_step=step;cd.record_passes=tape;
        okay(ctx.init(cd),"init actual reorder Ctx");okay(ctx.bind_thread(),"bind actual reorder thread");
        need(!ctx.checks_device_status(),"reorder enabled deferred status");need(ctx.records_passes()==tape,"unexpected replay eligibility");
        for(auto& b:bindings){b.base=rad_buf_ptr(&ctx,b.handle);need(b.base!=nullptr,"actual reorder arena pointer missing");}
    }
    void pass(int64_t actual_rows,int ordinal) {
        need(actual_rows>0&&actual_rows<=max_rows&&(flat||actual_rows==fixed_rows),"invalid core fixture row profile");rows=actual_rows;
        f::Fixture current(dtype,tp,rows,heads,dim,flat);
        for(int input:{r::query,r::weights}) {
            auto& b=current.buffers[size_t(input)];auto* bytes=static_cast<uint8_t*>(b.data());
            for(int64_t i=0;i<b.bytes;++i)bytes[i]^=uint8_t(unsigned(ordinal)*37u+unsigned(i)*13u);
        }
        const auto expected=current.expected();std::array<std::vector<uint8_t>,r::operand_count> before,after;
        okay(rad_stream_sync(ctx.stream()),"drain before changing input bits");
        for(int i=0;i<r::operand_count;++i) {
            const auto& b=bindings[size_t(i)];auto& bytes=before[size_t(i)];bytes.assign(size_t(b.bytes),poison);
            const auto& payload=current.buffers[size_t(i)];need(b.offset+payload.bytes<=b.bytes-guard_bytes,"fixture slice overran tail guard");
            std::memcpy(bytes.data()+b.offset,payload.data(),size_t(payload.bytes));after[size_t(i)]=bytes;
            if(i>=r::query_out)std::memcpy(after[size_t(i)].data()+b.offset,expected[size_t(i)].data(),size_t(payload.bytes));
            if(host)std::memcpy(b.base,bytes.data(),bytes.size());else okay(rad_memcpy_async(b.base,bytes.data(),b.bytes,ctx.stream()),"upload complete guarded core buffer");
            const int64_t first=i<r::query_out?(flat?tp*rows:tp):rows;
            operands[size_t(i)]=RAD_B_N(b.handle,b.offset_elements,first);
        }
        if(host)rad_dev_host_wrote();
        active=this;RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=rows;batch.step=ordinal;
        const int status=ctx.run_step(&batch);active=nullptr;
        need(status>=0,std::string("actual public buffer issue failed: ")+ctx.step_error());okay(ctx.await_device_status(),"scratchless status await");
        okay(rad_stream_sync(ctx.stream()),"finish actual reorder pass");
        for(int i=0;i<r::operand_count;++i) {
            const auto& b=bindings[size_t(i)];std::vector<uint8_t> got(size_t(b.bytes),0);
            if(host)std::memcpy(got.data(),b.base,got.size());else{okay(rad_memcpy_async(got.data(),b.base,b.bytes,ctx.stream()),"read complete guarded core buffer");okay(rad_stream_sync(ctx.stream()),"finish exact-bit readback");}
            need(got==after[size_t(i)],"actual core reorder changed source/guard or produced wrong raw output bits");
        }
        size_t index=0;for(const auto& band:builder->program().ops[op].bands)
            need(band.dom[host?RAD_DOMAIN_HOST:RAD_DOMAIN_DEVICE].row->info->launch==original_launches[index++],"core checker replaced actual launch");
    }
};
inline void step(RadCtx* context,const RadBatch*) {
    if(active)(void)rad_issue(context,active->op,active->operands.data(),r::operand_count,active->rows);
    else rad_step_fail(context,"missing local reorder fixture owner");
}
} // namespace ds41_reorder_core

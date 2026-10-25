#include "glm_main_ordinary_loader.h"
#include "format/checkpoint.h"
#include "format/share.h"
#include "mem/pools.h"
#include "place/mover.h"
#include "place/native_inventory.h"
#include "runtime/cache_prepare4_session.h"
#include "rad_plugin.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <vector>

namespace rad {
namespace {
struct Failure {int code;const char* why;};
void need(bool value,int code,const char* why){if(!value)throw Failure{code,why};}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,RAD_E_SHAPE,"loader addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,RAD_E_SHAPE,"loader product overflow");return a*b;}
uint64_t align64(uint64_t a){return add(a,63)&~uint64_t(63);}
uint64_t page(uint64_t a){return a?add(a,4095)&~uint64_t(4095):0;}
void explanation(std::string* out,const char* text)noexcept{if(out)try{*out=text;}catch(...){}}
bool span_fits(const void* p,uint64_t bytes,const void* base,uint64_t extent)noexcept{
    if(!p||!base||!bytes||bytes>UINTPTR_MAX-uintptr_t(p)||extent>UINTPTR_MAX-uintptr_t(base))return false;
    return uintptr_t(p)>=uintptr_t(base)&&uintptr_t(p)+bytes<=uintptr_t(base)+extent;
}
struct Job {
    uint32_t rank=0;rad_weight weight=0;
    const CkptWeight* source=nullptr;
    std::array<PlaneShare,RAD_ENC_MAX_PLANES> shares;
    std::array<uint64_t,RAD_ENC_MAX_PLANES> offsets{};
    uint64_t input=0,source_bytes=0;
    int32_t lay_op=-1,lay_band=-1,lay_dom=-1,lay_operand=-1;
    const KernelRow* layout_row=nullptr;
};
struct Pinned {
    void* allocation=nullptr;uint8_t* host=nullptr;void* device=nullptr;uint64_t bytes=0;
    ~Pinned(){if(allocation)rad_dev_free(allocation,RAD_MEM_HOST_PINNED);}
    int open(uint64_t n){
        if(!n)return RAD_OK;
        if(n>uint64_t(INT64_MAX)||n>SIZE_MAX)return RAD_E_SHAPE;
        allocation=rad_dev_alloc(int64_t(n),RAD_MEM_HOST_PINNED);
        if(!allocation)return RAD_E_NOMEM;
        host=static_cast<uint8_t*>(rad_dev_host_ptr(allocation));device=rad_dev_device_ptr(allocation);bytes=n;
        if(!host||!device||uintptr_t(host)%4096||uintptr_t(device)%4096||
           n>UINTPTR_MAX-uintptr_t(host)||n>UINTPTR_MAX-uintptr_t(device))return RAD_E_ALIGN;
        return RAD_OK;
    }
};
void source_bounds(const CkptWeight& source){
    need(source.rank<=RAD_MAX_RANK&&source.rows>0&&source.cols>0&&
         uint64_t(source.rows)<=uint64_t(INT64_MAX)/128/uint64_t(source.cols)&&
         source.pieces.size()<=RAD_MAX_ROW_PARTS&&source.scales.size()<=RAD_MAX_ROW_PARTS&&
         !source.pieces.empty()&&rad_enc_valid(&source.enc),RAD_E_SHAPE,"bounded checkpoint source geometry");
    uint64_t elements=1;
    for(uint32_t d=0;d<source.rank;++d){need(source.shape[d]>0,RAD_E_SHAPE,"nonpositive source shape");elements=mul(elements,uint64_t(source.shape[d]));}
    need(elements==mul(uint64_t(source.rows),uint64_t(source.cols)),RAD_E_SHAPE,"source logical view changed");
}
void ordinary_source(const CkptWeight& source){
    // These excluded import families allocate band-sized hidden planes or perform
    // unrelated transforms. Actual pinned GLM ordinary sources use this native
    // ckpt_plane_rows path; refusal never invents a new conversion.
    need((source.trivial||source.imported)&&!source.block_importer&&!source.paro&&!source.wnorm&&source.derived.empty(),
         RAD_E_UNSUPPORTED,"ordinary source needs unsupported materialization");
    auto physical=[](const SrcTensor* t){
        need(t&&t->data&&t->bytes>0&&uint64_t(t->bytes)<=UINTPTR_MAX-uintptr_t(t->data)&&
             t->shape.size()<=RAD_MAX_RANK&&!t->gptq_group&&!t->ct_group&&!t->pbits,
             RAD_E_UNSUPPORTED,"ordinary source is not bounded native tensor storage");
        uint64_t total=1;for(const auto n:t->shape){need(n>0,RAD_E_SHAPE,"physical source shape");total=mul(total,uint64_t(n));}
        need(total<=uint64_t(INT64_MAX)/128,RAD_E_SHAPE,"physical source product cap");
    };
    for(const auto* t:source.pieces)physical(t);
    for(const auto* t:source.scales)physical(t);
}
void layout_fields(const Program& program,const WeightInfo& w,Job* job){
    need(w.n_sel>0&&w.n_sel<=RAD_ENC_MAX_PLANES&&w.decl.n_row_parts>=0&&
         w.decl.n_row_parts<=RAD_MAX_ROW_PARTS&&w.n_part_spans>=0&&w.n_part_spans<=RAD_MAX_ROW_PARTS&&
         w.stored_bytes>0&&w.align>0&&w.align<=4096&&4096%w.align==0,
         RAD_E_SHAPE,"bounded selected/stored weight geometry/alignment");
    for(int k=0;k<w.n_sel;++k)need(w.sel[k]>=0&&w.sel[k]<w.enc.n_planes&&w.sel_rows[k]>0&&w.sel_cols[k]>0,
         RAD_E_SHAPE,"selected plane outside source encoding");
    if(w.identity&&!w.widen)need(w.n_sel==1,RAD_E_SHAPE,"identity stored weight needs one selected plane");
    if(w.widen){
        need(w.n_sel==1&&w.stored_dtype==RAD_F32&&
             (w.enc.plane[w.sel[0]].dtype==RAD_BF16||w.enc.plane[w.sel[0]].dtype==RAD_F16)&&
             mul(mul(uint64_t(w.sel_rows[0]),uint64_t(w.sel_cols[0])),4)==uint64_t(w.stored_bytes),
             RAD_E_SHAPE,"exact F16/BF16 widening geometry");
    }else if(!w.identity){
        need(w.lay_op>0&&size_t(w.lay_op)<program.ops.size()&&w.lay_band>=0&&
             size_t(w.lay_band)<program.ops[size_t(w.lay_op)].bands.size()&&w.lay_dom>=0&&w.lay_dom<RAD_N_DOMAINS&&w.lay_operand>=0,
             RAD_E_SHAPE,"resolved ordinary relayout position");
        const auto& resolved=program.ops[size_t(w.lay_op)].bands[size_t(w.lay_band)].dom[w.lay_dom];
        need(resolved.row&&resolved.row->info&&resolved.row->info->relayout,RAD_E_UNSUPPORTED,"ordinary selected relayout missing");
        job->lay_op=w.lay_op;job->lay_band=w.lay_band;job->lay_dom=w.lay_dom;job->lay_operand=w.lay_operand;
        job->layout_row=resolved.row;
    }
}
uint64_t checked_share_bytes(const PlaneShare& share){
    need(share.rows>0&&share.cols>0&&share.row_bytes>0&&share.rects.size()&&
         share.rects.size()<=RAD_MAX_ROW_PARTS,RAD_E_SHAPE,"bounded source share");
    const int64_t rb=rad_enc_row_bytes(share.dtype,share.cols);
    need(rb>0,RAD_E_SHAPE,"source share row bytes");
    const auto dense=mul(uint64_t(share.rows),uint64_t(rb));uint64_t actual=0;
    for(const auto& r:share.rects){
        need(r.row0>=0&&r.rows>0&&r.byte0>=0&&r.bytes>0&&r.byte0<=share.row_bytes&&
             r.bytes<=share.row_bytes-r.byte0,RAD_E_SHAPE,"source share rectangle bounds");
        actual=add(actual,mul(uint64_t(r.rows),uint64_t(r.bytes)));
    }
    need(dense==actual&&dense<=uint64_t(INT64_MAX),RAD_E_SHAPE,"source share dense extent");return dense;
}
// Same row-selection/copy equations as Engine's fill_from_checkpoint. The one
// difference in storage is caller-owned reusable column scratch, with no resize
// or unbounded fallback in this loop. Native ckpt_plane_rows math is unchanged.
int fill(const Job& job,const WeightInfo& w,uint8_t* input,std::span<uint8_t> column){
    const auto& source=*job.source;std::string why;
    for(int k=0;k<w.n_sel;++k){
        const auto& share=job.shares[size_t(k)];auto* at=input+job.offsets[size_t(k)];
        int64_t pr=0,pc=0;rad_enc_plane_dims(&source.enc.plane[w.sel[k]],source.rows,source.cols,&pr,&pc);
        const int64_t crb=rad_enc_row_bytes(source.enc.plane[w.sel[k]].dtype,pc);
        if(crb<=0)return RAD_E_SHAPE;
        for(const auto& r:share.rects){
            if(r.byte0==0&&r.bytes==share.row_bytes){
                const uint64_t offset=mul(uint64_t(r.row0),uint64_t(share.row_bytes));
                const uint64_t bytes=mul(uint64_t(r.rows),uint64_t(share.row_bytes));
                if(offset%uint64_t(crb)||bytes%uint64_t(crb)||offset/uint64_t(crb)>uint64_t(pr)||
                   bytes/uint64_t(crb)>uint64_t(pr)-offset/uint64_t(crb))return RAD_E_SHAPE;
                const int s=ckpt_plane_rows(source,w.sel[k],int64_t(offset/uint64_t(crb)),int64_t(bytes/uint64_t(crb)),at,&why);
                if(s<0)return s;
                at+=bytes;continue;
            }
            const int64_t step=std::max<int64_t>(1,(64ll<<20)/std::max<int64_t>(crb,1));
            for(int64_t r0=0;r0<r.rows;r0+=step){
                const int64_t nr=std::min(step,r.rows-r0);const auto bytes=mul(uint64_t(nr),uint64_t(crb));
                if(bytes>column.size())return RAD_E_SCRATCH;
                const int s=ckpt_plane_rows(source,w.sel[k],r.row0+r0,nr,column.data(),&why);
                if(s<0)return s;
                for(int64_t row=0;row<nr;++row){std::memcpy(at,column.data()+row*crb+r.byte0,size_t(r.bytes));at+=r.bytes;}
            }
        }
    }
    return RAD_OK;
}
}
struct GlmMainOrdinaryLoader::Impl {
    std::array<GlmMainOrdinaryRankSpec,4> ranks;
    std::array<std::shared_ptr<const NativeInventory>,4> inventories;
    GlmMainOrdinaryLoaderQuote quote;
    std::vector<Job> jobs;
    std::atomic<bool> attempted{false};
};
GlmMainOrdinaryLoader::GlmMainOrdinaryLoader(std::unique_ptr<Impl> p):p_(std::move(p)){}
GlmMainOrdinaryLoader::~GlmMainOrdinaryLoader()=default;
const GlmMainOrdinaryLoaderQuote& GlmMainOrdinaryLoader::quote()const noexcept{return p_->quote;}

int GlmMainOrdinaryLoader::prepare(std::span<const GlmMainOrdinaryRankSpec> ranks,
    const GlmMainOrdinaryLoaderLimits& limits,std::unique_ptr<GlmMainOrdinaryLoader>* out,
    GlmMainOrdinaryLoaderQuote* quote,std::string* why)noexcept{
    try{
        need(out&&!*out&&quote&&ranks.size()==4&&limits.max_jobs&&limits.max_jobs<=16384&&
             limits.max_weights_per_rank&&limits.max_weights_per_rank<=100000&&limits.max_metadata_bytes&&limits.max_metadata_bytes<=(128u<<20)&&
             limits.max_input_bytes&&limits.max_input_bytes<=(2ull<<30)&&limits.max_output_bytes<=(2ull<<30)&&
             limits.max_column_bytes<=(64u<<20)&&limits.max_buffer_bytes&&limits.max_buffer_bytes<=(4ull<<30)+(64u<<20),
             RAD_E_INVAL,"ordinary loader request/caps");
        size_t candidates=0;
        for(size_t rank=0;rank<4;++rank){const auto& r=ranks[rank];
            need(r.program&&r.plan&&r.sources&&r.plan->native_inventory&&
                 r.plan->strategy==Strategy::NativeBacked&&!r.plan->dynamic&&
                 r.plan->rank==int(rank)&&r.plan->world_size==4&&r.plan->weight.size()==r.program->weights.size()&&
                 r.program->ctx.rank==int(rank)&&r.program->ctx.world_size==4&&r.program->ctx.max_tok==1&&
                 r.program->ctx.max_seqs==1&&!r.program->ctx.max_spec&&r.program->weights.size()<=limits.max_weights_per_rank&&
                 r.plan->native_inventory->program()==r.program&&
                 r.plan->native_inventory->limits().backend_profile==NativeBackendProfile::GlmMainTp4AqlFailStopV1,
                 RAD_E_UNSUPPORTED,"actual four-rank Native plan/source identity");
            need(r.plan->native_inventory->validate(*r.program,*r.plan)>=0,RAD_E_STATE,"ordinary loader initial authority validation");
            for(size_t w=1;w<r.program->weights.size();++w)
                if(!r.plan->native_inventory->find(r.plan->native_inventory->unit_for_weight(rad_weight(w)))){
                    need(candidates<limits.max_jobs,RAD_E_SCRATCH,"ordinary loader job count cap");++candidates;
                }
        }
        // Maximum per-job rectangle ownership and one temporary weight_share
        // view are admitted before constructing any retained job/share arrays.
        const uint64_t each=add(sizeof(Job),mul(RAD_ENC_MAX_PLANES*RAD_MAX_ROW_PARTS,sizeof(PlaneShare::Rect)));
        const uint64_t metadata=add(16384,add(sizeof(Impl)+sizeof(GlmMainOrdinaryLoader),mul(candidates,each)));
        need(metadata<=limits.max_metadata_bytes,RAD_E_SCRATCH,"ordinary loader metadata reservation");
        auto p=std::make_unique<Impl>();p->quote.metadata_requested=metadata;p->quote.source_fill_scratch_upper=4096;
        p->jobs.reserve(candidates);
        for(size_t rank=0;rank<4;++rank){p->ranks[rank]=ranks[rank];p->inventories[rank]=ranks[rank].plan->native_inventory;
            const auto& r=ranks[rank];
            for(size_t wi=1;wi<r.program->weights.size();++wi){const auto& w=r.program->weights[wi];
                const auto* source=r.sources->find(w.source);
                if(!source){need(w.decl.optional&&!p->inventories[rank]->find(p->inventories[rank]->unit_for_weight(rad_weight(wi))),
                    RAD_E_NOTFOUND,"ordinary loader required/native source absent");++p->quote.optional_missing;continue;}
                source_bounds(*source);
                need(w.enc_known&&rad_enc_equal(&source->enc,&w.enc),RAD_E_FORMAT,"ordinary loader actual source encoding differs");
                Job job;job.rank=uint32_t(rank);job.weight=rad_weight(wi);job.source=source;layout_fields(*r.program,w,&job);
                std::string error;
                const int shared=weight_share(w,source->enc,source->rank,source->shape,4,int(rank),job.shares.data(),&error);
                need(shared>=0,shared,"ordinary loader actual TP plane share refused");
                for(int k=0;k<w.n_sel;++k){const auto bytes=checked_share_bytes(job.shares[size_t(k)]);
                    job.offsets[size_t(k)]=job.input;job.input=add(job.input,align64(bytes));job.source_bytes=add(job.source_bytes,bytes);}
                const auto unit=p->inventories[rank]->unit_for_weight(rad_weight(wi));
                if(p->inventories[rank]->find(unit)){
                    need(p->inventories[rank]->validate_unit(*r.program,*r.plan,unit)>=0,RAD_E_STATE,"typed native omission authority differs");
                    ++p->quote.omitted_views;p->quote.omitted_source_bytes=add(p->quote.omitted_source_bytes,job.source_bytes);continue;
                }
                // Source/share validation ABOVE precedes this typed omission and
                // any destination/publication below (the stock loader order).
                ordinary_source(*source);
                need(r.plan->weight[wi].tier==Tier::VRAM&&r.plan->weight[wi].site==Site::Device,
                     RAD_E_UNSUPPORTED,"ordinary loader needs fixed DEVICE resident destination");
                if(w.identity&&!w.widen)need(checked_share_bytes(job.shares[0])==uint64_t(w.stored_bytes),RAD_E_FORMAT,"ordinary identity share/stored extent differs");
                uint64_t column=0;
                for(int k=0;k<w.n_sel;++k){int64_t pr=0,pc=0;rad_enc_plane_dims(&source->enc.plane[w.sel[k]],source->rows,source->cols,&pr,&pc);
                    const auto crb=rad_enc_row_bytes(source->enc.plane[w.sel[k]].dtype,pc);need(crb>0,RAD_E_SHAPE,"ordinary physical row bytes");
                    for(const auto& cut:job.shares[size_t(k)].rects)if(cut.byte0||cut.bytes!=job.shares[size_t(k)].row_bytes){
                        need(uint64_t(crb)<=limits.max_column_bytes&&cut.row0<=pr&&cut.rows<=pr-cut.row0&&cut.byte0<=crb&&cut.bytes<=crb-cut.byte0,
                             RAD_E_SCRATCH,"ordinary column scratch/physical rectangle cap");
                        const auto rows=std::min<int64_t>(cut.rows,std::max<int64_t>(1,(64ll<<20)/crb));
                        column=std::max(column,mul(uint64_t(rows),uint64_t(crb)));
                    }
                }
                p->quote.input_bytes=std::max(p->quote.input_bytes,job.input);
                if(!w.identity||w.widen)p->quote.output_bytes=std::max(p->quote.output_bytes,uint64_t(w.stored_bytes));
                p->quote.column_bytes=std::max(p->quote.column_bytes,column);
                p->quote.ordinary_source_bytes=add(p->quote.ordinary_source_bytes,job.source_bytes);
                p->quote.ordinary_stored_bytes=add(p->quote.ordinary_stored_bytes,uint64_t(w.stored_bytes));
                if(w.widen)++p->quote.widen_jobs;else if(w.identity)++p->quote.identity_jobs;else ++p->quote.relayout_jobs;
                p->jobs.push_back(std::move(job));
            }
        }
        p->quote.jobs=p->jobs.size();p->quote.pinned_page_bytes=add(page(p->quote.input_bytes),page(p->quote.output_bytes));
        p->quote.buffer_bytes=add(p->quote.pinned_page_bytes,p->quote.column_bytes);
        need(p->quote.jobs&&p->quote.input_bytes<=limits.max_input_bytes&&p->quote.output_bytes<=limits.max_output_bytes&&
             p->quote.column_bytes<=limits.max_column_bytes&&p->quote.buffer_bytes<=limits.max_buffer_bytes,
             RAD_E_SCRATCH,"ordinary loader actual input/output/column buffer cap");
        auto result=std::unique_ptr<GlmMainOrdinaryLoader>(new GlmMainOrdinaryLoader(std::move(p)));
        for(size_t rank=0;rank<4;++rank)need(result->p_->inventories[rank]->validate(*ranks[rank].program,*ranks[rank].plan)>=0,
             RAD_E_STATE,"ordinary loader final authority validation");
        *quote=result->p_->quote;*out=std::move(result);return RAD_OK;
    }catch(const Failure& e){explanation(why,e.why);return e.code;}
    catch(const std::bad_alloc&){explanation(why,"ordinary loader metadata allocation refused");return RAD_E_NOMEM;}
    catch(...){explanation(why,"ordinary loader unexpected prepare failure");return RAD_E_STATE;}
}

int GlmMainOrdinaryLoader::execute(std::span<const GlmMainOrdinaryRuntimeRank> runtime,
    GlmMainOrdinaryLoaderReceipt* out,std::string* why)noexcept{
    try{
        need(out&&runtime.size()==4,RAD_E_INVAL,"ordinary runtime ranks/output");
        bool expected=false;need(p_->attempted.compare_exchange_strong(expected,true),RAD_E_STATE,"ordinary loader already attempted");
        const auto generation=p_->inventories[0]->source_generation();uint64_t model=0;
        for(size_t rank=0;rank<4;++rank){const auto& r=runtime[rank];const auto& spec=p_->ranks[rank];
            need(r.mover&&r.pools&&r.compute&&r.binding&&r.device>=0&&r.binding->runtime_admitted()&&r.binding->rank()==int(rank)&&
                 !r.binding->active()&&r.binding->source_generation()==generation&&r.binding->model_generation()&&
                 r.pools->native_inventory()==p_->inventories[rank].get()&&
                 r.mover->record_plan_matches(*spec.program,*spec.plan)&&r.mover->native_record_compute_stream()==r.compute&&
                 p_->inventories[rank]->source_generation()==generation,
                 RAD_E_UNSUPPORTED,"ordinary runtime needs actual admitted four-rank owners/streams");
            if(rank)need(r.binding->model_generation()==model,RAD_E_STATE,"ordinary runtime Engine generation differs");else model=r.binding->model_generation();
            need(p_->inventories[rank]->validate(*spec.program,*spec.plan)>=0,RAD_E_STATE,"ordinary runtime authority changed");
        }
        need(!device_is_host()&&!std::strcmp(device_backend_name(),"aql"),RAD_E_UNSUPPORTED,"ordinary runtime is the actual AQL-only profile");
        // Validate ALL destinations/source identities before any payload read or
        // upload. Never allocate a destination or silently use a selected sentinel.
        for(const auto& job:p_->jobs){const auto& r=runtime[job.rank];const auto& spec=p_->ranks[job.rank];const auto& w=spec.program->weights[job.weight];
            const auto* slot=r.mover->residency()->lookup(job.weight);
            need(slot&&slot->ptr&&!slot->ready&&w.stored_bytes>0&&w.align>0&&uintptr_t(slot->ptr)%uint64_t(w.align)==0&&
                 r.pools->weights().bytes()>0&&
                 span_fits(slot->ptr,uint64_t(w.stored_bytes),r.pools->weights().base(),uint64_t(r.pools->weights().bytes()))&&
                 spec.sources->find(w.source)==job.source&&rad_enc_equal(&job.source->enc,&w.enc),RAD_E_STATE,"ordinary destination/source changed");
            if(!w.identity&&!w.widen)need(w.lay_op==job.lay_op&&w.lay_band==job.lay_band&&w.lay_dom==job.lay_dom&&w.lay_operand==job.lay_operand&&
                 spec.program->ops[size_t(job.lay_op)].bands[size_t(job.lay_band)].dom[job.lay_dom].row==job.layout_row,
                 RAD_E_STATE,"ordinary selected relayout changed");
        }
        const int bound=rad_dev_set(runtime[0].device);if(bound<0)runtime[0].binding->fail_stop(bound);
        Pinned input,output;int allocated=input.open(p_->quote.input_bytes);if(allocated<0)return allocated;
        allocated=output.open(p_->quote.output_bytes);if(allocated<0)return allocated;
        std::vector<uint8_t> column(size_t(p_->quote.column_bytes));
        GlmMainOrdinaryLoaderReceipt receipt;receipt.quote=p_->quote;
        for(const auto& job:p_->jobs){const auto& r=runtime[job.rank];const auto& spec=p_->ranks[job.rank];auto& w=spec.program->weights[job.weight];
            const int selected=rad_dev_set(r.device);if(selected<0)r.binding->fail_stop(selected);
            std::memset(input.host,0,size_t(job.input));
            const int read=fill(job,w,input.host,column);if(read<0){explanation(why,"actual ordinary checkpoint plane fill refused");return read;}
            const void* upload=input.device;
            if(w.widen){
                const auto n=mul(uint64_t(w.sel_rows[0]),uint64_t(w.sel_cols[0]));const auto dtype=w.enc.plane[w.sel[0]].dtype;
                for(uint64_t i=0;i<n;++i)reinterpret_cast<float*>(output.host)[i]=rad_load_f32(input.host+job.offsets[0],dtype,int64_t(i));
                upload=output.device;
            }else if(!w.identity){
                RadTensor planes[RAD_ENC_MAX_PLANES]{};weight_planes(w,planes);
                for(int k=0;k<w.n_sel;++k)planes[k].data=input.host+job.offsets[size_t(k)];
                const auto& resolved=spec.program->ops[size_t(job.lay_op)].bands[size_t(job.lay_band)].dom[job.lay_dom];
                const int layout=resolved.row->info->relayout(resolved.geom.params(),resolved.geom.n_params(),job.lay_operand,
                    &w.enc,w.sel,planes,w.n_sel,output.host,w.stored_bytes);
                if(layout<0){explanation(why,"actual selected ordinary relayout refused");return layout;}
                upload=output.device;
            }
            const auto* slot=r.mover->residency()->lookup(job.weight);
            rad_dev_host_wrote();
            const int copied=rad_memcpy_async(slot->ptr,upload,w.stored_bytes,r.compute);
            if(copied<0)r.binding->fail_stop(copied); // may have enqueued: NO unwind/free
            ++receipt.completed_uploads;
            const int drained=rad_stream_sync(r.compute);
            if(drained<0)r.binding->fail_stop(drained); // failure never proves retirement
            ++receipt.successful_drains;++receipt.completed_jobs;
            receipt.source_bytes=add(receipt.source_bytes,job.source_bytes);
            receipt.upload_bytes=add(receipt.upload_bytes,uint64_t(w.stored_bytes));
            const auto& placement=spec.plan->weight[job.weight];
            w.ptr=slot->ptr;w.host_ptr=nullptr;w.tier=placement.tier;w.site=placement.site;w.slab_slot=placement.slab_slot;w.file_offset=0;
        }
        for(size_t rank=0;rank<4;++rank){const auto& spec=p_->ranks[rank];
            for(size_t wi=1;wi<spec.program->weights.size();++wi)if(p_->inventories[rank]->find(p_->inventories[rank]->unit_for_weight(rad_weight(wi)))){
                auto& w=spec.program->weights[wi];const auto& place=spec.plan->weight[wi];w.tier=place.tier;w.site=place.site;w.slab_slot=place.slab_slot;w.file_offset=0;
            }
        }
        *out=receipt;return RAD_OK;
    }catch(const Failure& e){explanation(why,e.why);return e.code;}
    catch(const std::bad_alloc&){explanation(why,"ordinary loader bounded storage allocation refused");return RAD_E_NOMEM;}
    catch(...){explanation(why,"ordinary loader unexpected execution failure");return RAD_E_STATE;}
}
} // namespace rad

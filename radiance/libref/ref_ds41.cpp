/* DeepSeek1017 semantic host kernels. Activation/cache quantization is deliberately outside
 * these FP32 oracles; future optimized kernels must match the declared numerical contract. */
#include "ref_common.h"
#include "ref_ops.h"
#include <arch/rad_ds41_ref.h>
namespace ds = rad::deepseek41::ref;
namespace {
bool tensor(const RadTensor& t,uint32_t dtype,std::initializer_list<int64_t> shape) {
    if(!t.data || t.dtype!=dtype || t.rank!=shape.size())return false;
    size_t index=0;
    for(auto extent:shape)if(extent<=0 || t.shape[index++]!=extent)return false;
    return true;
}
std::vector<float> read(const RadTensor& t) {
    std::vector<float> out(size_t(ref::numel(&t)));
    for(size_t i=0;i<out.size();++i)out[i]=ref::ldt(&t,ref::offlin(&t,int64_t(i)));
    return out;
}
void write(const RadTensor& t,ds::Floats values) {
    for(size_t i=0;i<values.size();++i)ref::stt(&t,ref::offlin(&t,int64_t(i)),values[i]);
}
uint32_t activation_dtype(const RadArgs* a) {
    const auto dt=rad_dtype_parse(ref::p_str(a,"dtype","f32"));
    return dt==RAD_F32 || dt==RAD_F16 || dt==RAD_BF16 ? dt : uint32_t(RAD_DT_INVALID);
}
}
extern "C" int ref_ds41_hc_prepare(const RadArgs* a,RadStream) {
    if(!ref::have(a,7))return RAD_E_INVAL;
    const auto dt=activation_dtype(a);if(dt==RAD_DT_INVALID)return RAD_E_DTYPE;
    const int64_t n=ref::p_int(a,"n",0),hc=ref::p_int(a,"hc",0),iters=ref::p_int(a,"iterations",0);
    if(n<=0 || hc<2 || hc>8 || n>INT64_MAX/(hc*hc*(hc+2)) || iters<1 || iters>256 || a->t[0].rank!=2)return RAD_E_SHAPE;
    const int64_t rows=ref::band_rows(a->t[0].shape[0],ref::p_int(a,"M",0)),width=hc*n,mixes=hc*(hc+2);
    if(!tensor(a->t[0],dt,{rows,width}) || !tensor(a->t[1],RAD_F32,{mixes,width}) ||
       !tensor(a->t[2],RAD_F32,{3}) || !tensor(a->t[3],RAD_F32,{mixes}) ||
       !tensor(a->t[4],RAD_F32,{rows,hc}) || !tensor(a->t[5],RAD_F32,{rows,hc}) ||
       !tensor(a->t[6],RAD_F32,{rows,hc*hc}))return RAD_E_SHAPE;
    const auto h=read(a->t[0]),fn=read(a->t[1]),scale=read(a->t[2]),base=read(a->t[3]);
    std::vector<float> pre(rows*hc),post(rows*hc),comb(rows*hc*hc);
    for(int64_t row=0;row<rows;++row) {
        ds::MhcMix mix;
        const int status=ds::mhc_prepare(ds::Floats(h).subspan(row*width,width),n,hc,fn,scale,base,int(iters),
                             ref::p_f32(a,"eps",0),ref::p_f32(a,"hc_eps",0),&mix);
        if(status!=RAD_OK)return status;
        std::copy(mix.pre.begin(),mix.pre.end(),pre.begin()+row*hc);
        std::copy(mix.post.begin(),mix.post.end(),post.begin()+row*hc);
        std::copy(mix.comb.begin(),mix.comb.end(),comb.begin()+row*hc*hc);
    }
    write(a->t[4],pre);write(a->t[5],post);write(a->t[6],comb);return RAD_OK;
}
extern "C" int ref_ds41_hc_collapse(const RadArgs* a,RadStream) {
    if(!ref::have(a,3))return RAD_E_INVAL;
    const auto dt=activation_dtype(a);if(dt==RAD_DT_INVALID)return RAD_E_DTYPE;
    const int64_t n=ref::p_int(a,"n",0),hc=ref::p_int(a,"hc",0);
    if(n<=0 || hc<2 || hc>8 || n>INT64_MAX/hc || a->t[0].rank!=2)return RAD_E_SHAPE;
    const int64_t rows=ref::band_rows(a->t[0].shape[0],ref::p_int(a,"M",0));
    if(!tensor(a->t[0],dt,{rows,hc*n}) || !tensor(a->t[1],RAD_F32,{rows,hc}) || !tensor(a->t[2],dt,{rows,n}))return RAD_E_SHAPE;
    const auto h=read(a->t[0]),pre=read(a->t[1]);std::vector<float> out(rows*n);
    for(int64_t row=0;row<rows;++row) {
        const int status=ds::mhc_collapse(ds::Floats(h).subspan(row*hc*n,hc*n),ds::Floats(pre).subspan(row*hc,hc),ds::Mutable(out).subspan(row*n,n));
        if(status!=RAD_OK)return status;
    }
    write(a->t[2],out);return RAD_OK;
}
extern "C" int ref_ds41_router_topk(const RadArgs* a,RadStream) {
    if(!ref::have(a,4))return RAD_E_INVAL;
    const int64_t experts=ref::p_int(a,"experts",0),topk=ref::p_int(a,"topk",0),normalize=ref::p_int(a,"normalize",1);
    if(experts<=0 || experts>INT32_MAX || topk<=0 || topk>experts || normalize<0 || normalize>1 || a->t[0].rank!=2)return RAD_E_SHAPE;
    const int64_t rows=ref::band_rows(a->t[0].shape[0],ref::p_int(a,"M",0));
    if(!tensor(a->t[0],RAD_F32,{rows,experts}) || !tensor(a->t[1],RAD_F32,{experts}) ||
       !tensor(a->t[2],RAD_I32,{rows,topk}) || !tensor(a->t[3],RAD_F32,{rows,topk}))return RAD_E_SHAPE;
    const auto logits=read(a->t[0]),bias=read(a->t[1]);std::vector<int32_t> ids(rows*topk);std::vector<float> weights(rows*topk);
    for(int64_t row=0;row<rows;++row) {
        ds::Routing result;const int status=ds::route(ds::Floats(logits).subspan(row*experts,experts),bias,topk,
                 ref::p_f32(a,"temperature",0),ref::p_f32(a,"route_scale",0),normalize!=0,&result);
        if(status!=RAD_OK)return status;
        std::copy(result.ids.begin(),result.ids.end(),ids.begin()+row*topk);
        std::copy(result.weights.begin(),result.weights.end(),weights.begin()+row*topk);
    }
    for(size_t i=0;i<ids.size();++i)((int32_t*)a->t[2].data)[ref::offlin(&a->t[2],int64_t(i))]=ids[i];
    write(a->t[3],weights);return RAD_OK;
}
extern "C" int ref_ds41_sparse_attn(const RadArgs* a,RadStream) {
    if(!ref::have(a,5))return RAD_E_INVAL;
    const auto dt=activation_dtype(a);if(dt==RAD_DT_INVALID)return RAD_E_DTYPE;
    const int64_t heads=ref::p_int(a,"heads",0),dim=ref::p_int(a,"head_dim",0),kvrows=ref::p_int(a,"kv_rows",0),topk=ref::p_int(a,"topk",0);
    if(heads<=0 || dim<=0 || heads>INT64_MAX/dim || kvrows<=0 || topk<=0 || a->t[0].rank!=3)return RAD_E_SHAPE;
    const int64_t rows=ref::band_rows(a->t[0].shape[0],ref::p_int(a,"M",0));
    if(!tensor(a->t[0],dt,{rows,heads,dim}) || !tensor(a->t[1],dt,{kvrows,dim}) ||
       !tensor(a->t[2],RAD_F32,{heads}) || !tensor(a->t[3],RAD_I32,{rows,topk}) || !tensor(a->t[4],dt,{rows,heads,dim}))return RAD_E_SHAPE;
    const auto query=read(a->t[0]),kv=read(a->t[1]),sinks=read(a->t[2]);std::vector<float> out(query.size());std::vector<int32_t> ids(topk);
    for(int64_t row=0;row<rows;++row) {
        for(int64_t i=0;i<topk;++i)ids[i]=((const int32_t*)a->t[3].data)[ref::offlin(&a->t[3],row*topk+i)];
        const int status=ds::sparse_attention(ds::Floats(query).subspan(row*heads*dim,heads*dim),kv,heads,dim,sinks,ids,
                   ref::p_f32(a,"scale",0),ds::Mutable(out).subspan(row*heads*dim,heads*dim));
        if(status!=RAD_OK)return status;
    }
    write(a->t[4],out);return RAD_OK;
}

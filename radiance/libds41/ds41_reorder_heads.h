/* Local activation reorder after a separate all-gather. No communication, scratch, payload
 * interpretation, head normalization, cache replication or architecture registration. */
#pragma once
#include "rad_plugin.h"
#include <climits>
#include <cstdint>
namespace rad::ds41::reorder_heads {
enum Operand {query=0,weights,query_out,weights_out,operand_count};
struct Launch {
    const void* q=nullptr;const void* w=nullptr;void* y=nullptr;void* v=nullptr;
    int64_t rows=0,heads=0,dim=0,tp=0,query_elements=0,head_elements=0;
    uint32_t dtype=0;int32_t element_bytes=0;
};
static_assert(sizeof(Launch)<=128,"raw launch arguments must fit the existing core AQL segment");
inline bool multiply(int64_t a,int64_t b,int64_t* out) {
    if(a<0||b<0||(b&&a>INT64_MAX/b))return false;
    *out=a*b;return true;
}
inline bool overlaps(const void* a,int64_t na,const void* b,int64_t nb) {
    const auto x=uintptr_t(a),y=uintptr_t(b);
    return x<=y?uint64_t(y-x)<uint64_t(na):uint64_t(x-y)<uint64_t(nb);
}
struct Geometry {int64_t capacity=0,tp=0,heads=0,dim=0;uint32_t dtype=0;int32_t bytes=0;};
inline int geometry(const RadParam* p,int np,Geometry* out,bool declaring=false) {
    if(!p||np<=0||!out)return RAD_E_INVAL;
    Geometry g;auto get=[&](const char* key){return declaring?rad_param_getdim(p,np,key,0):rad_param_geti(p,np,key,0);};
    g.capacity=get("M");g.tp=get("tp");g.heads=get("heads");g.dim=get("head_dim");
    g.dtype=rad_dtype_parse(rad_param_gets(p,np,"dtype",nullptr));
    if(g.dtype!=RAD_F32&&g.dtype!=RAD_F16&&g.dtype!=RAD_BF16)return RAD_E_DTYPE;
    g.bytes=g.dtype==RAD_F32?4:2;
    if(g.capacity<=0||g.capacity>INT_MAX||g.tp<=0||g.tp>65536||g.heads<=0||g.heads>INT_MAX||g.dim<=0||g.dim>INT_MAX)return RAD_E_SHAPE;
    int64_t count=0,weight_bytes=0;
    if(!multiply(g.tp,g.heads,&count)||!multiply(count,g.capacity,&count)||!multiply(count,4,&weight_bytes)||
       !multiply(count,g.dim,&count)||!multiply(count,g.bytes,&count))return RAD_E_SHAPE;
    *out=g;return RAD_OK;
}
inline int contiguous(const RadTensor& t,uint32_t dtype,const int64_t* shape,unsigned rank,int64_t* span) {
    if(!t.data||!span)return RAD_E_INVAL;
    if(t.dtype!=dtype)return RAD_E_DTYPE;
    if(t.rank!=rank)return RAD_E_SHAPE;
    int64_t elements=1;
    for(int axis=int(rank)-1;axis>=0;--axis) {
        if(shape[axis]<=0||t.shape[axis]!=shape[axis])return RAD_E_SHAPE;
        if(t.stride[axis]!=elements)return RAD_E_STRIDE;
        if(!multiply(elements,shape[axis],&elements))return RAD_E_SHAPE;
    }
    const int width=rad_dtype_bits(dtype)/8;
    if(width<=0||uintptr_t(t.data)%unsigned(width))return RAD_E_ALIGN;
    if(!multiply(elements,width,span)||uintptr_t(t.data)>UINTPTR_MAX-uint64_t(*span))return RAD_E_SHAPE;
    return RAD_OK;
}
inline int validate(const RadArgs* a,Launch* out) {
    if(!a||!out||!a->t||a->n_t!=operand_count)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(a->p,a->n_p,&g);if(rc!=RAD_OK)return rc;
    const bool flat=a->t[query].rank==3;
    if(!flat&&a->t[query].rank!=4)return RAD_E_SHAPE;
    if(flat&&(a->t[query].shape[0]<=0||a->t[query].shape[0]%g.tp))return RAD_E_SHAPE;
    const int64_t rows=flat?a->t[query].shape[0]/g.tp:a->t[query].shape[1];
    if(rows<=0||rows>g.capacity)return RAD_E_SHAPE;
    const int64_t qshape[]{g.tp,rows,g.heads,g.dim},wshape[]{g.tp,rows,g.heads},
                  yshape[]{rows,g.tp*g.heads,g.dim},vshape[]{rows,g.tp*g.heads},
                  fqshape[]{g.tp*rows,g.heads,g.dim},fwshape[]{g.tp*rows,g.heads};
    int64_t spans[operand_count]{};
    int checked=contiguous(a->t[query],g.dtype,flat?fqshape:qshape,flat?3:4,&spans[query]);if(checked!=RAD_OK)return checked;
    checked=contiguous(a->t[weights],RAD_F32,flat?fwshape:wshape,flat?2:3,&spans[weights]);if(checked!=RAD_OK)return checked;
    checked=contiguous(a->t[query_out],g.dtype,yshape,3,&spans[query_out]);if(checked!=RAD_OK)return checked;
    checked=contiguous(a->t[weights_out],RAD_F32,vshape,2,&spans[weights_out]);if(checked!=RAD_OK)return checked;
    for(int i=0;i<operand_count;++i)for(int j=0;j<i;++j)
        if(overlaps(a->t[i].data,spans[i],a->t[j].data,spans[j]))return RAD_E_INVAL;
    Launch p;p.q=a->t[query].data;p.w=a->t[weights].data;
    p.y=a->t[query_out].data;p.v=a->t[weights_out].data;
    p.rows=rows;p.heads=g.heads;p.dim=g.dim;p.tp=g.tp;p.dtype=g.dtype;p.element_bytes=g.bytes;
    p.query_elements=spans[query]/g.bytes;p.head_elements=spans[weights]/4;*out=p;return RAD_OK;
}
inline int shape(const RadParam* p,int np,int operand,RadOpdDesc* out) {
    if(!out||operand<0||operand>=operand_count)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(p,np,&g,true);if(rc!=RAD_OK)return rc;
    RadOpdDesc d{};d.idx_const=-1;d.dtype=operand==weights||operand==weights_out?uint32_t(RAD_F32):g.dtype;
    const int64_t qshape[]{g.tp,g.capacity,g.heads,g.dim},wshape[]{g.tp,g.capacity,g.heads},
                  yshape[]{g.capacity,g.tp*g.heads,g.dim},vshape[]{g.capacity,g.tp*g.heads};
    d.rank=operand==query?4:operand==weights||operand==query_out?3:2;
    const auto* dims=operand==query?qshape:operand==weights?wshape:operand==query_out?yshape:vshape;
    for(unsigned i=0;i<d.rank;++i)d.shape[i]=dims[i];
    *out=d;return RAD_OK;
}
} // namespace rad::ds41::reorder_heads
extern "C" {
int ds41_reorder_heads(const RadArgs*,RadStream);
int ds41_ref_reorder_heads(const RadArgs*,RadStream) noexcept;
}

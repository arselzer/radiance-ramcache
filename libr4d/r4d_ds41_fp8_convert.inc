/* Independent native FP8 aliases. The original bodies and exports above remain literal.
 * Only ds_to_e4m3/ds_e4m3 conversion changes on d89's proven finite domain. This SOURCE prototype
 * still needs actual original-vs-alias ISA admission and full byte gates in this compiler context. */
namespace fp8rt=rad_ds41_fp8_runtime;
struct Fp8Local { unsigned long long fast=0,fallback=0; };
struct Fp8Probe { uint8_t*code=nullptr;float*scale=nullptr;float*product=nullptr;long pitch=0; };
__device__ __forceinline__ bool fp8_hardware_built() {
#if defined(__gfx1201__)
    return true;
#else
    return false;
#endif
}
template<bool Audit>
__device__ __forceinline__ float fp8_converted(float v,float scale,Fp8Local& count,
                                              uint8_t* saved_code=nullptr) {
    const bool fast=fp8_hardware_built()&&fp8rt::hardware_domain(__float_as_uint(v),__float_as_uint(scale));
    const float q=ds_clamp(v/scale,-448.0f,448.0f);
    uint32_t code;float decoded;
#if defined(__gfx1201__)
    /* A gfx1201 compilation without these installed-toolchain intrinsics must fail compilation,
     * never produce an apparently hardware-capable alias that silently contains software only. */
    if(fast) {
        code=(uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q,q,0,false)&255u;
        /* q is finite/clamped here, so hardware encode never emits the NaN code. */
        decoded=__builtin_amdgcn_cvt_f32_fp8(code,0);
    } else
#endif
    {code=ds_to_e4m3(q);decoded=ds_e4m3(code);}
    if(saved_code)*saved_code=(uint8_t)code;
    if constexpr(Audit){count.fast+=fast?1u:0u;count.fallback+=fast?0u:1u;}
    float product;
#if defined(__gfx1201__)
    /* d89's admitted component operand order. Must be re-admitted against BOTH actual old
     * native exports in this DSO; C++ text does not prove NaN payload operand priority. */
    __asm__("v_mul_f32_e32 %0, %1, %2":"=v"(product):"v"(scale),"v"(decoded));
#else
    product=decoded*scale; // core admission refuses this target before allocating/issuing a CTX
#endif
    return product;
}
__device__ __forceinline__ void fp8_checked_add(unsigned long long*p,unsigned long long v,unsigned long long*bad) {
    if(!v)return;unsigned long long old=atomicCAS(p,0ull,0ull);
    for(;;){if(old>~0ull-v){atomicExch(bad,1ull);return;}
        const auto seen=atomicCAS(p,old,old+v);if(seen==old)return;old=seen;}
}
__device__ __forceinline__ unsigned long long fp8_down(unsigned long long v,unsigned d) {
    return (unsigned long long)__shfl_down((uint32_t)v,d,32)|
           ((unsigned long long)__shfl_down((uint32_t)(v>>32),d,32)<<32);
}
/* One optional checked integer update per workgroup, after all original arithmetic/stores.
 * Audit=false instantiates no shared storage, reductions, atomics or counter reads. */
template<bool Audit>
__device__ __forceinline__ void fp8_audit_block(unsigned long long*out,Fp8Local c) {
    if constexpr(Audit){
        __shared__ unsigned long long sums[8][2];
        for(unsigned d=16;d;d>>=1){c.fast+=fp8_down(c.fast,d);c.fallback+=fp8_down(c.fallback,d);}
        if(!(threadIdx.x&31)){sums[threadIdx.x/32][0]=c.fast;sums[threadIdx.x/32][1]=c.fallback;}
        __syncthreads();
        if(!threadIdx.x){Fp8Local total;
            for(unsigned w=0;w<8;++w){total.fast+=sums[w][0];total.fallback+=sums[w][1];}
            if(total.fast||total.fallback){fp8_checked_add(out+fp8rt::Fast,total.fast,out+fp8rt::Overflow);
                fp8_checked_add(out+fp8rt::Fallback,total.fallback,out+fp8rt::Overflow);
                fp8_checked_add(out+fp8rt::Blocks,1ull,out+fp8rt::Overflow);}
        }
    }
}
int fp8_audit_arg(const RadArgs*a,int base,unsigned long long**out) {
    *out=nullptr;if(a->n_t>base+1)return R4D_NO(RAD_E_INVAL);
    if(a->n_t<=base||!a->t[base].data)return RAD_OK;
    const auto&t=a->t[base];
    if(t.dtype!=RAD_I64||t.rank!=1||t.shape[0]!=fp8rt::kWords||t.stride[0]!=1||((uintptr_t)t.data&7u))
        return R4D_NO(RAD_E_SHAPE);
    *out=(unsigned long long*)t.data;return RAD_OK;
}
template<bool Audit,bool Probe=false,bool Candidate=true>
__global__ __launch_bounds__(256) void fp8_quant_kernel(uint16_t*x,long xp,int n,long rows,
                                                      int fmt,int sc,unsigned long long*audit,Fp8Probe probe) {
    Fp8Local count;
    const int col=blockIdx.x*256+threadIdx.x;const bool in=col<n;
    for(long r=blockIdx.y;r<rows;r+=gridDim.y){
        uint16_t*row=x+r*xp;const float v=in?ds_bf(row[col]):0.0f;
        float a=fabsf(v);a=a==a?a:0.0f;
        const float amax=r4d_xor_max<32>(a);
        if(in){const float scale=block_scale(amax,fmt,sc);uint8_t code=0;float product;
            if constexpr(Candidate)product=fp8_converted<Audit>(v,scale,count,Probe?&code:nullptr);
            else {product=roundtrip(v,scale,fmt);if constexpr(Probe)code=(uint8_t)ds_to_e4m3(ds_clamp(v/scale,-448.0f,448.0f));}
            row[col]=ds_bf16(product);
            if constexpr(Probe){const long at=r*probe.pitch+col;probe.code[at]=code;probe.scale[at]=scale;probe.product[at]=product;}}
    }
    fp8_audit_block<Audit>(audit,count);
}
template<bool Audit,bool Probe=false,bool Candidate=true>
__global__ __launch_bounds__(256) void fp8_rope_kernel(uint16_t*x,long xp,int n,long rows,FzRope r,
                                                     int fmt,int sc,unsigned long long*audit,Fp8Probe probe) {
    Fp8Local count;
    const int col=blockIdx.x*256+threadIdx.x;
    const bool in=col<n;
    const int hh=col/r.hd,d=col-hh*r.hd-r.off;
    const bool rot=in&&d>=0&&d<2*r.half;
    const int i=d>>1;
    const bool is_a=(d&1)==0;
    for(long m=blockIdx.y;m<rows;m+=gridDim.y){
        uint16_t*row=x+m*xp;
        const uint32_t h=in?(uint32_t)row[col]:0u;
        const uint32_t hp=(uint32_t)__shfl_xor((int)h,1);
        float v=ds_bf((uint16_t)h);
        const long p=r.pos[m*r.ps];
        if(rot&&p>=0&&p<r.tab_rows){
            const float c=r.cs[p*r.cp+(long)i*r.cst];
            const float sn=r.sign*r.cs[p*r.cp+(long)(r.half+i)*r.cst];
            const float va=is_a?v:ds_bf((uint16_t)hp),vb=is_a?ds_bf((uint16_t)hp):v;
            v=ds_bf(is_a?ds_bf16(va*c-vb*sn):ds_bf16(va*sn+vb*c));
        }
        float a=fabsf(v);a=a==a?a:0.0f;
        const float amax=r4d_xor_max<32>(a);
        if(in){const float scale=block_scale(amax,fmt,sc);uint8_t code=0;float product;
            if constexpr(Candidate)product=fp8_converted<Audit>(v,scale,count,Probe?&code:nullptr);
            else {product=roundtrip(v,scale,fmt);if constexpr(Probe)code=(uint8_t)ds_to_e4m3(ds_clamp(v/scale,-448.0f,448.0f));}
            row[col]=ds_bf16(product);
            if constexpr(Probe){const long at=m*probe.pitch+col;probe.code[at]=code;probe.scale[at]=scale;probe.product[at]=product;}}
    }
    fp8_audit_block<Audit>(audit,count);
}
int fp8_quant_body(const RadArgs*a,RadStream s) {
    if(!a||a->n_t<1||!a->t[0].data)return RAD_E_INVAL;
    const long n=(long)r4d_geti_or(a,"n",0),b=(long)r4d_geti_or(a,"block",0);
    int fmt,sc;if(!fz_quant_fmt(a,&fmt,&sc))return R4D_NO(RAD_E_UNSUPPORTED);
    /* Unknown geometry uses the WHOLE original body before any candidate issue. */
    if(!fp8rt::quant_scope(n,b,fmt,sc)){RadArgs old=*a;old.n_t=1;return quant_rt_body(&old,s);}
    const auto&x=a->t[0];
    if(x.dtype!=RAD_BF16||x.rank!=2||x.shape[0]<1||x.shape[1]!=n||!mat(x,RAD_BF16,x.shape[0],n))
        return R4D_NO(RAD_E_SHAPE);
    const long xp=(long)x.stride[0];
    unsigned long long*audit=nullptr;const int rc=fp8_audit_arg(a,1,&audit);if(rc<0)return rc;
    const auto grid=dim3((unsigned)((n+255)/256),grid_rows(x.shape[0]));
    const int slot=fp8rt::slot(fp8rt::Quant,n);
    if(audit)return R4D_EMIT(s,fp8_quant_kernel<true>,grid,dim3(256),0,
        (uint16_t*)x.data,xp,(int)n,(long)x.shape[0],fmt,sc,audit+slot*fp8rt::FieldCount,Fp8Probe{});
    return R4D_EMIT(s,fp8_quant_kernel<false>,grid,dim3(256),0,
        (uint16_t*)x.data,xp,(int)n,(long)x.shape[0],fmt,sc,nullptr,Fp8Probe{});
}
int fp8_rope_body(const RadArgs*a,RadStream s) {
    if(!a||a->n_t<3||!a->t[0].data||!a->t[1].data||!a->t[2].data)return RAD_E_INVAL;
    const auto&x=a->t[0];if(x.rank!=2||x.shape[0]<1)return R4D_NO(RAD_E_SHAPE);
    FzRope r;const int rc=fz_rope_args(a,x,a->t[1],a->t[2],x.shape[0],&r);if(rc<0)return R4D_NO(rc);
    const long n=(long)r.hd*(r.pairs/r.half),b=(long)r4d_geti_or(a,"block",0);
    int fmt,sc;if(!fz_quant_fmt(a,&fmt,&sc))return R4D_NO(RAD_E_UNSUPPORTED);
    if(!fp8rt::quant_scope(n,b,fmt,sc)||!fp8rt::rope_scope(n,r.hd,r.half,r.off,r.pairs,r.gptj,r.sign)){
        RadArgs old=*a;old.n_t=3;return rope_rt_body(&old,s);}
    unsigned long long*audit=nullptr;const int ar=fp8_audit_arg(a,3,&audit);if(ar<0)return ar;
    const dim3 grid((unsigned)((n+255)/256),grid_rows(x.shape[0]));
    const int slot=fp8rt::slot(fp8rt::Rope,n);
    if(audit)return R4D_EMIT(s,fp8_rope_kernel<true>,grid,dim3(256),0,
        (uint16_t*)x.data,(long)x.stride[0],(int)n,(long)x.shape[0],r,fmt,sc,audit+slot*fp8rt::FieldCount,Fp8Probe{});
    return R4D_EMIT(s,fp8_rope_kernel<false>,grid,dim3(256),0,
        (uint16_t*)x.data,(long)x.stride[0],(int)n,(long)x.shape[0],r,fmt,sc,nullptr,Fp8Probe{});
}

/* Unregistered sideband oracle. Original-product instantiations call the unchanged roundtrip
 * helper, never candidate(expected). The actual old public exports remain the BF16 authority.
 * No serving graph declares/uses this export or these transient buffers. */
int fp8_probe_body(const RadArgs*a,RadStream s,int op,int candidate) {
    if(!a||(op!=fp8rt::Quant&&op!=fp8rt::Rope)||(candidate!=0&&candidate!=1))return RAD_E_INVAL;
    const int base=op==fp8rt::Quant?1:3;
    if(a->n_t!=base+3||!a->t[0].data)return RAD_E_INVAL;
    const auto&x=a->t[0];if(x.rank!=2||x.shape[0]<1||!mat(x,RAD_BF16,x.shape[0],x.shape[1]))return RAD_E_SHAPE;
    const long n=x.shape[1],M=x.shape[0];
    int fmt,sc;if(!fz_quant_fmt(a,&fmt,&sc)||!fp8rt::quant_scope(n,r4d_geti_or(a,"block",0),fmt,sc))return RAD_E_UNSUPPORTED;
    FzRope r{};
    if(op==fp8rt::Rope){const int rc=fz_rope_args(a,x,a->t[1],a->t[2],M,&r);if(rc<0)return rc;
        if(!fp8rt::rope_scope(n,r.hd,r.half,r.off,r.pairs,r.gptj,r.sign))return RAD_E_UNSUPPORTED;}
    const auto&c=a->t[base];const auto&scale=a->t[base+1];const auto&product=a->t[base+2];
    if(!c.data||!scale.data||!product.data||!mat(c,RAD_U8,M,n)||!mat(scale,RAD_F32,M,n)||!mat(product,RAD_F32,M,n)||
       c.stride[0]!=scale.stride[0]||c.stride[0]!=product.stride[0])return RAD_E_SHAPE;
    Fp8Probe p{(uint8_t*)c.data,(float*)scale.data,(float*)product.data,(long)c.stride[0]};
    const dim3 grid((unsigned)((n+255)/256),grid_rows(M));
    if(op==fp8rt::Quant){
        if(candidate)return R4D_EMIT(s,(fp8_quant_kernel<false,true,true>),grid,dim3(256),0,(uint16_t*)x.data,(long)x.stride[0],(int)n,M,fmt,sc,nullptr,p);
        return R4D_EMIT(s,(fp8_quant_kernel<false,true,false>),grid,dim3(256),0,(uint16_t*)x.data,(long)x.stride[0],(int)n,M,fmt,sc,nullptr,p);
    }
    if(candidate)return R4D_EMIT(s,(fp8_rope_kernel<false,true,true>),grid,dim3(256),0,(uint16_t*)x.data,(long)x.stride[0],(int)n,M,r,fmt,sc,nullptr,p);
    return R4D_EMIT(s,(fp8_rope_kernel<false,true,false>),grid,dim3(256),0,(uint16_t*)x.data,(long)x.stride[0],(int)n,M,r,fmt,sc,nullptr,p);
}

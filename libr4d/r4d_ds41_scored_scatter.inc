/* Included AFTER the unchanged router_scored_kernel, in the same namespace.
 * One block owns every selected ID and sort output: no global arrival barrier. */
template<bool INV>
__global__ __launch_bounds__(256) void ds41_scored_scatter_m1_kernel(
    const float* logits,const float* bias,long bp,int32_t* ids,float* weights,
    int32_t* sorted,int32_t* off,int32_t* cnt,int32_t* inv,int base,int norm,float factor) {
    __shared__ r4d_moesort::ScatterLds<8> sl;
    const int lane=threadIdx.x&31;
    if(threadIdx.x<96)sl.cnt[threadIdx.x]=0;
    if(threadIdx.x<32) {
        /* OWN12 sqrtsoftplus branch, keys and selected-order sum copied from
         * router_scored_kernel<12,kSqrtSoftplus>; its old body is untouched. */
        float p[12],bv[12];
#pragma unroll
        for(int i=0;i<12;++i){const int e=lane+32*i;p[i]=logits[e];bv[i]=bias[(long)e*bp];}
        float t[12];
#pragma unroll
        for(int i=0;i<12;++i)t[i]=p[i];
        ds_exp_n(t);ds_log1p_n(t);
#pragma unroll
        for(int i=0;i<12;++i)p[i]=sqrtf(p[i]>20.0f?p[i]:t[i]);
        unsigned long long key[12];
#pragma unroll
        for(int i=0;i<12;++i)key[i]=ds_rank_key(p[i]+bv[i],lane+32*i);
        unsigned long long below=~0ull,mine=0;
        for(int j=0;j<6;++j){unsigned long long best=0;
#pragma unroll
            for(int i=0;i<12;++i){const auto c=key[i]<below?key[i]:0ull;best=c>best?c:best;}
            best=ds_wave_max_u64(best);if(lane==j)mine=best;below=best;
        }
        const int e=mine?(int)~(uint32_t)mine:0;float chosen=0;
#pragma unroll
        for(int i=0;i<12;++i){const float v=__shfl(p[i],e&31,32);if((e>>5)==i)chosen=v;}
        float total=0;
        for(int j=0;j<6;++j)total+=__int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen),j));
        const float mult=norm?factor/(total+1e-20f):factor;
        if(lane<6){ids[lane]=e;weights[lane]=chosen*mult;
            const int local=e-base;sl.e[lane]=local>=0&&local<96?local:-1;}
    }
    lds_barrier();
    r4d_moesort::scatter_small_place<8,INV>(sorted,off,cnt,6,96,sl,inv);
}

/* Read operands can alias each other. Every output is disjoint from every
 * input/output; range arithmetic includes strided bias and rejects overflow. */
bool ds41_scored_scatter_disjoint(const RadTensor*t,int n) {
    uintptr_t lo[8]={},hi[8]={};
    for(int i=0;i<n;++i){if(!t[i].data)continue;uint64_t last=0;
        for(uint32_t d=0;d<t[i].rank;++d){
            if(t[i].shape[d]<1||t[i].stride[d]<1)return false;
            const uint64_t shape=(uint64_t)t[i].shape[d]-1,stride=(uint64_t)t[i].stride[d];
            if(shape&&stride>(UINT64_MAX-last)/shape)return false;last+=shape*stride;
        }
        if(last>(UINT64_MAX/4)-1)return false;
        const uint64_t bytes=(last+1)*4;lo[i]=(uintptr_t)t[i].data;
        if(bytes>UINTPTR_MAX-lo[i])return false;hi[i]=lo[i]+bytes;
    }
    for(int i=2;i<n;++i)if(lo[i])for(int j=0;j<i;++j)if(lo[j]&&lo[i]<hi[j]&&lo[j]<hi[i])return false;
    return true;
}
std::atomic<uint32_t> ds41_scored_scatter_paths{0};
int ds41_scored_scatter_body(const RadArgs*a,RadStream stream) {
    if(!a||!a->t||a->n_t!=8)return RAD_E_INVAL;
    const auto*t=a->t;const auto&lg=t[0];const auto&bias=t[1];
    const int64_t M=r4d_geti_or(a,"M",0),ne=r4d_geti_or(a,"n_expert",0),nl=r4d_geti_or(a,"n_local",0),
        k=r4d_geti_or(a,"top_k",0),base=r4d_geti_or(a,"expert_base",-1);
    const char*score=r4d_gets(a,"score");const char*protect=r4d_gets(a,"protect");const char*dt=r4d_gets(a,"dtype");
    const float factor=(float)r4d_getf(a,"route_scale",0);const int norm=r4d_geti_or(a,"norm",0)!=0;
    if(a->world_size!=4||a->rank<0||a->rank>=4||M!=1||ne!=384||nl!=96||k!=6||base!=a->rank*96||
       !score||std::strcmp(score,"sqrtsoftplus")||!dt||std::strcmp(dt,"f32")||(protect&&*protect))return RAD_E_UNSUPPORTED;
    if(!std::isfinite(factor)||factor<=0)return RAD_E_INVAL;
    for(int i=0;i<7;++i)if(!t[i].data)return RAD_E_INVAL;
    for(int i=0;i<8;++i)if(t[i].data&&((uintptr_t)t[i].data&3))return RAD_E_SHAPE;
    if(lg.dtype!=RAD_F32||bias.dtype!=RAD_F32||t[2].dtype!=RAD_I32||t[3].dtype!=RAD_F32||
       t[4].dtype!=RAD_I32||t[5].dtype!=RAD_I32||t[6].dtype!=RAD_I32||(t[7].data&&t[7].dtype!=RAD_I32))return RAD_E_DTYPE;
    const auto row=[](const RadTensor&x,int64_t n){return x.rank==2&&x.shape[0]==1&&x.shape[1]==n&&
        x.stride[0]>=n&&x.stride[1]==1;};
    const auto vec=[](const RadTensor&x,int64_t n){return x.rank==1&&x.shape[0]==n&&x.stride[0]==1;};
    if(!row(lg,384)||bias.rank!=1||bias.shape[0]!=384||bias.stride[0]<1||!row(t[2],6)||!row(t[3],6)||
       !vec(t[4],6)||!vec(t[5],97)||!vec(t[6],96)||(t[7].data&&!vec(t[7],6))||
       !ds41_scored_scatter_disjoint(t,8))return RAD_E_SHAPE;
    int rc;
#define DS41_SCORED(INV) R4D_EMIT(stream,(ds41_scored_scatter_m1_kernel<INV>),dim3(1),dim3(256),0, \
        (const float*)lg.data,(const float*)bias.data,(long)bias.stride[0],(int32_t*)t[2].data,(float*)t[3].data, \
        (int32_t*)t[4].data,(int32_t*)t[5].data,(int32_t*)t[6].data,(int32_t*)t[7].data,(int)base,norm,factor)
    if(t[7].data)rc=DS41_SCORED(true);else rc=DS41_SCORED(false);
#undef DS41_SCORED
    if(rc==RAD_OK){const uint32_t bit=1u<<a->rank;
        if(!(ds41_scored_scatter_paths.load(std::memory_order_relaxed)&bit)&&
           !(ds41_scored_scatter_paths.fetch_or(bit,std::memory_order_relaxed)&bit))
            std::fprintf(stderr,"[r4d] ds41 scored scatter active: rank=%d M=1 path=sqrtsoftplus+stable_scatter inverse=%d\n",a->rank,t[7].data?1:0);}
    return rc;
}

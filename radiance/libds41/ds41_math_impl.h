/* Straight scalar FP32 equations, shared by the experimental GPU and a CPU emulator. GPU math
 * library transcendentals may differ slightly from the independent std:: oracle. No fast-math
 * or FMA contraction is part of this contract. Source + derived validation finish globally
 * before publication; checked dry-run never touches output. Workspace is caller supplied. */
#pragma once
#include "ds41_math.h"
#if defined(__HIPCC__)
#define DS41_MATH_HD __host__ __device__ inline
#else
#define DS41_MATH_HD inline
#endif
namespace rad::ds41::math {
namespace packed=rad::ds41::packed;
DS41_MATH_HD int64_t count(const RadTensor& t) {
    int64_t n=1;for(unsigned i=0;i<t.rank;++i)n*=t.shape[i];return n;
}
DS41_MATH_HD int64_t offset(const RadTensor& t,int64_t flat) {
    int64_t out=0;
    for(int i=int(t.rank)-1;i>=0;--i) {out+=(flat%t.shape[i])*t.stride[i];flat/=t.shape[i];}
    return out;
}
DS41_MATH_HD float load(const RadTensor& t,int64_t flat) {
    const int64_t i=offset(t,flat);
    if(t.dtype==RAD_F32)return static_cast<const float*>(t.data)[i];
    const uint16_t h=static_cast<const uint16_t*>(t.data)[i];
    return t.dtype==RAD_F16?packed::half(h):packed::from_bits(uint32_t(h)<<16);
}
DS41_MATH_HD void store(const RadTensor& t,int64_t flat,float value) {
    const int64_t i=offset(t,flat);
    if(t.dtype==RAD_F32)static_cast<float*>(t.data)[i]=value;
    else static_cast<uint16_t*>(t.data)[i]=t.dtype==RAD_F16?packed::to_half(value):packed::to_bf16(value);
}
DS41_MATH_HD int32_t id(const RadTensor& t,int64_t flat) {return static_cast<const int32_t*>(t.data)[offset(t,flat)];}
DS41_MATH_HD bool finite(float x) {return (packed::float_bits(x)&0x7fffffffU)<0x7f800000U;}
DS41_MATH_HD bool negative_infinity(float x) {return packed::float_bits(x)==0xff800000U;}
DS41_MATH_HD float sigmoid(float x) {
    if(x>=0)return 1.f/(1.f+::expf(-x));
    const float e=::expf(x);return e/(1.f+e);
}
DS41_MATH_HD float unbiased(const Plan& p,int64_t row,int64_t expert) {
    const float x=load(p.t[0],row*p.experts+expert)/p.temperature;
    return ::sqrtf(x>20.f ? x : ::log1pf(::expf(x)));
}
DS41_MATH_HD int input_tensors(const Plan& p) {
    return p.op==Op::hc_prepare?4:p.op==Op::hc_collapse?2:p.op==Op::router_topk?2:0;
}
DS41_MATH_HD int64_t preflight_items(const Plan& p) {
    if(p.op==Op::sparse_attn)return count(p.t[0])+p.heads+p.rows*p.topk;
    int64_t items=0;for(int i=0;i<input_tensors(p);++i)items+=count(p.t[i]);return items;
}
DS41_MATH_HD int check_item(const Plan& p,int64_t item) {
    if(p.op!=Op::sparse_attn) {
        for(int i=0;i<input_tensors(p);++i) {
            const int64_t n=count(p.t[i]);if(item<n)return finite(load(p.t[i],item))?RAD_OK:RAD_E_INVAL;item-=n;
        }
        return RAD_E_INVAL;
    }
    const int64_t q=count(p.t[0]);
    if(item<q)return finite(load(p.t[0],item))?RAD_OK:RAD_E_INVAL;
    item-=q;
    if(item<p.heads) {const float s=load(p.t[2],item);return finite(s)||negative_infinity(s)?RAD_OK:RAD_E_INVAL;}
    item-=p.heads;
    const int32_t selected=id(p.t[3],item);
    if(selected< -1||selected>=p.kv_rows)return RAD_E_SHAPE;
    if(selected>=0)for(int64_t d=0;d<p.dim;++d)
        if(!finite(load(p.t[1],int64_t(selected)*p.dim+d)))return RAD_E_INVAL;
    return RAD_OK;
}
// All arithmetic-validation paths are output-free. The same expressions run again only
// after the entire immutable batch has passed the ordered derived preflight.
DS41_MATH_HD bool representable(const RadTensor& t,float value) {
    if(!finite(value))return false;
    if(t.dtype==RAD_F32)return true;
    const uint16_t code=t.dtype==RAD_F16?packed::to_half(value):packed::to_bf16(value);
    return t.dtype==RAD_F16?(code&0x7c00u)!=0x7c00u:(code&0x7f80u)!=0x7f80u;
}
template<bool Write> DS41_MATH_HD int checked_store(const RadTensor& t,int64_t i,float value) {
    if(!representable(t,value))return RAD_E_INVAL;
    if constexpr(Write)store(t,i,value);
    return RAD_OK;
}
template<bool Write> DS41_MATH_HD int checked_prepare_row(const Plan& p,int64_t row,float* work) {
    const int64_t width=p.hc*p.n,mixes=p.hc*(p.hc+2);float square=0;
    for(int64_t c=0;c<width;++c) {
        const float v=load(p.t[0],row*width+c),product=v*v;
        if(!finite(product))return RAD_E_INVAL;
        square=square+product;if(!finite(square))return RAD_E_INVAL;
    }
    const float mean=square/float(width),norm=mean+p.eps;
    if(!finite(mean)||!finite(norm)||!(norm>0))return RAD_E_INVAL;
    const float root=::sqrtf(norm),inv=1.f/root;
    if(!finite(root)||!(root>0)||!finite(inv))return RAD_E_INVAL;
    float* coeff=work;float* comb=work+80;
    for(int64_t r=0;r<mixes;++r) {
        float value=0;
        for(int64_t c=0;c<width;++c) {
            const float product=load(p.t[0],row*width+c)*load(p.t[1],r*width+c);
            if(!finite(product))return RAD_E_INVAL;
            value=value+product;if(!finite(value))return RAD_E_INVAL;
        }
        coeff[r]=value*inv;if(!finite(coeff[r]))return RAD_E_INVAL;
    }
    for(int64_t i=0;i<p.hc;++i) {
        const float pre_scaled=coeff[i]*load(p.t[2],0),post_scaled=coeff[p.hc+i]*load(p.t[2],1);
        if(!finite(pre_scaled)||!finite(post_scaled))return RAD_E_INVAL;
        const float pre_arg=pre_scaled+load(p.t[3],i),post_arg=post_scaled+load(p.t[3],p.hc+i);
        if(!finite(pre_arg)||!finite(post_arg))return RAD_E_INVAL;
        const float pre_gate=sigmoid(pre_arg),post_gate=sigmoid(post_arg);
        if(!finite(pre_gate)||!finite(post_gate))return RAD_E_INVAL;
        const float pre=pre_gate+p.hc_eps,post=2.f*post_gate;
        if(checked_store<Write>(p.t[4],row*p.hc+i,pre)!=RAD_OK||
           checked_store<Write>(p.t[5],row*p.hc+i,post)!=RAD_OK)return RAD_E_INVAL;
    }
    for(int64_t src=0;src<p.hc;++src) {
        float maximum=packed::from_bits(0xff800000u);
        for(int64_t dst=0;dst<p.hc;++dst) {
            const auto i=2*p.hc+src*p.hc+dst;const float scaled=coeff[i]*load(p.t[2],2);
            if(!finite(scaled))return RAD_E_INVAL;
            const float v=scaled+load(p.t[3],i);if(!finite(v))return RAD_E_INVAL;
            comb[src*p.hc+dst]=v;if(v>maximum)maximum=v;
        }
        float sum=0;
        for(int64_t dst=0;dst<p.hc;++dst) {
            auto& v=comb[src*p.hc+dst];const float shifted=v-maximum;
            if(!finite(shifted))return RAD_E_INVAL;
            v=::expf(shifted);if(!finite(v))return RAD_E_INVAL;
            sum=sum+v;if(!finite(sum))return RAD_E_INVAL;
        }
        if(!(sum>0))return RAD_E_INVAL;
        for(int64_t dst=0;dst<p.hc;++dst) {
            const float divided=comb[src*p.hc+dst]/sum;
            if(!finite(divided))return RAD_E_INVAL;
            comb[src*p.hc+dst]=divided+p.hc_eps;if(!finite(comb[src*p.hc+dst]))return RAD_E_INVAL;
        }
    }
    for(int64_t iteration=0;iteration<p.iterations;++iteration) {
        if(iteration)for(int64_t src=0;src<p.hc;++src) {
            float sum=0;for(int64_t dst=0;dst<p.hc;++dst) {
                sum=sum+comb[src*p.hc+dst];if(!finite(sum))return RAD_E_INVAL;
            }
            const float denominator=sum+p.hc_eps;
            if(!finite(denominator)||!(denominator>0))return RAD_E_INVAL;
            for(int64_t dst=0;dst<p.hc;++dst) {
                comb[src*p.hc+dst]/=denominator;if(!finite(comb[src*p.hc+dst]))return RAD_E_INVAL;
            }
        }
        for(int64_t dst=0;dst<p.hc;++dst) {
            float sum=0;for(int64_t src=0;src<p.hc;++src) {
                sum=sum+comb[src*p.hc+dst];if(!finite(sum))return RAD_E_INVAL;
            }
            const float denominator=sum+p.hc_eps;
            if(!finite(denominator)||!(denominator>0))return RAD_E_INVAL;
            for(int64_t src=0;src<p.hc;++src) {
                comb[src*p.hc+dst]/=denominator;if(!finite(comb[src*p.hc+dst]))return RAD_E_INVAL;
            }
        }
    }
    for(int64_t i=0;i<p.hc*p.hc;++i)
        if(checked_store<Write>(p.t[6],row*p.hc*p.hc+i,comb[i])!=RAD_OK)return RAD_E_INVAL;
    return RAD_OK;
}
template<bool Write> DS41_MATH_HD int checked_collapse_element(const Plan& p,int64_t element) {
    const int64_t row=element/p.n,d=element%p.n;float sum=0;
    for(int64_t c=0;c<p.hc;++c) {
        const float product=load(p.t[1],row*p.hc+c)*load(p.t[0],row*p.hc*p.n+c*p.n+d);
        if(!finite(product))return RAD_E_INVAL;
        sum=sum+product;if(!finite(sum))return RAD_E_INVAL;
    }
    return checked_store<Write>(p.t[2],element,sum);
}
DS41_MATH_HD int checked_unbiased(const Plan& p,int64_t row,int64_t expert,float* result) {
    const float x=load(p.t[0],row*p.experts+expert)/p.temperature;
    if(!finite(x))return RAD_E_INVAL;
    float softplus=x;
    if(!(x>20.f)) {
        const float exponential=::expf(x);if(!finite(exponential))return RAD_E_INVAL;
        softplus=::log1pf(exponential);if(!finite(softplus)||softplus<0)return RAD_E_INVAL;
    }
    *result=::sqrtf(softplus);return finite(*result)?RAD_OK:RAD_E_INVAL;
}
template<bool Write> DS41_MATH_HD int checked_router_row(const Plan& p,int64_t row,float* work) {
    float* base=work;float* scores=work+p.experts;float* selected_weights=work+2*p.experts;
    for(int64_t e=0;e<p.experts;++e) {
        if(checked_unbiased(p,row,e,&base[e])!=RAD_OK)return RAD_E_INVAL;
        scores[e]=base[e]+load(p.t[1],e);if(!finite(scores[e]))return RAD_E_INVAL;
    }
    float total=0;float previous=0;int32_t previous_id=-1;
    for(int64_t rank=0;rank<p.topk;++rank) {
        int32_t best=-1;float best_score=packed::from_bits(0xff800000u);
        for(int32_t e=0;e<p.experts;++e) {
            const float score=scores[e];
            const bool eligible=rank==0||score<previous||(score==previous&&e>previous_id);
            if(eligible&&(best<0||score>best_score||(score==best_score&&e<best))) {best=e;best_score=score;}
        }
        if(best<0||best>=p.experts)return RAD_E_INVAL;
        previous=best_score;previous_id=best;
        if constexpr(Write)static_cast<int32_t*>(p.t[2].data)[offset(p.t[2],row*p.topk+rank)]=best;
        selected_weights[rank]=base[best];total=total+base[best];if(!finite(total))return RAD_E_INVAL;
    }
    const float denominator=total+1e-20f;
    if(!finite(denominator)||!(denominator>0))return RAD_E_INVAL;
    for(int64_t rank=0;rank<p.topk;++rank) {
        float weight=selected_weights[rank];
        if(p.normalize&&p.topk>1)weight/=denominator;
        if(!finite(weight))return RAD_E_INVAL;
        if(checked_store<Write>(p.t[3],row*p.topk+rank,weight*p.route_scale)!=RAD_OK)return RAD_E_INVAL;
    }
    return RAD_OK;
}
template<bool Write> DS41_MATH_HD int checked_attention_head(const Plan& p,int64_t job,float* work) {
    const int64_t row=job/p.heads,head=job%p.heads;
    float* scores=work;float* result=work+p.topk;
    const float sink=load(p.t[2],head);float maximum=sink;
    for(int64_t d=0;d<p.dim;++d)result[d]=0.f;
    for(int64_t i=0;i<p.topk;++i) {
        const int32_t selected=id(p.t[3],row*p.topk+i);
        if(selected<0) {scores[i]=packed::from_bits(0xff800000u);continue;}
        float dot=0;
        for(int64_t d=0;d<p.dim;++d) {
            const float product=load(p.t[0],job*p.dim+d)*load(p.t[1],int64_t(selected)*p.dim+d);
            if(!finite(product))return RAD_E_INVAL;
            dot=dot+product;if(!finite(dot))return RAD_E_INVAL;
        }
        scores[i]=dot*p.scale;if(!finite(scores[i]))return RAD_E_INVAL;
        if(scores[i]>maximum)maximum=scores[i];
    }
    if(!negative_infinity(maximum)) {
        const float sink_shift=sink-maximum;
        if(!negative_infinity(sink)&&!finite(sink_shift))return RAD_E_INVAL;
        float denominator=::expf(sink_shift);if(!finite(denominator))return RAD_E_INVAL;
        for(int64_t i=0;i<p.topk;++i) {
            const int32_t selected=id(p.t[3],row*p.topk+i);if(selected<0)continue;
            const float shift=scores[i]-maximum;if(!finite(shift))return RAD_E_INVAL;
            const float probability=::expf(shift);if(!finite(probability))return RAD_E_INVAL;
            denominator=denominator+probability;if(!finite(denominator))return RAD_E_INVAL;
            for(int64_t d=0;d<p.dim;++d) {
                const float product=probability*load(p.t[1],int64_t(selected)*p.dim+d);
                if(!finite(product))return RAD_E_INVAL;
                result[d]=result[d]+product;if(!finite(result[d]))return RAD_E_INVAL;
            }
        }
        if(!(denominator>0))return RAD_E_INVAL;
        for(int64_t d=0;d<p.dim;++d) {
            result[d]/=denominator;if(!finite(result[d]))return RAD_E_INVAL;
        }
    }
    for(int64_t d=0;d<p.dim;++d)
        if(checked_store<Write>(p.t[4],job*p.dim+d,result[d])!=RAD_OK)return RAD_E_INVAL;
    return RAD_OK;
}
// Compatibility entry points; public callers first dry-run every batch job.
DS41_MATH_HD void prepare_row(const Plan& p,int64_t row,float* work) {(void)checked_prepare_row<true>(p,row,work);}
DS41_MATH_HD void collapse_element(const Plan& p,int64_t i) {(void)checked_collapse_element<true>(p,i);}
DS41_MATH_HD void router_row(const Plan& p,int64_t row,float* work) {(void)checked_router_row<true>(p,row,work);}
DS41_MATH_HD void attention_head(const Plan& p,int64_t job,float* work) {(void)checked_attention_head<true>(p,job,work);}
inline int64_t workspace_floats(const Plan& p) {
    return p.op==Op::hc_prepare?144:p.op==Op::router_topk?2*p.experts+p.topk:p.op==Op::sparse_attn?p.topk+p.dim:0;
}
DS41_MATH_HD int64_t derived_jobs(const Plan& p) {
    return p.op==Op::hc_collapse?p.rows*p.n:p.op==Op::sparse_attn?p.rows*p.heads:p.rows;
}
template<bool Write> DS41_MATH_HD int checked_job(const Plan& p,int64_t job,float* work) {
    if(p.op==Op::hc_prepare)return checked_prepare_row<Write>(p,job,work);
    if(p.op==Op::hc_collapse)return checked_collapse_element<Write>(p,job);
    if(p.op==Op::router_topk)return checked_router_row<Write>(p,job,work);
    return checked_attention_head<Write>(p,job,work);
}
} // namespace rad::ds41::math
#undef DS41_MATH_HD

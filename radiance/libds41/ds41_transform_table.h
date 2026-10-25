/* Checked HOST table construction, explicitly capped by caller position count/byte budget.
 * No training-context allocation is inferred and no model weights are introduced. */
#pragma once
#include "ds41_transform.h"
#include "../arch/deepseek41/config.h"
#include <array>
#include <span>
namespace rad::ds41::transform {
struct RopeSpec {
    int64_t rotary=0,original_context=0;float theta=0,factor=0,beta_fast=0,beta_slow=0;
};
struct TableReceipt {int64_t positions=0,rotary=0,bytes=0;};
inline int validate_spec(const RopeSpec& s) {
    if(s.rotary<=0||s.rotary>max_head_dim||(s.rotary&1)||s.original_context<0)return RAD_E_SHAPE;
    return s.theta>1&&s.factor>0&&s.beta_fast>0&&s.beta_slow>0&&std::isfinite(s.theta)&&std::isfinite(s.factor)&&
           std::isfinite(s.beta_fast)&&std::isfinite(s.beta_slow)?RAD_OK:RAD_E_INVAL;
}
inline int select_spec(const deepseek41::Config& c,int64_t layer,RopeSpec* out) {
    if(!out)return RAD_E_INVAL;
    if(layer<0||layer>=c.n_layers||uint64_t(layer)>=c.layers.size()||c.rope_type!="yarn")return RAD_E_SHAPE;
    const int ratio=c.layers[size_t(layer)].compress_ratio;if(ratio<0||ratio>2)return RAD_E_SHAPE;
    RopeSpec s;s.rotary=c.rope_head_dim;s.original_context=ratio?c.original_seq_len:0;
    s.theta=float(ratio?c.compress_rope_theta:c.rope_theta);s.factor=float(c.rope_factor);
    s.beta_fast=float(c.beta_fast);s.beta_slow=float(c.beta_slow);
    const int status=validate_spec(s);if(status!=RAD_OK)return status;*out=s;return RAD_OK;
}
inline int build_pairs(const RopeSpec& s,int64_t positions,uint64_t max_byte_budget,
                        std::vector<float>* out,TableReceipt* receipt=nullptr) {
    if(!out)return RAD_E_INVAL;
    int status=validate_spec(s);if(status!=RAD_OK)return status;
    if(positions<=0||positions>max_positions)return RAD_E_SHAPE;
    int64_t elements=0,bytes=0;
    if(!math::multiply(positions,s.rotary,&elements)||!math::multiply(elements,4,&bytes)||
       uint64_t(bytes)>SIZE_MAX)return RAD_E_SHAPE;
    if(uint64_t(bytes)>max_byte_budget)return RAD_E_NOMEM;
    std::array<float,size_t(max_head_dim/2)> inverse{};float low=0,high=0;
    if(s.original_context) {
        auto corrected=[&](float rotations) {return double(s.rotary)*std::log(double(s.original_context)/(double(rotations)*2*std::acos(-1.)))/(2*std::log(double(s.theta)));};
        low=float(std::max(std::floor(corrected(s.beta_fast)),0.));high=float(std::min(std::ceil(corrected(s.beta_slow)),double(s.rotary-1)));
        if(!std::isfinite(low)||!std::isfinite(high))return RAD_E_INVAL;
    }
    float maximum=0;
    for(int64_t pair=0;pair<s.rotary/2;++pair) {
        float frequency=1.f/std::pow(s.theta,float(2*pair)/float(s.rotary));
        if(s.original_context) {
            const float smooth=1.f-std::clamp((float(pair)-low)/std::max(high-low,1e-3f),0.f,1.f);
            frequency=(frequency/s.factor)*(1.f-smooth)+frequency*smooth;
        }
        if(!std::isfinite(frequency)||frequency<0)return RAD_E_INVAL;
        inverse[size_t(pair)]=frequency;maximum=std::max(maximum,frequency);
    }
    // This establishes finite angles for every smaller position/frequency before allocating or
    // mutating caller output. Stack inverse-frequency scratch is bounded at 512 floats.
    if(!std::isfinite(float(positions-1)*maximum))return RAD_E_INVAL;
    try {
        std::vector<float> result(size_t(elements),0.f);
        for(int64_t position=0;position<positions;++position)for(int64_t pair=0;pair<s.rotary/2;++pair) {
            const float angle=float(position)*inverse[size_t(pair)];
            result[size_t(position*s.rotary+2*pair)]=std::cos(angle);
            result[size_t(position*s.rotary+2*pair+1)]=std::sin(angle);
        }
        *out=std::move(result);if(receipt)*receipt={positions,s.rotary,bytes};return RAD_OK;
    } catch(const std::bad_alloc&) {return RAD_E_NOMEM;}
}
} // namespace rad::ds41::transform

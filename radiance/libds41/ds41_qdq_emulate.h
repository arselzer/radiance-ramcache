#pragma once
#include "ds41_qdq.h"
#include "../arch/deepseek41/qdq_ref.h"
namespace rad::ds41::qdq {
inline int emulate(const RadArgs* a){
    Plan p;const int checked=validate(a,&p);if(checked!=RAD_OK)return checked;
    if(!deepseek41::qdq::supported_f32_environment())return RAD_E_UNSUPPORTED;
    *p.status=RAD_OK;
    // Establish every mask before any input payload read.
    if(p.masked)for(int64_t row=0;row<p.rows;++row){const int32_t bit=math::id(p.t[emit],row);if(bit!=0&&bit!=1)return *p.status=RAD_E_INVAL;}
    const int width=qdq_math::width(p.profile);
    for(int pass=0;pass<2;++pass)for(int64_t row=0;row<p.rows;++row){
        const bool active=!p.masked||math::id(p.t[emit],row)!=0;
        if(!active){if(pass)for(int64_t d=0;d<p.dim;++d)static_cast<uint16_t*>(p.t[output].data)[math::offset(p.t[output],row*p.dim+d)]=0;continue;}
        for(int64_t group=0;group<p.groups;++group){
            uint16_t source[32]{},values[32]{};
            for(int i=0;i<width;++i)source[i]=static_cast<const uint16_t*>(p.t[input].data)[math::offset(p.t[input],row*p.dim+group*width+i)];
            const int rc=qdq_math::group(p.profile,source,values);if(rc!=RAD_OK)return *p.status=rc;
            if(pass)for(int i=0;i<width;++i)static_cast<uint16_t*>(p.t[output].data)[math::offset(p.t[output],row*p.dim+group*width+i)]=values[i];
        }
    }
    return RAD_OK;
}
} // namespace rad::ds41::qdq

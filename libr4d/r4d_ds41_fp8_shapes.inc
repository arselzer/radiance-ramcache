#include "../arch/common/rad_ds41_fp8_runtime.h"
extern "C" int r4d_shape_quant_roundtrip_fp8_convert(const RadParam*p,int n,int i,RadOpdDesc*out) {
    if(i==1){if(!out)return RAD_E_INVAL;*out=opd((uint32_t)RAD_I64,rad_ds41_fp8_runtime::kWords);return RAD_OK;}
    return r4d_shape_quant_roundtrip(p,n,i,out);
}
extern "C" int r4d_shape_rope_rows_roundtrip_fp8_convert(const RadParam*p,int n,int i,RadOpdDesc*out) {
    if(i==3){if(!out)return RAD_E_INVAL;*out=opd((uint32_t)RAD_I64,rad_ds41_fp8_runtime::kWords);return RAD_OK;}
    return r4d_shape_rope_rows_roundtrip(p,n,i,out);
}

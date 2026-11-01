#include "../arch/common/rad_ds41_fp4_runtime.h"
extern "C" int r4d_shape_quant_roundtrip_fp4_classify(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (i == 1) { if (!out) return RAD_E_INVAL; *out = opd((uint32_t)RAD_I64, rad_ds41_fp4_runtime::kWords); return RAD_OK; }
    return r4d_shape_quant_roundtrip(p,n,i,out);
}

extern "C" int r4d_shape_quant_roundtrip_fp4_classify_kvpack(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (i == 2) { if (!out) return RAD_E_INVAL; *out = opd((uint32_t)RAD_I64, rad_ds41_fp4_runtime::kWords); return RAD_OK; }
    return r4d_shape_quant_roundtrip_kvpack(p,n,i,out);
}

extern "C" int r4d_shape_rmsnorm_rope_roundtrip_fp4_classify(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (i == 5) { if (!out) return RAD_E_INVAL; *out = opd((uint32_t)RAD_I64, rad_ds41_fp4_runtime::kWords); return RAD_OK; }
    return r4d_shape_rmsnorm_rope_roundtrip(p,n,i,out);
}

extern "C" int r4d_shape_rmsnorm_rope_roundtrip_fp4_classify_kvpack(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (i == 6) { if (!out) return RAD_E_INVAL; *out = opd((uint32_t)RAD_I64, rad_ds41_fp4_runtime::kWords); return RAD_OK; }
    return r4d_shape_rmsnorm_rope_roundtrip_kvpack(p,n,i,out);
}

extern "C" int r4d_shape_rope_rows_roundtrip_fp4_classify(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (i == 3) { if (!out) return RAD_E_INVAL; *out = opd((uint32_t)RAD_I64, rad_ds41_fp4_runtime::kWords); return RAD_OK; }
    return r4d_shape_rope_rows_roundtrip(p,n,i,out);
}

extern "C" int r4d_shape_rope_rows_roundtrip_fp4_classify_kvpack(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (i == 4) { if (!out) return RAD_E_INVAL; *out = opd((uint32_t)RAD_I64, rad_ds41_fp4_runtime::kWords); return RAD_OK; }
    return r4d_shape_rope_rows_roundtrip_kvpack(p,n,i,out);
}

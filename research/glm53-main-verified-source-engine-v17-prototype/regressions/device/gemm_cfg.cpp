// Narrow packaging dependency of unchanged r4d_moe.hip, outside serving/SRC.
// The original TU also emits unregistered GEMM launchers. Their cfg_for() calls
// r4d_cfg_gemm, normally supplied by the complete r4d_rows.cpp registry.
// Reproduce ONLY that accessor, including ordinary integer parameter overrides.
// Source verified against pinned libr4d/r4d_rows.cpp:
//   tGemmSkinny (line4916): wv1, sk0, mb0, npw1, nt1;
//   tune_i/r4d_cfg_gemm (lines9628..9644): first integer parameter, else default,
//   converted to int. No GEMM row is registered/executed by this fixture.
#include "r4d_plugin.h"

namespace {
int configured(const RadArgs* a,const char* key,int fallback){
    long long value=0;
    return rad_args_geti(a,key,&value)?static_cast<int>(value):fallback;
}
}
extern "C" void r4d_cfg_gemm(const RadArgs* a,R4dGemmCfg* out){
    out->wv=configured(a,"wv",1);
    out->sk=configured(a,"sk",0);
    out->mb=configured(a,"mb",0);
    out->npw=configured(a,"npw",1);
    out->nt=configured(a,"nt",1);
}

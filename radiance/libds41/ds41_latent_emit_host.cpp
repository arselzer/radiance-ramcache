#include "ds41_latent_emit_emulate.h"
namespace rad::ds41::latent_emit {
int prepare(const RadArgs* a,Plan* out){return validate(a,out);}
}
extern "C" int ds41_ref_latent_store_emitted(const RadArgs* a,RadStream) noexcept {
    if(!a)return RAD_E_INVAL;
    int32_t flag=RAD_OK;RadArgs host=*a;host.scratch=&flag;host.scratch_bytes=sizeof flag;
    return rad::ds41::latent_emit::emulate(&host);
}

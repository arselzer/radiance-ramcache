#include "ds41_qdq_emulate.h"
extern "C" int ds41_ref_qdq_candidate(const RadArgs* a,RadStream) noexcept {
    if(!a)return RAD_E_INVAL;
    int32_t status=RAD_OK;RadArgs local=*a;local.scratch=&status;local.scratch_bytes=sizeof status;
    return rad::ds41::qdq::emulate(&local);
}

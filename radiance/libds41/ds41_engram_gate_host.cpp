#include "ds41_engram_gate_emulate.h"
extern "C" int ds41_ref_engram_gate(const RadArgs* args,RadStream) noexcept {
    if(!args)return RAD_E_INVAL;
    // Generic host Ctx scratch can still be a device-arena address. No async
    // host status is promised: use a private CPU flag and return the error.
    int32_t status=RAD_OK;RadArgs host=*args;host.scratch=&status;host.scratch_bytes=sizeof status;
    return rad::ds41::engram_gate::emulate(&host);
}

#include "ds41_compress_finish_emulate.h"
extern "C" int ds41_ref_compress_finish(const RadArgs* args,RadStream) noexcept {
    if(!args)return RAD_E_INVAL;
    // Host rows never dereference the caller's possibly DEVICE-arena scratch.
    int32_t status=RAD_OK;RadArgs host=*args;host.scratch=&status;host.scratch_bytes=sizeof status;
    return rad::ds41::compress_finish::emulate(&host);
}

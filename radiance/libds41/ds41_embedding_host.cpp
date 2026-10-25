/* CPU host rows return payload failures directly and never touch caller/device scratch. */
#include "ds41_embedding_emulate.h"
#include <new>
namespace rad::ds41::embedding {
int prepare(const RadArgs* a,Plan* p){return validate(a,p);}
}
extern "C" int ds41_ref_embedding(const RadArgs* a,RadStream) noexcept {
    if(!a)return RAD_E_INVAL;
    int32_t flag=RAD_OK;RadArgs host=*a;host.scratch=&flag;host.scratch_bytes=sizeof flag;
    try{return rad::ds41::embedding::emulate(&host);}
    catch(const std::bad_alloc&){return RAD_E_NOMEM;}
    catch(...){return RAD_E_STATE;}
}

#include "profile.h"
#include <cstring>
#include <new>
namespace glm53::engine_core_contract {
namespace {void explain(std::string* why,const char* s)noexcept{if(why)try{*why=s;}catch(...){}}}
bool ColdSourceSeal::permits_zero_head_metadata(const RadModelMeta& m,const RadBuildCtx& c)const noexcept{
    return owner_&&source_.fingerprint&&admit_declaration_shape(&m,&c)==RAD_OK;
}
int seal_source(const InspectedSource& inspected,std::unique_ptr<const ColdSourceSeal>* out,std::string* why){
    const auto& q=inspected.quote();
    if(!out||*out)return RAD_E_INVAL;
    if(!inspected.owner_)return RAD_E_INVAL;
    const int audit=inspected.owner_->audit(why);if(audit<0)return audit;
    if(!q.fingerprint||q.physical_tensors!=219||q.payload_bytes!=1020544||q.vocabulary!=V||
       !q.header_bytes||q.header_bytes>65536||q.file_bytes!=8+q.header_bytes+q.payload_bytes||
       q.file_bytes>2*MiB||q.artifact_files!=4||!q.sidecar_bytes||q.sidecar_bytes>131072||
       q.vocab.tokens!=V||q.vocab.bos!=257||q.vocab.eos!=256||q.vocab.eot!=256||q.vocab.eog_count!=2||
       q.vocab.add_bos||q.vocab.add_eos||q.vocab.bos_terminal||!q.vocab.eos_terminal||!q.vocab.im_end_terminal){explain(why,"closed actual synthetic source/tokenizer quote");return RAD_E_FORMAT;}
    try{auto ready=std::unique_ptr<const ColdSourceSeal>(new ColdSourceSeal(q,inspected.owner_));*out=std::move(ready);return RAD_OK;}
    catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int quote_admission(const ColdSourceSeal& source,const BackendQuote& b,AdmissionQuote* out,std::string* why){
    if(!out)return RAD_E_INVAL;
    const int audit=source.owner().audit(why);if(audit<0)return audit;
    if(!source.source().fingerprint||b.loader_threads!=2||b.streams!=3||!b.stack_bytes||b.stack_bytes>8*MiB||b.guard_bytes>4096)
        return RAD_E_UNSUPPORTED;
    AdmissionQuote q;q.source_admitted=true;
    if(b.backend==Backend::Host){
        if(!b.matching_private_size_quote||!b.host_stream_control_bytes||b.host_stream_control_bytes>(128u<<10)){
            explain(why,"actual matching HOST Task/ring/HostStream quote unavailable");return RAD_E_UNSUPPORTED;}
        q.owned_threads=5;q.backend_stream_bytes=3*(b.stack_bytes+b.guard_bytes+b.host_stream_control_bytes);
    }else if(b.backend==Backend::Aql){q.owned_threads=2;q.backend_stream_bytes=38538240;}
    else return RAD_E_UNSUPPORTED;
    if(q.backend_stream_bytes>38538240||q.categories>q.cap)return RAD_E_SCRATCH;
    q.backend_admitted=true;
    // No Engine integration, HOST RAII/loader retirement implementation or
    // actual private backend-size query exists in this package. A quote cannot
    // manufacture the missing activation capability, even with supplied flags.
    q.engine_eligible=false;*out=q;return RAD_OK;
}
} // namespace glm53::engine_core_contract

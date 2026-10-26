// Explicit real AQL same-Engine declaration prefix; no source factory or model.
#include "engine.h"
#include "engine_glm_main.h"
#include "device/device.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace rad;
namespace {
std::atomic<bool> observing{false};
std::atomic<uint64_t> device_allocations{0},pinned_allocations{0},pinned_requested{0},streams{0},events{0},copies{0},clears{0};
void need(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
struct Observe {Observe(){observing=true;}~Observe(){observing=false;}};
}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t bytes,int kind){
 if(observing.load()){if(kind==RAD_MEM_DEVICE)++device_allocations;
  if(kind==RAD_MEM_HOST_PINNED){++pinned_allocations;if(bytes>0)pinned_requested+=uint64_t(bytes);}}
 return __real_rad_dev_alloc(bytes,kind);
}
extern "C" int __real_rad_stream_create(RadStream*,int);
extern "C" int __wrap_rad_stream_create(RadStream* out,int priority){if(observing.load())++streams;return __real_rad_stream_create(out,priority);}
extern "C" int __real_rad_event_create(RadEvent*);
extern "C" int __wrap_rad_event_create(RadEvent* out){if(observing.load())++events;return __real_rad_event_create(out);}
extern "C" int __real_rad_event_create_as(RadEvent*,unsigned);
extern "C" int __wrap_rad_event_create_as(RadEvent* out,unsigned flags){if(observing.load())++events;return __real_rad_event_create_as(out,flags);}
extern "C" int __real_rad_memcpy_async(void*,const void*,int64_t,RadStream);
extern "C" int __wrap_rad_memcpy_async(void* to,const void* from,int64_t bytes,RadStream stream){if(observing.load())++copies;return __real_rad_memcpy_async(to,from,bytes,stream);}
extern "C" int __real_rad_memcpy_2d_async(void*,int64_t,const void*,int64_t,int64_t,int64_t,RadStream);
extern "C" int __wrap_rad_memcpy_2d_async(void* to,int64_t dp,const void* from,int64_t sp,int64_t width,int64_t height,RadStream stream){if(observing.load())++copies;return __real_rad_memcpy_2d_async(to,dp,from,sp,width,height,stream);}
extern "C" int __real_rad_memset_async(void*,int,int64_t,RadStream);
extern "C" int __wrap_rad_memset_async(void* to,int value,int64_t bytes,RadStream stream){if(observing.load())++clears;return __real_rad_memset_async(to,value,bytes,stream);}
int main(int argc,char** argv){try{
 need(argc==7&&!std::strcmp(argv[1],"--run-actual-declaration"),"explicit MODEL HOME BUNDLE PROOF TRUSTED_ID required");
 for(int i=2;i<=5;++i){const std::string value=argv[i];need(!value.empty()&&value[0]=='/'&&value.size()<=4096,"absolute bounded paths");}
 const std::string id=argv[6];need(!id.empty()&&id.size()<=20&&std::all_of(id.begin(),id.end(),[](unsigned char c){return c>='0'&&c<='9';}),"bounded decimal bundle ID");
 GlmMainEngineOptions options;options.checkpoint=argv[2];options.home=argv[3];options.bundle_root=argv[4];options.source_proof=argv[5];
 options.expected_bundle_id=std::stoull(id);options.prefill_max_tok=4;
 std::unique_ptr<Engine> engine;need(GlmMainEngineHarness::create(options,&engine)==RAD_OK,"actual closed R4 Engine cold request");
 GlmMainDeclarationContext witness;int status=RAD_E_STATE,retry=RAD_OK;
 {Observe scope;status=GlmMainEngineHarness::declaration_preflight(*engine,&witness);retry=engine->start();}
 GlmMainEngineSnapshot snapshot;GlmMainEngineHarness::snapshot(*engine,&snapshot);
 bool valid=status==RAD_OK&&retry<0&&witness.requested_max_tok==4&&witness.declared_ranks==4&&witness.engine_retired&&
  !snapshot.source_admitted&&!snapshot.transport_ready&&!snapshot.source_bytes_verified&&!snapshot.mapped_bytes_compared&&
  !witness.source_admitted&&!witness.session_present&&!witness.provider_present&&!witness.native_owner_present&&
  !witness.application_stream_present&&!witness.context_initialized&&!witness.registry_instances&&
  !witness.vmem_committed_before&&!witness.vmem_committed_after&&device_allocations==0&&streams==0&&events==0&&copies==0&&clears==0;
 for(size_t i=0;i<4;++i)valid=valid&&witness.program_max_tok[i]==4&&witness.route_max_rows[i]==4&&witness.route_maps[i]==42&&
  witness.quantum[i]==4&&witness.attn_block[i]==4&&witness.state_chunk[i]==1&&!witness.model_pool_bytes[i];
 nlohmann::ordered_json report{{"schema","glm53-actual-same-engine-declaration-context-v1"},{"passed",valid},{"status",status},
  {"requested_max_tok",witness.requested_max_tok},{"declared_ranks",witness.declared_ranks},{"program_max_tok",witness.program_max_tok},
  {"route_max_rows",witness.route_max_rows},{"route_maps",witness.route_maps},{"actual_quantum",witness.quantum},
  {"actual_attention_block",witness.attn_block},{"actual_state_chunk",witness.state_chunk},{"model_pool_bytes",witness.model_pool_bytes},
  {"registry_instances",witness.registry_instances},{"vmem_committed_before",witness.vmem_committed_before},{"vmem_committed_after",witness.vmem_committed_after},
  {"source_admitted",snapshot.source_admitted},{"native_bytes_verified",snapshot.source_bytes_verified},{"raw_mapped_bytes_compared",snapshot.mapped_bytes_compared},
  {"session_present",witness.session_present},{"provider_present",witness.provider_present},{"native_owner_present",witness.native_owner_present},
  {"application_stream_present",witness.application_stream_present},{"context_initialized",witness.context_initialized},
  {"rad_DEVICE_allocation_calls",device_allocations.load()},{"rad_stream_create_calls",streams.load()},{"rad_event_create_calls",events.load()},
  {"rad_async_copy_calls",copies.load()},{"rad_async_clear_calls",clears.load()},
  {"sampler_pinned_allocation_calls",pinned_allocations.load()},{"sampler_pinned_requested_bytes",pinned_requested.load()},
  {"engine_retired",witness.engine_retired},{"repeat_start_status",retry},{"failed_phase",uint32_t(snapshot.failed_phase)},
  {"GPU_model_execution",false},{"full_source_factory_called",false},
  {"scope","same production Engine declaration prefix/actual all4 Program and geometry witnesses, closed before publication; AQL/HIP/driver/plugins and Sampler pinned resources may exist; rad API wrappers exclude opaque driver/vendor activity, not a zero-GPU-activity or RSS claim"}};
 std::cout<<report.dump(2)<<'\n';return valid?0:1;
}catch(const std::exception& e){std::cerr<<"FAIL actual Engine declaration context: "<<e.what()<<'\n';return 1;}}

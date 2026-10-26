// HOST refusal/one-attempt lifecycle for the actual declaration-only entry.
// No synthetic successful Engine/Program/source/DEVICE capability is fabricated.
#include "engine.h"
#include "engine_glm_main.h"
#include "device/device.h"
#include <array>
#include <cstdarg>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace rad;
namespace {
thread_local bool observing=false;
uint64_t calls=0;size_t checks=0;
void need(bool yes,const char* why){++checks;if(!yes)throw std::runtime_error(why);}
struct Observe {Observe(){need(!observing,"one test scope");observing=true;}~Observe(){observing=false;}};
GlmMainEngineOptions options(int rows){GlmMainEngineOptions o;o.checkpoint="/closed/model";o.home="/closed/home";
 o.bundle_root="/closed/bundle";o.source_proof="/closed/proof";o.expected_bundle_id=17;o.prefill_max_tok=rows;return o;}
void refuse_before_open(int rows){
 std::unique_ptr<Engine> engine;need(GlmMainEngineHarness::create(options(rows),&engine)==RAD_OK,"cold actual Engine shell");
 GlmMainDeclarationContext out;out.requested_max_tok=91;out.declared_ranks=3;out.engine_retired=true;
 std::array<unsigned char,sizeof(out)> before{};std::memcpy(before.data(),&out,sizeof(out));
 {Observe scope;const auto prior=calls;
  need(GlmMainEngineHarness::declaration_preflight(*engine,nullptr)==RAD_E_INVAL,"null output refuses before attempt");
  need(GlmMainEngineHarness::declaration_preflight(*engine,&out)==RAD_E_UNSUPPORTED,"HOST never simulates actual AQL declaration");
  need(std::memcmp(before.data(),&out,sizeof(out))==0,"failed actual declaration leaves valid caller object byte-exact");
  need(calls==prior,"HOST refusal before source open/read/device alloc/application stream");
  need(GlmMainEngineHarness::declaration_preflight(*engine,&out)==RAD_E_UNSUPPORTED&&engine->start()==RAD_E_UNSUPPORTED,"first error sticky across declaration and real start; no retry source work");
  need(calls==prior,"retry is no source/backend work");
 }
 GlmMainEngineSnapshot snapshot;need(GlmMainEngineHarness::snapshot(*engine,&snapshot)==RAD_OK,"closed actual Engine snapshot");
 need(snapshot.attempts==1&&snapshot.retired&&snapshot.phase==GlmMainEnginePhase::Terminal&&
      snapshot.failed_phase==GlmMainEnginePhase::ColdAdmitted&&snapshot.first_status==RAD_E_UNSUPPORTED&&
      !snapshot.source_admitted&&!snapshot.transport_ready&&!snapshot.source_bytes_verified,"single attempt terminal ownership without source capability");
 need(GlmMainEngineHarness::retire(*engine)==RAD_OK,"idempotent closed retirement");
}
}
extern "C" int __real_open(const char*,int,...);
extern "C" int __wrap_open(const char* path,int flags,...){if(observing)++calls;
 if((flags&O_CREAT)||((flags&O_TMPFILE)==O_TMPFILE)){va_list args;va_start(args,flags);const mode_t mode=va_arg(args,mode_t);va_end(args);return __real_open(path,flags,mode);}return __real_open(path,flags);}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* out,size_t n,off_t off){if(observing)++calls;return __real_pread(fd,out,n,off);}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t n,int kind){if(observing)++calls;return __real_rad_dev_alloc(n,kind);}
extern "C" int __real_rad_stream_create(RadStream*,int);
extern "C" int __wrap_rad_stream_create(RadStream* out,int device){if(observing)++calls;return __real_rad_stream_create(out,device);}
int main(int argc,char** argv){try{
 need(argc==2&&!std::strcmp(argv[1],"--run-host-declaration-context"),"explicit HOST declaration lifecycle controls");
 need(device_is_host(),"HOST-only negative controls");refuse_before_open(1);refuse_before_open(4);
 std::cout<<"PASS same-Engine declaration HOST refusal/lifecycle: 2 groups "<<checks<<" checks; no positive actual AQL declaration/source/model claim\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL declaration HOST controls: "<<e.what()<<'\n';return 1;}}

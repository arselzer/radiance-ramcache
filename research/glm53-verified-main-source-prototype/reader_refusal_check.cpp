// HOST, negative metadata only. Never constructs fake complete coverage,
// a private verified object, model payload, GPU/Engine or RuntimeReady.
#include "main_bundle_reader.h"
#include "allocation_hooks.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
namespace vm=glm53::verified_main;namespace fs=std::filesystem;using Json=nlohmann::json;
namespace {size_t checks=0,groups=0,refusals=0;std::atomic<bool> tracing{false};std::atomic<size_t> opens{0},reads{0},device_allocations{0};
void need(bool b,const char* why){++checks;if(!b)throw std::runtime_error(why);}
void trace(){opens=reads=device_allocations=0;tracing=true;}void stop(){tracing=false;}
uint64_t checksum(const std::string& s){uint64_t h=14695981039346656037ull;for(unsigned char c:s){h^=c;h*=1099511628211ull;}return h;}
struct Temp {fs::path path;Temp(){char p[]="/tmp/glm53-main-reader-negative-XXXXXX";const auto* made=::mkdtemp(p);need(made,"temporary negative metadata directory");path=made;}~Temp(){std::error_code e;fs::remove_all(path,e);}};
void file(const fs::path& p,const std::string& s){std::ofstream f(p,std::ios::binary);f.write(s.data(),std::streamsize(s.size()));need(bool(f),"negative metadata write");}
void scalar_refusals(){++groups;rad::Checkpoint ck;vm::Request request{"/tmp/absent-not-a-model","/tmp/absent-proof",1};
 for(int mutation=0;mutation<14;++mutation){auto q=request;vm::ReaderLimits l;std::shared_ptr<const vm::VerifiedBundle> out;
  switch(mutation){case 0:q.expected_bundle_id=0;break;case 1:l.metadata_bytes=1;break;case 2:l.rank_metadata_bytes=1;break;case 3:l.authority_bytes=0;break;case 4:l.authority_bytes=l.rank_metadata_bytes+1;break;case 5:l.manifest_bytes=0;break;case 6:l.proof_bytes=131073;break;case 7:l.index_bytes=(16u<<20)+1;break;case 8:l.header_bytes=0;break;case 9:l.window_bytes=0;break;case 10:l.window_bytes=65537;break;case 11:l.max_json_nodes=300001;break;case 12:l.max_json_nodes=1;break;case 13:l.rank_metadata_bytes=8u<<20;l.authority_bytes=1u<<20;break;}
  trace();const auto status=vm::VerifiedBundle::open(q,ck,l,&out);stop();need(status!=vm::Error::ok&&!out&&!opens&&!reads&&!device_allocations,"scalar/cap refusal reached file/payload/device work or published");++refusals;}
 trace();const auto bad=vm::VerifiedBundle::open(request,ck,{},nullptr);stop();need(bad==vm::Error::invalid&&!opens&&!reads,"null output refused before I/O");++refusals;
}
void subset_is_never_authority(){++groups;Temp t;rad::Checkpoint ck;const std::string proof="negative-metadata-only-not-a-model-proof";file(t.path/"source-proof.json",proof);
 Json identity{{"format_profile","glm53-main-tp4-canonical-mxfp4-e8m0-g32-v1"},{"source_content_id",1},{"mode","subset_component_NONREADY"},{"world",4},{"records",8},{"ordered_catalog_checksum",1},{"layouts",Json::array()},{"shards",Json::array()}};
 const auto id=checksum(identity.dump());Json bundle{{"schema","glm53-main-four-rank-data-bundle-v1"},{"identity",identity},{"bundle_id",id},{"source_proof_checksum",checksum(proof)},{"source_proof_complete",true},{"trusted_producer_transcript_not_independent_Cpp_crypto",true},{"runtime_ready",false}};
 // Deliberately untrusted subset metadata. No complete record/shard/payload
 // universe is fabricated; exact failure must precede proof/source inspection.
 file(t.path/"bundle.json",bundle.dump());vm::Request request{t.path.string(),(t.path/"source-proof.json").string(),id};std::shared_ptr<const vm::VerifiedBundle> out;std::string why;
 trace();const auto status=vm::VerifiedBundle::open(request,ck,{},&out,&why);stop();need(status==vm::Error::identity&&!out&&why=="complete four-rank data domain, never subset","subset did not reach intended complete-domain guard");need(opens>0&&reads>0&&!device_allocations,"negative persisted metadata seam was not exercised");++refusals;
 // Intended first allocation refusal, with preexisting files and unchanged caller.
 allocation_fault::begin();allocation_fault::remaining=0;const auto oom=vm::VerifiedBundle::open(request,ck,{},&out);allocation_fault::end();need(oom==vm::Error::no_memory&&!out&&!allocation_fault::live,"reader OOM publication/cleanup");++refusals;
 auto limited=vm::ReaderLimits{};limited.metadata_bytes=8u<<20;trace();const auto budget=vm::VerifiedBundle::open(request,ck,limited,&out);stop();need(budget==vm::Error::budget&&!out&&!opens&&!reads,"reader bulk metadata reservation failed after opening files");++refusals;
 bundle["bundle_id"]=id+1;file(t.path/"bundle.json",bundle.dump());need(vm::VerifiedBundle::open(request,ck,{},&out)==vm::Error::identity&&!out,"foreign trusted common ID accepted");++refusals;
}
}
extern "C" int __real_open(const char*,int,...);extern "C" int __real_openat(int,const char*,int,...);extern "C" ssize_t __real_pread(int,void*,size_t,off_t);extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" int __wrap_open(const char* p,int flags,...){if(tracing)++opens;if(flags&O_CREAT){va_list ap;va_start(ap,flags);const auto mode=va_arg(ap,int);va_end(ap);return __real_open(p,flags,mode);}return __real_open(p,flags);}
extern "C" int __wrap_openat(int fd,const char* p,int flags,...){if(tracing)++opens;if(flags&O_CREAT){va_list ap;va_start(ap,flags);const auto mode=va_arg(ap,int);va_end(ap);return __real_openat(fd,p,flags,mode);}return __real_openat(fd,p,flags);}
extern "C" ssize_t __wrap_pread(int fd,void* p,size_t n,off_t at){if(tracing)++reads;return __real_pread(fd,p,n,at);}
extern "C" void* __wrap_rad_dev_alloc(int64_t n,int kind){if(tracing)++device_allocations;return __real_rad_dev_alloc(n,kind);}
int main(int argc,char** argv){try{need(argc==2&&!std::strcmp(argv[1],"--run-host-refusals"),"explicit --run-host-refusals required");scalar_refusals();subset_is_never_authority();std::cout<<"PASS actual-main Reader refusals groups="<<groups<<" checks="<<checks<<" refusals="<<refusals<<"; no complete/source/RuntimeReady authority or model/GPU execution\n";return 0;}catch(const std::exception& e){stop();allocation_fault::end();std::cerr<<"FAIL actual-main Reader refusals: "<<e.what()<<'\n';return 1;}}

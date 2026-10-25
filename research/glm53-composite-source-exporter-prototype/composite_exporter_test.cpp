// Actual metadata-only Builder/Planner/PlaneShare; no kernel, GPU or model execution.
#include "native_record_store.h"
#include "glm53_weight_allocation.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <new>
#include <numeric>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
namespace ce=rad::composite_export;namespace gp=rad::glm53_weight_probe;namespace fs=std::filesystem;
using Json=nlohmann::json;
namespace oom {thread_local int64_t remaining=-1;
void test(){if(remaining>=0&&remaining--==0){remaining=-1;throw std::bad_alloc();}}
void* allocate(size_t n){test();
  if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* aligned(size_t n,size_t a){test();void* p=nullptr;
  if(::posix_memalign(&p,std::max(a,sizeof(void*)),n?n:1))throw std::bad_alloc();return p;}}
void* operator new(size_t n){return oom::allocate(n);}void* operator new[](size_t n){return oom::allocate(n);}
void* operator new(size_t n,const std::nothrow_t&)noexcept{try{return oom::allocate(n);}catch(...){return nullptr;}}
void* operator new[](size_t n,const std::nothrow_t&)noexcept{try{return oom::allocate(n);}catch(...){return nullptr;}}
void* operator new(size_t n,std::align_val_t a){return oom::aligned(n,size_t(a));}void* operator new[](size_t n,std::align_val_t a){return oom::aligned(n,size_t(a));}
void* operator new(size_t n,std::align_val_t a,const std::nothrow_t&)noexcept{try{return oom::aligned(n,size_t(a));}catch(...){return nullptr;}}
void* operator new[](size_t n,std::align_val_t a,const std::nothrow_t&)noexcept{try{return oom::aligned(n,size_t(a));}catch(...){return nullptr;}}
void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}void operator delete[](void* p,size_t)noexcept{std::free(p);}
void operator delete(void* p,const std::nothrow_t&)noexcept{std::free(p);}void operator delete[](void* p,const std::nothrow_t&)noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t)noexcept{std::free(p);}void operator delete[](void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete(void* p,size_t,std::align_val_t)noexcept{std::free(p);}void operator delete[](void* p,size_t,std::align_val_t)noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&)noexcept{std::free(p);}void operator delete[](void* p,std::align_val_t,const std::nothrow_t&)noexcept{std::free(p);}
namespace faults {int reads=-1,writes=-1,arm_read_after_writes=-1;uint64_t read_calls=0,write_calls=0;int after_reads=-1;void(*after)()=nullptr;const char* create_destination=nullptr;bool created_destination=false;}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);extern "C" ssize_t __real_pwrite(int,const void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* p,size_t n,off_t o){++faults::read_calls;
  if(faults::reads>=0&&faults::reads--==0){faults::reads=-1;errno=EIO;return -1;}const auto k=__real_pread(fd,p,n,o);if(k>0&&faults::after&&faults::after_reads--==0){const auto f=faults::after;faults::after=nullptr;f();}return k;}
extern "C" ssize_t __wrap_pwrite(int fd,const void* p,size_t n,off_t o){++faults::write_calls;
  if(faults::writes>=0&&faults::writes--==0){faults::writes=-1;errno=EIO;return -1;}const auto k=__real_pwrite(fd,p,n,o);if(k>0&&faults::create_destination){const char* destination=faults::create_destination;faults::create_destination=nullptr;faults::created_destination=::mkdir(destination,0700)==0;}
  if(k>0&&faults::arm_read_after_writes>=0&&faults::arm_read_after_writes--==0){faults::arm_read_after_writes=-1;faults::reads=1;}return k;}
namespace {
size_t cases=0,checks=0,compared=0,padding=0;const char* current="startup";
void need(bool b,const std::string& why){++checks;
  if(!b)throw std::runtime_error(why);}
void good(ce::Status e,const std::string& why){need(e==ce::Status::ok,why+": "+ce::name(e));}
struct Temp {fs::path path;Temp(){char n[]="/tmp/glm53-composite-XXXXXX";const char* p=::mkdtemp(n);need(p,"mkdtemp");path=p;}~Temp(){std::error_code e;fs::remove_all(path,e);}};
void write(const fs::path& p,std::span<const std::byte> b){std::ofstream f(p,std::ios::binary|std::ios::trunc);need(bool(f),"write open");f.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));need(bool(f),"write complete");}
std::vector<std::byte> read(const fs::path& p){const auto n=fs::file_size(p);need(n<=2u<<20,"bounded read fixture");std::vector<std::byte> b(size_t(n),std::byte{});std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));need(bool(f)||b.empty(),"bounded read");return b;}
uint8_t pattern(int layer,int expert,ce::Part part,int plane,uint64_t row,uint64_t byte){
 const uint32_t seed=uint32_t(17+layer*71+expert*29+int(part)*41+plane*113);
 if(layer&&plane){const uint32_t word=0x3e000000u|((seed*97u+uint32_t(row)*193u+uint32_t(byte/4)*359u)&0x007fffffu);return uint8_t(word>>(8*(byte%4)));}
 if(plane)return uint8_t(120+(seed+row+byte)%9);
 return uint8_t(seed+row*73+byte*19+(row/128)*11);
}
struct Fixture {
 Temp temp;rad::Registry registry;ce::SourceCatalog catalog;std::map<std::string,std::vector<std::byte>> originals;
 std::array<std::unique_ptr<rad::Builder>,4> builders;std::array<rad::Plan,4> plans;std::vector<ce::RecordSpec> specs;
 Fixture(){catalog.source_id="complete-synthetic-separated-physical-planes-v1";catalog.revision="synthetic-not-checkpoint";
  for(int l=0;l<2;++l)for(int e=0;e<2;++e)for(bool down:{false,true}){
   ce::SourceTensor t;t.logical=gp::logical(l,e,down);t.layer=l;t.expert=e;t.projection=down?ce::Projection::down:ce::Projection::gate_up;
   t.encoding=gp::encoding(l?gp::Format::fp8_f32:gp::Format::mxfp4);t.shape={down?128:1024,down?512:128};
   for(int p=0;p<2;++p){auto& plane=t.planes[size_t(p)];std::vector<std::byte> joined;
    const uint64_t rows=p&&l?uint64_t(t.shape[0]/128):uint64_t(t.shape[0]);
    const uint64_t rb=p?(l?uint64_t(t.shape[1]/128*4):uint64_t(t.shape[1]/32)):(l?uint64_t(t.shape[1]):uint64_t(t.shape[1]/2));
    const int np=down?1:2;
    for(int part=0;part<np;++part){ce::PhysicalPlane x;x.part=down?ce::Part::down:(part?ce::Part::up:ce::Part::gate);x.layer=l;x.expert=e;x.dtype=t.encoding.plane[p].dtype;
     x.rows=rows/uint64_t(np);x.row_bytes=rb;x.offset=64;x.bytes=x.rows*rb;x.complete=true;
     x.tensor="synthetic.layer."+std::to_string(l)+".expert."+std::to_string(e)+"."+std::to_string(int(x.part))+(p?".scale":".codes");
     x.path=(temp.path/(x.tensor+".bin")).string();std::vector<std::byte> file(size_t(x.offset+x.bytes+64),std::byte{0xa5});
     for(uint64_t row=0;row<x.rows;++row)for(uint64_t b=0;b<rb;++b)file[size_t(x.offset+row*rb+b)]=std::byte(pattern(l,e,x.part,p,row,b));
     x.checksum=ce::checksum({file.data()+x.offset,size_t(x.bytes)});joined.insert(joined.end(),file.begin()+int64_t(x.offset),file.begin()+int64_t(x.offset+x.bytes));
     write(x.path,file);originals.emplace(x.path,std::move(file));plane.pieces.push_back(std::move(x));
    }
    plane.bytes=joined.size();plane.checksum=ce::checksum(joined);
   }catalog.tensors.push_back(std::move(t));
  }
  gp::Inventory inv;inv.revision="synthetic-layout-only";inv.main_layers=1;inv.first_dense=0;inv.mtp_layers=1;inv.experts=2;inv.topk=2;inv.hidden=128;inv.nff=512;inv.layers={{0,gp::Format::mxfp4},{1,gp::Format::fp8_f32}};
  for(int rank=0;rank<4;++rank){builders[size_t(rank)]=gp::build(registry,inv,rank);rad::Config c;c.placement="all_vram";rad::PlannerInput pi{&builders[size_t(rank)]->program(),&c,rank,4,nullptr,0,true,false};need(rad::Planner(pi).plan(&plans[size_t(rank)])==RAD_OK,"actual pure Planner");}
  specs=capture(catalog);
 }
 std::vector<ce::RecordSpec> capture(const ce::SourceCatalog& c){std::vector<ce::RecordSpec> out;for(int rank=0;rank<4;++rank)for(size_t u=0;u<plans[size_t(rank)].units.size();++u)if(plans[size_t(rank)].units[u].expert>=0){ce::RecordSpec s;std::string why;good(ce::capture_unit(builders[size_t(rank)]->program(),plans[size_t(rank)],u,c,&s,&why),"capture actual shares: "+why);out.push_back(std::move(s));}return out;}
 ce::Receipt pack(const fs::path& dst){ce::Receipt out;std::string why;good(ce::pack(specs,catalog,dst.string(),{},&out,&why),"pack: "+why);return out;}
 std::unique_ptr<ce::Reader> open(const fs::path& dir,const ce::Receipt& rc){std::unique_ptr<ce::Reader> out;std::string why;good(ce::Reader::open(dir.string(),specs,catalog,rc.manifest_checksum,{},&out,&why),"open: "+why);return out;}
 std::vector<std::byte> expected(const ce::RecordSpec& spec){
  const auto k=spec.key();const bool mtp=k.layer==1;need(k.world==4&&k.rank<4&&k.expert>=0&&k.expert<2&&k.record_class==(mtp?ce::RecordClass::mtp_fp8_f32:ce::RecordClass::main_mxfp4_e8m0),"oracle source key");
  const std::array<uint64_t,4> off=mtp?std::array<uint64_t,4>{0,32768,36864,53248}:std::array<uint64_t,4>{0,16384,20480,28672};
  const std::array<uint64_t,4> sizes=mtp?std::array<uint64_t,4>{32768,8,16384,4}:std::array<uint64_t,4>{16384,1024,8192,512};
  const uint64_t stride=mtp?57344:32768;need(spec.stride()==stride&&spec.member_offsets()==off&&spec.member_bytes()==sizes,"actual metadata class footprint");std::vector<std::byte> out(size_t(stride),std::byte{});
  // Independent explicit source coordinates, not PlaneShare or composite-read helpers.
  for(int m=0;m<4;++m){const bool dn=m>=2,sc=m%2;const uint64_t rows=dn?(sc&&mtp?1:128):(sc&&mtp?2:256),rb=sizes[size_t(m)]/rows;
   for(uint64_t row=0;row<rows;++row){ce::Part part=ce::Part::down;uint64_t sr=row,sb=uint64_t(k.rank)*rb;
    if(!dn){const uint64_t half=sc&&mtp?1:128;part=row<half?ce::Part::gate:ce::Part::up;sr=uint64_t(k.rank)*half+row%half;sb=0;}
    for(uint64_t b=0;b<rb;++b)out[size_t(off[size_t(m)]+row*rb+b)]=std::byte(pattern(k.layer,k.expert,part,sc?1:0,sr,sb+b));
   }
  }
  padding+=size_t(stride-std::accumulate(sizes.begin(),sizes.end(),uint64_t{}));return out;
 }
 bool clean(const std::string& prefix){for(const auto& p:fs::directory_iterator(temp.path))if(p.path().filename().string().starts_with(prefix))return false;return true;}
};
void roundtrip(){++cases;Fixture f;const auto dst=f.temp.path/"ready";const auto rc=f.pack(dst);need(rc.records==16&&rc.segments==4&&rc.total_bytes==720896&&rc.peak_io_buffer==4096&&rc.metadata_reserved==(16u<<20)&&rc.peak_input_fds==2,"bounded four-rank publication receipt");auto reader=f.open(dst,rc);
 for(const auto& s:f.specs){std::vector<std::byte> out;good(reader->read(s.key(),&out),"record read");need(out==f.expected(s),"independent all native/nibble/F32/padding bytes");compared+=out.size();}
 for(const auto& [p,b]:f.originals)need(read(p)==b,"physical source mutated");
 const std::string hello="hello";need(ce::checksum({reinterpret_cast<const std::byte*>(hello.data()),hello.size()})==0xa430d84680aabd0bull,"FNV known answer");
}
void capture_errors(){++cases;Fixture f;auto& p=f.builders[0]->program();auto& plan=f.plans[0];const auto u=size_t(std::find_if(plan.units.begin(),plan.units.end(),[](const auto& x){return x.expert>=0;})-plan.units.begin());const auto h=plan.units[u].weights[0];const auto old=p.weights[h];
 for(int kind=0;kind<8;++kind){if(kind==0)p.weights[h].stored_bytes++;
  if(kind==1)p.weights[h].stored_dtype=RAD_BF16;
  if(kind==2)p.weights[h].layout_tag="forged";
  if(kind==3)p.weights[h].widen=true;
  if(kind==4)p.weights[h].shard_hi=INT64_MAX;
  if(kind==5)p.weights[h].enc.transform[0]='x';
  if(kind==6)p.weights[h].stored_rank=3;
  if(kind==7){p.weights[h].stored_shape[0]/=2;p.weights[h].stored_shape[1]*=2;}
  auto out=f.specs[0];const auto keep=out.metadata_json();need(ce::capture_unit(p,plan,u,f.catalog,&out)!=ce::Status::ok&&out.metadata_json()==keep,"malformed native metadata output atomicity");p.weights[h]=old;}
 const auto dh=plan.units[u].weights[2];const auto down=p.weights[dh];p.weights[dh].shard_lo=1;p.weights[dh].shard_hi=129;ce::RecordSpec split;need(ce::capture_unit(p,plan,u,f.catalog,&split)!=ce::Status::ok,"down K cut splits packed/scale block");p.weights[dh]=down;
 const auto before=plan.units[u].offset[1];plan.units[u].offset[1]=0;ce::RecordSpec out;need(ce::capture_unit(p,plan,u,f.catalog,&out)!=ce::Status::ok,"overlapping members");plan.units[u].offset[1]=before;
}
void piece_errors(){++cases;Fixture f;
 for(int kind=0;kind<13;++kind){auto c=f.catalog;auto& x=c.tensors[0].planes[0].pieces[0];
  if(kind==0)x.complete=false;
  if(kind==1)x.part=ce::Part::up;
  if(kind==2)x.layer=1;
  if(kind==3)x.expert=1;
  if(kind==4)x.dtype=RAD_BF16;
  if(kind==5)x.rows--;
  if(kind==6)x.bytes--;
  if(kind==7)x.offset=UINT64_MAX;
  if(kind==8)std::swap(c.tensors[0].planes[0].pieces[0],c.tensors[0].planes[0].pieces[1]);
  if(kind==9)c.tensors[0].planes[0].pieces.pop_back();
  if(kind==10)x.bytes=x.row_bytes;
  if(kind==11){x.rows=1;x.bytes=x.row_bytes;}
  if(kind==12)x.path=std::string(513,'x');
  const auto u=size_t(std::find_if(f.plans[0].units.begin(),f.plans[0].units.end(),[](const auto& x){return x.expert>=0;})-f.plans[0].units.begin());
  ce::RecordSpec out=f.specs[0];const auto old=out.metadata_json();need(ce::capture_unit(f.builders[0]->program(),f.plans[0],u,c,&out)!=ce::Status::ok&&out.metadata_json()==old,"physical source coverage/map refusal");
 }
 // The real saved samples are partial rows only. The complete-plane API refuses this
 // exact evidence category without reading/downloading those actual files.
 auto partial=f.catalog;partial.tensors[0].planes[0].pieces[0].complete=false;partial.source_id="actual-partial-row-evidence-not-native-ready";
 ce::Receipt out{55};need(ce::pack(f.specs,partial,(f.temp.path/"partial").string(),{},&out)!=ce::Status::ok&&out==ce::Receipt{55}&&f.clean("partial"),"partial row evidence became a full record");
}
void source_errors(){++cases;Fixture f;const auto dst=f.temp.path/"no-ready";ce::Receipt out{77};
 auto c=f.catalog;c.tensors.back().planes[1].pieces.back().checksum^=1;auto specs=f.capture(c);need(ce::pack(specs,c,dst.string(),{},&out)==ce::Status::checksum&&out==ce::Receipt{77}&&f.clean("no-ready"),"late physical checksum failure");
 c=f.catalog;c.tensors.back().planes[1].checksum^=1;specs=f.capture(c);need(ce::pack(specs,c,dst.string(),{},&out)==ce::Status::checksum&&f.clean("no-ready"),"ordered logical checksum failure");
 const auto& x=f.catalog.tensors.front().planes[0].pieces.front();fs::resize_file(x.path,x.offset+x.row_bytes);need(ce::pack(f.specs,f.catalog,dst.string(),{},&out)!=ce::Status::ok&&out==ce::Receipt{77}&&f.clean("no-ready"),"truncated physical plane published");
}
void io_transactions(){++cases;
 for(int kind=0;kind<2;++kind){Fixture f;const auto dst=f.temp.path/"io-failed";const auto path=dst.string();ce::Receipt out{81};const auto before=faults::write_calls;
  if(kind)faults::arm_read_after_writes=12;else faults::writes=12;
  const auto e=ce::pack(f.specs,f.catalog,path,{},&out);faults::writes=faults::reads=faults::arm_read_after_writes=-1;
  need(e==ce::Status::io&&faults::write_calls>before+12&&out==ce::Receipt{81}&&f.clean("io-failed"),"late read/write did not clean exclusive owned generation");
  f.pack(dst);need(fs::exists(dst/"manifest.json"),"I/O failure prevented fresh independent retry");
 }
}
fs::path change_path;bool changed=false;
void change_stamp(){struct stat st{};if(::stat(change_path.c_str(),&st))return;timespec times[2]{st.st_atim,st.st_mtim};++times[1].tv_sec;changed=::utimensat(AT_FDCWD,change_path.c_str(),times,0)==0;}
void output_errors(){++cases;Fixture f;const auto dst=f.temp.path/"ready";const auto rc=f.pack(dst);auto kept=f.open(dst,rc);auto* ptr=kept.get();const auto raw=read(dst/"manifest.json");const auto base=Json::parse(reinterpret_cast<const char*>(raw.data()),reinterpret_cast<const char*>(raw.data()+raw.size()));
 change_path=dst/"manifest.json";changed=false;faults::after_reads=1;faults::after=change_stamp;
 const auto admission=ce::Reader::open(dst.string(),f.specs,f.catalog,rc.manifest_checksum,{},&kept);faults::after=nullptr;
 need(changed&&admission==ce::Status::identity&&kept.get()==ptr,"manifest mutation during segment admission changed ready Reader");
 for(int kind=0;kind<6;++kind){auto j=base;
  if(kind==0)j["records"][0]["spec"]["rank"]=3;
  if(kind==1)j["records"][0]["spec"]["class"]="other";
  if(kind==2)j["source_catalog"]["source_id"]="foreign";
  if(kind==3)j["records"][0]["offset"]=UINT64_MAX;
  if(kind==4)j["checkpoint_payload_verified"]=true;
  if(kind==5)j["segments"][0]["name"]="../foreign";
  const auto text=j.dump();const auto b=std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),text.size());write(dst/"manifest.json",b);need(ce::Reader::open(dst.string(),f.specs,f.catalog,ce::checksum(b),{},&kept)!=ce::Status::ok&&kept.get()==ptr,"malformed composite manifest changed caller Reader");
 }
 write(dst/"manifest.json",raw);auto seg=read(dst/"segment-00000000.bin");seg[18000]^=std::byte{1};write(dst/"segment-00000000.bin",seg);std::vector<std::byte> out(3,std::byte{9}),old=out;need(kept->read(f.specs[0].key(),&out)!=ce::Status::ok&&out==old,"changed generation changed caller bytes");
 std::unique_ptr<ce::Reader> failed;need(ce::Reader::open(dst.string(),f.specs,f.catalog,rc.manifest_checksum,{},&failed)==ce::Status::checksum&&!failed,"native output corruption not checked");
 auto j=base;j["records"][0]["checksum"]=ce::checksum({seg.data(),32768});const auto text=j.dump();const auto forged=std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),text.size());write(dst/"manifest.json",forged);need(ce::Reader::open(dst.string(),f.specs,f.catalog,ce::checksum(forged),{},&failed)==ce::Status::invalid&&!failed,"nonzero padding accepted with recomputed checksum");
}
void budgets_publish(){++cases;Fixture f;const auto path=(f.temp.path/"limit").string();ce::Receipt out{99};
 for(int kind=0;kind<8;++kind){ce::Limits l;
  if(kind==0)l.max_records=15;
  if(kind==1){l.max_total_bytes=l.max_segment_bytes=4096;}
  if(kind==2)l.max_segment_bytes=4096;
  if(kind==3)l.max_manifest_bytes=1;
  if(kind==4)l.max_physical_planes=1;
  if(kind==5)l.max_sources=1;
  if(kind==6)l.max_metadata_bytes=1;
  if(kind==7)l.max_open_fds=1;
  const auto reads=faults::read_calls;need(ce::pack(f.specs,f.catalog,path,l,&out)==(kind>=6?ce::Status::invalid:ce::Status::budget)&&out==ce::Receipt{99}&&faults::read_calls==reads&&f.clean("limit"),"cap refusal occurred after payload I/O/publication");
 }
 auto missing=f.specs;missing.pop_back();need(ce::pack(missing,f.catalog,path,{},&out)==ce::Status::identity&&f.clean("limit"),"missing rank/expert coverage accepted");
 const auto late=(f.temp.path/"late-publish").string();faults::create_destination=late.c_str();faults::created_destination=false;
 const auto writes=faults::write_calls;const auto late_error=ce::pack(f.specs,f.catalog,late,{},&out);faults::create_destination=nullptr;
 need(faults::created_destination&&late_error==ce::Status::exists&&faults::write_calls>writes&&out==ce::Receipt{99}&&fs::is_empty(late),"late NOREPLACE collision replaced destination/receipt");
 need(fs::remove(late)&&f.clean("late-publish"),"late publication collision leaked owned staging");
 const auto ready=f.pack(path);const auto original=read(fs::path(path)/"manifest.json");need(ce::pack(f.specs,f.catalog,path,{},&out)==ce::Status::exists&&out==ce::Receipt{99}&&read(fs::path(path)/"manifest.json")==original,"NOREPLACE existing destination violated");need(ready.total_bytes==720896,"output budget");
}
void allocation_failure(){++cases;Fixture f;const auto path=(f.temp.path/"oom").string();ce::Receipt out{63};const auto before=faults::read_calls;
 oom::remaining=0;const auto e=ce::pack(f.specs,f.catalog,path,{},&out);oom::remaining=-1;
 need(e==ce::Status::no_memory&&out==ce::Receipt{63}&&faults::read_calls==before&&f.clean("oom"),"OOM changed receipt or opened payload");f.pack(path);
}
}
int main(){try{using Test=void(*)();const std::array<std::pair<const char*,Test>,8> tests{{{"roundtrip",roundtrip},{"capture_errors",capture_errors},{"piece_errors",piece_errors},{"source_errors",source_errors},{"io_transactions",io_transactions},{"output_errors",output_errors},{"budgets_publish",budgets_publish},{"allocation_failure",allocation_failure}}};for(const auto& [name,test]:tests){current=name;test();}need(cases==8&&compared==720896&&padding==118688,"exact source evidence counters");std::cout<<"PASS "<<cases<<" synthetic composite exporter cases, "<<checks<<" checks; native_bytes="<<compared<<" padding="<<padding<<" records=16 ranks=4 window=4096 max-input-FDs=2; actual metadata only, no R4D/Core/GPU/TP runtime or real checkpoint readiness\n";return 0;}catch(const std::exception& e){oom::remaining=-1;std::cerr<<"FAIL case="<<current<<" "<<e.what()<<'\n';return 1;}}

// Tiny generated payloads and actual Builder/Planner/weight_share only; no kernels/devices.
#include "native_record_store.h"
#include "glm53_weight_allocation.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
namespace es=rad::expert_store;
namespace gp=rad::glm53_weight_probe;
namespace fs=std::filesystem;
using Json=nlohmann::json;
namespace {
size_t cases=0,checks=0,compared_bytes=0,padding_bytes=0;
void need(bool v,const std::string& why){++checks;if(!v)throw std::runtime_error(why);}
void good(es::Status status,const std::string& why){need(status==es::Status::ok,why+": "+es::name(status));}
struct Temp {fs::path path;Temp(){char p[]="/tmp/glm53-record-store-XXXXXX";const char* d=::mkdtemp(p);need(d,"mkdtemp");path=d;}~Temp(){std::error_code e;fs::remove_all(path,e);}};
void write(const fs::path& p,std::span<const std::byte> bytes){std::ofstream f(p,std::ios::binary|std::ios::trunc);need(bool(f),"open fixture output");f.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));need(bool(f),"write tiny fixture");}
std::vector<std::byte> read(const fs::path& p){const auto n=fs::file_size(p);need(n<=2u<<20,"test file cap");std::vector<std::byte> b(size_t(n),std::byte{0});std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));need(bool(f)||b.empty(),"read tiny fixture");return b;}
uint64_t rounded(uint64_t n){return (n+4095)/4096*4096;}
uint8_t pattern(int layer,int expert,bool down,int plane,uint64_t row,uint64_t byte){
    const uint32_t seed=uint32_t(layer*71+expert*29+int(down)*17+plane*113);
    if(layer==1&&plane==1){const uint32_t word=0x3e000000u|((seed*97u+uint32_t(row)*193u+uint32_t(byte/4)*359u)&0x007fffffu);return uint8_t(word>>(8*(byte%4)));}
    const uint8_t x=uint8_t(seed+row*73+byte*19+(row/128)*11);
    if(layer==1&&plane==0){const unsigned q=unsigned(x)%254;return uint8_t(q>=127?q+1:q);}
    if(plane==1)return uint8_t(125+(seed+row+byte)%5);
    return x;
}
struct Fixture {
    Temp temp;rad::Registry registry;es::SourceCatalog catalog;std::vector<std::byte> source_image;
    std::array<std::unique_ptr<rad::Builder>,4> builders;std::array<rad::Plan,4> plans;std::vector<es::RecordSpec> specs;
    Fixture(){
        catalog.source_id="synthetic-source-bytes-v1";catalog.revision="synthetic-not-checkpoint-b5688-source-layout";
        const auto image=temp.path/"source.bin";
        for(int layer=0;layer<2;++layer)for(int expert=0;expert<2;++expert)for(bool down:{false,true}){
            es::SourceTensor t;t.logical=gp::logical(layer,expert,down);t.projection=down?es::Projection::down:es::Projection::gate_up;t.encoding=gp::encoding(layer?gp::Format::fp8_f32:gp::Format::mxfp4);t.shape={down?128:1024,down?512:128};
            for(int plane=0;plane<2;++plane){const uint64_t rows=plane&&layer?uint64_t(t.shape[0]/128):uint64_t(t.shape[0]);const uint64_t row_bytes=plane?(layer?uint64_t(t.shape[1]/128*4):uint64_t(t.shape[1]/32)):(layer?uint64_t(t.shape[1]):uint64_t(t.shape[1]/2));const uint64_t offset=rounded(source_image.size()+4096),bytes=rows*row_bytes;source_image.resize(size_t(offset+bytes+4096),std::byte{0xa5});for(uint64_t r=0;r<rows;++r)for(uint64_t b=0;b<row_bytes;++b)source_image[size_t(offset+r*row_bytes+b)]=std::byte(pattern(layer,expert,down,plane,r,b));t.planes[size_t(plane)]={image.string(),offset,bytes,es::checksum({source_image.data()+offset,size_t(bytes)})};}
            catalog.tensors.push_back(std::move(t));
        }
        write(image,source_image);
        gp::Inventory inv;inv.revision="synthetic-layout-only";inv.main_layers=1;inv.first_dense=0;inv.mtp_layers=1;inv.experts=2;inv.topk=2;inv.hidden=128;inv.nff=512;inv.layers={{0,gp::Format::mxfp4},{1,gp::Format::fp8_f32}};
        for(int rank=0;rank<4;++rank){builders[size_t(rank)]=gp::build(registry,inv,rank);rad::Config config;config.placement="all_vram";rad::PlannerInput pi{&builders[size_t(rank)]->program(),&config,rank,4,nullptr,0,true,false};need(rad::Planner(pi).plan(&plans[size_t(rank)])==RAD_OK,"actual pure Planner");}
        specs=capture(catalog);
    }
    std::vector<es::RecordSpec> capture(const es::SourceCatalog& c){std::vector<es::RecordSpec> result;for(int rank=0;rank<4;++rank)for(size_t u=0;u<plans[size_t(rank)].units.size();++u){if(plans[size_t(rank)].units[u].expert<0)continue;es::RecordSpec spec;std::string why;const auto status=es::capture_unit(builders[size_t(rank)]->program(),plans[size_t(rank)],u,c,&spec,&why);good(status,"capture actual weight_share: "+why);result.push_back(std::move(spec));}return result;}
    std::vector<std::byte> expected(const es::RecordSpec& spec){
        const auto key=spec.key();need(key.world==4&&key.rank<4,"expected rank/world");const bool mtp=key.record_class==es::RecordClass::mtp_fp8_f32;need(key.layer==int(mtp),"expected class/layer");
        const std::array<uint64_t,4> offsets=mtp?std::array<uint64_t,4>{0,32768,36864,53248}:std::array<uint64_t,4>{0,16384,20480,28672};const std::array<uint64_t,4> bytes=mtp?std::array<uint64_t,4>{32768,8,16384,4}:std::array<uint64_t,4>{16384,1024,8192,512};const uint64_t stride=mtp?57344:32768;
        need(spec.stride()==stride&&spec.member_offsets()==offsets&&spec.member_bytes()==bytes,"actual per-rank native class offsets/bytes");std::vector<std::byte> out(size_t(stride),std::byte{0});
        // Independent coordinates: NOT PlaneShare rects or packer helpers. Each GU half
        // selects this rank's128 rows; scale ROW axes shrink128 only for MTP. DN selects
        // rank*128 K columns in EVERY source row; scales shrink K by32 or128.
        for(int member=0;member<4;++member){const bool down=member>=2,scale=member%2;const uint64_t rows=down?(scale&&mtp?1:128):(scale&&mtp?2:256);const uint64_t row_bytes=bytes[size_t(member)]/rows;
            for(uint64_t r=0;r<rows;++r){uint64_t source_row=r,source_byte=0;if(!down){const uint64_t half=scale&&mtp?1:128;source_row=(r/half)*(half*4)+uint64_t(key.rank)*half+r%half;}else source_byte=uint64_t(key.rank)*row_bytes;
                for(uint64_t b=0;b<row_bytes;++b)out[size_t(offsets[size_t(member)]+r*row_bytes+b)]=std::byte(pattern(key.layer,key.expert,down,scale?1:0,source_row,source_byte+b));
            }
        }
        padding_bytes+=size_t(stride-std::accumulate(bytes.begin(),bytes.end(),uint64_t(0)));return out;
    }
    es::Receipt pack(const fs::path& destination){es::Receipt receipt;std::string why;const auto status=es::pack(specs,catalog,destination.string(),es::Limits{},&receipt,&why);good(status,"pack: "+why);return receipt;}
    std::unique_ptr<es::Reader> open(const fs::path& store,const es::Receipt& receipt,const es::Limits& l={}){std::unique_ptr<es::Reader> reader;std::string why;const auto status=es::Reader::open(store.string(),specs,catalog,receipt.manifest_checksum,l,&reader,&why);good(status,"open: "+why);return reader;}
};
void roundtrip(){++cases;Fixture f;const auto store=f.temp.path/"ready";const auto receipt=f.pack(store);need(receipt.records==16&&receipt.segments==12&&receipt.total_bytes==720896&&receipt.peak_io_buffer==65536,"segmented streaming receipt");auto reader=f.open(store,receipt);for(const auto& spec:f.specs){std::vector<std::byte> got;good(reader->read(spec.key(),&got),"read exact native record");const auto expected=f.expected(spec);need(got==expected,"independent every-byte ROWparts2/COL/scales/padding oracle");compared_bytes+=got.size();}need(read(f.temp.path/"source.bin")==f.source_image,"source bytes mutated");const std::string hello="hello";need(es::checksum({reinterpret_cast<const std::byte*>(hello.data()),hello.size()})==0xa430d84680aabd0bull,"independent FNV1a64 known answer");}
void capture_refusals(){++cases;Fixture f;auto& p=f.builders[0]->program();const auto u=std::find_if(f.plans[0].units.begin(),f.plans[0].units.end(),[](const auto& v){return v.expert>=0;})-f.plans[0].units.begin();const auto w=f.plans[0].units[size_t(u)].weights[0];const auto saved=p.weights[w];
    auto refuse=[&]{es::RecordSpec out=f.specs[0];const auto old=out.metadata_json();need(es::capture_unit(p,f.plans[0],size_t(u),f.catalog,&out)!=es::Status::ok&&out.metadata_json()==old,"capture malformed layout mutated output");p.weights[w]=saved;};
    p.weights[w].stored_bytes+=1;refuse();p.weights[w].stored_dtype=RAD_BF16;refuse();p.weights[w].layout_tag="wrong.native.layout";refuse();p.weights[w].widen=true;refuse();p.weights[w].shard_hi=INT64_MAX;refuse();p.weights[w].enc.transform[0]='x';refuse();
    const auto original=f.plans[0];f.plans[0].units[size_t(u)].offset[1]=0;refuse();f.plans[0]=original;f.plans[0].units[size_t(u)].slot_bytes+=4096;refuse();f.plans[0]=original;
    auto bad=f.catalog;bad.tensors[0].planes[0].offset=UINT64_MAX;es::RecordSpec spec;need(es::capture_unit(p,f.plans[0],size_t(u),bad,&spec)!=es::Status::ok,"source offset overflow accepted");bad=f.catalog;bad.tensors[0].shape[0]=INT64_MAX;need(es::capture_unit(p,f.plans[0],size_t(u),bad,&spec)!=es::Status::ok,"source dimensions overflow accepted");}
void source_failures(){++cases;Fixture f;const auto store=f.temp.path/"never_ready";es::Receipt out{123,456,789,1,2,3};const auto original=out;
    auto bad=f.catalog;bad.revision="wrong-source-revision";need(es::pack(f.specs,bad,store.string(),es::Limits{},&out)==es::Status::identity,"stale source fingerprint accepted");need(out.manifest_checksum==original.manifest_checksum&&!fs::exists(store),"failed source published");
    bad=f.catalog;bad.tensors.back().planes[1].checksum^=1;const auto changed=f.capture(bad);need(es::pack(changed,bad,store.string(),es::Limits{},&out)==es::Status::checksum,"late source checksum not verified");need(!fs::exists(store)&&out.total_bytes==original.total_bytes,"late checksum published store/receipt");
    const auto n=fs::file_size(f.temp.path/"source.bin");fs::resize_file(f.temp.path/"source.bin",n-4097);need(es::pack(f.specs,f.catalog,store.string(),es::Limits{},&out)!=es::Status::ok&&!fs::exists(store),"truncated source published");}
void metadata_failures(){++cases;Fixture f;const auto store=f.temp.path/"ready";const auto receipt=f.pack(store);auto kept=f.open(store,receipt);auto* const pointer=kept.get();const auto original=read(store/"manifest.json");const auto baseline=Json::parse(reinterpret_cast<const char*>(original.data()),reinterpret_cast<const char*>(original.data()+original.size()));
    auto attempt=[&](Json j){const auto text=j.dump();const auto bytes=std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),text.size());write(store/"manifest.json",bytes);const auto status=es::Reader::open(store.string(),f.specs,f.catalog,es::checksum(bytes),es::Limits{},&kept);need(status!=es::Status::ok&&kept.get()==pointer,"malformed store accepted/output replaced");};
    auto j=baseline;j["records"][0]["spec"]["rank"]=3;attempt(j);j=baseline;j["records"][0]["spec"]["class"]="fp8_e4m3_f32_b128";attempt(j);j=baseline;j["records"][0]["spec"]["members"][0]["layout_tag"]="forged";attempt(j);j=baseline;j["records"][0]["offset"]=UINT64_MAX;attempt(j);j=baseline;j["records"][1]["offset"]=0;attempt(j);j=baseline;j["records"][0]["bytes"]=-1;attempt(j);j=baseline;j["segments"][0]["name"]="../source.bin";attempt(j);j=baseline;j["checkpoint_headers_verified"]=true;attempt(j);j=baseline;j["source_catalog"]["source_id"]="other-source";attempt(j);j=baseline;j["segments"].erase(j["segments"].end()-1);attempt(j);
    const std::string raw="{\"schema\":\"radiance-synthetic-native-record-store-v1\","+baseline.dump().substr(1);const auto duplicate=std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw.data()),raw.size());write(store/"manifest.json",duplicate);need(es::Reader::open(store.string(),f.specs,f.catalog,es::checksum(duplicate),es::Limits{},&kept)==es::Status::invalid&&kept.get()==pointer,"duplicate manifest key accepted");
    write(store/"manifest.json",original);auto l=es::Limits{};l.max_manifest_nodes=1;need(es::Reader::open(store.string(),f.specs,f.catalog,receipt.manifest_checksum,l,&kept)==es::Status::budget&&kept.get()==pointer,"manifest node cap ignored");
    std::vector<std::byte> unchanged(7,std::byte{0x5a});const auto before=unchanged;need(kept->read(f.specs[0].key(),&unchanged)==es::Status::identity&&unchanged==before,"bound reader ignored manifest mutation");write(store/"manifest.json",original);auto fresh=f.open(store,receipt);auto key=f.specs[0].key();key.rank=4;need(fresh->read(key,&unchanged)==es::Status::identity&&unchanged==before,"wrong TP rank read changed output");
    auto wrong=f.catalog;wrong.source_id="new-source";auto specs=f.capture(wrong);need(es::Reader::open(store.string(),specs,wrong,receipt.manifest_checksum,es::Limits{},&kept)==es::Status::identity&&kept.get()==pointer,"fresh source fingerprint not rechecked");}
void payload_failures(){
    ++cases;Fixture f;const auto store=f.temp.path/"ready";const auto receipt=f.pack(store);auto reader=f.open(store,receipt);const auto segment=store/"segment-00000000.bin";const auto original_manifest=read(store/"manifest.json");auto bytes=read(segment);const auto pristine=bytes;bytes[100]^=std::byte{1};write(segment,bytes);std::vector<std::byte> out(3,std::byte{7}),before=out;
    need(reader->read(f.specs[0].key(),&out)==es::Status::identity&&out==before,"mutated segment read was not atomic");std::unique_ptr<es::Reader> failed;need(es::Reader::open(store.string(),f.specs,f.catalog,receipt.manifest_checksum,es::Limits{},&failed)==es::Status::checksum&&!failed,"corrupt record checksum accepted");
    bytes=pristine;bytes[18000]=std::byte{1};write(segment,bytes);auto manifest=Json::parse(reinterpret_cast<const char*>(original_manifest.data()),reinterpret_cast<const char*>(original_manifest.data()+original_manifest.size()));manifest["records"][0]["checksum"]=es::checksum({bytes.data(),32768});const auto text=manifest.dump();const auto forged=std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),text.size());write(store/"manifest.json",forged);
    need(es::Reader::open(store.string(),f.specs,f.catalog,es::checksum(forged),es::Limits{},&failed)==es::Status::invalid&&!failed,"nonzero padding accepted after checksum recomputation");
    write(segment,pristine);write(store/"manifest.json",original_manifest);fs::resize_file(segment,bytes.size()-1);need(es::Reader::open(store.string(),f.specs,f.catalog,receipt.manifest_checksum,es::Limits{},&failed)!=es::Status::ok&&!failed,"truncated segment accepted");
}
void budget_and_existing(){++cases;Fixture f;const auto dest=f.temp.path/"not_ready";es::Receipt out{99};auto missing=f.catalog;missing.tensors[0].planes[0].path=(f.temp.path/"absent").string();
    for(int which=0;which<5;++which){es::Limits l;if(which==0)l.max_records=15;if(which==1){l.max_total_bytes=4096;l.max_segment_bytes=4096;}if(which==2)l.max_segment_bytes=4096;if(which==3)l.max_manifest_bytes=1;if(which==4)l.max_sources=1;const auto s=es::pack(f.specs,missing,dest.string(),l,&out);need(s==es::Status::budget&&!fs::exists(dest)&&out.manifest_checksum==99,"limit did not refuse before missing-source I/O/publication");}
    const auto receipt=f.pack(dest);auto limited=es::Limits{};limited.max_read_bytes=32768;auto reader=f.open(dest,receipt,limited);const auto mtp=std::find_if(f.specs.begin(),f.specs.end(),[](const auto& s){return s.key().record_class==es::RecordClass::mtp_fp8_f32;});std::vector<std::byte> keep(1,std::byte{4});need(reader->read(mtp->key(),&keep)==es::Status::budget&&keep==std::vector<std::byte>{std::byte{4}},"one-record allocation cap");
    need(es::pack(f.specs,f.catalog,dest.string(),es::Limits{},&out)==es::Status::exists&&out.manifest_checksum==99,"existing store overwritten");need(read(dest/"manifest.json").size()>0,"existing manifest lost");}
int late_write_child(){
    Fixture f;const auto dest=f.temp.path/"unpublished";std::signal(SIGXFSZ,SIG_IGN);struct rlimit r{40000,40000};need(!::setrlimit(RLIMIT_FSIZE,&r),"isolated child RLIMIT_FSIZE");es::Receipt result{77};const auto status=es::pack(f.specs,f.catalog,dest.string(),es::Limits{},&result);bool owned_temp=false;for(const auto& e:fs::directory_iterator(f.temp.path))if(e.path().filename().string().find("unpublished.staging.")==0)owned_temp=true;
    need(status==es::Status::io&&result.manifest_checksum==77&&!fs::exists(dest)&&!owned_temp,"late EFBIG path published/leaked owned staging");return 0;
}
void late_write_failure(){
    ++cases;const pid_t child=::fork();need(child>=0,"fork immediate exec HOST fixture");
    if(child==0){::execl("/proc/self/exe","record_store_test","--late-write-child",static_cast<char*>(nullptr));std::_Exit(5);}
    int status=0;while(::waitpid(child,&status,0)<0){need(errno==EINTR,"waitpid");}
    need(WIFEXITED(status)&&WEXITSTATUS(status)==0,"exec child failed actual EFBIG cleanup");
}
}
int main(int argc,char** argv){try{if(argc==2&&!std::strcmp(argv[1],"--late-write-child"))return late_write_child();need(argc==1,"no GPU/model mode exists");roundtrip();capture_refusals();source_failures();metadata_failures();payload_failures();budget_and_existing();late_write_failure();std::cout<<"PASS "<<cases<<" offline synthetic record-store cases, "<<checks<<" checks, compared_native_bytes="<<compared_bytes<<", all_padding_bytes="<<padding_bytes<<"; actual Builder/Planner/weight_share, no native GPU row resolution or model payload\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}

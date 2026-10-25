// Copied bounded synthetic S1 fixture scaffolding; frozen store source remains unchanged.
#include "native_record_store.h"
#include "glm53_weight_allocation.h"
#pragma once
#include <atomic>
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
namespace bridge_test {
inline size_t cases=0,compared_bytes=0,padding_bytes=0;
inline std::atomic<size_t> checks=0;
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
} // namespace bridge_test

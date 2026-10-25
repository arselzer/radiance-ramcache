#include "file_provider.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
using namespace glm53::cache;
namespace {
size_t checks=0,cases=0;
void require(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
uint64_t independent_hash(std::span<const std::byte> bytes){uint64_t h=14695981039346656037ull;for(auto b:bytes){h^=std::to_integer<uint8_t>(b);h*=1099511628211ull;}return h;}
struct Temporary {
    int fd=-1;std::string path;
    Temporary(){char name[]="/tmp/glm53-provider-test-XXXXXX";fd=::mkstemp(name);if(fd<0)throw std::runtime_error("mkstemp");path=name;}
    ~Temporary(){if(fd>=0)::close(fd);::unlink(path.c_str());}
    void put(std::span<const std::byte> bytes){require(::ftruncate(fd,0)==0,"truncate fixture");size_t at=0;while(at<bytes.size()){const ssize_t n=::pwrite(fd,bytes.data()+at,bytes.size()-at,off_t(at));if(n<0&&errno==EINTR)continue;require(n>0,"write fixture");at+=size_t(n);}}
    void text(const std::string& value){put({reinterpret_cast<const std::byte*>(value.data()),value.size()});}
};
struct Fixture {
    Temporary file;FileIdentity identity;std::vector<std::byte> expected;std::vector<RecordRef> records;
    Fixture(){
        identity.source_id="synthetic-source";identity.model_id="synthetic-model";identity.layout_id="opaque-synthetic-layout";identity.class_lengths={{"main@synthetic",8},{"mtp@synthetic",16}};identity.max_file_bytes=1024;identity.max_record_bytes=16;identity.max_records=32;
        expected={std::byte{0},std::byte{1},std::byte{2},std::byte{255},std::byte{4},std::byte{5},std::byte{6},std::byte{7}};for(unsigned i=0;i<16;++i)expected.push_back(std::byte(0xA0u+i));
        RecordRef a;a.key={identity.source_id,identity.model_id,identity.layout_id,"main@synthetic",4,0,3,0};a.length=8;a.checksum=independent_hash(std::span(expected).first(8));
        RecordRef b;b.key={identity.source_id,identity.model_id,identity.layout_id,"mtp@synthetic",4,0,45,1};b.offset=8;b.length=16;b.checksum=independent_hash(std::span(expected).subspan(8));records={a,b};file.put(expected);
    }
    std::unique_ptr<FileProvider> open(){std::unique_ptr<FileProvider> result;require(FileProvider::open(file.path,identity,records,&result)==Error::ok,"open valid provider");return result;}
};
void exact_read_and_classes(){++cases;Fixture f;auto provider=f.open();
    const std::array<std::byte,5> hello{std::byte{'h'},std::byte{'e'},std::byte{'l'},std::byte{'l'},std::byte{'o'}};require(fnv1a64(hello)==0xA430D84680AABD0Bull,"published FNV1a64 hello vector");
    for(size_t r=0;r<f.records.size();++r){std::vector<std::byte> guarded(size_t(f.records[r].length)+2,std::byte{0xCC});const auto result=provider->read(f.records[r],std::span(guarded).subspan(1,guarded.size()-2));require(result.error==Error::ok&&result.bytes_read==f.records[r].length,"exact record read");require(guarded.front()==std::byte{0xCC}&&guarded.back()==std::byte{0xCC},"provider guards");require(std::equal(guarded.begin()+1,guarded.end()-1,f.expected.begin()+ptrdiff_t(f.records[r].offset)),"independent bytes");}
    const auto s=provider->stats();require(s.completed==2&&s.bytes_read==24&&s.pread_calls==2,"real POSIX I/O counts");
}
void immutable_metadata_refusals(){++cases;Fixture f;auto provider=f.open();const auto* original=provider.get();
    for(int fault=0;fault<13;++fault){auto id=f.identity;auto refs=f.records;
        if(fault==0)id.schema="other-schema";
        if(fault==1)id.rank=4;
        if(fault==2)refs[1].key.rank=1;
        if(fault==3)refs[0].key.source_id="different-source";
        if(fault==4)refs[0].key.layout_id="different-layout";
        if(fault==5)refs[1].key.record_class="main@synthetic";
        if(fault==6)refs[0].offset=UINT64_MAX-1;
        if(fault==7)refs[1].offset=16;
        if(fault==8)refs[1].offset=0;
        if(fault==9){refs[1].key=refs[0].key;refs[1].length=8;}
        if(fault==10)id.max_records=1;
        if(fault==11)id.max_total_identity_bytes=1;
        if(fault==12)id.max_file_bytes=8;
        require(FileProvider::open(f.file.path,id,refs,&provider)!=Error::ok,"bad immutable manifest accepted");require(provider.get()==original,"failed open replaced provider");
    }
    std::array<std::byte,8> dst;dst.fill(std::byte{0xCC});const auto before=dst;
    for(int fault=0;fault<5;++fault){auto ref=f.records[0];if(fault==0)++ref.checksum;if(fault==1)++ref.offset;if(fault==2)ref.key.expert=2;if(fault==3)ref.key.rank=1;if(fault==4)ref.key.record_class="mtp@synthetic";require(provider->read(ref,dst).error==Error::identity_mismatch,"forged ref accepted");require(dst==before,"metadata refusal wrote bytes");}
    require(provider->stats().pread_calls==0,"forged record reached pread");
}
void corrupt_and_truncated_files(){++cases;
    {Fixture f;auto bytes=f.expected;bytes.back()^=std::byte{1};f.file.put(bytes);auto provider=f.open();std::array<std::byte,16> dst{};const auto r=provider->read(f.records[1],dst);require(r.error==Error::checksum&&r.bytes_read==16,"corrupt payload not caught");}
    {Fixture f;auto provider=f.open();require(::ftruncate(f.file.fd,9)==0,"truncate after open");std::array<std::byte,16> dst;dst.fill(std::byte{0xCC});const auto before=dst;const auto r=provider->read(f.records[1],dst);require(r.error==Error::identity_mismatch&&r.bytes_read==0,"immutable extent change not caught");require(dst==before,"changed file read wrote destination");std::unique_ptr<FileProvider> other;require(FileProvider::open(f.file.path,f.identity,f.records,&other)==Error::invalid,"truncated record manifest accepted");}
    {Fixture f;const std::string link=f.file.path+".symlink";require(::symlink(f.file.path.c_str(),link.c_str())==0,"make symlink");std::unique_ptr<FileProvider> provider;require(FileProvider::open(link,f.identity,f.records,&provider)!=Error::ok,"symlink file accepted");::unlink(link.c_str());}
}
void cancellation_and_output_sizes(){++cases;Fixture f;auto provider=f.open();std::array<std::byte,8> dst;dst.fill(std::byte{0xCC});const auto before=dst;std::stop_source stop;stop.request_stop();require(provider->read(f.records[0],dst,stop.get_token()).error==Error::cancelled,"cancelled read accepted");require(dst==before,"cancelled read wrote bytes");require(provider->read(f.records[0],std::span(dst).first(7)).error==Error::invalid,"short destination accepted");require(dst==before,"bad destination size wrote bytes");require(provider->stats().pread_calls==0,"refusal attempted read");}
void cache_dedup_and_failed_batch(){++cases;Fixture f;auto corrupted=f.expected;corrupted.back()^=std::byte{1};f.file.put(corrupted);auto provider=f.open();Limits limits;limits.hot_bytes=8;limits.cold_bytes=64;limits.max_record_bytes=16;limits.max_records=8;limits.max_batch=8;Cache cache(limits,*provider);Batch held;require(cache.acquire(std::span(f.records).first(1),&held)==Error::ok,"valid first cache record");const auto old=held.leases[0].bytes().data();const auto attempts=provider->stats().attempts;const std::array<RecordRef,3> demand{f.records[0],f.records[1],f.records[0]};require(cache.acquire(demand,&held)==Error::checksum,"corrupt late record accepted");require(held.leases.size()==1&&held.leases[0].bytes().data()==old,"failed batch changed prior output");require(std::equal(held.leases[0].bytes().begin(),held.leases[0].bytes().end(),f.expected.begin()),"held good record changed");require(provider->stats().attempts==attempts+1,"hit/dedup caused extra I/O");}
void cache_real_io_and_duplicate_union(){++cases;Fixture f;auto provider=f.open();Limits limits;limits.hot_bytes=8;limits.cold_bytes=64;limits.max_record_bytes=16;limits.max_records=8;limits.max_batch=8;Cache cache(limits,*provider);const std::array<RecordRef,4> demand{f.records[1],f.records[0],f.records[1],f.records[0]};Batch first,second;require(cache.acquire(demand,&first)==Error::ok,"mixed-class batch");require(first.leases.size()==2&&first.leases[0].ref()==f.records[1]&&first.leases[1].ref()==f.records[0],"first-occurrence unique order");require(cache.acquire(demand,&second)==Error::ok,"repeated live batch");require(provider->stats().completed==2&&provider->stats().bytes_read==24,"duplicate union reread file");require(cache.stats().duplicate_demands>=4,"duplicate demands not counted");const auto use=cache.accounting().usage();require(use.hot_bytes<=8&&use.cold_bytes<=64,"charged byte budget exceeded");}
nlohmann::json catalog_json(){nlohmann::json j{{"schema","expert-cache-records-v1"},{"source_revision","b5688f25491202978c19c4d036eef579f61bbe07"},{"tp_rank",0},{"source","synthetic metadata fixture of executed catalog geometry"},{"layers",nlohmann::json::array()}};for(int layer=3;layer<=45;++layer)j["layers"].push_back({{"layer",layer},{"experts",288},{"record_bytes",layer==45?6299648:3342336},{"class",layer==45?"fp8_e4m3_f32_b128":"mxfp4_e8m0_g32"}});return j;}
void catalog_validation(){++cases;Temporary file;const auto good=catalog_json();file.text(good.dump());Rank0Catalog out;require(load_rank0_catalog(file.path,&out)==Error::ok,"valid pinned catalog metadata");require(out.layers.size()==43&&out.find(45)->record_bytes==6299648,"catalog actual stride");for(int fault=0;fault<6;++fault){auto j=good;if(fault==0)j["tp_rank"]=1;if(fault==1)j["layers"][0]["experts"]=0;if(fault==2)j["layers"][0]["record_bytes"]=true;if(fault==3)j["layers"][1]["layer"]=3;if(fault==4)j["layers"][42]["record_bytes"]=6292992;if(fault==5)j["source_revision"]="wrong";file.text(j.dump());require(load_rank0_catalog(file.path,&out)!=Error::ok,"malformed catalog accepted");require(out.layers.size()==43&&out.find(45)->record_bytes==6299648,"failed catalog parse mutated output");}}
Rank0Catalog tiny_catalog(){Rank0Catalog catalog;catalog.layers={{3,288,3342336,"mxfp4_e8m0_g32"},{45,288,6299648,"fp8_e4m3_f32_b128"}};return catalog;}
void route_parse_and_atomic_refusals(){++cases;Temporary file;const auto catalog=tiny_catalog();RouteTrace out;
    file.text("step,phase,layer,experts\n0,decode,3,1;2;1\n1,prefill,45,0;287\n");require(load_route_trace(file.path,catalog,{},&out)==Error::ok,"CSV routes");require(out.events.size()==2&&out.events[0].experts==std::vector<int32_t>({1,2,1}),"CSV duplicate routes lost");
    file.text("{\"schema\":\"glm53-route-unions-v1\",\"synthetic\":true,\"events\":[{\"step\":0,\"phase\":\"decode\",\"layer\":3,\"experts\":[[1,2],[1]]}]}");require(load_route_trace(file.path,catalog,{},&out)==Error::ok&&out.synthetic,"JSON nested rows");require(out.events[0].experts==std::vector<int32_t>({1,2,1}),"JSON duplicate rows lost");
    const auto original=out.provenance;
    const std::array<std::string,12> bad{
        "step,phase,layer,experts\n0,decode,3,288\n","step,phase,layer,experts\n0,decode,3,-1\n","step,phase,layer,experts\n0,decode,3,1;\n","step,phase,layer,experts\n0,decode,3,1,2\n","step,phase,layer,experts\n0,unknown,3,1\n","step,phase,layer,experts\n18446744073709551616,decode,3,1\n","step,phase,layer,experts\n1,decode,3,1\n0,decode,3,1\n",
        "[{\"step\":true,\"phase\":\"decode\",\"layer\":3,\"experts\":[1]}]","[{\"step\":0,\"phase\":\"decode\",\"layer\":3,\"experts\":[1.5]}]","[{\"step\":0,\"phase\":\"decode\",\"layer\":2,\"experts\":[1]}]",std::string(100,'[')+std::string(100,']'),"[]"};
    for(const auto& text:bad){file.text(text);require(load_route_trace(file.path,catalog,{},&out)!=Error::ok,"malformed route accepted");require(out.synthetic&&out.provenance==original&&out.events.size()==1&&out.events[0].experts.size()==3,"failed route parse mutated output");}
    file.text("{\"step\":0,\"phase\":\"decode\",\"layer\":3,\"experts\":[1]}\n{\"step\":1,\"phase\":\"decode\",\"layer\":45,\"experts\":[2]}\n");require(load_route_trace(file.path,catalog,{},&out)==Error::ok&&out.events.size()==2,"JSONL routes");
    TraceLimits small;small.max_routes_per_event=1;require(load_route_trace(file.path,catalog,small,&out)==Error::ok,"one route per JSONL event");small.max_events=1;require(load_route_trace(file.path,catalog,small,&out)==Error::too_large,"event cap");small={};small.max_line_bytes=8;require(load_route_trace(file.path,catalog,small,&out)==Error::too_large,"line cap");
}
void real_stride_zero_budget_no_io(){++cases;struct Never:Provider{size_t calls=0;ReadResult read(const RecordRef&,std::span<std::byte>,std::stop_token)override{++calls;return {Error::io,0};}} provider;Limits limits;limits.hot_bytes=0;limits.cold_bytes=0;Cache cache(limits,provider);RecordRef actual;actual.key={"b5688f25491202978c19c4d036eef579f61bbe07","GLM53-metadata-only","executed-rank0-layout","mxfp4_e8m0_g32",4,0,3,0};actual.length=3342336;Batch out;require(cache.acquire({&actual,1},&out)==Error::budget,"real stride zero-budget refusal");require(provider.calls==0&&cache.stats().bytes_read==0,"real stride attempted payload I/O");const auto use=cache.accounting().usage();require(use.hot_bytes==0&&use.cold_bytes==0&&use.live_records==0,"real stride allocated payload");}
}
int main(){try{exact_read_and_classes();immutable_metadata_refusals();corrupt_and_truncated_files();cancellation_and_output_sizes();cache_dedup_and_failed_batch();cache_real_io_and_duplicate_union();catalog_validation();route_parse_and_atomic_refusals();real_stride_zero_budget_no_io();std::cout<<"PASS: "<<cases<<" bounded provider/trace CPU cases, "<<checks<<" checks; synthetic records, no model/GPU\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL after "<<cases<<" cases: "<<e.what()<<'\n';return 1;}}

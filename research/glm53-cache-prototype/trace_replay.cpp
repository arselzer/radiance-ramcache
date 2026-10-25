// Bounded synthetic-file I/O replay. No checkpoint payload, GLM kernels or locality claim.
#include "file_provider.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <unistd.h>
using namespace glm53::cache;
namespace {
using Json=nlohmann::json;
void ensure(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
uint64_t integer(const std::string& text){uint64_t n=0;const auto r=std::from_chars(text.data(),text.data()+text.size(),n);ensure(!text.empty()&&r.ec==std::errc()&&r.ptr==text.data()+text.size(),"invalid numeric option");return n;}
std::byte expected_byte(int32_t layer,int32_t expert,size_t i){return std::byte((uint64_t(layer)*17+uint64_t(expert)*31+uint64_t(i)*13+(i>>5))&255u);}
struct Temporary {
    std::string path;int fd=-1;
    Temporary(){char name[]="/tmp/glm53-record-replay-XXXXXX";fd=::mkstemp(name);if(fd<0)throw std::runtime_error("mkstemp failed");path=name;}
    ~Temporary(){if(fd>=0)::close(fd);if(!path.empty())::unlink(path.c_str());}
    void write(std::span<const std::byte> bytes){size_t at=0;while(at<bytes.size()){const ssize_t n=::write(fd,bytes.data()+at,bytes.size()-at);if(n<0&&errno==EINTR)continue;ensure(n>0,"fixture write failed");at+=size_t(n);}}
};
RouteTrace demonstration(){
    RouteTrace trace;trace.synthetic=true;trace.provenance="synthetic_demo_v1: constructed locality/shift/scan, NOT measured GLM routes";
    for(uint64_t step=0;step<12;++step)for(int layer:{3,45}){
        RouteEvent e;e.step=step;e.phase=step>=10?"prefill":"decode";e.layer=layer;
        const int count=step>=10?32:8,base=step>=6&&step<10?8:0;
        for(int row=0;row<2;++row)for(int i=0;i<count;++i)e.experts.push_back(base+i);
        trace.events.push_back(std::move(e));
    }
    return trace;
}
}
int main(int argc,char** argv){
    try{
        std::string catalog_path,trace_path;uint64_t hot=0,cold=0;bool hot_set=false,cold_set=false,retain=false;
        for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--retain-previous"){retain=true;continue;}ensure(i+1<argc,"missing option value");const std::string value=argv[++i];
            if(arg=="--catalog")catalog_path=value;else if(arg=="--trace")trace_path=value;else if(arg=="--hot-bytes"){hot=integer(value);hot_set=true;}else if(arg=="--cold-bytes"){cold=integer(value);cold_set=true;}else throw std::runtime_error("unknown option: "+arg);
        }
        ensure(!catalog_path.empty()&&hot_set&&cold_set,"usage: trace_replay --catalog PINNED_RANK0.json --hot-bytes SYNTHETIC_BYTES --cold-bytes SYNTHETIC_BYTES [--trace CSV_OR_JSON] [--retain-previous]");
        ensure(hot<=8u*1024u*1024u&&cold<=8u*1024u*1024u&&hot+cold<=8u*1024u*1024u,"synthetic cache backing budget cap8MiB");
        Rank0Catalog catalog;std::string why;ensure(load_rank0_catalog(catalog_path,&catalog,&why)==Error::ok,"catalog: "+why);
        RouteTrace trace;if(trace_path.empty())trace=demonstration();else ensure(load_route_trace(trace_path,catalog,{},&trace,&why)==Error::ok,"trace: "+why);
        FileIdentity id;id.source_id=catalog.source_revision+":synthetic-bytes";id.model_id="synthetic-glm53-cost-metadata-v1";id.layout_id="opaque-synthetic-v1";
        for(const auto& layer:catalog.layers)id.class_lengths.emplace(layer.record_class+"@synthetic",layer.layer==45?256u:128u);
        std::map<std::pair<int32_t,int32_t>,RecordRef> records;
        for(const auto& e:trace.events)for(int32_t expert:e.experts){const auto* layer=catalog.find(e.layer);ensure(layer!=nullptr,"missing catalog layer");
            RecordRef ref;ref.key={id.source_id,id.model_id,id.layout_id,layer->record_class+"@synthetic",4,0,e.layer,expert};ref.length=e.layer==45?256:128;records.emplace(std::make_pair(e.layer,expert),std::move(ref));}
        ensure(records.size()<=id.max_records,"oversized synthetic record set");
        Temporary file;std::vector<RecordRef> manifest;uint64_t offset=0;
        for(auto& [key,ref]:records){(void)key;ensure(ref.length<=id.max_file_bytes-offset,"fixture file exceeds4MiB cap");ref.offset=offset;std::vector<std::byte> bytes(size_t(ref.length));for(size_t i=0;i<bytes.size();++i)bytes[i]=expected_byte(ref.key.layer,ref.key.expert,i);ref.checksum=fnv1a64(bytes);file.write(bytes);offset+=ref.length;manifest.push_back(ref);}
        std::unique_ptr<FileProvider> provider;ensure(FileProvider::open(file.path,id,manifest,&provider,&why)==Error::ok,"provider open: "+why);
        Limits limits;limits.hot_bytes=hot;limits.cold_bytes=cold;limits.max_record_bytes=256;limits.max_records=16384;limits.max_batch=4096;Cache cache(limits,*provider);
        Batch previous;Json events=Json::array();uint64_t raw_routes=0,unique_routes=0,hot_hit_bytes=0,cold_hit_bytes=0,miss_bytes=0,weighted_hot=0,weighted_cold=0,weighted_miss=0,weighted_io=0,weighted_attempted=0,max_union=0,served=0,refused=0;
        for(const auto& e:trace.events){
            const auto* layer=catalog.find(e.layer);std::vector<RecordRef> demand;std::set<int32_t> unique;
            for(int32_t expert:e.experts){demand.push_back(records.at({e.layer,expert}));unique.insert(expert);}
            const uint64_t actual_union=uint64_t(unique.size())*layer->record_bytes;max_union=std::max(max_union,actual_union);weighted_attempted+=actual_union;raw_routes+=e.experts.size();unique_routes+=unique.size();
            const auto before=cache.stats();Batch batch;const auto error=cache.acquire(demand,&batch);const auto after=cache.stats();
            previous.leases.clear(); // Retain the preceding event only through this attempt, even on refusal.
            weighted_io+=(after.provider_reads-before.provider_reads)*layer->record_bytes;
            const uint64_t small=layer->layer==45?256:128;
            Json row{{"step",e.step},{"phase",e.phase},{"layer",e.layer},{"raw_routes",e.experts.size()},{"unique_experts",unique.size()},{"status",to_string(error)},{"catalog_union_bytes_metadata_only",actual_union}};
            if(error==Error::ok){
                ensure(batch.leases.size()==unique.size(),"selected expert lost/dedup incorrect");std::set<int32_t> delivered;
                for(const auto& lease:batch.leases){ensure(lease.ref().key.layer==e.layer&&delivered.insert(lease.ref().key.expert).second,"wrong/duplicate lease key");const auto bytes=lease.bytes();ensure(bytes.size()==small,"wrong synthetic record length");for(size_t i=0;i<bytes.size();++i)ensure(bytes[i]==expected_byte(e.layer,lease.ref().key.expert,i),"independent synthetic byte mismatch");}
                ensure(delivered==unique,"missing selected expert");++served;
                const uint64_t h=after.hot_hits-before.hot_hits,c=after.cold_hits-before.cold_hits,m=after.misses-before.misses;
                hot_hit_bytes+=h*small;cold_hit_bytes+=c*small;miss_bytes+=m*small;weighted_hot+=h*layer->record_bytes;weighted_cold+=c*layer->record_bytes;weighted_miss+=m*layer->record_bytes;
                row["synthetic_hot_hit_bytes"]=h*small;row["synthetic_cold_hit_bytes"]=c*small;row["synthetic_miss_bytes"]=m*small;
                if(retain)previous=std::move(batch);
            }else{++refused;row["experts_executed"]=0;row["partial_expert_service"]=false;}
            events.push_back(std::move(row));
        }
        previous.leases.clear();const auto s=cache.stats();const auto io=provider->stats();
        Json report{{"schema","glm53-cache-file-replay-v1"},{"synthetic_record_payloads",true},{"trace_marked_synthetic",trace.synthetic},{"trace_provenance",trace.provenance},{"actual_GLM_locality_claim",false},{"actual_model_payload_read",false},{"retained_previous_batch",retain},{"catalog_source_revision",catalog.source_revision},{"catalog_scope",catalog.source},
            {"events",events},{"served_events",served},{"refused_events",refused},{"raw_routes",raw_routes},{"unique_event_experts",unique_routes},{"duplicate_routes",raw_routes-unique_routes},
            {"synthetic",{{"file_bytes",offset},{"hot_budget_bytes",hot},{"cold_budget_bytes",cold},{"peak_charged_hot_bytes",s.peak_hot_bytes},{"peak_charged_cold_bytes",s.peak_cold_bytes},{"hot_hit_bytes",hot_hit_bytes},{"cold_hit_bytes",cold_hit_bytes},{"miss_bytes",miss_bytes},{"provider_reads",s.provider_reads},{"cache_bytes_read",s.bytes_read},{"pread_calls",io.pread_calls},{"actual_file_bytes_read",io.bytes_read},{"failed_file_reads",io.failed},{"admission_bypasses",s.admission_bypasses},{"budget_refusals",s.budget_refusals}}},
            {"actual_catalog_costs_metadata_only",{{"scope","conditional full-record request arithmetic on synthetic cache decisions (including refused fills); hit/miss bytes cover successful events only; NOT an actual-stride cache run/capacity proof"},{"attempted_union_bytes",weighted_attempted},{"max_current_union_bytes",max_union},{"hot_hit_bytes",weighted_hot},{"cold_hit_bytes",weighted_cold},{"miss_bytes",weighted_miss},{"conditional_requested_read_bytes",weighted_io}}}};
        std::cout<<report.dump(2)<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"trace_replay: "<<e.what()<<'\n';return 1;}
}

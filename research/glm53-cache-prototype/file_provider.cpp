#include "file_provider.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <climits>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
namespace glm53::cache {
namespace {
Error fail(Error e,const char* text,std::string* why) noexcept {if(why)try{*why=text;}catch(...){}return e;}
bool key_less(const Key& a,const Key& b){return std::tie(a.source_id,a.model_id,a.layout_id,a.record_class,a.tp,a.rank,a.layer,a.expert)<std::tie(b.source_id,b.model_id,b.layout_id,b.record_class,b.tp,b.rank,b.layer,b.expert);}
bool stamp_equal(const struct stat& a,const struct stat& b){
    return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&
        a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&
        a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}
struct FD {int value=-1;~FD(){if(value>=0)::close(value);}};
Error text_file(const std::string& path,size_t cap,std::string* out){
    if(path.empty()||path.find('\0')!=std::string::npos||!cap)return Error::invalid;
    FD fd;fd.value=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);if(fd.value<0)return Error::io;
    struct stat before{};if(::fstat(fd.value,&before)||!S_ISREG(before.st_mode)||before.st_size<0)return Error::io;
    if(uint64_t(before.st_size)>cap)return Error::too_large;
    std::string result(size_t(before.st_size),'\0');size_t at=0;
    while(at<result.size()){
        const ssize_t n=::pread(fd.value,result.data()+at,result.size()-at,off_t(at));
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return Error::io;
        at+=size_t(n);
    }
    struct stat after{};if(::fstat(fd.value,&after)||!stamp_equal(before,after))return Error::identity_mismatch;
    *out=std::move(result);return Error::ok;
}
bool json_depth_ok(std::string_view text,unsigned maximum){
    unsigned depth=0;bool quoted=false,escaped=false;
    for(char ch:text){
        if(quoted){if(escaped)escaped=false;else if(ch=='\\')escaped=true;else if(ch=='"')quoted=false;continue;}
        if(ch=='"')quoted=true;
        else if(ch=='['||ch=='{'){if(++depth>maximum)return false;}
        else if(ch==']'||ch=='}'){if(!depth)return false;--depth;}
    }
    return depth==0&&!quoted;
}
int64_t json_integer(const nlohmann::json& j,int64_t minimum,int64_t maximum){
    if(!j.is_number_integer()||j.is_boolean())throw std::invalid_argument("expected exact integer");
    if(j.is_number_unsigned()){
        const uint64_t n=j.get<uint64_t>();if(n<uint64_t(minimum)||n>uint64_t(maximum))throw std::invalid_argument("integer overflow");return int64_t(n);
    }
    const int64_t n=j.get<int64_t>();if(n<minimum||n>maximum)throw std::invalid_argument("integer out of range");return n;
}
uint64_t csv_integer(std::string_view text,uint64_t max){
    uint64_t n=0;const auto result=std::from_chars(text.data(),text.data()+text.size(),n);
    if(text.empty()||result.ec!=std::errc()||result.ptr!=text.data()+text.size()||n>max)throw std::invalid_argument("invalid CSV integer");
    return n;
}
void event_validate(RouteEvent& event,const Rank0Catalog& catalog,size_t max_routes){
    if(event.phase!="decode"&&event.phase!="prefill")throw std::invalid_argument("unknown trace phase");
    const auto* layer=catalog.find(event.layer);if(!layer||event.experts.size()>max_routes)throw std::invalid_argument("unknown/oversized route union");
    for(int32_t expert:event.experts)if(expert<0||expert>=layer->experts)throw std::invalid_argument("expert ID outside catalog");
}
RouteEvent json_event(const nlohmann::json& j,const Rank0Catalog& catalog,const TraceLimits& limits){
    if(!j.is_object()||!j.contains("step")||!j.contains("phase")||!j.contains("layer")||!j.contains("experts"))throw std::invalid_argument("missing event fields");
    RouteEvent event;event.step=uint64_t(json_integer(j.at("step"),0,INT64_MAX));event.phase=j.at("phase").get<std::string>();event.layer=int32_t(json_integer(j.at("layer"),0,INT32_MAX));
    const auto& routes=j.at("experts");if(!routes.is_array())throw std::invalid_argument("experts must be array");
    auto add=[&](const nlohmann::json& value){if(event.experts.size()>=limits.max_routes_per_event)throw std::length_error("too many route IDs");event.experts.push_back(int32_t(json_integer(value,0,INT32_MAX)));};
    for(const auto& row:routes){if(row.is_array()){for(const auto& value:row)add(value);}else add(row);}
    event_validate(event,catalog,limits.max_routes_per_event);return event;
}
void append(RouteTrace& trace,RouteEvent event,const TraceLimits& limits){
    if(trace.events.size()>=limits.max_events)throw std::length_error("too many events");
    if(!trace.events.empty()&&event.step<trace.events.back().step)throw std::invalid_argument("decreasing trace step");
    trace.events.push_back(std::move(event));
}
}
struct FileProvider::Impl {
    int fd=-1;struct stat stamp{};FileIdentity identity;std::vector<RecordRef> manifest;
    std::atomic<uint64_t> attempts{0},completed{0},failed{0},pread_calls{0},bytes_read{0};
    ~Impl(){if(fd>=0)::close(fd);}
};
FileProvider::FileProvider(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
FileProvider::~FileProvider()=default;
Error FileProvider::open(const std::string& path,const FileIdentity& id,std::span<const RecordRef> manifest,std::unique_ptr<FileProvider>* out,std::string* why) noexcept {
    try{
        if(!out||path.empty()||path.find('\0')!=std::string::npos||id.schema!=file_schema||id.source_id.empty()||id.model_id.empty()||id.layout_id.empty()||!id.tp||id.tp>64||id.rank>=id.tp||!id.max_records||!id.max_record_bytes||!id.max_file_bytes||!id.max_identity_bytes||!id.max_total_identity_bytes||id.max_file_bytes>64u*1024u*1024u||id.max_record_bytes>16u*1024u*1024u||id.max_records>65536||id.max_identity_bytes>1024||id.max_total_identity_bytes>16u*1024u*1024u||id.class_lengths.empty()||id.class_lengths.size()>32)return fail(Error::invalid,"invalid immutable file identity/limits",why);
        if(manifest.size()>id.max_records)return fail(Error::metadata_limit,"manifest record cap",why);
        for(const auto* value:{&id.source_id,&id.model_id,&id.layout_id})if(value->size()>id.max_identity_bytes)return fail(Error::metadata_limit,"identity string cap",why);
        for(const auto& [name,length]:id.class_lengths)if(name.empty()||name.size()>id.max_identity_bytes||!length||length>id.max_record_bytes)return fail(Error::invalid,"invalid class geometry",why);
        auto impl=std::make_unique<Impl>();impl->identity=id;
        impl->fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);if(impl->fd<0)return fail(Error::io,"open failed",why);
        if(::fstat(impl->fd,&impl->stamp)||!S_ISREG(impl->stamp.st_mode)||impl->stamp.st_size<0)return fail(Error::io,"regular file required",why);
        if(uint64_t(impl->stamp.st_size)>id.max_file_bytes)return fail(Error::too_large,"file extent cap",why);
        size_t identities=0;std::vector<std::pair<uint64_t,uint64_t>> ranges;ranges.reserve(manifest.size());
        for(const auto& record:manifest){
            const auto& k=record.key;
            if(k.source_id!=id.source_id||k.model_id!=id.model_id||k.layout_id!=id.layout_id||k.tp!=id.tp||k.rank!=id.rank||k.layer<0||k.expert<0)return fail(Error::identity_mismatch,"record identity/rank mismatch",why);
            const auto found=id.class_lengths.find(k.record_class);
            if(found==id.class_lengths.end()||record.length!=found->second)return fail(Error::identity_mismatch,"record class/length mismatch",why);
            for(const auto* value:{&k.source_id,&k.model_id,&k.layout_id,&k.record_class}){
                if(value->size()>id.max_identity_bytes||value->size()>id.max_total_identity_bytes-identities)return fail(Error::metadata_limit,"manifest identity cap",why);
                identities+=value->size();
            }
            const uint64_t extent=uint64_t(impl->stamp.st_size);
            if(record.offset>extent||record.length>extent-record.offset||record.offset>uint64_t(std::numeric_limits<off_t>::max())||record.length>uint64_t(std::numeric_limits<off_t>::max())-record.offset)return fail(Error::invalid,"record offset/length outside immutable extent",why);
            ranges.emplace_back(record.offset,record.offset+record.length);
        }
        std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)if(ranges[i].first<ranges[i-1].second)return fail(Error::invalid,"overlapping record extents",why);
        impl->manifest.assign(manifest.begin(),manifest.end());std::sort(impl->manifest.begin(),impl->manifest.end(),[](const auto& a,const auto& b){return key_less(a.key,b.key);});
        for(size_t i=1;i<impl->manifest.size();++i)if(impl->manifest[i-1].key==impl->manifest[i].key)return fail(Error::identity_mismatch,"duplicate immutable record key",why);
        auto result=std::unique_ptr<FileProvider>(new FileProvider(std::move(impl)));*out=std::move(result);if(why)why->clear();return Error::ok;
    }catch(const std::bad_alloc&){return Error::no_memory;}catch(...){return Error::invalid;}
}
ReadResult FileProvider::read(const RecordRef& ref,std::span<std::byte> dst,std::stop_token stop){
    auto& p=*impl_;p.attempts.fetch_add(1,std::memory_order_relaxed);size_t at=0;
    auto error=[&](Error e){p.failed.fetch_add(1,std::memory_order_relaxed);return ReadResult{e,at};};
    if(stop.stop_requested())return error(Error::cancelled);
    const auto found=std::lower_bound(p.manifest.begin(),p.manifest.end(),ref.key,[](const RecordRef& a,const Key& b){return key_less(a.key,b);});
    if(found==p.manifest.end()||!(*found==ref))return error(Error::identity_mismatch);
    if(dst.size()!=ref.length||(!dst.data()&&dst.size())||dst.size()>UINTPTR_MAX-reinterpret_cast<uintptr_t>(dst.data()))return error(Error::invalid);
    struct stat before{};if(::fstat(p.fd,&before)||!stamp_equal(p.stamp,before))return error(Error::identity_mismatch);
    while(at<dst.size()){
        if(stop.stop_requested())return error(Error::cancelled);
        p.pread_calls.fetch_add(1,std::memory_order_relaxed);
        const ssize_t n=::pread(p.fd,dst.data()+at,std::min(dst.size()-at,size_t(SSIZE_MAX)),off_t(ref.offset+at));
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return error(Error::io);
        at+=size_t(n);p.bytes_read.fetch_add(uint64_t(n),std::memory_order_relaxed);
    }
    struct stat after{};if(::fstat(p.fd,&after)||!stamp_equal(p.stamp,after))return error(Error::identity_mismatch);
    if(stop.stop_requested())return error(Error::cancelled);
    if(fnv1a64(dst)!=ref.checksum)return error(Error::checksum);
    p.completed.fetch_add(1,std::memory_order_relaxed);return {Error::ok,at};
}
FileStats FileProvider::stats() const noexcept {const auto& p=*impl_;return {p.attempts.load(),p.completed.load(),p.failed.load(),p.pread_calls.load(),p.bytes_read.load()};}
const CatalogLayer* Rank0Catalog::find(int32_t layer)const noexcept{for(const auto& value:layers)if(value.layer==layer)return &value;return nullptr;}
Error load_rank0_catalog(const std::string& path,Rank0Catalog* out,std::string* why)noexcept{
    try{
        if(!out)return Error::invalid;
        std::string raw;const auto read=text_file(path,1024u*1024u,&raw);if(read!=Error::ok)return fail(read,"catalog read refused",why);
        if(!json_depth_ok(raw,8))return Error::invalid;
        const auto j=nlohmann::json::parse(raw);if(j.at("schema")!="expert-cache-records-v1"||json_integer(j.at("tp_rank"),0,63)!=0||j.at("source_revision")!="b5688f25491202978c19c4d036eef579f61bbe07")return fail(Error::identity_mismatch,"wrong pinned rank0 catalog",why);
        Rank0Catalog result;result.source_revision=j.at("source_revision").get<std::string>();result.source=j.at("source").get<std::string>();if(result.source.size()>4096||!j.at("layers").is_array()||j.at("layers").size()!=43)return Error::invalid;
        for(const auto& row:j.at("layers")){
            CatalogLayer value;value.layer=int32_t(json_integer(row.at("layer"),3,45));value.experts=int32_t(json_integer(row.at("experts"),288,288));value.record_bytes=uint64_t(json_integer(row.at("record_bytes"),1,1LL<<40));value.record_class=row.at("class").get<std::string>();
            if(result.find(value.layer)||value.record_class!=(value.layer==45?"fp8_e4m3_f32_b128":"mxfp4_e8m0_g32")||value.record_bytes!=(value.layer==45?6299648u:3342336u))return fail(Error::identity_mismatch,"catalog class/stride mismatch",why);
            result.layers.push_back(std::move(value));
        }
        std::sort(result.layers.begin(),result.layers.end(),[](const auto& a,const auto& b){return a.layer<b.layer;});*out=std::move(result);if(why)why->clear();return Error::ok;
    }catch(const std::bad_alloc&){return Error::no_memory;}catch(...){return fail(Error::invalid,"invalid catalog JSON",why);}
}
Error load_route_trace(const std::string& path,const Rank0Catalog& catalog,const TraceLimits& limits,RouteTrace* out,std::string* why)noexcept{
    try{
        if(!out||!limits.max_events||!limits.max_routes_per_event||!limits.max_line_bytes||limits.max_events>10000||limits.max_routes_per_event>4096||limits.max_bytes>1024u*1024u||limits.max_line_bytes>65536||limits.max_line_bytes>limits.max_bytes)return Error::invalid;
        std::string raw;const auto read=text_file(path,limits.max_bytes,&raw);if(read!=Error::ok)return fail(read,"trace read refused",why);
        if(raw.find('\0')!=std::string::npos)return Error::invalid;
        for(size_t at=0;at<raw.size();){const auto end=raw.find('\n',at);const size_t length=(end==std::string::npos?raw.size():end)-at;
            if(length>limits.max_line_bytes)return Error::too_large;
            if(end==std::string::npos)break;
            at=end+1;}

        RouteTrace trace;trace.provenance="external caller-supplied trace; locality/provenance unverified";
        const auto first=raw.find_first_not_of(" \t\r\n");if(first==std::string::npos)return Error::invalid;
        if(raw[first]=='['||raw[first]=='{'){
            if(!json_depth_ok(raw,16))return Error::invalid;
            const auto root=nlohmann::json::parse(raw,nullptr,false);
            if(!root.is_discarded()){
                const nlohmann::json* events=&root;
                if(root.is_object()&&root.contains("events")){
                    if(root.at("schema")!="glm53-route-unions-v1"||!root.at("synthetic").is_boolean())return Error::invalid;
                    trace.synthetic=root.at("synthetic").get<bool>();trace.provenance=trace.synthetic?"explicit synthetic input trace":"external explicitly non-synthetic trace; provenance unverified";events=&root.at("events");
                }
                if(events->is_array()){for(const auto& event:*events)append(trace,json_event(event,catalog,limits),limits);}
                else append(trace,json_event(*events,catalog,limits),limits);
            }else{
                size_t at=0;while(at<raw.size()){const size_t end=raw.find('\n',at),length=(end==std::string::npos?raw.size():end)-at;if(length>limits.max_line_bytes)return Error::too_large;
                    const auto line=std::string_view(raw).substr(at,length);if(line.find_first_not_of(" \t\r")!=std::string_view::npos)append(trace,json_event(nlohmann::json::parse(line),catalog,limits),limits);if(end==std::string::npos)break;at=end+1;}
            }
        }else{
            size_t at=0;bool header=false;
            while(at<raw.size()){
                const size_t end=raw.find('\n',at),length=(end==std::string::npos?raw.size():end)-at;if(length>limits.max_line_bytes)return Error::too_large;
                auto line=std::string_view(raw).substr(at,length);if(!line.empty()&&line.back()=='\r')line.remove_suffix(1);
                if(!header){if(line!="step,phase,layer,experts")return Error::invalid;header=true;}
                else if(!line.empty()){
                    if(line.find('"')!=std::string_view::npos)return Error::invalid;
                    std::array<std::string_view,4> parts{};size_t from=0;
                    for(size_t i=0;i<3;++i){const auto comma=line.find(',',from);if(comma==std::string_view::npos)return Error::invalid;parts[i]=line.substr(from,comma-from);from=comma+1;}parts[3]=line.substr(from);if(parts[3].find(',')!=std::string_view::npos)return Error::invalid;
                    RouteEvent event;event.step=csv_integer(parts[0],INT64_MAX);event.phase=std::string(parts[1]);event.layer=int32_t(csv_integer(parts[2],INT32_MAX));
                    if(!parts[3].empty()){size_t pos=0;while(true){const size_t semi=parts[3].find(';',pos);const auto value=parts[3].substr(pos,semi==std::string_view::npos?semi:semi-pos);if(event.experts.size()>=limits.max_routes_per_event)return Error::too_large;event.experts.push_back(int32_t(csv_integer(value,INT32_MAX)));if(semi==std::string_view::npos)break;pos=semi+1;}}
                    event_validate(event,catalog,limits.max_routes_per_event);append(trace,std::move(event),limits);
                }
                if(end==std::string::npos)break;
                at=end+1;
            }
        }
        if(trace.events.empty())return Error::invalid;
        *out=std::move(trace);if(why)why->clear();return Error::ok;
    }catch(const std::bad_alloc&){return Error::no_memory;}catch(const std::length_error&){return fail(Error::too_large,"trace cap exceeded",why);}catch(...){return fail(Error::invalid,"invalid trace field/JSON",why);}
}
} // namespace glm53::cache

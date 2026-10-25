#include "main_records.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace rad::glm_main_records {
using Json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
constexpr uint64_t seed=14695981039346656037ull,prime=1099511628211ull,page=4096;
constexpr const char* schema="radiance-native-record-store-v2";
constexpr const char* origin="complete_selected_glm53_main_checkpoint_tensors_v1";
struct Failure {Status status;const char* why;};
void need(bool b,Status s,const char* why){if(!b)throw Failure{s,why};}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,Status::invalid,"byte addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,Status::invalid,"byte multiplication overflow");return a*b;}
uint64_t hash(uint64_t h,const std::byte* p,size_t n){for(size_t i=0;i<n;++i){h^=std::to_integer<uint8_t>(p[i]);h*=prime;}return h;}
uint64_t hash_text(const std::string& s){return hash(seed,reinterpret_cast<const std::byte*>(s.data()),s.size());}
void explain(std::string* out,const char* why)noexcept{if(out)try{*out=why;}catch(...){}}
template<class Fn>Status operation(std::string* why,Fn&& f)noexcept{try{f();return Status::ok;}catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){explain(why,"allocation refused; caller preserved");return Status::no_memory;}catch(...){explain(why,"record operation exception; caller preserved");return Status::invalid;}}
void text(const std::string& s,size_t cap){need(!s.empty()&&s.size()<=cap&&s.find('\0')==std::string::npos,Status::invalid,"empty/oversized/NUL text");}
bool leaf(const std::string& s){return !s.empty()&&s.size()<=160&&s!="."&&s!=".."&&s.find('/')==std::string::npos&&s.find('\0')==std::string::npos;}
void limits(const Limits& l){need(l.max_records==1&&l.io_chunk_bytes>=4096&&l.io_chunk_bytes<=65536&&l.max_input_fds>=1&&l.max_input_fds<=2&&
    l.max_index_bytes&&l.max_index_bytes<=(16u<<20)&&l.max_config_bytes&&l.max_config_bytes<=(2u<<20)&&l.max_header_bytes&&l.max_header_bytes<=(4u<<20)&&l.max_manifest_bytes>=65536&&l.max_manifest_bytes<=(1u<<20)&&
    l.max_source_file_bytes&&l.max_source_file_bytes<=(64ull<<30)&&l.max_selected_tensor_bytes&&l.max_selected_tensor_bytes<=(16u<<20)&&
    l.max_record_bytes&&l.max_record_bytes<=(4u<<20)&&l.max_read_bytes&&l.max_read_bytes<=(4u<<20)&&
    l.max_metadata_bytes>=(2u<<20)&&l.max_metadata_bytes<=(256u<<20),Status::budget,"fixed actual-one-record limits");}
struct Fd {int n=-1;Fd()=default;explicit Fd(int f):n(f){}~Fd(){if(n>=0)::close(n);}Fd(Fd&& a)noexcept:n(a.n){a.n=-1;}Fd& operator=(Fd&& a)noexcept{if(n>=0)::close(n);n=a.n;a.n=-1;return *this;}Fd(const Fd&)=delete;};
FileStamp stamp(int fd,bool directory=false){struct stat s{};need(!::fstat(fd,&s)&&s.st_size>=0&&(directory?S_ISDIR(s.st_mode):S_ISREG(s.st_mode)),Status::io,"file kind/stat");
    return {uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),s.st_mtim.tv_sec,s.st_mtim.tv_nsec,s.st_ctim.tv_sec,s.st_ctim.tv_nsec};}
Json stamp_json(const FileStamp& s){return {{"device",s.device},{"inode",s.inode},{"bytes",s.bytes},{"mtime",{s.mtime_sec,s.mtime_nsec}},{"ctime",{s.ctime_sec,s.ctime_nsec}}};}
void read_exact(int fd,void* p,uint64_t at,size_t n){need(at<=uint64_t(INT64_MAX)&&n<=uint64_t(INT64_MAX)-at,Status::invalid,"read extent overflow");size_t done=0;while(done<n){const auto k=::pread(fd,static_cast<std::byte*>(p)+done,n-done,off_t(at+done));if(k<0&&errno==EINTR)continue;need(k>0,Status::io,"short/failed pread");done+=size_t(k);}}
void write_exact(int fd,const void* p,uint64_t at,size_t n){need(at<=uint64_t(INT64_MAX)&&n<=uint64_t(INT64_MAX)-at,Status::invalid,"write extent overflow");size_t done=0;while(done<n){const auto k=::pwrite(fd,static_cast<const std::byte*>(p)+done,n-done,off_t(at+done));if(k<0&&errno==EINTR)continue;need(k>0,Status::io,"short/failed pwrite");done+=size_t(k);}}
uint64_t stream_hash(int fd,uint64_t at,uint64_t bytes,std::vector<std::byte>& buffer,bool zero=false){uint64_t result=seed;for(uint64_t done=0;done<bytes;){const size_t n=size_t(std::min<uint64_t>(bytes-done,buffer.size()));read_exact(fd,buffer.data(),add(at,done),n);
    if(zero)need(std::all_of(buffer.begin(),buffer.begin()+n,[](auto b){return b==std::byte{0};}),Status::checksum,"nonzero canonical record padding");result=hash(result,buffer.data(),n);done+=n;}return result;}
uint64_t number(const Json& j){need(j.is_number_integer()&&!j.is_boolean(),Status::invalid,"unsigned extent required");if(j.is_number_unsigned())return j.get<uint64_t>();const auto n=j.get<int64_t>();need(n>=0,Status::invalid,"negative extent");return uint64_t(n);}
Json parse(const std::string& s,size_t max_nodes){size_t nodes=0;std::vector<std::set<std::string>> keys;
    return Json::parse(s,[&](int depth,Json::parse_event_t event,Json& v){need(depth<=16&&++nodes<=max_nodes,Status::budget,"JSON depth/node cap");
        if(v.is_string())need(v.get_ref<const std::string&>().size()<=4096,Status::budget,"JSON string cap");
        if(event==Json::parse_event_t::object_start)keys.emplace_back();
        if(event==Json::parse_event_t::key)need(!keys.empty()&&keys.back().insert(v.get<std::string>()).second,Status::identity,"duplicate JSON key");
        if(event==Json::parse_event_t::object_end){need(!keys.empty(),Status::invalid,"JSON object nesting");keys.pop_back();}return true;});}
struct FileProof {std::string name,path;FileStamp stamp;uint64_t header_checksum=0,header_bytes=0;};
struct Piece {std::string tensor;uint32_t file=0,projection=0,plane=0;uint64_t offset=0,bytes=0,rows=0,row_bytes=0,checksum=0;};
struct Member {std::vector<Piece> pieces;std::array<glm_main_metadata::PhysicalSlice,2> cuts{};uint32_t count=0;};
// Bounded two-file broker reused from the composite exporter: close the LRU
// BEFORE reopening and prove every observed file against one captured stamp.
struct Broker {
    struct Open {uint32_t file;Fd fd;uint64_t tick;};
    const std::vector<FileProof>& files;size_t cap;uint64_t clock=0;size_t peak=0;std::vector<Open> open;
    Broker(const std::vector<FileProof>& f,size_t n):files(f),cap(n){open.reserve(n);}
    int get(uint32_t id){need(id<files.size(),Status::identity,"source file ID outside recipe");for(auto& f:open)if(f.file==id){f.tick=++clock;need(stamp(f.fd.n)==files[id].stamp,Status::identity,"source changed while open");return f.fd.n;}
        if(open.size()==cap){const auto i=std::min_element(open.begin(),open.end(),[](const auto& a,const auto& b){return a.tick<b.tick;});open.erase(i);}
        Fd fd(::open(files[id].path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,Status::io,"open source payload file");need(stamp(fd.n)==files[id].stamp,Status::identity,"source replaced before reopen");
        open.push_back({id,std::move(fd),++clock});peak=std::max(peak,open.size());return open.back().fd.n;}
    void audit(){for(size_t n=0;n<files.size();++n)need(stamp(get(uint32_t(n)))==files[n].stamp,Status::identity,"source changed before publication");}
};
struct Temp {
    std::string path;Fd directory;bool published=false;
    explicit Temp(const fs::path& dest){const auto pattern=(dest.parent_path()/(dest.filename().string()+".staging.XXXXXX")).string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);char* created=::mkdtemp(name.data());need(created,Status::io,"private output directory");
        try{path=created;directory=Fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(directory.n>=0,Status::io,"private directory descriptor");}catch(...){::rmdir(created);throw;}}
    ~Temp(){if(published||path.empty())return;if(directory.n>=0){::unlinkat(directory.n,"segment-00000000.bin",0);::unlinkat(directory.n,"manifest.json",0);}::rmdir(path.c_str());}
};
void padding(int fd,const RecordInfo& info,std::vector<std::byte>& buffer){uint64_t previous=0;for(size_t n=0;n<4;++n){const auto at=info.member_offsets[n],bytes=info.member_bytes[n];need(at>=previous&&at<=info.length&&bytes<=info.length-at,Status::identity,"record member overlap/range");
    if(at>previous)(void)stream_hash(fd,info.offset+previous,at-previous,buffer,true);previous=at+bytes;}
    if(previous<info.length)(void)stream_hash(fd,info.offset+previous,info.length-previous,buffer,true);}
std::string raw_name(int layer,int expert,uint32_t projection,uint32_t plane){char name[160];const char* p=projection==0?"gate":projection==1?"up":"down";
    const int n=std::snprintf(name,sizeof name,"model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight%s",layer,expert,p,plane?"_scale":"");need(n>0&&size_t(n)<sizeof name,Status::invalid,"tensor name bound");return name;}
} // namespace
struct Capture {
    Receipt receipt;RecordInfo record;glm_main_metadata::RecordLayout layout;
    std::vector<FileProof> files;std::array<Member,4> members;
    std::string root,index_path,config_path;FileStamp root_stamp,index_stamp,config_stamp;uint64_t index_checksum=0,config_checksum=0;
    Json source_catalog,spec; // owned immutable typed capture, no borrowed pointers
};
namespace {
std::shared_ptr<Capture> capture(rad::Checkpoint& ck,const rad::CheckpointSources& sources,const rad::Program& p,const rad::Plan& plan,
                                 std::span<const int32_t> selected,const Limits& l,std::vector<std::byte>& window){
    need(selected.size()==1,Status::budget,"first actual-payload profile requires exactly one rank-local complete expert");
    glm_main_metadata::Quote geometry;std::string reason;glm_main_metadata::Limits geometry_limits;
    geometry_limits.max_metadata_bytes=std::min<uint64_t>(64u<<20,l.max_metadata_bytes-l.io_chunk_bytes-65536);
    const int quoted=glm_main_metadata::quote(ck,sources,p,plan,{},geometry_limits,&geometry,&reason);
    need(quoted==RAD_OK,quoted==RAD_E_NOMEM?Status::no_memory:quoted==RAD_E_SCRATCH?Status::budget:Status::unsupported,"fresh actual main geometry/source/layout quote refused");
    const auto found=std::find_if(geometry.records.begin(),geometry.records.end(),[&](const auto& row){return row.unit==selected[0];});
    need(found!=geometry.records.end()&&found->stride>0&&uint64_t(found->stride)<=l.max_record_bytes,Status::budget,"selected actual unit/record byte cap");
    auto result=std::make_shared<Capture>();result->layout=*found;result->receipt.metadata_reserved=add(geometry.construction_requested,l.io_chunk_bytes+65536);
    geometry=glm_main_metadata::Quote{}; // release full-domain quote before source JSON/payload staging
    text(ck.path(),4096);need(fs::is_directory(ck.path())&&!fs::is_symlink(ck.path()),Status::unsupported,"actual directory checkpoint required");result->root=fs::canonical(ck.path()).string();
    Fd root(::open(result->root.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(root.n>=0,Status::io,"actual source root");result->root_stamp=stamp(root.n,true);
    result->index_path=result->root+"/model.safetensors.index.json";Fd index(::open(result->index_path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(index.n>=0,Status::io,"actual checkpoint index");result->index_stamp=stamp(index.n);
    need(result->index_stamp.bytes&&result->index_stamp.bytes<=l.max_index_bytes,Status::budget,"bounded actual index file");
    uint64_t parser_reserved=add((2u<<20)+l.io_chunk_bytes+65536,mul(result->index_stamp.bytes,8));
    need(parser_reserved<=l.max_metadata_bytes&&result->receipt.metadata_reserved<=l.max_metadata_bytes,Status::budget,"metadata precharge before index parse");
    std::string index_text(size_t(result->index_stamp.bytes),'\0');read_exact(index.n,index_text.data(),0,index_text.size());need(stamp(index.n)==result->index_stamp,Status::identity,"index changed while reading");result->index_checksum=hash_text(index_text);
    const auto index_json=parse(index_text,1000000);need(index_json.is_object()&&index_json.contains("weight_map")&&index_json.at("weight_map").is_object(),Status::identity,"actual index weight_map");
    result->config_path=result->root+"/config.json";
    {Fd config(::open(result->config_path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(config.n>=0,Status::io,"actual source config");result->config_stamp=stamp(config.n);
        need(result->config_stamp.bytes&&result->config_stamp.bytes<=l.max_config_bytes,Status::budget,"bounded source config");
        parser_reserved=add(parser_reserved,mul(result->config_stamp.bytes,2));need(parser_reserved<=l.max_metadata_bytes,Status::budget,"source config metadata reservation");
        std::string config_text(size_t(result->config_stamp.bytes),'\0');read_exact(config.n,config_text.data(),0,config_text.size());need(stamp(config.n)==result->config_stamp,Status::identity,"config changed while reading");result->config_checksum=hash_text(config_text);}
    std::map<std::string,Json> headers;uint64_t complete_bytes=0;
    Json logical=Json::array();
    for(size_t m=0;m<4;++m){const auto& member=result->layout.members[m];auto& destination=result->members[m];destination.count=member.slices;destination.cuts=member.physical;
        uint64_t logical_hash=seed;Json physical=Json::array();
        for(size_t part=0;part<member.slices;++part){const auto& cut=member.physical[part];const auto name=raw_name(result->layout.layer,result->layout.expert,uint32_t(cut.projection),cut.plane);
            const auto entry=index_json.at("weight_map").find(name);need(entry!=index_json.at("weight_map").end()&&entry->is_string(),Status::identity,"actual tensor absent from checkpoint index");const auto file_name=entry->get<std::string>();need(leaf(file_name),Status::invalid,"unsafe indexed source file leaf");
            auto file=std::find_if(result->files.begin(),result->files.end(),[&](const auto& f){return f.name==file_name;});
            if(file==result->files.end()){
                FileProof f;f.name=file_name;f.path=result->root+"/"+file_name;Fd fd(::open(f.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,Status::io,"actual indexed source file");f.stamp=stamp(fd.n);
                need(f.stamp.bytes>=8&&f.stamp.bytes<=l.max_source_file_bytes,Status::budget,"whole physical file extent bound");uint64_t bytes=0;read_exact(fd.n,&bytes,0,8);
                need(bytes&&bytes<=l.max_header_bytes&&bytes<=f.stamp.bytes-8,Status::budget,"actual safetensors header extent");
                parser_reserved=add(parser_reserved,mul(bytes,8));need(parser_reserved<=l.max_metadata_bytes,Status::budget,"metadata precharge before selected header parse");
                std::string header(size_t(bytes),'\0');read_exact(fd.n,header.data(),8,header.size());need(stamp(fd.n)==f.stamp,Status::identity,"source changed during header read");f.header_bytes=bytes;f.header_checksum=hash_text(header);
                auto parsed=parse(header,300000);need(parsed.is_object(),Status::identity,"actual safetensors header object");headers.emplace(file_name,std::move(parsed));result->files.push_back(std::move(f));file=result->files.end()-1;
            }
            const auto file_id=uint32_t(file-result->files.begin());const auto& h=headers.at(file_name).at(name);
            need(h.is_object()&&h.at("dtype")=="U8"&&h.at("shape")==Json(cut.raw_shape)&&h.at("data_offsets").is_array()&&h.at("data_offsets").size()==2,Status::identity,"actual raw tensor header shape/type/extent");
            const auto begin=number(h.at("data_offsets")[0]),end=number(h.at("data_offsets")[1]);
            need(end>=begin&&end-begin==uint64_t(cut.raw_bytes)&&begin<=file->stamp.bytes-8-file->header_bytes&&uint64_t(cut.raw_bytes)<=file->stamp.bytes-8-file->header_bytes-begin,Status::identity,"actual tensor range outside physical file");
            const auto* mapped=ck.find(name);need(mapped&&mapped->data&&mapped->name==name&&mapped->rad_dtype==RAD_U8&&mapped->shape==std::vector<int64_t>({cut.raw_shape[0],cut.raw_shape[1]})&&mapped->bytes==cut.raw_bytes,Status::identity,"actual same opened raw mapping");
            Piece piece;piece.tensor=name;piece.file=file_id;piece.projection=uint32_t(cut.projection);piece.plane=cut.plane;piece.offset=add(8+file->header_bytes,begin);piece.bytes=cut.raw_bytes;piece.rows=cut.raw_shape[0];piece.row_bytes=cut.row_bytes;
            complete_bytes=add(complete_bytes,piece.bytes);need(complete_bytes<=l.max_selected_tensor_bytes,Status::budget,"complete selected tensor read bound");
            Fd fd(::open(file->path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0&&stamp(fd.n)==file->stamp,Status::identity,"same physical source before complete range comparison");
            uint64_t physical_hash=seed;
            for(uint64_t done=0;done<piece.bytes;){const auto n=size_t(std::min<uint64_t>(piece.bytes-done,window.size()));read_exact(fd.n,window.data(),piece.offset+done,n);
                need(!std::memcmp(window.data(),static_cast<const std::byte*>(mapped->data)+done,n),Status::identity,"mixed mapped checkpoint and physical source payload");
                physical_hash=hash(physical_hash,window.data(),n);logical_hash=hash(logical_hash,window.data(),n);done+=n;}
            need(stamp(fd.n)==file->stamp,Status::identity,"source changed during complete selected tensor comparison");piece.checksum=physical_hash;destination.pieces.push_back(piece);
            physical.push_back({{"tensor",piece.tensor},{"file",file_name},{"projection",piece.projection},{"plane",piece.plane},{"offset",piece.offset},{"bytes",piece.bytes},{"rows",piece.rows},{"row_bytes",piece.row_bytes},{"checksum",piece.checksum}});
        }
        logical.push_back({{"source",p.weights[member.weight].source},{"plane",member.plane},{"logical_plane_checksum",logical_hash},{"physical",physical}});
    }
    need(complete_bytes==13369344&&result->files.size()<=6,Status::identity,"complete actual six-tensor single-expert source closure");
    Broker audit(result->files,l.max_input_fds);audit.audit();need(stamp(index.n)==result->index_stamp&&stamp(root.n,true)==result->root_stamp,Status::identity,"source root/index changed during capture");
    Json files=Json::array();for(const auto& f:result->files)files.push_back({{"name",f.name},{"stamp",stamp_json(f.stamp)},{"header_bytes",f.header_bytes},{"header_checksum",f.header_checksum}});
    result->source_catalog={{"profile",origin},{"generation_scope","selected_raw_bundle_common_across_rank_cuts"},{"checkpoint_root",result->root},{"index_stamp",stamp_json(result->index_stamp)},{"index_checksum",result->index_checksum},
        {"config_stamp",stamp_json(result->config_stamp)},{"config_checksum",result->config_checksum},{"files",files},{"logical_planes",logical},
        {"intended_upstream_pin","amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07"},{"upstream_full_file_SHA_verified_by_this_code",false},{"whole_checkpoint_payload_verified",false}};
    const uint64_t generation=hash_text(result->source_catalog.dump());need(generation,Status::identity,"zero actual source generation checksum");
    result->receipt.source_generation=generation;result->receipt.source_config_bytes=result->config_stamp.bytes;result->receipt.source_index_bytes=result->index_stamp.bytes;result->receipt.complete_tensor_read_bytes=complete_bytes;result->receipt.physical_tensors=6;
    result->receipt.metadata_reserved=std::max(result->receipt.metadata_reserved,parser_reserved);need(result->receipt.metadata_reserved<=l.max_metadata_bytes,Status::budget,"final source/spec metadata reservation");
    const auto& row=result->layout;auto& record=result->record;record.key={row.world,row.rank,row.layer,row.expert};record.length=row.stride;
    Json members=Json::array();for(size_t m=0;m<4;++m){const auto& w=p.weights[row.members[m].weight];record.member_offsets[m]=row.offsets[m];record.member_bytes[m]=row.bytes[m];Json rects=Json::array();
        for(size_t n=0;n<row.members[m].slices;++n){const auto& cut=row.members[m].physical[n];rects.push_back({{"part",uint32_t(cut.projection)},{"row0",cut.row0},{"rows",cut.rows},{"byte0",cut.byte0},{"bytes",cut.bytes_per_row},{"row_bytes",cut.row_bytes}});}
        members.push_back({{"weight",w.name},{"source",w.source},{"projection",m<2?"gate_up":"down"},{"role",m%2?"scale":"codes"},{"dtype",row.dtype[m]},{"shape",row.shape[m]},
            {"stored_rank",2},{"stored_shape",row.shape[m]},{"stored_dtype",row.dtype[m]},{"alignment",w.align},{"layout_tag",w.layout_tag},{"identity",w.identity},{"offset",row.offsets[m]},{"bytes",row.bytes[m]},
            {"shard",w.decl.shard},{"span",{w.shard_lo,w.shard_hi}},{"row_parts",m<2?Json::array({512,512}):Json::array()},{"rectangles",rects}});
    }
    result->spec={{"world",row.world},{"rank",row.rank},{"layer",row.layer},{"expert",row.expert},{"class","mxfp4_e8m0_g32"},{"stride",row.stride},{"source_checksum",generation},{"members",members}};
    result->receipt.peak_io_buffer=window.size();result->receipt.records=1;return result;
}
} // namespace
const char* name(Status s)noexcept{switch(s){case Status::ok:return "ok";case Status::invalid:return "invalid";case Status::budget:return "budget";case Status::identity:return "identity";case Status::io:return "io";case Status::checksum:return "checksum";case Status::exists:return "exists";case Status::unsupported:return "unsupported";case Status::no_memory:return "no_memory";}return "unknown";}
uint64_t checksum(std::span<const std::byte> bytes)noexcept{return hash(seed,bytes.data(),bytes.size());}
Receipt PublishedRecords::receipt()const noexcept{return capture_?capture_->receipt:Receipt{};}
Status pack_main_records(rad::Checkpoint& ck,const rad::CheckpointSources& sources,const rad::Program& p,const rad::Plan& plan,std::span<const int32_t> selected,
                         const std::string& destination,const Limits& l,PublishedRecords* out,std::string* why)noexcept{return operation(why,[&]{
    need(out,Status::invalid,"null publication output");limits(l);text(destination,4096);const fs::path dest(destination);need(dest.is_absolute()&&leaf(dest.filename().string()),Status::invalid,"new absolute destination required");
    struct stat exists{};need(::lstat(destination.c_str(),&exists)!=0&&errno==ENOENT,Status::exists,"destination exists or cannot be checked");
    std::vector<std::byte> buffer(l.io_chunk_bytes);auto c=capture(ck,sources,p,plan,selected,l,buffer);Broker inputs(c->files,l.max_input_fds);
    Temp temporary(dest);Fd segment(::openat(temporary.directory.n,"segment-00000000.bin",O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));need(segment.n>=0,Status::io,"exclusive native segment");
    std::fill(buffer.begin(),buffer.end(),std::byte{0});for(uint64_t done=0;done<c->record.length;){const auto n=size_t(std::min<uint64_t>(c->record.length-done,buffer.size()));write_exact(segment.n,buffer.data(),done,n);done+=n;}
    for(size_t m=0;m<4;++m){const auto& member=c->members[m];uint64_t output=c->record.member_offsets[m];
        for(size_t part=0;part<member.count;++part){const auto& piece=member.pieces[part];const auto& cut=member.cuts[part];const int fd=inputs.get(piece.file);
            for(int64_t row=0;row<cut.rows;++row){const uint64_t at=add(piece.offset,add(mul(uint64_t(cut.row0+row),uint64_t(cut.row_bytes)),uint64_t(cut.byte0)));
                for(uint64_t done=0;done<uint64_t(cut.bytes_per_row);){const auto n=size_t(std::min<uint64_t>(uint64_t(cut.bytes_per_row)-done,buffer.size()));read_exact(fd,buffer.data(),at+done,n);write_exact(segment.n,buffer.data(),output,n);output+=n;done+=n;}}
        }
        need(output==c->record.member_offsets[m]+c->record.member_bytes[m],Status::identity,"native member copy coverage");c->receipt.selected_copy_bytes=add(c->receipt.selected_copy_bytes,c->record.member_bytes[m]);
    }
    need(stamp(segment.n).bytes==c->record.length,Status::identity,"actual output segment extent");c->record.checksum=stream_hash(segment.n,0,c->record.length,buffer);padding(segment.n,c->record,buffer);inputs.audit();
    Fd source_index(::open(c->index_path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));Fd source_root(::open(c->root.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));
    need(source_index.n>=0&&source_root.n>=0&&stamp(source_index.n)==c->index_stamp&&stamp(source_root.n,true)==c->root_stamp,Status::identity,"source root/index changed before publication");
    need(!::fsync(segment.n),Status::io,"native segment fsync");c->receipt.total_bytes=c->record.length;c->receipt.peak_input_fds=std::max(inputs.peak,size_t(1));
    Json manifest{{"schema",schema},{"origin",origin},{"checkpoint_headers_verified","selected_tensor_contracts"},{"checkpoint_payload_verified","selected_complete_tensors"},
        {"layout_provenance","actual stock Builder/Planner/DEVICE canonical identity geometry"},{"source_catalog",c->source_catalog},{"source_checksum",c->receipt.source_generation},
        {"segments",Json::array({{{"name","segment-00000000.bin"},{"rank",c->record.key.rank},{"bytes",c->record.length}}})},
        {"records",Json::array({{{"spec",c->spec},{"segment",0},{"offset",0},{"bytes",c->record.length},{"checksum",c->record.checksum}}})}};
    const auto serialized=manifest.dump();need(serialized.size()<=l.max_manifest_bytes,Status::budget,"actual manifest byte bound");c->receipt.manifest_checksum=hash_text(serialized);
    Fd metadata(::openat(temporary.directory.n,"manifest.json",O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));need(metadata.n>=0,Status::io,"exclusive manifest");write_exact(metadata.n,serialized.data(),0,serialized.size());need(!::fsync(metadata.n)&&!::fsync(temporary.directory.n),Status::io,"manifest/directory fsync");
    metadata=Fd{}; // retirement is complete; keep fixed descriptor bound8 at final config audit
    Fd parent(::open(dest.parent_path().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(parent.n>=0,Status::io,"publication parent descriptor");
    PublishedRecords ready;ready.capture_=c; // ALL fallible result work precedes rename.
    // Re-audit AFTER manifest/result/parent preparation. A late manifest-write
    // error or source change must not slip through an earlier source audit.
    inputs.audit();
    need(stamp(source_index.n)==c->index_stamp&&stamp(source_root.n,true)==c->root_stamp,Status::identity,"source index/root changed at final commit gate");
    {Fd config(::open(c->config_path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(config.n>=0&&stamp(config.n)==c->config_stamp,Status::identity,"source config changed at final commit gate");}
    const int published=::renameat2(AT_FDCWD,temporary.path.c_str(),AT_FDCWD,destination.c_str(),1u);const int error=errno;
    need(published==0,error==EEXIST?Status::exists:(error==ENOSYS||error==EINVAL||error==EOPNOTSUPP)?Status::unsupported:Status::io,"atomic NOREPLACE publication failed");
    temporary.published=true;(void)::fsync(parent.n);*out=std::move(ready);
});}
struct MainReader::Impl {Fd directory,manifest,segment;FileStamp directory_stamp,manifest_stamp,segment_stamp;std::shared_ptr<const Capture> capture;Limits limits;};
MainReader::MainReader(std::unique_ptr<Impl> p):impl_(std::move(p)){}MainReader::~MainReader()=default;
Status MainReader::open(const std::string& path,const PublishedRecords& published,const Limits& l,std::unique_ptr<MainReader>* out,std::string* why)noexcept{return operation(why,[&]{
    need(out&&!*out&&published.capture_,Status::invalid,"Reader requires empty output and retained private publication capture");limits(l);text(path,4096);auto p=std::make_unique<Impl>();p->capture=published.capture_;p->limits=l;
    need(p->capture->record.length<=l.max_record_bytes,Status::budget,"Reader actual record extent cap");p->directory=Fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(p->directory.n>=0,Status::io,"native generation directory");p->directory_stamp=stamp(p->directory.n,true);
    p->manifest=Fd(::openat(p->directory.n,"manifest.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(p->manifest.n>=0,Status::io,"native manifest descriptor");p->manifest_stamp=stamp(p->manifest.n);
    need(p->manifest_stamp.bytes&&p->manifest_stamp.bytes<=l.max_manifest_bytes&&
         add(mul(p->manifest_stamp.bytes,8),l.io_chunk_bytes+65536)<=l.max_metadata_bytes,Status::budget,"Reader manifest/parser precharge cap");
    std::string text(size_t(p->manifest_stamp.bytes),'\0');read_exact(p->manifest.n,text.data(),0,text.size());
    need(stamp(p->manifest.n)==p->manifest_stamp&&hash_text(text)==p->capture->receipt.manifest_checksum,Status::checksum,"independent published manifest checksum");const auto j=parse(text,65536);
    need(j.is_object()&&j.size()==9&&j.at("schema")==schema&&j.at("origin")==origin&&j.at("checkpoint_headers_verified")=="selected_tensor_contracts"&&j.at("checkpoint_payload_verified")=="selected_complete_tensors"&&
        j.at("layout_provenance")=="actual stock Builder/Planner/DEVICE canonical identity geometry"&&j.at("source_catalog")==p->capture->source_catalog&&number(j.at("source_checksum"))==p->capture->receipt.source_generation,
        Status::identity,"strict actual-source schema/provenance/fresh captured identity");
    const auto& segments=j.at("segments");const auto& records=j.at("records");need(segments.is_array()&&segments.size()==1&&records.is_array()&&records.size()==1,Status::identity,"single actual rank/expert generation closure");
    const auto& s=segments[0];const auto& r=records[0];const auto& expected=p->capture->record;
    need(s.is_object()&&s.size()==3&&s.at("name")=="segment-00000000.bin"&&number(s.at("rank"))==expected.key.rank&&number(s.at("bytes"))==expected.length&&
         r.is_object()&&r.size()==5&&r.at("spec")==p->capture->spec&&number(r.at("segment"))==0&&number(r.at("offset"))==0&&number(r.at("bytes"))==expected.length&&number(r.at("checksum"))==expected.checksum,
         Status::identity,"actual canonical record rank/member/source recipe mismatch");
    p->segment=Fd(::openat(p->directory.n,"segment-00000000.bin",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(p->segment.n>=0,Status::io,"native segment descriptor");p->segment_stamp=stamp(p->segment.n);
    need(p->segment_stamp.bytes==expected.length,Status::identity,"native segment physical extent");std::vector<std::byte> buffer(l.io_chunk_bytes);
    need(stream_hash(p->segment.n,0,expected.length,buffer)==expected.checksum,Status::checksum,"native record complete checksum");padding(p->segment.n,expected,buffer);
    need(stamp(p->segment.n)==p->segment_stamp&&stamp(p->manifest.n)==p->manifest_stamp&&stamp(p->directory.n,true)==p->directory_stamp,Status::identity,"generation changed during Reader validation");
    auto ready=std::unique_ptr<MainReader>(new MainReader(std::move(p)));*out=std::move(ready);
});}
Status MainReader::read(const Key& key,std::vector<std::byte>* out,std::string* why)const noexcept{return operation(why,[&]{
    need(out&&impl_&&key==impl_->capture->record.key,Status::identity,"unknown rank/layer/expert record key");const auto& info=impl_->capture->record;
    need(info.length<=impl_->limits.max_read_bytes&&info.length<=SIZE_MAX,Status::budget,"one actual record read byte cap");need(stamp(impl_->manifest.n)==impl_->manifest_stamp&&stamp(impl_->segment.n)==impl_->segment_stamp&&stamp(impl_->directory.n,true)==impl_->directory_stamp,Status::identity,"generation changed since Reader binding");
    std::vector<std::byte> ready(size_t(info.length),std::byte{0});read_exact(impl_->segment.n,ready.data(),0,ready.size());need(checksum(ready)==info.checksum,Status::checksum,"selected record checksum");
    need(stamp(impl_->manifest.n)==impl_->manifest_stamp&&stamp(impl_->segment.n)==impl_->segment_stamp&&stamp(impl_->directory.n,true)==impl_->directory_stamp,Status::identity,"generation changed during selected read");*out=std::move(ready);
});}
const Receipt& MainReader::receipt()const noexcept{return impl_->capture->receipt;}
std::span<const RecordInfo> MainReader::records()const noexcept{return {&impl_->capture->record,1};}
Status MainReader::export_catalog(std::shared_ptr<const VerifiedCatalog>* out,std::string* why)const noexcept{return operation(why,[&]{
    need(out&&impl_,Status::invalid,"normalized catalog output/state");
    need(stamp(impl_->directory.n,true)==impl_->directory_stamp&&stamp(impl_->manifest.n)==impl_->manifest_stamp&&stamp(impl_->segment.n)==impl_->segment_stamp,
         Status::identity,"generation changed before normalized export");
    auto result=std::shared_ptr<VerifiedCatalog>(new VerifiedCatalog);result->directory_=impl_->directory_stamp;result->manifest_=impl_->manifest_stamp;result->segment_=impl_->segment_stamp;
    result->record_=impl_->capture->record;result->receipt_=impl_->capture->receipt;
    need(stamp(impl_->directory.n,true)==result->directory_&&stamp(impl_->manifest.n)==result->manifest_&&stamp(impl_->segment.n)==result->segment_,
         Status::identity,"generation changed during normalized export");*out=std::move(result);
});}
} // namespace rad::glm_main_records

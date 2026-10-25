// Reuse the passed v3 POSIX/hash/record/Reader primitives byte-for-byte in this
// translation unit. CMake must NOT also compile main_records.cpp into this target.
#include "main_records.cpp"
#include "bundle_records.h"
#include "pinned_files.h"
#include <optional>
namespace glm53::main_bundle {
namespace mr=main_records;
using mr::Json;using mr::Fd;using mr::FileProof;using mr::Broker;
using mr::FileStamp;
using mr::need;using mr::add;using mr::mul;using mr::hash;using mr::hash_text;
using mr::number;using mr::parse;using mr::stamp;using mr::stamp_json;
using mr::read_exact;using mr::write_exact;using mr::stream_hash;using mr::text;
namespace fs=std::filesystem;
namespace {
constexpr uint64_t stride=3342336,records_per_rank=12096,full_records=48384;
constexpr const char* format="glm53-main-tp4-canonical-mxfp4-e8m0-g32-v1";
struct Raw {uint32_t file=0;uint64_t offset=0,bytes=0,rows=0,width=0,checksum=0;};
struct RawGroup {std::array<Raw,6> planes{};uint64_t component=0;};
struct Header {uint32_t file=0;uint64_t bytes=0,tick=0;Json parsed;};
bool selected(Mode mode,int layer,int expert){return mode==Mode::CompleteMainData||(layer==3&&expert==0)||(layer==44&&expert==1);}
size_t group_index(Mode mode,int layer,int expert){
    need(layer>=3&&layer<=44&&expert>=0&&expert<288,Status::identity,"main key outside complete domain");
    if(mode==Mode::CompleteMainData)return size_t(layer-3)*288+size_t(expert);
    need(selected(mode,layer,expert),Status::identity,"subset is exactly two declared complete experts");return layer==3?0:1;
}
FileStamp proof_stamp(const Json& j){
    need(j.is_object()&&j.size()==5&&j.at("mtime").is_array()&&j.at("mtime").size()==2&&j.at("ctime").is_array()&&j.at("ctime").size()==2,Status::identity,"proof file stamp");
    return {number(j.at("device")),number(j.at("inode")),number(j.at("bytes")),
        int64_t(number(j.at("mtime")[0])),int64_t(number(j.at("mtime")[1])),int64_t(number(j.at("ctime")[0])),int64_t(number(j.at("ctime")[1]))};
}
void valid_limits(const Limits& l,Mode mode){
    need(mode==Mode::CompleteMainData||mode==Mode::SubsetComponent,Status::unsupported,"explicit complete/subset mode");
    need(l.io_chunk_bytes>=4096&&l.io_chunk_bytes<=65536&&l.max_input_fds>=1&&l.max_input_fds<=2&&
         l.max_index_bytes&&l.max_index_bytes<=(16u<<20)&&l.max_header_bytes&&l.max_header_bytes<=(4u<<20)&&
         l.max_config_bytes&&l.max_config_bytes<=(2u<<20)&&l.max_manifest_bytes>=65536&&l.max_manifest_bytes<=(4u<<20)&&
         l.max_proof_bytes&&l.max_proof_bytes<=131072&&l.max_metadata_bytes>=(2u<<20)&&l.max_metadata_bytes<=(256u<<20)&&
         l.max_total_bytes&&l.max_total_bytes<=(uint64_t(200)<<30)&&l.max_records&&l.max_records<=full_records,
         Status::budget,"closed assembler resource limits");
    const uint64_t count=mode==Mode::CompleteMainData?full_records:8;
    need(count<=l.max_records&&mul(count,stride)<=l.max_total_bytes,Status::budget,"explicit complete/subset native-byte admission");
    const uint64_t layer_records=mode==Mode::CompleteMainData?288:1;
    need(add(65536,mul(layer_records,8192+4096+256))<=l.max_manifest_bytes,
         Status::budget,"layer metadata worst-case precharge before JSON/payload work");
}
struct Tree {
    std::string path;Fd directory;bool published=false;
    explicit Tree(const fs::path& destination){auto pattern=(destination.parent_path()/(destination.filename().string()+".bundle.XXXXXX")).string();
        std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);const auto* p=::mkdtemp(name.data());need(p,Status::io,"private bundle staging");
        try{path=p;directory=Fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(directory.n>=0,Status::io,"bundle staging descriptor");}catch(...){::rmdir(p);throw;}}
    ~Tree(){if(published||path.empty())return;
        if(directory.n>=0){char child[128];for(int rank=0;rank<4;++rank)for(int layer=3;layer<=44;++layer){
            std::snprintf(child,sizeof child,"rank-%d-layer-%02d/segment-00000000.bin",rank,layer);::unlinkat(directory.n,child,0);
            std::snprintf(child,sizeof child,"rank-%d-layer-%02d/manifest.json",rank,layer);::unlinkat(directory.n,child,0);
            std::snprintf(child,sizeof child,"rank-%d-layer-%02d",rank,layer);::unlinkat(directory.n,child,AT_REMOVEDIR);}
            ::unlinkat(directory.n,"source-proof.json",0);::unlinkat(directory.n,"bundle.json",0);}
        ::rmdir(path.c_str());}
};
void new_file(int directory,const char* name,const std::string& data){Fd f(::openat(directory,name,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));need(f.n>=0,Status::io,"new owned metadata file");write_exact(f.n,data.data(),0,data.size());need(!::fsync(f.n),Status::io,"owned metadata fsync");}
}
struct Capture {
    Receipt receipt;Mode mode=Mode::SubsetComponent;std::string source_proof,bundle;
    std::vector<Record> records;std::vector<Shard> shards;
    std::array<RankLayout,4> layouts{};
};
Receipt PublishedBundle::receipt()const noexcept{return capture_?capture_->receipt:Receipt{};}
std::span<const Record> PublishedBundle::records()const noexcept{return capture_?std::span<const Record>(capture_->records):std::span<const Record>{};}
std::span<const Shard> PublishedBundle::shards()const noexcept{return capture_?std::span<const Shard>(capture_->shards):std::span<const Shard>{};}
std::span<const RankLayout> PublishedBundle::layouts()const noexcept{return capture_?std::span<const RankLayout>(capture_->layouts):std::span<const RankLayout>{};}
struct Assembler::Impl {
    Limits limits;Mode mode;std::string destination,root;FileStamp root_stamp;
    const rad::Checkpoint* checkpoint_identity=nullptr; // identity only; caller retains immutable owner
    std::shared_ptr<Capture> capture;std::unique_ptr<Tree> temporary;
    std::vector<FileProof> files;std::map<std::string,uint32_t> file_ids;
    std::unique_ptr<Broker> inputs;Json index;std::vector<Header> headers;uint64_t clock=0;
    std::vector<RawGroup> groups;std::vector<std::byte> window;
    uint64_t source_content_id=0,base_metadata=0,geometry_metadata=0;
    bool fatal=false,finished=false;int next_rank=0;
    void charge(uint64_t headers_bytes=0) {
        const auto n=add(base_metadata,add(geometry_metadata,add(mul(headers_bytes,8),mul(limits.max_manifest_bytes,8))));
        need(n<=limits.max_metadata_bytes,Status::budget,"requested assembler metadata reservation before allocation");
        capture->receipt.metadata_reserved=std::max(capture->receipt.metadata_reserved,n);
    }
    void audit(){Fd source(::open(root.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(source.n>=0&&stamp(source.n,true)==root_stamp,Status::identity,"original checkpoint root changed");inputs->audit();}
    Header& header(uint32_t file){for(auto& h:headers)if(h.file==file){h.tick=++clock;return h;}
        if(headers.size()==2){const auto oldest=std::min_element(headers.begin(),headers.end(),[](const auto& a,const auto& b){return a.tick<b.tick;});headers.erase(oldest);}
        const int fd=inputs->get(file);uint64_t size=0;read_exact(fd,&size,0,8);
        need(size&&size<=limits.max_header_bytes&&size<=files[file].stamp.bytes-8,Status::budget,"source header cap");
        uint64_t live=size;for(const auto& h:headers)live=add(live,h.bytes);charge(live);
        std::string bytes(size_t(size),'\0');read_exact(fd,bytes.data(),8,bytes.size());need(stamp(fd)==files[file].stamp,Status::identity,"source changed during header read");
        Header h;h.file=file;h.bytes=size;h.tick=++clock;h.parsed=parse(bytes,300000);need(h.parsed.is_object(),Status::identity,"source header object");headers.push_back(std::move(h));return headers.back();
    }
    RawGroup& group(rad::Checkpoint& ck,int layer,int expert){auto& result=groups[group_index(mode,layer,expert)];if(result.component)return result;
        Json identity=Json::array();
        for(uint32_t projection=0;projection<3;++projection)for(uint32_t plane=0;plane<2;++plane){
            const auto name=mr::raw_name(layer,expert,projection,plane);const auto owner=index.at("weight_map").find(name);
            need(owner!=index.at("weight_map").end()&&owner->is_string(),Status::identity,"source tensor absent from pinned index");const auto found=file_ids.find(owner->get<std::string>());
            need(found!=file_ids.end(),Status::identity,"unknown indexed source owner");const auto id=found->second;auto& h=header(id);const auto tensor=h.parsed.find(name);
            const int64_t rows=projection==2?4096:2048,width=projection==2?(plane?64:1024):(plane?128:2048);
            need(tensor!=h.parsed.end()&&tensor->at("dtype")=="U8"&&tensor->at("shape")==Json::array({rows,width})&&tensor->at("data_offsets").is_array()&&tensor->at("data_offsets").size()==2,
                 Status::identity,"complete raw six-tensor header contract");
            const auto begin=number(tensor->at("data_offsets")[0]),end=number(tensor->at("data_offsets")[1]);const auto bytes=mul(uint64_t(rows),uint64_t(width));
            need(end>=begin&&end-begin==bytes&&end<=files[id].stamp.bytes-8-h.bytes,Status::identity,"source tensor file range");
            auto& raw=result.planes[projection*2+plane];raw={id,add(8+h.bytes,begin),bytes,uint64_t(rows),uint64_t(width),mr::seed};
            const auto* mapped=ck.find(name);need(mapped&&mapped->data&&mapped->rad_dtype==RAD_U8&&mapped->shape==std::vector<int64_t>({rows,width})&&mapped->bytes==int64_t(bytes),Status::identity,"same opened complete raw source mapping");
            const int fd=inputs->get(id);
            for(uint64_t done=0;done<bytes;){const auto n=size_t(std::min<uint64_t>(bytes-done,window.size()));read_exact(fd,window.data(),raw.offset+done,n);
                need(!std::memcmp(window.data(),static_cast<const std::byte*>(mapped->data)+done,n),Status::identity,"mixed mapped/physical source during component verification");raw.checksum=hash(raw.checksum,window.data(),n);done+=n;}
            need(stamp(fd)==files[id].stamp,Status::identity,"source changed during complete tensor proof");
            capture->receipt.complete_tensor_read_bytes=add(capture->receipt.complete_tensor_read_bytes,bytes);
            identity.push_back({{"name",name},{"file",files[id].name},{"offset",raw.offset},{"bytes",raw.bytes},{"checksum",raw.checksum}});
        }
        result.component=hash_text(Json{{"source_content",source_content_id},{"scope","complete_six_raw_tensors"},{"planes",identity}}.dump());need(result.component,Status::identity,"zero component identity");return result;
    }
};
Assembler::Assembler(std::unique_ptr<Impl> p):p_(std::move(p)){}Assembler::~Assembler()=default;
Status Assembler::begin(rad::Checkpoint& ck,const std::string& proof_path,Mode mode,const std::string& destination,const Limits& l,std::unique_ptr<Assembler>* out,std::string* why)noexcept{
    return mr::operation(why,[&]{need(out&&!*out,Status::invalid,"empty assembler output");valid_limits(l,mode);text(destination,4096);text(proof_path,4096);
        auto p=std::make_unique<Impl>();p->limits=l;p->mode=mode;p->checkpoint_identity=&ck;p->capture=std::make_shared<Capture>();p->capture->mode=mode;
        need(fs::path(destination).is_absolute()&&mr::leaf(fs::path(destination).filename().string()),Status::invalid,"new absolute bundle destination");
        const auto parent=fs::canonical(fs::path(destination).parent_path());const auto dest=parent/fs::path(destination).filename();p->destination=dest.string();
        need(fs::is_directory(ck.path())&&!fs::is_symlink(ck.path()),Status::unsupported,"ordinary checkpoint root");p->root=fs::canonical(ck.path()).string();
        for(auto at=dest;!at.empty();){need(at!=fs::path(p->root),Status::invalid,"bundle output must be outside source");const auto up=at.parent_path();if(up==at)break;at=up;}
        struct stat exists{};need(::lstat(p->destination.c_str(),&exists)&&errno==ENOENT,Status::exists,"new bundle destination only");
        Fd proof(::open(proof_path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(proof.n>=0,Status::io,"trusted source proof descriptor");const auto proof_stamp_before=stamp(proof.n);
        need(proof_stamp_before.bytes&&proof_stamp_before.bytes<=l.max_proof_bytes&&
             add(4u<<20,mul(proof_stamp_before.bytes,8))<=l.max_metadata_bytes,Status::budget,"source proof parser precharge cap");p->capture->source_proof.resize(size_t(proof_stamp_before.bytes));
        read_exact(proof.n,p->capture->source_proof.data(),0,p->capture->source_proof.size());need(stamp(proof.n)==proof_stamp_before,Status::identity,"proof changed while reading");
        const auto j=parse(p->capture->source_proof,20000);need(j.at("schema")=="glm53-source-file-hash-proof-v1"&&j.at("repo")=="amd/GLM-5.3-Flash-Quark-MXFP4"&&
            j.at("revision")=="b5688f25491202978c19c4d036eef579f61bbe07"&&j.at("upstream_info_sha256")==pinned_info_sha256&&j.at("root")==p->root&&j.at("files").is_array()&&j.at("files").size()==72,
            Status::identity,"exact pinned source proof domain");
        const bool full=j.at("mode")=="complete";need((full||j.at("mode")=="subset")&&j.at("full_payload_hashes_verified")==full&&(mode!=Mode::CompleteMainData||full),Status::identity,"full mode requires trusted complete72-file proof");
        p->root_stamp=proof_stamp(j.at("root_stamp"));p->files.reserve(72);Json content=Json::array();uint64_t hashed=0;
        for(size_t i=0;i<72;++i){const auto& expected=pinned_files[i];const auto& f=j.at("files")[i];const bool verified=full||!std::string(expected.name).ends_with(".safetensors");
            need(f.at("name")==expected.name&&number(f.at("bytes"))==expected.bytes&&f.at("algorithm")==expected.algorithm&&f.at("expected_hash")==expected.hash&&
                number(f.at("hashed_bytes"))==(verified?expected.bytes:0)&&(verified?f.at("observed_hash")==expected.hash:f.at("observed_hash").is_null()),Status::identity,"exact expected hash/stamp proof row");
            FileProof file;file.name=expected.name;file.path=p->root+"/"+file.name;file.stamp=proof_stamp(f.at("stamp"));need(file.stamp.bytes==expected.bytes,Status::identity,"source expected extent/stamp");
            p->file_ids.emplace(file.name,uint32_t(i));p->files.push_back(std::move(file));if(verified)hashed=add(hashed,expected.bytes);
            content.push_back({{"name",expected.name},{"bytes",expected.bytes},{"algorithm",expected.algorithm},{"hash",expected.hash}});
        }
        need(number(j.at("hashed_bytes"))==hashed&&number(j.at("window_bytes"))==65536,Status::identity,"trusted proof counters");
        p->capture->receipt.source_proof_complete=full;p->capture->receipt.source_proof_hashed_bytes=hashed;
        p->source_content_id=hash_text(Json{{"repo",j.at("repo")},{"revision",j.at("revision")},{"files",content}}.dump());need(p->source_content_id,Status::identity,"zero source content identity");
        const auto count=mode==Mode::CompleteMainData?full_records:8;const auto groups=mode==Mode::CompleteMainData?records_per_rank:2;const auto shards=mode==Mode::CompleteMainData?168:8;
        const auto index_id=p->file_ids.at("model.safetensors.index.json"),config_id=p->file_ids.at("config.json");
        need(p->files[index_id].stamp.bytes<=l.max_index_bytes&&p->files[config_id].stamp.bytes<=l.max_config_bytes,Status::budget,"source index/config extent caps");
        p->base_metadata=4u<<20;
        p->base_metadata=add(p->base_metadata,mul(p->files[index_id].stamp.bytes,8));
        p->base_metadata=add(p->base_metadata,mul(p->files[config_id].stamp.bytes,2));
        p->base_metadata=add(p->base_metadata,mul(l.max_proof_bytes,8));
        p->base_metadata=add(p->base_metadata,mul(count,2*sizeof(Record)));
        p->base_metadata=add(p->base_metadata,mul(shards,2*sizeof(Shard)+512));
        p->base_metadata=add(p->base_metadata,mul(groups,sizeof(RawGroup)));
        p->base_metadata=add(p->base_metadata,l.io_chunk_bytes);
        p->geometry_metadata=64u<<20;p->charge(2*l.max_header_bytes);p->geometry_metadata=0;
        p->capture->receipt.metadata_output_upper_bytes=add(l.max_proof_bytes,mul(shards+1,l.max_manifest_bytes));
        p->capture->records.reserve(size_t(count));p->capture->shards.reserve(size_t(shards));p->groups.resize(size_t(groups));p->window.resize(l.io_chunk_bytes);p->headers.reserve(2);
        p->inputs=std::make_unique<Broker>(p->files,l.max_input_fds);p->audit();const int index=p->inputs->get(index_id);
        std::string index_text(size_t(p->files[index_id].stamp.bytes),'\0');read_exact(index,index_text.data(),0,index_text.size());need(stamp(index)==p->files[index_id].stamp,Status::identity,"source index changed during parse");
        p->index=parse(index_text,1000000);need(p->index.at("weight_map").is_object()&&p->index.at("weight_map").size()==76108,Status::identity,"actual complete source index inventory");
        p->temporary=std::make_unique<Tree>(dest);new_file(p->temporary->directory.n,"source-proof.json",p->capture->source_proof);p->capture->receipt.metadata_written_bytes=p->capture->source_proof.size();p->capture->receipt.io_window_bytes=l.io_chunk_bytes;
        *out=std::unique_ptr<Assembler>(new Assembler(std::move(p)));
    });
}
namespace {
Json record_spec(const rad::Program& program,const main_metadata::RecordLayout& row,uint64_t component){
    Json members=Json::array();
    for(size_t m=0;m<4;++m){const auto h=row.members[m].weight;need(h&&h<program.weights.size(),Status::identity,"actual captured member handle");const auto& w=program.weights[h];Json cuts=Json::array();
        for(size_t part=0;part<row.members[m].slices;++part){const auto& q=row.members[m].physical[part];cuts.push_back({{"part",uint32_t(q.projection)},{"row0",q.row0},{"rows",q.rows},{"byte0",q.byte0},{"bytes",q.bytes_per_row},{"row_bytes",q.row_bytes}});}
        members.push_back({{"weight",w.name},{"source",w.source},{"projection",m<2?"gate_up":"down"},{"role",m%2?"scale":"codes"},
            {"dtype",row.dtype[m]},{"shape",row.shape[m]},{"stored_rank",2},{"stored_shape",row.shape[m]},{"stored_dtype",row.dtype[m]},
            {"alignment",w.align},{"layout_tag",w.layout_tag},{"identity",w.identity},{"offset",row.offsets[m]},{"bytes",row.bytes[m]},
            {"shard",w.decl.shard},{"span",{w.shard_lo,w.shard_hi}},{"row_parts",m<2?Json::array({512,512}):Json::array()},{"rectangles",cuts}});
    }
    return {{"world",row.world},{"rank",row.rank},{"layer",row.layer},{"expert",row.expert},{"class","mxfp4_e8m0_g32"},{"stride",row.stride},{"source_checksum",component},{"members",members}};
}
Json raw_components(const RawGroup& raw,int layer,int expert,const std::vector<FileProof>& files){Json tensors=Json::array();
    for(uint32_t projection=0;projection<3;++projection)for(uint32_t plane=0;plane<2;++plane){const auto& q=raw.planes[projection*2+plane];
        tensors.push_back({{"name",mr::raw_name(layer,expert,projection,plane)},{"file",files[q.file].name},{"offset",q.offset},{"bytes",q.bytes},{"rows",q.rows},{"row_bytes",q.width},{"checksum",q.checksum}});}
    return {{"layer",layer},{"expert",expert},{"component_source_id",raw.component},{"scope","complete_six_raw_tensors"},{"tensors",tensors}};
}
}
Status Assembler::append_rank(rad::Checkpoint& ck,const rad::CheckpointSources& sources,const rad::Program& program,const rad::Plan& plan,std::string* why)noexcept{
    if(!p_||p_->finished||p_->fatal)return Status::invalid;
    auto& p=*p_;const auto status=mr::operation(why,[&]{
        need(&ck==p.checkpoint_identity&&program.ctx.rank==p.next_rank&&p.next_rank<4&&program.ctx.world_size==4&&plan.rank==p.next_rank&&plan.world_size==4,
             Status::identity,"same retained source and exactly once ordered logical ranks");
        p.audit();p.geometry_metadata=64u<<20;uint64_t header_bytes=0;for(const auto& h:p.headers)header_bytes=add(header_bytes,h.bytes);p.charge(header_bytes);
        main_metadata::Quote geometry;main_metadata::Limits gl;gl.max_metadata_bytes=64u<<20;std::string reason;
        const int quoted=main_metadata::quote(ck,sources,program,plan,{},gl,&geometry,&reason);
        need(quoted==RAD_OK,quoted==RAD_E_NOMEM?Status::no_memory:quoted==RAD_E_SCRATCH?Status::budget:Status::unsupported,"actual once/rank source/layout geometry refused");
        ++p.capture->receipt.graph_quotes;p.geometry_metadata=geometry.construction_requested;p.charge(header_bytes);
        need(geometry.records.size()==records_per_rank&&geometry.world==4&&geometry.rank==uint32_t(p.next_rank),Status::identity,"actual complete main geometry universe");
        size_t packed_records=0;
        for(int layer=3;layer<=44;++layer){if(p.mode==Mode::SubsetComponent&&layer!=3&&layer!=44)continue;
            char child[64];const int nn=std::snprintf(child,sizeof child,"rank-%d-layer-%02d",p.next_rank,layer);need(nn>0&&size_t(nn)<sizeof child,Status::invalid,"shard leaf bound");
            need(!::mkdirat(p.temporary->directory.n,child,0700),Status::io,"exclusive layer shard directory");Fd directory(::openat(p.temporary->directory.n,child,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(directory.n>=0,Status::io,"layer shard descriptor");
            Fd segment(::openat(directory.n,"segment-00000000.bin",O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));need(segment.n>=0,Status::io,"exclusive canonical layer segment");
            Shard shard;shard.name=child;shard.rank=uint32_t(p.next_rank);shard.layer=layer;shard.first=p.capture->records.size();Json records=Json::array(),components=Json::array();
            for(int expert=0;expert<288;++expert){if(!selected(p.mode,layer,expert))continue;
                const auto ordinal=size_t(layer-3)*288+size_t(expert);const auto& row=geometry.records[ordinal];
                need(row.world==4&&row.rank==uint32_t(p.next_rank)&&row.layer==layer&&row.expert==expert&&row.stride==int64_t(stride),Status::identity,"actual sorted complete geometry key/stride");
                RankLayout layout;layout.world=row.world;layout.rank=row.rank;layout.stride=uint64_t(row.stride);layout.offsets=row.offsets;layout.bytes=row.bytes;layout.dtype=row.dtype;layout.shape=row.shape;
                for(size_t m=0;m<4;++m){const auto& w=program.weights[row.members[m].weight];layout.span[m]={w.shard_lo,w.shard_hi};}
                if(!packed_records)p.capture->layouts[size_t(p.next_rank)]=layout;
                else need(p.capture->layouts[size_t(p.next_rank)]==layout,Status::identity,"every record matches immutable rank layout template");
                auto& raw=p.group(ck,layer,expert);Record output;output.native.key={4,uint32_t(p.next_rank),layer,expert};output.native.offset=mul(shard.count,stride);output.native.length=stride;
                output.shard=uint32_t(p.capture->shards.size());output.component_source_id=raw.component;
                uint64_t member_end=0;
                for(size_t m=0;m<4;++m){need(row.offsets[m]>=0&&row.bytes[m]>0&&uint64_t(row.offsets[m])==member_end,Status::identity,"closed actual record has contiguous canonical members");
                    output.native.member_offsets[m]=uint64_t(row.offsets[m]);output.native.member_bytes[m]=uint64_t(row.bytes[m]);member_end=add(member_end,uint64_t(row.bytes[m]));
                    const auto& member=row.members[m];need(member.slices==(m<2?2u:1u),Status::identity,"actual physical member slice count");uint64_t written=0;
                    for(size_t part=0;part<member.slices;++part){const auto& cut=member.physical[part];const auto projection=uint32_t(cut.projection);need(projection<3&&cut.plane==m%2,Status::identity,"actual physical projection role");
                        const auto& input=raw.planes[projection*2+uint32_t(m%2)];const auto name=mr::raw_name(layer,expert,projection,uint32_t(m%2));const auto* mapped=ck.find(name);
                        need(mapped&&mapped->data&&mapped->bytes==int64_t(input.bytes)&&cut.row0>=0&&cut.rows>0&&cut.byte0>=0&&cut.bytes_per_row>0&&
                             uint64_t(cut.row_bytes)==input.width&&uint64_t(cut.row0)<=input.rows&&uint64_t(cut.rows)<=input.rows-uint64_t(cut.row0)&&
                             uint64_t(cut.byte0)<=input.width&&uint64_t(cut.bytes_per_row)<=input.width-uint64_t(cut.byte0),Status::identity,"actual complete physical cut inside source");
                        const int fd=p.inputs->get(input.file);const auto rows_per_band=p.window.size()/size_t(input.width);need(rows_per_band,Status::budget,"one source row must fit bounded copy window");
                        for(uint64_t r=0;r<uint64_t(cut.rows);){const auto rows=std::min<uint64_t>(rows_per_band,uint64_t(cut.rows)-r);const auto at=add(input.offset,mul(uint64_t(cut.row0)+r,input.width));const auto bytes=size_t(mul(rows,input.width));
                            read_exact(fd,p.window.data(),at,bytes);p.capture->receipt.source_band_read_bytes=add(p.capture->receipt.source_band_read_bytes,bytes);need(!std::memcmp(p.window.data(),static_cast<const std::byte*>(mapped->data)+(at-input.offset),bytes),Status::identity,"current rank mapped/source band identity");
                            // Compact COL rectangles in place, forward in the same64KiB buffer.
                            // This avoids thousands of tiny per-row syscalls without a second
                            // record/fused tensor buffer or changing any byte/nibble.
                            for(uint64_t k=0;k<rows;++k)std::memmove(p.window.data()+k*uint64_t(cut.bytes_per_row),p.window.data()+k*input.width+uint64_t(cut.byte0),size_t(cut.bytes_per_row));
                            const auto selected_bytes=mul(rows,uint64_t(cut.bytes_per_row));write_exact(segment.n,p.window.data(),add(output.native.offset,add(uint64_t(row.offsets[m]),written)),size_t(selected_bytes));
                            written=add(written,selected_bytes);r+=rows;
                        }
                        need(stamp(fd)==p.files[input.file].stamp,Status::identity,"source changed during bounded copy");
                    }
                    need(written==uint64_t(row.bytes[m]),Status::identity,"native member complete byte coverage");p.capture->receipt.selected_copy_bytes=add(p.capture->receipt.selected_copy_bytes,written);
                }
                need(member_end==stride,Status::identity,"actual main record has no hidden tail padding");
                output.native.checksum=stream_hash(segment.n,output.native.offset,stride,p.window);mr::padding(segment.n,output.native,p.window);
                auto spec=record_spec(program,row,raw.component);const auto spec_text=spec.dump();need(spec_text.size()<=8192,Status::budget,"bounded actual record metadata");output.spec_checksum=hash_text(spec_text);
                records.push_back({{"spec",std::move(spec)},{"segment",0},{"offset",output.native.offset},{"bytes",stride},{"checksum",output.native.checksum}});
                auto component=raw_components(raw,layer,expert,p.files);need(component.dump().size()<=4096,Status::budget,"bounded per-expert physical source metadata");
                components.push_back(std::move(component));p.capture->records.push_back(output);++shard.count;++packed_records;
            }
            need(shard.count==(p.mode==Mode::CompleteMainData?288u:1u),Status::identity,"exact layer-shard domain closure");shard.bytes=mul(shard.count,stride);need(stamp(segment.n).bytes==shard.bytes&&!::fsync(segment.n),Status::io,"complete canonical layer segment fsync/extent");
            Json manifest{{"schema","radiance-native-record-store-v2"},{"origin","verified_glm53_main_layer_records_v1"},
                {"checkpoint_headers_verified","actual_complete_main_geometry"},{"checkpoint_payload_verified",p.capture->receipt.source_proof_complete?"trusted_parent_complete72_file_hash_proof_plus_own_selected_complete_tensors":"own_selected_complete_tensors_only"},
                {"layout_provenance","actual stock Builder/Planner/DEVICE canonical identity geometry"},
                {"source_catalog",{{"format_profile",format},{"source_content_id",p.source_content_id},{"proof_reference","../source-proof.json"},{"components",Json::array()}}},
                {"source_checksum",p.source_content_id},{"segments",Json::array({{{"name","segment-00000000.bin"},{"rank",shard.rank},{"bytes",shard.bytes}}})},{"records",Json::array()}};
            manifest.at("records")=std::move(records);
            manifest.at("source_catalog").at("components")=std::move(components);
            const auto serialized=manifest.dump();need(serialized.size()<=p.limits.max_manifest_bytes,Status::budget,"bounded layer manifest serialization");shard.manifest_checksum=hash_text(serialized);
            new_file(directory.n,"manifest.json",serialized);p.capture->receipt.metadata_written_bytes=add(p.capture->receipt.metadata_written_bytes,serialized.size());need(!::fsync(directory.n),Status::io,"layer directory fsync");p.capture->shards.push_back(std::move(shard));
        }
        need(packed_records==(p.mode==Mode::CompleteMainData?records_per_rank:2),Status::identity,"exact per-rank catalog closure");p.audit();
        p.capture->receipt.peak_input_fds=std::max(p.capture->receipt.peak_input_fds,p.inputs->peak);++p.next_rank;p.geometry_metadata=0;
    });
    if(status!=Status::ok)p.fatal=true;return status;
}
Status Assembler::finish(PublishedBundle* out,std::string* why)noexcept{
    if(!p_||p_->finished||p_->fatal)return Status::invalid;auto& p=*p_;
    const auto status=mr::operation(why,[&]{need(out&&p.next_rank==4&&p.capture->receipt.graph_quotes==4,Status::identity,"all four once-quoted rank catalogs required");
        const auto expected=p.mode==Mode::CompleteMainData?full_records:8;need(p.capture->records.size()==expected&&p.capture->shards.size()==(p.mode==Mode::CompleteMainData?168u:8u),Status::identity,"complete mode/subset record and shard closure");
        need(p.capture->receipt.selected_copy_bytes==mul(expected,stride)&&p.capture->receipt.complete_tensor_read_bytes==mul(expected,stride),
             Status::identity,"exact complete-domain selected/source byte closure");
        uint64_t catalog=mr::seed;Json shards=Json::array();
        for(const auto& record:p.capture->records){const auto& key=record.native.key;const auto encoded=Json{{"world",key.world},{"rank",key.rank},{"layer",key.layer},{"expert",key.expert},
            {"shard",record.shard},{"offset",record.native.offset},{"length",record.native.length},{"checksum",record.native.checksum},{"component",record.component_source_id},{"spec_checksum",record.spec_checksum},
            {"member_offsets",record.native.member_offsets},{"member_bytes",record.native.member_bytes}}.dump();catalog=hash(catalog,reinterpret_cast<const std::byte*>(encoded.data()),encoded.size());}
        for(const auto& shard:p.capture->shards)shards.push_back({{"name",shard.name},{"rank",shard.rank},{"layer",shard.layer},{"bytes",shard.bytes},{"first",shard.first},{"records",shard.count},{"manifest_checksum",shard.manifest_checksum}});
        // The final ID is outside the shard manifests. Nothing is rewritten to
        // insert it there, so their finalized hashes cannot be circular.
        Json layouts=Json::array();for(const auto& layout:p.capture->layouts)layouts.push_back({{"world",layout.world},{"rank",layout.rank},{"stride",layout.stride},
            {"offsets",layout.offsets},{"bytes",layout.bytes},{"dtype",layout.dtype},{"shape",layout.shape},{"span",layout.span}});
        Json identity{{"format_profile",format},{"source_content_id",p.source_content_id},{"mode",p.mode==Mode::CompleteMainData?"complete_main_data":"subset_component_NONREADY"},
            {"world",4},{"records",expected},{"ordered_catalog_checksum",catalog},{"layouts",layouts},{"shards",shards}};
        p.capture->receipt.bundle_id=hash_text(identity.dump());need(p.capture->receipt.bundle_id,Status::identity,"zero final common bundle identity");
        p.capture->receipt.records=p.capture->records.size();p.capture->receipt.shards=p.capture->shards.size();p.capture->receipt.total_bytes=mul(expected,stride);
        p.capture->receipt.complete_main_domain=p.mode==Mode::CompleteMainData;
        Json bundle{{"schema","glm53-main-four-rank-data-bundle-v1"},{"identity",identity},{"bundle_id",p.capture->receipt.bundle_id},
            {"source_proof_checksum",hash_text(p.capture->source_proof)},{"source_proof_complete",p.capture->receipt.source_proof_complete},
            {"trusted_producer_transcript_not_independent_Cpp_crypto",true},{"runtime_ready",false}};
        p.capture->bundle=bundle.dump();need(p.capture->bundle.size()<=p.limits.max_manifest_bytes,Status::budget,"bounded final bundle metadata");new_file(p.temporary->directory.n,"bundle.json",p.capture->bundle);p.capture->receipt.metadata_written_bytes=add(p.capture->receipt.metadata_written_bytes,p.capture->bundle.size());
        need(!::fsync(p.temporary->directory.n),Status::io,"bundle directory fsync");Fd parent(::open(fs::path(p.destination).parent_path().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(parent.n>=0,Status::io,"bundle parent descriptor");
        PublishedBundle result;result.capture_=p.capture; // final nonfallible commit is prepared
        p.audit();const int renamed=::renameat2(AT_FDCWD,p.temporary->path.c_str(),AT_FDCWD,p.destination.c_str(),1u);const int error=errno;
        need(!renamed,error==EEXIST?Status::exists:(error==ENOSYS||error==EINVAL||error==EOPNOTSUPP)?Status::unsupported:Status::io,"final common-bundle NOREPLACE publication");
        p.temporary->published=true;p.finished=true;(void)::fsync(parent.n);*out=std::move(result);
    });
    if(status!=Status::ok)p.fatal=true;return status;
}
Status verify(const std::string& path,const PublishedBundle& published,const Limits& limits,std::string* why)noexcept{
    return mr::operation(why,[&]{need(bool(published.capture_),Status::invalid,"independent retained publication capture required");const auto& capture=*published.capture_;
        valid_limits(limits,capture.mode);
        need(capture.receipt.metadata_reserved<=limits.max_metadata_bytes,Status::budget,"verification metadata request exceeds caller cap before I/O");
        Fd root(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(root.n>=0,Status::io,"published bundle root");const auto root_before=stamp(root.n,true);
        auto metadata=[](int fd,size_t cap,FileStamp* saved){need(fd>=0,Status::io,"published metadata descriptor");const auto before=stamp(fd);need(before.bytes&&before.bytes<=cap,Status::budget,"published metadata cap");std::string data(size_t(before.bytes),'\0');read_exact(fd,data.data(),0,data.size());need(stamp(fd)==before,Status::identity,"metadata changed while verifying");*saved=before;return data;};
        Fd bundle_file(::openat(root.n,"bundle.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));Fd proof_file(::openat(root.n,"source-proof.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));FileStamp bundle_stamp,proof_stamp;
        need(metadata(bundle_file.n,limits.max_manifest_bytes,&bundle_stamp)==capture.bundle&&metadata(proof_file.n,limits.max_proof_bytes,&proof_stamp)==capture.source_proof,Status::checksum,"exact finalized bundle/proof bytes");
        std::vector<std::byte> buffer(limits.io_chunk_bytes);
        for(size_t si=0;si<capture.shards.size();++si){const auto& expected=capture.shards[si];Fd directory(::openat(root.n,expected.name.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(directory.n>=0,Status::io,"published layer directory");const auto dir_before=stamp(directory.n,true);
            Fd manifest_file(::openat(directory.n,"manifest.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));FileStamp manifest_stamp;
            const auto serialized=metadata(manifest_file.n,limits.max_manifest_bytes,&manifest_stamp);need(hash_text(serialized)==expected.manifest_checksum,Status::checksum,"independent layer manifest checksum");const auto manifest=parse(serialized,1000000);
            need(manifest.at("schema")=="radiance-native-record-store-v2"&&manifest.at("origin")=="verified_glm53_main_layer_records_v1"&&
                manifest.at("source_catalog").at("proof_reference")=="../source-proof.json"&&manifest.at("records").is_array()&&manifest.at("records").size()==expected.count&&
                manifest.at("segments").is_array()&&manifest.at("segments").size()==1&&manifest.at("segments")[0].at("name")=="segment-00000000.bin"&&number(manifest.at("segments")[0].at("bytes"))==expected.bytes,
                Status::identity,"actual layer manifest domain and segment contract");
            Fd segment(::openat(directory.n,"segment-00000000.bin",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(segment.n>=0,Status::io,"published segment");const auto before=stamp(segment.n);need(before.bytes==expected.bytes,Status::identity,"published segment extent");
            for(size_t i=0;i<expected.count;++i){const auto& record=capture.records[expected.first+i];const auto& row=manifest.at("records")[i];const auto& key=record.native.key;const auto& spec=row.at("spec");
                need(record.shard==si&&hash_text(spec.dump())==record.spec_checksum&&number(spec.at("world"))==key.world&&number(spec.at("rank"))==key.rank&&number(spec.at("layer"))==uint64_t(key.layer)&&number(spec.at("expert"))==uint64_t(key.expert)&&
                    number(row.at("segment"))==0&&number(row.at("offset"))==record.native.offset&&number(row.at("bytes"))==record.native.length&&number(row.at("checksum"))==record.native.checksum,
                    Status::identity,"every immutable record key/spec/range");
                need(stream_hash(segment.n,record.native.offset,record.native.length,buffer)==record.native.checksum,Status::checksum,"every complete native record checksum");mr::padding(segment.n,record.native,buffer);
            }
            need(stamp(segment.n)==before&&stamp(manifest_file.n)==manifest_stamp&&stamp(directory.n,true)==dir_before,Status::identity,"published layer changed during validation");
        }
        need(stamp(bundle_file.n)==bundle_stamp&&stamp(proof_file.n)==proof_stamp&&stamp(root.n,true)==root_before,Status::identity,"published bundle changed during validation");
    });
}
} // namespace glm53::main_bundle

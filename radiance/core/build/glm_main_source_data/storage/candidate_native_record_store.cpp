#include "candidate_native_record_store.h"
#include "format/share.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace rad::glm_main_native_store {
using Json=nlohmann::json;
struct SpecData {
    struct Member {std::string source,role;size_t plane=0;uint64_t offset=0,bytes=0;PlaneShare share;Json metadata;};
    Key key;uint64_t stride=0,source_fingerprint=0;std::vector<Member> members;Json metadata;
};
namespace {
constexpr uint64_t alignment=4096,seed=14695981039346656037ull,prime=1099511628211ull;
struct Failure {Status status;std::string text;};
[[noreturn]] void refuse(Status s,const std::string& m){throw Failure{s,m};}
void check(bool v,Status s,const std::string& m){if(!v)refuse(s,m);}
uint64_t add(uint64_t a,uint64_t b){check(b<=UINT64_MAX-a,Status::invalid,"addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){check(!a||b<=UINT64_MAX/a,Status::invalid,"multiplication overflow");return a*b;}
uint64_t positive(int64_t n){check(n>0,Status::invalid,"nonpositive geometry");return uint64_t(n);}
uint64_t number(const Json& j){check(j.is_number_integer()&&!j.is_boolean(),Status::invalid,"integer field required");if(j.is_number_unsigned())return j.get<uint64_t>();const auto n=j.get<int64_t>();check(n>=0,Status::invalid,"negative integer");return uint64_t(n);}
uint64_t hash_bytes(uint64_t h,const std::byte* p,size_t n){for(size_t i=0;i<n;++i){h^=std::to_integer<uint8_t>(p[i]);h*=prime;}return h;}
uint64_t hash_text(const std::string& s){return hash_bytes(seed,reinterpret_cast<const std::byte*>(s.data()),s.size());}
void string_ok(const std::string& s,size_t cap){check(!s.empty()&&s.size()<=cap&&s.find('\0')==std::string::npos,Status::invalid,"empty/oversized/NUL string");}
void limits_ok(const Limits& l){check(l.max_total_bytes&&l.max_total_bytes<=uint64_t(INT64_MAX)&&l.max_segment_bytes>=alignment&&l.max_segment_bytes%alignment==0&&l.max_segment_bytes<=l.max_total_bytes&&l.max_record_bytes&&l.max_record_bytes<=uint64_t(INT64_MAX)&&l.max_source_bytes&&l.max_source_bytes<=uint64_t(INT64_MAX)&&l.max_read_bytes<=SIZE_MAX&&l.max_manifest_bytes&&l.max_manifest_bytes<=(16u<<20)&&l.max_records&&l.max_records<=65536&&l.max_sources&&l.max_sources<=65536&&l.max_segments&&l.max_segments<=65536&&l.max_string_bytes&&l.max_string_bytes<=4096&&l.io_chunk_bytes>=4096&&l.io_chunk_bytes<=(1u<<20)&&l.max_manifest_nodes&&l.max_manifest_nodes<=1048576,Status::invalid,"invalid explicit limits");}
RecordClass classify(const RadEncoding& e){
    check(e.n_planes==2&&std::memchr(e.scheme,0,RAD_ENC_STR)&&std::memchr(e.transform,0,RAD_ENC_STR),Status::invalid,"encoding strings/planes");for(int i=0;i<2;++i)check(std::memchr(e.plane[i].role,0,RAD_ENC_STR),Status::invalid,"plane role termination");
    const auto main=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32),mtp=rad_enc_affine(RAD_F8E4M3,RAD_F32,128,128);
    if(rad_enc_equal(&e,&main))return RecordClass::main_mxfp4_e8m0;
    if(rad_enc_equal(&e,&mtp))return RecordClass::mtp_fp8_f32;
    refuse(Status::unsupported,"only canonical main/MTP encodings are supported");
}
const char* cls(RecordClass c){if(c==RecordClass::main_mxfp4_e8m0)return "mxfp4_e8m0_g32";if(c==RecordClass::mtp_fp8_f32)return "fp8_e4m3_f32_b128";refuse(Status::invalid,"unknown record class");}
Json enc_json(const RadEncoding& e){Json planes=Json::array();for(int k=0;k<2;++k){const auto& p=e.plane[k];planes.push_back({{"role",p.role},{"dtype",p.dtype},{"kind",p.kind},{"block",{p.block[0],p.block[1]}},{"extent",{p.extent[0],p.extent[1]}}});}return {{"scheme",e.scheme},{"transform",e.transform},{"planes",planes}};}
const SourceTensor& source(const SourceCatalog& c,const std::string& logical){const auto i=std::find_if(c.tensors.begin(),c.tensors.end(),[&](const auto& t){return t.logical==logical;});check(i!=c.tensors.end(),Status::identity,"missing logical source");return *i;}
Json catalog_json(const SourceCatalog& c,const Limits& l){
    string_ok(c.source_id,l.max_string_bytes);string_ok(c.revision,l.max_string_bytes);check(!c.tensors.empty()&&c.tensors.size()<=l.max_sources,Status::budget,"source count cap");
    std::vector<const SourceTensor*> sorted;sorted.reserve(c.tensors.size());for(const auto& t:c.tensors)sorted.push_back(&t);std::sort(sorted.begin(),sorted.end(),[](auto a,auto b){return a->logical<b->logical;});
    Json tensors=Json::array();uint64_t total=0;std::string last;
    for(const auto* t:sorted){string_ok(t->logical,l.max_string_bytes);check(t->logical!=last,Status::identity,"duplicate source");last=t->logical;const auto type=classify(t->encoding);const auto rows=positive(t->shape[0]),cols=positive(t->shape[1]);check(rows<=1048576&&cols<=1048576&&mul(rows,cols)<=uint64_t(INT64_MAX),Status::budget,"source dimensions cap");check(t->projection==Projection::gate_up||t->projection==Projection::down,Status::invalid,"projection");
        Json planes=Json::array();for(int k=0;k<2;++k){const auto& p=t->planes[size_t(k)];string_ok(p.path,4096);int64_t pr=0,pc=0;rad_enc_plane_dims(&t->encoding.plane[k],int64_t(rows),int64_t(cols),&pr,&pc);const uint64_t row_bytes=uint64_t(rad_enc_row_bytes(t->encoding.plane[k].dtype,pc));const uint64_t bytes=mul(positive(pr),row_bytes);check(p.bytes==bytes&&add(p.offset,p.bytes)<=uint64_t(INT64_MAX),Status::invalid,"source plane extent/geometry");total=add(total,bytes);check(total<=l.max_source_bytes,Status::budget,"total source plane cap");planes.push_back({{"role",t->encoding.plane[k].role},{"offset",p.offset},{"bytes",p.bytes},{"checksum",p.checksum}});}
        tensors.push_back({{"logical",t->logical},{"projection",t->projection==Projection::gate_up?"gate_up":"down"},{"class",cls(type)},{"shape",t->shape},{"encoding",enc_json(t->encoding)},{"planes",planes}});
    }
    return {{"source_id",c.source_id},{"revision",c.revision},{"tensors",tensors}};
}
bool key_less(const Key& a,const Key& b){return std::tie(a.world,a.rank,a.layer,a.expert,a.record_class)<std::tie(b.world,b.rank,b.layer,b.expert,b.record_class);}
std::vector<const SpecData*> specs_ok(std::span<const RecordSpec> specs,const SourceCatalog& catalog,const Limits& l,uint64_t* fingerprint){
    limits_ok(l);const auto cj=catalog_json(catalog,l);*fingerprint=hash_text(cj.dump());check(!specs.empty()&&specs.size()<=l.max_records,Status::budget,"record count cap");std::vector<const SpecData*> v;v.reserve(specs.size());uint64_t total=0;
    for(const auto& spec:specs){const auto* d=spec.data();check(d&&d->members.size()==4&&d->source_fingerprint==*fingerprint,Status::identity,"empty/stale source/spec fingerprint");check(d->stride&&d->stride%alignment==0&&d->stride<=l.max_record_bytes&&d->stride<=l.max_segment_bytes,Status::budget,"record/segment cap");total=add(total,d->stride);check(total<=l.max_total_bytes,Status::budget,"store total cap");v.push_back(d);}
    std::sort(v.begin(),v.end(),[](auto a,auto b){return key_less(a->key,b->key);});for(size_t i=1;i<v.size();++i)check(!(v[i-1]->key==v[i]->key),Status::identity,"duplicate native record key");return v;
}
struct Fd {int n=-1;Fd()=default;explicit Fd(int v):n(v){}~Fd(){if(n>=0)::close(n);}Fd(Fd&& o)noexcept:n(o.n){o.n=-1;}Fd& operator=(Fd&& o)noexcept{if(this!=&o){if(n>=0)::close(n);n=o.n;o.n=-1;}return *this;}Fd(const Fd&)=delete;};
struct Stamp {dev_t dev;ino_t ino;off_t size;timespec mt,ct;bool operator==(const Stamp& o)const{return dev==o.dev&&ino==o.ino&&size==o.size&&mt.tv_sec==o.mt.tv_sec&&mt.tv_nsec==o.mt.tv_nsec&&ct.tv_sec==o.ct.tv_sec&&ct.tv_nsec==o.ct.tv_nsec;}};
Stamp stamp(int fd){struct stat s{};check(!::fstat(fd,&s)&&S_ISREG(s.st_mode)&&s.st_size>=0,Status::io,"regular file fstat");return {s.st_dev,s.st_ino,s.st_size,s.st_mtim,s.st_ctim};}
FileStamp normalized(const Stamp& s){return {uint64_t(s.dev),uint64_t(s.ino),uint64_t(s.size),int64_t(s.mt.tv_sec),int64_t(s.mt.tv_nsec),int64_t(s.ct.tv_sec),int64_t(s.ct.tv_nsec)};}
FileStamp public_stamp(const struct stat& s){return {uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),int64_t(s.st_mtim.tv_sec),int64_t(s.st_mtim.tv_nsec),int64_t(s.st_ctim.tv_sec),int64_t(s.st_ctim.tv_nsec)};}
FileStamp directory_stamp(int fd){struct stat s{};check(!::fstat(fd,&s)&&S_ISDIR(s.st_mode)&&s.st_size>=0,Status::io,"directory fstat");return public_stamp(s);}
FileStamp regular_stamp(int fd){struct stat s{};check(!::fstat(fd,&s)&&S_ISREG(s.st_mode)&&s.st_size>=0,Status::io,"regular file fstat");return public_stamp(s);}
void read_exact(int fd,void* dst,uint64_t at,size_t n){check(add(at,n)<=uint64_t(INT64_MAX),Status::invalid,"pread offset overflow");size_t done=0;while(done<n){const auto k=::pread(fd,static_cast<std::byte*>(dst)+done,n-done,off_t(at+done));if(k<0&&errno==EINTR)continue;check(k>0,Status::io,"short/failed pread");done+=size_t(k);}}
void write_exact(int fd,const void* src,uint64_t at,size_t n){check(add(at,n)<=uint64_t(INT64_MAX),Status::invalid,"pwrite offset overflow");size_t done=0;while(done<n){const auto k=::pwrite(fd,static_cast<const std::byte*>(src)+done,n-done,off_t(at+done));if(k<0&&errno==EINTR)continue;check(k>0,Status::io,"short/failed pwrite");done+=size_t(k);}}
uint64_t stream_hash(int fd,uint64_t offset,uint64_t bytes,std::vector<std::byte>& buf,bool zero=false){uint64_t h=seed;for(uint64_t at=0;at<bytes;){const auto n=size_t(std::min<uint64_t>(bytes-at,buf.size()));read_exact(fd,buf.data(),add(offset,at),n);if(zero)check(std::all_of(buf.begin(),buf.begin()+n,[](auto v){return v==std::byte{0};}),Status::invalid,"nonzero record padding");h=hash_bytes(h,buf.data(),n);at+=n;}return h;}
struct Input {Fd file;Stamp initial;};
using Inputs=std::map<std::string,Input>;
Inputs open_inputs(const SourceCatalog& c,const Limits& l,std::vector<std::byte>& buf){Inputs out;for(const auto& t:c.tensors)for(const auto& p:t.planes){auto i=out.find(p.path);if(i==out.end()){Fd fd(::open(p.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));check(fd.n>=0,Status::io,"open source");const auto s=stamp(fd.n);check(uint64_t(s.size)<=l.max_source_bytes,Status::budget,"source file extent cap");i=out.emplace(p.path,Input{std::move(fd),s}).first;}check(p.offset<=uint64_t(i->second.initial.size)&&p.bytes<=uint64_t(i->second.initial.size)-p.offset,Status::invalid,"source range outside actual file");check(stream_hash(i->second.file.n,p.offset,p.bytes,buf)==p.checksum,Status::checksum,"source plane checksum");}return out;}
struct TempDir {
    std::string path;Fd fd;bool published=false;
    explicit TempDir(const std::filesystem::path& dest){std::string pattern=(dest.parent_path()/(dest.filename().string()+".staging.XXXXXX")).string();std::vector<char> text(pattern.begin(),pattern.end());text.push_back(0);char* p=::mkdtemp(text.data());check(p,Status::io,"create private staging directory");try{path=p;fd=Fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));check(fd.n>=0,Status::io,"open staging directory");}catch(...){::rmdir(p);throw;}}
    ~TempDir(){if(published||path.empty())return;const int copy=fd.n>=0?::dup(fd.n):-1;DIR* d=copy>=0?::fdopendir(copy):nullptr;if(d){while(const auto* e=::readdir(d)){if(!std::strncmp(e->d_name,"segment-",8)||!std::strcmp(e->d_name,"manifest.json"))::unlinkat(fd.n,e->d_name,0);}::closedir(d);}else if(copy>=0)::close(copy);::rmdir(path.c_str());}
};
std::string segment_name(size_t n){char name[48];std::snprintf(name,sizeof name,"segment-%08zu.bin",n);return name;}
void padding(int fd,uint64_t base,const SpecData& spec,std::vector<std::byte>& buf){uint64_t at=0;for(const auto& m:spec.members){check(m.offset>=at,Status::invalid,"overlapping stored members");if(m.offset>at)stream_hash(fd,add(base,at),m.offset-at,buf,true);at=add(m.offset,m.bytes);}check(at<=spec.stride,Status::invalid,"member past stride");if(at<spec.stride)stream_hash(fd,add(base,at),spec.stride-at,buf,true);}
void explain(std::string* why,const char* message)noexcept{if(why){try{*why=message;}catch(...){}}}
template<class Fn>Status operation(std::string* why,Fn&& fn){try{fn();return Status::ok;}catch(const Failure& e){explain(why,e.text.c_str());return e.status;}catch(const std::bad_alloc&){explain(why,"allocation failed");return Status::no_memory;}catch(const std::exception& e){explain(why,e.what());return Status::invalid;}catch(...){explain(why,"non-standard exception");return Status::invalid;}}
} // namespace
const char* name(Status s)noexcept{switch(s){case Status::ok:return "ok";case Status::invalid:return "invalid";case Status::budget:return "budget";case Status::io:return "io";case Status::checksum:return "checksum";case Status::identity:return "identity";case Status::exists:return "exists";case Status::no_memory:return "no_memory";case Status::unsupported:return "unsupported";}return "unknown";}
uint64_t checksum(std::span<const std::byte> b)noexcept{return hash_bytes(seed,b.data(),b.size());}
Key RecordSpec::key()const{return data_?data_->key:Key{};}uint64_t RecordSpec::stride()const{return data_?data_->stride:0;}
std::array<uint64_t,4> RecordSpec::member_offsets()const{std::array<uint64_t,4> r{};if(data_){for(size_t i=0;i<4;++i)r[i]=data_->members[i].offset;}return r;}
std::array<uint64_t,4> RecordSpec::member_bytes()const{std::array<uint64_t,4> r{};if(data_){for(size_t i=0;i<4;++i)r[i]=data_->members[i].bytes;}return r;}
std::string RecordSpec::metadata_json()const{return data_?data_->metadata.dump():"null";}
Status capture_unit(const Program& p,const Plan& plan,size_t unit,const SourceCatalog& catalog,RecordSpec* out,std::string* why){return capture_unit(p,plan,unit,catalog,Limits{},out,why);}
Status capture_unit(const Program& p,const Plan& plan,size_t unit,const SourceCatalog& catalog,const Limits& limits,RecordSpec* out,std::string* why){return operation(why,[&]{
    check(out&&unit<plan.units.size(),Status::invalid,"capture output/unit");limits_ok(limits);const auto cj=catalog_json(catalog,limits);const auto& mu=plan.units[unit];check(p.ctx.world_size>=1&&p.ctx.world_size<=64&&p.ctx.rank>=0&&p.ctx.rank<p.ctx.world_size&&plan.rank==p.ctx.rank&&plan.world_size==p.ctx.world_size,Status::identity,"rank/world agreement");check(mu.layer>=0&&mu.expert>=0&&mu.weights.size()==4&&mu.offset.size()==4,Status::invalid,"four-member expert movement unit required");
    auto d=std::make_shared<SpecData>();d->key.world=uint32_t(plan.world_size);d->key.rank=uint32_t(plan.rank);d->key.layer=mu.layer;d->key.expert=mu.expert;d->stride=positive(mu.slot_bytes);d->source_fingerprint=hash_text(cj.dump());uint64_t end=0;std::set<std::pair<Projection,std::string>> roles;
    Json members=Json::array();bool first=true;
    for(size_t k=0;k<4;++k){check(mu.weights[k]>0&&mu.weights[k]<p.weights.size(),Status::invalid,"weight handle range");const auto& w=p.weights[mu.weights[k]];string_ok(w.name,limits.max_string_bytes);const auto& t=source(catalog,w.source);const auto type=classify(t.encoding);if(first){d->key.record_class=type;first=false;}else check(type==d->key.record_class,Status::identity,"mixed record classes");check(w.enc_known&&classify(w.enc)==type&&rad_enc_equal(&w.enc,&t.encoding)&&w.identity&&!w.widen&&w.layout_tag.empty()&&w.n_sel==1&&w.sel[0]>=0&&w.sel[0]<2,Status::unsupported,"source encoding/selected/native identity layout");const auto pi=size_t(w.sel[0]);const auto& ep=t.encoding.plane[pi];const std::string role=ep.role;check((role=="codes"||role=="scale")&&roles.emplace(t.projection,role).second,Status::invalid,"duplicate/missing member role");check(w.decl.group.layer==mu.layer&&w.decl.group.expert==mu.expert&&w.decl.rank==2&&w.stored_dtype==ep.dtype&&w.decl.dtype==ep.dtype,Status::identity,"stored dtype/group/geometry");check(w.decl.shard==(t.projection==Projection::gate_up?RAD_SHARD_ROW:RAD_SHARD_COL),Status::invalid,"projection shard orientation");check(w.sel_rows[0]>0&&w.sel_cols[0]>0&&w.sel_rows[0]<=1048576&&w.sel_cols[0]<=1048576&&mul(uint64_t(w.sel_rows[0]),uint64_t(w.sel_cols[0]))<=uint64_t(INT64_MAX),Status::invalid,"selected dimensions");check(w.shard_lo>=0&&w.shard_hi>w.shard_lo&&w.shard_hi<=1048576&&w.n_part_spans==0&&w.decl.n_row_parts==(t.projection==Projection::gate_up?2:0),Status::invalid,"bounded explicit ROWparts2/COL span profile");for(int n=0;n<w.decl.n_row_parts;++n)check(w.decl.row_parts[n]>0&&w.decl.row_parts[n]<=1048576,Status::invalid,"row part bounds");
        PlaneShare shares[RAD_ENC_MAX_PLANES];std::string reason;check(weight_share(w,t.encoding,2,t.shape.data(),plan.world_size,plan.rank,shares,&reason)==RAD_OK,Status::invalid,"actual weight_share refused: "+reason);const auto& sh=shares[0];const uint64_t native_bytes=mul(positive(sh.rows),uint64_t(rad_enc_row_bytes(sh.dtype,sh.cols)));check(native_bytes==positive(w.stored_bytes)&&w.decl.shape[0]==sh.rows&&w.decl.shape[1]==sh.cols,Status::identity,"stored bytes/native share mismatch");check(w.align>0&&w.align<=int64_t(alignment)&&mu.offset[k]>=0&&uint64_t(mu.offset[k])%alignment==0&&uint64_t(mu.offset[k])>=end,Status::invalid,"member offset/alignment/overlap");const uint64_t off=uint64_t(mu.offset[k]);end=add(off,native_bytes);check(end<=d->stride&&d->stride%alignment==0,Status::invalid,"member outside class stride");
        Json rects=Json::array();uint64_t copied=0;for(const auto& r:sh.rects){check(r.row0>=0&&r.rows>0&&r.byte0>=0&&r.bytes>0&&sh.row_bytes>0&&r.byte0<=sh.row_bytes&&r.bytes<=sh.row_bytes-r.byte0,Status::invalid,"share rectangle");const auto extent=add(mul(uint64_t(r.row0+r.rows-1),uint64_t(sh.row_bytes)),add(uint64_t(r.byte0),uint64_t(r.bytes)));check(extent<=t.planes[pi].bytes,Status::invalid,"share outside source plane");copied=add(copied,mul(uint64_t(r.rows),uint64_t(r.bytes)));rects.push_back({{"row0",r.row0},{"rows",r.rows},{"byte0",r.byte0},{"bytes",r.bytes}});}check(copied==native_bytes,Status::invalid,"rectangle total mismatch");
        check(w.stored_rank<=RAD_MAX_RANK,Status::invalid,"stored rank cap");Json stored_shape=Json::array();for(uint32_t d0=0;d0<w.stored_rank;++d0){check(w.stored_shape[d0]>0&&w.stored_shape[d0]<=1048576,Status::invalid,"stored shape cap");stored_shape.push_back(w.stored_shape[d0]);}
        Json metadata{{"weight",w.name},{"source",t.logical},{"projection",t.projection==Projection::gate_up?"gate_up":"down"},{"role",role},{"dtype",ep.dtype},{"encoding",enc_json(w.enc)},{"shape",{sh.rows,sh.cols}},{"stored_rank",w.stored_rank},{"stored_shape",stored_shape},{"stored_dtype",w.stored_dtype},{"alignment",w.align},{"layout_tag",w.layout_tag},{"identity",w.identity},{"offset",off},{"bytes",native_bytes},{"shard",w.decl.shard},{"span",{w.shard_lo,w.shard_hi}},{"row_parts",Json::array()},{"row_bytes",sh.row_bytes},{"rectangles",rects}};for(int n=0;n<w.decl.n_row_parts;++n)metadata["row_parts"].push_back(w.decl.row_parts[n]);members.push_back(metadata);d->members.push_back({t.logical,role,pi,off,native_bytes,sh,metadata});
    }
    check(roles.size()==4&&positive(mu.bytes)==end&&end<=UINT64_MAX-(alignment-1)&&d->stride==(end+alignment-1)/alignment*alignment&&mu.slab_class>=0&&size_t(mu.slab_class)<plan.classes.size()&&plan.classes[size_t(mu.slab_class)].stride==int64_t(d->stride),Status::invalid,"incomplete/inconsistent native unit/class stride");d->metadata={{"world",d->key.world},{"rank",d->key.rank},{"layer",d->key.layer},{"expert",d->key.expert},{"class",cls(d->key.record_class)},{"stride",d->stride},{"source_checksum",d->source_fingerprint},{"members",members}};RecordSpec result;result.data_=std::move(d);*out=std::move(result);
});}
Status pack(std::span<const RecordSpec> specs,const SourceCatalog& catalog,const std::string& destination,const Limits& l,Receipt* out,std::string* why){return operation(why,[&]{
    check(out,Status::invalid,"pack receipt output");uint64_t source_hash=0;const auto ordered=specs_ok(specs,catalog,l,&source_hash);const auto dest=std::filesystem::path(destination);string_ok(destination,4096);check(!dest.filename().empty()&&dest.filename()!="."&&dest.filename()!="..",Status::invalid,"destination basename");struct stat existing{};check(::lstat(destination.c_str(),&existing)!=0&&errno==ENOENT,Status::exists,"destination already exists or is inaccessible");
    Json segments=Json::array(),records=Json::array();uint64_t total=0;uint32_t rank=UINT32_MAX;size_t segment=0;uint64_t used=0;
    for(const auto* spec:ordered){if(segments.empty()||rank!=spec->key.rank||spec->stride>l.max_segment_bytes-used){check(segments.size()<l.max_segments,Status::budget,"segment count cap");segment=segments.size();rank=spec->key.rank;used=0;segments.push_back({{"name",segment_name(segment)},{"rank",rank},{"bytes",uint64_t(0)}});}records.push_back({{"spec",spec->metadata},{"segment",segment},{"offset",used},{"bytes",spec->stride},{"checksum",UINT64_MAX}});used=add(used,spec->stride);segments[segment]["bytes"]=used;total=add(total,spec->stride);}
    Json manifest{{"schema","radiance-synthetic-native-record-store-v1"},{"origin","synthetic_fixture"},{"checkpoint_headers_verified",false},{"checkpoint_payload_verified",false},{"layout_provenance","actual Builder/Planner footprint; not actual R4D row resolution"},{"source_catalog",catalog_json(catalog,l)},{"source_checksum",source_hash},{"segments",segments},{"records",records}};
    check(manifest.dump().size()<=l.max_manifest_bytes,Status::budget,"manifest cap before I/O");std::vector<std::byte> buffer(l.io_chunk_bytes);auto inputs=open_inputs(catalog,l,buffer);TempDir temp(dest);std::vector<Fd> outputs;outputs.reserve(segments.size());
    for(size_t i=0;i<segments.size();++i){Fd fd(::openat(temp.fd.n,segment_name(i).c_str(),O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));check(fd.n>=0,Status::io,"create exclusive segment");outputs.push_back(std::move(fd));}
    for(size_t i=0;i<ordered.size();++i){const auto& spec=*ordered[i];const size_t si=size_t(number(records[i]["segment"]));const uint64_t base=number(records[i]["offset"]);auto& fd=outputs[si];std::fill(buffer.begin(),buffer.end(),std::byte{0});for(uint64_t at=0;at<spec.stride;){const auto n=size_t(std::min<uint64_t>(spec.stride-at,buffer.size()));write_exact(fd.n,buffer.data(),add(base,at),n);at+=n;}
        for(const auto& member:spec.members){const auto& src=source(catalog,member.source).planes[member.plane];auto& input=inputs.at(src.path);uint64_t destination_at=add(base,member.offset);
            for(const auto& rect:member.share.rects)for(int64_t row=0;row<rect.rows;++row){uint64_t source_at=add(src.offset,add(mul(uint64_t(rect.row0+row),uint64_t(member.share.row_bytes)),uint64_t(rect.byte0)));for(uint64_t at=0;at<uint64_t(rect.bytes);){const auto n=size_t(std::min<uint64_t>(uint64_t(rect.bytes)-at,buffer.size()));read_exact(input.file.n,buffer.data(),add(source_at,at),n);write_exact(fd.n,buffer.data(),destination_at,n);destination_at=add(destination_at,n);at+=n;}}
            check(destination_at==add(add(base,member.offset),member.bytes),Status::invalid,"copy extent mismatch");
        }
        records[i]["checksum"]=stream_hash(fd.n,base,spec.stride,buffer);padding(fd.n,base,spec,buffer);
    }
    for(auto& [path,input]:inputs){(void)path;check(stamp(input.file.n)==input.initial,Status::identity,"source changed during pack");}
    for(size_t i=0;i<outputs.size();++i){check(uint64_t(stamp(outputs[i].n).size)==number(segments[i]["bytes"]),Status::invalid,"written segment extent");check(!::fsync(outputs[i].n),Status::io,"segment fsync");}
    manifest["records"]=records;const auto text=manifest.dump();check(text.size()<=l.max_manifest_bytes,Status::budget,"final manifest cap");Fd metadata(::openat(temp.fd.n,"manifest.json",O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));check(metadata.n>=0,Status::io,"create manifest last");write_exact(metadata.n,text.data(),0,text.size());check(!::fsync(metadata.n)&&!::fsync(temp.fd.n),Status::io,"manifest/directory fsync");
    const auto parent=dest.parent_path().empty()?std::filesystem::path("."):dest.parent_path();Fd parentfd(::open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC));
    const Receipt ready{hash_text(text),total,source_hash,ordered.size(),segments.size(),buffer.size()};
    // ALL allocating/fallible preparation is complete before the single publication syscall.
    const long published=::syscall(SYS_renameat2,AT_FDCWD,temp.path.c_str(),AT_FDCWD,destination.c_str(),1u);const int publish_errno=errno;
    if(published!=0){if(publish_errno==ENOSYS||publish_errno==EINVAL||publish_errno==EOPNOTSUPP)refuse(Status::unsupported,"atomic renameat2 NOREPLACE is unavailable; no fallback");refuse(publish_errno==EEXIST?Status::exists:Status::io,"atomic no-replace publication failed");}
    temp.published=true;
    // Parent fsync is best effort: namespace crash-durability is documented separately from
    // complete/atomic publication. No error is returned after publication has happened.
    if(parentfd.n>=0)(void)::fsync(parentfd.n);
    *out=ready;
});}
struct Reader::Impl {
    struct Segment {Fd file;Stamp initial;uint64_t bytes=0;uint32_t rank=0;};
    struct Record {RecordSpec spec;size_t segment=0;uint64_t offset=0,checksum=0;};
    Fd directory,manifest_file;FileStamp directory_stamp{};Stamp manifest_stamp{};std::vector<Segment> segments;std::vector<Record> records;Limits limits;Receipt published;
};
Reader::Reader(std::unique_ptr<Impl> p):impl_(std::move(p)){}Reader::~Reader()=default;
const Receipt& Reader::receipt()const noexcept{return impl_->published;}
Status Reader::open(const std::string& path,std::span<const RecordSpec> expected,const SourceCatalog& catalog,uint64_t expected_manifest_checksum,const Limits& l,std::unique_ptr<Reader>* out,std::string* why){return operation(why,[&]{
    check(out,Status::invalid,"reader output");uint64_t source_hash=0;const auto ordered=specs_ok(expected,catalog,l,&source_hash);string_ok(path,4096);auto p=std::make_unique<Impl>();p->limits=l;p->directory=Fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));check(p->directory.n>=0,Status::io,"open immutable store directory");p->directory_stamp=directory_stamp(p->directory.n);Fd mf(::openat(p->directory.n,"manifest.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));check(mf.n>=0,Status::io,"open manifest");const auto before=stamp(mf.n);check(uint64_t(before.size)<=l.max_manifest_bytes&&before.size>0,Status::budget,"manifest file cap");std::string text(size_t(before.size),0);read_exact(mf.n,text.data(),0,text.size());check(stamp(mf.n)==before,Status::identity,"manifest changed during read");check(hash_text(text)==expected_manifest_checksum,Status::checksum,"published manifest fingerprint mismatch");
    size_t nodes=0;std::vector<std::set<std::string>> object_keys;const auto manifest=Json::parse(text,[&](int depth,Json::parse_event_t event,Json& value){check(depth<=16&&++nodes<=l.max_manifest_nodes,Status::budget,"manifest depth/node cap");if(value.is_string())check(value.get_ref<const std::string&>().size()<=l.max_string_bytes,Status::budget,"manifest string cap");if(event==Json::parse_event_t::object_start)object_keys.emplace_back();if(event==Json::parse_event_t::key){check(!object_keys.empty()&&object_keys.back().insert(value.get<std::string>()).second,Status::invalid,"duplicate manifest key");}if(event==Json::parse_event_t::object_end){check(!object_keys.empty(),Status::invalid,"manifest object nesting");object_keys.pop_back();}return true;});check(manifest.is_object()&&manifest.size()==9&&manifest.at("schema")=="radiance-synthetic-native-record-store-v1"&&manifest.at("origin")=="synthetic_fixture"&&manifest.at("checkpoint_headers_verified")==false&&manifest.at("checkpoint_payload_verified")==false&&manifest.at("layout_provenance")=="actual Builder/Planner footprint; not actual R4D row resolution",Status::identity,"schema/provenance/verification claim");
    check(manifest.at("source_catalog")==catalog_json(catalog,l)&&number(manifest.at("source_checksum"))==source_hash,Status::identity,"fresh source fingerprint mismatch");const auto& segs=manifest.at("segments");const auto& rows=manifest.at("records");check(segs.is_array()&&!segs.empty()&&segs.size()<=l.max_segments&&rows.is_array()&&rows.size()==ordered.size(),Status::budget,"manifest record/segment count");p->segments.reserve(segs.size());p->records.reserve(rows.size());uint64_t total=0;std::vector<uint64_t> covered(segs.size(),0);
    for(size_t i=0;i<segs.size();++i){const auto& s=segs[i];check(s.is_object()&&s.size()==3&&s.at("name")==segment_name(i),Status::invalid,"segment filename/metadata");const auto n=number(s.at("bytes")),rank=number(s.at("rank"));check(n>0&&n<=l.max_segment_bytes&&n%alignment==0&&rank<64,Status::budget,"segment bytes/rank cap");Fd fd(::openat(p->directory.n,segment_name(i).c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));check(fd.n>=0,Status::io,"open segment");const auto st=stamp(fd.n);check(uint64_t(st.size)==n,Status::invalid,"segment actual extent mismatch");p->segments.push_back({std::move(fd),st,n,uint32_t(rank)});}
    std::vector<std::byte> buffer(l.io_chunk_bytes);
    for(size_t i=0;i<rows.size();++i){const auto& r=rows[i];const auto* spec=ordered[i];check(r.is_object()&&r.size()==5&&r.at("spec")==spec->metadata,Status::identity,"fresh native rank/class/layout/member fingerprint mismatch");const auto si=number(r.at("segment")),offset=number(r.at("offset")),bytes=number(r.at("bytes")),crc=number(r.at("checksum"));check(si<p->segments.size()&&bytes==spec->stride&&offset%alignment==0,Status::invalid,"record segment/extent");auto& s=p->segments[size_t(si)];check(s.rank==spec->key.rank&&offset==covered[size_t(si)]&&offset<=s.bytes&&bytes<=s.bytes-offset,Status::invalid,"record overlap/gap/rank/range");covered[size_t(si)]=add(offset,bytes);total=add(total,bytes);check(total<=l.max_total_bytes,Status::budget,"actual store total cap");check(stream_hash(s.file.n,offset,bytes,buffer)==crc,Status::checksum,"native record checksum");padding(s.file.n,offset,*spec,buffer);const auto found=std::find_if(expected.begin(),expected.end(),[&](const auto& v){return v.data()==spec;});check(found!=expected.end(),Status::identity,"expected spec ownership");p->records.push_back({*found,size_t(si),offset,crc});}
    for(size_t i=0;i<p->segments.size();++i){check(covered[i]==p->segments[i].bytes&&stamp(p->segments[i].file.n)==p->segments[i].initial,Status::identity,"uncovered/changed segment");}
    check(stamp(mf.n)==before&&directory_stamp(p->directory.n)==p->directory_stamp,Status::identity,"directory/manifest changed during startup");
    p->manifest_stamp=before;p->manifest_file=std::move(mf);p->published={expected_manifest_checksum,total,source_hash,rows.size(),segs.size(),buffer.size()};*out=std::unique_ptr<Reader>(new Reader(std::move(p)));
});}
Status Reader::export_catalog(std::shared_ptr<const VerifiedCatalog>* out,std::string* why)const{return operation(why,[&]{
    check(out&&impl_,Status::invalid,"catalog export state/output");
    check(directory_stamp(impl_->directory.n)==impl_->directory_stamp&&stamp(impl_->manifest_file.n)==impl_->manifest_stamp,Status::identity,"catalog export directory/manifest changed");
    auto result=std::shared_ptr<VerifiedCatalog>(new VerifiedCatalog);
    result->directory_=impl_->directory_stamp;result->manifest_=normalized(impl_->manifest_stamp);result->receipt_=impl_->published;
    result->segments_.reserve(impl_->segments.size());result->records_.reserve(impl_->records.size());
    for(size_t i=0;i<impl_->segments.size();++i){const auto& segment=impl_->segments[i];
        const auto captured=regular_stamp(segment.file.n);
        check(captured==normalized(segment.initial),Status::identity,"segment changed before catalog export");
        result->segments_.push_back({segment_name(i),captured,segment.rank});
    }
    for(const auto& record:impl_->records){
        check(record.segment<=UINT32_MAX,Status::budget,"normalized segment index");
        result->records_.push_back({record.spec.key(),uint32_t(record.segment),record.offset,record.spec.stride(),record.checksum,record.spec.member_offsets(),record.spec.member_bytes()});
    }
    for(size_t i=0;i<impl_->segments.size();++i)check(regular_stamp(impl_->segments[i].file.n)==result->segments_[i].stamp,Status::identity,"segment changed during catalog export");
    check(directory_stamp(impl_->directory.n)==impl_->directory_stamp&&stamp(impl_->manifest_file.n)==impl_->manifest_stamp,Status::identity,"catalog changed during export");
    *out=std::move(result);
});}
Status Reader::read(const Key& key,std::vector<std::byte>* out,std::string* why)const{return operation(why,[&]{
    check(out&&impl_,Status::invalid,"read output/state");check(stamp(impl_->manifest_file.n)==impl_->manifest_stamp,Status::identity,"manifest changed since binding");const auto it=std::find_if(impl_->records.begin(),impl_->records.end(),[&](const auto& r){return r.spec.key()==key;});check(it!=impl_->records.end(),Status::identity,"unknown rank/class/expert key");const auto bytes=it->spec.stride();check(bytes<=impl_->limits.max_read_bytes&&bytes<=SIZE_MAX,Status::budget,"one-record read allocation cap");auto& segment=impl_->segments[it->segment];check(stamp(segment.file.n)==segment.initial,Status::identity,"segment changed since validation");std::vector<std::byte> ready(size_t(bytes),std::byte{0});read_exact(segment.file.n,ready.data(),it->offset,ready.size());check(checksum(ready)==it->checksum,Status::checksum,"read record checksum");check(stamp(segment.file.n)==segment.initial&&stamp(impl_->manifest_file.n)==impl_->manifest_stamp,Status::identity,"store changed during read");*out=std::move(ready);
});}
} // namespace rad::glm_main_native_store

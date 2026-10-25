#include "physical_source_bridge.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace glm53::engine_core_physical {
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
constexpr uint64_t seed=14695981039346656037ull,prime=1099511628211ull;
constexpr size_t window=4096,metadata_reservation=8u<<20;
struct Failure {Status code;const char* message;};
void require(bool yes,Status code,const char* message){if(!yes)throw Failure{code,message};}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,Status::invalid,"extent overflow");return a+b;}
uint64_t product(uint64_t a,uint64_t b){require(!a||b<=UINT64_MAX/a,Status::invalid,"geometry overflow");return a*b;}
uint64_t hash(uint64_t h,const std::byte* data,size_t n)noexcept{for(size_t i=0;i<n;++i){h^=std::to_integer<uint8_t>(data[i]);h*=prime;}return h;}
void string_ok(const std::string& s){require(!s.empty()&&s.size()<=512&&s.find('\0')==std::string::npos,Status::invalid,"bounded nonempty string");}
const char* part_name(Part p){return p==Part::gate?"gate":p==Part::up?"up":"down";}
struct Fd {
    int n=-1;explicit Fd(int value=-1):n(value){}~Fd(){if(n>=0)::close(n);}
    Fd(const Fd&)=delete;Fd& operator=(const Fd&)=delete;
};
struct Stamp {
    dev_t dev;ino_t ino;off_t bytes;timespec mt,ct;
    bool operator==(const Stamp& s)const{return dev==s.dev&&ino==s.ino&&bytes==s.bytes&&mt.tv_sec==s.mt.tv_sec&&mt.tv_nsec==s.mt.tv_nsec&&ct.tv_sec==s.ct.tv_sec&&ct.tv_nsec==s.ct.tv_nsec;}
};
Stamp stamp(int fd){struct stat s{};require(!::fstat(fd,&s)&&S_ISREG(s.st_mode)&&s.st_size>=0,Status::io,"regular source fstat");return {s.st_dev,s.st_ino,s.st_size,s.st_mtim,s.st_ctim};}
void read_exact(int fd,std::byte* bytes,size_t n,uint64_t at){require(add(at,n)<=uint64_t(INT64_MAX),Status::invalid,"read range");for(size_t done=0;done<n;){const auto got=::pread(fd,bytes+done,n-done,off_t(at+done));if(got<0&&errno==EINTR)continue;require(got>0,Status::io,"physical source read");done+=size_t(got);}}
void write_exact(int fd,const std::byte* bytes,size_t n,uint64_t at){require(add(at,n)<=uint64_t(INT64_MAX),Status::invalid,"write range");for(size_t done=0;done<n;){const auto got=::pwrite(fd,bytes+done,n-done,off_t(at+done));if(got<0&&errno==EINTR)continue;require(got>0,Status::io,"canonical source write");done+=size_t(got);}}
std::string file_name(size_t index){char name[40];std::snprintf(name,sizeof name,"plane-%04zu.bin",index);return name;}
struct Temp {
    std::string path;int fd=-1;bool published=false;
    explicit Temp(const std::string& destination){
        std::string pattern=destination+".staging.XXXXXX";std::vector<char> text(pattern.begin(),pattern.end());text.push_back(0);
        path.reserve(pattern.size());char* p=::mkdtemp(text.data());require(p,Status::io,"private source staging directory");
        try{path=p;fd=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,Status::io,"source staging directory fd");}
        catch(...){::rmdir(p);throw;}
    }
    ~Temp(){if(!published&&fd>=0){for(size_t i=0;i<128;++i){char name[40];std::snprintf(name,sizeof name,"plane-%04zu.bin",i);::unlinkat(fd,name,0);}::unlinkat(fd,"manifest.json",0);::rmdir(path.c_str());}if(fd>=0)::close(fd);}
};
void typed_sources(const rad::Program& program,const rad::Plan& plan,const Catalog& catalog){
    require(program.ctx.world_size==1&&program.ctx.rank==0&&plan.world_size==1&&plan.rank==0&&program.weights.size()<=512&&plan.units.size()<=128&&plan.unit_of_weight.size()==program.weights.size(),Status::identity,"actual bounded TP1 Program/Plan");
    // Validate borrowed strings BEFORE any owning representative map allocation.
    for(const auto& weight:program.weights)require(weight.source.size()<=512&&weight.name.size()<=512&&weight.layout_tag.size()<=512,Status::invalid,"bounded actual Program strings");
    const auto main=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
    std::array<bool,512> seen{};
    std::map<std::string,std::tuple<int32_t,int32_t,native::Projection,const rad::WeightInfo*,const rad::WeightInfo*>> actual;
    size_t units=0;
    for(size_t unit_index=0;unit_index<plan.units.size();++unit_index){const auto& unit=plan.units[unit_index];if(unit.expert<0)continue;
        require(unit.layer>=0&&unit.layer<8&&unit.expert<4&&unit.weights.size()==4&&unit.offset.size()==4&&unit.slot_bytes==32768&&unit.bytes==29184,Status::identity,"actual P3 unit shape");++units;
        constexpr int64_t offsets[]{0,16384,20480,28672},bytes[]{16384,1024,8192,512};
        constexpr int64_t shape[4][2]{{256,128},{256,4},{128,128},{128,4}};
        for(size_t member=0;member<4;++member){const auto handle=unit.weights[member];
            require(handle&&handle<program.weights.size()&&!seen[handle]&&plan.unit_of_weight[handle]==int32_t(unit_index),Status::identity,"actual unique unit/member map");seen[handle]=true;
            const auto& weight=program.weights[handle];const auto dtype=member%2?RAD_E8M0:RAD_FP4E2M1;
            if(!(unit.offset[member]==offsets[member]&&weight.align>0&&weight.align<=4096&&4096%weight.align==0&&weight.stored_bytes==bytes[member]&&
                weight.decl.rank==2&&weight.stored_rank==2&&weight.decl.dtype==dtype&&weight.stored_dtype==dtype&&
                weight.decl.shape[0]==shape[member][0]&&weight.decl.shape[1]==shape[member][1]&&weight.stored_shape[0]==shape[member][0]&&weight.stored_shape[1]==shape[member][1]&&
                weight.sel_rows[0]==shape[member][0]&&weight.sel_cols[0]==shape[member][1]&&
                weight.decl.group.layer==unit.layer&&weight.decl.group.expert==unit.expert&&weight.decl.group.slot==int(member/2)&&
                weight.decl.shard==(member<2?RAD_SHARD_ROW:RAD_SHARD_COL)&&weight.n_part_spans==0&&weight.shard_lo==0&&weight.shard_hi==(member<2?shape[member][0]/2:shape[member][1])&&
                weight.decl.n_row_parts==(member<2?2:0)&&(member>=2||(weight.decl.row_parts[0]==128&&weight.decl.row_parts[1]==128))))std::fprintf(stderr,"AUTHORITY member=%zu offset=%lld align=%lld bytes=%lld dtype=%u/%u shape=%lld,%lld decl=%lld,%lld sel=%lld,%lld group=%d,%d,%d unit=%d,%d shard=%d parts=%d/%d span=%lld,%lld rows=%lld,%lld\n",member,(long long)unit.offset[member],(long long)weight.align,(long long)weight.stored_bytes,weight.decl.dtype,weight.stored_dtype,(long long)weight.stored_shape[0],(long long)weight.stored_shape[1],(long long)weight.decl.shape[0],(long long)weight.decl.shape[1],(long long)weight.sel_rows[0],(long long)weight.sel_cols[0],weight.decl.group.layer,weight.decl.group.expert,weight.decl.group.slot,unit.layer,unit.expert,weight.decl.shard,weight.decl.n_row_parts,weight.n_part_spans,(long long)weight.shard_lo,(long long)weight.shard_hi,(long long)weight.decl.row_parts[0],(long long)weight.decl.row_parts[1]);
            require(unit.offset[member]==offsets[member]&&weight.align>0&&weight.align<=4096&&4096%weight.align==0&&weight.stored_bytes==bytes[member]&&
                weight.decl.rank==2&&weight.stored_rank==2&&weight.decl.dtype==dtype&&weight.stored_dtype==dtype&&
                weight.decl.shape[0]==shape[member][0]&&weight.decl.shape[1]==shape[member][1]&&weight.stored_shape[0]==shape[member][0]&&weight.stored_shape[1]==shape[member][1]&&
                weight.sel_rows[0]==shape[member][0]&&weight.sel_cols[0]==shape[member][1]&&
                weight.decl.group.layer==unit.layer&&weight.decl.group.expert==unit.expert&&weight.decl.group.slot==int(member/2)&&
                weight.decl.shard==(member<2?RAD_SHARD_ROW:RAD_SHARD_COL)&&weight.n_part_spans==0&&weight.shard_lo==0&&weight.shard_hi==(member<2?shape[member][0]/2:shape[member][1])&&
                weight.decl.n_row_parts==(member<2?2:0)&&(member>=2||(weight.decl.row_parts[0]==128&&weight.decl.row_parts[1]==128)),Status::identity,"actual canonical member shape/dtype/group/offset/shard");
        }
        for(size_t projection=0;projection<2;++projection){const auto a=unit.weights[projection*2],b=unit.weights[projection*2+1];
            require(a&&b&&a<program.weights.size()&&b<program.weights.size(),Status::identity,"actual P3 weight handles");const auto& code=program.weights[a];const auto& scale=program.weights[b];
            require(code.source==scale.source&&code.enc_known&&scale.enc_known&&rad_enc_equal(&code.enc,&main)&&rad_enc_equal(&scale.enc,&main)&&code.identity&&scale.identity&&!code.widen&&!scale.widen&&code.stored_rank==2&&scale.stored_rank==2&&code.n_sel==1&&scale.n_sel==1&&code.sel[0]==0&&scale.sel[0]==1,Status::identity,"actual canonical source members");
            require(actual.emplace(code.source,std::make_tuple(unit.layer,unit.expert,projection?native::Projection::down:native::Projection::gate_up,&code,&scale)).second,Status::identity,"actual source name aliases multiple units");
        }
    }
    require(units==32&&actual.size()==64&&catalog.tensors.size()==64,Status::identity,"actual complete P3 source coverage");
    for(const auto& tensor:catalog.tensors){const auto it=actual.find(tensor.expected.logical);require(it!=actual.end(),Status::identity,"physical source foreign to actual Program");
        const auto& [layer,expert,projection,code,scale]=it->second;
        require(tensor.layer==layer&&tensor.expert==expert&&tensor.expected.projection==projection&&rad_enc_equal(&tensor.expected.encoding,&code->enc)&&tensor.expected.shape==std::array<int64_t,2>{code->stored_shape[0],code->stored_shape[1]}&&tensor.expected.planes[0].bytes==uint64_t(code->stored_bytes)&&tensor.expected.planes[1].bytes==uint64_t(scale->stored_bytes),Status::identity,"physical source group/shape differs from actual Program");
    }
}
Json preflight(const Catalog& c,const std::string& destination){
    string_ok(c.source_id);string_ok(c.revision);string_ok(destination);
    require(fs::path(destination).is_absolute()&&destination.size()<=480,Status::invalid,"bounded absolute destination");
    require(c.origin=="complete_synthetic_physical_tp1_v1",Status::identity,"partial/checkpoint source is not complete synthetic authority");
    require(c.tensors.size()==64,Status::identity,"exact64logical TP1 sources");
    const auto main=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
    std::set<std::string> logical_names,physical_names;
    unsigned coverage[8][4]{};size_t count=0;uint64_t source_bytes=0;
    Json tensors=Json::array();
    for(const auto& t:c.tensors){const auto& s=t.expected;string_ok(s.logical);
        require(logical_names.insert(s.logical).second,Status::identity,"duplicate logical source");
        require(t.layer>=0&&t.layer<8&&t.expert>=0&&t.expert<4,Status::identity,"closed source group");
        const bool gu=s.projection==native::Projection::gate_up;
        require(gu||s.projection==native::Projection::down,Status::invalid,"projection role");
        const unsigned bit=gu?1u:2u;require(!(coverage[t.layer][t.expert]&bit),Status::identity,"duplicate group projection");coverage[t.layer][t.expert]|=bit;
        require(rad_enc_equal(&s.encoding,&main),Status::unsupported,"only main FP4/E8M0 TP1 bridge profile");
        require(s.shape==std::array<int64_t,2>{gu?256:128,128},Status::identity,"exact full TP1 source geometry");
        Json planes=Json::array();
        for(size_t p=0;p<2;++p){const uint64_t rows=gu?256:128,row_bytes=p?4:64,parts=gu?2:1;
            require(s.planes[p].bytes==product(rows,row_bytes)&&t.planes[p].size()==parts,Status::identity,"complete logical plane coverage");
            Json pieces=Json::array();
            for(size_t k=0;k<parts;++k){const auto& x=t.planes[p][k];string_ok(x.tensor);string_ok(x.path);
                const auto role=gu?(k?Part::up:Part::gate):Part::down;
                require(x.part==role&&x.tensor==s.logical+"/"+part_name(role)+"/"+(p?"scale":"codes"),Status::identity,"ordered physical role/tensor identity");
                require(physical_names.insert(x.tensor).second,Status::identity,"physical tensor reused");
                require(x.complete&&x.dtype==(p?RAD_E8M0:RAD_FP4E2M1)&&x.rows==rows/parts&&x.row_bytes==row_bytes&&x.bytes==product(x.rows,x.row_bytes),Status::identity,"full physical dtype/rows/bytes");
                require(add(x.offset,x.bytes)<=uint64_t(INT64_MAX),Status::invalid,"physical offset range");
                source_bytes=add(source_bytes,x.bytes);require(source_bytes<=2u<<20,Status::invalid,"source payload cap");++count;
                pieces.push_back({{"tensor",x.tensor},{"path",x.path},{"part",part_name(x.part)},{"dtype",x.dtype},{"offset",x.offset},{"rows",x.rows},{"row_bytes",x.row_bytes},{"bytes",x.bytes},{"checksum",x.checksum},{"complete",x.complete}});
            }
            planes.push_back({{"role",p?"scale":"codes"},{"expected_bytes",s.planes[p].bytes},{"expected_checksum",s.planes[p].checksum},{"pieces",pieces}});
        }
        tensors.push_back({{"logical",s.logical},{"layer",t.layer},{"expert",t.expert},{"projection",gu?"gate_up":"down"},{"shape",s.shape},{"planes",planes}});
    }
    for(const auto& layer:coverage)for(const auto bits:layer)require(bits==3,Status::identity,"missing complete group source");
    require(count==192&&source_bytes==835584,Status::identity,"closed physical source bytes/count");
    return {{"schema","p3-complete-physical-source-bundle-v1"},{"origin",c.origin},{"source_id",c.source_id},{"revision",c.revision},{"tensors",tensors},{"checkpoint_headers_verified",false},{"checkpoint_payload_verified",false}};
}
} // namespace
uint64_t checksum(const std::byte* p,size_t n)noexcept{return hash(seed,p,n);}
bool same_catalog(const native::SourceCatalog& a,const native::SourceCatalog& b)noexcept{
    if(a.source_id!=b.source_id||a.revision!=b.revision||a.tensors.size()!=b.tensors.size())return false;
    for(size_t i=0;i<a.tensors.size();++i){const auto& x=a.tensors[i];const auto& y=b.tensors[i];if(x.logical!=y.logical||x.projection!=y.projection||x.shape!=y.shape||!rad_enc_equal(&x.encoding,&y.encoding))return false;
        for(size_t p=0;p<2;++p){const auto& u=x.planes[p];const auto& v=y.planes[p];if(u.path!=v.path||u.offset!=v.offset||u.bytes!=v.bytes||u.checksum!=v.checksum)return false;}}
    return true;
}
Status materialize(const rad::Program& program,const rad::Plan& plan,const Catalog& c,const std::string& destination,Result* out,std::string* why){
    try{
        require(out,Status::invalid,"source bundle output");typed_sources(program,plan,c);auto manifest=preflight(c,destination);
        struct stat existing{};if(!::lstat(destination.c_str(),&existing))throw Failure{Status::exists,"source destination exists"};require(errno==ENOENT,Status::io,"source destination check");
        std::array<std::byte,window> buffer{};std::map<std::string,Stamp> stamps;
        Result ready;ready.metadata_reservation=metadata_reservation;ready.window_bytes=window;
        const auto physical_text=manifest.dump();require(physical_text.size()<=1u<<20,Status::invalid,"physical manifest cap");
        ready.physical_fingerprint=checksum(reinterpret_cast<const std::byte*>(physical_text.data()),physical_text.size());
        char identity[40];std::snprintf(identity,sizeof identity,"physical-%016llx",static_cast<unsigned long long>(ready.physical_fingerprint));
        ready.sources.source_id=identity;ready.sources.revision="complete-synthetic-physical-tp1-v1";ready.sources.tensors.reserve(64);
        // Verify ALL physical and ordered virtual plane checksums before creating output.
        for(const auto& t:c.tensors)for(size_t p=0;p<2;++p){uint64_t logical=seed;
            for(const auto& piece:t.planes[p]){Fd input(::open(piece.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));require(input.n>=0,Status::io,"open physical tensor");const auto s=stamp(input.n);
                require(uint64_t(s.bytes)<=2u<<20&&piece.offset<=uint64_t(s.bytes)&&piece.bytes<=uint64_t(s.bytes)-piece.offset,Status::identity,"complete physical file extent");
                const auto [it,inserted]=stamps.emplace(piece.path,s);require(inserted||it->second==s,Status::identity,"physical file changed");uint64_t physical=seed;
                for(uint64_t at=0;at<piece.bytes;){const auto n=size_t(std::min<uint64_t>(window,piece.bytes-at));read_exact(input.n,buffer.data(),n,piece.offset+at);physical=hash(physical,buffer.data(),n);logical=hash(logical,buffer.data(),n);ready.source_read_bytes+=n;at+=n;}
                require(physical==piece.checksum,Status::checksum,"complete physical tensor checksum");require(stamp(input.n)==s,Status::identity,"physical file changed during verification");ready.source_bytes+=piece.bytes;++ready.physical_pieces;
            }require(logical==t.expected.planes[p].checksum,Status::checksum,"ordered virtual source checksum");
        }
        Temp temp(destination);size_t plane_index=0;Json outputs=Json::array();
        for(const auto& t:c.tensors){native::SourceTensor target;
            // Physical source paths are authoritative; never copy ignored caller plane paths.
            target.logical=t.expected.logical;target.projection=t.expected.projection;target.shape=t.expected.shape;target.encoding=t.expected.encoding;
            for(size_t p=0;p<2;++p){const auto name=file_name(plane_index++);Fd output(::openat(temp.fd,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));require(output.n>=0,Status::io,"create canonical plane");uint64_t at_out=0,logical=seed;
                for(const auto& piece:t.planes[p]){Fd input(::open(piece.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));require(input.n>=0,Status::io,"reopen physical tensor");const auto s=stamp(input.n);require(stamps.at(piece.path)==s,Status::identity,"source replaced before copy");uint64_t physical=seed;
                    for(uint64_t at=0;at<piece.bytes;){const auto n=size_t(std::min<uint64_t>(window,piece.bytes-at));read_exact(input.n,buffer.data(),n,piece.offset+at);physical=hash(physical,buffer.data(),n);logical=hash(logical,buffer.data(),n);write_exact(output.n,buffer.data(),n,at_out);at_out+=n;ready.source_read_bytes+=n;at+=n;}
                    require(physical==piece.checksum&&stamp(input.n)==s,Status::identity,"source changed during canonical copy");
                }
                require(at_out==t.expected.planes[p].bytes&&logical==t.expected.planes[p].checksum,Status::checksum,"canonical plane equality");require(!::fsync(output.n),Status::io,"canonical plane fsync");
                const auto path=(fs::path(destination)/name).string();string_ok(path);target.planes[p]={path,0,at_out,logical};ready.output_bytes+=at_out;
                outputs.push_back({{"name",name},{"bytes",at_out},{"checksum",logical}});
            }ready.sources.tensors.push_back(std::move(target));
        }
        for(const auto& [path,s]:stamps){Fd input(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));require(input.n>=0&&stamp(input.n)==s,Status::identity,"physical final audit");}
        typed_sources(program,plan,c);
        manifest["canonical_planes"]=std::move(outputs);manifest["canonical_source_id"]=ready.sources.source_id;const auto text=manifest.dump();require(text.size()<=1u<<20,Status::invalid,"source manifest final cap");
        Fd metadata(::openat(temp.fd,"manifest.json",O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));require(metadata.n>=0,Status::io,"source manifest create");write_exact(metadata.n,reinterpret_cast<const std::byte*>(text.data()),text.size(),0);require(!::fsync(metadata.n)&&!::fsync(temp.fd),Status::io,"source manifest/directory fsync");
        ready.manifest_checksum=checksum(reinterpret_cast<const std::byte*>(text.data()),text.size());
        const auto parent=fs::path(destination).parent_path();Fd parent_fd(::open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC));
        const auto renamed=::syscall(SYS_renameat2,AT_FDCWD,temp.path.c_str(),AT_FDCWD,destination.c_str(),1u);const int saved=errno;
        if(renamed!=0)throw Failure{saved==EEXIST?Status::exists:(saved==ENOSYS||saved==EINVAL||saved==EOPNOTSUPP)?Status::unsupported:Status::io,"atomic source publication failed"};
        temp.published=true;if(parent_fd.n>=0)(void)::fsync(parent_fd.n);*out=std::move(ready);return Status::ok;
    }catch(const Failure& f){if(why){try{*why=f.message;}catch(...){}}return f.code;}
     catch(const std::bad_alloc&){return Status::no_memory;}
     catch(...){return Status::invalid;}
}
} // namespace glm53::engine_core_physical

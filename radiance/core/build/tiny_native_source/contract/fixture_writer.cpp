#include "fixture_writer.h"
#include "format/checkpoint.h"
#include "text/tokenizer.h"
#include "tests/tiny_model.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <type_traits>
namespace glm53::engine_core_contract {
namespace {
using Json=nlohmann::ordered_json;namespace fs=std::filesystem;
struct Failure {int status;const char* why;};
void need(bool b,int s,const char* why){if(!b)throw Failure{s,why};}
void explain(std::string* out,const char* s)noexcept{if(out)try{*out=s;}catch(...){}}
uint64_t hash(uint64_t h,const uint8_t* p,size_t n)noexcept{for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
struct Fd {int n=-1;explicit Fd(int x):n(x){}~Fd(){if(n>=0)::close(n);}Fd(const Fd&)=delete;};
void write_all(int fd,const void* p,size_t n){const auto* q=static_cast<const uint8_t*>(p);while(n){const auto k=::write(fd,q,n);if(k<0&&errno==EINTR)continue;need(k>0,RAD_E_IO,"synthetic fixture write");q+=k;n-=size_t(k);}}
void text_file(const fs::path& path,const std::string& text){Fd fd(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));need(fd.n>=0,RAD_E_IO,"fixture sidecar open");write_all(fd.n,text.data(),text.size());need(!::fsync(fd.n),RAD_E_IO,"fixture sidecar fsync");}
struct Temp {
    std::string path;int dir=-1;bool committed=false;
    explicit Temp(const std::string& destination){auto pattern=destination+".staging.XXXXXX";std::vector<char> chars(pattern.begin(),pattern.end());chars.push_back(0);path.reserve(pattern.size());char* s=::mkdtemp(chars.data());need(s,RAD_E_IO,"fixture temporary");try{path=s;dir=::open(s,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);need(dir>=0,RAD_E_IO,"fixture temporary directory");}catch(...){::rmdir(s);throw;}}
    ~Temp()noexcept{if(!committed){if(dir>=0)for(const auto* file:{"model.safetensors","config.json","tokenizer.json","tokenizer_config.json"})::unlinkat(dir,file,0);::rmdir(path.c_str());}if(dir>=0)::close(dir);}
};
uint64_t file_size(const fs::path& p,uint64_t cap){struct stat s{};need(!::lstat(p.c_str(),&s)&&S_ISREG(s.st_mode)&&s.st_size>=0&&uint64_t(s.st_size)<=cap,RAD_E_FORMAT,"bounded regular fixture file");return uint64_t(s.st_size);}
uint64_t fingerprint(const fs::path& p,uint64_t* first_word){Fd fd(::open(p.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,RAD_E_IO,"fixture fingerprint open");std::array<uint8_t,4096> bytes{};uint64_t h=14695981039346656037ull,total=0;for(;;){const auto n=::read(fd.n,bytes.data(),bytes.size());if(n<0&&errno==EINTR)continue;need(n>=0,RAD_E_IO,"fixture fingerprint read");if(!n)break;if(!total&&first_word){need(n>=8,RAD_E_FORMAT,"fixture header length");std::memcpy(first_word,bytes.data(),8);}total+=uint64_t(n);need(total<=2*MiB,RAD_E_FORMAT,"fixture fingerprint bound");h=hash(h,bytes.data(),size_t(n));}return h;}
void read_at(int fd,void* dst,size_t n,uint64_t offset){auto* p=static_cast<uint8_t*>(dst);while(n){const auto got=::pread(fd,p,n,off_t(offset));if(got<0&&errno==EINTR)continue;need(got>0,RAD_E_IO,"bounded exact physical source read");p+=got;n-=size_t(got);offset+=uint64_t(got);}}
void audit_file_payload(const fs::path& file,uint64_t header_bytes,const std::vector<TensorSpec>& specs){
    Fd fd(::open(file.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,RAD_E_IO,"actual physical payload audit open");
    uint64_t offset=8+header_bytes;std::array<uint8_t,4096> bytes{};
    for(const auto& s:specs){for(uint64_t at=0;at<s.bytes;){const auto n=size_t(std::min<uint64_t>(bytes.size(),s.bytes-at));read_at(fd.n,bytes.data(),n,offset+at);
        for(size_t i=0;i<n;++i)need(bytes[i]==expected_byte(s,at+i),RAD_E_FORMAT,"current physical file payload differs from inspected mapping contract");at+=n;}offset+=s.bytes;}
}
Json configuration(){return {{"model_type","glm53_native_tiny"},{"_name_or_path","SYNTHETIC-STARTUP-METADATA-ONLY"},{"num_hidden_layers",L},{"hidden_size",H},{"intermediate_size",F},{"num_attention_heads",0},{"num_key_value_heads",0},{"head_dim",0},{"num_experts",E},{"num_experts_per_tok",K},{"vocab_size",V},{"max_position_embeddings",T},{"rms_norm_eps",1e-5},{"tie_word_embeddings",false},{"bos_token_id",257},{"eos_token_id",256},{"quantization_config",{{"quant_method","quark"},{"global_quant_config",{{"weight",{{"dtype","fp4"},{"qscheme","per_group"},{"group_size",32},{"scale_format","e8m0"}}}}}}}};}
std::string tokenizer_fixture(){std::vector<std::string> extras;for(int i=0;i<61;++i)extras.push_back("<|tiny_extra_"+std::to_string(i)+"|>");return tiny::tokenizer_json(extras);}
std::string bounded_text(const fs::path& path,uint64_t cap){const auto n=file_size(path,cap);std::string text(size_t(n),' ');Fd fd(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,RAD_E_IO,"bounded sidecar open");read_at(fd.n,text.data(),text.size(),0);return text;}
} // namespace
std::string tokenizer_configuration_json(){return Json{{"bos_token","<|im_start|>"},{"eos_token","<|endoftext|>"},{"add_bos_token",false},{"add_eos_token",false}}.dump();}
SourceCheckpoint::SourceCheckpoint(std::unique_ptr<rad::Checkpoint> ck,std::string root,std::array<Stamp,4> artifacts):checkpoint_(std::move(ck)),root_(std::move(root)),artifacts_(artifacts){}
SourceCheckpoint::~SourceCheckpoint()=default;
rad::Checkpoint& SourceCheckpoint::checkpoint()const noexcept{return *checkpoint_;}
std::array<SourceCheckpoint::Stamp,4> SourceCheckpoint::stamps(const std::string& root){
    std::array<Stamp,4> result{};size_t i=0;
    for(const char* name:{"model.safetensors","config.json","tokenizer.json","tokenizer_config.json"}){struct stat s{};const auto p=fs::path(root)/name;
        need(!::lstat(p.c_str(),&s)&&S_ISREG(s.st_mode)&&s.st_size>=0,RAD_E_FORMAT,"opened source artifact identity");
        result[i++]={uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),uint64_t(s.st_mode),int64_t(s.st_mtim.tv_sec),int64_t(s.st_mtim.tv_nsec),int64_t(s.st_ctim.tv_sec),int64_t(s.st_ctim.tv_nsec)};}
    return result;
}
int SourceCheckpoint::audit(std::string* why)const{
    try{need(checkpoint_&&root_.size()<=480&&checkpoint_->path().size()<=480&&checkpoint_->dir().size()<=480&&checkpoint_->tok_dir().size()<=480&&fs::canonical(root_).string()==root_&&
             fs::canonical(checkpoint_->path()).string()==root_&&fs::canonical(checkpoint_->dir()).string()==root_&&
             fs::canonical(checkpoint_->tok_dir()).string()==root_,RAD_E_STATE,"opened Checkpoint path/dir/tokenizer identity changed");
        const int bounded=preflight_source_files(root_,why);if(bounded<0)return bounded;
        need(stamps(root_)==artifacts_,RAD_E_STATE,"opened source artifact changed or replaced");
        need(checkpoint_->tensors().size()==219&&checkpoint_->imports()==rad::Checkpoint::Import::QuarkMx,RAD_E_STATE,"opened Checkpoint object mutated");return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int open_source_checkpoint(const std::string& directory,std::shared_ptr<const SourceCheckpoint>* out,std::string* why){
    static_assert(sizeof(SourceCheckpoint)<=1024,"source owner metadata must stay within its explicit small request quote");
    try{need(out&&!*out,RAD_E_INVAL,"opened source owner output must be empty");const int bounded=preflight_source_files(directory,why);if(bounded<0)return bounded;
        auto root=fs::canonical(directory).string();need(root.size()<=480,RAD_E_INVAL,"canonical source root bound");const auto artifacts=SourceCheckpoint::stamps(root);
        auto ck=std::make_unique<rad::Checkpoint>();const int status=ck->open(root);if(status<0)return status;
        need(SourceCheckpoint::stamps(root)==artifacts,RAD_E_STATE,"source changed during actual Checkpoint open");
        auto ready=std::shared_ptr<const SourceCheckpoint>(new SourceCheckpoint(std::move(ck),std::move(root),artifacts));
        const int audit=ready->audit(why);if(audit<0)return audit;*out=std::move(ready);return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
std::vector<TensorSpec> tensor_specs(){
    std::vector<TensorSpec> out;out.reserve(219);
    auto add=[&](std::string name,const char* dt,std::initializer_list<int64_t> shape,uint64_t bytes,Role role,int l=-1,int e=-1,int p=-1){out.push_back({std::move(name),dt,shape,bytes,role,l,e,p});};
    for(int l=0;l<L;++l){for(int e=0;e<E;++e)for(int p=0;p<3;++p){const auto base="tiny.layers."+std::to_string(l)+".experts."+std::to_string(e)+"."+(p==0?"gate":p==1?"up":"down")+"_proj.weight";
        add(base,"U8",{F,H/2},8192,Role::ExpertCodes,l,e,p);add(base+"_scale","U8",{F,H/32},512,Role::ExpertScale,l,e,p);}
        const auto base="tiny.layers."+std::to_string(l);add(base+".router.weight","F32",{E,H},2048,Role::Router,l);
        add(base+".router.correction","F32",{E},16,Role::Correction,l);add(base+".ffn_norm.weight","F32",{H},512,Role::FfnNorm,l);
    }
    add("tiny.embed.weight","BF16",{V,H},81920,Role::Embedding);add("tiny.head.weight","BF16",{V,H},81920,Role::Head);add("tiny.final_norm.weight","F32",{H},512,Role::FinalNorm);
    return out;
}
uint8_t expected_byte(const TensorSpec& s,uint64_t at)noexcept{
    const auto word=[&](uint64_t i){uint32_t h=uint32_t(i+1)*0x9e3779b9u^uint32_t(s.layer+3)*0x85ebca6bu^uint32_t(s.expert+5)*0xc2b2ae35u^uint32_t(s.projection+7)*97u;h^=h>>16;return h;};
    if(s.role==Role::ExpertCodes){const auto nibble=[&](uint64_t k){const auto h=word(k);return uint8_t(1+h%7u)|uint8_t(((h>>13)&1u)<<3);};return uint8_t(nibble(2*at)|(nibble(2*at+1)<<4));}
    if(s.role==Role::ExpertScale)return uint8_t(124+word(at)%3);
    const bool bf16=s.dtype=="BF16";const uint64_t i=at/(bf16?2:4);float x=0;
    if(s.role==Role::FfnNorm||s.role==Role::FinalNorm)x=1.f+float(int(i%3)-1)/32.f;
    else if(s.role==Role::Correction)x=float(int(i)-1)/16.f;
    else if(s.role==Role::Router)x=float(int((i*7+uint64_t(s.layer)*3)%17)-8)/64.f;
    else x=float(int((i*13+(s.role==Role::Head?7:3))%31)-15)/64.f;
    if(bf16){const uint16_t bits=rad_f32_to_bf16(x);return uint8_t(bits>>((at%2)*8));}
    const auto bits=std::bit_cast<uint32_t>(x);return uint8_t(bits>>((at%4)*8));
}
int write_fixture(const std::string& destination,FixtureFiles* out,std::string* why){
    try{
        need(out&&out->directory.empty(),RAD_E_INVAL,"fixture output must be empty");
        need(std::endian::native==std::endian::little&&fs::path(destination).is_absolute()&&destination.size()<=480&&destination.find('\0')==std::string::npos,RAD_E_INVAL,"bounded little-endian fixture destination");
        need(category_sum<admitted,RAD_E_SCRATCH,"whole tiny fixed pre-admission");
        const auto specs=tensor_specs();need(specs.size()==219,RAD_E_SHAPE,"exact physical tensor count");
        Json header=Json::object();header["__metadata__"]={{"format","pt"},{"scope","complete synthetic Native startup metadata fixture"}};uint64_t payload=0;
        for(const auto& s:specs){need(s.name.size()<=512&&s.bytes<=131072,RAD_E_SHAPE,"bounded physical tensor");header[s.name]={{"dtype",s.dtype},{"shape",s.shape},{"data_offsets",{payload,payload+s.bytes}}};payload+=s.bytes;}
        need(payload==1020544,RAD_E_SHAPE,"exact checkpoint payload contract");
        auto h=header.dump();while(h.size()%8)h+=' ';need(h.size()<=65536&&8+h.size()+payload<=2*MiB,RAD_E_SCRATCH,"checkpoint header/file quote");
        const auto tokenizer=tokenizer_fixture(),config=configuration().dump(),tokenizer_config=tokenizer_configuration_json();need(tokenizer_config.size()<=4096&&tokenizer.size()+config.size()+tokenizer_config.size()<=131072,RAD_E_SCRATCH,"three sidecar quote");
        FixtureFiles ready{destination,payload,8+h.size()+payload,4,tokenizer.size()+config.size()+tokenizer_config.size()};static_assert(std::is_nothrow_move_assignable_v<FixtureFiles>);
        Temp temp(destination);Fd fd(::open((temp.path+"/model.safetensors").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));need(fd.n>=0,RAD_E_IO,"fixture tensor file open");
        const uint64_t header_bytes=h.size();write_all(fd.n,&header_bytes,8);write_all(fd.n,h.data(),h.size());std::array<uint8_t,4096> bytes{};
        for(const auto& s:specs)for(uint64_t at=0;at<s.bytes;){const auto n=size_t(std::min<uint64_t>(bytes.size(),s.bytes-at));for(size_t i=0;i<n;++i)bytes[i]=expected_byte(s,at+i);write_all(fd.n,bytes.data(),n);at+=n;}
        need(!::fsync(fd.n),RAD_E_IO,"fixture tensor fsync");text_file(fs::path(temp.path)/"config.json",config);text_file(fs::path(temp.path)/"tokenizer.json",tokenizer);text_file(fs::path(temp.path)/"tokenizer_config.json",tokenizer_config);
        Fd dir(::open(temp.path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(dir.n>=0&&!::fsync(dir.n),RAD_E_IO,"fixture directory fsync");
        const int renamed=int(::syscall(SYS_renameat2,AT_FDCWD,temp.path.c_str(),AT_FDCWD,destination.c_str(),1u));need(renamed==0,errno==EEXIST?RAD_E_INVAL:RAD_E_IO,"fixture NOREPLACE publication");temp.committed=true;*out=std::move(ready);return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int preflight_source_files(const std::string& directory,std::string* why){
    try{
        need(std::endian::native==std::endian::little&&fs::path(directory).is_absolute()&&directory.size()<=480&&directory.find('\0')==std::string::npos,RAD_E_INVAL,"bounded fixture source directory");
        struct stat d{};need(!::lstat(directory.c_str(),&d)&&S_ISDIR(d.st_mode),RAD_E_FORMAT,"ordinary fixture directory");
        struct Dir {DIR* value;~Dir(){if(value)::closedir(value);}} listing{::opendir(directory.c_str())};need(listing.value,RAD_E_IO,"bounded source directory listing");
        size_t entries=0;while(const auto* entry=::readdir(listing.value)){const char* n=entry->d_name;if(!std::strcmp(n,".")||!std::strcmp(n,".."))continue;
            need(!std::strcmp(n,"model.safetensors")||!std::strcmp(n,"config.json")||!std::strcmp(n,"tokenizer.json")||!std::strcmp(n,"tokenizer_config.json"),RAD_E_FORMAT,"extra source file outside closed fixture profile");++entries;}
        need(entries==4,RAD_E_FORMAT,"exact closed source directory count");
        const auto path=fs::path(directory)/"model.safetensors";const auto bytes=file_size(path,2*MiB);
        need(file_size(fs::path(directory)/"config.json",131072)+file_size(fs::path(directory)/"tokenizer.json",131072)+file_size(fs::path(directory)/"tokenizer_config.json",4096)<=131072,RAD_E_SCRATCH,"pre-open three-sidecar bound");
        // Close token-ID/count/chain metadata before the real parser allocates it.
        need(Json::parse(bounded_text(fs::path(directory)/"tokenizer.json",131072))==Json::parse(tokenizer_fixture()),RAD_E_FORMAT,"exact bounded synthetic tokenizer envelope");
        const auto tc=Json::parse(bounded_text(fs::path(directory)/"tokenizer_config.json",4096));
        need(tc.is_object()&&tc.size()==4&&tc.contains("bos_token")&&tc.contains("eos_token")&&tc.contains("add_bos_token")&&tc.contains("add_eos_token")&&
             tc.at("bos_token").is_string()&&tc.at("eos_token").is_string()&&tc.at("bos_token").get_ref<const std::string&>().size()<=64&&tc.at("eos_token").get_ref<const std::string&>().size()<=64&&
             tc.at("add_bos_token").is_boolean()&&tc.at("add_eos_token").is_boolean(),RAD_E_FORMAT,"bounded tokenizer sibling schema");
        Fd fd(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,RAD_E_IO,"pre-open checkpoint");uint64_t n=0;
        need(::pread(fd.n,&n,8,0)==8&&n&&n<=65536&&bytes==8+n+1020544,RAD_E_FORMAT,"pre-open header and exact extent");
        std::string h(size_t(n),' ');need(::pread(fd.n,h.data(),h.size(),8)==ssize_t(h.size()),RAD_E_IO,"bounded source header read");
        const auto header=Json::parse(h);need(header.is_object()&&header.size()==220&&header.contains("__metadata__"),RAD_E_FORMAT,"pre-open physical header count");
        uint64_t offset=0;for(const auto& s:tensor_specs()){need(header.contains(s.name),RAD_E_FORMAT,"pre-open physical header name");const auto& a=header.at(s.name);
            need(a.is_object()&&a.at("dtype")==s.dtype&&a.at("shape")==s.shape&&a.at("data_offsets")==Json::array({offset,offset+s.bytes}),RAD_E_FORMAT,"pre-open physical header geometry");offset+=s.bytes;}
        need(offset==1020544,RAD_E_SHAPE,"pre-open payload extent");return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_FORMAT;}
}
int inspect_vocab_contract(const std::string& directory,VocabQuote* out,std::string* why){
    try{need(out,RAD_E_INVAL,"vocabulary output required");const int bounded=preflight_source_files(directory,why);if(bounded<0)return bounded;
        rad::VocabBuild build;need(rad::parse_tokenizer_json((fs::path(directory)/"tokenizer.json").c_str(),build)==RAD_OK,RAD_E_FORMAT,"actual Core tokenizer+sibling parser");rad::Vocab vocab;
        need(vocab.load(std::move(build))==RAD_OK,RAD_E_FORMAT,"actual Core Vocab load");VocabQuote q;q.tokens=uint32_t(vocab.n_tokens());q.bos=vocab.bos();q.eos=vocab.eos();q.eot=vocab.eot();q.eog_count=uint32_t(vocab.eog().size());
        q.add_bos=vocab.add_bos();q.add_eos=vocab.add_eos();q.bos_terminal=vocab.is_eog(257);q.eos_terminal=vocab.is_eog(256);q.im_end_terminal=vocab.is_eog(258);
        need(q.tokens==V&&q.bos==257&&q.eos==256&&q.eot==256&&!q.add_bos&&!q.add_eos&&q.eog_count==2&&!q.bos_terminal&&q.eos_terminal&&q.im_end_terminal&&
             vocab.text(257)=="<|im_start|>"&&vocab.text(256)=="<|endoftext|>"&&vocab.find("<|im_start|>")==257&&vocab.find("<|endoftext|>")==256&&
             vocab.type(256)==RAD_TT_CONTROL&&vocab.type(257)==RAD_TT_CONTROL&&vocab.type(258)==RAD_TT_CONTROL,RAD_E_SHAPE,"actual Core BOS/EOS/EOT/token spelling/automatic-add contract");
        *out=q;return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_FORMAT;}
}
int inspect_checkpoint_source(const std::shared_ptr<const SourceCheckpoint>& owner,const std::string& directory,std::unique_ptr<const InspectedSource>* out,std::string* why){
    try{
        need(out&&!*out&&owner,RAD_E_INVAL,"inspected source output/owner must be valid and empty");
        need(directory.size()<=480&&directory.find('\0')==std::string::npos&&fs::path(directory).is_absolute(),RAD_E_INVAL,"bounded inspected source directory");
        need(fs::canonical(directory).string()==owner->root(),RAD_E_INVAL,"mixed Checkpoint and inspected directory identity");
        const int audit=owner->audit(why);if(audit<0)return audit;
        const int bounded=preflight_source_files(directory,why);if(bounded<0)return bounded;const auto model=fs::path(directory)/"model.safetensors";auto& ck=owner->checkpoint();
        SourceQuote q;q.file_bytes=file_size(model,2*MiB);q.artifact_files=4;q.sidecar_bytes=file_size(fs::path(directory)/"config.json",131072)+file_size(fs::path(directory)/"tokenizer.json",131072)+file_size(fs::path(directory)/"tokenizer_config.json",4096);need(q.sidecar_bytes<=131072,RAD_E_SCRATCH,"source three-sidecar bound");
        q.fingerprint=fingerprint(model,&q.header_bytes);need(q.header_bytes&&q.header_bytes<=65536&&ck.imports()==rad::Checkpoint::Import::QuarkMx,RAD_E_FORMAT,"actual Quark source import");
        const auto specs=tensor_specs();need(ck.tensors().size()==219,RAD_E_SHAPE,"actual physical source count");audit_file_payload(model,q.header_bytes,specs);
        for(const auto& s:specs){const auto* actual=ck.find(s.name);const auto dtype=s.dtype=="U8"?RAD_U8:s.dtype=="F32"?RAD_F32:RAD_BF16;
            need(actual&&actual->rad_dtype==dtype&&actual->shape==s.shape&&actual->bytes==int64_t(s.bytes)&&actual->data,RAD_E_SHAPE,"actual physical tensor name/type/shape/bytes");
            const auto* p=static_cast<const uint8_t*>(actual->data);for(uint64_t i=0;i<s.bytes;++i)need(p[i]==expected_byte(s,i),RAD_E_FORMAT,"actual synthetic physical payload differs");q.payload_bytes+=s.bytes;++q.physical_tensors;
        }
        need(q.payload_bytes==1020544&&q.file_bytes==8+q.header_bytes+q.payload_bytes,RAD_E_SHAPE,"exact source byte extent");
        const int vocabulary=inspect_vocab_contract(directory,&q.vocab,why);if(vocabulary<0)return vocabulary;q.vocabulary=q.vocab.tokens;
        RadBuildCtx ctx{};ctx.world_size=1;ctx.max_tok=T;ctx.max_seqs=1;ctx.max_ctx=T;
        need(admit_declaration_shape(&ck.meta().m,&ctx)==RAD_OK,RAD_E_SHAPE,"actual loaded zero-head synthetic metadata");
        need(fingerprint(model,nullptr)==q.fingerprint,RAD_E_STATE,"source changed during inspection");const int final_audit=owner->audit(why);if(final_audit<0)return final_audit;
        auto ready=std::unique_ptr<const InspectedSource>(new InspectedSource(q,owner));*out=std::move(ready);return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
} // namespace glm53::engine_core_contract

// Manual bounded CPU converter test. No model/kernel launch; synthetic bytes only.
#include "tests/tiny_model.h"
#include "format/checkpoint.h"
#include "format/radfile.h"
#include "format/recipe.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/prctl.h>
using Json=nlohmann::ordered_json;
namespace fs=std::filesystem;
using namespace rad;
namespace {
size_t checks=0,groups=0,converter_calls=0,preflight_refusals=0,hybrid_refusals=0,reuse_predicate_refusals=0;
uint64_t bounded_container_count=0,bounded_container_bytes=0,bounded_plane_bytes=0,bounded_header_reserved=0;
void need(bool ok,const std::string& why){++checks;if(!ok)throw std::runtime_error(why);}
void good(int rc,const std::string& why){need(rc==RAD_OK,why+" status="+std::to_string(rc));}
const std::string prefix="model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
const std::string ng="blk.0.ple_ngram.weight";
const std::string expert="model.language_model.layers.0.mlp.experts.0.";
constexpr std::array<int64_t,3> rows{3,5,7};
constexpr const char* exact_rule="blk.*.ple_ngram.weight rtn codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=0.00019931793212890625\n";
std::vector<uint8_t> finite_codes(){std::vector<uint8_t> out;for(unsigned i=0;i<256;++i)if((i&127)!=127)out.push_back(uint8_t(i));return out;}
std::string shard(int i){return prefix+"shard_"+std::to_string(i)+".weight";}
std::vector<uint8_t> bytes(size_t count,int seed){const auto all=finite_codes();std::vector<uint8_t> out(count);for(size_t i=0;i<count;++i)out[i]=all[(i*37+size_t(seed))%all.size()];return out;}
enum class Variant {good,missing_all,missing_late,wrong_finite,wrong_dtype,wrong_shape,wrong_global,nan_payload};
std::vector<uint8_t> bf16_bytes(size_t n,bool old,int seed){std::vector<uint8_t> data(2*n);
    for(size_t i=0;i<n;++i){const uint16_t bits=uint16_t((old?0x3e80:0xbf80)+((i+size_t(seed))%32));data[2*i]=uint8_t(bits);data[2*i+1]=uint8_t(bits>>8);}return data;}
struct Fixture {fs::path path;std::vector<uint8_t> ngram,gate,up,down,retained;std::array<std::vector<uint8_t>,3> scales,protected_codes;bool hybrid=false,old=false;};
Fixture fixture(const fs::path& path,Variant variant,bool hybrid=false,bool old=false){need(!fs::exists(path)&&fs::create_directory(path),"NEW fixture directory");Fixture f;f.path=path;f.hybrid=hybrid;f.old=old;
    std::vector<tiny::Tensor> ngs,aliases,weights;Json index=Json::object();Json map=Json::object();
    auto add=[&](std::vector<tiny::Tensor>& to,const char* file,tiny::Tensor tensor){map[tensor.name]=file;to.push_back(std::move(tensor));};
    size_t start=0;
    for(int i=0;i<3;++i){auto data=bytes(size_t(rows[size_t(i)])*160,int(start%254));
        // Logical order is a continuous deterministic finite-code stream across
        // unaligned physical shard boundaries, with every254 finite byte covered.
        const auto all=finite_codes();for(size_t j=0;j<data.size();++j)data[j]=all[((start+j)*37+(old?1:0))%254];
        if(variant==Variant::nan_payload&&i==2)data.back()=0x7f;
        f.ngram.insert(f.ngram.end(),data.begin(),data.end());start+=data.size();
        add(ngs,"ngram-source.safetensors",{shard(i),"F8_E4M3",{rows[size_t(i)],160},std::move(data)});
        if(variant!=Variant::missing_all&&!(variant==Variant::missing_late&&i==2)){
            std::vector<uint8_t> scale{0x51,0x39};std::string dt="BF16";std::vector<int64_t> shape{1};
            if(variant==Variant::wrong_finite&&i==2)scale[0]=0x52;
            if(variant==Variant::wrong_dtype&&i==2){dt="F32";scale={0,0,0,0};}
            if(variant==Variant::wrong_shape&&i==2)shape={1,1};
            add(aliases,"scale-aliases.safetensors",{shard(i)+"_scale",dt,shape,scale});}}
    std::vector<tiny::Tensor> parent;add(parent,"parent-scale.safetensors",{prefix+"weight_scale","BF16",{1},{uint8_t(variant==Variant::wrong_global?0x52:0x51),0x39}});
    f.gate=old?bf16_bytes(128*128,true,1):bytes(128*128,1);f.up=old?bf16_bytes(128*128,true,53):bytes(128*128,53);f.down=old?bf16_bytes(128*128,true,107):bytes(128*128,107);
    const char* names[]{"gate_proj.weight","up_proj.weight","down_proj.weight"};const std::vector<uint8_t>* codes[]{&f.gate,&f.up,&f.down};
    for(int i=0;i<3;++i){f.scales[size_t(i)]={uint8_t(0x80+16*i),0x3b};
        add(weights,"expert.safetensors",{expert+names[i],old?"BF16":"F8_E4M3",{128,128},*codes[i]});
        if(!old)add(weights,"expert.safetensors",{expert+names[i]+"_scale_inv","BF16",{1,1},f.scales[size_t(i)]});}
    if(hybrid){const std::string protected_prefix="model.language_model.layers.0.mlp.experts.1.";
        f.retained=bf16_bytes(2*128,old,9);add(weights,"expert.safetensors",{"model.language_model.retained.weight","BF16",{2,128},f.retained});
        for(int i=0;i<3;++i){f.protected_codes[size_t(i)]=old?bf16_bytes(128*128,true,13+i):bytes(128*128,131+17*i);
            add(weights,"expert.safetensors",{protected_prefix+names[i],old?"BF16":"F8_E4M3",{128,128},f.protected_codes[size_t(i)]});
            if(!old)add(weights,"expert.safetensors",{protected_prefix+names[i]+"_scale_inv","BF16",{1,1},{0x90,0x3b}});}}
    need(tiny::write_safetensors((path/"ngram-source.safetensors").string(),ngs)&&tiny::write_safetensors((path/"parent-scale.safetensors").string(),parent)&&
        tiny::write_safetensors((path/"expert.safetensors").string(),weights),"bounded synthetic physical sources");
    if(!aliases.empty())need(tiny::write_safetensors((path/"scale-aliases.safetensors").string(),aliases),"bounded alias sidecar");
    index["weight_map"]=map;
    Json cfg={{"model_type","qfn8bitfixture"},{"hidden_size",128},{"num_hidden_layers",1},{"num_attention_heads",1},{"num_key_value_heads",1},{"head_dim",128},{"intermediate_size",128},{"vocab_size",259},{"max_position_embeddings",16},
        {"quantization_config",{{"quant_method","fp8"},{"activation_scheme","dynamic"},{"weight_block_size",{128,128}}}}};
    if(hybrid)cfg["hybrid_fixture"]=1;
    need(tiny::write_text((path/"model.safetensors.index.json").string(),index.dump())&&tiny::write_text((path/"config.json").string(),cfg.dump())&&
        tiny::write_text((path/"tokenizer.json").string(),tiny::tokenizer_json()),"bounded synthetic metadata/tokenizer");
    uint64_t total=0;for(const auto& file:fs::directory_iterator(path)){need(file.is_regular_file()&&file.file_size()<=262144,"fixture per-file cap");total+=file.file_size();}
    need(total<1u<<20,"fixture aggregate<1MiB");return f;
}
// This is OUR overlay guard, not a stock converter refusal contract. Output
// encoding recipes alone cannot authenticate the input FP8 scale association.
bool alias_preflight(const Fixture& f){Checkpoint ck;if(ck.open(f.path.string())!=RAD_OK)return false;
    if(ck.meta().quant!="fp8"||ck.imports()!=Checkpoint::Import::None||ck.fp8_block_rows()!=128||ck.fp8_block_cols()!=128)return false;
    const auto scalar=[](const SrcTensor* t){return t&&t->rad_dtype==RAD_BF16&&t->shape==std::vector<int64_t>{1}&&t->bytes==2&&
        static_cast<const uint8_t*>(t->data)[0]==0x51&&static_cast<const uint8_t*>(t->data)[1]==0x39;};
    if(!scalar(ck.find(prefix+"weight_scale")))return false;
    for(int i=0;i<3;++i){const auto* t=ck.find(shard(i));if(!t||t->rad_dtype!=RAD_F8E4M3||t->shape!=std::vector<int64_t>{rows[size_t(i)],160}||t->bytes!=rows[size_t(i)]*160||!scalar(ck.find(shard(i)+"_scale")))return false;
        const auto* code=static_cast<const uint8_t*>(t->data);for(int64_t j=0;j<t->bytes;++j)if((code[j]&127)==127)return false;}
    return true;
}
uint64_t fingerprint(const fs::path& directory){std::vector<fs::path> files;for(const auto& f:fs::directory_iterator(directory))files.push_back(f.path());
    need(files.size()<=8,"bounded fixture file count");std::sort(files.begin(),files.end());uint64_t h=14695981039346656037ull,total=0;
    for(const auto& file:files){need(fs::is_regular_file(file)&&fs::file_size(file)<=262144,"bounded immutable source artifact");total+=fs::file_size(file);need(total<1u<<20,"bounded source hash bytes");
        for(const auto c:file.filename().string()){h^=uint8_t(c);h*=1099511628211ull;}
        std::ifstream input(file,std::ios::binary);need(bool(input),"source hash read");std::array<char,4096> block{};
        while(input){input.read(block.data(),block.size());for(std::streamsize i=0;i<input.gcount();++i){h^=uint8_t(block[size_t(i)]);h*=1099511628211ull;}}need(input.eof(),"complete source hash");}
    return h; // Noncryptographic regression witness, not hostile-writer security.
}
struct Scratch {fs::path root,home;Scratch(const std::string& arch,const std::string& kernel,const std::string& quant){char pattern[]="/tmp/qfn8bit-alias-XXXXXX";const auto* p=::mkdtemp(pattern);need(p,"private fixture root");root=p;home=root/"home";
        for(const char* d:{"kernels","architectures","quantizers"})fs::create_directories(home/d);
        fs::copy_file(arch,home/"architectures/alias_arch.so");fs::copy_file(kernel,home/"kernels/import_read.so");fs::copy_file(quant,home/"quantizers/libquant.so");}
    ~Scratch(){std::error_code error;fs::remove_all(root,error);}};
int child(const std::vector<std::string>& arguments,std::string* output,int seconds=60){
    // Reuse the stock test process helper but cap each read and total retained
    // log explicitly; no imported Python/framework/model runtime is involved.
    int pipefd[2];need(::pipe(pipefd)==0,"converter log pipe");std::vector<char*> argv;for(const auto& a:arguments)argv.push_back(const_cast<char*>(a.c_str()));argv.push_back(nullptr);
    const pid_t owner=::getpid();const pid_t pid=::fork();need(pid>=0,"converter fork");
    if(!pid){::close(pipefd[0]);::dup2(pipefd[1],STDOUT_FILENO);::dup2(pipefd[1],STDERR_FILENO);::close(pipefd[1]);
        if(::prctl(PR_SET_PDEATHSIG,SIGKILL)||::getppid()!=owner)::_Exit(126);
        ::setenv("ROCR_VISIBLE_DEVICES","99",1);::setenv("HIP_VISIBLE_DEVICES","99",1);::setenv("CUDA_VISIBLE_DEVICES","-1",1);::setenv("OMP_NUM_THREADS","1",1);::setenv("OMP_THREAD_LIMIT","1",1);::setenv("OMP_WAIT_POLICY","passive",1);
        ::execv(argv[0],argv.data());::_Exit(127);}
    ::close(pipefd[1]);const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);bool eof=false,failed=false,timed_out=false;std::array<char,4096> buffer{};
    while(!eof){if(std::chrono::steady_clock::now()>limit){timed_out=true;break;}pollfd p{pipefd[0],POLLIN|POLLHUP,0};const int ready=::poll(&p,1,100);if(ready<0&&errno==EINTR)continue;if(ready<0){failed=true;break;}if(!ready)continue;
        const auto n=::read(pipefd[0],buffer.data(),buffer.size());if(n<0&&errno==EINTR)continue;if(n<0){failed=true;break;}if(!n){eof=true;break;}
        if(output->size()+size_t(n)>65536){failed=true;break;}output->append(buffer.data(),size_t(n));}
    ::close(pipefd[0]);if(failed||timed_out)::kill(pid,SIGKILL);int status=0;bool killed=failed||timed_out;
    const auto forced_reap_limit=limit+std::chrono::seconds(2);
    for(;;){const auto got=::waitpid(pid,&status,WNOHANG);if(got==pid)break;if(got<0&&errno==EINTR)continue;need(got>=0,"converter waitpid");
        const auto now=std::chrono::steady_clock::now();if(now>limit&&!killed){timed_out=true;killed=true;::kill(pid,SIGKILL);}
        // Manual test-process containment if an OS-stuck child cannot be reaped.
        // PDEATHSIG and parent's independent process-group deadline remain active.
        if(now>forced_reap_limit)::_Exit(125);
        (void)::poll(nullptr,0,10);
    }
    need(!failed,"converter log/read cap");if(timed_out)return 124;
    return WIFEXITED(status)?WEXITSTATUS(status):128;
}
int convert(const Fixture& f,const Scratch& scratch,const std::string& binary,const fs::path& target,const std::string& rule,bool plan=false,bool clear_quant=true,std::string* observed_log=nullptr,const fs::path& reuse={}){
    need(alias_preflight(f),"overlay preflight MUST precede actual converter");need(!fs::exists(target),"new converter output");const auto recipe=target.string()+".recipe";
    need(tiny::write_text(recipe,rule),"tiny recipe");std::vector<std::string> args{binary,f.path.string(),"--arch","qfn8bitfixture","--home",scratch.home.string(),"--recipe",recipe,"-o",target.string()};if(plan)args.push_back("--plan-only");
    if(clear_quant){args.push_back("--set");args.push_back("quantization_config.quant_method=");}
    args.push_back("-v");if(!reuse.empty()){args.push_back("--reuse");args.push_back(reuse.string());}
    ++converter_calls;std::string log;const int rc=child(args,&log);std::cout<<"CONVERTER call="<<converter_calls<<" exit="<<rc<<" logBytes="<<log.size()<<"\n";if(rc)std::cerr<<log;if(observed_log)*observed_log=std::move(log);return rc;
}
void plane(const RadFile& f,const RadFileEntry& e,int k,const std::vector<uint8_t>& wanted){const auto& p=f.plane(e,k);need(p.bytes==wanted.size()&&!std::memcmp(f.data(p),wanted.data(),wanted.size()),"independent exact plane bytes");}
void result(const Fixture& source,const fs::path& path){RadFile f;good(f.open(path.c_str()),"actual output container");need(f.entry_count()==3,"exact three logical output weights");
    need(!*f.str(f.header().quant),"explicit empty-selector override recorded in output descriptor");
    const auto* table=f.find(ng);need(table&&table->n_planes==2,"ngram output two planes");const auto& enc=f.encoding(*table);const auto* scale=rad_enc_plane(&enc,"scale");
    need(enc.plane[0].dtype==RAD_F8E4M3&&scale&&scale->dtype==RAD_BF16&&!scale->block[0]&&!scale->block[1],"one global output scale encoding");plane(f,*table,0,source.ngram);plane(f,*table,1,{0x51,0x39});
    const auto* gu=f.find("fixture.expert.gate_up.weight");const auto* dn=f.find("fixture.expert.down.weight");need(gu&&dn&&gu->n_planes==2&&dn->n_planes==2,"unchanged native FP8 experts");
    std::vector<uint8_t> codes=source.gate;codes.insert(codes.end(),source.up.begin(),source.up.end());auto scales=source.scales[0];scales.insert(scales.end(),source.scales[1].begin(),source.scales[1].end());
    plane(f,*gu,0,codes);plane(f,*gu,1,scales);plane(f,*dn,0,source.down);plane(f,*dn,1,source.scales[2]);
    need(!*f.str(gu->quantizer)&&!*f.str(dn->quantizer),"expert conversion is copy, no requantizer");
}
using Names=std::set<std::string>;
const Names normal_names{"fixture.expert.gate_up.weight","fixture.expert.down.weight"};
const Names retained_names{ng,"fixture.retained.weight","fixture.protected.gate_up.weight","fixture.protected.down.weight"};
std::string hybrid_recipe(bool all_old=false,bool options_drift=false){
    std::string text=options_drift?exact_rule:"blk.*.ple_ngram.weight rtn codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=1.99317932128906e-4\n";
    text+="fixture.protected.*.weight cast dtype=bf16\nfixture.retained.weight cast dtype=bf16\n";
    if(all_old)text+="fixture.expert.*.weight cast dtype=bf16\n";
    return text;
}
Names catalog(const RadFile& file){Names names;for(int64_t i=0;i<file.entry_count();++i)names.insert(file.str(file.entry(i).name));return names;}
// Exact predicate used by current rad-convert (source pin tools/rad_convert.cpp):
// same encoding/quantizer/options/rank/all shapes and expert layer/ID/slot.
bool reusable(const RadFile& old,const RadFileEntry& a,const RadFile& candidate,const RadFileEntry& b){
    if(!rad_enc_equal(&old.encoding(a),&candidate.encoding(b))||std::strcmp(old.str(a.quantizer),candidate.str(b.quantizer))||
       std::strcmp(old.str(a.options),candidate.str(b.options))||a.rank!=b.rank)return false;
    if((a.expert>=0||b.expert>=0)&&(a.layer!=b.layer||a.expert!=b.expert||a.slot!=b.slot))return false;
    for(uint32_t i=0;i<RAD_MAX_RANK;++i)if(a.shape[i]!=b.shape[i])return false;
    return true;
}
bool complete_reuse_preflight(const RadFile& old,const RadFile& candidate,const std::string& rule,const Names& retain){
    if(old.entry_count()!=6||candidate.entry_count()!=6||std::strcmp(old.str(old.header().arch_id),candidate.str(candidate.header().arch_id))||
       *old.str(old.header().quant)||*candidate.str(candidate.header().quant))return false;
    Names complete=retain;for(const auto& name:normal_names)if(!complete.insert(name).second)return false;
    if(complete.size()!=6||catalog(old)!=complete||catalog(candidate)!=complete||retain!=retained_names)return false;
    Recipe recipe;std::string why;if(recipe.add_text(rule,"hybrid-fixture",&why)!=RAD_OK)return false;
    for(const auto& name:complete){const auto* a=old.find(name);const auto* b=candidate.find(name);if(!a||!b)return false;
        const auto* r=recipe.match(name);const std::string quantizer=r?r->quantizer:"",options=r?r->options_text():"";
        if(quantizer!=candidate.str(b->quantizer)||options!=candidate.str(b->options))return false;
        if(retain.contains(name)){if(!reusable(old,*a,candidate,*b))return false;}
        else{const auto& e=candidate.encoding(*b);const auto* s=rad_enc_plane(&e,"scale");
            if(r||e.n_planes!=2||e.plane[0].dtype!=RAD_F8E4M3||!s||s->dtype!=RAD_BF16||s->block[0]!=128||s->block[1]!=128||
               old.encoding(*a).n_planes!=1||old.encoding(*a).plane[0].dtype!=RAD_BF16||a->rank!=b->rank||a->layer!=0||b->layer!=0||a->expert!=0||b->expert!=0||a->slot!=b->slot)return false;
            for(uint32_t i=0;i<RAD_MAX_RANK;++i)if(a->shape[i]!=b->shape[i])return false;}
    }
    return true;
}
int guarded_hybrid_convert(const Fixture& source,const Scratch& scratch,const std::string& binary,const fs::path& output,const std::string& rule,
                           const RadFile& old,const RadFile& proposed,const Names& retain,std::string* log=nullptr){
    if(!complete_reuse_preflight(old,proposed,rule,retain)){++hybrid_refusals;return RAD_E_FORMAT;}
    return convert(source,scratch,binary,output,rule,false,true,log,fs::path(old.path()));
}
void equal_entry_planes(const RadFile& a,const RadFileEntry& ae,const RadFile& b,const RadFileEntry& be){
    need(ae.n_planes==be.n_planes,"all retained plane count");
    for(int k=0;k<ae.n_planes;++k){const auto& ap=a.plane(ae,k);const auto& bp=b.plane(be,k);
        need(ap.bytes==bp.bytes&&!std::memcmp(a.data(ap),b.data(bp),size_t(ap.bytes)),"every retained output plane is exactly old bytes");}
}
Names logged_names(const std::string& log,const std::string& marker,const Names& complete){Names names;std::istringstream lines(log);std::string line;
    while(std::getline(lines,line)){if(line.size()<marker.size()||line.compare(line.size()-marker.size(),marker.size(),marker))continue;
        size_t hits=0;for(const auto& name:complete){const auto at=line.find(name);if(at==std::string::npos)continue;
            const auto end=at+name.size();if(end<line.size()&&line[end]==' '){need(names.insert(name).second,"duplicate verbose per-weight evidence");++hits;}}
        need(hits==1,"exact known per-weight verbose evidence");
    }return names;
}
const RadFileEntry& required(const RadFile& file,const std::string& name){const auto* entry=file.find(name);need(entry,"required complete logical output name: "+name);return *entry;}
void bounded_open(RadFile& file,const fs::path& path){
    constexpr uint64_t blob_cap=256u<<10,plane_cap=200u<<10,container_cap=uint64_t(RAD_ALIGN_POOL)+blob_cap;
    need(fs::is_regular_file(path),"actual output regular file");const auto extent=fs::file_size(path);
    need(extent<=container_cap,"actual bounded output before mapping bytes="+std::to_string(extent)+" cap="+std::to_string(container_cap));
    good(file.open(path.c_str()),"bounded output open");const auto& h=file.header();
    need(h.data_off==RAD_ALIGN_POOL&&h.data_bytes<=blob_cap&&h.file_bytes==h.data_off+h.data_bytes&&h.file_bytes==extent,
         "actual2MiB container metadata/alignment reserve + bounded native blob");
    uint64_t payload=0;for(int64_t i=0;i<file.entry_count();++i){const auto& e=file.entry(i);
        for(int k=0;k<e.n_planes;++k){const auto bytes=file.plane(e,k).bytes;need(bytes<=plane_cap-payload,"closed tiny plane-payload bound");payload+=bytes;}}
    ++bounded_container_count;bounded_container_bytes+=extent;bounded_plane_bytes+=payload;bounded_header_reserved+=h.data_off;
    std::cout<<"CONTAINER bytes="<<extent<<" headerReserved="<<h.data_off<<" blob="<<h.data_bytes<<" planes="<<payload<<"\n";
}
void old_planes(const Fixture& source,const RadFile& old){need(old.entry_count()==6,"old actual six-entry catalog");
    plane(old,required(old,ng),0,source.ngram);plane(old,required(old,ng),1,{0x51,0x39});
    plane(old,required(old,"fixture.retained.weight"),0,source.retained);
    auto normal=source.gate;normal.insert(normal.end(),source.up.begin(),source.up.end());
    plane(old,required(old,"fixture.expert.gate_up.weight"),0,normal);plane(old,required(old,"fixture.expert.down.weight"),0,source.down);
    auto protect=source.protected_codes[0];protect.insert(protect.end(),source.protected_codes[1].begin(),source.protected_codes[1].end());
    plane(old,required(old,"fixture.protected.gate_up.weight"),0,protect);plane(old,required(old,"fixture.protected.down.weight"),0,source.protected_codes[2]);
}
void hybrid_planes(const Fixture& source,const RadFile& old,const RadFile& output){need(output.entry_count()==6,"hybrid complete output catalog");
    for(const auto& name:retained_names){const auto* a=old.find(name);const auto* b=output.find(name);need(a&&b&&reusable(old,*a,output,*b),"retained output metadata exactly old");equal_entry_planes(old,*a,output,*b);}
    auto gu=source.gate;gu.insert(gu.end(),source.up.begin(),source.up.end());auto scale=source.scales[0];scale.insert(scale.end(),source.scales[1].begin(),source.scales[1].end());
    const auto* g=output.find("fixture.expert.gate_up.weight");const auto* d=output.find("fixture.expert.down.weight");need(g&&d&&g->n_planes==2&&d->n_planes==2,"two native output expert planes");
    plane(output,*g,0,gu);plane(output,*g,1,scale);plane(output,*d,0,source.down);plane(output,*d,1,source.scales[2]);
    need(!*output.str(g->quantizer)&&!*output.str(d->quantizer),"new expert no numerical quantizer");
}
void predicate_guards(const RadFile& old,const RadFile& candidate){
    const auto& a=required(old,"fixture.protected.gate_up.weight");const auto& b=required(candidate,"fixture.protected.gate_up.weight");
    const auto& changed=required(candidate,"fixture.expert.gate_up.weight");need(reusable(old,a,candidate,b),"positive protected actual reuse metadata");
    for(int which=0;which<9;++which){auto altered=b;
        switch(which){case 0:++altered.rank;break;case 1:++altered.shape[0];break;case 2:++altered.shape[RAD_MAX_RANK-1];break;
            case 3:altered.quantizer=changed.quantizer;break;case 4:altered.options=changed.options;break;case 5:altered.enc=changed.enc;break;
            case 6:++altered.layer;break;case 7:++altered.expert;break;case 8:++altered.slot;break;}
        need(!reusable(old,a,candidate,altered),"copied actual metadata mutation refused by exact converter predicate index="+std::to_string(which));++reuse_predicate_refusals;
    }
    need(reusable(old,a,candidate,b),"read-only actual metadata remains usable after guard probes");
}
}
int main(int argc,char** argv){try{
    if(argc==2&&!std::strcmp(argv[1],"--internal-close-stdio-wait")){::close(STDOUT_FILENO);::close(STDERR_FILENO);for(;;)::pause();}
    need(argc==6&&!std::strcmp(argv[1],"--run-host"),"usage: --run-host ABS_RAD_CONVERT ABS_ARCH_MODULE ABS_IMPORT_READ_MODULE ABS_LIBQUANT");
    for(int i=2;i<6;++i)need(fs::path(argv[i]).is_absolute()&&fs::is_regular_file(argv[i]),"bounded actual binary/module path");
    Scratch scratch(argv[3],argv[4],argv[5]);++groups;
    const auto f=fixture(scratch.root/"good",Variant::good);need(alias_preflight(f),"positive authenticated aliases");const auto input_hash=fingerprint(f.path);
    const auto plan=scratch.root/"plan.rad";need(convert(f,scratch,argv[2],plan,exact_rule,true)==0&&!fs::exists(plan),"actual converter plan-only");
    const auto a=scratch.root/"exact.rad";need(convert(f,scratch,argv[2],a,exact_rule)==0,"actual fixed-scale conversion");result(f,a);
    const auto b=scratch.root/"legacy.rad";need(convert(f,scratch,argv[2],b,"blk.*.ple_ngram.weight rtn codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=1.99317932128906e-4\n")==0,"legacy recipe spelling");result(f,b);
    ++groups;size_t i=0;for(const auto variant:{Variant::missing_all,Variant::missing_late,Variant::wrong_finite,Variant::wrong_dtype,Variant::wrong_shape,Variant::wrong_global,Variant::nan_payload}){
        const auto bad=fixture(scratch.root/("bad"+std::to_string(i++)),variant);const auto calls=converter_calls;need(!alias_preflight(bad),"OUR preflight refuses missing/wrong scale or nonfinite late code");++preflight_refusals;
        need(converter_calls==calls&&!fs::exists(bad.path/"never.rad"),"refusal does not invoke converter or publish output");}
    ++groups;const auto c=scratch.root/"no-global-rule.rad";need(convert(f,scratch,argv[2],c,"")!=0&&!fs::exists(c),"actual converter-only architecture refuses per-row PLE output");
    const auto d=scratch.root/"invalid-rule.rad";need(convert(f,scratch,argv[2],d,"blk.*.ple_ngram.weight rtn codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=0\n")!=0&&!fs::exists(d),"actual invalid quantizer rule leaves no completed output");
    ++groups;const auto selector=scratch.root/"selector-refused.rad";std::string selector_log;
    need(convert(f,scratch,argv[2],selector,exact_rule,false,false,&selector_log)!=0&&!fs::exists(selector),"official fp8 descriptor cannot select empty-quant fixture architecture");
    need(selector_log.find("no architecture plugin claims 'qfn8bitfixture' at quantisation 'fp8'")!=std::string::npos,"intended exact plugin-selector refusal witness");
    ++groups;std::string closed_log;
    need(child({fs::absolute(argv[0]).string(),"--internal-close-stdio-wait"},&closed_log,1)==124&&closed_log.empty(),"EOF child is still reaped under deadline");
    ++groups;const auto old_source=fixture(scratch.root/"hybrid-old",Variant::good,true,true);
    const auto new_source=fixture(scratch.root/"hybrid-new",Variant::good,true,false);
    const auto old_hash=fingerprint(old_source.path),new_hash=fingerprint(new_source.path);
    need(old_source.retained!=new_source.retained&&old_source.ngram!=new_source.ngram,"retained old/source bytes deliberately differ");
    const auto old_path=scratch.root/"hybrid-old.rad",candidate_path=scratch.root/"hybrid-candidate.rad";
    need(convert(old_source,scratch,argv[2],old_path,hybrid_recipe(true))==0,"actual converter builds old BF16 protected/reference artifact");
    need(convert(new_source,scratch,argv[2],candidate_path,hybrid_recipe())==0,"actual converter candidate supplies planned metadata before reuse");
    RadFile old,candidate;bounded_open(old,old_path);bounded_open(candidate,candidate_path);old_planes(old_source,old);
    need(complete_reuse_preflight(old,candidate,hybrid_recipe(),retained_names),"complete retained-set exact converter reuse predicate");
    plane(candidate,required(candidate,"fixture.retained.weight"),0,new_source.retained);
    for(const auto& name:{"fixture.protected.gate_up.weight","fixture.protected.down.weight"}){const auto* a=old.find(name);const auto* b=candidate.find(name);
        need(a&&b&&a->n_planes==1&&b->n_planes==1&&old.plane(*a,0).bytes==candidate.plane(*b,0).bytes&&
             std::memcmp(old.data(old.plane(*a,0)),candidate.data(candidate.plane(*b,0)),size_t(old.plane(*a,0).bytes)),"protected old BF16 differs from current FP8-dequantized source candidate");}
    ++groups;const auto drift_path=scratch.root/"hybrid-option-drift.rad";
    need(convert(new_source,scratch,argv[2],drift_path,hybrid_recipe(false,true))==0,"actual options-drift probe output");
    RadFile drift;bounded_open(drift,drift_path);const auto* old_ng=old.find(ng);const auto* drift_ng=drift.find(ng);
    need(old_ng&&drift_ng&&rad_enc_equal(&old.encoding(*old_ng),&drift.encoding(*drift_ng))&&
         std::strcmp(old.str(old_ng->options),drift.str(drift_ng->options)),"equivalent scalar encoding still has different recorded option text");
    auto refuse=[&](const RadFile& proposed,const std::string& rule,const Names& retain){const auto before=converter_calls;const auto absent=scratch.root/"never-hybrid.rad";
        need(guarded_hybrid_convert(new_source,scratch,argv[2],absent,rule,old,proposed,retain)==RAD_E_FORMAT,"OUR strict retained-set preflight refuses before reuse conversion");
        need(converter_calls==before&&!fs::exists(absent),"hybrid refusal leaves converter-call count and output unchanged");};
    refuse(drift,hybrid_recipe(false,true),retained_names);
    auto incomplete=retained_names;incomplete.erase("fixture.retained.weight");refuse(candidate,hybrid_recipe(),incomplete);
    auto overlap=retained_names;overlap.erase("fixture.retained.weight");overlap.insert("fixture.expert.down.weight");refuse(candidate,hybrid_recipe(),overlap);
    predicate_guards(old,candidate);
    need(complete_reuse_preflight(old,candidate,hybrid_recipe(),retained_names),"good preflight restored after refused attempts");
    ++groups;const auto hybrid_path=scratch.root/"hybrid-ready.rad";std::string hybrid_log;
    need(guarded_hybrid_convert(new_source,scratch,argv[2],hybrid_path,hybrid_recipe(),old,candidate,retained_names,&hybrid_log)==0,"actual converter hybrid reuse");
    RadFile hybrid;bounded_open(hybrid,hybrid_path);need(complete_reuse_preflight(old,hybrid,hybrid_recipe(),retained_names),"every final logical encoding/identity/shape verified");hybrid_planes(new_source,old,hybrid);
    const auto complete=catalog(hybrid);need(logged_names(hybrid_log," reused",complete)==retained_names,"EXACT per-weight retained reuse evidence");
    need(logged_names(hybrid_log," as is",complete)==normal_names,"EXACT new expert raw-copy evidence");
    need(fingerprint(old_source.path)==old_hash&&fingerprint(new_source.path)==new_hash,"both distinct source checkpoint generations immutable");
    need(fingerprint(f.path)==input_hash,"all converter input artifacts unchanged");
    need(groups==8&&preflight_refusals==7&&hybrid_refusals==3&&reuse_predicate_refusals==9&&converter_calls==10&&bounded_container_count==4,"honest exact matrix counts");std::cout<<"PASS synthetic CPU converter groups="<<groups<<" checks="<<checks<<" calls="<<converter_calls<<" overlayRefusals="<<preflight_refusals<<" hybridRefusals="<<hybrid_refusals<<" predicateRefusals="<<reuse_predicate_refusals<<" retainedEntries=4 nativeExpertEntries=2 retainedPlanes=5 newPlanes=4 finiteFP8Codes=254 payload=2400B boundedContainerBytes="<<bounded_container_bytes<<" boundedPlaneBytes="<<bounded_plane_bytes<<" boundedHeaderReserved="<<bounded_header_reserved<<" no inference\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" checks="<<checks<<" converterCalls="<<converter_calls<<"\n";return 1;}}

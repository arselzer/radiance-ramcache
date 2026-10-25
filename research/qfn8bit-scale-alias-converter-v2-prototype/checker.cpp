// Manual bounded CPU converter test. No model/kernel launch; synthetic bytes only.
#include "tests/tiny_model.h"
#include "format/checkpoint.h"
#include "format/radfile.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sys/prctl.h>
using Json=nlohmann::ordered_json;
namespace fs=std::filesystem;
using namespace rad;
namespace {
size_t checks=0,groups=0,converter_calls=0,preflight_refusals=0;
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
struct Fixture {fs::path path;std::vector<uint8_t> ngram,gate,up,down;std::array<std::vector<uint8_t>,3> scales;};
Fixture fixture(const fs::path& path,Variant variant){need(!fs::exists(path)&&fs::create_directory(path),"NEW fixture directory");Fixture f;f.path=path;
    std::vector<tiny::Tensor> ngs,aliases,weights;Json index=Json::object();Json map=Json::object();
    auto add=[&](std::vector<tiny::Tensor>& to,const char* file,tiny::Tensor tensor){map[tensor.name]=file;to.push_back(std::move(tensor));};
    size_t start=0;
    for(int i=0;i<3;++i){auto data=bytes(size_t(rows[size_t(i)])*160,int(start%254));
        // Logical order is a continuous deterministic finite-code stream across
        // unaligned physical shard boundaries, with every254 finite byte covered.
        const auto all=finite_codes();for(size_t j=0;j<data.size();++j)data[j]=all[((start+j)*37)%254];
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
    f.gate=bytes(128*128,1);f.up=bytes(128*128,53);f.down=bytes(128*128,107);
    const char* names[]{"gate_proj.weight","up_proj.weight","down_proj.weight"};const std::vector<uint8_t>* codes[]{&f.gate,&f.up,&f.down};
    for(int i=0;i<3;++i){f.scales[size_t(i)]={uint8_t(0x80+16*i),0x3b};
        add(weights,"expert.safetensors",{expert+names[i],"F8_E4M3",{128,128},*codes[i]});
        add(weights,"expert.safetensors",{expert+names[i]+"_scale_inv","BF16",{1,1},f.scales[size_t(i)]});}
    need(tiny::write_safetensors((path/"ngram-source.safetensors").string(),ngs)&&tiny::write_safetensors((path/"parent-scale.safetensors").string(),parent)&&
        tiny::write_safetensors((path/"expert.safetensors").string(),weights),"bounded synthetic physical sources");
    if(!aliases.empty())need(tiny::write_safetensors((path/"scale-aliases.safetensors").string(),aliases),"bounded alias sidecar");
    index["weight_map"]=map;
    Json cfg={{"model_type","qfn8bitfixture"},{"hidden_size",128},{"num_hidden_layers",1},{"num_attention_heads",1},{"num_key_value_heads",1},{"head_dim",128},{"intermediate_size",128},{"vocab_size",259},{"max_position_embeddings",16},
        {"quantization_config",{{"quant_method","fp8"},{"activation_scheme","dynamic"},{"weight_block_size",{128,128}}}}};
    need(tiny::write_text((path/"model.safetensors.index.json").string(),index.dump())&&tiny::write_text((path/"config.json").string(),cfg.dump())&&
        tiny::write_text((path/"tokenizer.json").string(),tiny::tokenizer_json()),"bounded synthetic metadata/tokenizer");
    uint64_t total=0;for(const auto& file:fs::directory_iterator(path)){need(file.is_regular_file()&&file.file_size()<=131072,"fixture per-file cap");total+=file.file_size();}
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
    for(const auto& file:files){need(fs::is_regular_file(file)&&fs::file_size(file)<=131072,"bounded immutable source artifact");total+=fs::file_size(file);need(total<1u<<20,"bounded source hash bytes");
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
int convert(const Fixture& f,const Scratch& scratch,const std::string& binary,const fs::path& target,const std::string& rule,bool plan=false,bool clear_quant=true,std::string* observed_log=nullptr){
    need(alias_preflight(f),"overlay preflight MUST precede actual converter");need(!fs::exists(target),"new converter output");const auto recipe=target.string()+".recipe";
    need(tiny::write_text(recipe,rule),"tiny recipe");std::vector<std::string> args{binary,f.path.string(),"--arch","qfn8bitfixture","--home",scratch.home.string(),"--recipe",recipe,"-o",target.string()};if(plan)args.push_back("--plan-only");
    if(clear_quant){args.push_back("--set");args.push_back("quantization_config.quant_method=");}
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
    need(fingerprint(f.path)==input_hash,"all converter input artifacts unchanged");
    need(groups==5&&preflight_refusals==7&&converter_calls==6,"honest exact matrix counts");std::cout<<"PASS synthetic CPU converter groups="<<groups<<" checks="<<checks<<" calls="<<converter_calls<<" overlayRefusals="<<preflight_refusals<<" finiteFP8Codes=254 payload=2400B no inference\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" checks="<<checks<<" converterCalls="<<converter_calls<<"\n";return 1;}}

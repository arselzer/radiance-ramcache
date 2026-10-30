/* Actual HOST Mover/Planner integration over private synthetic file bytes.
 * No model Source capability, kernel arithmetic, GPU, server or external asset.
 * Optional genuine tiny-Engine pre-Pools case needs the matching whole-Core
 * export/link setup in the accompanying prototype's enable_engine_case.cmake. */
#include "rad_test.h"
#include "device/device.h"
#include "device/directio.h"
#include "place/mover.h"
#include "place/planner.h"
#include "tiny_model.h"
#ifdef RAD_NATIVE_CACHE_ENGINE_CASE
#include "engine.h"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
namespace {
constexpr int64_t MiB=1ll<<20, KiB=1ll<<10;
uint8_t source_byte(uint64_t i) {
    uint64_t z=i+0x9e3779b97f4a7c15ull;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
    z=(z^(z>>27))*0x94d049bb133111ebull;
    return uint8_t((z^(z>>31))>>56);
}
void need(int code,const char* what) {
    if(code<0)throw std::runtime_error(std::string(what)+": "+rad_strerror(code));
}
void exact_write(int fd,const uint8_t* p,size_t n) {
    while(n){const ssize_t k=::write(fd,p,n);if(k<0&&errno==EINTR)continue;
        if(k<=0)throw std::runtime_error("synthetic file write");p+=k;n-=size_t(k);}
}
struct Image {
    std::string path;int fd=-1;std::vector<uint8_t> bytes;
    explicit Image(int64_t size):bytes(size_t(size)) {
        for(size_t i=0;i<bytes.size();++i)bytes[i]=source_byte(i);
        std::string pattern=radtest::tmpdir()+"/cache-mover-XXXXXX";
        fd=::mkstemp(pattern.data());if(fd<0)throw std::runtime_error("mkstemp on O_DIRECT-capable disk");
        path=pattern;
        try{exact_write(fd,bytes.data(),bytes.size());if(::fsync(fd))throw std::runtime_error("fsync synthetic file");}
        catch(...){::close(fd);fd=-1;::unlink(path.c_str());throw;}
    }
    ~Image(){if(fd>=0)::close(fd);if(!path.empty())::unlink(path.c_str());}
    Image(const Image&)=delete;Image& operator=(const Image&)=delete;
};
/* Identical role as place_test's fake_dev/reversing_relayout fixture: a metadata
 * row only, not a fabricated real-model row or a launchable neural kernel. */
int relayout(const RadParam*,int,int,const RadEncoding*,const int*,const RadTensor* p,int n,void* dst,int64_t cap) {
    if(n!=2||!p||!dst||!p[0].data||!p[1].data)return RAD_E_INVAL;
    const int64_t a=p[0].shape[0]*p[0].shape[1],b=p[1].shape[0]*p[1].shape[1];
    if(a<=0||b<=0||a>INT64_MAX-b||a+b!=cap)return RAD_E_SHAPE;
    auto* d=static_cast<uint8_t*>(dst);const auto* x=static_cast<const uint8_t*>(p[0].data);
    for(int64_t i=0;i<a;++i)d[i]=uint8_t(x[a-1-i]^0x5a);
    std::memcpy(d+a,p[1].data,size_t(b));return RAD_OK;
}
struct Fixture {
    Program prog;RadKernelInfo info{};KernelRow row;
    Fixture(){info.name="synthetic-cache-relayout";info.op="moe_gemm";info.domain=RAD_DOMAIN_DEVICE;
        info.relayout=relayout;row.info=&info;prog.ctx.rank=0;prog.ctx.world_size=1;prog.weights.resize(1);}
    rad_weight weight(std::string name,int64_t n,int access,int layer,int expert) {
        WeightInfo w;w.name=std::move(name);w.decl.dtype=RAD_U8;w.decl.rank=1;w.decl.shape[0]=n;
        w.decl.access=access;w.decl.shard=RAD_SHARD_NONE;w.decl.group.layer=layer;w.decl.group.expert=expert;
        w.logical_bytes=w.stored_bytes=n;w.align=RAD_ALIGN_UNIT;
        prog.weights.push_back(std::move(w));return rad_weight(prog.weights.size()-1);
    }
    void op(const char* name,const std::vector<rad_weight>& ws) {
        OpInfo o;o.op=name;o.index=int32_t(prog.ops.size());o.weights=ws;Band b;b.hi=1<<20;
        b.dom[RAD_DOMAIN_DEVICE].row=&row;b.miss[RAD_DOMAIN_HOST]="fixture forces staged device-site bytes";
        o.bands.push_back(b);
        for(auto h:ws){auto& w=prog.weights[h];if(w.first_use_op<0)w.first_use_op=o.index;w.last_use_op=o.index;}
        prog.ops.push_back(std::move(o));
    }
    void experts() {
        const auto embed=weight("token_embd.weight",MiB,RAD_ACCESS_VOCAB,-1,-1);op("embed",{embed});
        for(int l=0;l<3;++l){std::vector<rad_weight> g,d;
            for(int e=0;e<2;++e){const auto stem="blk."+std::to_string(l)+".expert."+std::to_string(e);
                g.push_back(weight(stem+".gu",768*KiB+17,RAD_ACCESS_CONDITIONAL,l,e));
                d.push_back(weight(stem+".dn",256*KiB-17,RAD_ACCESS_CONDITIONAL,l,e));}
            op("moe_gemm",g);op("moe_gemm",d);}
    }
    void dense() {
        const auto embed=weight("token_embd.weight",MiB,RAD_ACCESS_VOCAB,-1,-1);op("embed",{embed});
        for(int l=0;l<3;++l){const auto h=weight("blk."+std::to_string(l)+".large",33*MiB,
            RAD_ACCESS_PER_TOKEN,l,-1);op("gemm",{h});}
    }
    int64_t lay_out(bool make_relaid) {
        int64_t off=4*int64_t(RAD_ALIGN_UNIT);
        for(size_t h=1;h<prog.weights.size();++h){auto& w=prog.weights[h];w.file_offset=uint64_t(off);
            if(make_relaid&&w.decl.group.expert>=0){
                constexpr int64_t tail=137;const int64_t first=w.stored_bytes-tail;
                w.identity=false;w.layout_tag="synthetic.reversed";w.lay_op=w.first_use_op;
                w.lay_band=0;w.lay_dom=RAD_DOMAIN_DEVICE;w.lay_operand=0;
                w.enc.n_planes=2;w.enc.plane[0].dtype=w.enc.plane[1].dtype=RAD_U8;
                w.n_sel=2;w.sel[0]=0;w.sel[1]=1;
                w.sel_rows[0]=w.sel_rows[1]=1;w.sel_cols[0]=first;w.sel_cols[1]=tail;
                w.file_plane_offset[0]=uint64_t(off);
                w.file_plane_offset[1]=uint64_t(align_up(off+first+257,256));
                off=align_up(int64_t(w.file_plane_offset[1])+tail,RAD_ALIGN_UNIT);
            }else off=align_up(off+w.stored_bytes,RAD_ALIGN_UNIT);
        }
        return off+RAD_ALIGN_UNIT;
    }
};
Config plan_config(bool dense) {
    Config c;c.placement=dense?"layer_offload":"expert_tiered";c.deterministic=true;
    c.vram_weights_mib=dense?70:6;c.host_pool_mib=133;c.weights_disk_tier=true;return c;
}
Plan planned(Fixture& f,const Config& c,bool relaid) {
    PlannerInput in{&f.prog,&c,0,1,nullptr,0};in.file_relayout=relaid;
    Plan p;need(Planner(in).plan(&p),"actual Planner");return p;
}
std::vector<int32_t> two_ssd_experts(const Plan& p) {
    for(int l=0;l<3;++l){std::vector<int32_t> out;
        for(size_t u=0;u<p.units.size();++u)if(p.units[u].layer==l&&p.units[u].expert>=0&&p.units[u].tier==Tier::SSD)out.push_back(int32_t(u));
        if(out.size()==2)return out;}
    throw std::runtime_error("actual Plan did not leave a complete two-expert SSD layer");
}
/* The real pread counts are observed process-wide, not intercepted. /proc/self/io
 * itself consumes one read and its returned byte count; remove ONLY that known
 * observation from the A->B delta. No physical NVMe/cache-miss claim is made.
 * Each case is serial: no unrelated test/child runs between its observations. */
struct Io {
    uint64_t chars=0,calls=0,observation_chars=0;
    static Io capture(){
        const int fd=::open("/proc/self/io",O_RDONLY|O_CLOEXEC);if(fd<0)throw std::runtime_error("/proc/self/io required");
        std::array<char,4096> b{};ssize_t n;
        do{n=::read(fd,b.data(),b.size()-1);}while(n<0&&errno==EINTR);
        const int close_rc=::close(fd);if(n<=0||n>=ssize_t(b.size()-1)||close_rc)throw std::runtime_error("bounded /proc/self/io read");
        std::istringstream in(std::string(b.data(),size_t(n)));std::string key;uint64_t value;Io x;bool rc=false,sc=false;
        while(in>>key>>value){if(key=="rchar:"){x.chars=value;rc=true;}else if(key=="syscr:"){x.calls=value;sc=true;}}
        if(!rc||!sc)throw std::runtime_error("/proc/self/io counters missing");x.observation_chars=uint64_t(n);return x;
    }
};
struct ReadDelta{uint64_t bytes=0,calls=0;};
ReadDelta delta(const Io& a,const Io& b){
    if(b.chars<a.chars+a.observation_chars||b.calls<a.calls+1)throw std::runtime_error("nonmonotone process read counters");
    return {b.chars-a.chars-a.observation_chars,b.calls-a.calls-1};
}
struct Transport {
    RadStream compute=nullptr;RadEvent done=nullptr;std::unique_ptr<Mover> mover;
    void open(Fixture& f,const Plan& p,const Image& image,int64_t cache,std::atomic<uint64_t>& calls){
        need(rad_stream_create(&compute,0),"HOST compute stream");need(rad_event_create(&done),"HOST done event");
        mover=std::make_unique<Mover>();mover->set_file_relayout([&f,&calls](const WeightInfo& w,const RadTensor* ps,void* dst,int64_t n){
            calls.fetch_add(1,std::memory_order_relaxed);const auto* k=weight_layout_kernel(f.prog,w);
            return k&&k->relayout?k->relayout(nullptr,0,w.lay_operand,&w.enc,w.sel,ps,w.n_sel,dst,n):RAD_E_STATE;});
        MoverConfig c;c.container_path=image.path;c.native_file_cache_bytes=cache;
        need(mover->init(f.prog,p,c),"actual HOST Mover init");need(mover->bind_compute_stream(compute),"bind HOST compute");
    }
    void drain(){if(mover&&mover->transfer_stream())need(rad_stream_sync(mover->transfer_stream()),"real HOST transfer drain");
        if(compute)need(rad_stream_sync(compute),"real HOST compute drain");}
    ~Transport(){
        if(mover&&mover->transfer_stream()&&rad_stream_sync(mover->transfer_stream())<0)std::abort();
        if(compute&&rad_stream_sync(compute)<0)std::abort();mover.reset();
        if(done)rad_event_destroy(done);if(compute)rad_stream_destroy(compute);
    }
};
std::vector<uint8_t> expected(const WeightInfo& w,const Image& image){
    std::vector<uint8_t> out(size_t(w.stored_bytes));
    if(w.identity)std::memcpy(out.data(),image.bytes.data()+w.file_offset,out.size());
    else{const int64_t first=w.sel_cols[0],tail=w.sel_cols[1];
        for(int64_t i=0;i<first;++i)out[size_t(i)]=uint8_t(image.bytes[size_t(w.file_plane_offset[0]+first-1-i)]^0x5a);
        std::memcpy(out.data()+first,image.bytes.data()+w.file_plane_offset[1],size_t(tail));}
    return out;
}
void check_published(Transport& t,const Fixture& f,const Image& image,rad_weight h){
    const auto* s=t.mover->residency()->lookup(h);REQUIRE(s&&s->ptr);
    if(s->ready)need(rad_event_wait(t.compute,s->ready),"compute follows actual ready event");
    const auto want=expected(f.prog.weights[h],image);std::vector<uint8_t> got(want.size());
    need(rad_memcpy_async(got.data(),s->ptr,int64_t(got.size()),t.compute),"actual HOST readback copy");
    need(rad_stream_sync(t.compute),"readback complete");CHECK(got==want);
}
void routed_pass(Transport& t,const Fixture& f,const Plan& p,const Image& image,const std::vector<int32_t>& units){
    std::vector<int32_t> staged;need(t.mover->stage_file_units(units,t.mover->file_buffer(0),t.mover->file_buffer_bytes(),nullptr,t.done,&staged),"actual routed file stage");
    REQUIRE(staged==units);need(rad_event_wait(t.compute,t.done),"real done->compute dependency");
    for(auto u:units)for(auto h:p.units[size_t(u)].weights)check_published(t,f,image,h);
    t.drain();t.mover->unstage_units(staged);
    for(auto u:units)for(auto h:p.units[size_t(u)].weights){const auto* s=t.mover->residency()->lookup(h);CHECK(s&&s->ptr==nullptr);}
}
void routed_case(bool cached,bool relaid,bool mutate){
    Fixture f;f.experts();Image image(f.lay_out(relaid));const auto config=plan_config(false);const auto p=planned(f,config,relaid);
    const auto units=two_ssd_experts(p);std::atomic<uint64_t> calls{0};Transport t;t.open(f,p,image,cached?8*MiB:0,calls);
    const auto a=Io::capture();routed_pass(t,f,p,image,units);const auto b=Io::capture();const auto cold=delta(a,b);
    CHECK(cold.calls>=4);CHECK(cold.bytes>0);CHECK_EQ(calls.load(),relaid?uint64_t(4):uint64_t(0));
    const auto before_relaid=t.mover->stats().file_relaid;
    const auto c=Io::capture();routed_pass(t,f,p,image,units);const auto d=Io::capture();const auto warm=delta(c,d);
    if(cached){CHECK_EQ(warm.calls,uint64_t(0));CHECK_EQ(warm.bytes,uint64_t(0));CHECK_EQ(calls.load(),relaid?uint64_t(4):uint64_t(0));CHECK_EQ(t.mover->stats().file_relaid,before_relaid);}
    else{CHECK(warm.calls>=4);CHECK(warm.bytes>0);CHECK_EQ(calls.load(),relaid?uint64_t(8):uint64_t(0));}
    CHECK_EQ(t.mover->stats().file_units,uint64_t(4));
    std::printf("  actual HOST Mover cached=%d relaid=%d cold logical reads=%llu/%lluB warm=%llu/%lluB callbacks=%llu\n",int(cached),int(relaid),
        (unsigned long long)cold.calls,(unsigned long long)cold.bytes,(unsigned long long)warm.calls,(unsigned long long)warm.bytes,(unsigned long long)calls.load());
    if(mutate){
        REQUIRE(cached);const auto before=t.mover->stats();const auto callback_before=calls.load();
        std::vector<WeightSlot> slots;for(auto u:units)for(auto h:p.units[size_t(u)].weights)slots.push_back(*t.mover->residency()->lookup(h));
        REQUIRE_EQ(::ftruncate(image.fd,off_t(image.bytes.size()+1)),0);REQUIRE_EQ(::fsync(image.fd),0);
        std::vector<int32_t> staged;CHECK_EQ(t.mover->stage_file_units(units,t.mover->file_buffer(0),t.mover->file_buffer_bytes(),nullptr,t.done,&staged),RAD_E_STATE);
        t.drain();CHECK_EQ(calls.load(),callback_before);CHECK_EQ(t.mover->stats().h2d_bytes,before.h2d_bytes);CHECK_EQ(t.mover->stats().file_units,before.file_units);
        size_t i=0;for(auto u:units)for(auto h:p.units[size_t(u)].weights){const auto* s=t.mover->residency()->lookup(h);CHECK(s&&s->ptr==slots[i].ptr&&s->ready==slots[i].ready&&s->generation==slots[i].generation);++i;}
    }
}
} // namespace

TEST(native_cache_real_host_relayout_cold_warm_and_epoch_refusal){routed_case(true,true,true);}
TEST(native_cache_real_host_scalar_default_rereads_and_relayouts){routed_case(false,true,false);}
TEST(native_cache_real_host_identity_cold_warm){routed_case(true,false,false);}

TEST(native_cache_real_host_large_identity_demand_keeps_original_bypass){
    Fixture f;f.dense();Image image(f.lay_out(false));const auto config=plan_config(true);const auto p=planned(f,config,false);
    REQUIRE_EQ(p.prefetch.size(),size_t(3));for(const auto& pf:p.prefetch){const auto& u=p.units[size_t(pf.unit)];REQUIRE(u.tier==Tier::SSD);REQUIRE(u.bytes>RAD_PLACE_FILE_WINDOW_BYTES);}
    std::atomic<uint64_t> calls{0};Transport t;t.open(f,p,image,128*MiB,calls);int dispatch=0;
    for(int pass=0;pass<2;++pass){const auto before=Io::capture();const auto reads=t.mover->stats().ssd_reads;
        for(int op=0;op<=int(f.prog.ops.size());++op){
            for(int attempt=0;attempt<64;++attempt){
                need(t.mover->begin_dispatch(dispatch++,nullptr),"real demand dispatch");need(t.mover->prefetch_to_op(op),"real scheduled demand stage");t.drain();
                bool ready=true;for(size_t h=1;h<f.prog.weights.size();++h)if(f.prog.weights[h].first_use_op==op&&f.prog.weights[h].decl.group.layer>=0){
                    const auto* s=t.mover->residency()->lookup(rad_weight(h));if(!s||!s->ptr)ready=false;}
                if(ready){for(size_t h=1;h<f.prog.weights.size();++h)if(f.prog.weights[h].first_use_op==op&&f.prog.weights[h].decl.group.layer>=0)check_published(t,f,image,rad_weight(h));break;}
                REQUIRE(attempt<63);
            }
        }
        t.drain();const auto after=Io::capture();const auto observed=delta(before,after);CHECK(observed.calls>=3);CHECK(observed.bytes>=uint64_t(3*33*MiB));
        CHECK_EQ(t.mover->stats().ssd_reads-reads,uint64_t(3));
        std::printf("  >32MiB identity pass%d actual demand reads=%llu/%llu logicalB\n",pass,(unsigned long long)observed.calls,(unsigned long long)observed.bytes);
    }
    CHECK_EQ(calls.load(),uint64_t(0));const auto report=t.mover->report();
    CHECK(has(report,"native file RAM cache: 0 hits"));CHECK(has(report,", 0 misses"));
}

#ifdef RAD_NATIVE_CACHE_ENGINE_CASE
TEST(native_cache_actual_engine_reserves_impossible_capacity_before_pools){
    std::ifstream in("/proc/meminfo");std::string key;uint64_t kb=0;bool found=false;
    while(in>>key){if(key=="MemTotal:"){in>>kb;found=true;break;}std::string line;std::getline(in,line);}
    REQUIRE(found&&kb>0&&kb<uint64_t(INT64_MAX/1024));
    const int64_t cache_mib=int64_t((kb+1023)/1024)+1024;
    tiny::TempDir dir("native_cache_engine");REQUIRE(dir.ok());tiny::Qwen35 shape;shape.layers=2;shape.ctx=512;REQUIRE(tiny::write_qwen35(dir.path,shape));
    Argv args({"--model",dir.path,"--radiance-home",home(),"--kernels","libref","--debug-accept-reference-kernels",
        "--embedding-placement","vram","--vision-tower-placement","vram","--ngram-placement","disk","--no-prefix-cache",
        "--max-model-len","512","--max-num-seqs","1","--max-num-batched-tokens","4","--num-speculative-tokens","0",
        "--host-pool-mib","4","--weights-disk-tier","--expert-ram-cache-total-mib",std::to_string(cache_mib)});
    Config config;REQUIRE_EQ(config_parse(args.argc(),args.argv.data(),&config),RAD_OK);
    REQUIRE(config.embedding_placement==TablePlacement::Vram&&config.tower_placement==TablePlacement::Vram&&config.ngram_placement==TablePlacement::Disk&&!config.prefix_cache);
    int status=RAD_OK;
    const auto log=stderr_of([&]{Engine engine(config);status=engine.start();});
    CHECK_EQ(status,RAD_E_NOMEM);CHECK(has(log,"native expert RAM cache: cache, remaining host allocations and --host-headroom-mib do not fit"));CHECK(!has(log,"pools:"));
    std::printf("  genuine tiny Engine cache request%lldMiB > real MemTotal%lluKiB refused before Pools, no serve/request\n",(long long)cache_mib,(unsigned long long)kb);
}
#endif

int main(int argc,char** argv){
#ifdef RAD_HAVE_HIP
    std::fprintf(stderr,"This integration checker requires a HOST-only Core build; refusing HIP/AQL binary.\n");return 2;
#else
    if(!device_is_host()){std::fprintf(stderr,"HOST backend required before any Mover allocation.\n");return 2;}
    return radtest::run_all(argc,argv);
#endif
}

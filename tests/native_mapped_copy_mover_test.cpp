/* Actual HOST Mover/Planner integration over private synthetic file bytes.
 * No model Source capability, kernel arithmetic, GPU, server or external asset.
 * Optional genuine tiny-Engine pre-Pools case needs the matching whole-Core
 * export/link setup in the accompanying prototype's enable_engine_case.cmake. */
#include "rad_test.h"
#include "device/device.h"
#include "device/directio.h"
#include "place/mover.h"
#include "place/planner.h"
#include "place/host_expert_pool.h"
#include "place/readahead.h"
#include "place/router_predictor.h"
#include "place/stager.h"
#include "tiny_model.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#ifdef RAD_NATIVE_CACHE_ENGINE_CASE
#include "engine.h"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <cerrno>
#include <chrono>
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
#include <thread>
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
                g.push_back(weight(stem+".gu",int64_t(NativeMappedFileCache::GU),RAD_ACCESS_CONDITIONAL,l,e));
                d.push_back(weight(stem+".dn",int64_t(NativeMappedFileCache::DN),RAD_ACCESS_CONDITIONAL,l,e));}
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
        //Use exact native transport sizes/tag. Relayout remains the explicitly synthetic
        //reversal callback above; this fixture never claims real MXFP4 arithmetic.
        for(auto& w:prog.weights)if(w.decl.group.expert>=0)w.layout_tag="ggml_mxfp4";
        return off+RAD_ALIGN_UNIT;
    }
};
Config plan_config(bool dense) {
    Config c;c.placement=dense?"layer_offload":"expert_tiered";c.deterministic=true;
    c.vram_weights_mib=dense?70:80;c.host_pool_mib=dense?133:201;c.weights_disk_tier=true;return c;
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
} //namespace

namespace rad {
struct NativeMappedMoverHostTest {
    struct Driver {
        struct Copy {void* dst;const void* source;int64_t bytes;bool mapped=false,complete=false;std::vector<uint8_t> expected;};
        struct Mark {RadEvent event=nullptr;size_t through=0;};
        Mover* owner=nullptr;
        std::vector<Copy> copies;
        std::array<Mark,RAD_PLACE_FILE_WINDOWS> marks{};
        size_t copy_attempts=0,validated=0,records=0,syncs=0,drains=0;
        size_t fail_copy_at=SIZE_MAX;
        bool fail_record=false,fail_sync=false,fail_drain=false;
        static int copy(void* context,void* dst,const void* source,int64_t bytes,RadStream) {
            auto& d=*static_cast<Driver*>(context);
            const size_t attempt=d.copy_attempts++;
            if(attempt==d.fail_copy_at)return RAD_E_STATE;
            const bool mapped=alive(*d.owner,source,size_t(bytes));
            if(!mapped&&!window_source(*d.owner,source,size_t(bytes)))return RAD_E_STATE;
            const auto* first=static_cast<const uint8_t*>(source);
            d.copies.push_back({dst,source,bytes,mapped,false,{first,first+bytes}});return RAD_OK;
        }
        int complete(size_t through) {
            if(through>copies.size())return RAD_E_STATE;
            for(size_t i=0;i<through;++i)if(!copies[i].complete){
                auto& copy=copies[i];
                //Dereference only while its real Lease OR reserved pinned window remains live.
                if(copy.mapped?!alive(*owner,copy.source,size_t(copy.bytes)):
                               !window_source(*owner,copy.source,size_t(copy.bytes)))return RAD_E_STATE;
                if(std::memcmp(copy.source,copy.expected.data(),size_t(copy.bytes)))return RAD_E_STATE;
                std::memcpy(copy.dst,copy.source,size_t(copy.bytes));copy.complete=true;++validated;
                std::vector<uint8_t>().swap(copy.expected);
            }
            return RAD_OK;
        }
        static int record(void* context,RadEvent event,RadStream stream) {
            auto& d=*static_cast<Driver*>(context);++d.records;
            if(d.fail_record)return RAD_E_STATE;
            Mark* mark=nullptr;
            for(auto& m:d.marks)if(m.event==event){mark=&m;break;}
            if(!mark)for(auto& m:d.marks)if(!m.event){mark=&m;break;}
            if(!mark)return RAD_E_STATE;
            *mark={event,d.copies.size()};return rad_event_record(event,stream);
        }
        static int sync(void* context,RadEvent event) {
            auto& d=*static_cast<Driver*>(context);++d.syncs;
            if(d.fail_sync)return RAD_E_STATE;
            for(auto& m:d.marks)if(m.event==event){
                RAD_TRY(d.complete(m.through));return rad_event_sync(event);
            }
            return RAD_E_STATE;
        }
        static int drain(void* context,RadStream stream) {
            auto& d=*static_cast<Driver*>(context);++d.drains;
            if(d.fail_drain)return RAD_E_STATE;
            RAD_TRY(d.complete(d.copies.size()));return rad_stream_sync(stream);
        }
        size_t pending()const{size_t result=0;for(const auto& c:copies)result+=!c.complete;return result;}
    };
    static bool alive(const Mover& m,const void* source,size_t bytes) {
        for(const auto& batch:m.mapped_windows_)for(const auto& lease:batch.leases)
            if(lease.active()&&lease.device()==source&&lease.bytes()==bytes)return true;
        return false;
    }
    static bool window_source(const Mover& m,const void* source,size_t bytes) {
        const uintptr_t at=reinterpret_cast<uintptr_t>(source),base=reinterpret_cast<uintptr_t>(m.windows_);
        if(!base||!bytes||at<base)return false;
        const size_t offset=at-base,W=size_t(RAD_PLACE_FILE_WINDOW_BYTES);
        return offset<size_t(m.n_windows_)*W&&bytes<=W-offset%W;
    }
    static void attach(Mover& m,Driver& d){d.owner=&m;m.mapped_copy_hooks_={&d,Driver::copy,Driver::record,Driver::sync,Driver::drain};}
    static NativeMappedFileCache::Snapshot stats(const Mover& m){return m.mapped_file_cache_.stats();}
    static bool enabled(const Mover& m){return m.mapped_file_cache_on_;}
    static int drain(Mover& m){return m.mapped_file_drain();}
    static int retire(Mover& m,int window){return m.retire_file_window(window);}
    static int clear(Mover& m){return m.mapped_file_cache_.clear();}
    static int last_window(const Mover& m){return(m.win_next_+m.n_windows_-1)%m.n_windows_;}
    static void next_window(Mover& m,int window){m.win_next_=window;}
    static const Program* program(const Mover& m){return m.prog_;}
    static int64_t budget(const Mover& m){return m.cfg_.native_file_cache_bytes;}
    static int prime(Mover& m,const WeightInfo& w){
        std::vector<uint8_t> out(size_t(w.stored_bytes));Mover::RelayIn input;bool hit=false;
        return m.read_native_file(w,out.data(),int64_t(out.size()),&input,&hit);
    }
    static bool eligible(const WeightInfo& w){return Mover::mapped_file_eligible(w);}
    static void serial_reader(Mover& m){m.relay_threads_=1;}
};
} //namespace rad

namespace {
using Access=NativeMappedMoverHostTest;
struct Environment {
    std::array<const char*,4> names{"RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE",
        "RADIANCE_EXPERIMENTAL_NATIVE_FILE_WORKERS","RADIANCE_EXPERIMENTAL_NATIVE_CACHE_CONCURRENT_COPY",
        "RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS"};
    std::array<std::string,4> old;std::array<bool,4> present{};
    explicit Environment(const char* mapped="1",const char* slabs="0"){
        for(size_t i=0;i<names.size();++i){const auto* p=std::getenv(names[i]);present[i]=p;old[i]=p?p:"";}
        if(mapped)::setenv(names[0],mapped,1);else ::unsetenv(names[0]);
        ::setenv(names[1],"0",1);::setenv(names[2],"0",1);
        if(slabs)::setenv(names[3],slabs,1);else ::unsetenv(names[3]);
    }
    ~Environment(){for(size_t i=0;i<names.size();++i)if(present[i])::setenv(names[i],old[i].c_str(),1);else ::unsetenv(names[i]);}
};
struct Case {
    Environment environment;
    Fixture fixture;
    std::unique_ptr<Image> image;
    Config config;
    Plan plan;
    std::vector<int32_t> units;
    std::atomic<uint64_t> calls{0};
    Access::Driver driver; //outlives Transport: it is consulted during Mover destruction
    Transport transport;
    Case(bool relaid,int64_t cache,const char* opt="1",bool native_tag=true,const char* slabs="0"):environment(opt,slabs){
        fixture.experts();image=std::make_unique<Image>(fixture.lay_out(relaid));
        if(!native_tag)for(auto& w:fixture.prog.weights)if(w.decl.group.expert>=0)w.layout_tag="ordinary-window-fixture";
        config=plan_config(false);plan=planned(fixture,config,relaid);units=two_ssd_experts(plan);
        transport.open(fixture,plan,*image,cache,calls);
        if(Access::enabled(*transport.mover))Access::attach(*transport.mover,driver);
    }
    ~Case(){
        //A failed assertion must not leave an intentionally failing HOST seam armed at teardown.
        driver.fail_drain=driver.fail_sync=driver.fail_record=false;driver.fail_copy_at=SIZE_MAX;
        if(transport.mover&&Access::drain(*transport.mover)!=RAD_OK)std::abort();
    }
    void stage(const std::vector<int32_t>& selected,int buffer=0,
               NativeFileCacheReadPolicy policy=NativeFileCacheReadPolicy::Normal){
        std::vector<int32_t> staged;
        need(transport.mover->stage_file_units(selected,transport.mover->file_buffer(buffer),
            transport.mover->file_buffer_bytes(),nullptr,transport.done,&staged,nullptr,nullptr,0,nullptr,policy),"actual mapped-source routed stage");
        REQUIRE(staged==selected);
    }
    void check(const std::vector<int32_t>& selected){
        need(Access::drain(*transport.mover),"fixture proves fake consumer complete before lease release");
        for(int32_t u:selected)for(auto h:plan.units[size_t(u)].weights)check_published(transport,fixture,*image,h);
        transport.drain();transport.mover->unstage_units(selected);
    }
};
void basic_case(bool relaid){
    Case c(relaid,26*MiB);const std::vector<int32_t> a{c.units[0]},b{c.units[1]};
    REQUIRE(Access::enabled(*c.transport.mover));
    c.stage(a);c.check(a);const auto cold=Access::stats(*c.transport.mover);
    CHECK_EQ(cold.logical_payload,size_t(18800640));CHECK_EQ(cold.cache.resident_entries,size_t(2));
    CHECK(cold.external_metadata+cold.cache.charge()<=size_t(26*MiB));
    const auto callbacks=c.calls.load();const auto io0=Io::capture();c.stage(a);const auto io1=Io::capture();
    const auto io=delta(io0,io1);CHECK_EQ(io.calls,uint64_t(0));CHECK_EQ(io.bytes,uint64_t(0));
    CHECK_EQ(c.calls.load(),callbacks);CHECK_EQ(c.driver.pending(),size_t(2));
    CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    CHECK_EQ(Access::clear(*c.transport.mover),RAD_E_STATE);
    //An unrelated cold expert cannot evict the held LRU owner; it takes the ordinary window path.
    c.stage(b,1);CHECK_EQ(c.driver.pending(),size_t(4));
    CHECK(Access::stats(*c.transport.mover).cache.leased_tail_bypasses>=2);
    c.check(b);c.check(a);CHECK_EQ(c.driver.validated,size_t(6));
    CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));
    CHECK_EQ(c.transport.mover->stats().mapped_source_bytes,uint64_t(18800640));
    CHECK(has(c.transport.mover->report(),"logical payload"));CHECK(has(c.transport.mover->report(),"conservative mapped backing"));
}
} //namespace

TEST(mapped_source_actual_mover_identity_hit_copy_and_leased_tail_fallback){basic_case(false);}
TEST(mapped_source_actual_mover_relayout_hit_copy_and_leased_tail_fallback){basic_case(true);}

TEST(mapped_source_actual_mover_same_layer_GU_hit_then_DN_miss_is_nonblocking){
    Case c(true,18*MiB);const std::vector<int32_t> a{c.units[0]};
    Access::serial_reader(*c.transport.mover); //force the exact GU-hit then DN-miss order
    const auto& unit=c.plan.units[size_t(a[0])];REQUIRE_EQ(unit.weights.size(),size_t(2));
    const auto& gu=c.fixture.prog.weights[unit.weights[0]];
    REQUIRE_EQ(gu.stored_bytes,int64_t(NativeMappedFileCache::GU));
    need(Access::prime(*c.transport.mover,gu),"prime only actual GU read/relayout");
    c.stage(a);CHECK_EQ(c.driver.pending(),size_t(2));
    CHECK_EQ(Access::stats(*c.transport.mover).cache.leased_tail_bypasses,uint64_t(1));
    CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(1));
    c.check(a);CHECK_EQ(c.driver.validated,size_t(2));
}

TEST(mapped_source_actual_mover_copy_event_failure_retains_until_proven_drain){
    Case c(false,26*MiB);const std::vector<int32_t> a{c.units[0]};c.stage(a);c.check(a);
    c.driver.fail_record=true;std::vector<int32_t> staged;
    CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
    CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    c.driver.fail_drain=true;const auto* original=Access::program(*c.transport.mover);const auto cap=Access::budget(*c.transport.mover);
    Fixture other;other.experts();MoverConfig changed;changed.native_file_cache_bytes=18*MiB;
    CHECK_EQ(c.transport.mover->init(other.prog,c.plan,changed),RAD_E_STATE);
    CHECK(Access::program(*c.transport.mover)==original);CHECK_EQ(Access::budget(*c.transport.mover),cap);
    CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    c.driver.fail_drain=false;c.driver.fail_record=false;
    need(Access::drain(*c.transport.mover),"failed stage prefix completes without publishing");
    CHECK_EQ(c.driver.validated,size_t(4));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));
    for(auto h:c.plan.units[size_t(a[0])].weights)CHECK(c.transport.mover->residency()->lookup(h)->ptr==nullptr);
    c.stage(a);c.check(a); //only a subsequent completely successful stage may publish
}

TEST(mapped_source_actual_mover_partial_copy_failure_and_window_event_generations){
    Case c(false,26*MiB);const std::vector<int32_t> a{c.units[0]};c.stage(a);c.check(a);
    c.driver.fail_copy_at=c.driver.copy_attempts+1;std::vector<int32_t> staged;
    CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
    CHECK_EQ(c.driver.pending(),size_t(1));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    c.driver.fail_copy_at=SIZE_MAX;need(Access::drain(*c.transport.mover),"partial enqueue drain");
    c.stage(a);const int window=Access::last_window(*c.transport.mover);CHECK_EQ(c.driver.pending(),size_t(2));
    c.driver.fail_sync=true;CHECK_EQ(Access::retire(*c.transport.mover,window),RAD_E_STATE);
    CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    c.driver.fail_sync=false;need(Access::retire(*c.transport.mover,window),"same recorded generation completion");
    CHECK_EQ(c.driver.pending(),size_t(0));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));
    const auto old_syncs=c.driver.syncs;Access::next_window(*c.transport.mover,window);c.stage(a);
    CHECK_EQ(c.driver.syncs,old_syncs);CHECK_EQ(c.driver.pending(),size_t(2));
    need(Access::retire(*c.transport.mover,window),"new event generation completion");c.check(a);
    CHECK_EQ(c.driver.pending(),size_t(0));
}

TEST(mapped_source_actual_mover_cold_fallback_record_failure_requires_drain_before_reuse){
    //Both relaid/native routing and the original direct-ext path need the same window proof.
    for(bool native_tag:{true,false}){
        Case c(false,26*MiB,"1",native_tag);const std::vector<int32_t> a{c.units[0]};
        c.driver.fail_record=true;std::vector<int32_t> staged;
        CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
            c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
        CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(c.driver.validated,size_t(0));
        CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));
        const int window=Access::last_window(*c.transport.mover);const auto attempts=c.driver.copy_attempts;
        c.driver.fail_drain=true;CHECK_EQ(Access::retire(*c.transport.mover,window),RAD_E_STATE);
        Access::next_window(*c.transport.mover,window);
        const auto io0=Io::capture();
        CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
            c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
        const auto io1=Io::capture();CHECK_EQ(delta(io0,io1).calls,uint64_t(0));
        CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(c.driver.copy_attempts,attempts);
        for(auto h:c.plan.units[size_t(a[0])].weights)CHECK(c.transport.mover->residency()->lookup(h)->ptr==nullptr);
        c.driver.fail_drain=false;c.driver.fail_record=false;
        need(Access::retire(*c.transport.mover,window),"cold fallback sources drain before pinned window reuse");
        CHECK_EQ(c.driver.pending(),size_t(0));CHECK_EQ(c.driver.validated,size_t(2));
        Access::next_window(*c.transport.mover,window);c.stage(a);c.check(a);
        CHECK_EQ(c.driver.validated,size_t(4));
    }
}

TEST(mapped_source_actual_mover_epoch_change_refuses_cached_source_before_copy){
    Case c(true,26*MiB);const std::vector<int32_t> a{c.units[0]};c.stage(a);c.check(a);
    REQUIRE_EQ(::ftruncate(c.image->fd,off_t(c.image->bytes.size()+1)),0);REQUIRE_EQ(::fsync(c.image->fd),0);
    const auto before=c.transport.mover->stats();const auto attempts=c.driver.copy_attempts;
    std::vector<int32_t> staged;CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
    CHECK_EQ(c.driver.copy_attempts,attempts);CHECK_EQ(c.driver.pending(),size_t(0));
    CHECK_EQ(c.transport.mover->stats().h2d_bytes,before.h2d_bytes);
    for(auto h:c.plan.units[size_t(a[0])].weights)CHECK(c.transport.mover->residency()->lookup(h)->ptr==nullptr);
    CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged,nullptr,nullptr,0,nullptr,NativeFileCacheReadPolicy::ReadOnly),RAD_E_STATE);
    CHECK_EQ(c.driver.copy_attempts,attempts);CHECK_EQ(c.driver.pending(),size_t(0));
    CHECK_EQ(c.transport.mover->stats().h2d_bytes,before.h2d_bytes);
}

TEST(mapped_source_actual_mover_ReadOnly_bypasses_unchanged_helper_without_admission_or_touch){
    Case c(true,26*MiB);const std::vector<int32_t> a{c.units[0]},b{c.units[1]};
    c.stage(a);c.check(a);const auto before=Access::stats(*c.transport.mover);const auto callbacks=c.calls.load();
    const auto io0=Io::capture();c.stage(a,0,NativeFileCacheReadPolicy::ReadOnly);const auto io1=Io::capture();
    CHECK(delta(io0,io1).calls>=2);CHECK_EQ(c.driver.pending(),size_t(2));
    CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));c.check(a);
    CHECK_EQ(c.calls.load(),callbacks+2);
    for(int pass=0;pass<2;++pass){c.stage(b,0,NativeFileCacheReadPolicy::ReadOnly);c.check(b);}
    const auto after=Access::stats(*c.transport.mover);
    CHECK_EQ(after.cache.hits,before.cache.hits);CHECK_EQ(after.cache.misses,before.cache.misses);
    CHECK_EQ(after.cache.evictions,before.cache.evictions);CHECK_EQ(after.cache.inserts,before.cache.inserts);
    CHECK_EQ(after.logical_payload,before.logical_payload);CHECK_EQ(after.allocations,before.allocations);
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_hits,uint64_t(0));
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_misses,uint64_t(6));
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_miss_bytes,uint64_t(3)*18800640);
    const auto warm0=Io::capture();c.stage(a);const auto warm1=Io::capture();
    CHECK_EQ(delta(warm0,warm1).calls,uint64_t(0));CHECK_EQ(c.driver.pending(),size_t(2));c.check(a);
    CHECK(has(c.transport.mover->report(),"ReadOnly policy: bypass"));
}

TEST(mapped_source_actual_mover_destructor_drains_before_cache_allocator_teardown){
    Case c(false,26*MiB);const std::vector<int32_t> a{c.units[0]};c.stage(a);c.check(a);c.stage(a);
    CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(c.driver.validated,size_t(2));
    c.transport.mover.reset();
    CHECK_EQ(c.driver.pending(),size_t(0));CHECK_EQ(c.driver.validated,size_t(4));CHECK(c.driver.drains>=1);
}

TEST(mapped_source_exact_optin_and_narrow_native_transport_scope){
    for(const char* env:{static_cast<const char*>(nullptr),"0","true","01"}){
        Case c(false,26*MiB,env);CHECK(!Access::enabled(*c.transport.mover));
    }
    WeightInfo w;w.decl.access=RAD_ACCESS_CONDITIONAL;w.decl.shard=RAD_SHARD_NONE;
    w.decl.group.layer=0;w.decl.group.expert=0;w.layout_tag="ggml_mxfp4";w.stored_bytes=NativeMappedFileCache::GU;
    CHECK(Access::eligible(w));w.decl.group.layer=40;CHECK(!Access::eligible(w));
    w.decl.group.layer=0;w.decl.group.expert=96;CHECK(!Access::eligible(w));
    w.decl.group.expert=0;w.layout_tag="other";CHECK(!Access::eligible(w));
    w.layout_tag="ggml_mxfp4";--w.stored_bytes;CHECK(!Access::eligible(w));
}

namespace ds41_mapped {
struct CacheHostTest {
    static void before_copy(Cache& cache,void(*hook)(void*)noexcept,void* context){
        std::lock_guard<std::mutex> owner(cache.owner_mutex_);
        if(!cache.state_)std::abort();
        std::lock_guard<std::mutex> lock(cache.state_->mutex);
        cache.state_->test_before_copy=hook;cache.state_->test_copy_context=context;
    }
    static Cache::Allocation idle(Cache& cache){
        std::lock_guard<std::mutex> owner(cache.owner_mutex_);
        if(!cache.state_)std::abort();
        std::lock_guard<std::mutex> lock(cache.state_->mutex);
        return cache.state_->idle;
    }
};
} //namespace ds41_mapped
namespace rad {
struct NativeMappedFileCacheHostTest {
    static ds41_mapped::Cache& cache(NativeMappedFileCache& c){return c.cache_;}
    static ds41_mapped::SlabCache& slabs(NativeMappedFileCache& c){return c.slab_cache_;}
    static void reject_alias(NativeMappedFileCache& c,bool reject){c.test_reject_alias_=reject;}
    static size_t exhaust_records(NativeMappedFileCache& c){const auto previous=c.first_free_;c.first_free_=SIZE_MAX;return previous;}
    static void restore_records(NativeMappedFileCache& c,size_t previous){c.first_free_=previous;}
};
} //namespace rad
namespace {
using Adapter=NativeMappedFileCache;
using AdapterTest=NativeMappedFileCacheHostTest;
bool guard_tail_intact(const uint8_t* host,size_t bytes){
    if(!host)return false;
    for(size_t n=0;n<64;++n)
        if(host[-ptrdiff_t(64)+ptrdiff_t(n)]!=0xbd||host[bytes+n]!=0||host[bytes+64+n]!=0xbd)return false;
    return true;
}
struct PoisonBeforePublish {
    Adapter& adapter;size_t key=0,bytes=0,calls=0;bool okay=true;
    static void run(void* context)noexcept{
        auto& state=*static_cast<PoisonBeforePublish*>(context);++state.calls;
        auto& cache=AdapterTest::cache(state.adapter);
        const auto allocation=ds41_mapped::CacheHostTest::idle(cache);
        state.okay=state.okay&&allocation.bytes==state.bytes+64&&guard_tail_intact(allocation.host,state.bytes);
        Adapter::Lease invisible;
        state.okay=state.okay&&state.adapter.borrow(state.key,state.bytes,&invisible)==Adapter::Status::Miss;
        //Force stale nonzero payload just before the unchanged full memcpy, while unpublished.
        if(allocation.host&&allocation.bytes>=state.bytes)std::memset(allocation.host,0xd3,state.bytes);else state.okay=false;
    }
    void arm(size_t next_key,size_t next_bytes){
        key=next_key;bytes=next_bytes;
        ds41_mapped::CacheHostTest::before_copy(AdapterTest::cache(adapter),run,this);
    }
};
std::vector<uint8_t> adapter_pattern(size_t bytes,uint64_t offset){
    std::vector<uint8_t> result(bytes);
    for(size_t i=0;i<bytes;++i)result[i]=source_byte(i+offset);
    return result;
}
void check_adapter_hit(Adapter& adapter,size_t key,const std::vector<uint8_t>& expected){
    Adapter::Lease lease;REQUIRE(adapter.borrow(key,expected.size(),&lease)==Adapter::Status::Hit);
    CHECK_EQ(lease.bytes(),expected.size());CHECK(std::memcmp(lease.host(),expected.data(),expected.size())==0);
    CHECK(guard_tail_intact(lease.host(),expected.size()));CHECK(lease.device()!=nullptr);
    REQUIRE(lease.release()==Adapter::Status::Ok);
}
} //namespace
TEST(mapped_source_guard_tail_only_init_full_payload_before_publication_and_reuse){
    Environment environment;
    Adapter adapter;need(adapter.configure(15*MiB,3,0),"bounded real HOST mapped adapter");
    PoisonBeforePublish poison{adapter};
    const auto gu=adapter_pattern(Adapter::GU,7101),dn=adapter_pattern(Adapter::DN,8101);
    poison.arm(0,gu.size());REQUIRE(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::Ok);
    CHECK(poison.okay);CHECK_EQ(poison.calls,size_t(1));check_adapter_hit(adapter,0,gu);
    Adapter::Lease held;REQUIRE(adapter.borrow(0,gu.size(),&held)==Adapter::Status::Hit);
    REQUIRE(adapter.insert(1,dn.data(),dn.size())==Adapter::Status::Bypass);
    CHECK_EQ(adapter.stats().api.allocate_attempts,uint64_t(1));CHECK_EQ(poison.calls,size_t(1));
    REQUIRE(held.release()==Adapter::Status::Ok);
    poison.arm(1,dn.size());REQUIRE(adapter.insert(1,dn.data(),dn.size())==Adapter::Status::Ok);
    CHECK(poison.okay);CHECK_EQ(poison.calls,size_t(2));check_adapter_hit(adapter,1,dn);
    Adapter::Lease stale;CHECK(adapter.borrow(0,gu.size(),&stale)==Adapter::Status::Miss);
    auto stats=adapter.stats();CHECK_EQ(stats.api.allocate_attempts,uint64_t(2));CHECK_EQ(stats.allocations,uint64_t(2));
    CHECK_EQ(stats.api.allocate_null_returns,uint64_t(0));CHECK_EQ(stats.api.alias_rejections,uint64_t(0));
    CHECK_EQ(stats.api.free_invocations,uint64_t(1));CHECK_EQ(stats.public_free_invocations,uint64_t(1));
    CHECK(stats.api.allocate_ns>=stats.api.allocate_max_ns);CHECK(stats.api.free_ns>=stats.api.free_max_ns);
    need(adapter.clear(),"all real adapter mappings retired after leases");stats=adapter.stats();
    CHECK_EQ(stats.api.free_invocations,uint64_t(2));CHECK_EQ(stats.logical_payload,size_t(0));
}
TEST(mapped_source_guard_tail_only_init_failure_cleanup_API_census_and_retry){
    Environment environment;
    Adapter adapter;need(adapter.configure(15*MiB,3,0),"real HOST adapter failure fixture");
    const auto gu=adapter_pattern(Adapter::GU,9101);
    const size_t previous=AdapterTest::exhaust_records(adapter);
    CHECK(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::Allocation);
    CHECK_EQ(adapter.stats().api.allocate_attempts,uint64_t(0));
    AdapterTest::restore_records(adapter,previous);
    AdapterTest::reject_alias(adapter,true);
    CHECK(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::Allocation);
    auto stats=adapter.stats();CHECK_EQ(stats.api.allocate_attempts,uint64_t(1));
    CHECK_EQ(stats.api.allocate_null_returns,uint64_t(0));CHECK_EQ(stats.api.alias_rejections,uint64_t(1));
    CHECK_EQ(stats.allocations,uint64_t(1));CHECK_EQ(stats.api.free_invocations,uint64_t(1));
    CHECK_EQ(stats.public_free_invocations,uint64_t(1));CHECK_EQ(stats.logical_payload,size_t(0));
    Adapter::Lease invisible;CHECK(adapter.borrow(0,gu.size(),&invisible)==Adapter::Status::Miss);
    AdapterTest::reject_alias(adapter,false);
    PoisonBeforePublish poison{adapter};poison.arm(0,gu.size());
    REQUIRE(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::Ok);
    CHECK(poison.okay);CHECK_EQ(poison.calls,size_t(1));check_adapter_hit(adapter,0,gu);
    need(adapter.clear(),"retry owner cleanup after real alias-failure cleanup");stats=adapter.stats();
    CHECK_EQ(stats.api.allocate_attempts,uint64_t(2));CHECK_EQ(stats.allocations,uint64_t(2));
    CHECK_EQ(stats.api.free_invocations,uint64_t(2));CHECK_EQ(stats.public_free_invocations,uint64_t(2));
    CHECK(stats.api.allocate_ns>=stats.api.allocate_max_ns);CHECK(stats.api.free_ns>=stats.api.free_max_ns);
    need(adapter.configure(15*MiB,3,0),"fresh namespace resets API counters");stats=adapter.stats();
    CHECK_EQ(stats.api.allocate_attempts,uint64_t(0));CHECK_EQ(stats.api.free_invocations,uint64_t(0));
}

//V6 exercises the actual Mover with the same delayed copies, data and Planner;
//only the new slab opt-in/cache budget differs. No fake GPU numerical claim.
TEST(mapped_slab_actual_mover_identity_relayout_hits_retain_one_paired_owner){
    for(bool relaid:{false,true}){
        Case c(relaid,80*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]},b{c.units[1]};
        c.stage(a);c.check(a);auto cold=Access::stats(*c.transport.mover);
        REQUIRE(cold.slab_policy);CHECK_EQ(cold.allocations,uint64_t(1));CHECK_EQ(cold.public_free_invocations,uint64_t(0));
        CHECK_EQ(cold.slabs.retained_slab_owners,size_t(1));
        for(const auto& k:cold.slabs.classes){CHECK_EQ(k.capacity_slots,size_t(4));CHECK_EQ(k.resident_slots,size_t(1));CHECK_EQ(k.idle_slots,size_t(3));}
        CHECK(cold.external_metadata+cold.cache.charge()<=size_t(80*MiB));
        c.stage(b,1);c.check(b);const auto calls=c.calls.load();const auto io0=Io::capture();c.stage(a);const auto io1=Io::capture();
        CHECK_EQ(delta(io0,io1).calls,uint64_t(0));CHECK_EQ(c.calls.load(),calls);
        auto warm=Access::stats(*c.transport.mover);CHECK_EQ(warm.cache.lease_refs,size_t(2));
        CHECK_EQ(warm.allocations,cold.allocations);CHECK_EQ(warm.public_free_invocations,uint64_t(0));
        CHECK_EQ(Access::clear(*c.transport.mover),RAD_E_STATE);CHECK_EQ(Access::stats(*c.transport.mover).api.free_invocations,uint64_t(0));
        c.check(a);CHECK(has(c.transport.mover->report(),"retained-slab ledger"));CHECK(has(c.transport.mover->report(),"class GU:"));
    }
}
TEST(mapped_slab_actual_mover_GU_lease_then_DN_miss_uses_idle_class_without_wait){
    Case c(true,80*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};Access::serial_reader(*c.transport.mover);
    const auto& unit=c.plan.units[size_t(a[0])];REQUIRE_EQ(unit.weights.size(),size_t(2));
    need(Access::prime(*c.transport.mover,c.fixture.prog.weights[unit.weights[0]]),"slab prime GU only");
    c.stage(a);auto stats=Access::stats(*c.transport.mover);
    CHECK_EQ(stats.cache.lease_refs,size_t(1));CHECK_EQ(stats.cache.leased_tail_bypasses,uint64_t(0));
    CHECK_EQ(stats.allocations,uint64_t(1));CHECK_EQ(stats.public_free_invocations,uint64_t(0));
    CHECK_EQ(stats.slabs.classes[0].leased_slots,size_t(1));CHECK_EQ(stats.slabs.classes[1].resident_slots,size_t(1));
    CHECK_EQ(c.driver.pending(),size_t(2));c.check(a);
}
TEST(mapped_slab_actual_mover_class_tail_lease_bypasses_then_reuses_without_free){
    Case c(true,80*MiB,"1",true,"1");std::vector<int32_t>all;
    for(size_t u=0;u<c.plan.units.size();++u)if(c.plan.units[u].tier==Tier::SSD&&c.plan.units[u].expert>=0)all.push_back(int32_t(u));
    REQUIRE_EQ(all.size(),size_t(6));
    for(size_t k=0;k<4;++k){c.stage({all[k]});c.check({all[k]});}
    const auto full=Access::stats(*c.transport.mover);CHECK_EQ(full.allocations,uint64_t(1));
    for(const auto& cl:full.slabs.classes){CHECK_EQ(cl.capacity_slots,size_t(4));CHECK_EQ(cl.resident_slots,size_t(4));CHECK_EQ(cl.idle_slots,size_t(0));}
    c.stage({all[0]}); //the actual window holds both leases until its copy completes
    for(size_t k=1;k<4;++k)for(auto h:c.plan.units[size_t(all[k])].weights)
        need(Access::prime(*c.transport.mover,c.fixture.prog.weights[h]),"same-class hits make leased expert the tail");
    c.stage({all[4]},1);auto held=Access::stats(*c.transport.mover);
    CHECK_EQ(held.cache.leased_tail_bypasses,uint64_t(2));CHECK_EQ(held.cache.lease_refs,size_t(2));
    CHECK_EQ(held.allocations,full.allocations);CHECK_EQ(held.public_free_invocations,uint64_t(0));
    CHECK_EQ(c.driver.pending(),size_t(4));c.check({all[4]});c.check({all[0]});
    c.stage({all[4]},1);c.check({all[4]});const auto reused=Access::stats(*c.transport.mover);
    CHECK(reused.cache.evictions>=2);CHECK_EQ(reused.allocations,full.allocations);CHECK_EQ(reused.public_free_invocations,uint64_t(0));
    CHECK_EQ(reused.cache.payload_charge,full.cache.payload_charge);
}
TEST(mapped_slab_actual_mover_cold_and_warm_record_failure_retains_before_drain){
    for(bool warm:{false,true}){
        Case c(false,80*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};
        if(warm){c.stage(a);c.check(a);}c.driver.fail_record=true;std::vector<int32_t>staged;
        CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
            c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
        CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,warm?size_t(2):size_t(0));
        const int window=Access::last_window(*c.transport.mover);const auto attempts=c.driver.copy_attempts;
        c.driver.fail_drain=true;CHECK_EQ(Access::retire(*c.transport.mover,window),RAD_E_STATE);
        Access::next_window(*c.transport.mover,window);const auto before=Io::capture();
        CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
            c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged),RAD_E_STATE);
        const auto after=Io::capture();CHECK_EQ(delta(before,after).calls,uint64_t(0));
        CHECK_EQ(c.driver.copy_attempts,attempts);CHECK_EQ(Access::stats(*c.transport.mover).public_free_invocations,uint64_t(0));
        for(auto h:c.plan.units[size_t(a[0])].weights)CHECK(c.transport.mover->residency()->lookup(h)->ptr==nullptr);
        c.driver.fail_drain=c.driver.fail_record=false;need(Access::retire(*c.transport.mover,window),"slab copy/fallback completion ACK");
        CHECK_EQ(c.driver.pending(),size_t(0));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));
        c.stage(a);c.check(a);
    }
}
namespace ds41_mapped {
struct SlabCacheHostTest {
    static void before_copy(SlabCache& c,void(*hook)(void*)noexcept,void* context){
        std::lock_guard<std::mutex>w(c.writer_mutex_);std::lock_guard<std::mutex>o(c.owner_mutex_);
        if(!c.state_)std::abort();std::lock_guard<std::mutex>l(c.state_->mutex);
        c.state_->test_before_copy=hook;c.state_->test_copy_context=context;
    }
    static std::pair<uint8_t*,size_t> copying(SlabCache& c){
        std::lock_guard<std::mutex>o(c.owner_mutex_);if(!c.state_)return{};std::lock_guard<std::mutex>l(c.state_->mutex);
        for(size_t i=0;i<c.state_->used_slabs;++i){auto&b=c.state_->slabs[i];if(!b.ready)continue;
            for(size_t j=0;j<2*b.pairs;++j)if(b.keys[j]==SlabCache::busy)return{SlabCache::host(b,j),SlabCache::bytes_for(b,SlabCache::slot_class(b,j))};}
        return{};
    }
};
} //namespace ds41_mapped

namespace {
struct SlabPoisonBeforePublish {
    Adapter& adapter;size_t key=0,bytes=0,calls=0;bool okay=true;
    static void run(void* context)noexcept{
        auto& x=*static_cast<SlabPoisonBeforePublish*>(context);++x.calls;
        auto p=ds41_mapped::SlabCacheHostTest::copying(AdapterTest::slabs(x.adapter));
        x.okay=x.okay&&p.second==x.bytes&&guard_tail_intact(p.first,x.bytes);
        Adapter::Lease invisible;x.okay=x.okay&&x.adapter.borrow(x.key,x.bytes,&invisible)==Adapter::Status::Miss;
        if(p.first&&p.second==x.bytes)std::memset(p.first,0xd7,p.second);else x.okay=false;
    }
    void arm(size_t next,size_t n){key=next;bytes=n;ds41_mapped::SlabCacheHostTest::before_copy(AdapterTest::slabs(adapter),run,this);}
};
} //namespace
TEST(mapped_slab_real_HOST_adapter_full_payload_guards_reuse_and_terminal_once){
    Environment environment("1","1");Adapter adapter;need(adapter.configure(22*MiB,4,0),"bounded one-pair slab");
    const auto gu=adapter_pattern(Adapter::GU,12001),dn=adapter_pattern(Adapter::DN,13001);SlabPoisonBeforePublish poison{adapter};
    poison.arm(0,gu.size());REQUIRE(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::Ok);
    CHECK(poison.okay);check_adapter_hit(adapter,0,gu);Adapter::Lease held;REQUIRE(adapter.borrow(0,gu.size(),&held)==Adapter::Status::Hit);
    poison.arm(1,dn.size());REQUIRE(adapter.insert(1,dn.data(),dn.size())==Adapter::Status::Ok);
    CHECK(poison.okay);check_adapter_hit(adapter,1,dn);
    CHECK(adapter.insert(2,gu.data(),gu.size())==Adapter::Status::Bypass);CHECK_EQ(poison.calls,size_t(2));
    REQUIRE(held.release()==Adapter::Status::Ok);poison.arm(2,gu.size());REQUIRE(adapter.insert(2,gu.data(),gu.size())==Adapter::Status::Ok);
    CHECK(poison.okay);check_adapter_hit(adapter,2,gu);auto before=adapter.stats();
    CHECK_EQ(before.allocations,uint64_t(1));CHECK_EQ(before.public_free_invocations,uint64_t(0));
    CHECK_EQ(before.slabs.classes[0].capacity_slots,size_t(1));CHECK_EQ(before.slabs.classes[1].capacity_slots,size_t(1));
    CHECK(before.external_metadata+before.cache.charge()<=size_t(22*MiB));
    need(adapter.clear(),"terminal slab free invocation");const auto once=adapter.stats();
    CHECK_EQ(once.api.free_invocations,uint64_t(1));CHECK_EQ(once.slabs.terminal_free_invoked_owners,size_t(1));
    CHECK_EQ(once.slabs.terminal_backing_charge,before.slabs.retained_backing_charge);
    need(adapter.clear(),"idempotent terminal cleanup");CHECK_EQ(adapter.stats().api.free_invocations,uint64_t(1));
    CHECK_EQ(adapter.configure(22*MiB,4,0),RAD_E_STATE); //no spend-after-void-free refund
}
/* RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET's growth limit (set_growth_limit): 0 grows nothing (a
 * bypass, no allocation); held at what the cache holds, the slab fills its free slots and then
 * replaces in LRU order without a second slab; lifted, it grows again. Every hit reads the bytes
 * inserted: the limit changes only whether a unit is cached, never its bytes. */
TEST(mapped_slab_growth_limit_holds_growth_and_replaces_within_the_slabs){
    Environment environment("1","1");Adapter adapter;
    adapter.set_growth_limit(0); //configure quotes unlimited (its minimum quote), whatever the limit was
    need(adapter.configure(128*MiB,8,0),"room for a second slab");CHECK_EQ(adapter.growth_limit(),SIZE_MAX);
    const auto a=adapter_pattern(Adapter::GU,21001);
    adapter.set_growth_limit(0);
    CHECK(adapter.insert(0,a.data(),a.size())==Adapter::Status::Bypass);
    CHECK_EQ(adapter.stats().allocations,uint64_t(0));CHECK(adapter.growth_refusals()>0);
    adapter.set_growth_limit(SIZE_MAX);
    REQUIRE(adapter.insert(0,a.data(),a.size())==Adapter::Status::Ok);check_adapter_hit(adapter,0,a);
    const auto grown=adapter.stats();CHECK_EQ(grown.allocations,uint64_t(1));
    const size_t slots=grown.slabs.classes[0].capacity_slots;REQUIRE(slots>=1&&slots+3<8);
    CHECK_EQ(adapter.growth_charged(),grown.mapped_policy_charge);
    adapter.set_growth_limit(adapter.growth_charged());
    for(size_t k=1;k<slots+3;++k){
        const auto p=adapter_pattern(Adapter::GU,30000+k);
        REQUIRE(adapter.insert(k,p.data(),p.size())==Adapter::Status::Ok);check_adapter_hit(adapter,k,p);
    }
    CHECK_EQ(adapter.stats().allocations,uint64_t(1));
    {Adapter::Lease gone;CHECK(adapter.borrow(0,a.size(),&gone)==Adapter::Status::Miss);} //replaced, LRU
    adapter.set_growth_limit(SIZE_MAX);
    const auto z=adapter_pattern(Adapter::GU,40001);
    REQUIRE(adapter.insert(slots+3,z.data(),z.size())==Adapter::Status::Ok);check_adapter_hit(adapter,slots+3,z);
    CHECK_EQ(adapter.stats().allocations,uint64_t(2));
    need(adapter.clear(),"terminal cleanup");
}
/* The per-entry owners (no slabs): a limit at what is held bypasses a new key and keeps the old. */
TEST(mapped_entry_growth_limit_bypasses_without_evicting){
    Environment environment("1","0");Adapter adapter;need(adapter.configure(64*MiB,8,0),"per-entry budget");
    const auto a=adapter_pattern(Adapter::GU,51001),b=adapter_pattern(Adapter::DN,52001);
    adapter.set_growth_limit(0);
    CHECK(adapter.insert(0,a.data(),a.size())==Adapter::Status::Bypass);CHECK_EQ(adapter.stats().allocations,uint64_t(0));
    adapter.set_growth_limit(SIZE_MAX);
    REQUIRE(adapter.insert(0,a.data(),a.size())==Adapter::Status::Ok);
    adapter.set_growth_limit(adapter.growth_charged());
    CHECK(adapter.insert(1,b.data(),b.size())==Adapter::Status::Bypass);
    CHECK_EQ(adapter.stats().allocations,uint64_t(1));check_adapter_hit(adapter,0,a);
    adapter.set_growth_limit(SIZE_MAX);
    REQUIRE(adapter.insert(1,b.data(),b.size())==Adapter::Status::Ok);check_adapter_hit(adapter,1,b);
    need(adapter.clear(),"terminal cleanup");
}
/* The packed layout (RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE) is a hard tag: its slots have their
 * own sizes, only reserve_packed/borrow_packed reach them and their leases say packed(), and every
 * block_mxfp4 call -- the card's copy path -- misses or bypasses (counted), even for a resident key,
 * so a packed slot is never copied to a card and no block bytes land in one. A block cache refuses
 * the packed calls; the packed layout needs slabs and valid sizes. */
TEST(mapped_slab_real_HOST_adapter_packed_layout_is_CPU_only){
    const ds41_mapped::SlabCache::Sizes packed{11999296,6001984};
    {Environment noslabs("1","0");Adapter a;CHECK_EQ(a.configure(22*MiB,4,0,0,Adapter::Layout::Packed,packed),RAD_E_INVAL);}
    Environment environment("1","1");
    {Adapter a;CHECK_EQ(a.configure(22*MiB,4,0,0,Adapter::Layout::Packed,{packed[0],packed[0]}),RAD_E_INVAL);}
    Adapter adapter;need(adapter.configure(22*MiB,4,0,0,Adapter::Layout::Packed,packed),"packed one-pair slab");
    CHECK(adapter.packed());CHECK(adapter.slot_bytes()==packed);CHECK(adapter.stats().slab_policy);
    const auto gu=adapter_pattern(Adapter::GU,15001);
    CHECK(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::Bypass);
    {Adapter::Write w;CHECK(adapter.reserve_fill(0,Adapter::GU,&w)==Adapter::Status::Bypass);CHECK(w.empty());}
    Adapter::Lease none;CHECK(adapter.borrow(0,Adapter::GU,&none)==Adapter::Status::Miss);CHECK(none.empty());
    CHECK(adapter.borrow_readonly(0,Adapter::GU,&none)==Adapter::Status::Miss);
    Adapter::Write w;REQUIRE(adapter.reserve_packed(0,0,&w)==Adapter::Status::Ok);
    CHECK(w.packed());CHECK_EQ(w.bytes(),packed[0]);REQUIRE(w.host()!=nullptr);
    CHECK_EQ(reinterpret_cast<uintptr_t>(w.host())%64,uintptr_t(0));
    std::memset(w.host(),0x5a,packed[0]);
    Adapter::Lease first;REQUIRE(w.publish(&first)==Adapter::Status::Ok);
    CHECK(first.packed());CHECK_EQ(first.bytes(),packed[0]);CHECK(first.release()==Adapter::Status::Ok);CHECK(!first.packed());
    Adapter::Lease hit;REQUIRE(adapter.borrow_packed(0,0,&hit)==Adapter::Status::Hit);
    CHECK(hit.packed());CHECK_EQ(hit.bytes(),packed[0]);CHECK(hit.host()[0]==0x5a&&hit.host()[packed[0]-1]==0x5a);
    CHECK(guard_tail_intact(hit.host(),packed[0]));
    Adapter::Lease blockwise;CHECK(adapter.borrow(0,packed[0],&blockwise)==Adapter::Status::Miss);   //resident, still refused
    CHECK(adapter.present(0));
    Adapter::Lease cls;CHECK(adapter.borrow_packed(0,1,&cls)==Adapter::Status::Layout);   //another class than the entry's
    CHECK(adapter.borrow_packed(0,2,&cls)==Adapter::Status::Invalid);
    CHECK(hit.release()==Adapter::Status::Ok);
    const auto st=adapter.stats();CHECK(st.packed);CHECK(st.slot_bytes==packed);CHECK_EQ(st.layout_refusals,uint64_t(5));
    CHECK_EQ(st.slabs.classes[0].capacity_logical_bytes,packed[0]*st.slabs.classes[0].capacity_slots);
    need(adapter.clear(),"packed terminal cleanup");
    Adapter block;need(block.configure(22*MiB,4,0),"block one-pair slab");CHECK(!block.packed());
    Adapter::Lease bl;CHECK(block.borrow_packed(0,0,&bl)==Adapter::Status::Layout);
    Adapter::Write bw;CHECK(block.reserve_packed(0,0,&bw)==Adapter::Status::Layout);
    CHECK(block.slot_bytes()==ds41_mapped::SlabCache::block_sizes);
    need(block.clear(),"block terminal cleanup");
}
/* Large-chunk prefill's read-only chunks (RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT): borrow_packed_readonly
 * leases a packed entry as borrow_packed does (the same bytes, a packed lease, the hit counted) but leaves
 * the LRU order alone: of two entries, the older one borrowed read-only is still the one the next
 * reservation evicts, where a touching borrow would have moved it to the front. */
TEST(mapped_slab_packed_readonly_borrow_keeps_the_lru_order){
    const ds41_mapped::SlabCache::Sizes packed{11999296,6001984};
    Environment environment("1","1");
    for(int touch=0;touch<2;++touch){
        Adapter adapter;need(adapter.configure(40*MiB,4,0,0,Adapter::Layout::Packed,packed),"packed two-pair slab");
        if(adapter.stats().slabs.classes[0].capacity_slots<2){SKIP("the slab holds fewer than two gate_up slots");return;}
        if(adapter.stats().slabs.classes[0].capacity_slots>2)continue;   //the eviction below needs exactly two
        for(size_t key:{size_t(0),size_t(1)}){
            Adapter::Write w;REQUIRE(adapter.reserve_packed(key,0,&w)==Adapter::Status::Ok);
            std::memset(w.host(),int(0x40+key),packed[0]);
            Adapter::Lease l;REQUIRE(w.publish(&l)==Adapter::Status::Ok);CHECK(l.release()==Adapter::Status::Ok);
        }
        const uint64_t hits0=adapter.stats().cache.hits;
        Adapter::Lease h;
        REQUIRE((touch?adapter.borrow_packed(0,0,&h):adapter.borrow_packed_readonly(0,0,&h))==Adapter::Status::Hit);
        CHECK(h.packed());CHECK_EQ(h.bytes(),packed[0]);CHECK(h.host()[0]==0x40);
        CHECK(h.release()==Adapter::Status::Ok);
        CHECK_EQ(adapter.stats().cache.hits,hits0+1);
        /* A third entry evicts the LRU tail: key 0 after a read-only borrow, key 1 after a touching one. */
        Adapter::Write w;REQUIRE(adapter.reserve_packed(2,0,&w)==Adapter::Status::Ok);
        Adapter::Lease l;REQUIRE(w.publish(&l)==Adapter::Status::Ok);CHECK(l.release()==Adapter::Status::Ok);
        CHECK_EQ(adapter.present(0),touch==1);
        CHECK_EQ(adapter.present(1),touch==0);
        Adapter::Lease cls;CHECK(adapter.borrow_packed_readonly(2,2,&cls)==Adapter::Status::Invalid);
        need(adapter.clear(),"packed terminal cleanup");
    }
    Adapter block;need(block.configure(22*MiB,4,0),"block one-pair slab");
    Adapter::Lease bl;CHECK(block.borrow_packed_readonly(0,0,&bl)==Adapter::Status::Layout);
    need(block.clear(),"block terminal cleanup");
}
TEST(mapped_slab_real_HOST_adapter_alias_failure_is_owned_until_terminal_cleanup){
    for(const char* flag:{static_cast<const char*>(nullptr),"0","true","01"}){
        Environment off("1",flag);Adapter a;need(a.configure(22*MiB,4,0),"slabs default off");CHECK(!a.stats().slab_policy);
    }
    Environment environment("1","1");Adapter adapter;need(adapter.configure(22*MiB,4,0),"slab failed alias owner fixture");
    const auto gu=adapter_pattern(Adapter::GU,14001);AdapterTest::reject_alias(adapter,true);
    CHECK(adapter.insert(0,gu.data(),gu.size())==Adapter::Status::AllocatorContract);auto failed=adapter.stats();
    CHECK(failed.cache.allocator_contract_broken);CHECK_EQ(failed.cache.resident_entries,size_t(0));
    CHECK_EQ(failed.slabs.retained_slab_owners,size_t(1));CHECK_EQ(failed.api.alias_rejections,uint64_t(1));
    CHECK_EQ(failed.api.free_invocations,uint64_t(0));CHECK(failed.cache.payload_charge>0);
    Adapter::Lease invisible;CHECK(adapter.borrow(0,gu.size(),&invisible)==Adapter::Status::AllocatorContract);
    AdapterTest::reject_alias(adapter,false);need(adapter.clear(),"malformed independent owner retired once");
    CHECK_EQ(adapter.stats().api.free_invocations,uint64_t(1));CHECK_EQ(adapter.stats().slabs.terminal_free_invoked_owners,size_t(1));
}

namespace {
void slab_readonly_hot_cold_case(bool relaid){
    Case c(relaid,22*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]},b{c.units[1]};
    const auto RO=NativeFileCacheReadPolicy::ReadOnly;
    c.stage(a);c.check(a);const auto initial=Access::stats(*c.transport.mover);
    REQUIRE(initial.slab_policy);REQUIRE_EQ(initial.slabs.classes[0].capacity_slots,size_t(1));
    REQUIRE_EQ(initial.slabs.classes[1].capacity_slots,size_t(1));
    const auto callbacks=c.calls.load();const auto warm0=Io::capture();c.stage(a,0,RO);const auto warm1=Io::capture();
    CHECK_EQ(delta(warm0,warm1).calls,uint64_t(0));CHECK_EQ(c.calls.load(),callbacks);
    CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    REQUIRE(c.driver.copies.size()>=2);CHECK(c.driver.copies[c.driver.copies.size()-1].mapped);
    CHECK(c.driver.copies[c.driver.copies.size()-2].mapped);CHECK_EQ(Access::clear(*c.transport.mover),RAD_E_STATE);
    //Both real source leases stay live while a different RO expert reads into the
    //other pinned window. No insertion, allocator call or GPU-duration mutex wait.
    const auto miss0=Io::capture();c.stage(b,1,RO);const auto miss1=Io::capture();
    CHECK(delta(miss0,miss1).calls>=2);CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    CHECK_EQ(c.driver.pending(),size_t(4));c.check(b);c.check(a);
    const auto again0=Io::capture();c.stage(b,1,RO);const auto again1=Io::capture();
    CHECK(delta(again0,again1).calls>=2);c.check(b);
    const auto readonly=Access::stats(*c.transport.mover);
    CHECK_EQ(readonly.cache.inserts,initial.cache.inserts);CHECK_EQ(readonly.cache.evictions,initial.cache.evictions);
    CHECK_EQ(readonly.allocations,initial.allocations);CHECK_EQ(readonly.public_free_invocations,uint64_t(0));
    CHECK_EQ(readonly.cache.payload_charge,initial.cache.payload_charge);
    CHECK_EQ(readonly.cache.hits,initial.cache.hits+2);CHECK_EQ(readonly.cache.misses,initial.cache.misses+4);
    CHECK_EQ(c.calls.load(),callbacks+(relaid?4:0));
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_hits,uint64_t(2));
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_misses,uint64_t(4));
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_hit_bytes,uint64_t(18800640));
    CHECK_EQ(c.transport.mover->stats().file_prefill_ro_miss_bytes,uint64_t(37601280));
    const auto a0=Io::capture();c.stage(a);const auto a1=Io::capture();
    CHECK_EQ(delta(a0,a1).calls,uint64_t(0));c.check(a); //RO B did not replace A.
    const auto b0=Io::capture();c.stage(b);const auto b1=Io::capture();
    CHECK(delta(b0,b1).calls>=2);c.check(b); //Normal still admits and evicts both A shapes.
    const auto cold0=Io::capture();c.stage(a,0,RO);const auto cold1=Io::capture();
    CHECK(delta(cold0,cold1).calls>=2);c.check(a);
    const auto hot0=Io::capture();c.stage(b,1,RO);const auto hot1=Io::capture();
    CHECK_EQ(delta(hot0,hot1).calls,uint64_t(0));c.check(b); //RO miss A left B resident.
    const auto normal0=Io::capture();c.stage(a);const auto normal1=Io::capture();
    CHECK(delta(normal0,normal1).calls>=2);c.check(a);
    const auto final=Access::stats(*c.transport.mover);
    CHECK_EQ(final.cache.evictions,initial.cache.evictions+4);CHECK_EQ(final.cache.inserts,initial.cache.inserts+4);
    CHECK_EQ(final.allocations,uint64_t(1));CHECK_EQ(final.api.free_invocations,uint64_t(0));
    CHECK(has(c.transport.mover->report(),"retained-slab hit lease without LRU touch"));
}
} //namespace
TEST(mapped_slab_ReadOnly_identity_warm_lease_and_cold_miss_never_admits){slab_readonly_hot_cold_case(false);}
TEST(mapped_slab_ReadOnly_relayout_warm_lease_and_cold_miss_never_admits){slab_readonly_hot_cold_case(true);}

TEST(mapped_slab_ReadOnly_warm_copy_record_failure_retains_until_checked_drain){
    Case c(true,22*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};c.stage(a);c.check(a);
    const auto RO=NativeFileCacheReadPolicy::ReadOnly;const auto before=Access::stats(*c.transport.mover);
    c.driver.fail_record=true;std::vector<int32_t>staged;
    CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged,nullptr,nullptr,0,nullptr,RO),RAD_E_STATE);
    CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    const int window=Access::last_window(*c.transport.mover);const auto attempts=c.driver.copy_attempts;
    c.driver.fail_drain=true;CHECK_EQ(Access::retire(*c.transport.mover,window),RAD_E_STATE);
    Access::next_window(*c.transport.mover,window);const auto io0=Io::capture();
    CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged,nullptr,nullptr,0,nullptr,RO),RAD_E_STATE);
    const auto io1=Io::capture();CHECK_EQ(delta(io0,io1).calls,uint64_t(0));
    CHECK_EQ(c.driver.copy_attempts,attempts);CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
    CHECK_EQ(Access::clear(*c.transport.mover),RAD_E_STATE);
    for(auto h:c.plan.units[size_t(a[0])].weights)CHECK(c.transport.mover->residency()->lookup(h)->ptr==nullptr);
    c.driver.fail_drain=c.driver.fail_record=false;need(Access::retire(*c.transport.mover,window),"RO source copies drained before release");
    CHECK_EQ(c.driver.pending(),size_t(0));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));
    c.stage(a,0,RO);c.check(a);const auto after=Access::stats(*c.transport.mover);
    CHECK_EQ(after.allocations,before.allocations);CHECK_EQ(after.cache.evictions,before.cache.evictions);
    CHECK_EQ(after.cache.inserts,before.cache.inserts);CHECK_EQ(after.public_free_invocations,uint64_t(0));
}

TEST(mapped_slab_ReadOnly_changed_file_epoch_refuses_before_borrow_IO_copy_or_publish){
    Case c(false,22*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};c.stage(a);c.check(a);
    REQUIRE_EQ(::ftruncate(c.image->fd,off_t(c.image->bytes.size()+1)),0);REQUIRE_EQ(::fsync(c.image->fd),0);
    const auto before=Access::stats(*c.transport.mover);const auto attempts=c.driver.copy_attempts;
    const auto h2d=c.transport.mover->stats().h2d_bytes;std::vector<int32_t>staged;const auto io0=Io::capture();
    CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),
        c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&staged,nullptr,nullptr,0,nullptr,NativeFileCacheReadPolicy::ReadOnly),RAD_E_STATE);
    const auto io1=Io::capture();const auto after=Access::stats(*c.transport.mover);
    CHECK_EQ(delta(io0,io1).calls,uint64_t(0));CHECK_EQ(c.driver.copy_attempts,attempts);CHECK_EQ(c.driver.pending(),size_t(0));
    CHECK_EQ(after.cache.hits,before.cache.hits);CHECK_EQ(after.cache.misses,before.cache.misses);
    CHECK_EQ(after.cache.lease_refs,size_t(0));CHECK_EQ(c.transport.mover->stats().h2d_bytes,h2d);
    for(auto h:c.plan.units[size_t(a[0])].weights)CHECK(c.transport.mover->residency()->lookup(h)->ptr==nullptr);
}

namespace {
/* The packed RAM cache through the actual HOST Mover: weights laid out as the release's MXFP4 planes
 * (codes [N, K/2], then e8m0 scales [N, K/32]), relaid on a read into block_mxfp4 as the kernel
 * does. Scales in the release's mix, except expert 1's down: one block in 500 outside the window,
 * more raw rounds than the default down budget (320). */
struct PackedEnvironment {
    bool had=false;std::string old;
    PackedEnvironment(){const auto*p=std::getenv("RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE");had=p;old=p?p:"";::setenv("RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE","1",1);}
    ~PackedEnvironment(){if(had)::setenv("RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE",old.c_str(),1);else::unsetenv("RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE");}
};
int64_t mx_rows(const WeightInfo& w){return w.stored_bytes==int64_t(NativeMappedFileCache::GU)?4608:5120;}
int64_t mx_cols(const WeightInfo& w){return w.stored_bytes==int64_t(NativeMappedFileCache::GU)?5120:2304;}
/* block_mxfp4 from planes: block u of row n is its e8m0, then value j < 16 in the low nibble of byte
 * j and j + 16 in its high one; a planes byte p holds values 2p (low) and 2p + 1 (high). */
void mx_blocks(const uint8_t* codes,const uint8_t* scales,int64_t N,int64_t K,uint8_t* out){
    const int64_t UK=K/32;
    for(int64_t n=0;n<N;++n)for(int64_t u=0;u<UK;++u){
        const uint8_t* c=codes+n*(K/2)+u*16;uint8_t* b=out+(n*UK+u)*17;b[0]=scales[n*UK+u];
        auto v=[&](int j){return j&1?c[j>>1]>>4:c[j>>1]&15;};
        for(int j=0;j<16;++j)b[1+j]=uint8_t(v(j)|(v(j+16)<<4));
    }
}
int mx_relayout(const RadParam*,int,int,const RadEncoding*,const int*,const RadTensor* p,int n,void* dst,int64_t cap){
    if(n!=2||!p||!dst||!p[0].data||!p[1].data)return RAD_E_INVAL;
    const int64_t N=p[0].shape[0],K=p[0].shape[1];
    if(K%32||p[1].shape[0]!=N||p[1].shape[1]!=K/32||cap!=N*(K/32)*17)return RAD_E_SHAPE;
    mx_blocks(static_cast<const uint8_t*>(p[0].data),static_cast<const uint8_t*>(p[1].data),N,K,static_cast<uint8_t*>(dst));
    return RAD_OK;
}
int64_t lay_out_mxfp4(Fixture& f){
    int64_t off=4*int64_t(RAD_ALIGN_UNIT);
    for(size_t h=1;h<f.prog.weights.size();++h){auto& w=f.prog.weights[h];w.file_offset=uint64_t(off);
        if(w.decl.group.expert<0){off=align_up(off+w.stored_bytes,RAD_ALIGN_UNIT);continue;}
        const int64_t N=mx_rows(w),K=mx_cols(w);
        w.identity=false;w.layout_tag="ggml_mxfp4";w.lay_op=w.first_use_op;
        w.lay_band=0;w.lay_dom=RAD_DOMAIN_DEVICE;w.lay_operand=0;
        w.enc.n_planes=2;w.enc.plane[0].dtype=RAD_FP4E2M1;w.enc.plane[1].dtype=RAD_E8M0;
        w.n_sel=2;w.sel[0]=0;w.sel[1]=1;w.sel_rows[0]=w.sel_rows[1]=N;w.sel_cols[0]=K;w.sel_cols[1]=K/32;
        w.file_plane_offset[0]=uint64_t(off);w.file_plane_offset[1]=uint64_t(off+N*K/2);
        off=align_up(off+N*K/2+N*(K/32),RAD_ALIGN_UNIT);
    }
    return off+RAD_ALIGN_UNIT;
}
/* Writes the scale planes' bytes into the image (and its file): the release's mix, one block in
 * `rare` outside the window. */
void write_scales(const Fixture& f,Image& image){
    for(size_t h=1;h<f.prog.weights.size();++h){const auto& w=f.prog.weights[h];if(w.decl.group.expert<0)continue;
        const int64_t n=mx_rows(w)*(mx_cols(w)/32);
        const bool over=w.decl.group.expert==1&&w.stored_bytes==int64_t(NativeMappedFileCache::DN);
        uint64_t s=0x9e3779b97f4a7c15ull*h+1;uint8_t* sc=image.bytes.data()+w.file_plane_offset[1];
        for(int64_t i=0;i<n;++i){s^=s<<13;s^=s>>7;s^=s<<17;const int p=int(s%100);
            sc[i]=uint8_t(p<2?119:p<50?120:p<98?121:122);
            if((s>>32)%(over?500u:5000u)==0)sc[i]=uint8_t((s>>8)&1?112:126);}
        if(::pwrite(image.fd,sc,size_t(n),off_t(w.file_plane_offset[1]))!=ssize_t(n))throw std::runtime_error("scale plane write");
    }
    if(::fsync(image.fd))throw std::runtime_error("fsync scale planes");
}
std::vector<uint8_t> image_blocks(const WeightInfo& w,const Image& image){
    std::vector<uint8_t> out(size_t(w.stored_bytes));
    mx_blocks(image.bytes.data()+w.file_plane_offset[0],image.bytes.data()+w.file_plane_offset[1],mx_rows(w),mx_cols(w),out.data());
    return out;
}
} //namespace
/* insert_unit_planes packs a missed unit into the packed cache, and borrow_host_unit lends packed
 * slots whose scales widen back to the container's and whose codes are its codes plane; a unit with
 * a weight over its raw-round budget stays out whole (counted once a unit); the card's copy path
 * reads the container, publishes the block_mxfp4 bytes, and neither reads nor fills the cache. */
TEST(mapped_packed_actual_mover_inserts_lends_and_keeps_the_card_out){
    Environment mapped("1","1");PackedEnvironment packed;
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    REQUIRE(t.mover->packed_cache());
    for(int32_t u:units)REQUIRE(t.mover->host_planes_eligible(u));
    Mover::HostPlanes p0,p1;p0.unit=units[0];p1.unit=units[1];
    Mover::HostPlanes* ps[2]={&p0,&p1};
    REQUIRE(t.mover->read_unit_planes(ps,2,1));
    std::vector<uint8_t> scratch;
    for(Mover::HostPlanes* p:ps){
        const bool over=plan.units[size_t(p->unit)].expert==1;
        CHECK_EQ(t.mover->insert_unit_planes(*p,&scratch),RAD_OK);
        CHECK_EQ(t.mover->unit_cached(p->unit),!over);
        Mover::HostUnit hu;const bool lent=t.mover->borrow_host_unit(p->unit,&hu);CHECK_EQ(lent,!over);
        if(!lent)continue;
        CHECK(hu.packed);REQUIRE_EQ(hu.n,2);
        for(int k=0;k<2;++k){
            const auto& w=f.prog.weights[p->w[k]];const int N=int(mx_rows(w)),K=int(mx_cols(w));
            CHECK_EQ(hu.bytes[k],w.stored_bytes);CHECK(hu.slot[k]<w.stored_bytes);
            ds41hx::Mx v;REQUIRE(ds41hx::packed_view(hu.host[k],size_t(hu.slot[k]),N,K,&v));
            CHECK(v.n_exc>0);CHECK(ds41hx::packed_scales_equal(v,image.bytes.data()+w.file_plane_offset[1]));
            CHECK(std::memcmp(v.codes,image.bytes.data()+w.file_plane_offset[0],size_t(N)*size_t(K/2))==0);
        }
        t.mover->release_host_unit(&hu);
    }
    CHECK_EQ(t.mover->insert_unit_planes(plan.units[size_t(units[0])].expert==1?p0:p1,&scratch),RAD_OK);
    const auto st=t.mover->packed_stats();
    CHECK_EQ(st.inserted,uint64_t(2));CHECK_EQ(st.over_budget,uint64_t(2));CHECK_EQ(st.over_budget_units,uint64_t(1));
    CHECK_EQ(st.round_trip_failed,uint64_t(0));
    /* The card's path for the cached unit: the container's bytes relaid, the cache untouched. */
    const int32_t cached=plan.units[size_t(units[0])].expert==1?units[1]:units[0];
    const auto before=Access::stats(*t.mover);std::vector<int32_t> staged;
    need(t.mover->stage_file_units({cached},t.mover->file_buffer(0),t.mover->file_buffer_bytes(),nullptr,t.done,&staged),"card path over a packed cache");
    REQUIRE(staged==std::vector<int32_t>{cached});
    need(Access::drain(*t.mover),"the card's copies complete");
    for(auto h:plan.units[size_t(cached)].weights){
        const auto* s=t.mover->residency()->lookup(h);REQUIRE(s&&s->ptr);
        if(s->ready)need(rad_event_wait(t.compute,s->ready),"compute follows the ready event");
        const auto want=image_blocks(f.prog.weights[h],image);std::vector<uint8_t> got(want.size());
        need(rad_memcpy_async(got.data(),s->ptr,int64_t(got.size()),t.compute),"readback");need(rad_stream_sync(t.compute),"readback done");
        CHECK(got==want);
    }
    t.drain();t.mover->unstage_units(staged);
    const auto after=Access::stats(*t.mover);
    CHECK_EQ(after.layout_refusals-before.layout_refusals,uint64_t(4));   //a borrow and an insert a weight
    CHECK_EQ(after.cache.inserts,before.cache.inserts);CHECK_EQ(after.cache.hits,before.cache.hits);
    CHECK_EQ(after.cache.lease_refs,size_t(0));CHECK_EQ(t.mover->stats().mapped_source_copies,uint64_t(0));
    CHECK(has(t.mover->report(),"packed RAM cache"));
}

/* RADIANCE_EXPERIMENTAL_DS41_INSERT_NT / _SLICE_KIB (the background inserts' ds41hx::PackCopy): a unit
 * inserted with its codes planes written by non-temporal stores in 4 KiB pieces lends slots holding
 * every byte one memcpy's pack writes there (the codes, header, 2-bit codes and raw rounds), and the
 * hook ran once a piece, for every byte of both codes planes. */
TEST(mapped_packed_insert_by_streaming_stores_in_pieces_writes_the_same_slots){
    Environment mapped("1","1");PackedEnvironment packed;
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    REQUIRE(t.mover->packed_cache());
    const int32_t u=plan.units[size_t(units[0])].expert==1?units[1]:units[0];   //write_scales: expert 1's down is over budget
    Mover::HostPlanes p;p.unit=u;Mover::HostPlanes* pp=&p;
    REQUIRE(t.mover->read_unit_planes(&pp,1,1));REQUIRE_EQ(p.n,2);
    struct Count{size_t calls=0,bytes=0;};Count cnt;
    ds41hx::PackCopy cp;cp.nt=true;cp.slice=4096;cp.ctx=&cnt;
    cp.before=+[](void* c,size_t n){Count* k=static_cast<Count*>(c);++k->calls;k->bytes+=n;};
    std::vector<uint8_t> scratch;
    CHECK_EQ(t.mover->insert_unit_planes(p,&scratch,&cp),RAD_OK);
    REQUIRE(t.mover->unit_cached(u));
    Mover::HostUnit hu;REQUIRE(t.mover->borrow_host_unit(u,&hu));REQUIRE(hu.packed);REQUIRE_EQ(hu.n,2);
    size_t codes=0,pieces=0;
    for(int k=0;k<2;++k){
        const auto& w=f.prog.weights[p.w[k]];const int N=int(mx_rows(w)),K=int(mx_cols(w));
        const int cap=w.stored_bytes==int64_t(NativeMappedFileCache::GU)?512:320;   //the default budgets
        const uint8_t* const c=image.bytes.data()+w.file_plane_offset[0];const uint8_t* const sc=image.bytes.data()+w.file_plane_offset[1];
        ds41hx::PackInfo pi;REQUIRE(ds41hx::pack_plan(sc,N,K,&pi));
        std::vector<uint8_t> want(ds41hx::packed_bytes(N,K,cap),0xA5);
        REQUIRE_EQ(ds41hx::pack_planes(c,sc,N,K,cap,pi,want.data(),want.size()),ds41hx::kPackOk);
        REQUIRE(hu.slot[k]>=int64_t(want.size()));
        CHECK(std::memcmp(hu.host[k],want.data(),want.size())==0);
        codes+=size_t(N)*size_t(K/2);pieces+=(size_t(N)*size_t(K/2)+4095)/4096;
    }
    t.mover->release_host_unit(&hu);
    CHECK_EQ(cnt.bytes,codes);CHECK_EQ(cnt.calls,pieces);
    CHECK_EQ(t.mover->packed_stats().inserted,uint64_t(2));CHECK_EQ(t.mover->packed_stats().round_trip_failed,uint64_t(0));
    t.drain();
}

/* Read-ahead in pieces (place/readahead.h) through the actual HOST Mover: lay_out_unit_planes lays
 * a unit out exactly as read_unit_planes does, its pieces read through read_piece (the container,
 * O_DIRECT, 64 KiB pieces, three readers, a demand read holding them back midway) leave the same
 * codes and scales planes as the demand read, and those planes insert into the RAM cache as the
 * demand read's would -- packed or not. */
namespace {
void read_ahead_case(bool packed){
    Environment mapped("1","1");std::unique_ptr<PackedEnvironment> pk;if(packed)pk=std::make_unique<PackedEnvironment>();
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    REQUIRE_EQ(t.mover->packed_cache(),packed);
    auto read=[](void* m,const PlaneExtent& p){return static_cast<Mover*>(m)->read_piece(p.file,p.off,p.len,p.dst,p.cap);};
    PieceReader r;REQUIRE(r.start(3,64*KiB));
    for(int32_t u:units){
        REQUIRE(t.mover->host_planes_eligible(u));
        Mover::HostPlanes demand,ahead;demand.unit=ahead.unit=u;Mover::HostPlanes* pd=&demand;
        REQUIRE(t.mover->read_unit_planes(&pd,1,1));
        PlaneExtent ext[2];REQUIRE(t.mover->lay_out_unit_planes(&ahead,ext));
        REQUIRE_EQ(ahead.n,2);
        for(int k=0;k<2;++k){
            CHECK_EQ(ahead.w[k],demand.w[k]);CHECK_EQ(ahead.rows[k],demand.rows[k]);CHECK_EQ(ahead.cols[k],demand.cols[k]);
            CHECK_EQ(ahead.a0[k],demand.a0[k]);CHECK_EQ(ahead.len[k],demand.len[k]);
            CHECK_EQ(ext[k].off,ahead.a0[k]);CHECK_EQ(ext[k].len,ahead.len[k]);CHECK(ext[k].dst==ahead.buf[k]);CHECK_EQ(ext[k].cap,ahead.cap[k]);
            CHECK(ahead.codes[k]-ahead.buf[k]==demand.codes[k]-demand.buf[k]);
            CHECK(ahead.scales[k]-ahead.buf[k]==demand.scales[k]-demand.buf[k]);
        }
        auto ra=std::make_shared<ReadAheadUnit>(&t,1,ext,2,64*KiB,+read,t.mover.get(),nullptr);
        r.demand_enter();r.submit(ra);
        CHECK_EQ(ra->started(),0);           /* held while the demand read is in flight */
        r.demand_leave();r.wait(*ra);
        REQUIRE(ra->whole());CHECK_EQ(ra->in(0),ra->pieces(0));CHECK_EQ(ra->in(1),ra->pieces(1));
        CHECK(t.mover->source_epochs_current());
        for(int k=0;k<2;++k){
            const auto& w=f.prog.weights[ahead.w[k]];const size_t codes=size_t(ahead.rows[k]*ahead.cols[k]/2),scales=size_t(ahead.rows[k]*ahead.cols[k]/32);
            CHECK(std::memcmp(ahead.codes[k],demand.codes[k],codes)==0);CHECK(std::memcmp(ahead.scales[k],demand.scales[k],scales)==0);
            CHECK(std::memcmp(ahead.codes[k],image.bytes.data()+w.file_plane_offset[0],codes)==0);
            CHECK(std::memcmp(ahead.scales[k],image.bytes.data()+w.file_plane_offset[1],scales)==0);
        }
        std::vector<uint8_t> scratch;
        CHECK_EQ(t.mover->insert_unit_planes(ahead,&scratch),RAD_OK);
        const bool over=packed&&plan.units[size_t(u)].expert==1;   /* write_scales: over the down budget */
        CHECK_EQ(t.mover->unit_cached(u),!over);
        Mover::HostUnit hu;const bool lent=t.mover->borrow_host_unit(u,&hu);CHECK_EQ(lent,!over);
        if(lent)t.mover->release_host_unit(&hu);
    }
    CHECK_EQ(r.stats().failed,uint64_t(0));
    CHECK_EQ(t.mover->read_piece(0,0,0,image.bytes.data(),4096),RAD_E_INVAL);
    t.drain();
}
} //namespace
TEST(mapped_read_ahead_pieces_read_and_insert_as_the_demand_read){read_ahead_case(false);}
TEST(mapped_packed_read_ahead_pieces_read_and_insert_as_the_demand_read){read_ahead_case(true);}
/* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW through the actual HOST Mover and the file stager's
 * read-ahead parts (stager.h's test friend): a multi-row post's union is planned as a one-token
 * post's list is -- this rank's uncached file-tier units of the layer, laid out by the mover and read
 * in pieces from the container, an id the rank does not own or one naming no file-tier unit passed
 * over, a unit already live not planned twice, the live cap and the too-late rule kept -- each unit
 * tagged with the post's rows and queued in the union's order; the planes a whole unit hands over are
 * the demand read's; planned and skipped units count under the post's rows, the deciding pass's
 * under its own. */
namespace rad {
struct EpCountsHostProbe {
    PieceReader r;   /* outlives the stager, whose destructor forgets its units on it */
    FileStager fs;
    void side_store_holds_what_the_packed_cache_refuses();   /* defined below */
    void multirow_union(Mover* m,const Plan& plan,const std::vector<int32_t>& units){
        REQUIRE(r.start(2,64*KiB));
        const int32_t layer=plan.units[size_t(units[0])].layer;
        const int32_t e0=plan.units[size_t(units[0])].expert,e1=plan.units[size_t(units[1])].expert;
        REQUIRE(e0>=0&&e0<96&&e1>=0&&e1<96&&layer>=0&&layer<RouterPredictor::kLayers);
        fs.plan_=&plan;fs.mover_=m;fs.hx_ra_=&r;fs.hx_ra_cap_=8;
        fs.hx_ra_li_.assign(RouterPredictor::kLayers,-1);fs.hx_ra_li_[size_t(layer)]=0;
        fs.hx_unit_of_.assign(1,{});fs.hx_unit_of_[0].fill(-1);
        fs.hx_unit_of_[0][size_t(e0)]=units[0];fs.hx_unit_of_[0][size_t(e1)]=units[1];
        fs.hx_ra_ok_.assign(plan.units.size(),0);
        for(int32_t u:units){REQUIRE(m->host_planes_eligible(u));fs.hx_ra_ok_[size_t(u)]=1;}
        const auto& by=fs.hx_stats_.ra_rows;const auto& hx=fs.hx_stats_;
        /* Pass 5, a 2-row verify pass: the union of row 0's {e1, ..} and row 1's {e0, ..}, with 200
         * (not this rank's: it owns 0..95) and 7 (no file-tier unit) among them. */
        const int32_t un[4]={e1,200,e0,7};
        fs.trace_pass_=5;fs.pass_tok_=2;fs.hx_ra_at_.store(uint64_t(5)<<32);
        fs.hx_ra_plan(5,layer,un,4,2);
        REQUIRE_EQ(fs.hx_ra_live_.size(),size_t(2));
        CHECK_EQ(fs.hx_ra_live_[0].unit,units[1]);CHECK_EQ(fs.hx_ra_live_[1].unit,units[0]);
        for(const auto& e:fs.hx_ra_live_){CHECK_EQ(e.rows,2);CHECK_EQ(e.pass,uint32_t(5));CHECK_EQ(e.li,0);}
        CHECK(fs.hx_ra_live_[0].u->order()<fs.hx_ra_live_[1].u->order());
        CHECK_EQ(by[2].planned,uint64_t(2));CHECK_EQ(by[1].planned,uint64_t(0));CHECK_EQ(hx.ra_planned,uint64_t(2));
        fs.hx_ra_plan(5,layer,un,4,2);   /* live already: nothing twice */
        CHECK_EQ(fs.hx_ra_live_.size(),size_t(2));CHECK_EQ(by[2].planned,uint64_t(2));
        for(const auto& e:fs.hx_ra_live_){r.wait(*e.u);CHECK(e.u->whole());}
        CHECK_EQ(r.stats().failed,uint64_t(0));
        /* The pass decides routed layer 0: e1's unit selected and missed, e0's not selected. */
        fs.hx_ra_begin(0);
        REQUIRE_EQ(fs.hx_ra_here_.size(),size_t(2));
        {
            bool ready=false;std::shared_ptr<ReadAheadUnit> urgent;
            auto p=fs.hx_ra_take(units[1],&ready,&urgent);
            REQUIRE(p!=nullptr);CHECK(ready);CHECK(!urgent);REQUIRE_EQ(p->n,2);
            Mover::HostPlanes demand;demand.unit=units[1];Mover::HostPlanes* pd=&demand;
            REQUIRE(m->read_unit_planes(&pd,1,2));
            for(int k=0;k<2;++k){
                const size_t codes=size_t(p->rows[k]*p->cols[k]/2),scales=size_t(p->rows[k]*p->cols[k]/32);
                CHECK(std::memcmp(p->codes[k],demand.codes[k],codes)==0);
                CHECK(std::memcmp(p->scales[k],demand.scales[k],scales)==0);
            }
        }
        for(auto& e:fs.hx_ra_here_)fs.hx_ra_drop(e,false);
        fs.hx_ra_here_.clear();
        CHECK_EQ(by[2].misses,uint64_t(1));CHECK_EQ(by[2].miss_planned,uint64_t(1));CHECK_EQ(by[2].whole,uint64_t(1));
        CHECK_EQ(by[2].dropped,uint64_t(1));CHECK(by[2].used_bytes>0);CHECK(by[2].dropped_bytes>0);
        /* Pass 6, a 3-row post with room for one live unit: one planned, the next past the cap. */
        fs.hx_ra_cap_=1;fs.trace_pass_=6;fs.pass_tok_=3;fs.hx_ra_at_.store(uint64_t(6)<<32);
        fs.hx_ra_plan(6,layer,un,4,3);
        CHECK_EQ(by[3].planned,uint64_t(1));CHECK_EQ(by[3].skipped,uint64_t(1));CHECK_EQ(hx.ra_capped,uint64_t(1));
        fs.hx_ra_drop_stale(7,-1);   /* the next pass begins: pass 6's unit is stale */
        CHECK(fs.hx_ra_live_.empty());CHECK_EQ(by[3].dropped,uint64_t(1));CHECK_EQ(hx.ra_stale,uint64_t(1));
        /* Too late: this rank has decided routed layer 0 of pass 8 already. */
        fs.hx_ra_cap_=8;fs.hx_ra_at_.store((uint64_t(8)<<32)|1u);
        fs.hx_ra_plan(8,layer,un,4,4);
        CHECK_EQ(by[4].planned,uint64_t(0));CHECK_EQ(by[4].skipped,uint64_t(1));CHECK_EQ(hx.ra_late,uint64_t(1));
        /* A one-row post (rows by default) counts under 1 row. */
        const int32_t one[2]={e0,e1};
        fs.hx_ra_at_.store(uint64_t(9)<<32);
        fs.hx_ra_plan(9,layer,one,2);
        CHECK_EQ(by[1].planned,uint64_t(2));
        fs.hx_ra_drop_stale(10,-1);
        CHECK_EQ(by[1].dropped,uint64_t(2));CHECK_EQ(hx.ra_planned,uint64_t(5));
        /* No plane mirror here: the finishing rule has no share to beat at any pass size. */
        CHECK_EQ(m->mirror_bytes(1),m->decode_mirror_bytes());CHECK_EQ(m->mirror_bytes(2),int64_t(0));
        r.forget(&fs);
    }
};
} //namespace rad
TEST(mapped_read_ahead_multirow_union_through_the_file_stager){
    Environment mapped("1","1");
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    {rad::EpCountsHostProbe p;p.multirow_union(t.mover.get(),plan,units);}
    t.drain();
}
/* read_unit_planes and its two halves (start_unit_planes / finish_unit_planes, which
 * RADIANCE_EXPERIMENTAL_DS41_MISS_POST uses) land the container's planes byte for byte and report the
 * call's timing; a second start while one is in flight (before it touches any buffer) and a finish
 * with none are refused; the kept readers serve both ways in turn. */
TEST(mapped_actual_mover_unit_planes_read_whole_or_posted_land_the_same_bytes){
    Environment mapped("1","1");
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    for(int32_t u:units)REQUIRE(t.mover->host_planes_eligible(u));
    auto same=[&](const Mover::HostPlanes& p){
        if(p.n!=2)return false;
        for(int k=0;k<2;++k){const auto& w=f.prog.weights[p.w[k]];const size_t N=size_t(mx_rows(w)),K=size_t(mx_cols(w));
            if(std::memcmp(p.codes[k],image.bytes.data()+w.file_plane_offset[0],N*K/2)||
               std::memcmp(p.scales[k],image.bytes.data()+w.file_plane_offset[1],N*(K/32)))return false;}
        return true;};
    for(int round=0;round<3;++round){
        Mover::HostPlanes a0,a1;a0.unit=units[0];a1.unit=units[1];Mover::HostPlanes* pa[2]={&a0,&a1};
        Mover::PlaneReadTiming ta;
        REQUIRE(t.mover->read_unit_planes(pa,2,1,&ta));
        CHECK(same(a0));CHECK(same(a1));
        CHECK_EQ(ta.units,2);CHECK_EQ(ta.reads,4);CHECK_EQ(ta.mirror_reads,0);CHECK(!ta.posted);
        CHECK(ta.container_ns>0);CHECK(ta.return_ns>=ta.container_ns);CHECK(ta.last_start_ns>=ta.first_ns);
        Mover::HostPlanes b0,b1;b0.unit=units[0];b1.unit=units[1];Mover::HostPlanes* pb[2]={&b0,&b1};
        REQUIRE(t.mover->start_unit_planes(pb,2,1));
        CHECK(t.mover->unit_planes_started());
        Mover::HostPlanes c0;c0.unit=units[0];Mover::HostPlanes* pc[1]={&c0};
        CHECK(!t.mover->start_unit_planes(pc,1,1));CHECK(c0.buf[0]==nullptr);   //one in flight
        Mover::PlaneReadTiming tb;
        REQUIRE(t.mover->finish_unit_planes(&tb));
        CHECK(!t.mover->unit_planes_started());
        CHECK(same(b0));CHECK(same(b1));
        CHECK(tb.posted);CHECK_EQ(tb.units,2);CHECK_EQ(tb.reads,4);CHECK(tb.return_ns>=tb.join_ns);
        CHECK_EQ(ta.rest,0);CHECK_EQ(tb.rest,0);CHECK_EQ(ta.rest_bytes,int64_t(0));
        CHECK(!t.mover->finish_unit_planes(&tb));   //none in flight
    }
    CHECK(has(t.mover->report(),"kept plane readers"));
}

/* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT through the actual HOST Mover: a unit laid out for the
 * read-ahead, its pieces read by the piece readers up to a stop (the last two held in flight), and
 * the rest of each extent read by read_unit_planes -- or posted, start_unit_planes and
 * finish_unit_planes -- with PlanesIn, beside a fresh unit in the same call, while the held pieces
 * land after it: the container's planes byte for byte, the unit never laid out again (its buffers
 * stay put), and only the rest read, counted in the call's timing. A unit not laid out, or bytes in
 * off the alignment, past their extent or negative, is refused before any read; a unit with every
 * byte in reads nothing. */
namespace {
struct HeldPieces {
    Mover* mover=nullptr;std::atomic<int> n{0};int hold=0;std::atomic<bool> go{false};
    static int read(void* c,const PlaneExtent& p){auto* h=static_cast<HeldPieces*>(c);
        if(h->n.fetch_add(1)>=h->hold)while(!h->go.load())std::this_thread::sleep_for(std::chrono::microseconds(50));
        return h->mover->read_piece(p.file,p.off,p.len,p.dst,p.cap);}
};
} //namespace
TEST(mapped_actual_mover_reads_a_read_ahead_units_rest_in_place){
    Environment mapped("1","1");
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    for(int32_t u:units)REQUIRE(t.mover->host_planes_eligible(u));
    auto same=[&](const Mover::HostPlanes& p){
        if(p.n!=2)return false;
        for(int k=0;k<2;++k){const auto& w=f.prog.weights[p.w[k]];const size_t N=size_t(mx_rows(w)),K=size_t(mx_cols(w));
            if(std::memcmp(p.codes[k],image.bytes.data()+w.file_plane_offset[0],N*K/2)||
               std::memcmp(p.scales[k],image.bytes.data()+w.file_plane_offset[1],N*(K/32)))return false;}
        return true;};
    for(int posted=0;posted<2;++posted){
        PieceReader r;REQUIRE(r.start(2,64*KiB));
        HeldPieces held;held.mover=t.mover.get();held.hold=5;   //pieces 5 and 6 wait in flight
        Mover::HostPlanes ahead;ahead.unit=units[0];PlaneExtent ext[2];REQUIRE(t.mover->lay_out_unit_planes(&ahead,ext));
        uint8_t* const bufs[2]={ahead.buf[0],ahead.buf[1]};
        auto ra=std::make_shared<ReadAheadUnit>(&t,1,ext,2,64*KiB,&HeldPieces::read,&held,nullptr);
        //However the case ends, no reader is left in `held` or the planes once they go.
        struct Settle{PieceReader& r;HeldPieces& h;std::shared_ptr<ReadAheadUnit>& u;
            ~Settle(){h.go=true;if(u){r.cancel(*u);r.wait(*u);}}} settle{r,held,ra};
        r.submit(ra);
        for(int i=0;i<50000&&!(held.n.load()==7&&ra->in()==5);++i)std::this_thread::sleep_for(std::chrono::microseconds(100));
        REQUIRE_EQ(ra->in(),5);REQUIRE_EQ(held.n.load(),7);
        REQUIRE_EQ(r.stop(*ra),7);
        Mover::PlanesIn in[2];in[0].bytes[0]=ra->front(0);in[0].bytes[1]=ra->front(1);
        CHECK_EQ(in[0].bytes[0],7*64*KiB);CHECK_EQ(in[0].bytes[1],int64_t(0));
        Mover::HostPlanes fresh;fresh.unit=units[1];Mover::HostPlanes* ps[2]={&ahead,&fresh};
        Mover::PlaneReadTiming tm;
        if(posted){REQUIRE(t.mover->start_unit_planes(ps,2,1,in));REQUIRE(t.mover->finish_unit_planes(&tm));CHECK(tm.posted);}
        else REQUIRE(t.mover->read_unit_planes(ps,2,1,&tm,in));
        CHECK(ahead.buf[0]==bufs[0]);CHECK(ahead.buf[1]==bufs[1]);
        CHECK_EQ(tm.units,2);CHECK_EQ(tm.reads,4);CHECK_EQ(tm.mirror_reads,0);CHECK_EQ(tm.rest,1);
        CHECK_EQ(tm.rest_bytes,ahead.len[0]+ahead.len[1]-in[0].bytes[0]);CHECK_EQ(tm.rest_mirror_bytes,int64_t(0));
        CHECK(!ra->whole());CHECK_EQ(ra->in(),5);   //the two held pieces land after the call
        held.go=true;r.wait(*ra);
        CHECK_EQ(ra->in(),7);CHECK(!ra->failed());
        CHECK(same(ahead));CHECK(same(fresh));
        CHECK(t.mover->source_epochs_current());
    }
    Mover::HostPlanes lay;lay.unit=units[0];PlaneExtent ext[2];REQUIRE(t.mover->lay_out_unit_planes(&lay,ext));
    Mover::HostPlanes* pl[1]={&lay};Mover::PlanesIn bad;
    bad.bytes[0]=64*KiB+100;CHECK(!t.mover->read_unit_planes(pl,1,1,nullptr,&bad));            //off the alignment
    bad.bytes[0]=0;bad.bytes[1]=lay.len[1]+4096;CHECK(!t.mover->read_unit_planes(pl,1,1,nullptr,&bad));   //past the extent
    bad.bytes[1]=-4096;CHECK(!t.mover->read_unit_planes(pl,1,1,nullptr,&bad));
    Mover::HostPlanes never;never.unit=units[0];Mover::HostPlanes* pn[1]={&never};Mover::PlanesIn some;some.bytes[0]=64*KiB;
    CHECK(!t.mover->read_unit_planes(pn,1,1,nullptr,&some));CHECK(never.buf[0]==nullptr);   //not laid out: never laid out here
    CHECK(!t.mover->start_unit_planes(pn,1,1,&some));CHECK(!t.mover->unit_planes_started());
    Mover::PlanesIn all;all.bytes[0]=lay.len[0];all.bytes[1]=lay.len[1];Mover::PlaneReadTiming tz;
    CHECK(t.mover->read_unit_planes(pl,1,1,&tz,&all));CHECK_EQ(tz.reads,0);CHECK_EQ(tz.rest,1);CHECK_EQ(tz.rest_bytes,int64_t(0));
    CHECK(!t.mover->start_unit_planes(pl,1,1,&all));   //nothing to post
}

/* read_unit_planes' `landed` (large-chunk prefill's streamed misses, FileStager::hx_stream_misses): each
 * unit handed over exactly once, by a reader, with its bytes in place when it is; the call's result and
 * bytes as without it. */
namespace {
struct Landed {
    std::atomic<int> calls{0};
    std::atomic<int> per[4]={};
    std::atomic<int> wrong{0};
    std::function<bool(int)> check;
};
void on_landed(void* ctx,int i){
    Landed& l=*(Landed*)ctx;
    l.calls.fetch_add(1);
    if(i<0||i>=4){l.wrong.fetch_add(1);return;}
    l.per[i].fetch_add(1);
    if(l.check&&!l.check(i))l.wrong.fetch_add(1);
}
}
TEST(mapped_actual_mover_unit_planes_landed_each_once_with_its_bytes){
    Environment mapped("1","1");
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    auto same=[&](const Mover::HostPlanes& p){
        if(p.n!=2)return false;
        for(int k=0;k<2;++k){const auto& w=f.prog.weights[p.w[k]];const size_t N=size_t(mx_rows(w)),K=size_t(mx_cols(w));
            if(std::memcmp(p.codes[k],image.bytes.data()+w.file_plane_offset[0],N*K/2)||
               std::memcmp(p.scales[k],image.bytes.data()+w.file_plane_offset[1],N*(K/32)))return false;}
        return true;};
    for(int round=0;round<3;++round){
        Mover::HostPlanes a0,a1;a0.unit=units[0];a1.unit=units[1];Mover::HostPlanes* pa[2]={&a0,&a1};
        Landed l;l.check=[&](int i){return same(*pa[i]);};
        Mover::PlaneReadTiming ta;
        REQUIRE(t.mover->read_unit_planes(pa,2,64,&ta,nullptr,false,nullptr,on_landed,&l));
        CHECK_EQ(l.calls.load(),2);CHECK_EQ(l.per[0].load(),1);CHECK_EQ(l.per[1].load(),1);CHECK_EQ(l.wrong.load(),0);
        CHECK(same(a0));CHECK(same(a1));
        CHECK_EQ(ta.units,2);CHECK_EQ(ta.reads,4);
        /* A posted call afterwards has no `landed` left over from this one. */
        Mover::HostPlanes b0;b0.unit=units[0];Mover::HostPlanes* pb[1]={&b0};
        REQUIRE(t.mover->start_unit_planes(pb,1,1));
        REQUIRE(t.mover->finish_unit_planes());
        CHECK(same(b0));CHECK_EQ(l.calls.load(),2);
    }
}

/* Mover::plane_spans and Mover::rest_mirror_tail (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT): with
 * nothing in, the reads read_unit_planes has always made of an extent (the container's head, the
 * mirror's tail, the scales; one read without a mirror); with bytes in, reads that tile the rest once
 * in order, the mirror's part whole, past the bytes in and ending the codes plane, or none; a rest's
 * mirror share scales with what is left of its unit, at most the w3 codes past the bytes in, on the
 * alignment. Shapes of a DeepSeek-V4.1 gate_up: codes [4608, 2560] (w3 the second half), scales
 * [4608, 160]. */
TEST(mover_plane_spans_tile_a_rest_and_keep_the_mirror_in_w3){
    constexpr int64_t A=4096,P=512*KiB;
    const int64_t end=4608ll*2560,len=end+4608ll*160,w3=end/2,tail=596*A,total=len+5120ll*1152+5120ll*72;
    Mover::PlaneSpan s[3];
    REQUIRE_EQ(Mover::plane_spans(len,0,end,tail,s),3);
    CHECK(s[0].off==0&&s[0].len==end-tail&&!s[0].mirror);
    CHECK(s[1].off==end-tail&&s[1].len==tail&&s[1].mirror);
    CHECK(s[2].off==end&&s[2].len==len-end&&!s[2].mirror);
    REQUIRE_EQ(Mover::plane_spans(len,0,0,0,s),1);CHECK(s[0].off==0&&s[0].len==len&&!s[0].mirror);
    CHECK_EQ(Mover::plane_spans(len,len,end,tail,s),0);CHECK_EQ(Mover::plane_spans(0,0,0,0,s),0);
    for(int64_t in=0;in<len;in+=P)for(int64_t t:{int64_t(0),A,int64_t(MiB),tail,w3,w3+A}){
        const int n=Mover::plane_spans(len,in,end,t,s);int64_t at=in;bool mirror=false;
        for(int j=0;j<n;++j){CHECK_EQ(s[j].off,at);CHECK(s[j].len>0);at+=s[j].len;
            if(s[j].mirror){mirror=true;CHECK(s[j].off>=in);CHECK_EQ(s[j].off+s[j].len,end);CHECK_EQ(s[j].len,t);}}
        CHECK_EQ(at,len);
        CHECK_EQ(mirror,t>0&&t<=end-in);
    }
    CHECK_EQ(Mover::rest_mirror_tail(tail,total,total,0,end,w3,A),tail);   //a whole unit: the whole tail
    const int64_t in7=7*P,left=total-in7,t7=Mover::rest_mirror_tail(tail,left,total,in7,end,w3,A);
    CHECK_EQ(t7%A,int64_t(0));CHECK(t7<=tail*left/total);CHECK(t7>=tail*left/total-A);
    CHECK_EQ(Mover::rest_mirror_tail(tail,total-end,total,end,end,w3,A),int64_t(0));     //the codes all in
    CHECK_EQ(Mover::rest_mirror_tail(tail,total/2,total,end-3*A,end,w3,A),3*A);         //no further than the bytes in
    CHECK_EQ(Mover::rest_mirror_tail(4*w3,total,total,0,end,w3,A),w3);                  //no more than w3
    CHECK_EQ(Mover::rest_mirror_tail(0,total,total,0,end,w3,A),int64_t(0));
    CHECK_EQ(Mover::rest_mirror_tail(tail,0,total,0,end,w3,A),int64_t(0));
}

/* RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE in the file stager (place/hx_side_store.h) on this file's packed
 * fixture, where expert 1's down plane has more raw rounds than its budget, so the packed cache refuses its
 * unit for good. Through the stager's test friend, with the actual HOST Mover:
 *  - the insert threads' loop (hx_bg_loop) offers a unit to the store only after the packed cache refused
 *    it: the unit that fits is cached and not held, the refused one held, and a later refusal of the same
 *    unit keeps the planes held first;
 *  - the read-ahead planner (hx_ra_plan) plans the refused unit before it is held, and neither the cached
 *    unit nor the held one after;
 *  - a layer that selects the held unit (hx_choose) takes its planes with no read: hx_miss_ready_ 2, the
 *    held buffers themselves, the read-ahead planned before they were held dropped as cached, nothing on
 *    the layer's read list (hx_unread_misses), counted as a side-store hit;
 *  - no insert is queued for it (hx_queue_inserts), while another miss of the layer is;
 *  - once the container changes, the layer refuses the held planes before any job (the epoch check, as
 *    for units read ahead whole). */
namespace {
/* The process's CPU pool for hx_choose (its schedule estimates and reservations; no job runs here): two
 * workers and a silu table of zeros, the switches unset again so no later case sees them. */
HostExpertPool* side_store_pool(){
    static HostExpertPool* p=[]()->HostExpertPool*{
        const std::string path=radtest::tmpdir()+"/silu_side_store.bin";
        std::vector<float> silu(65536,0.0f);FILE* f=std::fopen(path.c_str(),"wb");if(!f)return nullptr;
        const bool ok=std::fwrite(silu.data(),4,65536,f)==65536;std::fclose(f);if(!ok)return nullptr;
        ::setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS","2",1);::setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU",path.c_str(),1);
        HostExpertPool* got=HostExpertPool::get();
        ::unsetenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS");::unsetenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU");
        return got;}();
    return p;
}
} //namespace
namespace rad {
/* FileStager's test friend (stager.h), defined above with the multi-row case. The piece readers
 * outlive the stager: its destructor forgets its units on them. */
void EpCountsHostProbe::side_store_holds_what_the_packed_cache_refuses(){
    Environment mapped("1","1");PackedEnvironment packed;
    Fixture f;f.experts();f.info.relayout=mx_relayout;Image image(lay_out_mxfp4(f));write_scales(f,image);
    Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);const auto units=two_ssd_experts(plan);
    std::atomic<uint64_t> calls{0};Access::Driver driver;Transport t;t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    REQUIRE(t.mover->packed_cache());
    HostExpertPool* pool=side_store_pool();
    if(!pool||!__builtin_cpu_supports("avx2")){SKIP("no AVX2 or no CPU pool");return;}
    const int32_t held=plan.units[size_t(units[0])].expert==1?units[0]:units[1],fits=held==units[0]?units[1]:units[0];
    const int layer=plan.units[size_t(held)].layer,e_held=plan.units[size_t(held)].expert,e_fits=plan.units[size_t(fits)].expert;
    REQUIRE_EQ(e_held,1);REQUIRE_EQ(plan.units[size_t(fits)].layer,layer);REQUIRE_EQ(plan.rank,0);
    /* The stager as hx_init_impl leaves it for this rank's one routed layer: misses on the CPU, read-ahead
     * in pieces on these readers, the store's room two units. */
    REQUIRE(r.start(1,512*KiB));
    fs.mover_=t.mover.get();fs.plan_=&plan;fs.hx_pool_=pool;fs.hx_policy_=FileStager::kHxAll;fs.hx_miss_rows_=1;
    fs.hx_units_.reset(new Mover::HostUnit[FileStager::kHxUnits]);
    StagedLayer L;L.layer=layer;L.units={fits,held};fs.layers_={L};
    /* hx_choose is called directly here: simulate the decode pass-begin read-ahead admission. */
    fs.hx_ra_pass_=true;
    fs.hx_ra_=&r;fs.hx_ra_li_.assign(size_t(layer+1),-1);fs.hx_ra_li_[size_t(layer)]=0;
    std::array<int32_t,96> uo;uo.fill(-1);uo[size_t(e_held)]=held;uo[size_t(e_fits)]=fits;fs.hx_unit_of_={uo};
    fs.hx_ra_ok_.assign(plan.units.size(),0);fs.hx_ra_ok_[size_t(held)]=fs.hx_ra_ok_[size_t(fits)]=1;
    fs.hx_side_.configure(2);
    std::vector<int32_t> counts(96,0);counts[size_t(e_held)]=counts[size_t(e_fits)]=1;
    fs.counts_host_=counts.data();fs.counts_cap_=96;
    std::shared_ptr<ReadAheadUnit> ahead;
    {
        /* Pass 1, layer 0's misses, read as the layer would (the stager's own planes). */
        auto pf=fs.hx_planes_get(fits),ph=fs.hx_planes_get(held);
        Mover::HostPlanes* ps[2]={pf.get(),ph.get()};
        REQUIRE(t.mover->read_unit_planes(ps,2,1));
        /* Before anything is held or cached, the planner reads the refused unit ahead for pass 1. */
        const int32_t g1[1]={e_held};
        fs.hx_ra_plan(1,layer,g1,1);
        CHECK_EQ(fs.hx_stats_.ra_planned,uint64_t(1));
        {std::lock_guard<std::mutex> lk(fs.hx_ra_mu_);REQUIRE_EQ(fs.hx_ra_live_.size(),size_t(1));
         CHECK_EQ(fs.hx_ra_live_[0].unit,held);ahead=fs.hx_ra_live_[0].u;}
        REQUIRE(ahead!=nullptr);
        /* The insert threads' loop: the fitting unit is cached and not held; the refused one is held. */
        auto run_inserts=[&](std::vector<FileStager::HxInsert> jobs){
            {std::lock_guard<std::mutex> lk(fs.hx_bg_mu_);for(auto& j:jobs)fs.hx_bg_q_.push_back(std::move(j));fs.hx_bg_stop_=true;}
            std::thread bg([&]{fs.hx_bg_loop();});bg.join();
            std::lock_guard<std::mutex> lk(fs.hx_bg_mu_);fs.hx_bg_stop_=false;
        };
        run_inserts({FileStager::HxInsert{pf,1},FileStager::HxInsert{ph,1}});
        CHECK(t.mover->unit_cached(fits));CHECK(!t.mover->unit_cached(held));
        CHECK(t.mover->packed_refused(held));CHECK(!t.mover->packed_refused(fits));
        CHECK(fs.hx_side_.find(held)==ph);CHECK(!fs.hx_side_.holds(fits));CHECK_EQ(fs.hx_side_.stats().units,1);
        CHECK_EQ(fs.insert_stats().units,uint64_t(2));
        /* A later miss of the refused unit, inserted again: refused again, the planes held first stay. */
        auto ph2=fs.hx_planes_get(held);Mover::HostPlanes* p2=ph2.get();
        REQUIRE(t.mover->read_unit_planes(&p2,1,1));
        run_inserts({FileStager::HxInsert{ph2,1}});
        CHECK(fs.hx_side_.find(held)==ph);CHECK_EQ(fs.hx_side_.stats().units,1);CHECK_EQ(fs.hx_side_.stats().full,uint64_t(0));
        CHECK_EQ(t.mover->packed_stats().over_budget_units,uint64_t(1));
        ph2.reset();
        /* Pass 2's planner: the cached unit and the held one both count as in, so nothing is planned. */
        const int32_t g2[2]={e_held,e_fits};
        fs.hx_ra_plan(2,layer,g2,2);
        CHECK_EQ(fs.hx_stats_.ra_planned,uint64_t(1));
        {std::lock_guard<std::mutex> lk(fs.hx_ra_mu_);CHECK_EQ(fs.hx_ra_live_.size(),size_t(1));}
        /* Pass 1's layer selects both: the fitting unit lent from the packed cache, the held one handed its
         * planes, the read-ahead planned for it before it was held dropped as cached. */
        fs.trace_pass_=1;fs.pass_tok_=1;fs.want_={fits,held};
        fs.hx_choose(0);
        REQUIRE_EQ(fs.hx_miss_.size(),size_t(1));REQUIRE_EQ(fs.hx_miss_ready_.size(),size_t(1));
        CHECK(fs.hx_miss_[0]==ph);CHECK_EQ(int(fs.hx_miss_ready_[0]),2);
        CHECK_EQ(fs.hx_side_handed_,1);CHECK_EQ(fs.hx_ra_handed_,0);
        CHECK_EQ(fs.hx_stats_.side_hits,uint64_t(1));CHECK_EQ(fs.hx_stats_.side_d_hits,uint64_t(1));
        CHECK_EQ(fs.hx_stats_.ra_cached,uint64_t(1));CHECK(ahead->cancelled());
        CHECK(fs.hx_ra_here_.empty());
        {std::lock_guard<std::mutex> lk(fs.hx_ra_mu_);CHECK(fs.hx_ra_live_.empty());}
        CHECK_EQ(fs.hx_n_,1);CHECK_EQ(fs.hx_li_,0);CHECK(fs.want_.empty());
        r.wait(*ahead);   /* its pieces in flight land in its own buffers */
        /* Nothing for the layer to read. */
        std::vector<Mover::HostPlanes*> out;std::vector<Mover::PlanesIn> in;bool alone=true;
        CHECK(!fs.hx_unread_misses(&out,&in,&alone));CHECK(out.empty());CHECK(!alone);
        /* No insert queued for the held unit; another miss of the layer is queued, with its pass's rows. */
        fs.hx_queue_inserts(1);
        {std::lock_guard<std::mutex> lk(fs.hx_bg_mu_);CHECK(fs.hx_bg_q_.empty());}
        auto other=fs.hx_planes_get(fits);
        fs.hx_miss_={ph,other};fs.hx_miss_ready_={2,0};fs.hx_miss_ra_={nullptr,nullptr};
        fs.hx_queue_inserts(1);
        {std::lock_guard<std::mutex> lk(fs.hx_bg_mu_);REQUIRE_EQ(fs.hx_bg_q_.size(),size_t(1));
         CHECK(fs.hx_bg_q_.front().p==other);CHECK_EQ(fs.hx_bg_q_.front().rows,1);fs.hx_bg_q_.clear();}
        fs.hx_miss_={ph};fs.hx_miss_ready_={2};fs.hx_miss_ra_={nullptr};other.reset();
        /* The container changes under the held planes: the layer refuses them before any job. */
        REQUIRE_EQ(::ftruncate(image.fd,off_t(image.bytes.size()+1)),0);REQUIRE_EQ(::fsync(image.fd),0);
        CHECK(!t.mover->source_epochs_current());
        CHECK_EQ(fs.hx_submit_misses_now(1),RAD_E_IO);
        CHECK(fs.hx_jobs_.empty());CHECK_EQ(pool->work(),int64_t(0));
        /* Let go of everything the mover lent or the readers hold before the mover goes. */
        fs.hx_drain();
        fs.hx_side_.clear();
        r.forget(&fs);
    }
    ahead.reset();
    {std::lock_guard<std::mutex> lk(fs.hx_ra_mu_);fs.hx_ra_live_.clear();fs.hx_ra_here_.clear();}
    {std::lock_guard<std::mutex> lk(fs.hx_planes_mu_);fs.hx_planes_free_.clear();}
    CHECK_EQ(pool->backlog(),int64_t(0));
    fs.counts_host_=nullptr;fs.counts_cap_=0;fs.mover_=nullptr;fs.plan_=nullptr;
    t.drain();
}
} //namespace rad
TEST(mapped_packed_side_store_holds_refused_units_for_the_stager){
    rad::EpCountsHostProbe p;p.side_store_holds_what_the_packed_cache_refuses();
}

int main(int argc,char** argv){
#ifdef RAD_HAVE_HIP
    std::fprintf(stderr,"HOST-only matching fresh Core required; refusing GPU binary.\n");return 2;
#else
    if(!device_is_host()){std::fprintf(stderr,"HOST backend required.\n");return 2;}
    return radtest::run_all(argc,argv);
#endif
}

namespace {
struct DirectFillEnvironment {
    bool had=false;std::string old;
    DirectFillEnvironment(){const auto*p=std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL");had=p;old=p?p:"";::setenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL","1",1);}
    ~DirectFillEnvironment(){if(had)::setenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL",old.c_str(),1);else::unsetenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL");}
};
struct FillResidency {
    ResidencyTable* table;uint64_t epoch;std::vector<std::pair<rad_weight,WeightSlot>> slots;
    FillResidency(Case&c,const std::vector<int32_t>& units):table(c.transport.mover->residency()),epoch(table->epoch()){
        for(auto u:units)for(auto h:c.plan.units[size_t(u)].weights){const auto*w=table->lookup(h);REQUIRE(w);slots.push_back({h,*w});}}
    void unchanged(){CHECK_EQ(table->epoch(),epoch);for(const auto&x:slots){const auto*w=table->lookup(x.first);REQUIRE(w);
        CHECK(w->ptr==x.second.ptr);CHECK(w->ready==x.second.ready);CHECK_EQ(w->generation,x.second.generation);}}
};
void restore_relayout(Case&c){c.transport.mover->set_file_relayout([&c](const WeightInfo&w,const RadTensor*ps,void*dst,int64_t n){
    c.calls.fetch_add(1,std::memory_order_relaxed);const auto*k=weight_layout_kernel(c.fixture.prog,w);
    return k&&k->relayout?k->relayout(nullptr,0,w.lay_operand,&w.enc,w.sel,ps,w.n_sel,dst,n):RAD_E_STATE;});}
}
TEST(mapped_direct_fill_cold_identity_and_relayout_pin_cache_source_without_window_insertion_copy){
    for(bool relaid:{false,true}){DirectFillEnvironment direct;Case c(relaid,80*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};
        c.stage(a);const auto cold=Access::stats(*c.transport.mover);
        CHECK_EQ(cold.slabs.fill_reservations,uint64_t(2));CHECK_EQ(cold.slabs.fill_commits,uint64_t(2));CHECK_EQ(cold.slabs.fill_bytes,uint64_t(18800640));
        CHECK_EQ(cold.slabs.fills_active,size_t(0));CHECK_EQ(cold.cache.lease_refs,size_t(2));CHECK_EQ(c.driver.pending(),size_t(2));
        CHECK_EQ(c.transport.mover->stats().mapped_source_copies,uint64_t(2));c.check(a);
        const auto callbacks=c.calls.load();const auto before=Io::capture();c.stage(a);const auto after=Io::capture();
        CHECK_EQ(delta(before,after).calls,uint64_t(0));CHECK_EQ(c.calls.load(),callbacks);CHECK_EQ(Access::stats(*c.transport.mover).slabs.fill_commits,uint64_t(2));c.check(a);
    }
}
TEST(mapped_direct_fill_ReadOnly_miss_does_not_reserve_or_admit_then_warm_lease_works){
    DirectFillEnvironment direct;Case c(true,80*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};
    c.stage(a,0,NativeFileCacheReadPolicy::ReadOnly);auto cold=Access::stats(*c.transport.mover);
    CHECK_EQ(cold.slabs.fill_reservations,uint64_t(0));CHECK_EQ(cold.cache.resident_entries,size_t(0));CHECK_EQ(cold.cache.lease_refs,size_t(0));c.check(a);
    c.stage(a);c.check(a);const auto before=Io::capture();c.stage(a,0,NativeFileCacheReadPolicy::ReadOnly);const auto after=Io::capture();
    CHECK_EQ(delta(before,after).calls,uint64_t(0));CHECK_EQ(Access::stats(*c.transport.mover).slabs.fill_commits,uint64_t(2));
    CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));c.check(a);
}
TEST(mapped_direct_fill_partial_relayout_error_cancels_before_H2D_or_Residency_then_reuses){
    DirectFillEnvironment direct;Case c(true,80*MiB,"1",true,"1");Access::serial_reader(*c.transport.mover);const std::vector<int32_t>a{c.units[0]};FillResidency old(c,a);
    c.transport.mover->set_file_relayout([](const WeightInfo&,const RadTensor*,void*dst,int64_t n){std::memset(dst,0x9b,size_t(n/3));return RAD_E_STATE;});
    std::vector<int32_t>scratch;CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&scratch),RAD_E_STATE);
    old.unchanged();auto failed=Access::stats(*c.transport.mover);
    CHECK_EQ(failed.slabs.fill_aborts,uint64_t(2));CHECK_EQ(failed.slabs.fills_active,size_t(0));CHECK_EQ(failed.cache.resident_entries,size_t(0));CHECK_EQ(c.driver.copy_attempts,size_t(0));
    const auto allocations=failed.allocations;restore_relayout(c);c.stage(a);c.check(a);CHECK_EQ(Access::stats(*c.transport.mover).allocations,allocations);
}
TEST(mapped_direct_fill_epoch_change_after_pack_aborts_unpublished_reservation){
    DirectFillEnvironment direct;Case c(true,80*MiB,"1",true,"1");Access::serial_reader(*c.transport.mover);const std::vector<int32_t>a{c.units[0]};FillResidency old(c,a);
    c.transport.mover->set_file_relayout([&c](const WeightInfo&w,const RadTensor*ps,void*dst,int64_t n){
        const auto*k=weight_layout_kernel(c.fixture.prog,w);if(!k||!k->relayout)return int(RAD_E_STATE);
        const int rc=k->relayout(nullptr,0,w.lay_operand,&w.enc,w.sel,ps,w.n_sel,dst,n);if(rc<0)return rc;
        if(::ftruncate(c.image->fd,off_t(c.image->bytes.size()+1))||::fsync(c.image->fd))return int(RAD_E_STATE);return int(RAD_OK);});
    std::vector<int32_t>scratch;CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&scratch),RAD_E_STATE);
    old.unchanged();auto failed=Access::stats(*c.transport.mover);CHECK_EQ(failed.slabs.fill_aborts,uint64_t(1));
    CHECK_EQ(failed.slabs.fill_commits,uint64_t(0));CHECK_EQ(failed.slabs.fills_active,size_t(0));CHECK_EQ(failed.cache.resident_entries,size_t(0));CHECK_EQ(c.driver.copy_attempts,size_t(0));
}
TEST(mapped_direct_fill_cold_and_warm_record_drain_faults_keep_first_consumer_pin){
    for(bool warm:{false,true}){DirectFillEnvironment direct;Case c(false,80*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]};
        if(warm){c.stage(a);c.check(a);}FillResidency old(c,a);c.driver.fail_record=true;std::vector<int32_t>scratch;
        CHECK_EQ(c.transport.mover->stage_file_units(a,c.transport.mover->file_buffer(0),c.transport.mover->file_buffer_bytes(),nullptr,c.transport.done,&scratch),RAD_E_STATE);
        old.unchanged();CHECK_EQ(c.driver.pending(),size_t(2));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
        c.driver.fail_drain=true;CHECK_EQ(Access::drain(*c.transport.mover),RAD_E_STATE);CHECK_EQ(Access::clear(*c.transport.mover),RAD_E_STATE);old.unchanged();
        CHECK_EQ(Access::stats(*c.transport.mover).api.free_invocations,uint64_t(0));CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(2));
        c.driver.fail_drain=c.driver.fail_record=false;need(Access::drain(*c.transport.mover),"actual fake-consumer completion acknowledged");
        CHECK_EQ(Access::stats(*c.transport.mover).cache.lease_refs,size_t(0));c.stage(a);c.check(a);
    }
}
TEST(mapped_direct_fill_held_class_pressure_uses_original_window_without_second_admission){
    DirectFillEnvironment direct;Case c(false,26*MiB,"1",true,"1");const std::vector<int32_t>a{c.units[0]},b{c.units[1]};
    c.stage(a);const auto first=Access::stats(*c.transport.mover);CHECK_EQ(first.cache.lease_refs,size_t(2));
    c.stage(b,1);auto pressure=Access::stats(*c.transport.mover);
    CHECK_EQ(pressure.slabs.fill_commits,uint64_t(2));CHECK_EQ(pressure.cache.leased_tail_bypasses,uint64_t(2));
    CHECK_EQ(pressure.allocations,first.allocations);CHECK_EQ(pressure.cache.lease_refs,size_t(2));CHECK_EQ(c.driver.pending(),size_t(4));
    c.check(b);c.check(a);CHECK_EQ(Access::stats(*c.transport.mover).slabs.fills_active,size_t(0));
}
TEST(mapped_direct_fill_worker_callback_exception_retires_all_unpublished_producers){
    Environment mapped("1","1");DirectFillEnvironment direct;
    struct WorkersOn {bool had;std::string old;WorkersOn(){const auto*p=std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_FILE_WORKERS");had=p;old=p?p:"";::setenv("RADIANCE_EXPERIMENTAL_NATIVE_FILE_WORKERS","1",1);}
        ~WorkersOn(){if(had)::setenv("RADIANCE_EXPERIMENTAL_NATIVE_FILE_WORKERS",old.c_str(),1);else::unsetenv("RADIANCE_EXPERIMENTAL_NATIVE_FILE_WORKERS");}}workers;
    Fixture f;f.experts();Image image(f.lay_out(true));Config cfg=plan_config(false);Plan plan=planned(f,cfg,true);
    const std::vector<int32_t> selected{two_ssd_experts(plan)[0]};std::atomic<uint64_t>calls{0};Access::Driver driver;Transport t;
    t.open(f,plan,image,80*MiB,calls);Access::attach(*t.mover,driver);
    t.mover->set_file_relayout([](const WeightInfo&,const RadTensor*,void*dst,int64_t n)->int{std::memset(dst,0xaf,size_t(n/4));throw std::runtime_error("actual CPU worker fill failed");});
    const auto before=t.mover->residency()->epoch();std::vector<int32_t>scratch;
    CHECK_EQ(t.mover->stage_file_units(selected,t.mover->file_buffer(0),t.mover->file_buffer_bytes(),nullptr,t.done,&scratch),RAD_E_STATE);
    CHECK_EQ(t.mover->residency()->epoch(),before);const auto stats=Access::stats(*t.mover);
    CHECK(stats.slabs.fill_aborts>0);CHECK_EQ(stats.slabs.fills_active,size_t(0));CHECK_EQ(stats.slabs.fill_commits,uint64_t(0));
    CHECK_EQ(stats.cache.resident_entries,size_t(0));CHECK_EQ(stats.cache.lease_refs,size_t(0));CHECK_EQ(driver.copy_attempts,size_t(0));
    for(auto u:selected)for(auto h:plan.units[size_t(u)].weights)CHECK(t.mover->residency()->lookup(h)->ptr==nullptr);
}

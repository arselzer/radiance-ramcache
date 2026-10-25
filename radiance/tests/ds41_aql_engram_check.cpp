/* Explicit checker-only Rad/AQL Engram lifetime gate; never an automatic test. */
#include "../arch/deepseek41/engram_rad_staging.h"
#include "device/device.h"
#include <hip/hip_runtime_api.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace eg=rad::deepseek41::engram;
namespace {
constexpr size_t width=256,row_bytes=width*4,guard=256;
constexpr uint64_t cache_cap=16384,pinned_cap=262144,driver_allowance=65536;
constexpr uint8_t poison=0xbd;
int live_streams=0,live_core_owners=0;
uint64_t core_requested=0,core_peak=0;
void need(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
void okay(int status,const char* label){need(status==RAD_OK,std::string(label)+": "+rad_strerror(status));}
void hip_ok(hipError_t status,const char* label){need(status==hipSuccess,std::string(label)+": "+hipGetErrorString(status));}
std::vector<uint8_t> unhex(const std::string& value){
    need(!(value.size()&1)&&value.size()<=65536,"invalid bounded fixture hex");std::vector<uint8_t> out(value.size()/2);
    const auto nibble=[](char c)->unsigned{if(c>='0'&&c<='9')return unsigned(c-'0');if(c>='a'&&c<='f')return unsigned(c-'a'+10);throw std::runtime_error("bad hex digit");};
    for(size_t i=0;i<out.size();++i)out[i]=uint8_t((nibble(value[2*i])<<4)|nibble(value[2*i+1]));
    return out;
}
struct TinyFiles {
    std::string directory;std::array<std::string,2> paths;
    std::array<std::string,2> hashes{"fa2911bd11ade3bf67dec0fa029c6cd040b6f88040b4d92c9ca571cea7d5c5aa","1525c78cc395df31e91e29596fb50a47bd7f297d959ae85a0921dd33c868bd4d"};
    std::vector<uint8_t> primary_expected;
    explicit TinyFiles(const std::string& fixture){
        std::ifstream input(fixture,std::ios::binary|std::ios::ate);need(bool(input)&&input.tellg()>0&&input.tellg()<=65536,"fixture JSON is absent/too large");input.seekg(0);
        const auto j=nlohmann::json::parse(input);need(j.at("width")==256&&j.at("rows")==4,"wrong source fixture geometry");
        primary_expected=unhex(j.at("f32_hex").get<std::string>().substr(0,row_bytes*2));need(primary_expected.size()==row_bytes,"bad independent F32 row oracle");
        const std::array<const char*,2> exact_hex{"031425364758697a8b9cadbecfe0f102132435465768798a9bacbdcedff00112233445566778899aabbccddeef00112233445566778899aabbccddeeff102132435465768798a9bacb0028dda8433092d338dd81f7a390b50c6c8b00a82161ef9f91c755ae3159e6140f6bdc370100811566f064b07603329947bd84dc491a0034fdf841347b79ab3372f09d501373efe30004eab8aac7003683b3699edf8cf7008ef0000063a0327d2b2b628666e0626f94d7f6b7002896db363e083c72458e6b2cab2a61c45700a8ea173c294738d2a15c8dcb7939d08608","031425364758697a8b9cadbecfe0f102132435465768798a9bacbdcedff00112233445566778899aabbccddeef00112233445566778899aabbccddeeff102132435465768798a9bacb0028d2a8433092d338dd81f7a390b50c6c8b00a82161ef9f91c755ae3159e6140f6bdc370100811566f064b07603329947bd84dc491a0034fdf841347b79ab3372f09d501373efe30004eab8aac7003683b3699edf8cf7008ef0000063a0327d2b2b628666e0626f94d7f6b7002896db363e083c72458e6b2cab2a61c45700a8ea173c294738d2a15c8dcb7939d08608"};
        auto temporary=std::string("/tmp/radiance-ds41-aql-engram.XXXXXX");need(::mkdtemp(temporary.data())!=nullptr,"mkdtemp failed");directory=std::move(temporary);
        const auto prefix=unhex(j.at("prefix_hex"));need(prefix.size()==73,"wrong pinned fixture prefix");
        for(size_t i=0;i<2;++i){
            auto bytes=prefix;auto packed=unhex(j.at(i?"other_packed_hex":"packed_hex").get<std::string>().substr(0,288));bytes.insert(bytes.end(),packed.begin(),packed.end());
            need(bytes==unhex(exact_hex[i])&&bytes.size()==217,"tiny asset bytes differ from verified literal/SHA pin");
            paths[i]=directory+(i?"/layer14.iq4nl":"/layer1.iq4nl");const int fd=::open(paths[i].c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);need(fd>=0,"tiny asset create failed");
            size_t done=0;while(done<bytes.size()){const ssize_t n=::write(fd,bytes.data()+done,bytes.size()-done);if(n<0&&errno==EINTR)continue;if(n<=0){::close(fd);throw std::runtime_error("tiny asset write failed");}done+=size_t(n);}need(::close(fd)==0,"tiny asset close failed");
        }
    }
    ~TinyFiles(){
        for(const auto& path:paths)if(!path.empty())::unlink(path.c_str());
        if(!directory.empty())::rmdir(directory.c_str());
    }
    eg::Iq4NlAsset asset(size_t i)const{return {paths.at(i),hashes.at(i),"gguf-iq4_nl-block-v1",i?14:1,217,73,1,256};}
};
struct Stream {
    RadStream value=nullptr;
    Stream(){okay(rad_stream_create(&value,0),"create actual Rad stream");need(value!=nullptr,"null Rad stream");++live_streams;}
    ~Stream(){if(!value)return;if(rad_stream_sync(value)!=RAD_OK){std::fputs("unproven stream drain: retain queue/pins until process exit\n",stderr);return;}rad_stream_destroy(value);--live_streams;}
};
struct CoreMemory {
    struct State {void* allocation=nullptr;void* device=nullptr;void* host=nullptr;size_t bytes=0;int kind=RAD_MEM_DEVICE;std::shared_ptr<Stream> stream;std::shared_ptr<State> retained;};
    std::shared_ptr<State> state;
    CoreMemory(std::shared_ptr<Stream> stream,size_t bytes,int kind):state(std::make_shared<State>()){
        need(bytes>0&&bytes<=65536&&core_requested+bytes+cache_cap+pinned_cap+8192<=1048576,"explicit fixture cap exceeded");
        state->stream=std::move(stream);state->bytes=bytes;state->kind=kind;state->retained=state;
        state->allocation=rad_dev_alloc(int64_t(bytes),kind);need(state->allocation!=nullptr,"tiny core allocation failed");
        ++live_core_owners;core_requested+=bytes;core_peak=std::max(core_peak,core_requested);
        state->device=kind==RAD_MEM_DEVICE?state->allocation:rad_dev_device_ptr(state->allocation);
        state->host=kind==RAD_MEM_DEVICE?nullptr:rad_dev_host_ptr(state->allocation);need(state->device&&(kind==RAD_MEM_DEVICE||state->host),"core memory alias unavailable");
    }
    ~CoreMemory(){if(!state->allocation){state->retained.reset();return;}if(rad_stream_sync(state->stream->value)!=RAD_OK)return;
        // Core memory uses its existing trusted void teardown API; this is not
        // the checked physical-free accounting of the separate HIP pinned ring.
        rad_dev_free(state->allocation,state->kind);state->allocation=nullptr;core_requested-=state->bytes;--live_core_owners;state->retained.reset();}
    void* data()const{return static_cast<uint8_t*>(state->device)+guard;}
    void initialize(){okay(rad_memset_async(state->device,poison,int64_t(state->bytes),state->stream->value),"poison core guards");}
};
// Checker-only: physical pins are retained, with adopted budget, until all AQL
// work/owners/streams are gone. No attempt to invalidate the foreign-pointer
// locate cache via dummy/null core free, and no mid-run physical release/reuse.
class RetainedPins final:public eg::CheckedRadHostAllocator {
    struct Slot {void* pointer=nullptr;size_t usable=0,physical=0;uint64_t accounted=0;bool reserved=false,released=false;eg::StagingCharge charge;};
    struct State {std::array<Slot,2> slots;std::shared_ptr<State> retained;size_t allocations=0,frees=0;};
    struct Owner final:eg::HostRowAllocation {
        std::shared_ptr<State> state;size_t index;
        Owner(std::shared_ptr<State> s,size_t i):state(std::move(s)),index(i){}
        ~Owner()override{state->slots[index].released=true;}
        std::span<std::byte> memory()noexcept override{auto& s=state->slots[index];return {static_cast<std::byte*>(s.pointer),s.usable};}
        uint64_t accounted_bytes()const noexcept override{return state->slots[index].accounted;}
        void adopt_budget_charge(eg::StagingCharge& charge)noexcept override{auto& slot=state->slots[index];assert(!slot.charge.bytes());slot.charge=std::move(charge);}
    };
    std::shared_ptr<State> state_=std::make_shared<State>();
public:
    RetainedPins(){state_->retained=state_;}
    int allocation_charge(size_t bytes,uint64_t* out)const noexcept override{
        if(!out||!bytes||bytes>4096)return RAD_E_SHAPE;
        *out=4096+driver_allowance+sizeof(Slot)+sizeof(Owner)+512;return RAD_OK;
    }
    int allocate(size_t bytes,uint64_t maximum,std::shared_ptr<eg::HostRowAllocation>* out)noexcept override{
        uint64_t charge=0;if(!out||allocation_charge(bytes,&charge)!=RAD_OK||charge>maximum)return RAD_E_SHAPE;out->reset();
        try{
            size_t index=0;
            while(index<state_->slots.size()&&state_->slots[index].reserved)++index;
            if(index==state_->slots.size())return RAD_E_FULL;
            auto owner=std::make_shared<Owner>(state_,index);auto& slot=state_->slots[index];slot.reserved=true;slot.released=false;slot.usable=bytes;slot.physical=4096;slot.accounted=charge;
            int current=-1;const auto bound=hipGetDevice(&current);int result=bound==hipSuccess&&current==0?RAD_OK:RAD_E_DEVICE;
            if(result==RAD_OK){const auto status=hipHostMalloc(&slot.pointer,slot.physical,hipHostMallocPortable);result=status==hipSuccess&&slot.pointer?RAD_OK:RAD_E_DEVICE;if(slot.pointer)++state_->allocations;}
            if(result==RAD_OK){std::memset(slot.pointer,0xA7,slot.physical);hipPointerAttribute_t attributes{};void* alias=nullptr;
                if(hipPointerGetAttributes(&attributes,slot.pointer)!=hipSuccess||attributes.type!=hipMemoryTypeHost||
                   hipHostGetDevicePointer(&alias,slot.pointer,0)!=hipSuccess||!alias||reinterpret_cast<uintptr_t>(slot.pointer)%alignof(float))result=RAD_E_DEVICE;}
            *out=std::move(owner);return result; // Also return owner on partial allocation failure.
        }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
    }
    uint64_t retained_charge()const{uint64_t bytes=0;for(const auto& slot:state_->slots)bytes+=slot.charge.bytes();return bytes;}
    size_t allocations()const{return state_->allocations;}
    bool finalize_after_all_rad_owners(){
        need(live_streams==0&&live_core_owners==0,"cannot free foreign pins before all Rad owners/streams are gone");
        bool success=true;int current=-1;hip_ok(hipGetDevice(&current),"verify final pinned-free device");need(current==0,"pinned finalization changed rank device");
        for(auto& slot:state_->slots)if(slot.reserved){
            need(slot.released,"pinned owner remains live at finalization");
            if(slot.pointer){
                const auto* bytes=static_cast<const uint8_t*>(slot.pointer);
                need(std::all_of(bytes+slot.usable,bytes+slot.physical,[](uint8_t value){return value==0xA7;}),"pinned allocation tail guard changed");
                const auto freed=hipHostFree(slot.pointer);if(freed!=hipSuccess){success=false;continue;}slot.pointer=nullptr;++state_->frees;
            }
            slot.charge.reset();slot.reserved=false;
        }
        if(success){need(state_->allocations==state_->frees,"checked pinned frees did not match allocations");state_->retained.reset();}
        return success;
    }
};
struct Gate {
    void* value=nullptr;
    explicit Gate(RadStream stream){
        okay(rad_stream_gate(stream,&value),"create actual AQL gate");
        if(rad_gate_room(stream)<16){static_cast<void>(rad_gate_open(value));value=nullptr;throw std::runtime_error("not enough bounded AQL gate room");}
    }
    ~Gate(){if(value)static_cast<void>(rad_gate_open(value));}
    void open(){if(value){okay(rad_gate_open(value),"open actual AQL gate");value=nullptr;}}
};
struct Reader {std::shared_ptr<CoreMemory> output;RadStream expected;int calls=0;bool throwing=false;};
int read_row(RadStream stream,const eg::DeviceRowTarget& target,void* user){
    auto& reader=*static_cast<Reader*>(user);++reader.calls;if(stream!=reader.expected)return RAD_E_STATE;
    const int status=rad_memcpy_async(reader.output->data(),reinterpret_cast<const void*>(target.owner->address()+target.offset),row_bytes,stream);
    if(reader.throwing)throw std::runtime_error("intentional exception after actual D2D enqueue");
    return status;
}
std::vector<uint8_t> read(const std::shared_ptr<CoreMemory>& source,const std::shared_ptr<CoreMemory>& mapped){
    okay(rad_memcpy_async(mapped->state->device,source->state->device,int64_t(source->state->bytes),source->state->stream->value),"core guarded readback");
    okay(rad_stream_sync(source->state->stream->value),"finish core guarded readback");const auto* bytes=static_cast<const uint8_t*>(mapped->state->host);
    need(std::all_of(bytes,bytes+guard,[](uint8_t v){return v==poison;})&&std::all_of(bytes+guard+row_bytes,bytes+source->state->bytes,[](uint8_t v){return v==poison;}),"actual device redzone changed");return {bytes+guard,bytes+guard+row_bytes};
}
void run(const TinyFiles& files,const std::shared_ptr<RetainedPins>& pins,const std::shared_ptr<eg::StagingBudget>& budget){
    auto stream=std::make_shared<Stream>();auto input=std::make_shared<CoreMemory>(stream,row_bytes+2*guard,RAD_MEM_DEVICE);
    auto output=std::make_shared<CoreMemory>(stream,row_bytes+2*guard,RAD_MEM_DEVICE);auto mapped=std::make_shared<CoreMemory>(stream,row_bytes+2*guard,RAD_MEM_HOST_MAPPED);
    input->initialize();output->initialize();okay(rad_stream_sync(stream->value),"settle guarded core allocations");
    std::string why;std::shared_ptr<eg::Iq4NlLayerStores> source;eg::Iq4NlCacheOptions options;options.max_batch_rows=2;options.max_io_rows=1;
    okay(eg::Iq4NlLayerStores::open("aql-engram-tiny",{files.asset(0),files.asset(1)},cache_cap,options,&source,&why),"open actual bounded IQ4_NL row stores");
    const std::array<int64_t,1> ids{0};std::vector<float> expected_other(width),primary(width);
    okay(source->row_lookup()(1,ids,width,primary,&why),"decode primary CPU row");need(std::memcmp(primary.data(),files.primary_expected.data(),row_bytes)==0,"primary CPU decode differs from independent golden");
    okay(source->row_lookup()(14,ids,width,expected_other,&why),"decode distinct secondary asset");
    std::vector<uint8_t> other(row_bytes);std::memcpy(other.data(),expected_other.data(),row_bytes);need(other!=files.primary_expected,"asset separation fixture degenerate");
    std::shared_ptr<eg::RadEngramBackend> backend;
    okay(eg::RadEngramBackend::create_on_rank("aql-engram-tiny",{stream->value,stream,0,101},pins,{1},&backend,&why),"create actual Rad Engram backend");
    struct Close {std::shared_ptr<eg::RadEngramBackend> backend;~Close(){if(backend)static_cast<void>(backend->close());}} close{backend};
    std::shared_ptr<eg::EngramStagingPool> pool;okay(eg::EngramStagingPool::create("aql-engram-tiny",source,budget,backend,{1,1,width},&pool,&why),"create bounded actual staging pool");
    std::shared_ptr<eg::RadArenaRowBuffer> arena;okay(backend->bind_arena(input->data(),row_bytes,input,101,&arena,&why),"bind owned actual device span");
    const auto target=[&](size_t layer){return eg::DeviceRowTarget{arena,0,files.hashes[layer]};};
    const auto pending=[&](const eg::RowTransferTicket& ticket){need(!ticket.state().quiescent,"transfer retired before final-reader completion");eg::RowTransferTicket blocked;need(pool->submit(1,ids,target(0),&blocked)==RAD_E_FULL,"busy staging slot was overwritten/reused");};
    const auto drain=[&](eg::RowTransferTicket& ticket,int status){const auto result=backend->drain(ticket.backend_record(*backend));need(result.quiescent&&result.status==status,"actual Rad upload/reader drain mismatch");need(pool->reap()==0,"safe staging slot not reclaimed");};
    Reader reader{output,stream->value};eg::RowTransferTicket first;
    {Gate upload(stream->value);okay(pool->submit(1,ids,target(0),&first,{},&why),"submit gated actual upload");pending(first);upload.open();}
    okay(rad_stream_sync(stream->value),"finish upload only");pending(first);need(read(input,mapped)==files.primary_expected,"actual uploaded row differs");
    {Gate gate(stream->value);okay(backend->consume(first,read_row,&reader),"enqueue actual D2D reader");first.cancel();pending(first);gate.open();}
    drain(first,RAD_OK);need(read(output,mapped)==files.primary_expected,"cancel after submission invalidated/overwrote reader data");
    eg::RowTransferTicket second;
    {Gate gate(stream->value);okay(pool->submit(14,ids,target(1),&second,{},&why),"submit secondary asset");need(first.state().quiescent&&first.state().status==RAD_OK,"old event generation was not latched");reader.throwing=true;
        need(backend->consume(second,read_row,&reader)==RAD_E_STATE,"partial-reader callback exception was lost");pending(second);gate.open();}
    drain(second,RAD_E_STATE);need(read(output,mapped)==other,"throwing partial reader lost its uploaded row");reader.throwing=false;
    eg::RowTransferTicket cancelled;std::stop_source request;
    {Gate gate(stream->value);okay(pool->submit(1,ids,target(0),&cancelled,request.get_token(),&why),"submit cancellable upload");request.request_stop();cancelled.cancel();const int calls=reader.calls;
        need(backend->consume(cancelled,read_row,&reader)==RAD_E_CANCELLED&&reader.calls==calls,"cancelled request ran a new GPU reader");pending(cancelled);gate.open();}
    drain(cancelled,RAD_E_CANCELLED);need(read(output,mapped)==other,"cancel before reader overwrote output");
    okay(source->row_lookup()(1,ids,width,primary,&why),"request cancellation kept shared cache usable");
    eg::RowTransferTicket fresh;
    {Gate gate(stream->value);okay(pool->submit(14,ids,target(1),&fresh,{},&why),"submit fresh request after cancellation");okay(backend->consume(fresh,read_row,&reader),"consume fresh request");pending(fresh);
        need(first.state().quiescent&&second.state().quiescent,"reused events changed old completion generations");gate.open();}
    drain(fresh,RAD_OK);need(read(output,mapped)==other,"fresh post-cancel output differs");
    eg::RowTransferTicket manual;
    {Gate gate(stream->value);okay(pool->submit(1,ids,target(0),&manual,{},&why),"submit manual handoff");okay(backend->order_upload(manual),"order actual manual upload");
        okay(rad_memcpy_async(output->data(),input->data(),row_bytes,stream->value),"actual manual D2D consumer");need(backend->finish_no_readers(manual)==RAD_E_STATE,"no-reader proof accepted after reader handoff");
        okay(backend->finish_readers(manual),"seal actual manual final reader");pending(manual);gate.open();}
    drain(manual,RAD_OK);need(read(output,mapped)==files.primary_expected,"manual handoff result differs");
    eg::RowTransferTicket refused;auto wrong=target(0);wrong.source_sha256=files.hashes[1];need(pool->submit(1,ids,wrong,&refused)==RAD_E_SHAPE,"wrong asset identity accepted");
    const std::array<int64_t,1> outside{1};need(pool->submit(1,outside,target(0),&refused)==RAD_E_SHAPE,"out-of-table row accepted");
    const auto stats=backend->native_stats();need(stats.queued==5&&stats.retired==5&&stats.events==2&&stats.upload_fences==5&&stats.reader_fences==5,"bounded Rad event/generation counts differ");
    need(reader.calls==3&&pins->allocations()==1,"fixed staging ring/reader count grew unexpectedly");
    first.reset();second.reset();cancelled.reset();fresh.reset();manual.reset();refused.reset();wrong.owner.reset();arena.reset();
    need(pool->shutdown().quiescent,"pool shutdown left active work");pool.reset();need(pins->retained_charge()>0&&budget->stats().used==pins->retained_charge(),"pool close refunded quota before actual pinned free");
    need(backend->close().quiescent&&backend->native_stats().events==0,"actual Rad backend failed safe event teardown");close.backend.reset();backend.reset();
    okay(rad_stream_sync(stream->value),"last explicit AQL synchronization");
    std::printf("  five actual uploads/four D2D readers, two reused Rad events, one retained 4096B physical pin; staging_charge=%llu B, core_request_peak=%llu B\n",
                (unsigned long long)budget->stats().used,(unsigned long long)core_peak);
}
void usage(){std::puts("ds41-aql-engram-check --run-gpu --fixtures deepseek41-engram-store.json [--host-kargs]\nTiny actual Rad/AQL stream/events/copy gate; one visible gfx1201; no model/Pools/service.");}
}
int main(int argc,char** argv){
    bool approved=false,host_kargs=false;std::string fixtures;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--help"||arg=="-h"){usage();return 0;}if(arg=="--run-gpu")approved=true;
        else if(arg=="--host-kargs")host_kargs=true;else if(arg=="--fixtures"&&i+1<argc)fixtures=argv[++i];else{std::fprintf(stderr,"unknown/incomplete option: %s\n",arg.c_str());return 2;}}
    if(!approved||fixtures.empty()){usage();return 2;}
    std::shared_ptr<RetainedPins> pins;std::shared_ptr<eg::StagingBudget> budget;
    try{
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)okay(rad::aql_force_host_kargs(),"select host argument placement");
#else
        (void)host_kargs;throw std::runtime_error("checker requires HIP-enabled core");
#endif
        need(rad_dev_count()==1&&std::strcmp(rad::device_backend_name(),"aql")==0&&!rad::device_is_host(),"exactly one visible actual AQL card required");RadDeviceProps props{};
        okay(rad_dev_props(0,&props),"read selected card");need(!props.is_host_backend&&!std::strcmp(props.arch,"gfx1201"),"requires gfx1201");okay(rad_dev_set(0),"bind rank device");
        TinyFiles files(fixtures);pins=std::make_shared<RetainedPins>();budget=std::make_shared<eg::StagingBudget>(pinned_cap);run(files,pins,budget);
        need(live_streams==0&&live_core_owners==0&&core_requested==0,"Rad owners/streams survived gate scope");
        // NO Rad/AQL call after this boundary: checked native free may invalidate
        // cached foreign classifications, so this process immediately finishes.
        need(pins->finalize_after_all_rad_owners(),"checked physical pinned free failed; memory AND quota retained until exit");need(budget->stats().used==0,"staging quota not refunded after checked physical free");
        std::puts("PASS: actual Rad/AQL Engram upload/final-reader/cancel/throw/manual handoff/event reuse; 217B assets, bounded pinned ring, checked process-final pin release. No Ctx/serving/throughput claim.");return 0;
    }catch(const std::exception& error){
        // Failed/unproven teardown deliberately retains registry pins and quota.
        // Do not run an optimistic cleanup that could free an outstanding reader.
        std::fprintf(stderr,"FAIL: %s; retained_pinned_charge=%llu, live_Rad_streams=%d, live_core_owners=%d\n",error.what(),
            (unsigned long long)(pins?pins->retained_charge():0),live_streams,live_core_owners);return 1;
    }
}

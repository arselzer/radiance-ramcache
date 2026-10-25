/* Tiny, explicit-gated HIP staging lifetime checker. Not registered with CTest.
 * Synthetic 217-byte local file, bounded pinned rows and sub-KiB device targets.
 * No model payload, plugin scan, serving API, benchmark or device reset. */
#include "ds41_engram_hip.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace eg=rad::deepseek41::engram;
namespace {
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void hipcheck(hipError_t status){if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));}
void ok(int status,const char* label){if(status!=RAD_OK)throw std::runtime_error(std::string(label)+": native Engram status "+std::to_string(status));}
struct File {
    std::string path;std::vector<uint8_t> bytes;
    File(){
        char name[]="/tmp/ds41-engram-hip.XXXXXX";const int fd=::mkstemp(name);require(fd>=0,"mkstemp");path=name;
        for(int i=0;i<73;++i)bytes.push_back(uint8_t(i*17+3));
        for(int layer=0;layer<2;++layer)for(int row=0;row<4;++row){
            bytes.push_back(0);bytes.push_back(0x3c);
            for(int j=0;j<16;++j)bytes.push_back(uint8_t(((j+row+layer)%16)|(((15-j+row+layer)%16)<<4)));
        }
        size_t done=0;while(done<bytes.size()){
            const auto n=::write(fd,bytes.data()+done,bytes.size()-done);if(n<=0){::close(fd);::unlink(path.c_str());throw std::runtime_error("tiny fixture write");}done+=size_t(n);
        }
        ::close(fd);
    }
    ~File(){::unlink(path.c_str());}
    eg::Iq4NlAsset asset(int layer)const{
        return {path,"b828d527a8c9bdc9420c1f9a586fe75c6b5c808eb96e469ab9cf9bd2a8552aad","gguf-iq4_nl-block-v1",layer,int64_t(bytes.size()),layer==1?73:145,4,32};
    }
};
struct DeviceScope {
    int previous=0;explicit DeviceScope(int device){hipcheck(hipGetDevice(&previous));hipcheck(hipSetDevice(device));}
    ~DeviceScope(){static_cast<void>(hipSetDevice(previous));}
};
struct ConsumerArgs {const void* source=nullptr;void* destination=nullptr;size_t bytes=0;};
int copy_consumer(hipStream_t stream,void* p)noexcept{
    const auto& args=*static_cast<ConsumerArgs*>(p);
    return hipMemcpyAsync(args.destination,args.source,args.bytes,hipMemcpyDeviceToDevice,stream)==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
int throwing_consumer(hipStream_t stream,const eg::DeviceRowTarget&,void* p){
    ok(copy_consumer(stream,p),"throwing consumer queued copy");throw std::runtime_error("synthetic host callback failure after queued reader");
}
std::vector<float> expected(int layer,std::span<const int64_t> rows){
    constexpr int q[16]{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    std::vector<float> out;const int salt=layer==1?0:1;
    for(auto row:rows){for(int j=0;j<16;++j)out.push_back(float(q[(j+row+salt)%16]));for(int j=0;j<16;++j)out.push_back(float(q[(15-j+row+salt)%16]));}
    return out;
}
void verify(int device,const eg::DeviceRowTarget& target,int layer,std::span<const int64_t> rows){
    DeviceScope guard(device);std::vector<float> actual(rows.size()*32);const auto wanted=expected(layer,rows);
    hipcheck(hipMemcpy(actual.data(),reinterpret_cast<void*>(target.owner->address()+target.offset),actual.size()*sizeof(float),hipMemcpyDeviceToHost));
    require(actual==wanted,"decoded row upload differs from independent IQ4_NL codebook");
}
}
int main(int argc,char** argv){
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")){
        std::fprintf(stderr,"usage: ds41-engram-hip-selftest --run-gpu [device]\nRuns tiny GPU work; explicitly approved idle window only. Never called by CTest.\n");return 2;
    }
    std::shared_ptr<eg::HipEngramBackend> backend;
    try{
        int device=0;if(argc==3){size_t used=0;device=std::stoi(argv[2],&used);require(used==std::strlen(argv[2])&&device>=0,"invalid device");}
        int devices=0;hipcheck(hipGetDeviceCount(&devices));require(device<devices,"device out of range");
        hipDeviceProp_t properties{};hipcheck(hipGetDeviceProperties(&properties,device));
        require(!std::strncmp(properties.gcnArchName,"gfx1201",7)&&(!properties.gcnArchName[7]||properties.gcnArchName[7]==':'),"checker requires gfx1201");
        const int caller_device=devices>1?(device+1)%devices:device;hipcheck(hipSetDevice(caller_device));
        const auto tls=[&]{int current=-1;hipcheck(hipGetDevice(&current));require(current==caller_device,"backend did not restore caller device TLS");};
        eg::HipEngramOptions native;native.device=device;native.max_operations=2;native.max_host_allocations=4;native.max_device_allocations=8;
        ok(eg::HipEngramBackend::create("tiny-engram-hip",native,&backend),"create HIP backend");tls();
        auto host_budget=std::make_shared<eg::StagingBudget>(2*1024*1024),device_budget=std::make_shared<eg::StagingBudget>(2*1024*1024);
        File file;eg::Iq4NlCacheOptions cache;cache.max_batch_rows=4;cache.max_io_rows=4;
        std::shared_ptr<eg::Iq4NlLayerStores> source;ok(eg::Iq4NlLayerStores::open("tiny-engram-hip",{file.asset(1),file.asset(14)},16384,cache,&source),"open tiny layer stores");
        std::shared_ptr<eg::EngramStagingPool> pool;ok(eg::EngramStagingPool::create("tiny-engram-hip",source,host_budget,backend,{3,2,32},&pool),"create pinned staging pool");tls();
        std::shared_ptr<eg::HipEngramDeviceBuffer> allocation,readback,third,arena;
        ok(backend->allocate_device(512,device_budget,&allocation),"allocate first device target");ok(backend->allocate_device(512,device_budget,&readback),"allocate readback target");ok(backend->allocate_device(512,device_budget,&third),"allocate third target");tls();
        ok(backend->wrap_device(reinterpret_cast<void*>(allocation->address()+32),320,allocation,&arena),"wrap actual HIP arena suballocation");tls();
        std::shared_ptr<eg::HipEngramDeviceBuffer> invalid;float host=0;
        require(backend->wrap_device(&host,4,allocation,&invalid)==RAD_E_SHAPE,"host pointer accepted as device arena");
        require(backend->wrap_device(reinterpret_cast<void*>(allocation->address()+4080),64,allocation,&invalid)==RAD_E_SHAPE,"oversized native arena accepted");tls();
        const std::vector<int64_t> rows{2,0},one{1};
        eg::DeviceRowTarget target{arena,16,file.asset(1).sha256},second{readback,0,file.asset(14).sha256},last{third,0,file.asset(1).sha256};
        eg::RowTransferTicket first,two,blocked;std::stop_source stop;
        ok(pool->submit(1,rows,target,&first,stop.get_token()),"submit first rows");ok(backend->wait_upload(first),"wait first upload fence");tls();
        require(!first.state().quiescent,"DMA completion incorrectly released staging before final-reader proof");
        verify(device,target,1,rows);tls();
        ConsumerArgs copy{reinterpret_cast<void*>(target.owner->address()+target.offset),reinterpret_cast<void*>(readback->address()+128),rows.size()*32*sizeof(float)};
        ok(backend->enqueue_consumer(first,copy_consumer,&copy),"enqueue first D2D consumer");require(backend->finish_no_readers(first)==RAD_E_STATE,"no-reader seal ignored queued reader");tls();
        stop.request_stop();first.cancel();require(!first.state().quiescent,"request cancellation fabricated quiescence");
        ok(pool->submit(14,one,second,&two),"submit layer14 rows");
        require(pool->submit(1,one,last,&blocked)==RAD_E_FULL,"bounded native operation pool did not reject third upload");require(blocked.state().quiescent,"no-work operation exhaustion retained unsafe work");
        ok(backend->finish_readers(first),"seal first final reader");require(backend->enqueue_consumer(first,copy_consumer,&copy)==RAD_E_STATE,"consumer allowed after final-reader seal");
        auto first_record=first.backend_record(*backend);require(bool(first_record),"missing typed transfer record");
        require(backend->drain(first_record).quiescent,"sealed upload+reader stream failed to drain");tls();
        eg::DeviceRowTarget copied{readback,128,file.asset(1).sha256};verify(device,copied,1,rows);tls();
        ok(backend->finish_no_readers(two),"seal second no-reader scope");require(backend->drain(two.backend_record(*backend)).quiescent,"no-reader seal failed to drain");verify(device,second,14,one);tls();
        pool->reap();ok(pool->submit(1,one,last,&blocked),"reuse operation generation");require(first.state().quiescent,"reused event generation changed old ticket state");
        ok(backend->wait_upload(blocked),"wait reused upload fence");require(!blocked.state().quiescent,"new generation reused prior reader event state");
        ConsumerArgs partial{reinterpret_cast<void*>(last.owner->address()),reinterpret_cast<void*>(readback->address()),one.size()*32*sizeof(float)};
        require(backend->consume(blocked,throwing_consumer,&partial)==RAD_E_STATE,"throwing consumer did not report host callback error");
        require(backend->drain(blocked.backend_record(*backend)).quiescent,"throwing consumer left a healthy stream unsealed");verify(device,last,1,one);tls();
        eg::DeviceRowTarget partial_result{readback,0,file.asset(1).sha256};verify(device,partial_result,1,one);
        std::vector<float> cpu(32);ok(source->row_lookup()(1,one,32,cpu,nullptr),"shared cache after cancellation");
        require(pool->shutdown().quiescent,"pool shutdown not quiescent after final readers");
        const auto stats=backend->native_stats();require(stats.queued==3&&stats.active_operations==0&&stats.reader_fences==3,"native staging counters inconsistent");
        first.reset();two.reset();blocked.reset();target.owner.reset();second.owner.reset();last.owner.reset();copied.owner.reset();partial_result.owner.reset();
        // first_record is intentionally kept through this scope; it may retain the
        // first physical staging/destination allocation after pool destruction.
        pool.reset();arena.reset();allocation.reset();readback.reset();third.reset();
        require(host_budget->stats().used>0,"record did not retain charged physical host storage");
        require(!backend->close().quiescent,"close freed live record-owned allocation");tls();
        first_record.reset();require(host_budget->stats().used==0,"physical host free did not refund staging cap");
        require(device_budget->stats().used==0,"physical device free did not refund device cap");
        require(backend->close().quiescent,"final native close did not release stream/events");tls();
        std::fprintf(stdout,"PASS native Engram upload/final-reader fences, cancellation, callback exception, bounds, two layers, event generations, budgets and TLS (device %d)\n",device);
        return 0;
    }catch(const std::exception& e){
        std::fprintf(stderr,"FAIL native Engram checker: %s\n",e.what());if(backend)static_cast<void>(backend->close());return 1;
    }
}

// Reuses actual bounded TP1 HostOnly native source/Pools/Mover construction.
// This tests the new real transfer fence without granting TP4 HOST capability.
#include "regressions/compact/allocation_hooks.h"
#include "mover_wait_fixture.h"
namespace fence_fault {
bool tracing=false,fail_after_real=false;
RadStream compute=nullptr,last=nullptr;
size_t calls=0,negative_witness=0;
}
extern "C" int __real_rad_stream_sync(RadStream);
extern "C" int __wrap_rad_stream_sync(RadStream stream){
    const int real=__real_rad_stream_sync(stream);
    if(fence_fault::tracing){++fence_fault::calls;fence_fault::last=stream;
        if(fence_fault::fail_after_real&&stream!=fence_fault::compute&&real==RAD_OK){
            fence_fault::fail_after_real=false;++fence_fault::negative_witness;return RAD_E_IO;}}
    return real;
}
namespace {
size_t fence_groups=0,fence_refusals=0;
struct Session {
    Fixture& fixture;uint64_t token=0;
    explicit Session(Fixture& f):fixture(f){good(f.mover.native_record_session_open(&token),"actual raw Native Mover session");}
    ~Session(){fence_fault::tracing=false;fence_fault::fail_after_real=false;
        if(token&&(fixture.mover.record_restore(token)<0||fixture.mover.record_session_close(token)<0))std::abort();}
};
void trace(bool refuse=false){fence_fault::tracing=true;fence_fault::fail_after_real=refuse;fence_fault::calls=0;fence_fault::last=nullptr;}
void stop_trace(){fence_fault::tracing=false;fence_fault::fail_after_real=false;}
void actual_fence(const std::string& home,bool refuse){++fence_groups;
    Fixture f(home,true,3*32768,8,65536,false,true);
    need(f.inventory->limits().backend_profile==NativeBackendProfile::HostOnly&&
         !f.builder->program().requires_glm_main_tp4_cache_fail_stop(),"no TP4 HOST capability manufactured");
    fence_fault::compute=f.compute.value;Session session(f);
    trace();good(f.mover.native_record_wait_prepared(session.token,{}),"resident-only empty boundary uses actual transfer fence");
    stop_trace();need(fence_fault::calls==1&&fence_fault::last&&fence_fault::last!=f.compute.value,"empty boundary fences mover, not compute");
    const auto& owned=f.inventory->owned().front();const auto& unit=f.plan.units[size_t(owned.unit)];
    const auto regions=f.mover.native_record_regions();int32_t region=-1;
    for(size_t i=0;i<regions.size();++i)if(!regions[i].sentinel&&
        f.inventory->find(regions[i].exemplar)->klass==owned.klass){region=int32_t(i);break;}
    need(region>=0,"actual matching native consumer region");
    NativeRecordBatch leases;const uint32_t record=owned.record;
    good(f.source->acquire({&record,1},&leases),"actual bounded source lease");need(leases.leases.size()==1,"exact source union");
    const auto bytes=leases.leases[0].bytes;const auto lifetime=leases.leases[0].lifetime;
    const RecordPublication publication{owned.unit,region};
    trace();need(f.mover.native_record_wait_prepared(session.token,{&publication,1})<0,"unpublished batch cannot become READY");
    stop_trace();need(!fence_fault::calls,"unpublished refusal before stream fence");++fence_refusals;
    good(f.mover.native_record_upload(session.token,owned.unit,region,bytes),"checksum-bound actual native upload");
    trace();need(f.mover.native_record_wait_prepared(session.token,{})<0,"busy unpublished upload cannot pass empty READY");
    stop_trace();need(!fence_fault::calls,"busy empty refusal before stream fence");++fence_refusals;
    good(f.mover.record_publish(session.token,{&publication,1}),"actual Residency generation publication");
    std::array<WeightSlot,4> before{};for(size_t m=0;m<4;++m){const auto* slot=f.mover.residency()->lookup(unit.weights[m]);need(slot,"published member slot");before[m]=*slot;}
    auto changed=publication;++changed.region;
    trace();need(f.mover.native_record_wait_prepared(session.token,{&changed,1})<0,"same-size changed publication refused");
    stop_trace();need(!fence_fault::calls,"publication refusal before real wait");++fence_refusals;
    trace();need(f.mover.native_record_wait_prepared(session.token+1,{&publication,1})<0,"foreign token refused");
    stop_trace();need(!fence_fault::calls,"token refusal before real wait");++fence_refusals;
    trace(refuse);allocation_fault::begin();const int status=f.mover.native_record_wait_prepared(session.token,{&publication,1});
    const auto heap_calls=allocation_fault::calls;allocation_fault::end();stop_trace();
    need(status==(refuse?RAD_E_IO:RAD_OK),"actual mover fence outcome propagated");
    need(fence_fault::calls==1&&fence_fault::last&&fence_fault::last!=f.compute.value&&!heap_calls,"one REAL mover fence, no compute drain or heap allocation");
    need(leases.leases.size()==1&&leases.leases[0].lifetime==lifetime&&leases.leases[0].bytes.data()==bytes.data(),"source lease retained across completion/refusal");
    for(size_t m=0;m<4;++m){const auto* slot=f.mover.residency()->lookup(unit.weights[m]);
        need(slot&&slot->ptr==before[m].ptr&&slot->ready==before[m].ready&&slot->generation==before[m].generation,"wait retains exact publication/event/generation");
        need(!std::memcmp(slot->ptr,bytes.data()+unit.offset[m],size_t(f.builder->program().weights[unit.weights[m]].stored_bytes)),"real upload completed before successful READY or injected semantic refusal");}
    if(refuse){need(fence_fault::negative_witness==1,"semantic refusal witnessed after successful actual sync");
        trace();need(f.mover.native_record_wait_prepared(session.token,{&publication,1})<0,"failed upload generation remains terminal until explicit retirement");
        stop_trace();need(!fence_fault::calls,"sticky upload refusal does not drain/release");++fence_refusals;}
    good(f.mover.record_restore(session.token),"explicit normal retirement drains both actual streams");
    for(const auto h:unit.weights){const auto* slot=f.mover.residency()->lookup(h);need(slot&&slot->ptr&&!slot->ready,"sentinel restore occurs only in explicit retirement");}
    good(f.mover.native_record_guards_host(),"actual native allocation guards intact");
}
}
int main(int argc,char** argv){try{
    need(argc==3&&!std::strcmp(argv[1],"--radiance-home"),"usage: glm53_tp4_mover_fence_check --radiance-home MATCHING_NEW_HOST_HOME");
    need(device_is_host(),"HOST only, no TP4 payload or GPU execution");actual_fence(argv[2],false);actual_fence(argv[2],true);
    need(fence_groups==2&&fence_fault::negative_witness==1,"actual fence cases complete");
    std::cout<<"PASS actual Native mover fence groups="<<fence_groups<<" checks="<<checks<<" refusals="<<fence_refusals<<"; HostOnly component, no TP4 payload/Ready claim\n";return 0;
}catch(const std::exception& error){stop_trace();allocation_fault::end();std::cerr<<"FAIL actual Native mover fence: "<<error.what()<<'\n';return 1;}}

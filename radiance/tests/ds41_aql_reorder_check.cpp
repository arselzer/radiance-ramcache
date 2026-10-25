/* Explicit manual real-Builder/Ctx/AQL/replay gate. Never automatic *_test.cpp. */
#include "ds41_reorder_heads_core_fixture.h"
#include <cstdio>
#include <cstring>
#include <string>
namespace c=ds41_reorder_core;
namespace {
void usage(){std::puts("ds41-aql-reorder-check --run-gpu --home ISOLATED_BUILD_HOME [--host-kargs]\nTiny actual local reorder binding/replay gate; one visible gfx1201; no models/Pools/service.");}
}
int main(int argc,char** argv) {
    bool run=false,host_kargs=false;std::string home;
    for(int i=1;i<argc;++i) {
        const std::string option=argv[i];if(option=="--help"||option=="-h"){usage();return 0;}
        if(option=="--run-gpu")run=true;else if(option=="--host-kargs")host_kargs=true;
        else if(option=="--home"&&i+1<argc)home=argv[++i];else{std::fprintf(stderr,"unknown/incomplete option: %s\n",option.c_str());return 2;}
    }
    if(!run||home.empty()){usage();return 2;}
    // No explicit registry/device query/allocation/stream before this main gate. Linked HIP
    // fatbinary registration may have run before main; parent owns the idle execution window.
    try {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)c::okay(rad::aql_force_host_kargs(),"force host kernel argument rings");
#else
        (void)host_kargs;throw std::runtime_error("checker requires HIP-enabled rad_core");
#endif
        c::need(rad_dev_count()==1,"expose exactly one GPU via ROCR_VISIBLE_DEVICES");
        c::need(!rad::device_is_host()&&std::strcmp(rad::device_backend_name(),"aql")==0,"actual AQL backend required");
        RadDeviceProps props{};c::okay(rad_dev_props(0,&props),"read selected GPU");
        c::need(!props.is_host_backend&&std::strcmp(props.arch,"gfx1201")==0,"requires exactly one visible gfx1201");c::okay(rad_dev_set(0),"select only visible GPU");
        rad::Registry registry;c::okay(registry.load_plugin(home+"/experimental/kernels/libds41.so",0),"load isolated actual DS41 GPU rows");c::Stream stream;
        size_t passes=0,graphs=0;int64_t played=0,peak_arena=0;
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int tp:{1,2,4,8})for(bool flat:{false,true}) {
            c::Harness h(registry,stream.value,dtype,tp,flat,true);peak_arena=std::max(peak_arena,h.arena_bytes);
            if(flat)for(int rows:{1,3,5})for(int iteration=0;iteration<9;++iteration){h.pass(rows,iteration+100*rows);++passes;}
            else for(int iteration=0;iteration<9;++iteration){h.pass(c::fixed_rows,iteration);++passes;}
            c::need(h.ctx.records_passes()&&!h.ctx.checks_device_status(),"scratchless graph lost ordinary replay eligibility");
            c::need(h.ctx.pass_counts().played>0,"graph never reached real AQL tape replay");played+=h.ctx.pass_counts().played;++graphs;
        }
        c::okay(rad_stream_sync(stream.value),"final actual AQL drain");c::need(graphs==24&&passes==432&&played>=24,"actual reorder replay coverage count drifted");
        std::printf("PASS: %zu actual Builder/Ctx/AQL local reorder graphs, %zu bit-changing guarded passes, %lld real tape plays, both metadata profiles, TP1/2/4/8, three dtypes; peak one-graph arena=%lld B.\n",graphs,passes,(long long)played,(long long)peak_arena);
        std::puts("No launch replacement/status collector/scratch/Pools; stream argument/bounce rings and module/driver metadata excluded from arena receipt. Not a communication/multi-GPU/source-cache proof.");return 0;
    } catch(const std::exception& error) {
        c::active=nullptr;std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;
    }
}

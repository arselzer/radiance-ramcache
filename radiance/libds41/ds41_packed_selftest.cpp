/* Explicitly gated, tiny device checker for a later approved idle window. Not run by CTest.
 * CPU decode/rounding is independently tested against GGML + generic canonical oracles.
 * This checker compares device reductions and indexing, guards every allocation and exercises
 * malformed routed IDs/null expert pointers. It never opens model weights or serving APIs. */
#include <hip/hip_runtime.h>
#include "ds41_packed_args.h"
#include "ds41_latent.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace p=rad::ds41::packed;
namespace {
void check(hipError_t status) {
    if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
}
struct Device {
    void* base=nullptr;size_t size=0;
    static constexpr size_t guard=64;
    explicit Device(const std::vector<uint8_t>& bytes):size(bytes.size()) {
        check(hipMalloc(&base,size+2*guard));
        const auto status=hipMemset(base,0xA7,size+2*guard);
        if(status!=hipSuccess) {static_cast<void>(hipFree(base));base=nullptr;check(status);}
        const auto copied=hipMemcpy(data(),bytes.data(),size,hipMemcpyHostToDevice);
        if(copied!=hipSuccess) {static_cast<void>(hipFree(base));base=nullptr;check(copied);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device() {if(base)static_cast<void>(hipFree(base));}
    void* data() const {return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> bytes(size+2*guard);
        check(hipMemcpy(bytes.data(),base,bytes.size(),hipMemcpyDeviceToHost));
        if(!std::all_of(bytes.begin(),bytes.begin()+guard,[](uint8_t b){return b==0xA7;})||
           !std::all_of(bytes.end()-guard,bytes.end(),[](uint8_t b){return b==0xA7;}))
            throw std::runtime_error("device allocation red zone changed");
        return {bytes.begin()+guard,bytes.end()-guard};
    }
};
template<class T> std::vector<uint8_t> bytes(const std::vector<T>& values) {
    std::vector<uint8_t> out(values.size()*sizeof(T));std::memcpy(out.data(),values.data(),out.size());return out;
}
RadTensor matrix(void* data,uint32_t dtype,int64_t rows,int64_t cols,int64_t pitch) {
    RadTensor t{};t.data=data;t.dtype=dtype;t.rank=2;t.shape[0]=rows;t.shape[1]=cols;
    t.stride[0]=pitch;t.stride[1]=1;return t;
}
RadTensor vector(void* data,int64_t count) {
    RadTensor t{};t.data=data;t.dtype=RAD_I32;t.rank=1;t.shape[0]=count;t.stride[0]=1;return t;
}
void put(std::vector<uint8_t>& data,size_t i,uint32_t dtype,float v) {
    if(dtype==RAD_F32)std::memcpy(data.data()+4*i,&v,4);
    else {const auto h=dtype==RAD_BF16?p::to_bf16(v):p::to_half(v);std::memcpy(data.data()+2*i,&h,2);}
}
float get(const std::vector<uint8_t>& data,size_t i,uint32_t dtype) {
    if(dtype==RAD_F32) {float v;std::memcpy(&v,data.data()+4*i,4);return v;}
    uint16_t h;std::memcpy(&h,data.data()+2*i,2);return dtype==RAD_BF16?p::from_bits(uint32_t(h)<<16):p::half(h);
}
void verify_numbers(const std::vector<uint8_t>& actual,const std::vector<uint8_t>& expected,
                      uint32_t dtype,int64_t jobs,int64_t n,int64_t pitch,
                      const std::vector<int32_t>& errors) {
    const size_t width=size_t(rad_dtype_bits(dtype)/8);
    for(int64_t r=0;r<jobs;++r) {
        for(int64_t c=0;c<n;++c) {
            const size_t at=size_t(r*pitch+c);const float got=get(actual,at,dtype),want=get(expected,at,dtype);
            if(!errors.empty()&&errors[size_t(r)]!=0) {
                if(!std::isnan(got))throw std::runtime_error("malformed route did not produce NaN");
                continue;
            }
            const float rel=dtype==RAD_F32?0.0002f:dtype==RAD_F16?0.0011f:0.008f;
            const float abs=dtype==RAD_F32?0.00004f:0.0001f;
            if(!std::isfinite(got)||!std::isfinite(want)||std::abs(got-want)>abs+rel*std::abs(want)) {
                std::fprintf(stderr,"row=%lld col=%lld expected=%g actual=%g\n",(long long)r,(long long)c,want,got);
                throw std::runtime_error("device/CPU packed GEMM mismatch");
            }
        }
        const size_t lo=size_t(r*pitch+n)*width,hi=size_t((r+1)*pitch)*width;
        if(!std::equal(actual.begin()+lo,actual.begin()+hi,expected.begin()+lo))
            throw std::runtime_error("output row padding changed");
    }
}
void run(RadLaunchFn launch,p::Type type,bool routed,uint32_t ad,uint32_t yd,
          int64_t m,int64_t n,int64_t k,int malformed=0,bool logical_weights=false) {
    const int64_t e=3,jobs=routed?2*m:m,ap=k+4,yp=n+3;
    const int64_t rb=k/p::block_values(type)*p::block_bytes(type);
    std::mt19937 rng(unsigned(1017+k+n+m+malformed));
    std::vector<uint8_t> a(size_t(m*ap)*(rad_dtype_bits(ad)/8),0x3A);
    for(int64_t r=0;r<m;++r)for(int64_t c=0;c<k;++c)put(a,size_t(r*ap+c),ad,float(int(rng()%2001)-1000)/2048);
    std::vector<std::vector<uint8_t>> weights(size_t(routed?e:1));
    for(auto& w:weights) {
        w.resize(size_t(n*rb));for(auto& b:w)b=uint8_t(rng());
        for(size_t at=0;at<w.size();at+=size_t(p::block_bytes(type))) {
            w[at]=0;w[at+1]=0x28; // finite 1/32 F16 scale, no speculative easy-text effect.
            if(type==p::Type::q5_k) {w[at+2]=0;w[at+3]=0x24;}
        }
    }
    std::vector<int32_t> ids(size_t(jobs),0),rows(size_t(jobs),0),status(size_t(jobs),0x12345678),errors(size_t(jobs),0);
    for(int64_t j=0;j<jobs;++j) {ids[size_t(j)]=int32_t((j*2+1)%e);rows[size_t(j)]=int32_t((j*3+1)%m);}
    std::vector<const uint8_t*> host_table;for(const auto& w:weights)host_table.push_back(w.data());
    std::vector<uint8_t> expected(size_t(jobs*yp)*(rad_dtype_bits(yd)/8),0xD3);
    RadTensor t[6]{};t[0]=matrix(a.data(),ad,m,k,ap);t[1]=matrix(weights[0].data(),RAD_U8,n,rb,rb);
    if(routed) {
        t[1].data=host_table.data();t[1].rank=3;t[1].shape[0]=e;t[1].shape[1]=n;t[1].shape[2]=rb;
        t[1].stride[0]=0;t[1].stride[1]=rb;t[1].stride[2]=1;
        t[2]=vector(ids.data(),jobs);t[3]=vector(rows.data(),jobs);t[4]=matrix(expected.data(),yd,jobs,n,yp);t[5]=vector(status.data(),jobs);
    } else t[2]=matrix(expected.data(),yd,jobs,n,yp);
    if(logical_weights) {
        t[1].dtype=p::storage_dtype(type);
        if(routed) {t[1].shape[2]=k;t[1].stride[1]=k;}
        else {t[1].shape[1]=k;t[1].stride[0]=k;}
    }
    const char* an=rad_dtype_name(ad),*yn=rad_dtype_name(yd);
    const RadParam params[]{RAD_INT("M",m+1),RAD_INT("N",n),RAD_INT("K",k),RAD_STR("format",p::type_name(type)),
                           RAD_STR("a_dtype",an),RAD_STR("y_dtype",yn),RAD_INT("R",jobs+1),RAD_INT("n_expert",e)};
    RadArgs args{};args.t=t;args.n_t=routed?6:3;args.p=params;args.n_p=routed?8:6;
    const int reference=routed?ds41_ref_packed_routed(&args,nullptr):ds41_ref_packed_gemm(&args,nullptr);
    if(reference!=RAD_OK)throw std::runtime_error("valid CPU oracle refused");
    std::vector<std::unique_ptr<Device>> wdev;
    for(const auto& w:weights)wdev.push_back(std::make_unique<Device>(w));
    std::vector<const uint8_t*> device_table;for(const auto& w:wdev)device_table.push_back(static_cast<const uint8_t*>(w->data()));
    if(malformed==1) {ids.back()=int32_t(e);errors.back()=RAD_E_SHAPE;}
    if(malformed==2) {
        const auto bad_expert=ids.back();device_table[size_t(bad_expert)]=nullptr;
        for(size_t j=0;j<ids.size();++j)if(ids[j]==bad_expert)errors[j]=RAD_E_STATE;
    }
    auto original_status=std::vector<int32_t>(size_t(jobs),0x12345678);
    Device da(a),dy(std::vector<uint8_t>(expected.size(),0xD3));
    auto bt=bytes(device_table),bi=bytes(ids),br=bytes(rows),bs=bytes(original_status);
    Device dt(bt),di(bi),dr(br),ds(bs);
    t[0].data=da.data();t[1].data=routed?dt.data():wdev[0]->data();
    t[routed?4:2].data=dy.data();
    if(routed) {t[2].data=di.data();t[3].data=dr.data();t[5].data=ds.data();}
    if(launch(&args,nullptr)!=RAD_OK)throw std::runtime_error("device launch refused valid geometry");
    check(hipDeviceSynchronize());const auto actual=dy.read();
    verify_numbers(actual,expected,yd,jobs,n,yp,routed?errors:std::vector<int32_t>{});
    if(da.read()!=a||dt.read()!=bt||di.read()!=bi||dr.read()!=br)throw std::runtime_error("read-only operand changed");
    for(size_t j=0;j<wdev.size();++j)if(wdev[j]->read()!=weights[j])throw std::runtime_error("read-only weight changed");
    const auto received=ds.read();
    if(routed&&received!=bytes(errors))throw std::runtime_error("device route status mismatch");
    if(!routed&&received!=bs)throw std::runtime_error("unused status buffer changed");
}
void run_latent(RadLaunchFn store,RadLaunchFn gather,uint32_t dtype,int64_t d,int64_t ratio,int64_t b,int error=0) {
    const int64_t rows=4,blocks=3,pitch=d+3,eb=rad_dtype_bits(dtype)/8;
    std::mt19937 rng(unsigned(d+ratio+b+dtype));
    std::vector<uint8_t> input(size_t(rows*pitch*eb));for(auto& value:input)value=uint8_t(rng());
    std::vector<uint8_t> cache(size_t(blocks*b*(d/ratio)*eb),0xC7),output(input.size(),0xA1);
    std::vector<int64_t> first{0,b-ratio,b,2*b+ratio},last,positions;
    for(auto s:first) {last.push_back(s+ratio-1);positions.push_back(100*b+s%b+ratio-1);}
    if(error==1)positions.back()=-1;
    if(error==2) {last.back()=last.front();positions.back()=positions.front();}
    std::vector<int32_t> status(size_t(rows),0x12345678),wanted_status(size_t(rows),error==1?RAD_E_SHAPE:error==2?RAD_E_INVAL:RAD_OK);
    RadTensor ct{};ct.dtype=dtype;ct.rank=4;ct.shape[0]=blocks;ct.shape[1]=1;ct.shape[2]=b;ct.shape[3]=d/ratio;
    ct.stride[0]=b*d/ratio;ct.stride[1]=b*d/ratio;ct.stride[2]=d/ratio;ct.stride[3]=1;ct.data=cache.data();
    const RadParam params[]{RAD_INT("M",rows+1),RAD_INT("dimension",d),RAD_INT("ratio",ratio),RAD_INT("block_size",b),
        RAD_INT("n_blocks",blocks),RAD_STR("dtype",rad_dtype_name(dtype))};
    RadTensor t[]{matrix(input.data(),dtype,rows,d,pitch),vector(positions.data(),rows),vector(last.data(),rows),ct,vector(status.data(),rows)};
    t[1].dtype=RAD_I64;t[2].dtype=RAD_I64;
    RadArgs args{};args.t=t;args.n_t=5;args.p=params;args.n_p=6;
    const int ref=ds41_ref_latent_store(&args,nullptr);
    if((error==1||error==2)?ref>=0:ref!=RAD_OK)throw std::runtime_error("CPU latent store unexpected status");
    const auto expected_cache=cache;
    Device dx(input),dc(std::vector<uint8_t>(cache.size(),0xC7)),dp(bytes(positions)),dl(bytes(last)),ds(bytes(std::vector<int32_t>(size_t(rows),0x12345678)));
    t[0].data=dx.data();t[1].data=dp.data();t[2].data=dl.data();t[3].data=dc.data();t[4].data=ds.data();
    if(store(&args,nullptr)!=RAD_OK)throw std::runtime_error("GPU latent store enqueue refused");
    check(hipDeviceSynchronize());
    if(dc.read()!=expected_cache||ds.read()!=bytes(wanted_status))throw std::runtime_error("GPU latent store payload/status mismatch");
    if(dx.read()!=input||dp.read()!=bytes(positions)||dl.read()!=bytes(last))throw std::runtime_error("latent store changed read-only input");
    if(error==1||error==2)return;
    if(error==3)first.back()=-1;
    ct.data=cache.data();RadTensor g[]{ct,vector(first.data(),rows),matrix(output.data(),dtype,rows,d,pitch),vector(status.data(),rows)};
    g[1].dtype=RAD_I64;args.t=g;args.n_t=4;
    const int gather_ref=ds41_ref_latent_gather(&args,nullptr);
    if(error==3?gather_ref>=0:gather_ref!=RAD_OK)throw std::runtime_error("CPU latent gather unexpected status");
    Device df(bytes(first)),dy(std::vector<uint8_t>(output.size(),0xA1));
    g[0].data=dc.data();g[1].data=df.data();g[2].data=dy.data();g[3].data=ds.data();
    if(gather(&args,nullptr)!=RAD_OK)throw std::runtime_error("GPU latent gather enqueue refused");
    check(hipDeviceSynchronize());wanted_status.assign(size_t(rows),error==3?RAD_E_SHAPE:RAD_OK);
    if(dy.read()!=output||ds.read()!=bytes(wanted_status)||dc.read()!=cache||df.read()!=bytes(first))
        throw std::runtime_error("GPU latent gather payload/status/read-only mismatch");
}
}
int main(int argc,char** argv) {
    if((argc!=3&&argc!=4)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-packed-selftest --run-gpu /path/libds41.so [device]\n"
                            "Runs GPU work. Use only in an approved idle window; never called by CTest.\n");
        return 2;
    }
    void* plugin=nullptr;
    try {
        int device=0;
        if(argc==4) {size_t used=0;device=std::stoi(argv[3],&used);if(used!=std::strlen(argv[3])||device<0)throw std::runtime_error("invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        if(std::strncmp(props.gcnArchName,"gfx1201",7)||
           (props.gcnArchName[7]&&props.gcnArchName[7]!=':'))throw std::runtime_error("checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        const auto abi=reinterpret_cast<uint32_t(*)()>(dlsym(plugin,"rad_plugin_abi_version"));
        const auto dense=reinterpret_cast<RadLaunchFn>(dlsym(plugin,"ds41_packed_gemm"));
        const auto routed=reinterpret_cast<RadLaunchFn>(dlsym(plugin,"ds41_packed_routed"));
        const auto latent_store=reinterpret_cast<RadLaunchFn>(dlsym(plugin,"ds41_latent_store"));
        const auto latent_gather=reinterpret_cast<RadLaunchFn>(dlsym(plugin,"ds41_latent_gather"));
        if(!abi||abi()!=RAD_ABI_VERSION||!dense||!routed||!latent_store||!latent_gather)throw std::runtime_error("invalid experimental plugin ABI");
        size_t cases=0;
        for(unsigned f=0;f<6;++f)for(const auto ad:{RAD_F32,RAD_F16,RAD_BF16})for(const auto yd:{RAD_F32,RAD_F16,RAD_BF16})
            for(int scenario=0;scenario<3;++scenario) {
                const int64_t m=scenario==0?1:scenario==1?3:9,n=scenario==0?1:scenario==1?7:13,k=256*(scenario+1);
                run(dense,p::Type(f),false,ad,yd,m,n,k);++cases;
                run(routed,p::Type(f),true,ad,yd,m,n,k);++cases;
            }
        for(unsigned f=0;f<6;++f)for(int error:{1,2}) {run(routed,p::Type(f),true,RAD_F32,RAD_F32,3,7,256,error);++cases;}
        for(unsigned f=0;f<6;++f)for(bool route:{false,true}) {
            run(route?routed:dense,p::Type(f),route,RAD_F32,RAD_BF16,3,7,512,0,true);++cases;
        }
        size_t latent_cases=0;
        for(const auto dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int64_t dim:{128,512})for(int64_t ratio:{1,2})for(int64_t block:{4,8}) {
            run_latent(latent_store,latent_gather,dtype,dim,ratio,block);++latent_cases;
        }
        for(const auto dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int error:{1,2,3}) {
            run_latent(latent_store,latent_gather,dtype,512,2,4,error);++latent_cases;
        }
        check(hipDeviceSynchronize());dlclose(plugin);plugin=nullptr;
        std::printf("PASS: %zu packed + %zu latent GPU cases on %s, guarded inputs/outputs; not a model-quality/performance test\n",cases,latent_cases,props.gcnArchName);
        return 0;
    } catch(const std::exception& e) {
        if(plugin)dlclose(plugin);
        std::fprintf(stderr,"ds41-packed-selftest: %s\n",e.what());return 1;
    }
}

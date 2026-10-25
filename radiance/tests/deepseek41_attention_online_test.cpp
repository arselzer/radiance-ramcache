/* Independent hand/rational/long-double witnesses for the online64 BF16 numerator
 * candidate. No TileLang, native kernel, shared baseline replacement or model execution. */
#include "rad_test.h"
#include "../arch/deepseek41/attention_online_ref.h"
#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace a=rad::deepseek41::attention_online;
namespace {
long double exact_bf16(uint16_t code){
    const unsigned e=(code>>7)&255u,m=code&127u;const long double value=e?std::ldexp(static_cast<long double>(128u+m),int(e)-134):std::ldexp(static_cast<long double>(m),-133);
    return code&0x8000u?-value:value;
}
/* Independent nearest-neighbor oracle; no bit-add BF16 converter from the implementation. */
uint16_t nearest(long double value){
    const bool sign=std::signbit(value);const long double magnitude=std::fabs(value);
    if(!std::isfinite(value)||magnitude>exact_bf16(0x7F7F))throw std::runtime_error("test oracle finite range");
    unsigned lo=0,hi=0x7F7F;
    while(lo+1<hi){const unsigned mid=(lo+hi)/2;if(exact_bf16(uint16_t(mid))<magnitude)lo=mid;else hi=mid;}
    const long double dl=magnitude-exact_bf16(uint16_t(lo)),dh=exact_bf16(uint16_t(hi))-magnitude;
    const unsigned chosen=dl<dh?lo:dh<dl?hi:(lo&1u?hi:lo);
    return uint16_t(chosen|(sign?0x8000u:0));
}
struct Fixture {
    int64_t heads,dim,rows;std::vector<uint16_t> query,kv,guarded;std::vector<float> sinks;std::vector<int32_t> selected;
    std::unique_ptr<a::Workspace> workspace=std::make_unique<a::Workspace>();
    Fixture(int64_t h=1,int64_t d=2,int64_t n=2,int64_t k=2):heads(h),dim(d),rows(n),query(size_t(h*d),0),kv(size_t(n*d),0),guarded(size_t(h*d)+8,0xD3D3),sinks(size_t(h),-INFINITY),selected(size_t(k),0){}
    a::View view()const{return {heads,dim,rows,query,kv,sinks,selected,1.f};}
    std::span<uint16_t> output(){return {guarded.data()+4,size_t(heads*dim)};}
    void q(int64_t i,float x){query[size_t(i)]=nearest(x);}void value(int64_t i,float x){kv[size_t(i)]=nearest(x);}
};
void same_inputs(const Fixture& f,const std::vector<uint16_t>& q,const std::vector<uint16_t>& kv,const std::vector<float>& sinks,const std::vector<int32_t>& selected){CHECK(f.query==q);CHECK(f.kv==kv);CHECK(std::memcmp(f.sinks.data(),sinks.data(),sinks.size()*4)==0);CHECK(f.selected==selected);}
void poison_workspace(a::Workspace& workspace){
    static_assert(std::is_trivially_copyable_v<a::Workspace>);
    auto bytes=std::as_writable_bytes(std::span(&workspace,1));
    std::fill(bytes.begin(),bytes.end(),std::byte{0xB7});
}
void guards(const Fixture& f){for(int i=0;i<4;++i){CHECK_EQ(f.guarded[size_t(i)],0xD3D3);CHECK_EQ(f.guarded[f.guarded.size()-1-size_t(i)],0xD3D3);}}
struct Pages {
    void* data=nullptr;size_t bytes;
    explicit Pages(size_t count,int protection=PROT_READ|PROT_WRITE):bytes(count){data=mmap(nullptr,bytes,protection,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(data==MAP_FAILED)throw std::runtime_error("mmap");}
    ~Pages(){munmap(data,bytes);}
};
}
TEST(ds41_attention_online_candidate_counts_and_block_boundaries_match_exact_rational_oracle){
    static_assert(sizeof(a::Workspace)<72*1024);
    for(int count:{0,1,63,64,65,127,128,640})for(int heads:{1,3})for(int dim:{1,8}){
        Fixture f(heads,dim,1,count);std::fill(f.kv.begin(),f.kv.end(),nearest(1));std::fill(f.sinks.begin(),f.sinks.end(),0.f);
        const auto input=f.query,kv=f.kv;const auto sinks=f.sinks;const auto ids=f.selected;
        REQUIRE_EQ(a::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);
        const uint16_t expected=nearest(count?static_cast<long double>(count)/(count+1):0.L);
        for(auto code:f.output())CHECK_EQ(code,expected);
        same_inputs(f,input,kv,sinks,ids);guards(f);
    }
    Fixture maximum(32,1024,1,1);std::fill(maximum.kv.begin(),maximum.kv.end(),nearest(1));std::fill(maximum.sinks.begin(),maximum.sinks.end(),0.f);
    REQUIRE_EQ(a::evaluate(maximum.view(),maximum.output(),maximum.workspace.get()),RAD_OK);
    for(auto code:maximum.output())CHECK_EQ(code,nearest(.5L));
    guards(maximum);
}
TEST(ds41_attention_online_rounds_unnormalized_numerator_before_PV_and_not_normalized_probabilities){
    Fixture f;f.q(0,1);f.value(0,0);f.value(1,0);f.value(2,-2);f.value(3,1);f.selected={0,1};
    REQUIRE_EQ(a::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);
    const long double p=std::exp(-2.L),rounded_p=exact_bf16(nearest(p));
    const uint16_t correct=nearest(rounded_p/(1+p)),normalized_probability=nearest(p/(1+p));
    CHECK_EQ(correct,0x3DF5);CHECK_EQ(normalized_probability,0x3DF4);CHECK(correct!=normalized_probability);CHECK_EQ(f.output()[1],correct);guards(f);
    // The high-precision witness is deliberately away from the final BF16 midpoint;
    // it establishes the arithmetic boundary without claiming native exp/GEMM parity.
}
TEST(ds41_attention_online_preserves_duplicate_multiplicity_padding_slots_and_candidate_block_order){
    Fixture duplicate(1,1,2,2);duplicate.value(0,1);duplicate.value(1,3);duplicate.selected={0,1};
    REQUIRE_EQ(a::evaluate(duplicate.view(),duplicate.output(),duplicate.workspace.get()),RAD_OK);CHECK_EQ(duplicate.output()[0],nearest(2));
    duplicate.selected={0,0,1};REQUIRE_EQ(a::evaluate(duplicate.view(),duplicate.output(),duplicate.workspace.get()),RAD_OK);CHECK_EQ(duplicate.output()[0],nearest(5.L/3));
    Fixture order(1,2,2,65);order.q(0,1);order.value(0,0);order.value(1,4);order.value(2,-.5f);order.value(3,1);
    std::fill(order.selected.begin(),order.selected.end(),-1);order.selected[0]=0;order.selected[64]=1;
    REQUIRE_EQ(a::evaluate(order.view(),order.output(),order.workspace.get()),RAD_OK);const auto forward=order.output()[1];
    std::swap(order.selected[0],order.selected[64]);REQUIRE_EQ(a::evaluate(order.view(),order.output(),order.workspace.get()),RAD_OK);const auto reversed=order.output()[1];
    CHECK_EQ(forward,0x4037);CHECK_EQ(reversed,0x4038);CHECK(forward!=reversed);guards(order);
    // Removing/sorting padding would move both keys into one64-slot tile and change this function.
    for(int count:{63,64,65,127,128,640}){Fixture pad(1,1,1,count);pad.value(0,1);pad.sinks[0]=0;int valid=0;
        for(int i=0;i<count;++i){pad.selected[size_t(i)]=i%3?-1:0;if(i%3==0)++valid;}
        REQUIRE_EQ(a::evaluate(pad.view(),pad.output(),pad.workspace.get()),RAD_OK);CHECK_EQ(pad.output()[0],nearest(static_cast<long double>(valid)/(valid+1)));guards(pad);}
}
TEST(ds41_attention_online_sink_last_infinite_denominator_and_finite_initial_max_policies_are_explicit){
    Fixture f(1,1,1,1);f.value(0,1);f.sinks[0]=1000;
    REQUIRE_EQ(a::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);CHECK_EQ(f.output()[0],0);
    f.sinks[0]=-INFINITY;REQUIRE_EQ(a::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);CHECK_EQ(f.output()[0],nearest(1));
    for(int count:{0,1,65}){Fixture empty(1,1,0,count);std::fill(empty.selected.begin(),empty.selected.end(),-1);empty.sinks[0]=0;
        REQUIRE_EQ(a::evaluate(empty.view(),empty.output(),empty.workspace.get()),RAD_OK);CHECK_EQ(empty.output()[0],0);
        empty.sinks[0]=-INFINITY;std::fill(empty.guarded.begin(),empty.guarded.end(),0xD3D3);const auto before=empty.guarded;CHECK_EQ(a::evaluate(empty.view(),empty.output(),empty.workspace.get()),RAD_E_INVAL);CHECK(empty.guarded==before);
        empty.sinks[0]=-3e30f;CHECK_EQ(a::evaluate(empty.view(),empty.output(),empty.workspace.get()),RAD_E_INVAL);CHECK(empty.guarded==before);}
    Fixture floor(1,1,1,1);floor.q(0,1);floor.value(0,-2e30f);floor.sinks[0]=-1e30f;
    REQUIRE_EQ(a::evaluate(floor.view(),floor.output(),floor.workspace.get()),RAD_OK);CHECK_EQ(floor.output()[0],0); // score below finite-1e30 initial maximum
    floor.sinks[0]=-INFINITY;std::fill(floor.guarded.begin(),floor.guarded.end(),0xD3D3);const auto before=floor.guarded;CHECK_EQ(a::evaluate(floor.view(),floor.output(),floor.workspace.get()),RAD_E_INVAL);CHECK(floor.guarded==before);
}
TEST(ds41_attention_online_rejects_late_bad_IDs_or_nonfinite_selected_payload_without_mutating_output){
    for(int fault=0;fault<10;++fault){Fixture f(2,2,2,65);f.selected.back()=1;f.value(0,1);f.value(3,2);auto v=f.view();
        if(fault==0)f.selected.back()=2;
        if(fault==1)f.selected.back()=-2;
        if(fault==2)f.query.back()=0x7F80;
        if(fault==3)f.kv.back()=0x7FC1;
        if(fault==4)f.kv.back()=0xFF80;
        if(fault==5)f.sinks.back()=INFINITY;
        if(fault==6)f.sinks.back()=NAN;
        if(fault==7)v.scale=INFINITY;
        if(fault==8)v.scale=0;
        if(fault==9)v.scale=-1;
        const auto before=f.guarded;CHECK(a::evaluate(v,f.output(),f.workspace.get())<0);CHECK(f.guarded==before);
    }
    // Derived score overflow after finite BF16 payload validation also refuses transactionally.
    Fixture overflow(1,1,1,1);overflow.query[0]=0x7F7F;overflow.kv[0]=0x7F7F;const auto before=overflow.guarded;
    CHECK_EQ(a::evaluate(overflow.view(),overflow.output(),overflow.workspace.get()),RAD_E_INVAL);CHECK(overflow.guarded==before);
    // PV accumulation overflow is refused even though each BF16 payload value is finite.
    Fixture numerator(1,1,1,2);numerator.kv[0]=0x7F7F;const auto untouched=numerator.guarded;
    CHECK_EQ(a::evaluate(numerator.view(),numerator.output(),numerator.workspace.get()),RAD_E_INVAL);CHECK(numerator.guarded==untouched);
    Fixture late_head(2,1,1,1);late_head.query[1]=0x7F7F;late_head.kv[0]=0x7F7F;std::fill(late_head.sinks.begin(),late_head.sinks.end(),0.f);const auto before_heads=late_head.guarded;
    CHECK_EQ(a::evaluate(late_head.view(),late_head.output(),late_head.workspace.get()),RAD_E_INVAL);CHECK(late_head.guarded==before_heads);
}
TEST(ds41_attention_online_ignores_unselected_invalid_KV_and_establishes_all_IDs_before_KV_reads){
    Fixture f(1,2,2,1);f.value(0,1);f.kv[2]=0x7F80;f.kv[3]=0x7FC1;f.sinks[0]=0;
    REQUIRE_EQ(a::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);CHECK_EQ(f.output()[0],nearest(.5L));
    const size_t page=size_t(sysconf(_SC_PAGESIZE));REQUIRE(page>=4096);Pages storage(page*2);
    constexpr int64_t dim=1024;const int64_t rows=int64_t(storage.bytes/(size_t(dim)*2));
    auto* values=static_cast<uint16_t*>(storage.data);std::fill(values,values+page/2,nearest(1));REQUIRE_EQ(mprotect(static_cast<uint8_t*>(storage.data)+page,page,PROT_NONE),0);
    Fixture protected_rows(1,dim,1,1);auto v=protected_rows.view();v.kv_rows=rows;v.kv={values,size_t(rows*dim)};protected_rows.sinks[0]=0;
    REQUIRE_EQ(a::evaluate(v,protected_rows.output(),protected_rows.workspace.get()),RAD_OK);for(auto code:protected_rows.output())CHECK_EQ(code,nearest(.5L));
    Pages inaccessible(page*2,PROT_NONE);Fixture bad(1,dim,1,2);bad.selected={0,4};v=bad.view();v.kv_rows=4;v.kv={static_cast<const uint16_t*>(inaccessible.data),size_t(4*dim)};
    const auto before=bad.guarded;CHECK_EQ(a::evaluate(v,bad.output(),bad.workspace.get()),RAD_E_SHAPE);CHECK(bad.guarded==before);
}
TEST(ds41_attention_online_checks_exact_sizes_caps_ranges_and_all_workspace_aliases_before_scratch_mutation){
    for(int fault=0;fault<17;++fault){Fixture f;auto v=f.view();auto out=f.output();auto* workspace=f.workspace.get();std::vector<int32_t> excessive(641,0);std::vector<uint8_t> unaligned(sizeof(a::Workspace)+8);
        if(fault==0)v.heads=33;
        if(fault==1)v.dimension=1025;
        if(fault==2)v.dimension=0;
        if(fault==3)v.kv_rows=-1;
        if(fault==4)v.selected=excessive;
        if(fault==5)v.query=v.query.first(1);
        if(fault==6)v.kv=v.kv.first(3);
        if(fault==7)v.sinks={};
        if(fault==8)out={f.guarded.data()+4,3};
        if(fault==9)workspace=nullptr;
        if(fault==10)v.query={reinterpret_cast<const uint16_t*>(UINTPTR_MAX-1),2};
        if(fault==11)workspace=reinterpret_cast<a::Workspace*>(unaligned.data()+1);
        if(fault==12)workspace=reinterpret_cast<a::Workspace*>(UINTPTR_MAX-3);
        if(fault==13)v.selected={reinterpret_cast<const int32_t*>(UINTPTR_MAX-3),2};
        if(fault==14)out={static_cast<uint16_t*>(nullptr),2};
        if(fault==15)v.query={static_cast<const uint16_t*>(nullptr),2};
        if(fault==16)v.kv={static_cast<const uint16_t*>(nullptr),4};
        const auto before=f.guarded;poison_workspace(*f.workspace);std::vector<uint8_t> scratch(sizeof(a::Workspace));std::memcpy(scratch.data(),f.workspace.get(),scratch.size());
        CHECK(a::evaluate(v,out,workspace)<0);CHECK(f.guarded==before);CHECK(std::memcmp(scratch.data(),f.workspace.get(),scratch.size())==0);
    }
    for(int alias=0;alias<9;++alias){Fixture f;auto v=f.view();auto out=f.output();poison_workspace(*f.workspace);const auto before=f.guarded,input=f.query,kv=f.kv;const auto sinks=f.sinks;const auto selected=f.selected;
        std::vector<uint8_t> scratch(sizeof(a::Workspace));std::memcpy(scratch.data(),f.workspace.get(),scratch.size());
        if(alias==0)out={f.query.data(),2};
        if(alias==1)out={f.kv.data(),2};
        if(alias==2)out={reinterpret_cast<uint16_t*>(f.sinks.data()),2};
        if(alias==3)out={reinterpret_cast<uint16_t*>(f.selected.data()),2};
        if(alias==4)out={f.workspace->transactional_output.data(),2};
        if(alias==5)v.query={f.workspace->transactional_output.data(),2};
        if(alias==6)v.kv={f.workspace->transactional_output.data(),4};
        if(alias==7)v.sinks={f.workspace->numerator.data(),1};
        if(alias==8)v.selected={f.workspace->ids.data(),2};
        CHECK_EQ(a::evaluate(v,out,f.workspace.get()),RAD_E_INVAL);CHECK(f.guarded==before);same_inputs(f,input,kv,sinks,selected);CHECK(std::memcmp(scratch.data(),f.workspace.get(),scratch.size())==0);
    }
}
TEST(ds41_attention_online_refuses_non_RNE_and_x86_FTZ_DAZ_before_scratch_or_output_mutation){
    struct Restore {
        std::fenv_t environment{};
#if defined(__SSE__)
        unsigned mxcsr=0;
#endif
        Restore(){
            if(std::fegetenv(&environment)!=0)throw std::runtime_error("cannot capture floating environment");
#if defined(__SSE__)
            mxcsr=_mm_getcsr();
#endif
        }
        ~Restore(){
            static_cast<void>(std::fesetenv(&environment));
#if defined(__SSE__)
            _mm_setcsr(mxcsr);
#endif
        }
    } restore;
    REQUIRE(a::detail::supported_environment());
    Fixture f;poison_workspace(*f.workspace);
    const auto before=f.guarded,input=f.query,kv=f.kv;const auto sinks=f.sinks;const auto ids=f.selected;
    std::vector<uint8_t> scratch(sizeof(a::Workspace));std::memcpy(scratch.data(),f.workspace.get(),scratch.size());
    auto refusal=[&]{
        CHECK_EQ(a::evaluate(f.view(),f.output(),f.workspace.get()),RAD_E_UNSUPPORTED);
        CHECK(f.guarded==before);CHECK(std::memcmp(scratch.data(),f.workspace.get(),scratch.size())==0);
        same_inputs(f,input,kv,sinks,ids);
    };
    for(int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        REQUIRE_EQ(std::fesetround(mode),0);CHECK(!a::detail::supported_environment());refusal();CHECK_EQ(std::fegetround(),mode);
    }
    REQUIRE_EQ(std::fesetround(FE_TONEAREST),0);
#if defined(__SSE__)
    const unsigned baseline=_mm_getcsr();
    for(unsigned flag:{1u<<15,1u<<6}){ // FTZ and DAZ independently, with RNE unchanged.
        const unsigned control=(baseline&~((1u<<15)|(1u<<6)))|flag;
        _mm_setcsr(control);CHECK(!a::detail::supported_environment());refusal();
        CHECK_EQ(std::fegetround(),FE_TONEAREST);
        CHECK_EQ(_mm_getcsr()&~0x3Fu,control&~0x3Fu); // Probe arithmetic may set exception flags.
        _mm_setcsr(baseline);
    }
#endif
}
RAD_TEST_MAIN()

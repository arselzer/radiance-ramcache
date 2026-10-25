/* Actual CPU Torch score-expression receipts plus independent dyadic witnesses.
 * This establishes a scalar mathematical candidate, not TileLang/native/TP parity. */
#include "rad_test.h"
#include "../arch/deepseek41/index_score_bf16_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <memory>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace s=rad::deepseek41::index_score_bf16;
namespace {
const nlohmann::json& golden(){
    static const auto value=[] {
        const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/ds41-index-score-cpu.json";
        if(std::filesystem::file_size(path)>1000000)throw std::runtime_error("index fixture exceeds cap");
        std::ifstream input(path);if(!input)throw std::runtime_error("missing CPU Torch index-score fixture");
        return nlohmann::json::parse(input);
    }();return value;
}
long double exact(uint16_t code){
    const unsigned e=(code>>7)&255u,m=code&127u;
    const long double value=e?std::ldexp(static_cast<long double>(128u+m),int(e)-134):std::ldexp(static_cast<long double>(m),-133);
    return code&0x8000u?-value:value;
}
// Independent exact nearest-neighbor converter, not the implementation's bit-add converter.
uint16_t nearest(long double value){
    const bool sign=std::signbit(value);const long double magnitude=std::fabs(value);
    if(!std::isfinite(value)||magnitude>exact(0x7F7F))throw std::runtime_error("finite test oracle range");
    unsigned lo=0,hi=0x7F7F;
    while(lo+1<hi){const unsigned mid=(lo+hi)/2;if(exact(uint16_t(mid))<magnitude)lo=mid;else hi=mid;}
    const long double dl=magnitude-exact(uint16_t(lo)),dh=exact(uint16_t(hi))-magnitude;
    const unsigned chosen=dl<dh?lo:dh<dl?hi:(lo&1u?hi:lo);
    return uint16_t(chosen|(sign?0x8000u:0));
}
struct Fixture {
    int64_t heads,dimension,keys;
    std::vector<uint16_t> query,key,weights,guarded;
    std::unique_ptr<s::Workspace> workspace=std::make_unique<s::Workspace>();
    float factor=1;
    Fixture(int64_t h=2,int64_t d=2,int64_t n=2):heads(h),dimension(d),keys(n),query(size_t(h*d),0),key(size_t(n*d),0),weights(size_t(h),nearest(1)),guarded(size_t(n)+8,0xD3D3){}
    s::View view()const{return {heads,dimension,keys,query,key,weights,factor};}
    std::span<uint16_t> output(){return {guarded.data()+4,size_t(keys)};}
    void guards()const{for(size_t i=0;i<4;++i){CHECK_EQ(guarded[i],0xD3D3);CHECK_EQ(guarded[guarded.size()-1-i],0xD3D3);}}
};
void poison_workspace(s::Workspace& workspace){
    static_assert(std::is_trivially_copyable_v<s::Workspace>);
    auto bytes=std::as_writable_bytes(std::span(&workspace,1));
    std::fill(bytes.begin(),bytes.end(),std::byte{0xB7});
}
std::vector<uint8_t> snapshot(const s::Workspace& w){
    std::vector<uint8_t> bytes(sizeof(w));std::memcpy(bytes.data(),&w,sizeof(w));return bytes;
}
void same_workspace(const s::Workspace& w,const std::vector<uint8_t>& before){CHECK(std::memcmp(&w,before.data(),before.size())==0);}
}
TEST(ds41_index_score_bf16_matches_actual_unchanged_cpu_Torch_score_AST_stages){
    const auto& g=golden();CHECK_EQ(g.at("schema").get<std::string>(),"ds41-index-score-cpu-v1");
    CHECK_EQ(g.at("source_revision").get<std::string>(),"1bae0e1088993b9b82de08f59de5be5a34e42170");
    CHECK_EQ(g.at("source_model_sha256").get<std::string>(),"e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d");
    CHECK_EQ(g.at("torch_version").get<std::string>(),"2.14.1+cpu");CHECK_EQ(g.at("device").get<std::string>(),"cpu");CHECK_EQ(g.at("threads").get<int>(),1);
    for(const auto& stage:g.at("stage_dtypes"))CHECK_EQ(stage.get<std::string>(),"bfloat16");
    REQUIRE_EQ(g.at("cases").size(),6u);
    for(const auto& c:g.at("cases")){
        Fixture f(c.at("heads"),c.at("dimension"),c.at("keys"));
        f.query=c.at("query").get<std::vector<uint16_t>>();f.key=c.at("key").get<std::vector<uint16_t>>();f.weights=c.at("projected_weights").get<std::vector<uint16_t>>();
        f.factor=std::bit_cast<float>(c.at("factor_f32_bits").get<uint32_t>());
        const auto q=f.query,k=f.key,w=f.weights;
        REQUIRE_EQ(s::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);
        const auto scaled=c.at("scaled_weights").get<std::vector<uint16_t>>(),dot=c.at("dot").get<std::vector<uint16_t>>();
        const auto relu=c.at("relu").get<std::vector<uint16_t>>(),products=c.at("products").get<std::vector<uint16_t>>(),scores=c.at("scores").get<std::vector<uint16_t>>();
        REQUIRE_EQ(scaled.size(),size_t(f.heads));REQUIRE_EQ(dot.size(),size_t(f.heads*f.keys));REQUIRE_EQ(relu.size(),dot.size());REQUIRE_EQ(products.size(),dot.size());REQUIRE_EQ(scores.size(),size_t(f.keys));
        for(int64_t h=0;h<f.heads;++h){CHECK_EQ(f.workspace->scaled_weights[size_t(h)],scaled[size_t(h)]);
            for(int64_t j=0;j<f.keys;++j){const auto a=size_t(h*s::max_keys+j),b=size_t(h*f.keys+j);
                CHECK_EQ(f.workspace->dot[a],dot[b]);CHECK_EQ(f.workspace->relu[a],relu[b]);CHECK_EQ(f.workspace->products[a],products[b]);}}
        for(int64_t j=0;j<f.keys;++j){CHECK_EQ(f.output()[size_t(j)],scores[size_t(j)]);CHECK_EQ(f.workspace->transactional_scores[size_t(j)],scores[size_t(j)]);}
        CHECK(f.query==q);CHECK(f.key==k);CHECK(f.weights==w);f.guards();
    }
}
TEST(ds41_index_score_bf16_has_strict_score_order_reversal_from_existing_F32_equations_without_a_tie){
    const auto& c=golden().at("cases").back();REQUIRE_EQ(c.at("name").get<std::string>(),"strict_bf16_vs_f32_order_reversal");
    Fixture f(c.at("heads"),c.at("dimension"),c.at("keys"));f.query=c.at("query").get<std::vector<uint16_t>>();f.key=c.at("key").get<std::vector<uint16_t>>();f.weights=c.at("projected_weights").get<std::vector<uint16_t>>();f.factor=.25f;
    REQUIRE_EQ(s::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);
    // These input dyadics have exact F32 dot/head sums. Long-double direct equations
    // independently prove the F32 path's strict ordering, without any top-k implementation.
    std::vector<long double> unrounded(size_t(f.keys),0);
    for(int64_t j=0;j<f.keys;++j)for(int64_t h=0;h<f.heads;++h){long double dot=0;
        for(int64_t d=0;d<f.dimension;++d)dot+=exact(f.query[size_t(h*f.dimension+d)])*exact(f.key[size_t(j*f.dimension+d)]);
        unrounded[size_t(j)]+=std::max(dot,0.L)*exact(f.weights[size_t(h)])*.25L;}
    const auto a=c.at("strict_reversal").at("a").get<size_t>(),b=c.at("strict_reversal").at("b").get<size_t>();
    CHECK(exact(f.output()[a])<exact(f.output()[b]));CHECK(unrounded[a]>unrounded[b]);
    for(int64_t j=0;j<f.keys;++j)CHECK_EQ(std::bit_cast<uint32_t>(static_cast<float>(unrounded[size_t(j)])),c.at("baseline_f32_bits")[size_t(j)].get<uint32_t>());
    CHECK_EQ(f.output()[a],0xBF61);CHECK_EQ(f.output()[b],0xBF60);f.guards();
}
TEST(ds41_index_score_bf16_hand_oracle_caps_empty_and_signed_zero_boundaries){
    static_assert(sizeof(s::Workspace)<128*1024);
    for(int heads:{1,2,32})for(int dim:{1,8,1024})for(int keys:{0,1,63,64,65,128,640}){
        if(dim==1024&&keys>1)continue;
        Fixture f(heads,dim,keys);f.factor=.25f;
        for(int h=0;h<heads;++h){f.weights[size_t(h)]=nearest(h%2?-1.L:.5L);for(int d=0;d<dim;++d)f.query[size_t(h*dim+d)]=nearest(h%2?-1.L:1.L);}
        for(int j=0;j<keys;++j)for(int d=0;d<dim;++d)f.key[size_t(j*dim+d)]=nearest(j%2?-1.L:1.L);
        REQUIRE_EQ(s::evaluate(f.view(),f.output(),f.workspace.get()),RAD_OK);
        for(int j=0;j<keys;++j){long double expected=0;
            for(int h=0;h<heads;++h){const long double dot=(h%2?-1.L:1.L)*(j%2?-1.L:1.L)*dim;
                expected+=std::max(dot,0.L)*(h%2?-1.L:.5L)*.25L;}
            CHECK_EQ(f.output()[size_t(j)],nearest(expected));}
        f.guards();
    }
    Fixture zero(1,4,2);zero.weights[0]=0x8000;std::fill(zero.key.begin(),zero.key.end(),nearest(1));zero.factor=.5f;
    REQUIRE_EQ(s::evaluate(zero.view(),zero.output(),zero.workspace.get()),RAD_OK);
    CHECK_EQ(zero.workspace->scaled_weights[0],0x8000);CHECK_EQ(zero.workspace->dot[0],0);CHECK_EQ(zero.workspace->relu[0],0);CHECK_EQ(zero.workspace->products[0],0x8000);CHECK_EQ(zero.output()[0],0);
}
TEST(ds41_index_score_bf16_rejects_nonfinite_late_inputs_and_derived_overflow_transactionally){
    for(int fault=0;fault<11;++fault){Fixture f;auto v=f.view();
        if(fault==0)f.query.back()=0x7FC1;
        if(fault==1)f.key.back()=0x7F80;
        if(fault==2)f.weights.back()=0xFF80;
        if(fault==3)v.factor=INFINITY;
        if(fault==4)v.factor=0;
        if(fault==5)v.factor=-1;
        if(fault==6){f.weights.back()=0x7F7F;v.factor=2;}
        if(fault==7){f.query.back()=0x7F7F;f.key.back()=0x7F7F;}
        if(fault==8){f.query={0x7F7F,0x7F7F,0,0};f.key={nearest(1),nearest(1),0,0};}
        if(fault==9){std::fill(f.query.begin(),f.query.end(),nearest(1));std::fill(f.key.begin(),f.key.end(),nearest(1));f.weights.back()=0x7F7F;}
        if(fault==10){std::fill(f.query.begin(),f.query.end(),nearest(.5L));std::fill(f.key.begin(),f.key.end(),nearest(1));std::fill(f.weights.begin(),f.weights.end(),0x7F7F);}
        // Rebind after vector replacement, keeping explicitly changed factor.
        v.query=f.query;v.key=f.key;v.projected_weights=f.weights;
        const auto before=f.guarded,q=f.query,k=f.key,w=f.weights;
        CHECK(s::evaluate(v,f.output(),f.workspace.get())<0);CHECK(f.guarded==before);CHECK(f.query==q);CHECK(f.key==k);CHECK(f.weights==w);
    }
    // A safe first key cannot publish before the final key's arithmetic overflow.
    Fixture late(1,1,2);late.query[0]=0x7F7F;late.key={0,0x7F7F};const auto before=late.guarded;
    CHECK_EQ(s::evaluate(late.view(),late.output(),late.workspace.get()),RAD_E_INVAL);CHECK(late.guarded==before);
}
TEST(ds41_index_score_bf16_metadata_ranges_and_aliases_refuse_before_workspace_mutation){
    for(int fault=0;fault<21;++fault){Fixture f;auto v=f.view();auto out=f.output();auto* workspace=f.workspace.get();std::vector<uint8_t> unaligned(sizeof(s::Workspace)+8);
        if(fault==0)v.heads=33;
        if(fault==1)v.heads=0;
        if(fault==2)v.dimension=1025;
        if(fault==3)v.dimension=0;
        if(fault==4)v.keys=-1;
        if(fault==5)v.keys=641;
        if(fault==6)v.query=v.query.first(3);
        if(fault==7)v.key=v.key.first(3);
        if(fault==8)v.projected_weights=v.projected_weights.first(1);
        if(fault==9)out=out.first(1);
        if(fault==10)workspace=nullptr;
        if(fault==11)v.query={reinterpret_cast<const uint16_t*>(UINTPTR_MAX-1),4};
        if(fault==12)workspace=reinterpret_cast<s::Workspace*>(UINTPTR_MAX-1);
        if(fault==13)workspace=reinterpret_cast<s::Workspace*>(unaligned.data()+1);
        if(fault==14)v.key={static_cast<const uint16_t*>(nullptr),4};
        if(fault==15)out={static_cast<uint16_t*>(nullptr),2};
        if(fault==16)v.projected_weights={reinterpret_cast<const uint16_t*>(UINTPTR_MAX-1),2};
        if(fault==17)v.factor=NAN;
        if(fault==18)v.factor=-0.f;
        if(fault==19)v.query={reinterpret_cast<const uint16_t*>(unaligned.data()+1),4};
        if(fault==20)out={reinterpret_cast<uint16_t*>(UINTPTR_MAX-1),2};
        poison_workspace(*f.workspace);const auto scratch=snapshot(*f.workspace);const auto before=f.guarded;
        CHECK(s::evaluate(v,out,workspace)<0);CHECK(f.guarded==before);same_workspace(*f.workspace,scratch);
    }
    for(int alias=0;alias<7;++alias){Fixture f;auto v=f.view();auto out=f.output();poison_workspace(*f.workspace);
        if(alias==0)out={f.query.data(),2};
        if(alias==1)out={f.key.data(),2};
        if(alias==2)out={f.weights.data(),2};
        if(alias==3)out={f.workspace->transactional_scores.data(),2};
        if(alias==4)v.query={f.workspace->dot.data(),4};
        if(alias==5)v.key={f.workspace->relu.data(),4};
        if(alias==6)v.projected_weights={f.workspace->scaled_weights.data(),2};
        const auto before=f.guarded,q=f.query,k=f.key,w=f.weights;const auto scratch=snapshot(*f.workspace);
        CHECK_EQ(s::evaluate(v,out,f.workspace.get()),RAD_E_INVAL);CHECK(f.guarded==before);CHECK(f.query==q);CHECK(f.key==k);CHECK(f.weights==w);same_workspace(*f.workspace,scratch);
    }
}
TEST(ds41_index_score_bf16_refuses_non_RNE_and_FTZ_DAZ_without_mutation){
    struct Restore {
        std::fenv_t environment{};
#if defined(__SSE__)
        unsigned mxcsr=0;
#endif
        Restore(){if(std::fegetenv(&environment)!=0)throw std::runtime_error("fenv snapshot");
#if defined(__SSE__)
            mxcsr=_mm_getcsr();
#endif
        }
        ~Restore(){static_cast<void>(std::fesetenv(&environment));
#if defined(__SSE__)
            _mm_setcsr(mxcsr);
#endif
        }
    } restore;
    REQUIRE(s::detail::fp::supported_environment());Fixture f;
    poison_workspace(*f.workspace);const auto scratch=snapshot(*f.workspace);const auto before=f.guarded;
    auto refusal=[&]{CHECK_EQ(s::evaluate(f.view(),f.output(),f.workspace.get()),RAD_E_UNSUPPORTED);CHECK(f.guarded==before);same_workspace(*f.workspace,scratch);};
    for(int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){REQUIRE_EQ(std::fesetround(mode),0);refusal();CHECK_EQ(std::fegetround(),mode);}
    REQUIRE_EQ(std::fesetround(FE_TONEAREST),0);
#if defined(__SSE__)
    const unsigned baseline=_mm_getcsr();
    for(unsigned flag:{1u<<15,1u<<6}){_mm_setcsr((baseline&~((1u<<15)|(1u<<6)))|flag);refusal();_mm_setcsr(baseline);}
#endif
}
RAD_TEST_MAIN()

/* Accepted-finite arithmetic failures, with whole-batch output refusal. CPU only.
 * Ordinary equations still compare with the independent pinned scalar reference. */
#include "rad_test.h"
#include "../libds41/ds41_math_emulate.h"
#include "../arch/common/rad_ds41_ref.h"
#include <array>
namespace m=rad::ds41::math;
namespace ref=rad::deepseek41::ref;
namespace {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,bool pitch) {
        t.dtype=dtype;t.rank=shape.size();unsigned i=0;for(auto d:shape)t.shape[i++]=d;
        int64_t stride=pitch?2:1;
        for(int axis=int(t.rank)-1;axis>=0;--axis){t.stride[axis]=stride;stride=stride*t.shape[axis]+(pitch?3:0);}
        words.resize(size_t(stride*4+32)/4,0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count()const{int64_t n=1;for(unsigned i=0;i<t.rank;++i){n*=t.shape[i];}return n;}
    int64_t offset(int64_t flat)const{return m::offset(t,flat);}
    void set(int64_t flat,float value){rad_store_f32(words.data(),t.dtype,offset(flat),value);}
    float get(int64_t flat)const{return rad_load_f32(words.data(),t.dtype,offset(flat));}
    void id(int64_t flat,int32_t value){reinterpret_cast<int32_t*>(words.data())[offset(flat)]=value;}
    int32_t id(int64_t flat)const{return reinterpret_cast<const int32_t*>(words.data())[offset(flat)];}
    std::vector<float> values()const{std::vector<float> result;for(int64_t i=0;i<count();++i){result.push_back(get(i));}return result;}
};
struct Case {
    m::Op op;std::vector<Buffer> b;std::vector<RadTensor> tensors;std::vector<RadParam> params;
    std::array<int32_t,4> flag{711,712,713,714};
    Case(m::Op o,uint32_t dtype=RAD_F32,bool pitch=true):op(o) {
        if(op==m::Op::hc_prepare||op==m::Op::hc_collapse){
            params={RAD_INT("M",3),RAD_INT("n",2),RAD_INT("hc",2),RAD_STR("dtype",rad_dtype_name(dtype))};
            b.emplace_back(dtype,std::initializer_list<int64_t>{2,4},pitch);
            if(op==m::Op::hc_prepare){
                params.push_back(RAD_INT("iterations",2));params.push_back(RAD_F64("eps",1e-6));params.push_back(RAD_F64("hc_eps",1e-6));
                b.emplace_back(RAD_F32,std::initializer_list<int64_t>{8,4},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{3},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{8},pitch);
                b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,2},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,2},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,4},pitch);
            }else{b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,2},pitch);b.emplace_back(dtype,std::initializer_list<int64_t>{2,2},pitch);}
        }else if(op==m::Op::router_topk){
            params={RAD_INT("M",3),RAD_INT("experts",4),RAD_INT("topk",2),RAD_INT("normalize",1),RAD_F64("temperature",.75),RAD_F64("route_scale",1.5)};
            b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,4},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{4},pitch);b.emplace_back(RAD_I32,std::initializer_list<int64_t>{2,2},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,2},pitch);
        }else{
            params={RAD_INT("M",3),RAD_INT("heads",2),RAD_INT("head_dim",2),RAD_INT("kv_rows",3),RAD_INT("topk",2),RAD_F64("scale",.25),RAD_STR("dtype",rad_dtype_name(dtype))};
            b.emplace_back(dtype,std::initializer_list<int64_t>{2,2,2},pitch);b.emplace_back(dtype,std::initializer_list<int64_t>{3,2},pitch);b.emplace_back(RAD_F32,std::initializer_list<int64_t>{2},pitch);b.emplace_back(RAD_I32,std::initializer_list<int64_t>{2,2},pitch);b.emplace_back(dtype,std::initializer_list<int64_t>{2,2,2},pitch);
        }
        for(int i=0;i<first_output();++i)for(int64_t j=0;j<b[size_t(i)].count();++j){
            if(b[size_t(i)].t.dtype==RAD_I32)b[size_t(i)].id(j,int32_t(j%3));else b[size_t(i)].set(j,float((j*3+i)%9-4)/16.f);
        }
        if(op==m::Op::hc_prepare)for(int i=0;i<3;++i)b[2].set(i,.5f+float(i)*.25f);
        if(op==m::Op::sparse_attn)b[2].set(0,-INFINITY);
    }
    int first_output()const{return op==m::Op::hc_prepare?4:op==m::Op::hc_collapse?2:op==m::Op::router_topk?2:4;}
    RadArgs args(){tensors.clear();for(auto& buffer:b){buffer.t.data=buffer.words.data();tensors.push_back(buffer.t);}RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());a.scratch=flag.data();a.scratch_bytes=4;return a;}
    int run(){auto a=args();return m::emulate(op,&a);}
    void real(const char* key,double value){for(auto& p:params)if(!std::strcmp(p.key,key))p.dval=value;}
    void integer(const char* key,int64_t value){for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    void source_valid(){auto a=args();m::Plan plan;if(m::validate(op,&a,&plan)!=RAD_OK)throw std::runtime_error("derived witness has invalid metadata");for(int64_t i=0;i<m::preflight_items(plan);++i)if(m::check_item(plan,i)!=RAD_OK)throw std::runtime_error("derived witness has invalid source payload");}
};
void reference(Case& c){
    const auto x=c.b[0].values();
    if(c.op==m::Op::hc_prepare){
        for(int row=0;row<2;++row){ref::MhcMix mix;REQUIRE_EQ(ref::mhc_prepare(ref::Floats(x).subspan(size_t(row*4),4),2,2,c.b[1].values(),c.b[2].values(),c.b[3].values(),2,1e-6f,1e-6f,&mix),RAD_OK);
            for(int i=0;i<2;++i){c.b[4].set(row*2+i,mix.pre[size_t(i)]);c.b[5].set(row*2+i,mix.post[size_t(i)]);}for(int i=0;i<4;++i)c.b[6].set(row*4+i,mix.comb[size_t(i)]);}
    }else if(c.op==m::Op::hc_collapse){
        const auto pre=c.b[1].values();std::array<float,2> out{};for(int row=0;row<2;++row){REQUIRE_EQ(ref::mhc_collapse(ref::Floats(x).subspan(size_t(row*4),4),ref::Floats(pre).subspan(size_t(row*2),2),out),RAD_OK);for(int i=0;i<2;++i)c.b[2].set(row*2+i,out[size_t(i)]);}
    }else if(c.op==m::Op::router_topk){
        for(int row=0;row<2;++row){ref::Routing out;REQUIRE_EQ(ref::route(ref::Floats(x).subspan(size_t(row*4),4),c.b[1].values(),2,.75f,1.5f,true,&out),RAD_OK);for(int i=0;i<2;++i){c.b[2].id(row*2+i,out.ids[size_t(i)]);c.b[3].set(row*2+i,out.weights[size_t(i)]);}}
    }else{
        const auto kv=c.b[1].values(),sinks=c.b[2].values();std::array<int32_t,2> ids{};std::array<float,4> out{};for(int row=0;row<2;++row){for(int i=0;i<2;++i)ids[size_t(i)]=c.b[3].id(row*2+i);REQUIRE_EQ(ref::sparse_attention(ref::Floats(x).subspan(size_t(row*4),4),kv,2,2,sinks,ids,.25f,out),RAD_OK);for(int i=0;i<4;++i)c.b[4].set(row*4+i,out[size_t(i)]);}
    }
}
void rejected(Case& c){
    c.source_valid();std::vector<std::vector<uint32_t>> before;for(const auto& b:c.b)before.push_back(b.words);
    CHECK_EQ(c.run(),RAD_E_INVAL);CHECK_EQ(c.flag[0],RAD_E_INVAL);CHECK_EQ(c.flag[1],712);CHECK_EQ(c.flag[2],713);CHECK_EQ(c.flag[3],714);
    for(size_t i=0;i<c.b.size();++i)CHECK(c.b[i].words==before[i]);
}
void reset_good(Case& c,const Case& original){
    c.b=original.b;c.params=original.params;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
}
}
TEST(ds41_math_derived_normal_equations_match_independent_reference_all_dtypes_pitches){
    for(auto op:{m::Op::hc_prepare,m::Op::hc_collapse,m::Op::router_topk,m::Op::sparse_attn})for(auto dtype:{RAD_F32,RAD_F16,RAD_BF16})for(bool pitch:{false,true}){
        Case c(op,dtype,pitch),expected=c;reference(expected);const auto before=c.b;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
        for(int i=0;i<c.first_output();++i)CHECK(c.b[size_t(i)].words==before[size_t(i)].words);
        for(size_t i=size_t(c.first_output());i<c.b.size();++i){auto untouched=before[i];for(int64_t j=0;j<c.b[i].count();++j){if(c.b[i].t.dtype==RAD_I32){CHECK_EQ(c.b[i].id(j),expected.b[i].id(j));untouched.id(j,c.b[i].id(j));}else{CHECK_NEAR(c.b[i].get(j),expected.b[i].get(j),3e-6+3e-5*std::fabs(expected.b[i].get(j)));untouched.set(j,c.b[i].get(j));}}CHECK(c.b[i].words==untouched.words);}
    }
}
TEST(ds41_math_derived_prepare_refuses_late_square_dot_and_sinkhorn_zero_denominator){
    for(auto dtype:{RAD_F32,RAD_BF16,RAD_F16}){
        Case c(m::Op::hc_prepare,dtype),good=c;REQUIRE_EQ(c.run(),RAD_OK);
        for(int i=0;i<4;++i)c.b[0].set(i,0);
        if(dtype==RAD_F16){for(int i=4;i<8;++i){c.b[0].set(i,2);}
            for(int64_t i=0;i<c.b[1].count();++i)c.b[1].set(i,std::ldexp(1.f,127));}
        else{for(int i=4;i<8;++i){c.b[0].set(i,std::ldexp(1.f,100));}
            for(int64_t i=0;i<c.b[1].count();++i)c.b[1].set(i,std::ldexp(1.f,100));}
        rejected(c);reset_good(c,good);
    }
    Case zero(m::Op::hc_prepare);for(int64_t i=0;i<zero.b[1].count();++i){zero.b[1].set(i,0);}
    for(int i=0;i<8;++i){zero.b[3].set(i,i>=4&&i%2?-1000.f:0.f);}
    zero.real("hc_eps",0);rejected(zero);
}
TEST(ds41_math_derived_collapse_refuses_FP32_and_actual_F16_BF16_cast_overflow){
    for(int fault=0;fault<4;++fault){const auto dtype=fault==2?RAD_F16:fault==1||fault==3?RAD_BF16:RAD_F32;Case c(m::Op::hc_collapse,dtype),good=c;REQUIRE_EQ(c.run(),RAD_OK);
        for(int i=4;i<8;++i){c.b[0].set(i,0);}
        c.b[1].set(2,fault==3?257.f/256.f:2);c.b[1].set(3,0);
        c.b[0].set(4,fault==2?65504.f:fault==3?m::packed::from_bits(0x7F7F0000u):std::ldexp(1.f,127));
        rejected(c);reset_good(c,good);
    }
}
TEST(ds41_math_derived_router_refuses_late_quotient_and_top1_scaled_weight_overflow){
    for(int fault=0;fault<2;++fault){Case c(m::Op::router_topk),good=c;REQUIRE_EQ(c.run(),RAD_OK);
        if(fault==0){for(int i=0;i<4;++i){c.b[0].set(i,0);}
            for(int i=4;i<8;++i){c.b[0].set(i,4);}
            c.real("temperature",std::ldexp(1.,-126));}
        else{c.integer("topk",1);c.b[2].t.shape[1]=1;c.b[3].t.shape[1]=1;for(int i=0;i<4;++i){c.b[0].set(i,0);}
            for(int i=4;i<8;++i){c.b[0].set(i,std::ldexp(1.f,126));}
            c.real("temperature",1);c.real("route_scale",std::ldexp(1.,66));}
        rejected(c);reset_good(c,good);
    }
}
TEST(ds41_math_derived_sparse_refuses_late_head_score_and_finite_probability_numerator_overflow){
    for(auto dtype:{RAD_F32,RAD_BF16}){Case c(m::Op::sparse_attn,dtype),good=c;REQUIRE_EQ(c.run(),RAD_OK);
        for(int64_t i=0;i<c.b[0].count();++i){c.b[0].set(i,0);}
        c.b[2].set(0,-INFINITY);c.b[2].set(1,-INFINITY);
        // First sequence sees small key0 once; final sequence sees large key2 twice.
        c.b[3].id(0,0);c.b[3].id(1,-1);c.b[3].id(2,2);c.b[3].id(3,2);c.b[1].set(4,std::ldexp(1.f,127));c.b[1].set(5,0);
        rejected(c);reset_good(c,good);
    }
    Case head(m::Op::sparse_attn,RAD_F16);for(int64_t i=0;i<head.b[0].count();++i){head.b[0].set(i,0);}
    head.b[0].set(7,2);for(int64_t i=0;i<head.b[1].count();++i){head.b[1].set(i,2);}
    head.real("scale",std::ldexp(1.,127));rejected(head);
}
TEST(ds41_math_derived_preserves_router_ties_zero_weights_padding_and_positive_zero_gates){
    Case router(m::Op::router_topk);for(int64_t i=0;i<router.b[0].count();++i){router.b[0].set(i,0);}
    for(int i=0;i<4;++i){router.b[1].set(i,0);}
    REQUIRE_EQ(router.run(),RAD_OK);for(int row=0;row<2;++row){CHECK_EQ(router.b[2].id(row*2),0);CHECK_EQ(router.b[2].id(row*2+1),1);}
    for(int64_t i=0;i<router.b[0].count();++i){router.b[0].set(i,-1000);}
    REQUIRE_EQ(router.run(),RAD_OK);for(int64_t i=0;i<router.b[3].count();++i)CHECK_EQ(router.b[3].get(i),0);
    for(auto dtype:{RAD_F32,RAD_F16,RAD_BF16}){Case empty(m::Op::sparse_attn,dtype);for(int64_t i=0;i<empty.b[3].count();++i){empty.b[3].id(i,-1);}
        for(int i=0;i<2;++i){empty.b[2].set(i,-INFINITY);}
        REQUIRE_EQ(empty.run(),RAD_OK);for(float x:empty.b[4].values()){CHECK_EQ(x,0);CHECK(!std::signbit(x));}}
    Case gate(m::Op::hc_prepare);gate.real("hc_eps",0);for(int i=0;i<4;++i){gate.b[3].set(i,-1000);}
    REQUIRE_EQ(gate.run(),RAD_OK);for(float x:gate.b[5].values()){CHECK_EQ(x,0);CHECK(!std::signbit(x));}
}
TEST(ds41_math_derived_dry_run_never_reads_router_output_weights){
    Case router(m::Op::router_topk);auto a=router.args();m::Plan p;REQUIRE_EQ(m::validate(m::Op::router_topk,&a,&p),RAD_OK);
    for(int64_t i=0;i<router.b[3].count();++i)router.b[3].set(i,NAN);
    const auto old_ids=router.b[2].words,old_weights=router.b[3].words;std::vector<float> workspace(size_t(m::workspace_floats(p)));
    REQUIRE_EQ(m::checked_router_row<false>(p,0,workspace.data()),RAD_OK);CHECK(router.b[2].words==old_ids);CHECK(router.b[3].words==old_weights);
    CHECK_EQ(m::workspace_floats(p),2*p.experts+p.topk);
}
RAD_TEST_MAIN()

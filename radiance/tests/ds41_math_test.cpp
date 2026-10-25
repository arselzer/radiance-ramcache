/* CPU emulator of the exact GPU row equations versus the independent pinned DS41 reference.
 * No HIP runtime/device access. All numerical storage boundaries use the public ABI converters. */
#include "rad_test.h"
#include "../libds41/ds41_math_emulate.h"
#include "../arch/common/rad_ds41_ref.h"
#include <array>
#include <random>

namespace math=rad::ds41::math;
namespace ref=rad::deepseek41::ref;
namespace {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> dims,bool pitched=true) {
        t.dtype=dtype;t.rank=dims.size();unsigned i=0;for(auto d:dims)t.shape[i++]=d;
        int64_t stride=pitched?2:1;
        for(int axis=int(t.rank)-1;axis>=0;--axis) {t.stride[axis]=stride;stride=stride*t.shape[axis]+(pitched?3:0);}
        words.resize(size_t(stride*4+32)/4,0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count() const {int64_t n=1;for(unsigned i=0;i<t.rank;++i)n*=t.shape[i];return n;}
    int64_t offset(int64_t flat) const {
        int64_t at=0;for(int axis=int(t.rank)-1;axis>=0;--axis) {at+=(flat%t.shape[axis])*t.stride[axis];flat/=t.shape[axis];}return at;
    }
    float get(int64_t flat) const {return rad_load_f32(words.data(),t.dtype,offset(flat));}
    void set(int64_t flat,float value) {rad_store_f32(words.data(),t.dtype,offset(flat),value);}
    void set_id(int64_t flat,int32_t value) {reinterpret_cast<int32_t*>(words.data())[offset(flat)]=value;}
    int32_t get_id(int64_t flat) const {return reinterpret_cast<const int32_t*>(words.data())[offset(flat)];}
    std::vector<float> values() const {std::vector<float> v;for(int64_t i=0;i<count();++i)v.push_back(get(i));return v;}
};
struct Case {
    math::Op op;std::vector<Buffer> buffers;std::vector<RadTensor> tensors;std::vector<RadParam> params;
    std::array<int32_t,4> status{911,912,913,914};
    explicit Case(math::Op kind,uint32_t dtype=RAD_F32,bool pitched=true):op(kind) {
        const char* dt=rad_dtype_name(dtype);
        if(op==math::Op::hc_prepare||op==math::Op::hc_collapse) {
            params={RAD_INT("M",4),RAD_INT("n",6),RAD_INT("hc",4),RAD_STR("dtype",dt)};
            buffers.emplace_back(dtype,std::initializer_list<int64_t>{2,24},pitched);
            if(op==math::Op::hc_prepare) {
                params.push_back(RAD_INT("iterations",20));params.push_back(RAD_F64("eps",1e-6));params.push_back(RAD_F64("hc_eps",1e-6));
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{24,24},pitched);
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{3},pitched);
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{24},pitched);
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,4},pitched);
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,4},pitched);
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,16},pitched);
            } else {
                buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{2,4},pitched);
                buffers.emplace_back(dtype,std::initializer_list<int64_t>{2,6},pitched);
            }
        } else if(op==math::Op::router_topk) {
            params={RAD_INT("M",4),RAD_INT("experts",12),RAD_INT("topk",6),RAD_F64("temperature",.75),RAD_F64("route_scale",1.5),RAD_INT("normalize",1)};
            buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{3,12},pitched);
            buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{12},pitched);
            buffers.emplace_back(RAD_I32,std::initializer_list<int64_t>{3,6},pitched);
            buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{3,6},pitched);
        } else {
            params={RAD_INT("M",4),RAD_INT("heads",3),RAD_INT("head_dim",8),RAD_INT("kv_rows",7),RAD_INT("topk",4),RAD_F64("scale",.35),RAD_STR("dtype",dt)};
            buffers.emplace_back(dtype,std::initializer_list<int64_t>{2,3,8},pitched);
            buffers.emplace_back(dtype,std::initializer_list<int64_t>{7,8},pitched);
            buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{3},pitched);
            buffers.emplace_back(RAD_I32,std::initializer_list<int64_t>{2,4},pitched);
            buffers.emplace_back(dtype,std::initializer_list<int64_t>{2,3,8},pitched);
        }
        std::mt19937 rng(411017+unsigned(op));
        for(int i=0;i<first_output();++i)for(int64_t j=0;j<buffers[size_t(i)].count();++j) {
            if(buffers[size_t(i)].t.dtype==RAD_I32)buffers[size_t(i)].set_id(j,int32_t(j%5)-1);
            else buffers[size_t(i)].set(j,float(int(rng()%1001)-500)/512.f);
        }
        if(op==math::Op::hc_prepare) {buffers[2].set(0,.6f);buffers[2].set(1,.9f);buffers[2].set(2,1.2f);}
        if(op==math::Op::sparse_attn)buffers[2].set(2,-INFINITY);
        for(auto& b:buffers) {b.t.data=b.words.data();tensors.push_back(b.t);}
    }
    int first_output() const {return op==math::Op::hc_prepare?4:op==math::Op::hc_collapse?2:op==math::Op::router_topk?2:4;}
    RadArgs args() {
        RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());
        a.scratch=status.data();a.scratch_bytes=4;return a;
    }
    int run() {auto a=args();return math::emulate(op,&a);}
    void integer(const char* key,int64_t value) {for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    void real(const char* key,double value) {for(auto& p:params)if(!std::strcmp(p.key,key))p.dval=value;}
};
void expected(Case& c) {
    const auto h=c.buffers[0].values();
    if(c.op==math::Op::hc_prepare) {
        const auto fn=c.buffers[1].values(),sc=c.buffers[2].values(),base=c.buffers[3].values();
        for(int row=0;row<2;++row) {
            ref::MhcMix mix;REQUIRE_EQ(ref::mhc_prepare(ref::Floats(h).subspan(row*24,24),6,4,fn,sc,base,20,1e-6f,1e-6f,&mix),RAD_OK);
            for(int i=0;i<4;++i) {c.buffers[4].set(row*4+i,mix.pre[i]);c.buffers[5].set(row*4+i,mix.post[i]);}
            for(int i=0;i<16;++i)c.buffers[6].set(row*16+i,mix.comb[i]);
        }
    } else if(c.op==math::Op::hc_collapse) {
        const auto pre=c.buffers[1].values();std::vector<float> out(6);
        for(int row=0;row<2;++row) {
            REQUIRE_EQ(ref::mhc_collapse(ref::Floats(h).subspan(row*24,24),ref::Floats(pre).subspan(row*4,4),out),RAD_OK);
            for(int i=0;i<6;++i)c.buffers[2].set(row*6+i,out[i]);
        }
    } else if(c.op==math::Op::router_topk) {
        const auto bias=c.buffers[1].values();
        for(int row=0;row<3;++row) {
            ref::Routing result;REQUIRE_EQ(ref::route(ref::Floats(h).subspan(row*12,12),bias,6,.75f,1.5f,true,&result),RAD_OK);
            for(int i=0;i<6;++i) {c.buffers[2].set_id(row*6+i,result.ids[i]);c.buffers[3].set(row*6+i,result.weights[i]);}
        }
    } else {
        const auto kv=c.buffers[1].values(),sinks=c.buffers[2].values();std::vector<float> out(24);std::array<int32_t,4> ids{};
        for(int row=0;row<2;++row) {
            for(int i=0;i<4;++i)ids[i]=c.buffers[3].get_id(row*4+i);
            REQUIRE_EQ(ref::sparse_attention(ref::Floats(h).subspan(row*24,24),kv,3,8,sinks,ids,.35f,out),RAD_OK);
            for(int i=0;i<24;++i)c.buffers[4].set(row*24+i,out[i]);
        }
    }
}
void compare(Case& c) {
    Case oracle=c;expected(oracle);const auto before=c.buffers;
    REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.status[0],RAD_OK);
    CHECK_EQ(c.status[1],912);CHECK_EQ(c.status[2],913);CHECK_EQ(c.status[3],914);
    for(int i=0;i<c.first_output();++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    for(size_t i=size_t(c.first_output());i<c.buffers.size();++i) {
        const auto& actual=c.buffers[i];const auto& want=oracle.buffers[i];
        std::vector<bool> touched(actual.words.size()*4,false);
        for(int64_t j=0;j<actual.count();++j) {
            if(actual.t.dtype==RAD_I32)CHECK_EQ(actual.get_id(j),want.get_id(j));
            else CHECK_NEAR(actual.get(j),want.get(j),3e-6+3e-5*std::fabs(want.get(j)));
            const int bytes=rad_dtype_bits(actual.t.dtype)/8;
            for(int b=0;b<bytes;++b)touched[size_t(actual.offset(j)*bytes+b)]=true;
        }
        const auto* bytes=reinterpret_cast<const uint8_t*>(actual.words.data());
        const auto* old=reinterpret_cast<const uint8_t*>(before[i].words.data());
        for(size_t b=0;b<touched.size();++b)if(!touched[b])CHECK_EQ(bytes[b],old[b]);
    }
}
void rejects(Case& c,bool payload=true) {
    const auto before=c.buffers;const auto old_status=c.status;
    CHECK(c.run()<0);
    for(size_t i=0;i<before.size();++i)CHECK(c.buffers[i].words==before[i].words);
    if(payload)CHECK(c.status[0]<0);else CHECK(c.status==old_status);
    CHECK_EQ(c.status[1],912);CHECK_EQ(c.status[2],913);CHECK_EQ(c.status[3],914);
}
}

TEST(ds41_math_shared_gpu_equations_match_independent_oracle_at_all_storage_boundaries) {
    for(auto op:{math::Op::hc_prepare,math::Op::hc_collapse,math::Op::router_topk,math::Op::sparse_attn})
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(bool pitched:{false,true}) {Case c(op,dtype,pitched);compare(c);}
}
TEST(ds41_math_sinkhorn_exact_iteration_epsilon_order_and_positive_zero_gates) {
    for(int iters:{1,2,20,256})for(float eps:{0.f,1e-6f,.01f}) {
        Case c(math::Op::hc_prepare);c.integer("iterations",iters);c.real("hc_eps",eps);
        for(int i=0;i<24;++i)c.buffers[3].set(i,i<8?-1000.f:float(i-8)/8.f);
        const auto h=c.buffers[0].values();ref::MhcMix want;
        REQUIRE_EQ(ref::mhc_prepare(ref::Floats(h).first(24),6,4,c.buffers[1].values(),c.buffers[2].values(),c.buffers[3].values(),iters,1e-6f,eps,&want),RAD_OK);
        REQUIRE_EQ(c.run(),RAD_OK);
        for(int i=0;i<4;++i) {
            CHECK_EQ(c.buffers[4].get(i),eps);CHECK_EQ(c.buffers[5].get(i),0.f);
            CHECK(!std::signbit(c.buffers[5].get(i)));
        }
        for(int i=0;i<16;++i)CHECK_NEAR(c.buffers[6].get(i),want.comb[i],2e-7);
    }
}
TEST(ds41_math_router_has_exact_ties_and_correction_never_changes_weight_values) {
    Case c(math::Op::router_topk);
    for(int64_t i=0;i<c.buffers[0].count();++i)c.buffers[0].set(i,0.f);
    for(int i=0;i<12;++i)c.buffers[1].set(i,0.f);
    REQUIRE_EQ(c.run(),RAD_OK);
    for(int row=0;row<3;++row)for(int k=0;k<6;++k)CHECK_EQ(c.buffers[2].get_id(row*6+k),k);
    c.buffers[1].set(11,100.f);REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.buffers[2].get_id(0),11);
    for(int k=0;k<6;++k)CHECK_NEAR(c.buffers[3].get(k),.25f,1e-7);
    c.integer("normalize",0);REQUIRE_EQ(c.run(),RAD_OK);
    CHECK_NEAR(c.buffers[3].get(0),1.5f*std::sqrt(std::log1p(std::exp(0.f))),1e-7);
    c.integer("topk",1);c.tensors[2].shape[1]=1;c.tensors[3].shape[1]=1;c.integer("normalize",1);
    REQUIRE_EQ(c.run(),RAD_OK);CHECK_NEAR(c.buffers[3].get(0),1.5f*std::sqrt(std::log1p(std::exp(0.f))),1e-7);
}
TEST(ds41_math_sparse_sink_has_no_value_and_only_minus_one_is_padding) {
    Case c(math::Op::sparse_attn);
    for(int64_t i=0;i<c.buffers[0].count();++i)c.buffers[0].set(i,0.f);
    for(int64_t i=0;i<c.buffers[1].count();++i)c.buffers[1].set(i,2.f);
    for(int64_t i=0;i<c.buffers[3].count();++i)c.buffers[3].set_id(i,i%4<2?int32_t(i%2):-1);
    for(int64_t i=0;i<3;++i)c.buffers[2].set(i,std::log(2.f));
    REQUIRE_EQ(c.run(),RAD_OK);for(auto v:c.buffers[4].values())CHECK_NEAR(v,1.f,1e-7);
    for(int64_t i=0;i<c.buffers[3].count();++i)c.buffers[3].set_id(i,-1);
    for(int i=0;i<3;++i)c.buffers[2].set(i,-INFINITY);
    REQUIRE_EQ(c.run(),RAD_OK);for(auto v:c.buffers[4].values()) {CHECK_EQ(v,0.f);CHECK(!std::signbit(v));}
    c.buffers[3].set_id(c.buffers[3].count()-1,-2);rejects(c);
}
TEST(ds41_math_device_preflight_emulation_is_batch_atomic_and_bounds_safe) {
    for(auto op:{math::Op::hc_prepare,math::Op::hc_collapse,math::Op::router_topk,math::Op::sparse_attn}) {
        Case c(op);c.buffers[0].set(c.buffers[0].count()-1,NAN);rejects(c);
    }
    Case a(math::Op::sparse_attn);a.buffers[3].set_id(a.buffers[3].count()-1,7);rejects(a);
    Case b(math::Op::sparse_attn);b.buffers[2].set(2,INFINITY);rejects(b);
    Case c(math::Op::sparse_attn);c.buffers[1].set(6*8,NAN);REQUIRE_EQ(c.run(),RAD_OK); // key 6 unselected
    c.buffers[3].set_id(c.buffers[3].count()-1,6);rejects(c);
}
TEST(ds41_math_geometry_alias_stride_and_scratch_errors_do_not_touch_flag_or_outputs) {
    for(auto op:{math::Op::hc_prepare,math::Op::hc_collapse,math::Op::router_topk,math::Op::sparse_attn}) {
        Case c(op);const auto good=c.tensors;
        c.tensors.back().data=c.tensors.front().data;rejects(c,false);c.tensors=good;
        c.tensors.front().stride[c.tensors.front().rank-1]=0;rejects(c,false);c.tensors=good;
        c.tensors.front().stride[0]=INT64_MAX;rejects(c,false);c.tensors=good;
        c.tensors.front().data=nullptr;rejects(c,false);c.tensors=good;
        c.tensors.back().dtype=RAD_I16;rejects(c,false);c.tensors=good;
        auto args=c.args();math::Plan plan;
        args.scratch_bytes=3;CHECK(math::validate(op,&args,&plan)<0);
        args=c.args();args.scratch=nullptr;CHECK(math::validate(op,&args,&plan)<0);
        args=c.args();args.scratch=static_cast<uint8_t*>(args.scratch)+1;CHECK(math::validate(op,&args,&plan)<0);
        args=c.args();args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);CHECK_EQ(math::validate(op,&args,&plan),RAD_E_SHAPE);
        args=c.args();args.scratch=c.tensors.front().data;CHECK(math::validate(op,&args,&plan)<0);
        CHECK_EQ(c.status[0],911);CHECK_EQ(ds41_math_scratch(nullptr),rad::ds41::gpu_scratch_bytes<math::Plan>());
        CHECK(math::validate(op,nullptr,&plan)<0);CHECK(math::validate(op,&args,nullptr)<0);
    }
}
RAD_TEST_MAIN()

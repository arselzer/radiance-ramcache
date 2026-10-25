#include "quant_fixture.h"
#include "../libds41/ds41_ffn_fixture.h"
#include "../libds41/ds41_packed_args.h"
#include "../arch/deepseek41/block_ref.h"
#include <memory>

namespace ffn=rad::ds41::ffn;
namespace fix=ffn::testing;
namespace ref=rad::deepseek41::ref;
namespace {
const RadKernelInfo* actual_post() {
    static rad::Registry registry;static bool loaded=false;
    if(!loaded) {loaded=true;if(registry.load_plugin(home()+"/kernels/libref.so",0)!=RAD_OK)throw std::runtime_error("missing real libref plugin");}
    for(const auto& row:registry.rows())if(!std::strcmp(row.info->op,"mhc_post"))return row.info;
    throw std::runtime_error("missing actual mhc_post");
}
void rejects(fix::Fixture& c,bool payload=true) {
    const auto before=c.buffers;const auto status=c.flag;CHECK(c.run()<0);
    for(size_t i=0;i<before.size();++i)CHECK(c.buffers[i].words==before[i].words);
    if(payload)CHECK(c.flag[0]<0);else CHECK(c.flag==status);
    CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
void expected(fix::Fixture& c) {
    const int yi=int(c.buffers.size())-1;
    if(c.op==ffn::Op::expert_act) {
        const auto gate=c.buffers[0].values(),up=c.buffers[1].values();std::vector<float> activation(7);
        for(int row=0;row<3;++row) {
            REQUIRE_EQ(ref::swiglu(ref::Floats(gate).subspan(row*7,7),ref::Floats(up).subspan(row*7,7),1.25f,activation),RAD_OK);
            for(int i=0;i<7;++i)c.buffers[yi].setf(row*7+i,activation[i]*c.buffers[2].getf(row));
        }
    } else if(c.op==ffn::Op::moe_combine) {
        for(int row=0;row<3;++row)for(int d=0;d<7;++d) {
            float sum=0;for(int e=0;e<4;++e)for(int job=0;job<6;++job)
                if(c.buffers[1].geti(job)==e&&c.buffers[2].geti(job)==row)sum+=c.buffers[0].getf(job*7+d);
            c.buffers[yi].setf(row*7+d,sum+c.buffers[3].getf(row*7+d));
        }
    } else {
        const auto x=c.buffers[0].values(),h=c.buffers[1].values(),post=c.buffers[2].values(),comb=c.buffers[3].values();
        std::vector<float> output(28);
        for(int row=0;row<2;++row) {
            ref::MhcMix mix;mix.copies=4;mix.post.assign(post.begin()+row*4,post.begin()+(row+1)*4);mix.comb.assign(comb.begin()+row*16,comb.begin()+(row+1)*16);
            REQUIRE_EQ(ref::mhc_expand(ref::Floats(x).subspan(row*7,7),ref::Floats(h).subspan(row*28,28),mix,output),RAD_OK);
            for(int i=0;i<28;++i)c.buffers[yi].setf(row*28+i,output[i]);
        }
    }
}
void compare(fix::Fixture& c) {
    auto oracle=c;expected(oracle);const auto before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
    for(size_t i=0;i+1<c.buffers.size();++i)CHECK(c.buffers[i].words==before[i].words);
    CHECK(c.buffers.back().words==oracle.buffers.back().words);CHECK_EQ(c.flag[1],912);
}
struct Weight {
    const char* name;int values,bytes;int64_t rows,cols;std::vector<uint8_t> raw;std::vector<float> decoded;
    Weight(const char* format,int block_values,int block_bytes,int64_t r,int64_t c,unsigned seed)
        :name(format),values(block_values),bytes(block_bytes),rows(r),cols(c) {
        std::mt19937 rng(seed);raw.resize(size_t(rows*(cols/values)*bytes));for(auto& v:raw)v=uint8_t(rng());
        for(size_t at=0;at<raw.size();at+=size_t(bytes)) {raw[at]=0;raw[at+1]=0x18;if(!std::strcmp(name,"q5_K")){raw[at+2]=0;raw[at+3]=0x10;}}
        const auto* importer=qfix::reg().checkpoint_importer("gguf",name);if(!importer)throw std::runtime_error("missing real importer");
        RadCheckpointImportWeight weight{"tiny-FFN-weight",name,rows,cols};RadEncoding encoding{};
        if(importer->info->encoding(&weight,&encoding)!=RAD_OK)throw std::runtime_error("encoding failure");
        std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> planes;void* dst[RAD_ENC_MAX_PLANES]{};const void* src[RAD_ENC_MAX_PLANES]{};
        for(int p=0;p<encoding.n_planes;++p) {planes[p].resize(size_t(rad_enc_plane_bytes(&encoding.plane[p],rows,cols)));dst[p]=planes[p].data();src[p]=dst[p];}
        if(importer->info->import_rows(&weight,raw.data(),raw.size(),rows,dst)!=RAD_OK)throw std::runtime_error("import failure");
        rad::EncodedView view;rad::enc_view(encoding,rows,cols,src,&view);decoded.resize(size_t(rows*cols));
        if(rad::enc_decode_rows(view,0,rows,decoded.data())!=RAD_OK)throw std::runtime_error("generic decoder failure");
    }
};
}

TEST(ds41_ffn_weighted_activation_combine_and_existing_post_match_independent_oracles) {
    for(auto op:{ffn::Op::expert_act,ffn::Op::moe_combine,ffn::Op::mhc_post})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch) {
        fix::Fixture c(op,dtype,pitch);compare(c);
        if(op==ffn::Op::mhc_post) {
            const auto expected_words=c.buffers.back().words;auto a=c.args();REQUIRE_EQ(actual_post()->launch(&a,nullptr),RAD_OK);
            CHECK(c.buffers.back().words==expected_words);
        }
    }
}
TEST(ds41_ffn_route_weight_is_applied_before_the_only_down_input_cast) {
    std::mt19937 rng(1201017);bool distinguishes=false;
    fix::Fixture c(ffn::Op::expert_act,RAD_BF16,2);c.real("limit",0);
    for(int trial=0;trial<200&&!distinguishes;++trial) {
        const float g=float(int(rng()%3001)-1500)/128.f,u=float(int(rng()%3001)-1500)/128.f,w=float(1+rng()%999)/1000.f;
        c.buffers[0].setf(0,g);c.buffers[1].setf(0,u);c.buffers[2].setf(0,w);
        const float stored_g=c.buffers[0].getf(0),stored_u=c.buffers[1].getf(0);
        std::vector<float> act(1);REQUIRE_EQ(ref::swiglu(std::vector<float>{stored_g},std::vector<float>{stored_u},0,act),RAD_OK);
        const uint16_t correct=rad_f32_to_bf16(act[0]*w),wrong=rad_f32_to_bf16(rad_bf16_to_f32(rad_f32_to_bf16(act[0]))*w);
        REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(rad_f32_to_bf16(c.buffers.back().getf(0)),correct);distinguishes=correct!=wrong;
    }
    CHECK(distinguishes);
}
TEST(ds41_ffn_combine_sums_in_expert_id_order_before_shared_and_final_cast) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}) {
        fix::Fixture c(ffn::Op::moe_combine,dtype,2);
        const float big=dtype==RAD_F16?65504.f:1e8f,small=dtype==RAD_F16?1.f/1048576.f:1.f;
        c.buffers[0].setf(0,small);c.buffers[0].setf(7,big);c.buffers[0].setf(3*7,-big);c.buffers[3].setf(0,0);
        REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.buffers.back().getf(0),small);
        c.integer("tp_world",4);rejects(c,false);
    }
}
TEST(ds41_ffn_all_payloads_are_validated_before_any_output_write) {
    for(auto op:{ffn::Op::expert_act,ffn::Op::moe_combine,ffn::Op::mhc_post}) {
        fix::Fixture c(op);c.buffers[0].setf(c.buffers[0].count()-1,NAN);rejects(c);
    }
    for(int fault=0;fault<5;++fault) {
        fix::Fixture c(ffn::Op::moe_combine);
        if(fault==0)c.buffers[1].seti(5,4);
        if(fault==1)c.buffers[2].seti(5,3);
        if(fault==2) {c.buffers[1].seti(5,0);c.buffers[2].seti(5,2);}
        if(fault==3)c.buffers[1].seti(5,-1);
        if(fault==4)c.buffers[3].setf(c.buffers[3].count()-1,INFINITY);
        rejects(c);
    }
    fix::Fixture c(ffn::Op::expert_act);c.buffers[2].setf(2,-.25f);rejects(c);
}
TEST(ds41_ffn_metadata_refuses_aliases_bad_pitches_dtypes_and_scratch) {
    for(auto op:{ffn::Op::expert_act,ffn::Op::moe_combine,ffn::Op::mhc_post}) {
        fix::Fixture c(op);auto args=c.args();ffn::Plan p;const auto tensors=c.tensors;
        c.tensors.back().data=c.tensors.front().data;CHECK(ffn::validate(op,&args,&p)<0);c.tensors=tensors;
        c.tensors.front().stride[1]=0;CHECK(ffn::validate(op,&args,&p)<0);c.tensors=tensors;
        c.tensors.back().dtype=RAD_U32;CHECK(ffn::validate(op,&args,&p)<0);c.tensors=tensors;
        args.scratch_bytes=3;CHECK(ffn::validate(op,&args,&p)<0);CHECK_EQ(c.flag[0],911);
    }
}
TEST(ds41_ffn_tiny_native_packed_experts_through_existing_mhc_post_preserve_boundaries) {
    struct Format {const char* name;int values,bytes;};
    for(const auto format:{Format{"iq2_xs",256,74},Format{"q5_K",256,176},Format{"q8_0",32,34}})
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}) {
        constexpr int M=2,D=256,I=256,E=3,R=4;
        std::vector<Weight> weights;for(int e=0;e<E+1;++e)for(int kind=0;kind<3;++kind)weights.emplace_back(format.name,format.values,format.bytes,I,D,1017+e*3+kind);
        fix::Buffer x(dtype,{M,D},0);for(int64_t j=0;j<x.count();++j)x.setf(j,float(int(j*17%101)-50)/128.f);
        std::vector<int32_t> ids{2,0,1,0},rows{0,0,1,1},identity{0,1,2,3};std::vector<float> route_weights{.65f,.35f,.2f,.8f};
        fix::Buffer gate(dtype,{R,I},0),up(dtype,{R,I},0),down_input(dtype,{R,I},0),parts(dtype,{R,D},0),shared(dtype,{M,D},0),combined(dtype,{M,D},0);
        auto matrix=[](void* data,uint32_t dt,int64_t a,int64_t b){RadTensor t{};t.data=data;t.dtype=dt;t.rank=2;t.shape[0]=a;t.shape[1]=b;t.stride[0]=b;t.stride[1]=1;return t;};
        auto vector=[](void* data,int64_t a){RadTensor t{};t.data=data;t.dtype=RAD_I32;t.rank=1;t.shape[0]=a;t.stride[0]=1;return t;};
        const int64_t row_bytes=D/format.values*format.bytes;std::vector<RadParam> packed_params{RAD_INT("M",R),RAD_INT("N",I),RAD_INT("K",D),RAD_STR("format",format.name),RAD_STR("a_dtype",rad_dtype_name(dtype)),RAD_STR("y_dtype",rad_dtype_name(dtype)),RAD_INT("R",R),RAD_INT("n_expert",E)};
        for(int kind=0;kind<3;++kind) {
            std::vector<const uint8_t*> table;for(int e=0;e<E;++e)table.push_back(weights[size_t(e*3+kind)].raw.data());
            RadTensor wt{};wt.data=table.data();wt.dtype=RAD_U8;wt.rank=3;wt.shape[0]=E;wt.shape[1]=I;wt.shape[2]=row_bytes;wt.stride[0]=0;wt.stride[1]=row_bytes;wt.stride[2]=1;
            std::vector<int32_t> status(R,999);RadTensor t[]{matrix(kind==2?down_input.words.data():x.words.data(),dtype,kind==2?R:M,D),wt,vector(ids.data(),R),vector(kind==2?identity.data():rows.data(),R),matrix(kind==0?gate.words.data():kind==1?up.words.data():parts.words.data(),dtype,R,I),vector(status.data(),R)};
            RadArgs a{};a.t=t;a.n_t=6;a.p=packed_params.data();a.n_p=packed_params.size();REQUIRE_EQ(ds41_ref_packed_routed(&a,nullptr),RAD_OK);
            if(kind==1) {
                RadTensor weight=vector(route_weights.data(),R);weight.dtype=RAD_F32;RadTensor at[]{matrix(gate.words.data(),dtype,R,I),matrix(up.words.data(),dtype,R,I),weight,matrix(down_input.words.data(),dtype,R,I)};
                const RadParam ap[]{RAD_INT("M",R),RAD_INT("n",I),RAD_F64("limit",.5),RAD_STR("dtype",rad_dtype_name(dtype))};int32_t flag=99;
                RadArgs act{};act.t=at;act.n_t=4;act.p=ap;act.n_p=4;act.scratch=&flag;act.scratch_bytes=4;REQUIRE_EQ(ffn::emulate(ffn::Op::expert_act,&act),RAD_OK);
            }
        }
        fix::Buffer shared_gate(dtype,{M,I},0),shared_up(dtype,{M,I},0),shared_input(dtype,{M,I},0);
        std::vector<float> shared_weight(M,1.f);
        for(int kind=0;kind<3;++kind) {
            RadTensor t[]{matrix(kind==2?shared_input.words.data():x.words.data(),dtype,M,D),
                matrix(weights[size_t(E*3+kind)].raw.data(),RAD_U8,I,row_bytes),
                matrix(kind==0?shared_gate.words.data():kind==1?shared_up.words.data():shared.words.data(),dtype,M,I)};
            RadArgs a{};a.t=t;a.n_t=3;a.p=packed_params.data();a.n_p=6;REQUIRE_EQ(ds41_ref_packed_gemm(&a,nullptr),RAD_OK);
            if(kind==1) {
                RadTensor weight=vector(shared_weight.data(),M);weight.dtype=RAD_F32;
                RadTensor at[]{matrix(shared_gate.words.data(),dtype,M,I),matrix(shared_up.words.data(),dtype,M,I),weight,matrix(shared_input.words.data(),dtype,M,I)};
                const RadParam ap[]{RAD_INT("M",M),RAD_INT("n",I),RAD_F64("limit",.5),RAD_STR("dtype",rad_dtype_name(dtype))};int32_t flag=99;
                RadArgs act{};act.t=at;act.n_t=4;act.p=ap;act.n_p=4;act.scratch=&flag;act.scratch_bytes=4;
                REQUIRE_EQ(ffn::emulate(ffn::Op::expert_act,&act),RAD_OK);
            }
        }
        const auto precision=dtype==RAD_BF16?ref::Precision::bf16:ref::Precision::f32;
        // Generic canonical decodes supply the independent logical weights. Explicit storage
        // conversion below also covers F16, which the reference Precision enum does not encode.
        std::vector<float> oracle(size_t(M*D),0),shared_values(size_t(M*D));
        auto expert=[&](int e,int token,float weight) {
            const auto input=x.values();std::vector<float> g(I),u(I),a(I),out(D);
            REQUIRE_EQ(ref::linear(ref::Floats(input).subspan(token*D,D),weights[size_t(e*3)].decoded,I,g),RAD_OK);
            REQUIRE_EQ(ref::linear(ref::Floats(input).subspan(token*D,D),weights[size_t(e*3+1)].decoded,I,u),RAD_OK);
            for(int i=0;i<I;++i) {g[i]=dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(g[i])):ref::rounded(g[i],precision);u[i]=dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(u[i])):ref::rounded(u[i],precision);}
            REQUIRE_EQ(ref::swiglu(g,u,.5f,a),RAD_OK);
            for(float& v:a) {v*=weight;v=dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(v)):ref::rounded(v,precision);}
            REQUIRE_EQ(ref::linear(a,weights[size_t(e*3+2)].decoded,D,out),RAD_OK);
            for(float& v:out)v=dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(v)):ref::rounded(v,precision);
            if(e==E)for(int d=0;d<D;++d) {CHECK_EQ(shared.getf(token*D+d),out[d]);shared_values[size_t(token*D+d)]=out[d];}
            else for(int d=0;d<D;++d)oracle[size_t(token*D+d)]+=out[d];
        };
        for(int e=0;e<E;++e)for(int job=0;job<R;++job)if(ids[size_t(job)]==e)expert(e,rows[size_t(job)],route_weights[size_t(job)]);
        for(int token=0;token<M;++token)expert(E,token,1.f);
        RadTensor combine_t[]{matrix(parts.words.data(),dtype,R,D),vector(ids.data(),R),vector(rows.data(),R),matrix(shared.words.data(),dtype,M,D),matrix(combined.words.data(),dtype,M,D)};
        const RadParam combine_p[]{RAD_INT("M",M),RAD_INT("R",R),RAD_INT("n",D),RAD_INT("n_expert",E),RAD_STR("dtype",rad_dtype_name(dtype)),RAD_INT("tp_world",1)};
        int32_t flag=99;RadArgs ca{};ca.t=combine_t;ca.n_t=5;ca.p=combine_p;ca.n_p=6;ca.scratch=&flag;ca.scratch_bytes=4;REQUIRE_EQ(ffn::emulate(ffn::Op::moe_combine,&ca),RAD_OK);
        for(int64_t j=0;j<M*D;++j) {const float v=oracle[size_t(j)]+shared_values[size_t(j)];const float stored=dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(v)):ref::rounded(v,precision);CHECK_NEAR(combined.getf(j),stored,3e-6+2e-5*std::fabs(stored));}
        fix::Buffer residual(dtype,{M,4*D},0),post(RAD_F32,{M,4},0),comb(RAD_F32,{M,16},0),out(dtype,{M,4*D},0);
        for(int64_t j=0;j<residual.count();++j)residual.setf(j,float(int(j%29)-14)/32.f);
        for(int64_t j=0;j<post.count();++j)post.setf(j,float(j%4+1)/4.f);
        for(int64_t j=0;j<comb.count();++j)comb.setf(j,float((j/4)%4+j%4+1)/16.f);
        RadTensor pt[]{matrix(combined.words.data(),dtype,M,D),matrix(residual.words.data(),dtype,M,4*D),matrix(post.words.data(),RAD_F32,M,4),matrix(comb.words.data(),RAD_F32,M,16),matrix(out.words.data(),dtype,M,4*D)};
        const RadParam pp[]{RAD_INT("M",M),RAD_INT("n",D),RAD_INT("hc",4),RAD_STR("dtype",rad_dtype_name(dtype))};RadArgs pa{};pa.t=pt;pa.n_t=5;pa.p=pp;pa.n_p=4;pa.scratch=&flag;pa.scratch_bytes=4;
        REQUIRE_EQ(ffn::emulate(ffn::Op::mhc_post,&pa),RAD_OK);const auto expected=out.words;
        REQUIRE_EQ(actual_post()->launch(&pa,nullptr),RAD_OK);CHECK(out.words==expected);
    }
}
RAD_TEST_MAIN()

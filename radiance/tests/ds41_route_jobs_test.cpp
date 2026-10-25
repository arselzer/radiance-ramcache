#include "rad_test.h"
#include "../libds41/ds41_route_jobs_fixture.h"
#include "../libds41/ds41_route_jobs_report.h"
#include "../libds41/ds41_packed_args.h"
#include "runtime/ctx.h"
#include "device/device.h"
#include <limits>
#include <numeric>

namespace r=rad::ds41::route_jobs;
namespace fix=r::testing;
namespace {
void unchanged(const fix::Fixture& c,const std::vector<fix::Buffer>& before){for(size_t i=0;i<before.size();++i)CHECK(c.buffers[i].words==before[i].words);}
void rejects(fix::Fixture& c,bool payload){const auto before=c.buffers;const auto flag=c.flag;CHECK(c.run()<0);unchanged(c,before);
    if(payload)CHECK(c.flag[0]<0);else CHECK(c.flag==flag);for(size_t i=1;i<c.flag.size();++i)CHECK_EQ(c.flag[i],flag[i]);}
void literals(fix::Fixture& c){
    const std::array<int32_t,6> ids{0,1,3,4,1,2},tokens{0,0,1,1,2,2},identity{0,1,2,3,4,5};const std::array<int32_t,5> counts{1,2,1,1,1};
    const auto before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
    for(int64_t j=0;j<6;++j){CHECK_EQ(c.buffers[r::expert_ids].id(j),ids[size_t(j)]);CHECK_EQ(c.buffers[r::token_ids].id(j),tokens[size_t(j)]);
        CHECK_EQ(c.buffers[r::identity_rows].id(j),identity[size_t(j)]);CHECK_EQ(c.buffers[r::job_weights].words[size_t(c.buffers[r::job_weights].offset(j))],before[r::weights].words[size_t(before[r::weights].offset(j))]);}
    for(int i=0;i<5;++i)CHECK_EQ(c.buffers[r::expert_counts].id(i),counts[size_t(i)]);
    for(int i=0;i<2;++i)CHECK(c.buffers[i].words==before[i].words);
    for(int i=2;i<r::operand_count;++i){std::vector<bool> touched(c.buffers[i].words.size());for(int64_t j=0;j<c.buffers[i].count();++j)touched[size_t(c.buffers[i].offset(j))]=true;
        for(size_t j=0;j<touched.size();++j)if(!touched[j])CHECK_EQ(c.buffers[i].words[j],before[i].words[j]);}
}
thread_local const RadRouting* report=nullptr;
void reporting_step(RadCtx* ctx,const RadBatch*){if(rad_route_report(ctx,0,report)!=RAD_OK)rad_step_fail(ctx,"count handoff failed");}
RadTensor matrix(void* data,uint32_t dtype,int64_t m,int64_t n){RadTensor t{};t.data=data;t.dtype=dtype;t.rank=2;t.shape[0]=m;t.shape[1]=n;t.stride[0]=n;t.stride[1]=1;return t;}
RadTensor vector(void* data,uint32_t dtype,int64_t n){RadTensor t{};t.data=data;t.dtype=dtype;t.rank=1;t.shape[0]=n;t.stride[0]=1;return t;}
}
TEST(ds41_route_jobs_flattening_identity_histogram_and_pitches_match_independent_literals){for(int pitch=0;pitch<3;++pitch){fix::Fixture c(pitch);literals(c);}}
TEST(ds41_route_jobs_preserves_zero_and_finite_weight_bits_without_filtering_or_normalizing){
    fix::Fixture c(2);const std::array<float,6> values{0.f,-0.f,std::numeric_limits<float>::denorm_min(),1.f,17.f,std::numeric_limits<float>::max()};
    for(int i=0;i<6;++i)c.buffers[r::weights].weight(i,values[size_t(i)]);
    REQUIRE_EQ(c.run(),RAD_OK);
    for(int i=0;i<6;++i)CHECK_EQ(c.buffers[r::job_weights].words[size_t(c.buffers[r::job_weights].offset(i))],rad::ds41::packed::float_bits(values[size_t(i)]));
    CHECK_EQ(c.buffers[r::expert_counts].id(0),1);CHECK_EQ(c.buffers[r::expert_counts].id(1),2);
}
TEST(ds41_route_jobs_checks_entire_batch_before_clearing_histogram_or_any_output){
    for(int pitch=0;pitch<3;++pitch)for(int fault=0;fault<8;++fault){fix::Fixture c(pitch);
        if(fault==0)c.buffers[r::ids].id(5,-1);
        if(fault==1)c.buffers[r::ids].id(5,5);
        if(fault==2)c.buffers[r::weights].weight(5,-.25f);
        if(fault==3)c.buffers[r::weights].weight(5,NAN);
        if(fault==4)c.buffers[r::weights].weight(5,INFINITY);
        if(fault==5)c.buffers[r::weights].weight(5,-INFINITY);
        if(fault>=6){c.buffers[r::ids].id(5,c.buffers[r::ids].id(4));if(fault==7)c.buffers[r::weights].weight(5,0.f);}
        rejects(c,true);
    }
}
TEST(ds41_route_jobs_rejects_bad_capacity_dtype_stride_span_and_every_output_alias){
    for(int output=2;output<r::operand_count;++output)for(int other=0;other<r::operand_count;++other)if(output!=other){fix::Fixture c;auto a=c.args();const auto before=c.buffers;const auto flag=c.flag;
        c.tensors[output].data=c.tensors[other].data;CHECK(r::emulate(&a)<0);unchanged(c,before);CHECK(c.flag==flag);}
    for(int fault=0;fault<17;++fault){fix::Fixture c;auto a=c.args();const auto before=c.buffers;const auto flag=c.flag;
        if(fault==0)c.params[0].ival=2;
        if(fault==1)c.params[1].ival=6;
        if(fault==2)c.params[2].ival=4097;
        if(fault==3)c.params[0].ival=INT_MAX;
        if(fault==4)c.tensors[0].stride[1]=0;
        if(fault==5)c.tensors[0].stride[0]=1;
        if(fault==6)c.tensors[6].dtype=RAD_F32;
        if(fault==7)c.tensors[5].data=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==8)c.tensors[0].data=static_cast<uint8_t*>(c.tensors[0].data)+1;
        if(fault==9)a.scratch_bytes=3;
        if(fault==10)a.scratch=c.tensors[0].data;
        if(fault==11)a.scratch=c.tensors[5].data;
        if(fault==12)c.tensors[0].shape[0]=0;
        if(fault==13)c.tensors[6].shape[0]=5;
        if(fault==14)c.tensors[5].stride[0]=-1;
        if(fault==15)a.n_t=6;
        if(fault==16)a.scratch=static_cast<uint8_t*>(a.scratch)+1;
        CHECK(r::emulate(&a)<0);unchanged(c,before);CHECK(c.flag==flag);
    }
}
TEST(ds41_route_jobs_accepts_single_expert_and_maximum_topk_without_count_overflow){
    fix::Fixture one(0,4,1,1);REQUIRE_EQ(one.run(),RAD_OK);CHECK_EQ(one.buffers[r::expert_counts].id(0),4);
    fix::Fixture full(0,1,4096,4096);REQUIRE_EQ(full.run(),RAD_OK);
    for(int i=0;i<4096;++i){CHECK_EQ(full.buffers[r::expert_ids].id(i),i);CHECK_EQ(full.buffers[r::expert_counts].id(i),1);CHECK_EQ(full.buffers[r::token_ids].id(i),0);CHECK_EQ(full.buffers[r::identity_rows].id(i),i);}
}
TEST(ds41_route_jobs_contiguous_view_exposes_both_gemm_mappings_and_no_fake_sorting){
    for(int pitch=0;pitch<3;++pitch){fix::Fixture c(pitch);auto a=c.args();r::ContiguousJobs view;
        CHECK_EQ(r::contiguous_view(&a,&view),pitch==2?RAD_E_STRIDE:RAD_OK);
        if(pitch!=2){CHECK_EQ(view.routing.top_k,2);CHECK_EQ(view.routing.n_expert,5);CHECK_EQ(view.jobs,6);CHECK_EQ(view.rows,3);
            CHECK(view.routing.sorted_tok==nullptr);CHECK(view.routing.expert_offset==nullptr);CHECK(view.routing.expert_ids==c.tensors[2].data);CHECK(view.token_ids==c.tensors[3].data);CHECK(view.identity_rows==c.tensors[6].data);}
    }
}
TEST(ds41_route_jobs_actual_core_heat_handoff_consumes_counts_without_sorted_fields){
    if(!rad::device_is_host())SKIP("host backend only");
    fix::Fixture c;rad::Program p;p.meta.n_layers=1;p.meta.n_expert=5;p.weights.resize(1);p.buffers.resize(1);p.ops.resize(1);
    rad::Ctx ctx;rad::CtxDesc d;d.program=&p;d.arch_step=reporting_step;d.record_passes=false;REQUIRE_EQ(ctx.init(d),RAD_OK);
    auto a=c.args();c.tensors[r::expert_counts].data=rad_route_counts(&ctx,0,5);REQUIRE(a.t[r::expert_counts].data);REQUIRE_EQ(r::emulate(&a),RAD_OK);
    r::ContiguousJobs view;REQUIRE_EQ(r::contiguous_view(&a,&view),RAD_OK);RadBatch b{};b.n_tok=3;b.n_seq=1;b.step=0;b.phase=RAD_PHASE_PREFILL;
    report=&view.routing;REQUIRE_EQ(ctx.run_step(&b),RAD_OK);report=nullptr;REQUIRE_EQ(rad_stream_sync(ctx.stream()),RAD_OK);
    const int bank=ctx.next_histogram(0,-1,1);REQUIRE(bank>=0);const auto* counts=ctx.last_histogram(0,bank);REQUIRE(counts);
    const std::array<int32_t,5> expected{1,2,1,1,1};for(int i=0;i<5;++i)CHECK_EQ(counts[i],expected[size_t(i)]);
    RadRouting counts_only{};counts_only.expert_count=view.routing.expert_count;counts_only.n_expert=5;counts_only.top_k=2;
    ++b.step;report=&counts_only;REQUIRE_EQ(ctx.run_step(&b),RAD_OK);report=nullptr;REQUIRE_EQ(rad_stream_sync(ctx.stream()),RAD_OK);
    const auto* retained=ctx.last_routing(0);REQUIRE(retained);CHECK(retained->expert_ids==nullptr);CHECK(retained->sorted_tok==nullptr);CHECK(retained->expert_offset==nullptr);
    const int next=ctx.next_histogram(0,0,2);REQUIRE(next>=0);const auto* latest=ctx.last_histogram(0,next);REQUIRE(latest);for(int i=0;i<5;++i)CHECK_EQ(latest[i],expected[size_t(i)]);
}
TEST(ds41_route_jobs_token_and_identity_rows_drive_actual_packed_gate_up_and_down_gathers){
    fix::Fixture c;REQUIRE_EQ(c.run(),RAD_OK);auto a=c.args();r::ContiguousJobs view;REQUIRE_EQ(r::contiguous_view(&a,&view),RAD_OK);
    std::vector<std::vector<uint8_t>> weights(5,std::vector<uint8_t>(34));std::vector<const uint8_t*> table;
    for(int e=0;e<5;++e){weights[size_t(e)][0]=0;weights[size_t(e)][1]=0x3c;std::fill(weights[size_t(e)].begin()+2,weights[size_t(e)].end(),uint8_t(e+1));table.push_back(weights[size_t(e)].data());}
    const RadParam params[]{RAD_INT("M",6),RAD_INT("N",1),RAD_INT("K",32),RAD_INT("R",6),RAD_INT("n_expert",5),RAD_STR("format","q8_0"),RAD_STR("a_dtype","f32"),RAD_STR("y_dtype","f32")};
    for(bool down:{false,true}){const int m=down?6:3;std::vector<float> input(size_t(m*32)),output(6);std::vector<int32_t> status(6,999);
        for(int row=0;row<m;++row)for(int i=0;i<32;++i)input[size_t(row*32+i)]=float(row+1)/32.f;
        RadTensor wt{};wt.data=table.data();wt.dtype=RAD_U8;wt.rank=3;wt.shape[0]=5;wt.shape[1]=1;wt.shape[2]=34;wt.stride[0]=0;wt.stride[1]=34;wt.stride[2]=1;
        RadTensor tensors[]{matrix(input.data(),RAD_F32,m,32),wt,vector(const_cast<int32_t*>(view.routing.expert_ids),RAD_I32,6),
            vector(const_cast<int32_t*>(down?view.identity_rows:view.token_ids),RAD_I32,6),matrix(output.data(),RAD_F32,6,1),vector(status.data(),RAD_I32,6)};
        RadArgs packed{};packed.t=tensors;packed.n_t=6;packed.p=params;packed.n_p=8;REQUIRE_EQ(ds41_ref_packed_routed(&packed,nullptr),RAD_OK);
        for(int job=0;job<6;++job){const int input_row=down?job:job/2;CHECK_EQ(output[size_t(job)],float(input_row+1)*float(view.routing.expert_ids[job]+1));CHECK_EQ(status[size_t(job)],RAD_OK);}
    }
}
RAD_TEST_MAIN()

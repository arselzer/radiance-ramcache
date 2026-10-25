#include "rad_test.h"
#include "../libds41/ds41_compress_finish_fixture.h"
#include "../libds41/ds41_compress_fixture.h"
#include <sys/mman.h>
#include <limits>
namespace f=rad::ds41::compress_finish;
namespace fix=f::testing;
namespace comp=rad::ds41::compress;
namespace ref=rad::deepseek41::ref;
namespace {
void compare(fix::Fixture& c){
    const auto expected=c.expected(),before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
    for(int i=0;i<f::operand_count;++i)CHECK(c.buffers[size_t(i)].words==(i<f::unrotated?before:expected)[size_t(i)].words);
    CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
void reject(fix::Fixture& c){
    const auto before=c.buffers;CHECK(c.run()<0);CHECK(c.flag[0]<0);for(int i=0;i<f::operand_count;++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
struct Pages {
    uint8_t* ptr=nullptr;size_t bytes=0;
    explicit Pages(size_t n):bytes(n){ptr=static_cast<uint8_t*>(mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));if(ptr==MAP_FAILED){ptr=nullptr;throw std::runtime_error("mmap fixture failed");}}
    ~Pages(){if(ptr)munmap(ptr,bytes);}
    void protect(size_t off,size_t size){if(mprotect(ptr+off,size,PROT_NONE))throw std::runtime_error("mprotect fixture failed");}
};
}
TEST(ds41_compress_finish_cast_norm_rope_matches_independent_composition_for_all_scalar_profiles){
    for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})for(int64_t dim:{2,7,128,512,1024})for(int ratio:{1,2})for(int pitch:{0,1,2}){
        fix::Fixture c(ad,yd,dim,ratio,pitch);compare(c);
    }
}
TEST(ds41_compress_finish_odd_pool_chunks_match_reference_compressor_then_original_position_rope){
    for(int ratio:{1,2})for(int64_t dim:{128,512}){
        comp::testing::Fixture a(dim,ratio,RAD_F32,1,{3},{0},{0});REQUIRE_EQ(a.run(),RAD_OK);
        comp::testing::Fixture b(dim,ratio,RAD_F32,1,{4},{3},{0});b.buffer(comp::pending_state).words=a.buffer(comp::pending_state).words;REQUIRE_EQ(b.run(),RAD_OK);
        fix::Fixture gain_fixture(RAD_F32,RAD_BF16,dim,ratio);ref::Compressor reference(size_t(ratio),gain_fixture.buffers[f::gain].values(),1e-6f,"finish-odd-chunks",ref::Precision::bf16);
        for(const auto* chunk:{&a,&b}){
            fix::Fixture c(RAD_F32,RAD_BF16,dim,ratio,2,chunk->tokens);
            for(int64_t row=0;row<c.rows;++row){c.buffers[f::emit].id(row,chunk->buffer(comp::emit).geti(row));c.buffers[f::first_positions].id(row,chunk->buffer(comp::first_position).geti(row));
                for(int64_t d=0;d<dim;++d)c.buffers[f::pooled].set(row*dim+d,chunk->buffer(comp::emit).geti(row)?chunk->buffer(comp::pooled).getf(row*dim+d):NAN);}
            compare(c);
            for(int64_t row=0;row<c.rows;++row){std::vector<float> kv,score,latent;int64_t first=-1;
                for(int64_t d=0;d<dim;++d){kv.push_back(chunk->buffer(comp::projected_kv).getf(row*dim+d));if(ratio==2)score.push_back(chunk->buffer(comp::gate_scores).getf(row*dim+d));}
                REQUIRE_EQ(reference.push(kv,score,&latent,&first),RAD_OK);CHECK_EQ(latent.empty()?0:1,c.buffers[f::emit].id(row));
                if(!latent.empty()){CHECK_EQ(first,c.buffers[f::first_positions].id(row));for(int64_t d=0;d<dim;++d)CHECK_EQ(c.buffers[f::unrotated].get(row*dim+d),latent[size_t(d)]);}
            }
        }
    }
}
TEST(ds41_compress_finish_inactive_rows_are_positive_zero_active_prefix_retains_signed_zero){
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}){
        fix::Fixture c(dtype,dtype,7,1);for(int64_t d=0;d<c.dim;++d)c.buffers[f::gain].set(d,1.f);
        for(int64_t row=0;row<c.rows;++row)for(int64_t d=0;d<c.dim;++d)c.buffers[f::pooled].set(row*c.dim+d,-0.f);
        c.buffers[f::emit].id(2,0);c.buffers[f::first_positions].id(2,INT32_MIN);for(int64_t d=0;d<c.dim;++d)c.buffers[f::pooled].set(2*c.dim+d,NAN);compare(c);
        for(int output:{f::unrotated,f::rotated}){for(int64_t d=0;d<c.dim;++d){const float z=c.buffers[size_t(output)].get(2*c.dim+d);CHECK_EQ(z,0.f);CHECK(!std::signbit(z));}
            CHECK(std::signbit(c.buffers[size_t(output)].get(0)));}
    }
}
TEST(ds41_compress_finish_inactive_pooled_and_position_pages_are_never_dereferenced){
    const size_t page=size_t(sysconf(_SC_PAGESIZE));REQUIRE(page>=4096);
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}){
        fix::Fixture c(dtype,dtype,7,2,0,5);const auto expected=c.expected();Pages pooled(size_t(c.rows)*page),positions(size_t(c.rows)*page);
        const int width=rad_dtype_bits(dtype)/8;
        for(int64_t row=0;row<c.rows;++row){if(c.buffers[f::emit].id(row)){
                for(int64_t d=0;d<c.dim;++d)rad_store_f32(pooled.ptr+size_t(row)*page,dtype,d,c.buffers[f::pooled].get(row*c.dim+d));
                const auto pos=c.buffers[f::first_positions].id(row);std::memcpy(positions.ptr+size_t(row)*page,&pos,4);
            }else{pooled.protect(size_t(row)*page,page);positions.protect(size_t(row)*page,page);}}
        auto args=c.args();c.tensors[f::pooled].data=pooled.ptr;c.tensors[f::pooled].stride[0]=int64_t(page/size_t(width));c.tensors[f::first_positions].data=positions.ptr;c.tensors[f::first_positions].stride[0]=int64_t(page/4);
        REQUIRE_EQ(f::emulate(&args),RAD_OK);for(int output:{f::unrotated,f::rotated})CHECK(c.buffers[size_t(output)].words==expected[size_t(output)].words);
    }
    fix::Fixture all;for(int64_t row=0;row<all.rows;++row)all.buffers[f::emit].id(row,0);Pages none(page);none.protect(0,page);auto args=all.args();all.tensors[f::pooled].data=none.ptr;all.tensors[f::first_positions].data=none.ptr;
    REQUIRE_EQ(f::emulate(&args),RAD_OK);for(int output:{f::unrotated,f::rotated})for(int64_t at=0;at<all.rows*all.dim;++at){CHECK_EQ(all.buffers[size_t(output)].get(at),0.f);CHECK(!std::signbit(all.buffers[size_t(output)].get(at)));}
}
TEST(ds41_compress_finish_active_payload_and_derived_overflow_failures_are_whole_batch_atomic){
    for(int fault=0;fault<14;++fault){
        fix::Fixture c(RAD_F32,fault==10||fault==11?RAD_F16:RAD_F32,7,1);const int64_t row=c.rows-1;
        if(fault==0)c.buffers[f::emit].id(row,2);
        if(fault==1)c.buffers[f::emit].id(row,-1);
        if(fault==2)c.buffers[f::first_positions].id(row,-1);
        if(fault==3)c.buffers[f::first_positions].id(row,int32_t(c.positions));
        if(fault==4)c.buffers[f::first_positions].id(row,INT32_MAX);
        if(fault==5)c.buffers[f::pooled].set(row*c.dim+c.dim-1,NAN);
        if(fault==6)c.buffers[f::pooled].set(row*c.dim+c.dim-1,INFINITY);
        if(fault==7)c.buffers[f::gain].set(c.dim-1,NAN);
        if(fault==8)c.buffers[f::pairs].set(c.buffers[f::first_positions].id(row)*c.rotary,2.f);
        if(fault==9)c.buffers[f::pairs].set(c.buffers[f::first_positions].id(row)*c.rotary+1,NAN);
        if(fault==10)c.buffers[f::pooled].set(row*c.dim,1e10f); // finite source -> F16 input overflow
        if(fault==11)c.buffers[f::gain].set(c.dim-1,1e10f); // finite norm -> F16 output overflow
        if(fault==12)c.buffers[f::pooled].set(row*c.dim,1e30f); // F32 square/sum overflow
        if(fault==13)c.buffers[f::gain].set(c.dim-1,std::numeric_limits<float>::infinity());
        reject(c);
    }
    fix::Fixture odd(RAD_F32,RAD_BF16,7,2);odd.buffers[f::first_positions].id(5,3);reject(odd);
    fix::Fixture rotated(RAD_F32,RAD_F16,2,1,0,1);for(int64_t d=0;d<2;++d){rotated.buffers[f::pooled].set(d,1.f);rotated.buffers[f::gain].set(d,60000.f);}
    rotated.buffers[f::pairs].set(0,std::sqrt(0.5f));rotated.buffers[f::pairs].set(1,std::sqrt(0.5f));reject(rotated); // finite unrotated, overflowing rotated F16
    fix::Fixture unused;unused.buffers[f::pairs].set(63*unused.rotary,NAN);compare(unused);
    fix::Fixture last;last.buffers[f::first_positions].id(5,62);compare(last); // largest aligned table position
    fix::Fixture all_inactive;for(int64_t row=0;row<all_inactive.rows;++row)all_inactive.buffers[f::emit].id(row,0);all_inactive.buffers[f::gain].set(0,NAN);reject(all_inactive);
}
TEST(ds41_compress_finish_descriptor_scratch_and_alias_failures_precede_any_payload_or_output_write){
    for(int fault=0;fault<20;++fault){
        fix::Fixture c;auto args=c.args();const auto before=c.buffers;const auto flag=c.flag;
        if(fault==0)c.integer("M",0);
        if(fault==1)c.integer("n",1025);
        if(fault==2)c.integer("rotary_dim",3);
        if(fault==3)c.integer("rotary_dim",c.dim+2);
        if(fault==4)c.integer("ratio",3);
        if(fault==5)c.floating("eps",0);
        if(fault==6)c.floating("eps",NAN);
        if(fault==7)c.text("a_dtype","i32");
        if(fault==8)c.tensors[f::unrotated].dtype=RAD_U32;
        if(fault==9)c.tensors[f::pooled].rank=1;
        if(fault==10)c.tensors[f::rotated].data=nullptr;
        if(fault==11)c.tensors[f::pooled].stride[0]=INT64_MAX;
        if(fault==12)c.tensors[f::gain].stride[0]=-1;
        if(fault==13)c.tensors[f::pooled].data=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==14)c.tensors[f::pooled].data=static_cast<uint8_t*>(c.tensors[f::pooled].data)+1;
        if(fault==15)args.scratch_bytes=3;
        if(fault==16)args.scratch=nullptr;
        if(fault==17)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==18)args.scratch=c.tensors[f::pooled].data;
        if(fault==19)c.tensors[f::pairs].shape[0]=f::max_positions+1;
        f::Plan p;p.rows=777;CHECK(f::validate(&args,&p)<0);CHECK_EQ(p.rows,777);CHECK(f::emulate(&args)<0);CHECK(c.flag==flag);for(int i=0;i<f::operand_count;++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    }
    for(int output:{f::unrotated,f::rotated})for(int other=0;other<f::operand_count;++other)if(other!=output){
        fix::Fixture c;auto args=c.args();const auto before=c.buffers;const auto flag=c.flag;c.tensors[size_t(output)].data=c.tensors[size_t(other)].data;f::Plan p;
        CHECK(f::validate(&args,&p)<0);CHECK(f::emulate(&args)<0);CHECK(c.flag==flag);for(int i=0;i<f::operand_count;++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    }
}
TEST(ds41_compress_finish_native_reserved_span_and_host_private_flag_contract_are_explicit){
    fix::Fixture c;auto args=c.args();f::Plan p;REQUIRE_EQ(f::validate(&args,&p),RAD_OK);CHECK_EQ(args.scratch_bytes,4);CHECK_EQ(ds41_compress_finish_scratch(nullptr),rad::ds41::gpu_scratch_bytes<f::Plan>());
    CHECK_EQ(rad::ds41::validate_plan_storage(&args,p),RAD_E_INVAL);
    // Keep the deliberately aliased output extent inside a real owned allocation; the
    // refusal must come from native-reserve overlap, not an accidental neighboring span.
    std::vector<std::max_align_t> allocation(size_t(ds41_compress_finish_scratch(nullptr)+p.spans[f::rotated])/sizeof(std::max_align_t)+2);args.scratch=allocation.data();args.scratch_bytes=ds41_compress_finish_scratch(nullptr);
    REQUIRE_EQ(f::validate(&args,&p),RAD_OK);CHECK_EQ(rad::ds41::validate_plan_storage(&args,p),RAD_OK);--args.scratch_bytes;CHECK(rad::ds41::validate_plan_storage(&args,p)<0);++args.scratch_bytes;
    c.tensors[f::rotated].data=static_cast<uint8_t*>(args.scratch)+rad::ds41::device_plan_offset;REQUIRE_EQ(f::validate(&args,&p),RAD_OK);CHECK_EQ(rad::ds41::validate_plan_storage(&args,p),RAD_E_INVAL);
    auto expected=c.expected();args=c.args();const auto flag=c.flag;Pages inaccessible(size_t(sysconf(_SC_PAGESIZE)));inaccessible.protect(0,inaccessible.bytes);args.scratch=inaccessible.ptr;args.scratch_bytes=INT64_MAX;
    REQUIRE_EQ(ds41_ref_compress_finish(&args,nullptr),RAD_OK);CHECK(c.flag==flag);for(int output:{f::unrotated,f::rotated})CHECK(c.buffers[size_t(output)].words==expected[size_t(output)].words);
    fix::Fixture bad;bad.buffers[f::pooled].set(5*bad.dim,NAN);auto bad_args=bad.args();const auto before_bad=bad.buffers;const auto bad_flag=bad.flag;bad_args.scratch=inaccessible.ptr;bad_args.scratch_bytes=INT64_MAX;
    CHECK(ds41_ref_compress_finish(&bad_args,nullptr)<0);CHECK(bad.flag==bad_flag);for(int i=0;i<f::operand_count;++i)CHECK(bad.buffers[size_t(i)].words==before_bad[size_t(i)].words);
    fix::Fixture no_scratch;auto no_scratch_args=no_scratch.args();no_scratch_args.scratch=nullptr;no_scratch_args.scratch_bytes=0;CHECK_EQ(ds41_ref_compress_finish(&no_scratch_args,nullptr),RAD_OK);
    c.params[0]=RAD_RANGE("M",1,17);RadOpdDesc d{};REQUIRE_EQ(f::shape(c.params.data(),int(c.params.size()),f::unrotated,&d),RAD_OK);CHECK_EQ(d.shape[0],17);CHECK_EQ(d.shape[1],c.dim);
    d.shape[0]=998;CHECK_EQ(f::shape(c.params.data(),int(c.params.size()),f::pairs,&d),RAD_E_UNSUPPORTED);CHECK_EQ(d.shape[0],998);
}
RAD_TEST_MAIN()

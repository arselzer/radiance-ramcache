/* CPU proof of opaque latent copy contracts, using the separately checked core-compatible
 * state mapper as the address oracle. Raw F32/F16/BF16 payloads are copied bit for bit. */
#include "rad_test.h"
#include "rad_plugin.h"
#include "../libds41/ds41_latent_args.h"
#include <array>
#include <random>
#include <vector>
namespace ls=rad::ds41::latent;
namespace st=rad::deepseek41::state;
namespace {
RadTensor mat(void* data,uint32_t dtype,int64_t rows,int64_t cols,int64_t pitch) {
    RadTensor t{};t.data=data;t.dtype=dtype;t.rank=2;t.shape[0]=rows;t.shape[1]=cols;
    t.stride[0]=pitch;t.stride[1]=1;return t;
}
RadTensor vec(void* data,uint32_t dtype,int64_t n) {
    RadTensor t{};t.data=data;t.dtype=dtype;t.rank=1;t.shape[0]=n;t.stride[0]=1;return t;
}
struct Case {
    int64_t d,r,b,blocks=3,rows=4,pitch;uint32_t dtype;st::Layout layout;
    std::vector<uint8_t> input,cache,output;
    std::vector<int64_t> first,last,pos;
    std::vector<int32_t> status;
    std::array<RadParam,6> params;
    Case(uint32_t dt,int64_t dim,int64_t ratio,int64_t block):d(dim),r(ratio),b(block),pitch(d+3),dtype(dt) {
        if(st::make_layout(d,r,b,rad_dtype_bits(dtype)/8,&layout)!=RAD_OK)throw std::runtime_error("test layout failed");
        input.resize(size_t(rows*pitch*layout.element_bytes));cache.resize(size_t(blocks*layout.page_bytes),0xC7);
        output.resize(input.size(),0xA1);std::mt19937 rng(unsigned(d+r+b+dt));for(auto& value:input)value=uint8_t(rng());
        first={0,b-r,b,2*b+r};
        for(auto slot:first) {last.push_back(slot+r-1);pos.push_back(100*b+slot%b+r-1);}
        status.assign(size_t(rows),0x12345678);
        params={RAD_INT("M",rows+1),RAD_INT("dimension",d),RAD_INT("ratio",r),RAD_INT("block_size",b),
                RAD_INT("n_blocks",blocks),RAD_STR("dtype",rad_dtype_name(dtype))};
    }
    RadTensor tensor_cache() {
        RadTensor t{};t.data=cache.data();t.dtype=dtype;t.rank=4;t.shape[0]=blocks;t.shape[1]=1;t.shape[2]=b;t.shape[3]=d/r;
        t.stride[0]=b*d/r;t.stride[1]=b*d/r;t.stride[2]=d/r;t.stride[3]=1;return t;
    }
    std::array<RadTensor,5> store_tensors() {
        return {mat(input.data(),dtype,rows,d,pitch),vec(pos.data(),RAD_I64,rows),vec(last.data(),RAD_I64,rows),
                tensor_cache(),vec(status.data(),RAD_I32,rows)};
    }
    std::array<RadTensor,4> gather_tensors() {
        return {tensor_cache(),vec(first.data(),RAD_I64,rows),mat(output.data(),dtype,rows,d,pitch),vec(status.data(),RAD_I32,rows)};
    }
    RadArgs args(const RadTensor* tensors,int count) {
        RadArgs a{};a.t=tensors;a.n_t=count;a.p=params.data();a.n_p=params.size();return a;
    }
    std::vector<uint8_t> expected_cache() const {
        std::vector<uint8_t> result(cache.size(),0xC7);
        for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<d;++col) {
            // Entry index within this physical page, using checked mapping independent of
            // the host copy's flattened original-slot offset.
            const int64_t page=first[size_t(row)]/b,entry=(first[size_t(row)]%b)/r;
            int64_t offset=0;if(st::physical_byte(layout,page,entry,col,&offset)!=RAD_OK)throw std::runtime_error("map failed");
            std::memcpy(result.data()+offset,input.data()+(row*pitch+col)*layout.element_bytes,size_t(layout.element_bytes));
        }
        return result;
    }
};
}
TEST(ds41_latent_host_store_and_gather_preserve_every_payload_bit_and_padding) {
    for(auto dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int64_t d:{128,512})for(int64_t ratio:{1,2})for(int64_t block:{4,8,16}) {
        Case c(dtype,d,ratio,block);const auto original=c.input;auto tensors=c.store_tensors();auto args=c.args(tensors.data(),tensors.size());
        REQUIRE_EQ(ds41_ref_latent_store(&args,nullptr),RAD_OK);CHECK(c.cache==c.expected_cache());CHECK(c.input==original);
        for(auto status:c.status)CHECK_EQ(status,RAD_OK);
        auto gather=c.gather_tensors();args=c.args(gather.data(),gather.size());
        REQUIRE_EQ(ds41_ref_latent_gather(&args,nullptr),RAD_OK);
        for(int64_t row=0;row<c.rows;++row) {
            const size_t start=size_t(row*c.pitch*c.layout.element_bytes),n=size_t(c.layout.entry_bytes);
            CHECK(std::equal(c.output.begin()+start,c.output.begin()+start+n,c.input.begin()+start));
            CHECK(std::all_of(c.output.begin()+start+n,c.output.begin()+size_t((row+1)*c.pitch*c.layout.element_bytes),[](uint8_t x){return x==0xA1;}));
        }
    }
}
TEST(ds41_latent_store_rejects_incomplete_c2_bad_slots_and_duplicates_before_writes) {
    for(int kind=0;kind<4;++kind) {
        Case c(RAD_BF16,512,2,4);
        if(kind==0)--c.pos.back();
        if(kind==1)c.last.back()=c.blocks*c.b;
        if(kind==2) {c.last.back()=c.last[0];c.pos.back()=c.pos[0];}
        if(kind==3)c.pos.back()=-1;
        const auto before=c.cache;const auto status=c.status;auto tensors=c.store_tensors();auto args=c.args(tensors.data(),tensors.size());
        CHECK(ds41_ref_latent_store(&args,nullptr)<0);CHECK(c.cache==before);CHECK(c.status==status);
    }
}
TEST(ds41_latent_gather_rejects_unaligned_out_of_range_or_negative_slots_atomically) {
    for(int kind=0;kind<3;++kind) {
        Case c(RAD_F32,128,2,8);
        c.first.back()=kind==0?-1:kind==1?c.blocks*c.b:1;
        const auto before=c.output,status_bytes=c.cache;const auto status=c.status;
        auto tensors=c.gather_tensors();auto args=c.args(tensors.data(),tensors.size());
        CHECK(ds41_ref_latent_gather(&args,nullptr)<0);CHECK(c.output==before);CHECK(c.cache==status_bytes);CHECK(c.status==status);
    }
}
TEST(ds41_latent_metadata_rejects_bad_cache_shapes_strides_aliases_and_capacities) {
    Case c(RAD_F32,512,2,4);auto good=c.store_tensors();const auto before=c.cache;
    for(int kind=0;kind<8;++kind) {
        auto tensors=good;auto args=c.args(tensors.data(),tensors.size());
        if(kind==0)tensors[3].rank=3;
        if(kind==1)++tensors[3].shape[3];
        if(kind==2)++tensors[3].stride[0];
        if(kind==3)tensors[3].dtype=RAD_U8;
        if(kind==4)tensors[0].data=tensors[3].data;
        if(kind==5)tensors[4].data=tensors[1].data;
        if(kind==6)args.n_t=4;
        if(kind==7)tensors[0].data=nullptr;
        CHECK(ds41_ref_latent_store(&args,nullptr)<0);CHECK(c.cache==before);
    }
    auto params=c.params;params[0]=RAD_INT("M",1);auto args=c.args(good.data(),good.size());args.p=params.data();
    CHECK_EQ(ds41_ref_latent_store(&args,nullptr),RAD_E_SHAPE);
    params=c.params;params[3]=RAD_INT("block_size",3);args.p=params.data();CHECK_EQ(ds41_ref_latent_store(&args,nullptr),RAD_E_SHAPE);
}
RAD_TEST_MAIN()

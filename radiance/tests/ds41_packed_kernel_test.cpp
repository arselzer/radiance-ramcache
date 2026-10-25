/* CPU launch and ABI contract checks for the experimental native packed kernels. Numeric
 * expectations use the real importer and generic canonical decoder, never packed::decode. */
#include "quant_fixture.h"
#include "../libds41/ds41_packed_args.h"
#include <array>
#include <bit>
#include <random>

namespace pk=rad::ds41::packed;
namespace {
struct Format {pk::Type type;const char* name;int values,bytes;};
constexpr Format formats[]{
    {pk::Type::iq2_xxs,"iq2_xxs",256,66},{pk::Type::iq2_xs,"iq2_xs",256,74},
    {pk::Type::iq2_s,"iq2_s",256,82},{pk::Type::iq3_xxs,"iq3_xxs",256,98},
    {pk::Type::q5_k,"q5_K",256,176},{pk::Type::q8_0,"q8_0",32,34}};
void put16(uint8_t* p,uint16_t v) {p[0]=uint8_t(v);p[1]=uint8_t(v>>8);}
struct Imported {
    Format format;int64_t rows,cols;RadEncoding encoding{};
    std::vector<uint8_t> raw;std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> planes;
    std::vector<float> decoded;
    Imported(Format f,int64_t r,int64_t c,uint32_t seed):format(f),rows(r),cols(c) {
        raw.resize(size_t(rows*(cols/f.values)*f.bytes));std::mt19937 rng(seed);
        for(auto& b:raw)b=uint8_t(rng());
        for(size_t at=0;at<raw.size();at+=size_t(f.bytes)) {
            put16(raw.data()+at,uint16_t(0x2000+(rng()%32)*0x20));
            if(f.type==pk::Type::q5_k)put16(raw.data()+at+2,0x1800);
        }
        const auto* entry=qfix::reg().checkpoint_importer("gguf",f.name);
        if(!entry)throw std::runtime_error(std::string("missing GGUF importer ")+f.name);
        RadCheckpointImportWeight weight{"packed-kernel-oracle",f.name,rows,cols};
        if(entry->info->encoding(&weight,&encoding)!=RAD_OK)throw std::runtime_error("encoding failed");
        void* out[RAD_ENC_MAX_PLANES]{};const void* in[RAD_ENC_MAX_PLANES]{};
        for(int p=0;p<encoding.n_planes;++p) {
            planes[p].resize(size_t(rad_enc_plane_bytes(&encoding.plane[p],rows,cols)));
            out[p]=planes[p].data();in[p]=out[p];
        }
        if(entry->info->import_rows(&weight,raw.data(),raw.size(),rows,out)!=RAD_OK)
            throw std::runtime_error("import failed");
        rad::EncodedView view;rad::enc_view(encoding,rows,cols,in,&view);decoded.resize(size_t(rows*cols));
        if(rad::enc_decode_rows(view,0,rows,decoded.data())!=RAD_OK)throw std::runtime_error("canonical decode failed");
    }
    std::vector<RadTensor> tensors(bool payload=true) {
        std::vector<RadTensor> result;
        for(int p=0;p<encoding.n_planes;++p) {
            int64_t r=0,c=0;rad_enc_plane_dims(&encoding.plane[p],rows,cols,&r,&c);
            RadTensor t{};t.data=payload?planes[p].data():nullptr;t.dtype=encoding.plane[p].dtype;t.rank=2;
            t.shape[0]=r;t.shape[1]=c;t.stride[0]=c;t.stride[1]=1;result.push_back(t);
        }
        return result;
    }
};
struct Storage {
    std::vector<uint32_t> words;
    explicit Storage(size_t bytes):words((bytes+3)/4,0xA5A5A5A5u) {}
    void* data() {return words.data();}
    float load(uint32_t dtype,size_t i) const {
        const auto* p=reinterpret_cast<const uint8_t*>(words.data());
        if(dtype==RAD_F32) {float value;std::memcpy(&value,p+i*4,4);return value;}
        uint16_t h;std::memcpy(&h,p+i*2,2);return dtype==RAD_F16?rad_f16_to_f32(h):rad_bf16_to_f32(h);
    }
    void store(uint32_t dtype,size_t i,float value) {
        auto* p=reinterpret_cast<uint8_t*>(words.data());
        if(dtype==RAD_F32)std::memcpy(p+i*4,&value,4);
        else {const uint16_t h=dtype==RAD_F16?rad_f32_to_f16(value):rad_f32_to_bf16(value);std::memcpy(p+i*2,&h,2);}
    }
};
RadTensor matrix(void* p,uint32_t dtype,int64_t rows,int64_t cols,int64_t pitch) {
    RadTensor t{};t.data=p;t.dtype=dtype;t.rank=2;t.shape[0]=rows;t.shape[1]=cols;t.stride[0]=pitch;t.stride[1]=1;return t;
}
RadTensor vector(void* p,int64_t count) {
    RadTensor t{};t.data=p;t.dtype=RAD_I32;t.rank=1;t.shape[0]=count;t.stride[0]=1;return t;
}
struct Fixture {
    Format f;bool routed;int64_t input_rows=3,jobs,n=4,k,m=5,capacity=8,apitch,ypitch;
    uint32_t ad,yd;Storage a,y;std::vector<Imported> experts;
    std::vector<const uint8_t*> pointers;std::vector<int32_t> ids{2,0,2,1,0},rows{1,0,1,2,0},status;
    std::vector<RadParam> params;std::vector<RadTensor> tensors;
    Fixture(Format format,bool route,uint32_t a_dtype=RAD_F32,uint32_t y_dtype=RAD_F32,int64_t width=768,bool pitched=true)
        :f(format),routed(route),jobs(route?5:input_rows),k(width),apitch(k+(pitched?7:0)),ypitch(n+(pitched?5:0)),
         ad(a_dtype),yd(y_dtype),a(size_t(input_rows*apitch*4+32)),
         y(size_t(std::max(jobs*ypitch*4+32,n*(k/f.values)*f.bytes+32))),status(size_t(jobs),-777) {
        for(int64_t r=0;r<input_rows;++r)for(int64_t c=0;c<k;++c)
            a.store(ad,size_t(r*apitch+c),float((r*43+c*17)%131-65)/64.0f);
        for(int i=0;i<(routed?4:1);++i)experts.emplace_back(f,n,k,51017+i);
        for(auto& e:experts)pointers.push_back(e.raw.data());
        params={RAD_INT("M",m),RAD_INT("N",n),RAD_INT("K",k),RAD_STR("format",f.name),
                RAD_STR("a_dtype",rad_dtype_name(ad)),RAD_STR("y_dtype",rad_dtype_name(yd))};
        tensors={matrix(a.data(),ad,input_rows,k,apitch)};
        if(routed) {
            params.push_back(RAD_INT("R",capacity));params.push_back(RAD_INT("n_expert",pointers.size()));
            RadTensor wt{};wt.data=pointers.data();wt.dtype=RAD_U8;wt.rank=3;wt.shape[0]=pointers.size();
            wt.shape[1]=n;wt.shape[2]=nrow();wt.stride[0]=0;wt.stride[1]=nrow();wt.stride[2]=1;
            tensors.push_back(wt);tensors.push_back(vector(ids.data(),jobs));tensors.push_back(vector(rows.data(),jobs));
            tensors.push_back(matrix(y.data(),yd,jobs,n,ypitch));tensors.push_back(vector(status.data(),jobs));
        } else {
            tensors.push_back(matrix(experts[0].raw.data(),RAD_U8,n,nrow(),nrow()));
            tensors.push_back(matrix(y.data(),yd,jobs,n,ypitch));
        }
    }
    int64_t nrow() const {return k/f.values*f.bytes;}
    RadArgs args() const {RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());return a;}
    int run() {const auto arg=args();return routed?ds41_ref_packed_routed(&arg,nullptr):ds41_ref_packed_gemm(&arg,nullptr);}
    void set_int(const char* key,int64_t value) {for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    void set_str(const char* key,const char* value) {for(auto& p:params)if(!std::strcmp(p.key,key))p.sval=value;}
    Storage expected() const {
        Storage out=y;
        for(int64_t j=0;j<jobs;++j)for(int64_t row=0;row<n;++row) {
            const int e=routed?ids[size_t(j)]:0;const int64_t token=routed?rows[size_t(j)]:j;float sum=0;
            for(int64_t c=0;c<k;++c) {
                // Explicitly separate multiplication/addition like the host reference, with
                // canonical decoded weights and ABI activation conversions as the oracle.
                const volatile float product=a.load(ad,size_t(token*apitch+c))*experts[size_t(e)].decoded[size_t(row*k+c)];
                sum=sum+product;
            }
            out.store(yd,size_t(j*ypitch+row),sum);
        }
        return out;
    }
};
void rejects(Fixture& f) {
    const auto a=f.a.words,y=f.y.words;const auto ids=f.ids,rows=f.rows,status=f.status;const auto pointers=f.pointers;
    std::vector<std::vector<uint8_t>> weights;for(const auto& e:f.experts)weights.push_back(e.raw);
    CHECK(f.run()<0);CHECK(f.a.words==a);CHECK(f.y.words==y);CHECK(f.ids==ids);CHECK(f.rows==rows);
    CHECK(f.status==status);CHECK(f.pointers==pointers);
    for(size_t i=0;i<weights.size();++i)CHECK(f.experts[i].raw==weights[i]);
}
}

TEST(ds41_packed_host_dense_all_formats_and_scalar_dtype_pairs_match_canonical_matvec) {
    for(const auto f:formats)for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16}) {
        Fixture test(f,false,ad,yd);const auto expected=test.expected();const auto before=test.a.words;
        REQUIRE_EQ(test.run(),RAD_OK);CHECK(test.y.words==expected.words);CHECK(test.a.words==before);
        const auto arg=test.args();pk::Launch l;REQUIRE_EQ(pk::validate(&arg,false,&l),RAD_OK);
        CHECK_EQ(l.jobs,3);CHECK_EQ(l.g.m,5);CHECK_EQ(l.a_pitch,test.apitch);CHECK_EQ(l.y_pitch,test.ypitch);
    }
}
TEST(ds41_packed_host_routed_preserves_route_order_repeated_rows_and_unused_null_experts) {
    for(const auto f:formats)for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16}) {
        Fixture test(f,true,ad,yd);test.pointers[3]=nullptr;const auto expected=test.expected();
        REQUIRE_EQ(test.run(),RAD_OK);CHECK(test.y.words==expected.words);
        for(const auto status:test.status)CHECK_EQ(status,RAD_OK);
        for(int64_t n=0;n<test.n;++n)CHECK_EQ(test.y.load(yd,size_t(n)),test.y.load(yd,size_t(2*test.ypitch+n)));
    }
}
TEST(ds41_packed_opaque_runtime_weight_descriptors_keep_logical_shapes_and_native_byte_extents) {
    for(const auto f:formats)for(bool routed:{false,true}) {
        Fixture test(f,routed);const auto expected=test.expected();auto& w=test.tensors[1];
        w.dtype=pk::storage_dtype(f.type);
        CHECK_EQ(rad_dtype_bits(w.dtype),0);
        if(routed) {w.shape[2]=test.k;w.stride[1]=test.k;}
        else {w.shape[1]=test.k;w.stride[0]=test.k;}
        REQUIRE_EQ(test.run(),RAD_OK);CHECK(test.y.words==expected.words);
        const auto before=test.y.words;w.dtype=RAD_U8;
        CHECK_EQ(test.run(),RAD_E_SHAPE);CHECK(test.y.words==before);
        w.dtype=pk::storage_dtype(pk::Type((unsigned(f.type)+1)%6));
        CHECK_EQ(test.run(),RAD_E_DTYPE);CHECK(test.y.words==before);
    }
}
TEST(ds41_packed_host_contiguous_shapes_and_minimum_q8_block_width_work) {
    for(const auto f:formats) {
        Fixture test(f,false,RAD_F32,RAD_F32,f.values,false);const auto expected=test.expected();
        REQUIRE_EQ(test.run(),RAD_OK);CHECK(test.y.words==expected.words);
    }
}
TEST(ds41_packed_host_validates_every_route_before_writing_output_or_status) {
    for(const auto f:formats)for(int fault=0;fault<6;++fault) {
        Fixture test(f,true);
        if(fault==0)test.ids.back()=int32_t(test.pointers.size());
        if(fault==1)test.ids.back()=-1;
        if(fault==2)test.rows.back()=int32_t(test.input_rows);
        if(fault==3)test.rows.back()=-1;
        if(fault==4) {test.ids.back()=3;test.pointers[3]=nullptr;}
        if(fault==5) {test.ids.back()=3;test.pointers[3]=static_cast<const uint8_t*>(test.y.data());}
        rejects(test);
    }
}
TEST(ds41_packed_host_rejects_malformed_geometry_tensor_shapes_dtypes_pitches_and_caps) {
    for(bool route:{false,true}) {
        Fixture test(formats[1],route);const auto good=test.tensors;
        for(size_t p=0;p<good.size();++p) {
            test.tensors=good;test.tensors[p].data=nullptr;rejects(test);
            test.tensors=good;test.tensors[p].dtype=RAD_U16;rejects(test);
            test.tensors=good;test.tensors[p].rank=0;rejects(test);
        }
        for(int fault=0;fault<10;++fault) {
            test.tensors=good;
            if(fault==0)test.tensors[0].shape[0]=test.m+1;
            if(fault==1)test.tensors[0].shape[0]=0;
            if(fault==2)--test.tensors[0].shape[1];
            if(fault==3)test.tensors[0].stride[0]=test.k-1;
            if(fault==4)test.tensors[0].stride[1]=2;
            if(fault==5)test.tensors[0].stride[0]=INT64_MAX;
            if(fault==6)++test.tensors[route?4:2].shape[0];
            if(fault==7)test.tensors[route?4:2].stride[0]=test.n-1;
            if(fault==8)test.tensors[1].stride[0]=route?1:test.nrow()+1;
            if(fault==9)--test.tensors[1].shape[route?2:1];
            rejects(test);
        }
        test.tensors=good;
        for(const char* key:{"M","N","K"}) {
            const auto params=test.params;test.set_int(key,0);rejects(test);test.params=params;
            test.set_int(key,INT64_MAX);rejects(test);test.params=params;
        }
        const auto params=test.params;test.set_int("K",257);rejects(test);test.params=params;
        test.set_str("format","iq2_xs_typo");rejects(test);test.params=params;
        test.set_str("a_dtype","i32");rejects(test);test.params=params;
        test.set_str("y_dtype","f8e4m3");rejects(test);test.params=params;
        if(route) {
            test.set_int("R",test.jobs-1);rejects(test);test.params=params;
            test.tensors[2].shape[0]=test.capacity+1;rejects(test);test.tensors=good;
            test.tensors[2].stride[0]=2;rejects(test);test.tensors=good;
            test.tensors[1].stride[1]++;rejects(test);test.tensors=good;
        }
        auto args=test.args();pk::Launch l;
        CHECK(pk::validate(nullptr,route,&l)<0);CHECK(pk::validate(&args,route,nullptr)<0);
        args.n_t--;CHECK((route?ds41_ref_packed_routed(&args,nullptr):ds41_ref_packed_gemm(&args,nullptr))<0);
        args=test.args();args.p=nullptr;CHECK(pk::validate(&args,route,&l)<0);
    }
}
TEST(ds41_packed_host_refuses_output_status_and_selected_expert_aliases_and_bad_alignment) {
    for(bool route:{false,true}) {
        Fixture test(formats[1],route);const auto good=test.tensors;const int yi=route?4:2;
        test.tensors[yi].data=test.a.data();rejects(test);test.tensors=good;
        test.tensors[yi].data=test.experts[0].raw.data();rejects(test);test.tensors=good;
        test.tensors[0].data=static_cast<uint8_t*>(test.a.data())+1;rejects(test);test.tensors=good;
        test.tensors[yi].data=static_cast<uint8_t*>(test.y.data())+1;rejects(test);test.tensors=good;
        if(route) {
            test.tensors[5].data=test.ids.data();rejects(test);test.tensors=good;
            test.tensors[5].data=test.experts[0].raw.data();rejects(test);test.tensors=good;
            test.tensors[1].data=reinterpret_cast<uint8_t*>(test.pointers.data())+1;rejects(test);
        }
    }
}
TEST(ds41_packed_layout_roundtrip_selection_and_rank_local_geometry_match_canonical_planes) {
    const RadLayoutFn layout=&pk::layout;const RadRelayoutFn relayout=&pk::relayout;const RadUnrelayoutFn inverse=&pk::unrelayout;
    for(const auto f:formats) {
        Fixture test(f,false);auto& source=test.experts[0];auto planes=source.tensors(false);
        std::vector<int> selection;for(int i=0;i<source.encoding.n_planes;++i)selection.push_back(i);
        RadLayout out{};
        REQUIRE_EQ(layout(test.params.data(),test.params.size(),1,&source.encoding,selection.data(),planes.data(),planes.size(),&out),RAD_OK);
        CHECK_EQ(out.dtype,pk::storage_dtype(f.type));CHECK_EQ(out.rank,2u);CHECK_EQ(out.align,256);
        CHECK_EQ(std::string(out.tag),std::string(pk::runtime_tag(f.type)));
        CHECK_EQ(out.shape[0],test.n);CHECK_EQ(out.shape[1],test.nrow());CHECK_EQ(out.bytes,int64_t(source.raw.size()));
        CHECK(std::string(out.tag)!=std::string(rad::gguf::geometry(pk::gguf_type(f.type))->tag));
        planes=source.tensors();std::vector<uint8_t> packed(source.raw.size()+32,0x79);
        REQUIRE_EQ(relayout(test.params.data(),test.params.size(),1,&source.encoding,selection.data(),planes.data(),planes.size(),packed.data()+8,source.raw.size()+16),RAD_OK);
        CHECK(std::equal(source.raw.begin(),source.raw.end(),packed.begin()+8));
        CHECK(std::all_of(packed.begin(),packed.begin()+8,[](uint8_t b){return b==0x79;}));
        CHECK(std::all_of(packed.begin()+8+source.raw.size(),packed.end(),[](uint8_t b){return b==0x79;}));
        Imported restored(f,test.n,test.k,999);auto destinations=restored.tensors();
        REQUIRE_EQ(inverse(test.params.data(),test.params.size(),1,&source.encoding,selection.data(),packed.data()+8,source.raw.size(),destinations.data(),destinations.size()),RAD_OK);
        CHECK(restored.planes==source.planes);
        for(int fault=0;fault<7;++fault) {
            auto bad_selection=selection;auto bad_planes=planes;auto params=test.params;int operand=1;
            if(fault==0)operand=0;
            if(fault==1)bad_selection[0]=1;
            if(fault==2)bad_planes.pop_back();
            if(fault==3)bad_planes[0].stride[0]++;
            if(fault==4)params[1].ival++;
            if(fault==5)params[3].sval="q5_k"; // Native GGUF name is case-sensitive q5_K.
            if(fault==6)bad_planes[0].dtype=RAD_U16;
            std::vector<uint8_t> dst(source.raw.size(),0x6D);
            CHECK(layout(params.data(),params.size(),operand,&source.encoding,bad_selection.data(),bad_planes.data(),bad_planes.size(),&out)<0);
            CHECK(relayout(params.data(),params.size(),operand,&source.encoding,bad_selection.data(),bad_planes.data(),bad_planes.size(),dst.data(),dst.size())<0);
            CHECK(std::all_of(dst.begin(),dst.end(),[](uint8_t b){return b==0x6D;}));
        }
        std::vector<uint8_t> dst(source.raw.size(),0x6D);
        CHECK(relayout(test.params.data(),test.params.size(),1,&source.encoding,selection.data(),planes.data(),planes.size(),dst.data(),dst.size()-1)<0);
        CHECK(std::all_of(dst.begin(),dst.end(),[](uint8_t b){return b==0x6D;}));
    }
}
TEST(ds41_packed_shape_hooks_describe_capacity_and_refuse_a_fictitious_float_weight) {
    for(bool route:{false,true}) {
        Fixture test(formats[1],route);test.params[0]=RAD_RANGE("M",1,test.m);RadOpdDesc out{};
        for(int operand=0;operand<(route?6:3);++operand) {
            const int st=pk::shape(test.params.data(),test.params.size(),operand,&out,route);
            if(operand==1) {CHECK_EQ(st,RAD_E_UNSUPPORTED);continue;}
            REQUIRE_EQ(st,RAD_OK);
            if(operand==0) {CHECK_EQ(out.dtype,test.ad);CHECK_EQ(out.shape[0],test.m);CHECK_EQ(out.shape[1],test.k);}
            else if(operand==(route?4:2)) {CHECK_EQ(out.dtype,test.yd);CHECK_EQ(out.shape[0],route?test.capacity:test.m);CHECK_EQ(out.shape[1],test.n);}
            else {CHECK_EQ(out.dtype,uint32_t(RAD_I32));CHECK_EQ(out.shape[0],test.capacity);}
            if(route&&(operand==2||operand==3)) {
                CHECK_EQ(out.fill,RAD_FILL_INDEX);CHECK_EQ(out.idx_max,operand==2?int64_t(test.pointers.size()):test.m);
            }
        }
        CHECK(pk::shape(test.params.data(),test.params.size(),-1,&out,route)<0);
        CHECK(pk::shape(test.params.data(),test.params.size(),route?6:3,&out,route)<0);
        CHECK(pk::shape(test.params.data(),test.params.size(),0,nullptr,route)<0);
    }
}
TEST(ds41_packed_scalar_storage_helpers_match_abi_rounding_and_all_half_bit_patterns) {
    for(uint32_t h=0;h<65536;++h) {
        CHECK_EQ(std::bit_cast<uint32_t>(pk::half(h)),std::bit_cast<uint32_t>(rad_f16_to_f32(uint16_t(h))));
        for(uint32_t tail:{0x7FFFu,0x8000u,0x8001u}) {
            const float value=std::bit_cast<float>((h<<16)|tail);
            CHECK_EQ(pk::to_bf16(value),rad_f32_to_bf16(value));CHECK_EQ(pk::to_half(value),rad_f32_to_f16(value));
        }
    }
    std::mt19937 rng(411017);
    for(int i=0;i<100000;++i) {
        const float value=std::bit_cast<float>(uint32_t(rng()));
        CHECK_EQ(pk::to_half(value),rad_f32_to_f16(value));CHECK_EQ(pk::to_bf16(value),rad_f32_to_bf16(value));
    }
}
RAD_TEST_MAIN()

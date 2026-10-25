/* Actual Engine loader widening, using an existing supported tiny Qwen graph on HOST.
 * No serve()/HTTP/request, replacement kernel, manual converted payload or new architecture. */
#include "rad_test.h"
#include "tiny_model.h"
#include "device/device.h"
#include "engine.h"
#include <thread>

using namespace rad;

TEST(ds41_plain_loader_engine_start_exactly_widens_bf16_norm_payload){
    if(!device_is_host())SKIP("requires the HOST backend; this test never starts a GPU Engine");
    REQUIRE_EQ(rad_dev_count(),1);
    RadDeviceProps props{};REQUIRE_EQ(rad_dev_props(0,&props),RAD_OK);
    REQUIRE_EQ(props.vram_bytes,128ll*1024*1024);
    tiny::TempDir directory("plain-loader");REQUIRE(directory.ok());
    tiny::Qwen35 shape;shape.layers=2;shape.ctx=64;
    REQUIRE(tiny::write_qwen35(directory.path,shape));
    const auto checkpoint=directory/"model.safetensors";
    const auto checkpoint_bytes=std::filesystem::file_size(checkpoint);
    REQUIRE(checkpoint_bytes<=512u*1024u);
    const auto tensors=tiny::read_safetensors(checkpoint);
    uint64_t aligned_stage_bound=0;
    for(const auto& tensor:tensors)aligned_stage_bound+=(uint64_t(tensor.bytes.size())+4095u)/4096u*4096u;
    // This pinned tiny graph consumes each source once (fusing views can reduce page padding).
    // Include four whole-file copies plus every independently page-rounded selected source.
    // Driver/allocator metadata and host thread stacks are separate runtime overhead.
    constexpr uint64_t mib=1024u*1024u;
    const uint64_t source_and_stage_bound=4u*checkpoint_bytes+aligned_stage_bound;
    REQUIRE(source_and_stage_bound<=2u*mib);
    const unsigned hardware=std::thread::hardware_concurrency();
    const unsigned workers=std::max(2u,std::min(16u,hardware>2?hardware-2:2u));
    REQUIRE(workers<=16u);
    const uint64_t bounce_bound=uint64_t(workers)*8u*4u*mib;
    constexpr uint64_t device_pool_bound=128u*mib;
    REQUIRE(device_pool_bound+bounce_bound<=640u*mib);
    std::fprintf(stderr,"    loader bound: synthetic device/pools <=128MiB; %u workers' lazy bounce <=%lluMiB; source/staged <=%llu bytes; separate anonymous VA reserve4GiB; excludes runtime/allocator/thread-stack overhead\n",
                 workers,static_cast<unsigned long long>(bounce_bound/mib),static_cast<unsigned long long>(source_and_stage_bound));
    const tiny::Tensor* source=nullptr;
    for(const auto& tensor:tensors)if(tensor.name=="model.language_model.norm.weight")source=&tensor;
    REQUIRE(source!=nullptr);REQUIRE_EQ(source->dtype,std::string("BF16"));
    REQUIRE_EQ(source->shape.size(),size_t(1));REQUIRE_EQ(source->shape[0],shape.hidden);
    REQUIRE_EQ(source->bytes.size(),size_t(shape.hidden*2));
    const auto original_checkpoint=tiny::read_text(checkpoint);

    Config config;config.model=directory.path;config.radiance_home=home();config.kernel_hierarchy={"libref"};
    config.accept_reference_kernels=true;config.tp=1;config.max_tok=64;config.max_seqs=1;config.max_ctx=64;
    config.host_pool_mib=0;config.host_headroom_mib=256;config.gpu_headroom_mib=16;
    config.placement="all_vram";config.embedding_placement=TablePlacement::Vram;config.embedding_placement_given=true;
    config.prefix_cache=false;config.prefix_cache_mib=0;config.checkpoint_slots=0;config.n_spec=0;
    config.deterministic=true;config.debug_graph=false;config.profile_ops=false;
    {
        // The ordinary public start path includes declare, plan, real source reads, private
        // StagedWeight::run, upload/drain and Ctx binding. debug_graph would stop before load.
        Engine engine(std::move(config));REQUIRE_EQ(engine.start(),RAD_OK);
        const auto& program=engine.program();const WeightInfo* loaded=nullptr;
        for(const auto& weight:program.weights)if(weight.name=="output_norm.weight"){
            REQUIRE(loaded==nullptr);loaded=&weight;
        }
        REQUIRE(loaded!=nullptr);CHECK(loaded->enc_known);CHECK(loaded->identity);REQUIRE(loaded->widen);
        REQUIRE_EQ(loaded->enc.n_planes,1);REQUIRE_EQ(loaded->enc.plane[0].dtype,uint32_t(RAD_BF16));
        REQUIRE_EQ(loaded->n_sel,1);REQUIRE_EQ(loaded->decl.dtype,uint32_t(RAD_F32));
        REQUIRE_EQ(loaded->stored_dtype,uint32_t(RAD_F32));REQUIRE_EQ(loaded->decl.rank,uint32_t(1));
        REQUIRE_EQ(loaded->decl.shape[0],shape.hidden);REQUIRE_EQ(loaded->stored_bytes,shape.hidden*4);
        CHECK_EQ(loaded->layout_tag,std::string());REQUIRE(loaded->ptr!=nullptr);
        // HOST device memory is addressable here. Read the actual Engine-owned destination,
        // never source.data rebound as F32 and never a test-created widened weight buffer.
        const auto* bytes=static_cast<const uint8_t*>(loaded->ptr);
        for(int64_t i=0;i<shape.hidden;++i){
            uint16_t bf16_bits=0;std::memcpy(&bf16_bits,source->bytes.data()+size_t(i)*2,2);
            const uint32_t expected_bits=uint32_t(bf16_bits)<<16;
            uint32_t actual_bits=0;std::memcpy(&actual_bits,bytes+size_t(i)*4,4);
            CHECK_EQ(actual_bits,expected_bits);
        }
        CHECK_EQ(tiny::read_text(checkpoint),original_checkpoint);
        // Destruction occurs with the source directory and its expected bytes still alive.
        // Engine drains transfer streams, then Ctx streams, before releasing pools/plugins.
    }
    CHECK_EQ(tiny::read_text(checkpoint),original_checkpoint);
}

int main(int argc,char** argv){
    // Before the first explicit backend query: the normal CPU test environment already hides
    // cards, but this standalone binary also fixes its own bounded synthetic HOST capacity.
    // A HIP-enabled executable may perform linked registration before main; this is not a
    // claim that those constructors are suppressed. No GPU Engine can pass the test guard.
    if(setenv("ROCR_VISIBLE_DEVICES","99",1)||setenv("RADIANCE_HOST_DEVICES","1",1)||
       setenv("RADIANCE_HOST_VRAM_MIB","128",1)){
        std::fprintf(stderr,"unable to set isolated HOST loader test environment\n");return 2;
    }
    return radtest::run_all(argc,argv);
}

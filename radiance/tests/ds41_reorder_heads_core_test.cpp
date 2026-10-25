/* Real HOST plugin/Builder/Ctx binding proof for maxM>actualM. No launch replacement. */
#include "rad_test.h"
#include "ds41_reorder_heads_core_fixture.h"
namespace c=ds41_reorder_core;
TEST(ds41_reorder_heads_core_host_flat_row_slices_vary_M_without_changing_rank_order_or_bits) {
    if(!rad::device_is_host())SKIP("HOST-only actual buffer binding proof");
    rad::Registry registry;c::okay(registry.load_plugin(home()+"/experimental/kernels/libds41_ref.so",0),"load actual host DS41 rows");c::Stream stream;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int tp:{1,2,4,8}) {
        c::Harness flat(registry,stream.value,dtype,tp,true,false);
        for(int rows:{1,3,5})flat.pass(rows,rows+int(dtype));
        CHECK_EQ(flat.ctx.pass_counts().played,0);CHECK(!flat.ctx.checks_device_status());
    }
}
TEST(ds41_reorder_heads_core_host_original_rank_major_profile_preserves_public_operand_hazards) {
    if(!rad::device_is_host())SKIP("HOST-only actual buffer binding proof");
    rad::Registry registry;c::okay(registry.load_plugin(home()+"/experimental/kernels/libds41_ref.so",0),"load actual host DS41 rows");c::Stream stream;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int tp:{1,2,4,8}) {
        c::Harness original(registry,stream.value,dtype,tp,false,false);original.pass(c::fixed_rows,0);original.pass(c::fixed_rows,1);
        CHECK_EQ(original.ctx.pass_counts().played,0);CHECK(!original.ctx.checks_device_status());
    }
}
RAD_TEST_MAIN()

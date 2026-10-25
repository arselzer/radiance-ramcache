/* Reject unrelated malformed metadata before generic transform validation interprets it. */
#include "rad_test.h"
#include "../arch/common/rad_gguf_relayout.h"

namespace gg=rad::gguf;
namespace {
RadEncoding unrelated_transform() {
    auto enc=gg::canonical_encoding(gg::Format::iq2_xs);
    rad_enc_copy_str(enc.transform,"givens");
    rad_enc_add_table(&enc,"t.pairs",RAD_I16,1,8);
    rad_enc_add_table(&enc,"t.angle",RAD_F32,1,INT64_MAX);
    // Generic givens validity multiplies this extent by2. A native IQ adapter has no reason to
    // interpret it at all: this encoding already differs from its exact canonical contract.
    return enc;
}
}
TEST(gguf_repack_planner_declines_unrelated_overflowing_transform_atomically) {
    auto encoding=unrelated_transform();gg::Layout out{};out.bytes=1234;
    std::string why;
    CHECK_EQ(gg::plan(gg::Format::iq2_xs,encoding,1,256,{0,1,0,256},&out,&why),RAD_E_DTYPE);
    CHECK_EQ(out.bytes,1234);CHECK(!why.empty());
}
TEST(gguf_relayout_hooks_decline_unrelated_overflowing_transform_before_geometry) {
    auto encoding=unrelated_transform();RadTensor planes[RAD_ENC_MAX_PLANES]{};
    int selection[RAD_ENC_MAX_PLANES]{};
    for(int p=0;p<RAD_ENC_MAX_PLANES;++p)selection[p]=p;
    RadLayout out{};out.bytes=5678;
    CHECK_EQ(gg::abi_layout<gg::Format::iq2_xs>(nullptr,0,0,&encoding,selection,planes,
                                              encoding.n_planes,&out),RAD_E_DTYPE);
    CHECK_EQ(out.bytes,5678);
}
RAD_TEST_MAIN()

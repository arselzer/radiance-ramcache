/* Exact actual-checkpoint view/closed-owner regression. No file is opened,
 * no model weights exist, and no source capability is admitted by this test. */
#include "rad_test.h"
#include "format/checkpoint.h"
#include <array>

namespace {
void closed_owner_control(bool row_range) {
    rad::Checkpoint ck;
    rad::GgufFile closed;
    std::array<int64_t,48> storage{};
    rad::SrcTensor raw;
    raw.name="engram_offsets.weight";raw.ggml=27;raw.rad_dtype=RAD_I64;
    raw.shape={2,24};raw.data=storage.data();raw.bytes=sizeof(storage);
    raw.supplemental_gguf=&closed;raw.supplemental_tensor=raw.name;
    const auto* view=row_range?ck.rows(raw,1,1,"control"):ck.slice(raw,1,"control");
    REQUIRE(view);
    CHECK(view->supplemental_gguf==&closed);
    CHECK_EQ(view->supplemental_tensor,raw.name);
    CHECK_EQ(view->bytes,int64_t(24*sizeof(int64_t)));
    rad::CkptWeight w;
    w.name="control";w.trivial=true;w.rank=2;w.shape[0]=1;w.shape[1]=24;
    w.rows=1;w.cols=24;w.enc=rad_enc_plain(RAD_I64);
    w.pieces.push_back(view);w.scales.push_back(nullptr);
    std::array<uint64_t,24> out;out.fill(0x123456789abcdef0ULL);
    const auto before=out;
    std::string why;
    CHECK(rad::ckpt_plane_rows(w,0,0,1,out.data(),&why)<0);
    CHECK(out==before);
    CHECK(why.find("supplemental")!=std::string::npos);
    std::array<float,24> f;f.fill(123.0f);const auto fb=f;
    CHECK(rad::ckpt_rows_f32(w,0,1,f.data(),&why)<0);
    CHECK(f==fb);
}
}
TEST(sliced_control_retains_closed_owner_and_refuses_before_output) {closed_owner_control(false);}
TEST(row_control_retains_closed_owner_and_refuses_before_output) {closed_owner_control(true);}
RAD_TEST_MAIN()

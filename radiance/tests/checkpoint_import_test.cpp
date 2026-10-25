/* Lossless GGUF import through real checkpoint maps and the libquant plugin. The golden
 * synthetic blocks were decoded independently by GGML, not by the importer under test.
 * tests/data/golden/gguf-import.json records the reference commit and generator seed. */
#include "quant_fixture.h"
#include "format/checkpoint.h"
#include "format/radfile.h"
#include "tiny_model.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unistd.h>

using namespace rad;
namespace {
using Json = nlohmann::json;
Json fixtures() {
    std::ifstream file(std::string(RAD_GOLDEN_FIXTURES) + "/gguf-import.json");
    if (!file) throw std::runtime_error("missing GGUF import golden fixture");
    return Json::parse(file)["formats"];
}
std::vector<uint8_t> unhex(const std::string& text) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i < text.size(); i += 2)
        out.push_back((uint8_t)std::stoul(text.substr(i, 2), nullptr, 16));
    return out;
}
std::vector<float> expected(const Json& f) {
    const auto bytes = unhex(f.at("f32_hex"));
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}
struct Tensor { std::string name; uint32_t type; std::vector<int64_t> shape; std::vector<uint8_t> data; };
std::string write_gguf(const std::vector<Tensor>& tensors) {
    std::string path = radtest::tmp_path("gguf_import.XXXXXX.gguf");
    const int fd = mkstemps(path.data(), 5);
    if (fd < 0) throw std::runtime_error("mkstemp");
    close(fd);
    std::vector<uint8_t> bytes;
    auto put = [&]<class T>(T v) { auto* p = (const uint8_t*)&v; bytes.insert(bytes.end(), p, p + sizeof(v)); };
    auto string = [&](const std::string& s) { put((uint64_t)s.size()); bytes.insert(bytes.end(), s.begin(), s.end()); };
    auto pad = [&] { bytes.resize((bytes.size() + 31) / 32 * 32); };
    put(uint32_t(0x46554747)); put(uint32_t(3)); put(uint64_t(tensors.size())); put(uint64_t(1));
    string("general.architecture"); put(uint32_t(GGUF_STR)); string("radtest");
    uint64_t offset = 0;
    for (const auto& t : tensors) {
        string(t.name); put(uint32_t(t.shape.size()));
        for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it) put(uint64_t(*it));
        put(t.type); put(offset); offset += (t.data.size() + 31) / 32 * 32;
    }
    pad();
    for (const auto& t : tensors) { bytes.insert(bytes.end(), t.data.begin(), t.data.end()); pad(); }
    std::ofstream file(path, std::ios::binary); file.write((const char*)bytes.data(), bytes.size());
    return path;
}
Tensor tensor(const Json& f, const char* name = "packed", bool stack = false) {
    return {name, f.at("type"), stack ? std::vector<int64_t>{2, 2, 256} : std::vector<int64_t>{4, 256},
            unhex(f.at("packed_hex"))};
}
RadNameMap map(const char* declared, int mode, std::initializer_list<const char*> src,
                int index = -1, int dim = 0) {
    RadNameMap m{}; m.declared = declared; m.mode = mode; m.concat_dim = dim;
    for (int& i : m.src_index) i = -1;
    for (const char* s : src) { m.src[m.n_src] = s; m.src_index[m.n_src++] = index; }
    return m;
}
void compare(const std::vector<float>& got, const std::vector<float>& want, size_t offset = 0) {
    REQUIRE(got.size() + offset <= want.size());
    for (size_t i = 0; i < got.size(); ++i) CHECK_EQ(got[i], want[offset + i]);
}
Json iq2() {
    for (const auto& f : fixtures()) if (f.at("name") == "iq2_xs") return f;
    throw std::runtime_error("no iq2_xs fixture");
}
}

TEST(lossless_gguf_all_checkpoint_formats_match_independent_decoder) {
    auto& reg = qfix::reg();
    REQUIRE_EQ(reg.checkpoint_importers().size(), 19u);
    for (const auto& f : fixtures()) {
        const std::string type = f.at("name");
        const auto* row = reg.checkpoint_importer("gguf", type); REQUIRE(row);
        CHECK_EQ(row->info->block_values, f.at("block_values").get<int64_t>());
        CHECK_EQ(row->info->block_bytes, f.at("block_bytes").get<int64_t>());
        Checkpoint ck; REQUIRE_EQ(ck.open(write_gguf({tensor(f)})), RAD_OK); ck.set_importers(&reg);
        CHECK_EQ(ck.imports(), Checkpoint::Import::GgufBlocks);
        Program prog; prog.name_map.push_back(map("logical", RAD_MAP_COPY, {"packed"}));
        CkptWeight w; std::string why;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "logical", false, &w, &why), RAD_OK);
        REQUIRE(w.imported); REQUIRE(w.block_importer); CHECK(!w.trivial);
        const auto want = expected(f);
        std::vector<float> all(4 * 256), band(2 * 256);
        REQUIRE_EQ(ckpt_rows_f32(w, 0, 4, all.data(), &why), RAD_OK); compare(all, want);
        REQUIRE_EQ(ckpt_rows_f32(w, 1, 2, band.data(), &why), RAD_OK); compare(band, want, 256);
        std::vector<std::vector<uint8_t>> planes;
        REQUIRE_EQ(ckpt_import_band(w, 0, 4, &planes, &why), RAD_OK);
        for (int p = 0; p < w.enc.n_planes; ++p) {
            int64_t pr = 0, pc = 0; rad_enc_plane_dims(&w.enc.plane[p], w.rows, w.cols, &pr, &pc);
            std::vector<uint8_t> own(planes[p].size());
            REQUIRE_EQ(ckpt_plane_rows(w, p, 0, pr, own.data(), &why), RAD_OK); CHECK(own == planes[p]);
        }
    }
}
TEST(checkpoint_importers_preserve_legacy_quantizer_compatibility_and_cleanup) {
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(::home()+"/quantizers/libquant.so",0), RAD_OK);
    const auto n = reg.quantizers().size();
    REQUIRE_EQ(reg.load_plugin(::home()+"/testquant/radtest_quant.so",0), RAD_OK);
    CHECK_EQ(reg.quantizers().size(),n+1);CHECK_EQ(reg.checkpoint_importers().size(),19u);
    reg.close();CHECK(reg.quantizers().empty());CHECK(reg.checkpoint_importers().empty());
}
TEST(checkpoint_importers_reject_incomplete_malformed_and_duplicate_interfaces_transactionally) {
    for (int which=1;which<=5;++which) {
        Registry reg;REQUIRE_EQ(reg.load_plugin(::home()+"/quantizers/libquant.so",0), RAD_OK);
        const auto nq=reg.quantizers().size(),np=reg.plugins().size(),ni=reg.checkpoint_importers().size();
        const auto path=::home()+"/testquant/import/radtest_importbad_"+std::to_string(which)+".so";
        CHECK(reg.load_plugin(path,0)<0);
        CHECK_EQ(reg.quantizers().size(),nq);CHECK_EQ(reg.plugins().size(),np);
        CHECK_EQ(reg.checkpoint_importers().size(),ni);
    }
}
TEST(lossless_gguf_is_opt_in_for_conversion_and_does_not_enable_raw_serving) {
    Checkpoint ck; REQUIRE_EQ(ck.open(write_gguf({tensor(iq2())})), RAD_OK);
    Program prog; prog.name_map.push_back(map("logical", RAD_MAP_COPY, {"packed"}));
    CkptWeight w; std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "logical", false, &w, &why), RAD_OK);
    CHECK(!w.trivial); CHECK(!w.imported); CHECK(!w.block_importer);
}
TEST(lossless_gguf_does_not_alias_private_type_ids_to_standard_formats) {
    auto& reg=qfix::reg();CHECK(reg.checkpoint_importer("gguf","rad4_q4")==nullptr);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(write_gguf({{"private",43,{4,256},std::vector<uint8_t>(4*8*20)}})),RAD_OK);
    ck.set_importers(&reg);Program prog;prog.name_map.push_back(map("logical",RAD_MAP_COPY,{"private"}));
    CkptWeight w;std::string why;REQUIRE_EQ(ckpt_resolve(ck,prog,"logical",false,&w,&why),RAD_OK);
    CHECK(!w.imported);CHECK(!w.trivial);CHECK(!w.block_importer);
    std::vector<float> out(1024);CHECK(ckpt_rows_f32(w,0,4,out.data(),&why)<0);
}
TEST(lossless_gguf_expert_slice_reads_only_the_selected_expert) {
    const auto f = iq2(); Checkpoint ck;
    REQUIRE_EQ(ck.open(write_gguf({tensor(f, "experts", true)})), RAD_OK); ck.set_importers(&qfix::reg());
    Program prog; prog.name_map.push_back(map("expert1", RAD_MAP_COPY, {"experts"}, 1));
    CkptWeight w; std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "expert1", false, &w, &why), RAD_OK); REQUIRE(w.imported);
    CHECK_EQ(w.rows, 2); CHECK_EQ(w.cols, 256);
    std::vector<float> got(512); REQUIRE_EQ(ckpt_rows_f32(w, 0, 2, got.data(), &why), RAD_OK);
    compare(got, expected(f), 512);
}
TEST(lossless_gguf_row_join_band_crosses_source_boundary) {
    const auto f = iq2(); auto a = tensor(f, "a"), b = tensor(f, "b");
    const size_t half = a.data.size() / 2;
    a.data.resize(half); b.data.erase(b.data.begin(), b.data.begin() + half);
    a.shape = b.shape = {2, 256};
    Checkpoint ck; REQUIRE_EQ(ck.open(write_gguf({a,b})), RAD_OK); ck.set_importers(&qfix::reg());
    Program prog; prog.name_map.push_back(map("joined", RAD_MAP_CONCAT, {"a","b"}));
    CkptWeight w; std::string why; REQUIRE_EQ(ckpt_resolve(ck, prog, "joined", false, &w, &why), RAD_OK);
    REQUIRE(w.imported); std::vector<float> got(3 * 256);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 3, got.data(), &why), RAD_OK); compare(got, expected(f), 256);
}
TEST(lossless_gguf_refuses_column_joins_and_mixed_types) {
    const auto f = iq2(); auto a = tensor(f, "a"), b = tensor(f, "b");
    Checkpoint ck; REQUIRE_EQ(ck.open(write_gguf({a,b})), RAD_OK); ck.set_importers(&qfix::reg());
    Program prog; prog.name_map.push_back(map("columns", RAD_MAP_CONCAT, {"a","b"}, -1, 1));
    CkptWeight w; std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "columns", false, &w, &why), RAD_OK); CHECK(!w.imported);
    auto different = tensor(fixtures().at(0), "different");
    Checkpoint mixed; REQUIRE_EQ(mixed.open(write_gguf({a,different})), RAD_OK); mixed.set_importers(&qfix::reg());
    prog.name_map.clear(); prog.name_map.push_back(map("mixed", RAD_MAP_CONCAT, {"a","different"}));
    REQUIRE_EQ(ckpt_resolve(mixed, prog, "mixed", false, &w, &why), RAD_OK); CHECK(!w.imported);
}
TEST(lossless_gguf_import_checks_ranges_and_initializes_constants_without_payload) {
    const auto f = iq2(); const auto packed = unhex(f.at("packed_hex"));
    const auto* imp = qfix::reg().checkpoint_importer("gguf", "iq2_xs"); REQUIRE(imp);
    RadCheckpointImportWeight iw{"test", "iq2_xs", 4, 256}; RadEncoding enc{};
    REQUIRE_EQ(imp->info->encoding(&iw, &enc), RAD_OK);
    std::vector<std::vector<uint8_t>> bytes; void* ptr[RAD_ENC_MAX_PLANES]{};
    for (int p = 0; p < enc.n_planes; ++p) {
        bytes.emplace_back(rad_enc_plane_bytes(&enc.plane[p], 4, 256), 0); ptr[p] = bytes.back().data();
    }
    CHECK_EQ(imp->info->import_rows(&iw, packed.data(), packed.size() - 1, 4, ptr), RAD_E_SHAPE);
    CHECK_EQ(imp->info->import_rows(&iw, nullptr, packed.size(), 4, ptr), RAD_E_SHAPE);
    CHECK_EQ(imp->info->import_rows(&iw, packed.data(), packed.size(), 5, ptr), RAD_E_SHAPE);
    CHECK_OK(imp->info->import_rows(&iw, nullptr, 0, 0, ptr));
    const int grid = rad_enc_find(&enc, "grid"), scale = rad_enc_find(&enc, "scale.2");
    REQUIRE(grid >= 0 && scale >= 0);
    float factor = 0; std::memcpy(&factor, ptr[scale], 4); CHECK_EQ(factor, 0.125f);
    CHECK(std::any_of(bytes[grid].begin(), bytes[grid].end(), [](uint8_t b) { return b != 0; }));
    iw.cols = 255; CHECK_EQ(imp->info->encoding(&iw, &enc), RAD_E_SHAPE);
    iw.cols = 256; iw.rows = INT64_MAX;
    CHECK_EQ(imp->info->encoding(&iw, &enc), RAD_E_SHAPE);
    Checkpoint ck; REQUIRE_EQ(ck.open(write_gguf({tensor(f)})), RAD_OK); ck.set_importers(&qfix::reg());
    Program prog; prog.name_map.push_back(map("logical", RAD_MAP_COPY, {"packed"})); CkptWeight w;
    std::string why; REQUIRE_EQ(ckpt_resolve(ck, prog, "logical", false, &w, &why), RAD_OK);
    std::vector<std::vector<uint8_t>> planes;
    CHECK_EQ(ckpt_import_band(w, 4, 1, &planes, &why), RAD_E_SHAPE);
    CHECK_EQ(ckpt_import_band(w, INT64_MAX, 1, &planes, &why), RAD_E_SHAPE);
    CHECK_OK(ckpt_import_band(w, 0, 0, &planes, &why));
}
TEST(lossless_gguf_zero_row_import_handles_huge_representable_plane_geometry_without_overflow) {
    const auto* imp = qfix::reg().checkpoint_importer("gguf", "iq2_xs"); REQUIRE(imp);
    RadCheckpointImportWeight iw{"huge", "iq2_xs", 1, INT64_MAX / 256 * 256};
    RadEncoding enc{}; REQUIRE_EQ(imp->info->encoding(&iw, &enc), RAD_OK);
    // No large allocation or payload: only the fixed codebook and scalar are requested.
    std::vector<std::vector<uint8_t>> data((size_t)enc.n_planes);
    void* ptr[RAD_ENC_MAX_PLANES]{};
    for(int p=0;p<enc.n_planes;++p) {
        const auto& plane=enc.plane[p];
        if(plane.kind==RAD_PLANE_TILED&&plane.block[0]==1)continue;
        data[(size_t)p].assign((size_t)rad_enc_plane_bytes(&plane,1,256),0x5a);
        ptr[p]=data[(size_t)p].data();
    }
    REQUIRE_EQ(imp->info->import_rows(&iw,nullptr,0,0,ptr),RAD_OK);
    const auto huge_shared=data;
    iw.cols=256;
    for(auto& p:data)std::fill(p.begin(),p.end(),0);
    REQUIRE_EQ(imp->info->import_rows(&iw,nullptr,0,0,ptr),RAD_OK);
    CHECK(data==huge_shared);
    // Every refusal must leave even the earlier codebook unchanged.
    for(auto& p:data)std::fill(p.begin(),p.end(),0xa5);
    const auto before=data;
    iw.cols=INT64_MAX / 256 * 256;iw.rows=2;
    CHECK_EQ(imp->info->import_rows(&iw,nullptr,0,0,ptr),RAD_E_SHAPE);
    CHECK(data==before);
    iw.rows=1;
    CHECK_EQ(imp->info->import_rows(&iw,nullptr,1,0,ptr),RAD_E_SHAPE);
    CHECK(data==before);
    const int factor=rad_enc_find(&enc,"scale.2");REQUIRE(factor>=0);
    ptr[factor]=nullptr;
    CHECK_EQ(imp->info->import_rows(&iw,nullptr,0,0,ptr),RAD_E_INVAL);
    CHECK(data==before);
}
TEST(lossless_gguf_constructed_checkpoint_geometry_is_bounded_before_helpers_or_allocations) {
    const auto* imp=qfix::reg().checkpoint_importer("gguf","iq2_xs");REQUIRE(imp);
    CkptWeight w;w.name="constructed";w.rows=1;w.cols=INT64_MAX/256*256;
    w.imported=true;w.block_importer=imp->info;
    const RadCheckpointImportWeight iw{w.name.c_str(),"iq2_xs",w.rows,w.cols};
    REQUIRE_EQ(imp->info->encoding(&iw,&w.enc),RAD_OK);
    std::vector<std::vector<uint8_t>> out{{0x5a,0xa5}};std::string why;
    REQUIRE_EQ(ckpt_import_band(w,0,0,&out,&why),RAD_OK);
    const int grid=rad_enc_find(&w.enc,"grid");REQUIRE(grid>=0);
    CHECK(out[0].empty());CHECK_EQ(out[(size_t)grid].size(),4096u);
    std::vector<uint8_t> row(8,0x5a);
    REQUIRE_EQ(ckpt_plane_rows(w,grid,511,1,row.data(),&why),RAD_OK);
    CHECK(std::equal(row.begin(),row.end(),out[(size_t)grid].end()-8));
    CHECK_OK(ckpt_plane_rows(w,0,0,0,nullptr,&why));
    const auto before=out;
    auto bad=w;bad.rows=2;
    CHECK_EQ(ckpt_import_band(bad,0,0,&out,&why),RAD_E_SHAPE);CHECK(out==before);
    bad=w;bad.enc.n_planes=INT32_MAX;
    CHECK(ckpt_import_band(bad,0,0,&out,&why)<0);CHECK(out==before);
    CHECK(ckpt_plane_rows(bad,100,0,0,nullptr,&why)<0);CHECK(out==before);
    bad=w;bad.enc.plane[grid].extent[0]=1; // forged smaller codebook must not allocate undersized output
    CHECK(ckpt_import_band(bad,0,0,&out,&why)<0);CHECK(out==before);
    bad=w;bad.enc.plane[grid].extent[0]=INT64_MAX;
    CHECK(ckpt_import_band(bad,0,0,&out,&why)<0);CHECK(out==before);
    CHECK_EQ(ckpt_plane_rows(w,grid,0,1,nullptr,&why),RAD_E_INVAL);
}
TEST(lossless_gguf_import_band_refusals_preserve_existing_outputs_after_an_earlier_source_was_read) {
    const auto f=iq2();Checkpoint ck;REQUIRE_EQ(ck.open(write_gguf({tensor(f)})),RAD_OK);
    ck.set_importers(&qfix::reg());Program prog;
    prog.name_map.push_back(map("logical",RAD_MAP_COPY,{"packed"}));CkptWeight w;std::string why;
    REQUIRE_EQ(ckpt_resolve(ck,prog,"logical",false,&w,&why),RAD_OK);
    SrcTensor first=*w.pieces[0],last=first;
    first.shape={2,256};first.bytes/=2;
    last.shape=first.shape;last.bytes=first.bytes;
    last.data=(const uint8_t*)last.data+first.bytes;
    w.pieces={&first,&last};
    std::vector<std::vector<uint8_t>> output{{1,2,3},{4,5,6}};const auto before=output;
    --last.bytes;
    CHECK_EQ(ckpt_import_band(w,0,4,&output,&why),RAD_E_SHAPE);CHECK(output==before);
    ++last.bytes;last.ggml=43; // a private/colliding numeric type is not the registered IQ2 source
    CHECK_EQ(ckpt_import_band(w,0,4,&output,&why),RAD_E_SHAPE);CHECK(output==before);
    last.ggml=first.ggml;last.data=nullptr;
    CHECK_EQ(ckpt_import_band(w,0,4,&output,&why),RAD_E_SHAPE);CHECK(output==before);
    w.pieces={&first};
    CHECK_EQ(ckpt_import_band(w,0,4,&output,&why),RAD_E_SHAPE);CHECK(output==before);
}
TEST(lossless_gguf_rad_container_roundtrip_keeps_every_plane) {
    const auto f = iq2(); Checkpoint ck; REQUIRE_EQ(ck.open(write_gguf({tensor(f)})), RAD_OK);
    ck.set_importers(&qfix::reg()); Program prog;
    prog.name_map.push_back(map("logical", RAD_MAP_COPY, {"packed"})); CkptWeight src; std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "logical", false, &src, &why), RAD_OK);
    RadWriter writer; const auto path = radtest::tmp_path("lossless-import.rad");
    REQUIRE_EQ(writer.begin(path.c_str()), RAD_OK); writer.set_arch("radtest");
    RadWriter::Weight w; w.name = "logical"; w.enc = src.enc; w.rank = 2; w.shape[0] = 4; w.shape[1] = 256;
    const int64_t index = writer.add_weight(w); REQUIRE(index >= 0);
    for (int64_t row = 0; row < 4; ++row) {
        std::vector<std::vector<uint8_t>> planes; REQUIRE_EQ(ckpt_import_band(src, row, 1, &planes, &why), RAD_OK);
        for (int p = 0; p < src.enc.n_planes; ++p) {
            const auto& plane = src.enc.plane[p]; const bool per_row = plane.kind == RAD_PLANE_TILED && plane.block[0] == 1;
            if (!per_row && row) continue;
            const int64_t offset = per_row ? row * (int64_t)planes[p].size() : 0;
            REQUIRE_EQ(writer.write_plane(index, p, offset, planes[p].data(), planes[p].size()), RAD_OK);
        }
    }
    REQUIRE_EQ(writer.finish(), RAD_OK); RadFile file; REQUIRE_EQ(file.open(path.c_str()), RAD_OK);
    const auto* entry = file.find("logical"); REQUIRE(entry);
    const void* ptr[RAD_ENC_MAX_PLANES]{};
    for (int p = 0; p < src.enc.n_planes; ++p) ptr[p] = file.data(file.plane(*entry, p));
    EncodedView view; enc_view(src.enc, 4, 256, ptr, &view); std::vector<float> got(1024);
    REQUIRE_EQ(enc_decode_rows(view, 0, 4, got.data(), &why), RAD_OK); compare(got, expected(f));
}
TEST(lossless_gguf_converter_writes_published_values_without_a_recipe) {
    tiny::TempDir dir("gguf_cli"); REQUIRE(dir.ok());
    const auto home = dir / "home";
    std::filesystem::create_directories(home + "/architectures");
    std::filesystem::create_directories(home + "/quantizers");
    std::filesystem::create_directories(home + "/kernels");
    std::filesystem::create_symlink(RAD_IMPORT_ARCH, home + "/architectures/import.so");
    std::filesystem::create_symlink(RAD_IMPORT_KERNEL, home + "/kernels/fixture.so");
    std::filesystem::create_symlink(::home() + "/quantizers/libquant.so", home + "/quantizers/libquant.so");
    const auto tok = dir / "tokenizer.json"; REQUIRE(tiny::write_text(tok, tiny::tokenizer_json()));
    for (const auto& f : fixtures()) {
        const auto input = write_gguf({tensor(f)}), output = dir / "output.rad";
        std::filesystem::remove(output);
        std::string log;
        const int status = tiny::run({RAD_IMPORT_CONVERT, input, "-o", output, "--tokenizer", tok},
                         {"RADIANCE_HOME=" + home, "ROCR_VISIBLE_DEVICES=99", "OMP_NUM_THREADS=2",
                          "OMP_WAIT_POLICY=passive"}, &log);
        if (status) std::fprintf(stderr, "%s\n", log.c_str());
        REQUIRE_EQ(status, 0);
        RadFile file; REQUIRE_EQ(file.open(output.c_str()), RAD_OK);
        const auto* entry = file.find("logical"); REQUIRE(entry);
        const void* ptr[RAD_ENC_MAX_PLANES]{};
        for (uint32_t p = 0; p < entry->n_planes; ++p) ptr[p] = file.data(file.plane(*entry, p));
        EncodedView view; enc_view(file.encoding(*entry), 4, 256, ptr, &view);
        std::vector<float> got(1024); REQUIRE_EQ(enc_decode_rows(view, 0, 4, got.data()), RAD_OK);
        compare(got, expected(f));
        CHECK_EQ(std::string(file.str(entry->quantizer)), std::string());
    }
}
RAD_TEST_MAIN()

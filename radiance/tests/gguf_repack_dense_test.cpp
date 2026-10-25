/* Exact dense native blocks -> real importer -> packed bridge roundtrips.
 * Synthetic source blocks and independent GGML golden values; no model/GPU access. */
#include "quant_fixture.h"
#include "../arch/common/rad_gguf_repack.h"
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <random>

namespace gg = rad::gguf;
namespace {
using Json = nlohmann::json;
const Json& fixtures() {
    static const Json data = [] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES) + "/gguf-import.json");
        if (!file) throw std::runtime_error("missing independent GGUF golden fixture");
        return Json::parse(file).at("formats");
    }();
    return data;
}
const Json& fixture(gg::Format format) {
    const auto* geometry = gg::geometry(format);
    if (!geometry) throw std::runtime_error("unknown dense format");
    for (const auto& item : fixtures()) if (item.at("name") == geometry->name) return item;
    throw std::runtime_error("missing dense format fixture");
}
std::vector<uint8_t> unhex(const std::string& input) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < input.size(); i += 2)
        bytes.push_back(uint8_t(std::stoul(input.substr(i, 2), nullptr, 16)));
    return bytes;
}
struct Imported {
    gg::Format format;
    int64_t rows, cols;
    RadEncoding encoding{};
    std::vector<std::vector<uint8_t>> storage;
    Imported(gg::Format f, int64_t r, int64_t c, const std::vector<uint8_t>& packed)
        : format(f), rows(r), cols(c) {
        const auto* geometry = gg::geometry(f);
        const auto* entry = qfix::reg().checkpoint_importer("gguf", geometry->name);
        if (!entry) throw std::runtime_error("missing real GGUF importer");
        RadCheckpointImportWeight weight{"dense-repack-fixture", geometry->name, rows, cols};
        if (entry->info->encoding(&weight, &encoding) != RAD_OK)
            throw std::runtime_error("GGUF import encoding failed");
        void* ptr[RAD_ENC_MAX_PLANES]{};
        for (int p = 0; p < encoding.n_planes; ++p) {
            storage.emplace_back(size_t(rad_enc_plane_bytes(&encoding.plane[p], rows, cols)), 0);
            ptr[p] = storage.back().data();
        }
        if (entry->info->import_rows(&weight, packed.data(), packed.size(), rows, ptr) != RAD_OK)
            throw std::runtime_error("real GGUF import failed");
    }
    gg::View view() const {
        gg::View result;
        result.encoding = encoding; result.rows = rows; result.cols = cols;
        for (int p = 0; p < encoding.n_planes; ++p) result.planes[p].bytes = storage[p];
        return result;
    }
};
std::vector<uint8_t> source_region(gg::Format format, const std::vector<uint8_t>& source,
                                  int64_t cols, gg::Region region) {
    const auto& g = *gg::geometry(format);
    const int64_t source_row = cols / g.values * g.bytes;
    const int64_t row = region.cols / g.values * g.bytes;
    std::vector<uint8_t> result(size_t(region.rows * row));
    for (int64_t r = 0; r < region.rows; ++r) {
        const auto* begin = source.data() + (region.row + r) * source_row + region.col / g.values * g.bytes;
        std::copy(begin, begin + row, result.begin() + r * row);
    }
    return result;
}
void roundtrip(const Imported& input, const std::vector<uint8_t>& packed, gg::Region region) {
    const auto view = input.view();
    gg::Layout layout;
    std::string why;
    REQUIRE_EQ(gg::plan(input.format, view.encoding, view.rows, view.cols, region, &layout, &why), RAD_OK);
    const auto expected = source_region(input.format, packed, input.cols, region);
    REQUIRE_EQ(layout.bytes, int64_t(expected.size()));
    CHECK_EQ(layout.row_bytes, region.cols / gg::geometry(input.format)->values * gg::geometry(input.format)->bytes);
    std::vector<uint8_t> guarded(expected.size() + 32, 0xa5);
    // A larger valid output capacity must not cause writes beyond the planned extent.
    REQUIRE_EQ(gg::repack(input.format, view, region, std::span(guarded).subspan(8, expected.size() + 16), &why), RAD_OK);
    CHECK(std::equal(expected.begin(), expected.end(), guarded.begin() + 8));
    CHECK(std::all_of(guarded.begin(), guarded.begin() + 8, [](uint8_t b) { return b == 0xa5; }));
    CHECK(std::all_of(guarded.begin() + 8 + expected.size(), guarded.end(), [](uint8_t b) { return b == 0xa5; }));
}
constexpr std::array<gg::Format, 2> formats{gg::Format::q5_k, gg::Format::q8_0};
}

TEST(gguf_dense_repack_matches_native_fixture_bytes_and_independent_values) {
    for (auto format : formats) {
        const auto& f = fixture(format);
        const auto source = unhex(f.at("packed_hex"));
        Imported imported(format, 4, 256, source);
        roundtrip(imported, source, {0, 4, 0, 256});
        const auto expected_bytes = unhex(f.at("f32_hex"));
        std::vector<float> expected(expected_bytes.size() / 4), decoded(expected.size());
        std::memcpy(expected.data(), expected_bytes.data(), expected_bytes.size());
        const void* ptr[RAD_ENC_MAX_PLANES]{};
        for (int p = 0; p < imported.encoding.n_planes; ++p) ptr[p] = imported.storage[p].data();
        rad::EncodedView view;
        rad::enc_view(imported.encoding, 4, 256, ptr, &view);
        REQUIRE_EQ(rad::enc_decode_rows(view, 0, 4, decoded.data()), RAD_OK);
        for (size_t i = 0; i < expected.size(); ++i) CHECK_EQ(decoded[i], expected[i]);
    }
}
TEST(gguf_dense_repack_arbitrary_bits_preserve_codes_scales_and_minimums) {
    std::mt19937 random(411017);
    for (auto format : formats) for (int trial = 0; trial < 32; ++trial) {
        const auto& g = *gg::geometry(format);
        constexpr int64_t rows = 3, cols = 5120;
        std::vector<uint8_t> source(size_t(rows * (cols / g.values) * g.bytes));
        for (auto& value : source) value = uint8_t(random());
        Imported imported(format, rows, cols, source);
        roundtrip(imported, source, {0, rows, 0, cols});
        roundtrip(imported, source, {1, 2, g.values, 3 * g.values});
    }
}
TEST(gguf_dense_repack_preserves_every_half_bit_pattern_without_float_conversion) {
    for (auto format : formats) {
        const auto& g = *gg::geometry(format);
        constexpr int64_t rows = 65536;
        std::vector<uint8_t> source(size_t(rows * g.bytes), 0);
        for (int64_t row = 0; row < rows; ++row) {
            const uint16_t half = uint16_t(row);
            source[size_t(row * g.bytes)] = uint8_t(half);
            source[size_t(row * g.bytes + 1)] = uint8_t(half >> 8);
            if (format == gg::Format::q5_k) {
                const uint16_t minimum = uint16_t(~half);
                source[size_t(row * g.bytes + 2)] = uint8_t(minimum);
                source[size_t(row * g.bytes + 3)] = uint8_t(minimum >> 8);
            }
        }
        Imported imported(format, rows, g.values, source);
        roundtrip(imported, source, {0, rows, 0, g.values});
    }
}
TEST(gguf_dense_repack_supports_plane_pitches_and_row_bands) {
    for (auto format : formats) {
        const auto source = unhex(fixture(format).at("packed_hex"));
        Imported imported(format, 4, 256, source);
        auto view = imported.view();
        std::vector<std::vector<uint8_t>> padded;
        for (int p = 0; p < imported.encoding.n_planes; ++p) {
            int64_t rows = 0, cols = 0;
            rad_enc_plane_dims(&imported.encoding.plane[p], 4, 256, &rows, &cols);
            const size_t row_bytes = size_t(rad_enc_row_bytes(imported.encoding.plane[p].dtype, cols));
            const size_t pitch = row_bytes + size_t(p) + 3;
            padded.emplace_back(size_t(rows) * pitch, 0xc3);
            for (int64_t row = 0; row < rows; ++row)
                std::copy_n(imported.storage[p].data() + row * row_bytes, row_bytes, padded.back().data() + row * pitch);
            view.planes[p].bytes = padded.back(); view.planes[p].row_stride = pitch;
        }
        const auto before = padded;
        gg::Region region{1, 2, 0, 256};
        const auto expected = source_region(format, source, 256, region);
        std::vector<uint8_t> out(expected.size(), 0);
        REQUIRE_EQ(gg::repack(format, view, region, out), RAD_OK); CHECK(out == expected); CHECK(padded == before);
    }
}
TEST(gguf_dense_repack_all_rotating_tp_ranges_are_native_block_slices) {
    std::mt19937 random(2304);
    for (auto format : formats) {
        const auto& g = *gg::geometry(format);
        std::vector<uint8_t> source(size_t(3 * (2304 / g.values) * g.bytes));
        for (auto& byte : source) byte = uint8_t(random());
        Imported imported(format, 3, 2304, source);
        for (int layer = 0; layer < 4; ++layer) {
            int64_t col = 0;
            for (int rank = 0; rank < 4; ++rank) {
                const int64_t width = rank == layer ? 768 : 512;
                roundtrip(imported, source, {0, 3, col, width}); col += width;
            }
            CHECK_EQ(col, 2304);
        }
        if (format == gg::Format::q8_0) roundtrip(imported, source, {0, 3, 576, 576});
        else {
            gg::Layout ignored;
            CHECK_EQ(gg::plan(format, imported.encoding, 3, 2304, {0, 3, 576, 576}, &ignored), RAD_E_SHAPE);
        }
    }
}
TEST(gguf_dense_repack_refuses_wrong_group_and_transform_without_writing) {
    for (auto format : formats) {
        const auto source = unhex(fixture(format).at("packed_hex"));
        Imported imported(format, 4, 256, source);
        auto view = imported.view(); std::vector<uint8_t> out(source.size(), 0xa5);
        view.encoding.plane[1].block[1] = 128;
        CHECK(gg::repack(format, view, {0, 4, 0, 256}, out) < 0);
        CHECK(std::all_of(out.begin(), out.end(), [](uint8_t b) { return b == 0xa5; }));
        view = imported.view(); rad_enc_copy_str(view.encoding.transform, "fwht128");
        CHECK(gg::repack(format, view, {0, 4, 0, 256}, out) < 0);
        CHECK(std::all_of(out.begin(), out.end(), [](uint8_t b) { return b == 0xa5; }));
    }
}
TEST(gguf_dense_repack_rejects_short_buffers_invalid_regions_and_aliasing) {
    for (auto format : formats) {
        const auto source = unhex(fixture(format).at("packed_hex"));
        Imported imported(format, 4, 256, source);
        auto view = imported.view(); std::vector<uint8_t> out(source.size(), 0xa5);
        CHECK(gg::repack(format, view, {0, 4, 0, 256}, std::span(out).first(out.size() - 1)) < 0);
        CHECK(std::all_of(out.begin(), out.end(), [](uint8_t b) { return b == 0xa5; }));
        view.planes[0].bytes = view.planes[0].bytes.first(view.planes[0].bytes.size() - 1);
        CHECK(gg::repack(format, view, {0, 4, 0, 256}, out) < 0);
        view = imported.view(); view.planes[0].row_stride = 1;
        CHECK(gg::repack(format, view, {0, 4, 0, 256}, out) < 0);
        view = imported.view();
        for (gg::Region bad : {gg::Region{-1, 1, 0, 256}, {4, 1, 0, 256}, {0, 5, 0, 256},
                               {0, 1, -1, 256}, {0, 1, 0, 257}, {0, 1, INT64_MAX, 256}})
            CHECK(gg::repack(format, view, bad, out) < 0);
        imported.storage[0].resize(source.size() + 64, 0xc3); view = imported.view();
        const auto before = imported.storage[0];
        CHECK(gg::repack(format, view, {0, 4, 0, 256}, std::span(imported.storage[0]).first(source.size())) < 0);
        CHECK(imported.storage[0] == before);
    }
}
RAD_TEST_MAIN()

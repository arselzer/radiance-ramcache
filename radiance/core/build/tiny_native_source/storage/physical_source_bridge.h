#pragma once
#include "native_source.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace glm53::engine_core_physical {
namespace native = rad::engine_core_expert_store;
enum class Status {ok, invalid, identity, checksum, io, exists, unsupported, no_memory};
enum class Part {gate,up,down};
struct Piece {
    Part part=Part::gate;
    std::string tensor,path;
    uint32_t dtype=0;
    uint64_t offset=0,rows=0,row_bytes=0,bytes=0,checksum=0;
    bool complete=false;
};
struct Tensor {
    native::SourceTensor expected;
    int32_t layer=-1,expert=-1;
    std::array<std::vector<Piece>,2> planes;
};
struct Catalog {
    std::string source_id,revision;
    std::string origin="complete_synthetic_physical_tp1_v1";
    std::vector<Tensor> tensors;
};
struct Result {
    native::SourceCatalog sources;
    uint64_t physical_fingerprint=0,manifest_checksum=0,source_bytes=0,output_bytes=0;
    uint64_t source_read_bytes=0;
    size_t physical_pieces=0,window_bytes=0,metadata_reservation=0;
};
// Closed synthetic TP1 geometry:8layers/E4/H128/FF128, main FP4/E8M0 only.
// Complete physical sources are streamed into canonical source files. This is
// NOT a native generation; the unchanged S1 packer/reader still verifies it.
// Old caller output is preserved until one NOREPLACE source-bundle publication.
Status materialize(const rad::Program&,const rad::Plan&,const Catalog&,const std::string& destination,Result*,std::string* why=nullptr);
uint64_t checksum(const std::byte*,size_t)noexcept;
bool same_catalog(const native::SourceCatalog&,const native::SourceCatalog&)noexcept;
} // namespace glm53::engine_core_physical

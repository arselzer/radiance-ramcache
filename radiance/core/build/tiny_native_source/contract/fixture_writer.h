#pragma once
#include "profile.h"
#include <array>
#include <vector>
namespace glm53::engine_core_contract {
enum class Role {ExpertCodes,ExpertScale,Router,Correction,FfnNorm,Embedding,Head,FinalNorm};
struct TensorSpec {
    std::string name,dtype;
    std::vector<int64_t> shape;
    uint64_t bytes=0;Role role=Role::ExpertCodes;
    int layer=-1,expert=-1,projection=-1;
};
std::vector<TensorSpec> tensor_specs();
uint8_t expected_byte(const TensorSpec&,uint64_t at)noexcept;
struct FixtureFiles {std::string directory;uint64_t payload_bytes=0,file_bytes=0;uint32_t artifact_files=0;uint64_t sidecar_bytes=0;};
std::string tokenizer_configuration_json();
// Controlled synthetic checkpoint only. Atomic NOREPLACE; failure preserves out.
// No Engine, Native source/cache, pools, Ctx, Owner or model step is constructed.
int write_fixture(const std::string& destination,FixtureFiles* out,std::string* why=nullptr);
// Bound the four known files and actual safetensors header before Checkpoint::open
// allocates parser/model metadata. Closed synthetic input, not an arbitrary repo.
int preflight_source_files(const std::string& directory,std::string* why=nullptr);
// Actual Core sibling-config parser and loaded Vocab semantics; failure preserves
// caller quote. Preflight closes all auxiliary-file and token-ID allocation paths.
int inspect_vocab_contract(const std::string& directory,VocabQuote* out,std::string* why=nullptr);
int inspect_checkpoint_source(const std::shared_ptr<const SourceCheckpoint>&,const std::string& directory,
    std::unique_ptr<const InspectedSource>* out,std::string* why=nullptr);
} // namespace glm53::engine_core_contract

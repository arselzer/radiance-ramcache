#pragma once
// Closed actual GLM main TP4 startup helper. No serving/default-path activation.
#include "startup.h"
#include "place/planner.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace rad {
class Mover;class Pools;class CachePrepare4RankBinding;struct CkptWeight;
struct GlmMainOrdinaryRankSpec {
    Program* program=nullptr;
    const Plan* plan=nullptr; // final immutable NativeBacked plan
    const CheckpointSources* sources=nullptr; // SAME Engine-owned cache
};
struct GlmMainOrdinaryRuntimeRank {
    Mover* mover=nullptr;
    Pools* pools=nullptr;
    int device=-1;
    RadStream compute=nullptr; // Engine-owned, already bound to Mover
    CachePrepare4RankBinding* binding=nullptr; // admitted startup fail-stop capability
};
struct GlmMainOrdinaryLoaderLimits {
    uint64_t max_metadata_bytes=64u<<20;
    // Limits, not allocations: actual maxima are derived by prepare. The real
    // replicated BF16[154880,4096] embedding alone is1,268,776,960 bytes.
    uint64_t max_input_bytes=2ull<<30,max_output_bytes=2ull<<30;
    uint64_t max_column_bytes=64u<<20,max_buffer_bytes=(4ull<<30)+(64u<<20);
    size_t max_jobs=8192,max_weights_per_rank=100000;
};
struct GlmMainOrdinaryLoaderQuote {
    uint64_t metadata_requested=0,input_bytes=0,output_bytes=0,column_bytes=0,buffer_bytes=0;
    uint64_t pinned_page_bytes=0; // input/output rounded separately to4KiB; buffer_bytes adds column
    uint64_t ordinary_source_bytes=0,ordinary_stored_bytes=0,omitted_source_bytes=0;
    uint64_t source_fill_scratch_upper=0;
    size_t jobs=0,omitted_views=0,optional_missing=0,identity_jobs=0,widen_jobs=0,relayout_jobs=0;
};
struct GlmMainOrdinaryLoaderReceipt {
    GlmMainOrdinaryLoaderQuote quote;
    uint64_t source_bytes=0,upload_bytes=0;
    size_t completed_jobs=0,completed_uploads=0,successful_drains=0;
};
struct GlmMainOrdinarySourceEligibility {
    size_t pieces=0,scales=0;
    uint32_t encoding_planes=0;
    bool canonical_plain=false;
};
class GlmMainOrdinaryLoader final {
public:
    // Same pure source check used by prepare AFTER typed native omission.
    // No payload reads, source mutation, backend calls or allocation of planes.
    // Exact plain encoding has parallel NULL scale placeholders; affine needs
    // a real physical scale per piece. Output assigned only on success.
    static int validate_source(const CkptWeight&,GlmMainOrdinarySourceEligibility*,
                               std::string* why=nullptr)noexcept;
    // Metadata/source rectangles only: no tensor payload reads, backend calls,
    // Pools allocation or kernel instances. Validates exact source/share BEFORE
    // typed owned omission. Caller outputs remain unchanged on refusal. Programs,
    // final Plans and source cache remain immutable/alive through execute.
    static int prepare(std::span<const GlmMainOrdinaryRankSpec>,
        const GlmMainOrdinaryLoaderLimits&,std::unique_ptr<GlmMainOrdinaryLoader>* out,
        GlmMainOrdinaryLoaderQuote* quote,std::string* why=nullptr)noexcept;
    ~GlmMainOrdinaryLoader();
    // Exactly once. One reusable pinned input/output pair and bounded column
    // scratch; all destinations validated before first payload/copy. Actual
    // copy + successful REAL drain precede reuse. Ambiguous GPU errors invoke
    // the actual admitted binding's fail_stop BEFORE buffer cleanup.
    int execute(std::span<const GlmMainOrdinaryRuntimeRank>,
                GlmMainOrdinaryLoaderReceipt* out,std::string* why=nullptr)noexcept;
    const GlmMainOrdinaryLoaderQuote& quote()const noexcept;
    GlmMainOrdinaryLoader(const GlmMainOrdinaryLoader&)=delete;
private:
    struct Impl;explicit GlmMainOrdinaryLoader(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> p_;
};
} // namespace rad

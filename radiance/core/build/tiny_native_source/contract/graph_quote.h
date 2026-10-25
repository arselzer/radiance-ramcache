#pragma once
#include "profile.h"
#include "bound_sources.h"
#include "place/native_weight_authority.h"
namespace rad {class Program;class Plan;class CheckpointSources;}
namespace glm53::engine_core_contract {
struct GraphQuote {
    uint64_t physical_tensors=0,weights=0,expert_views=0,nonexpert_views=0,logical_sources=0;
    uint64_t units=0,expert_units=0,ordinary_views=0,ordinary_stored_bytes=0,ordinary_page_upper=0;
    uint64_t model_units=0,layer_units=0;
    uint64_t all_stored_bytes=0,actual_arena_scratch_bytes=0,bands=0;
    uint64_t explicit_full_spans=0,gate_up_rowparts=0,selected_plane_rects=0;
    rad::NativeAuthorityStats authority{};
    bool engine_eligible=false;
};
// Borrows the actual finished Program and pure metadata Plan; BoundSources retains
// the real Core Quark resolver AND its exact opened Checkpoint. Closed namespace,
// bounded header/tokenizer metadata, artifact-stat and import-plane identity
// audits read metadata, no physical weight payload.
// No W-element WeightInfo snapshot, capture, Pools, source payload read,
// Native owner, stream, Ctx or Engine construction. Failure preserves caller quote.
int quote_declared_graph(const ColdSourceSeal&,const rad::Program&,const rad::Plan&,
    const BoundSources&,GraphQuote*,std::string* why=nullptr);
} // namespace glm53::engine_core_contract

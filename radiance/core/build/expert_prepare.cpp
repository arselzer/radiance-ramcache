#include "rad_build.h"
#include "rad_expert_prepare.h"
#include <algorithm>
#include <limits>
namespace rad {
int Builder::decl_expert_route_map(const RadExpertRouteDecl* d){
    if(finished_){err(RAD_E_STATE,"expert route map after finish");return RAD_E_STATE;}
    if(!d||!d->entries||d->n_entries<=0||d->n_entries!=d->local_experts||d->local_experts>4096||d->canonical_experts<d->local_experts||d->canonical_experts>65536||d->expert_base<0||int64_t(d->expert_base)+d->local_experts>d->canonical_experts||d->layer<0||d->max_rows<=0||d->max_rows>(1<<20)||d->top_k<=0||d->top_k>d->canonical_experts||d->regular_experts<0||d->regular_experts>d->local_experts||d->classes<1||d->classes>4||!d->gu||d->gu>=prog_.ops.size()||prog_.expert_route_maps.size()>=256)return RAD_E_INVAL;
    if(std::any_of(prog_.expert_route_maps.begin(),prog_.expert_route_maps.end(),[&](const auto& m){return m.decl.layer==d->layer;}))return RAD_E_INVAL;
    size_t total=size_t(d->n_entries);for(const auto& m:prog_.expert_route_maps)total+=m.entries.size();if(total>65536)return RAD_E_SHAPE;
    for(const auto h:{d->counts,d->offsets,d->sorted,d->ids})if(!h||h>=prog_.buffers.size()||prog_.buffers[h].decl.dtype!=RAD_I32)return RAD_E_DTYPE;
    std::vector<uint8_t> canonical(size_t(d->canonical_experts),0);
    for(int i=0;i<d->n_entries;++i){const auto& e=d->entries[i];if(e.slot!=i||e.canonical<d->expert_base||e.canonical>=int64_t(d->expert_base)+d->local_experts||canonical[size_t(e.canonical)]||e.protected_expert!=(i>=d->regular_experts)||e.class_index<-1||e.class_index>=d->classes)return RAD_E_INVAL;canonical[size_t(e.canonical)]=1;
        if(!e.members[0]||!e.members[2]||bool(e.members[1])!=bool(e.members[3])||(e.protected_expert&&(e.members[1]||e.members[3])))return RAD_E_INVAL;
        for(const auto h:e.members)if(h){if(h>=prog_.weights.size()||prog_.weights[h].decl.group.layer!=d->layer)return RAD_E_INVAL;}
    }
    ExpertRouteMap map;map.decl=*d;map.decl.entries=nullptr;map.entries.assign(d->entries,d->entries+d->n_entries);
    // Prepare all growing containers before publishing map/lifetime metadata.
    prog_.expert_route_maps.reserve(prog_.expert_route_maps.size()+1);
    auto& reads=op_bufs_[d->gu].reads;reads.reserve(reads.size()+4);
    // Extends these CPU handoff reads through the first dependent GU in the real arena plan.
    const rad_buf read[]{d->counts,d->offsets,d->sorted,d->ids};const int status=op_reads(d->gu,read,4);if(status<0)return status;
    prog_.expert_route_maps.push_back(std::move(map));return RAD_OK;
}
}
extern "C" int rad_decl_expert_route_map(RadBuilder* b,const RadExpertRouteDecl* d){
    if(!b)return RAD_E_INVAL;
    try{return static_cast<rad::Builder*>(b)->decl_expert_route_map(d);}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_INVAL;}
}

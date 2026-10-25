#include "metadata_contract.h"
#include <algorithm>
#include <new>
#include <stdexcept>
namespace glm53::main_metadata {
namespace {
template<class T> struct Restore {T& target;T saved;explicit Restore(T& t):target(t),saved(t){}~Restore(){target=std::move(saved);}};
}
int contract_checks(rad::Checkpoint& ck,rad::CheckpointSources& sources,rad::Program& p,rad::Plan& plan,
                    ContractReceipt* out,std::string* why)noexcept{
    try{
        if(!out)return RAD_E_INVAL;ContractReceipt receipt;
        auto check=[&](bool b,const char* text){++receipt.checks;if(!b)throw std::runtime_error(text);};
        Limits limits;Quote baseline;check(quote(ck,sources,p,plan,{},limits,&baseline,nullptr)==RAD_OK,"actual main metadata baseline");
        check(baseline.main_units==12096&&baseline.records.size()==12096&&baseline.owned.size()==12096&&baseline.resident_main_units.empty(),"actual42x288 complete baseline");
        check(baseline.stored_bytes==baseline.source_selected_bytes&&baseline.physical_rectangles==72576,"actual complete six-rectangle sources");
        check(baseline.construction_requested==baseline.immutable_requested+baseline.scratch_requested&&baseline.construction_requested<=limits.max_metadata_bytes,"actual metadata precharge arithmetic");
        const auto unit=baseline.records.front().unit;check(unit>=0&&size_t(unit)<plan.units.size(),"actual first main unit handle");
        auto& u=plan.units[size_t(unit)];const auto h=u.weights[0],scale=u.weights[1],down=u.weights[2];
        auto refuse=[&](const char* text){Quote sentinel;sentinel.rank=987;sentinel.records.resize(1);sentinel.records[0].unit=654;
            const auto* retained=sentinel.records.data();check(quote(ck,sources,p,plan,{},limits,&sentinel,nullptr)<0,text);
            check(sentinel.rank==987&&sentinel.records.size()==1&&sentinel.records.data()==retained&&sentinel.records[0].unit==654,"refusal preserves caller output and storage");++receipt.refusals;};
        {Restore v(p.ctx.world_size);p.ctx.world_size=2;refuse("foreign Program world");}
        {Restore v(plan.rank);plan.rank=(plan.rank+1)%4;refuse("foreign Plan rank");}
        {Restore v(p.ctx.max_spec);p.ctx.max_spec=1;refuse("MTP input refuses main-only profile");}
        {Restore v(p.meta.n_expert_used);p.meta.n_expert_used=4;refuse("wrong actual top-k");}
        {Restore v(u.expert);u.expert=288;refuse("actual expert outside main domain");}
        {Restore v(u.layer);u.layer=45;refuse("MTP unit refuses main profile");}
        {Restore v(u.offset[1]);++u.offset[1];refuse("actual member offset mutation");}
        {Restore v(u.slot_bytes);u.slot_bytes+=4096;refuse("actual unit stride mutation");}
        {Restore v(plan.unit_of_weight[h]);plan.unit_of_weight[h]=-1;refuse("actual weight-unit map mutation");}
        {Restore v(p.weights[h].source);p.weights[h].source="foreign.main.source";refuse("foreign logical source name");}
        {Restore v(p.weights[h].decl.group.slot);p.weights[h].decl.group.slot=1;refuse("actual member role mutation");}
        {Restore v(p.weights[h].sel[0]);p.weights[h].sel[0]=1;refuse("wrong selected main plane");}
        {Restore v(p.weights[h].enc);p.weights[h].enc=rad_enc_plain(RAD_BF16);refuse("wrong main encoding");}
        {Restore v(p.weights[h].shard_lo);++p.weights[h].shard_lo;refuse("wrong actual TP4 start span");}
        {Restore v(p.weights[down].shard_hi);--p.weights[down].shard_hi;refuse("wrong actual DOWN end span");}
        {Restore v(p.weights[h].decl.row_parts[1]);--p.weights[h].decl.row_parts[1];refuse("wrong actual up half");}
        {Restore v(p.weights[scale].stored_dtype);p.weights[scale].stored_dtype=RAD_U8;refuse("wrong E8M0 stored dtype");}
        {Restore v(p.weights[h].stored_bytes);--p.weights[h].stored_bytes;refuse("wrong actual stored bytes");}
        {Restore v(p.weights[h].layout_tag);p.weights[h].layout_tag="unsupported-packed-form";refuse("unsupported real selected layout tag");}
        {Restore v(p.weights[h].identity);p.weights[h].identity=false;refuse("nonidentity layout refused");}
        {Restore v(p.weights[h].widen);p.weights[h].widen=true;refuse("widened layout refused");}
        {Restore v(limits.max_metadata_bytes);limits.max_metadata_bytes=baseline.construction_requested-1;refuse("exact metadata precharge-minus-one");}
        // The real cache must already hold the graph's actual named resolutions;
        // an empty same-checkpoint cache cannot substitute geometry alone.
        rad::CheckpointSources empty(ck);Quote sentinel;sentinel.rank=765;
        check(quote(ck,empty,p,plan,{},limits,&sentinel,nullptr)<0&&sentinel.rank==765,"empty source cache cannot seal actual source metadata");++receipt.refusals;
        // Deliberately stale name-map and stale ordered cache pieces are separate.
        auto map=std::find_if(p.name_map.begin(),p.name_map.end(),[&](const auto& m){return m.declared&&p.weights[h].source==m.declared;});
        check(map!=p.name_map.end()&&map->n_src==2,"actual GU ordered name map exists");
        {Restore a(map->src[0]);Restore b(map->src[1]);std::swap(map->src[0],map->src[1]);refuse("current GU name-map order mutation");}
        auto* resolved=const_cast<rad::CkptWeight*>(sources.find(p.weights[h].source));
        check(resolved&&resolved->pieces.size()==2&&resolved->scales.size()==2,"actual cached GU planes exist");
        {Restore a(resolved->pieces[0]);Restore b(resolved->pieces[1]);std::swap(resolved->pieces[0],resolved->pieces[1]);refuse("stale swapped GU code cache with current map restored");}
        {Restore a(resolved->scales[0]);Restore b(resolved->scales[1]);std::swap(resolved->scales[0],resolved->scales[1]);refuse("stale swapped GU scale cache with current map restored");}
        const int32_t resident=unit;Quote one;check(quote(ck,sources,p,plan,{&resident,1},limits,&one,nullptr)==RAD_OK&&one.owned.size()==12095&&one.resident_main_units==std::vector<int32_t>({unit}),"one explicit actual resident partitions main metadata");
        check(std::none_of(one.owned.begin(),one.owned.end(),[&](const auto& b){return b.unit==unit;}),"resident has no owned binding");
        const int32_t duplicate[]={unit,unit};Quote unchanged;unchanged.rank=543;
        check(quote(ck,sources,p,plan,duplicate,limits,&unchanged,nullptr)<0&&unchanged.rank==543,"duplicate resident refuses without publication");++receipt.refusals;
        Quote final;check(quote(ck,sources,p,plan,{},limits,&final,nullptr)==RAD_OK&&final.stored_bytes==baseline.stored_bytes&&final.owned.size()==baseline.owned.size(),"all actual metadata mutations restored");
        *out=receipt;return RAD_OK;
    }catch(const std::bad_alloc&){if(why)try{*why="contract allocation refused";}catch(...){}return RAD_E_NOMEM;}
    catch(const std::exception& e){if(why)try{*why=e.what();}catch(...){}return RAD_E_STATE;}
    catch(...){return RAD_E_STATE;}
}
}

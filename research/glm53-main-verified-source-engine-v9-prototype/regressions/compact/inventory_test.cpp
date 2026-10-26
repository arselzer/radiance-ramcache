#include "allocation_hooks.h"
#include "baseline_metadata.h"
#include "mutations.h"
#include "strict_p3/adapter_fixture.h"
namespace {
size_t compact_groups=0,compact_mutations=0,compact_oom=0;
void need(bool value,const std::string& why){need(value,why.c_str());}
void exact_inventory_fields(const std::string& home){++compact_groups;Fixture f(home,true);auto& program=f.builder->program();
    const auto unit=f.inventory->owned().front().unit;const auto weight=f.plan.units[size_t(unit)].weights.front();
    const auto original=program.weights[weight];const auto before=f.source->stats().payload_reads;const auto acquires=f.source->stats().acquires;
    need(f.inventory->validate(program,f.plan)==RAD_OK&&f.inventory->validate_unit(program,f.plan,unit)==RAD_OK,"initial compact authority validation");
    for(const auto& mutation:compact_tests::mutations()){
        program.weights[weight]=original;mutation.apply(program.weights[weight]);
        const bool want=compact_baseline::metadata(program.weights[weight],original,true);
        const int selected=f.inventory->validate_unit(program,f.plan,unit);
        const int all=f.inventory->validate(program,f.plan);
        need((selected==RAD_OK)==want&&(all==RAD_OK)==want,"actual inventory differential field: "+mutation.label);
        need(f.source->stats().payload_reads==before&&f.source->stats().acquires==acquires,"metadata refusal reached source I/O");++compact_mutations;
    }
    program.weights[weight]=original;good(f.inventory->validate(program,f.plan),"restored metadata");
    const auto& stats=f.inventory->authority_stats();
    need(stats.weights==program.weights.size()&&stats.reference_size==32&&stats.geometry_size==sizeof(WeightInfo)&&
         stats.admissible&&stats.construction_peak_bytes<=f.inventory->footprint().metadata_reservation,
         "actual inventory authority quote/type/count");
    std::cout<<"COMPACT_INVENTORY W="<<stats.weights<<" G="<<stats.geometries<<" strings="<<stats.strings
        <<" nameSourceBytes="<<stats.name_source_bytes<<" tagBytes="<<stats.layout_tag_bytes
        <<" authorityImmutable="<<stats.immutable_bytes<<" authorityScratch="<<stats.scratch_bytes
        <<" authorityPeak="<<stats.construction_peak_bytes<<" wholeMetadataReserved="<<f.inventory->footprint().metadata_reservation
        <<" oldWeightTermOnly="<<program.weights.size()*(2*sizeof(WeightInfo)+128)<<"\n";
    good(f.run(),"same real source/current route/FFN after restored metadata");
}
void map_classes_and_foreign(const std::string& home){++compact_groups;Fixture f(home,true);auto& program=f.builder->program();
    const auto unit=f.inventory->owned().front().unit;const auto w=f.plan.units[size_t(unit)].weights.front();
    const auto map=f.plan.unit_of_weight;const auto counts=f.source->stats();
    f.plan.unit_of_weight.clear();need(f.inventory->validate_runtime_header(program,f.plan)<0,"cleared map reached compact indexing");f.plan.unit_of_weight=map;
    f.plan.unit_of_weight.pop_back();need(f.inventory->validate_runtime_header(program,f.plan)<0,"shrunk map reached compact indexing");f.plan.unit_of_weight=map;
    f.plan.unit_of_weight[w]=unit+1;need(f.inventory->unit_for_weight(w)==unit&&f.inventory->validate_unit(program,f.plan,unit)<0,"mutable map replaced frozen expected unit");f.plan.unit_of_weight=map;
    f.plan.unit_of_weight[0]=unit;need(f.inventory->validate_runtime_header(program,f.plan)<0,"null-handle sentinel map mutation accepted");f.plan.unit_of_weight=map;
    const auto offset=f.plan.units[size_t(unit)].offset[0];f.plan.units[size_t(unit)].offset[0]+=4096;
    need(f.inventory->validate_unit(program,f.plan,unit)<0,"same-size selected unit offset mutation accepted");f.plan.units[size_t(unit)].offset[0]=offset;
    const auto classes=f.plan.classes;f.plan.classes.clear();need(f.inventory->validate_runtime_header(program,f.plan)<0,"cleared class count accepted");f.plan.classes=classes;
    const auto c=f.plan.units[size_t(unit)].slab_class;f.plan.classes[size_t(c)].stride+=4096;
    need(f.inventory->validate_unit(program,f.plan,unit)<0,"selected class stride mutation accepted");f.plan.classes=classes;
    ++f.plan.units[size_t(unit)].first_use_op;need(f.inventory->validate_unit(program,f.plan,unit)<0,"unit first-use remains independently guarded");--f.plan.units[size_t(unit)].first_use_op;
    const auto member=f.plan.units[size_t(unit)].weights[0];f.plan.units[size_t(unit)].weights[0]=f.plan.units[size_t(unit)].weights[1];
    need(f.inventory->validate_unit(program,f.plan,unit)<0,"immutable member order replaced");f.plan.units[size_t(unit)].weights[0]=member;
    Fixture other(home,false);need(f.inventory->validate(other.builder->program(),f.plan)<0,"foreign same-shaped Program accepted");
    std::vector<RecordOwnerBinding> bindings;for(const auto& x:f.inventory->owned())bindings.push_back({x.unit,x.record});
    const auto live_catalog=f.source->catalog();std::vector<NativeRecordDescriptor> catalog(live_catalog.begin(),live_catalog.end());
    const auto fixed=residents(f);need(bindings.size()>1,"foreign descriptor matrix needs distinct records");
    for(int field=0;field<4;++field){auto changed=catalog;auto& d=changed[bindings[field==3?1:0].record];
        if(field==0)d.rank=1;else if(field==1)d.world=4;else if(field==2)d.source_generation=0;
        else {++d.source_generation;if(!d.source_generation)++d.source_generation;}
        std::shared_ptr<const NativeInventory> refused;
        need(NativeInventory::bind(program,f.metadata_plan,changed,bindings,fixed,f.inventory->limits(),&refused)==RAD_E_INVAL&&!refused,
             "foreign rank/world/missing or mixed source identity published compact inventory");}
    good(f.inventory->validate(program,f.plan),"authority restored after negative metadata gates");
    const auto after=f.source->stats();need(after.acquires==counts.acquires&&after.payload_reads==counts.payload_reads,"pure authority checks reached source");
}
void inventory_bind_allocations(const std::string& home){++compact_groups;Fixture f(home,true);auto& program=f.builder->program();
    std::vector<RecordOwnerBinding> bindings;for(const auto& x:f.inventory->owned())bindings.push_back({x.unit,x.record});
    const auto fixed=residents(f);const auto catalog=f.source->catalog();const auto old_map=f.metadata_plan.unit_of_weight;
    const auto payload=payload_totals();const auto source=f.source->stats();
    auto limits=f.inventory->limits();limits.max_metadata_bytes=f.inventory->footprint().metadata_reservation;
    std::shared_ptr<const NativeInventory> ready;std::string why;
    good(NativeInventory::bind(program,f.metadata_plan,catalog,bindings,fixed,limits,&ready,&why),"exact compact precharge budget");
    need(ready->footprint().metadata_reservation==limits.max_metadata_bytes,"exact metadata quote changed");
    --limits.max_metadata_bytes;std::shared_ptr<const NativeInventory> refused;
    const int cap=NativeInventory::bind(program,f.metadata_plan,catalog,bindings,fixed,limits,&refused);
    need(cap==RAD_E_SCRATCH&&!refused&&payload_totals()==payload,"one-byte under-budget inventory allocated payload/published");
    limits=f.inventory->limits();auto existing=f.inventory;allocation_fault::begin();
    const int occupied=NativeInventory::bind(program,f.metadata_plan,catalog,bindings,fixed,limits,&existing);allocation_fault::end();
    need(occupied==RAD_E_INVAL&&existing==f.inventory&&allocation_fault::calls==0,"existing caller inventory replaced/allocated");
    bool succeeded=false;
    for(int64_t site=0;site<512;++site){std::shared_ptr<const NativeInventory> candidate;
        allocation_fault::begin();allocation_fault::remaining=site;
        const int status=NativeInventory::bind(program,f.metadata_plan,catalog,bindings,fixed,limits,&candidate);allocation_fault::end();
        if(status==RAD_OK){need(candidate&&candidate->validate(program,f.metadata_plan)==RAD_OK,"final successful bind invalid");succeeded=true;break;}
        need(status==RAD_E_NOMEM&&!candidate&&allocation_fault::live==0,"partial snapshot/credit leaked on bind allocation failure");++compact_oom;
        need(payload_totals()==payload&&f.metadata_plan.unit_of_weight==old_map,"bind failure changed Program/Plan/device payload");
    }
    const auto after=f.source->stats();need(succeeded&&compact_oom>=16&&after.acquires==source.acquires&&after.payload_reads==source.payload_reads,"all bind allocation sites/pure source boundary not exercised");
    good(f.inventory->validate(program,f.plan),"original live capability preserved after all allocator failures");
}
}
int main(int argc,char** argv){try{
    need(argc==3&&!std::strcmp(argv[1],"--radiance-home"),"usage: compact_native_inventory_test --radiance-home MATCHING_COMPACT_HOST_HOME");
    need(device_is_host(),"HOST only, no GPU execution");exact_inventory_fields(argv[2]);map_classes_and_foreign(argv[2]);inventory_bind_allocations(argv[2]);
    need(compact_groups==3&&compact_mutations>=600,"compact native matrix incomplete");
    std::cout<<"PASS compact NativeInventory groups="<<compact_groups<<" checks="<<checks<<" mutations="<<compact_mutations<<" OOMrefusals="<<compact_oom<<"; actual HOST inventory/source/owner, no full-model/GPU/TP claim\n";return 0;
}catch(const std::exception& error){allocation_fault::end();std::cerr<<"FAIL compact NativeInventory: "<<error.what()<<'\n';return 1;}}

#include "native_backing_owner.h"
#include "runtime/ctx.h"
#include "../build/tiny_component_quotes.h"
#include <algorithm>
namespace rad {
namespace {
uint64_t checksum(std::span<const std::byte> bytes){
    uint64_t h=14695981039346656037ull;
    for(const auto b:bytes){h^=std::to_integer<uint8_t>(b);h*=1099511628211ull;}
    return h;
}
}
struct NativeBackingOwner::Impl {
    Program* program=nullptr;
    const Ctx* context=nullptr;
    const Plan* plan=nullptr;
    Mover* mover=nullptr;
    std::shared_ptr<const NativeInventory> inventory;
    std::shared_ptr<NativeRecordSource> source;
    std::vector<WeightSlot> fixed;
    std::vector<uint32_t> request;
    std::vector<RecordPublication> publication;
    std::vector<int32_t> used;
    NativeRecordBatch held;
    NativeBackingStats stats;
    uint64_t token=0;
    int fatal=RAD_OK;
    bool quiesced=true;
};
int NativeBackingOwner::tiny_startup_quote(const Program& program,const Plan& plan,TinyOwnerQuote* out) noexcept {
    if(!out)return RAD_E_INVAL;
    if(tiny_quote::program_profile(program)<0||!plan.native_inventory||plan.native_inventory->program()!=&program)return RAD_E_UNSUPPORTED;
    try {
        const auto& inv=*plan.native_inventory;const int valid=inv.validate(program,plan);if(valid<0)return valid;
        TinyOwnerQuote q;q.sizeof_owner=sizeof(NativeBackingOwner);q.sizeof_impl=sizeof(Impl);q.sizeof_weight_slot=sizeof(WeightSlot);q.sizeof_publication=sizeof(RecordPublication);q.sizeof_lease=sizeof(NativeRecordLease);
        q.weights=program.weights.size();q.slots=inv.footprint().consumer_slots;q.classes=inv.classes().size();
        if(!q.slots||q.slots>512||!q.classes||q.classes>64)return RAD_E_UNSUPPORTED;
        auto& s=q.storage;s.embedded_object_bytes=sizeof(NativeBackingOwner);
        if(!tiny_quote::heap(s,1,sizeof(Impl))||!tiny_quote::heap(s,q.weights,sizeof(WeightSlot))||!tiny_quote::heap(s,q.slots,sizeof(uint32_t))||!tiny_quote::heap(s,q.slots,sizeof(RecordPublication))||!tiny_quote::heap(s,q.slots,sizeof(NativeRecordLease))||!tiny_quote::heap(s,q.classes,sizeof(int32_t)))return RAD_E_SHAPE;
        s.control_allowance=512;
        if(!tiny_quote::finish(s))return RAD_E_SCRATCH;
        *out=q;return RAD_OK;
    }catch(...){return RAD_E_STATE;}
}
NativeBackingOwner::NativeBackingOwner(std::unique_ptr<Impl> p):p_(std::move(p)){}
NativeBackingOwner::~NativeBackingOwner(){
    if(!p_)return;
    // Ctx admission is the final, non-fallible commit in expert_prepare_open. If it never
    // succeeded, there cannot be a WTAB retaining these consumers, so failed bringup can close.
    if(p_->token&&!p_->context&&quiesce()>=0&&p_->mover->record_session_close(p_->token)>=0)
        p_->token=0;
    // Normal ownership explicitly closes after Ctx shutdown. Never free live leases on failed
    // drains. Backing itself is Mover/Pools-owned, and must outlive Ctx and this Owner.
    if(p_->token)(void)p_.release();
}
int NativeBackingOwner::open(Program& program,const Plan& plan,Mover& mover,
                             std::shared_ptr<NativeRecordSource> source,
                             std::unique_ptr<NativeBackingOwner>* out){
    if(!out||*out||!source||!plan.native_inventory||
       !mover.record_plan_matches(program,plan))return RAD_E_INVAL;
    RAD_TRY(plan.native_inventory->validate(program,plan));
    try{
        auto p=std::make_unique<Impl>();
        p->program=&program;p->plan=&plan;p->mover=&mover;
        p->inventory=plan.native_inventory;p->source=std::move(source);
        const auto catalog=p->source->catalog();
        for(const auto& owned:p->inventory->owned())
            if(owned.record>=catalog.size()||catalog[owned.record]!=owned.descriptor)
                return RAD_E_INVAL;
        const auto slots=p->inventory->footprint().consumer_slots;
        p->request.reserve(slots);p->publication.reserve(slots);p->held.leases.reserve(slots);
        p->used.resize(p->inventory->classes().size(),0);
        p->fixed.resize(program.weights.size());
        for(size_t w=1;w<program.weights.size();++w){
            const auto u=p->inventory->unit_for_weight(rad_weight(w));
            if(p->inventory->find(u))continue;
            const auto* original=mover.residency()->lookup(rad_weight(w));
            if(!original||!original->ptr||original->ready)return RAD_E_STATE;
            p->fixed[w]=*original;
        }
        // Object allocation precedes session opening; no fallible allocation after publication.
        auto owner=std::unique_ptr<NativeBackingOwner>(new NativeBackingOwner(std::move(p)));
        RAD_TRY(mover.native_record_session_open(&owner->p_->token));
        *out=std::move(owner);
        return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
    catch(...){return RAD_E_STATE;}
}
int NativeBackingOwner::check_program(const Program& program)const{
    return p_->inventory->validate(program,*p_->plan);
}
bool NativeBackingOwner::accepts_native_inventory(const NativeInventory* inventory)const{
    return p_->token&&p_->fatal>=0&&inventory==p_->inventory.get();
}
int NativeBackingOwner::bind_native_context(const NativeInventory* inventory,const Ctx* context,RadStream stream){
    if(!accepts_native_inventory(inventory)||!context||
       context->program()!=p_->program||stream!=p_->mover->native_record_compute_stream()||
       p_->context)return RAD_E_INVAL;
    p_->context=context;
    return RAD_OK;
}
int NativeBackingOwner::quiesce(){
    if(!p_->token)return RAD_E_STATE;
    const int status=p_->mover->record_restore(p_->token);
    if(status<0){p_->quiesced=false;if(p_->fatal>=0)p_->fatal=status;return status;}
    p_->held.leases.clear();p_->stats.held_leases=0;p_->quiesced=true;
    ++p_->stats.retirements;
    return RAD_OK;
}
int NativeBackingOwner::close_after_context_shutdown(){
    RAD_TRY(quiesce());
    RAD_TRY(p_->mover->record_session_close(p_->token));
    p_->token=0;
    return RAD_OK;
}
int NativeBackingOwner::end_pass(RadStream,int status){
    if(status<0&&p_->fatal>=0)p_->fatal=status;
    return quiesce();
}
NativeBackingStats NativeBackingOwner::stats()const noexcept{return p_->stats;}
int NativeBackingOwner::prepare(const ExpertPrepareView& view,bool* changed){
    auto& p=*p_;
    auto fail=[&](int status){if(p.fatal>=0)p.fatal=status;return status;};
    if(p.fatal<0)return p.fatal;
    if(!changed||!p.token||view.program!=p.program||view.context!=p.context||!view.context||
       view.context->program()!=p.program||view.context->cur_stream()!=view.producer_stream||
       view.producer_stream!=p.mover->native_record_compute_stream()||
       view.rows<0||uint64_t(view.rows)>p.inventory->limits().max_rows||
       view.declaration.top_k!=p.inventory->limits().top_k)return fail(RAD_E_INVAL);
    *changed=false;++p.stats.prepares;
    // Check sizes before any mutable Plan indexing, then derive each selected unit from the
    // immutable map and validate its complete current layout/mappings before using it.
    const int current=p.inventory->validate_runtime_header(*p.program,*p.plan);
    if(current<0)return fail(current);
    const auto epoch=p.mover->residency()->epoch();
    const int retired=quiesce();
    if(retired<0)return retired;
    *changed=p.mover->residency()->epoch()!=epoch;
    p.request.clear();p.publication.clear();
    std::fill(p.used.begin(),p.used.end(),0);
    try{
        const auto catalog=p.source->catalog();
        for(const auto& entry:view.map){
            if(entry.slot<0||size_t(entry.slot)>=view.counts.size())return fail(RAD_E_SHAPE);
            if(view.counts[size_t(entry.slot)]<=0)continue;
            if(!entry.members[0])return fail(RAD_E_INVAL);
            const auto unit=p.inventory->unit_for_weight(entry.members[0]);
            if(p.inventory->validate_unit(*p.program,*p.plan,unit)<0)return fail(RAD_E_STATE);
            const auto* owned=p.inventory->find(unit);
            if(!owned){
                for(const auto w:entry.members){
                    if(!w)continue;
                    if(w>=p.fixed.size())return fail(RAD_E_INVAL);
                    const auto* slot=p.mover->residency()->lookup(w);const auto& expected=p.fixed[w];
                    if(!slot||!expected.ptr||slot->ptr!=expected.ptr||slot->ready!=expected.ready||
                       slot->generation!=expected.generation)return fail(RAD_E_STATE);
                }
                continue;
            }
            const auto& mu=p.plan->units[size_t(unit)];
            if(mu.layer!=view.declaration.layer||mu.expert!=entry.slot||mu.weights.size()!=4||
               !std::equal(mu.weights.begin(),mu.weights.end(),entry.members)||
               owned->record>=catalog.size()||
               catalog[owned->record]!=owned->descriptor)return fail(RAD_E_STATE);
            const auto& cls=p.inventory->classes()[size_t(owned->klass)];
            if(p.inventory->validate_unit(*p.program,*p.plan,cls.exemplar)<0)return fail(RAD_E_STATE);
            auto& used=p.used[size_t(owned->klass)];
            if(used>=cls.slots||p.request.size()>=p.inventory->footprint().consumer_slots)
                return fail(RAD_E_SCRATCH);
            p.request.push_back(owned->record);
            p.publication.push_back({unit,cls.first_region+1+used++});
        }
        if(p.request.empty())return RAD_OK;
        NativeRecordBatch next;
        ++p.stats.source_batches;
        const int status=p.source->acquire(p.request,&next);
        if(status<0)return fail(status);
        if(next.leases.size()!=p.request.size())return fail(RAD_E_STATE);
        for(size_t i=0;i<next.leases.size();++i){
            const auto* owned=p.inventory->find(p.publication[i].unit);
            const auto& lease=next.leases[i];
            if(lease.record!=p.request[i]||!lease.lifetime||!lease.bytes.data()||
               lease.bytes.size()!=uint64_t(owned->descriptor.stride)||
               checksum(lease.bytes)!=owned->descriptor.checksum)return fail(RAD_E_INVAL);
        }
        // Full union is verified before any landing/output mutation. Keep strong cache leases
        // through H2D and the entire GU/activation/DN/gather slice, including issue failure.
        p.held=std::move(next);p.quiesced=false;p.stats.held_leases=p.held.leases.size();
        p.stats.peak_held_leases=std::max(p.stats.peak_held_leases,p.stats.held_leases);
        for(size_t i=0;i<p.held.leases.size();++i){
            const auto& publication=p.publication[i];
            const int copied=p.mover->native_record_upload(p.token,publication.unit,publication.region,
                                                          p.held.leases[i].bytes);
            if(copied<0)return fail(copied);
            p.stats.source_bytes+=p.held.leases[i].bytes.size();
            p.stats.copy_bytes+=p.held.leases[i].bytes.size()+4096;
        }
        const int published=p.mover->record_publish(p.token,p.publication);
        if(published<0)return fail(published);
        p.stats.published_units+=p.publication.size();*changed=true;
        return RAD_OK;
    }catch(const std::bad_alloc&){return fail(RAD_E_NOMEM);}
    catch(...){return fail(RAD_E_STATE);}
}
} // namespace rad

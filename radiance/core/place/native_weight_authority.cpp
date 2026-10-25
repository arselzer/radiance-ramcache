#include "native_weight_authority.h"
#include "planner.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <iterator>
#include <new>
#include <type_traits>
namespace rad {
namespace {
struct Failure {int status;const char* message;};
void require(bool value,int status,const char* message){if(!value)throw Failure{status,message};}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,RAD_E_SHAPE,"authority addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){require(!a||b<=UINT64_MAX/a,RAD_E_SHAPE,"authority multiplication overflow");return a*b;}
template<class T,size_t N> bool same(const T(&a)[N],const T(&b)[N])noexcept{return std::equal(a,a+N,b);}
// Exact old metadata(false) numeric/encoding fields. Layout-tag identity is
// checked separately against an exact owned byte slice, never normalized.
bool geometry_fields(const WeightInfo& a,const WeightInfo& b)noexcept{
    const auto& x=a.decl;const auto& y=b.decl;
    return x.dtype==y.dtype&&x.rank==y.rank&&same(x.shape,y.shape)&&x.access==y.access&&
        x.shard==y.shard&&x.optional==y.optional&&x.n_row_parts==y.n_row_parts&&
        same(x.row_parts,y.row_parts)&&a.logical_bytes==b.logical_bytes&&
        a.shard_lo==b.shard_lo&&a.shard_hi==b.shard_hi&&a.n_part_spans==b.n_part_spans&&
        same(a.part_lo,b.part_lo)&&same(a.part_hi,b.part_hi)&&same(a.part_total,b.part_total)&&
        a.enc_known==b.enc_known&&rad_enc_equal(&a.enc,&b.enc)&&a.n_sel==b.n_sel&&
        same(a.sel,b.sel)&&same(a.sel_rows,b.sel_rows)&&same(a.sel_cols,b.sel_cols)&&
        a.identity==b.identity&&a.widen==b.widen&&a.stored_dtype==b.stored_dtype&&
        a.stored_rank==b.stored_rank&&same(a.stored_shape,b.stored_shape)&&
        a.stored_bytes==b.stored_bytes&&a.align==b.align;
}
// Fieldwise construction: ignored names/source/tag strings are NEVER copied.
// Their default std::strings stay empty; all compared tag bytes live in arena.
void copy_geometry(WeightInfo& d,const WeightInfo& s)noexcept{
    d.decl.dtype=s.decl.dtype;d.decl.rank=s.decl.rank;
    std::copy(std::begin(s.decl.shape),std::end(s.decl.shape),std::begin(d.decl.shape));
    d.decl.access=s.decl.access;d.decl.shard=s.decl.shard;d.decl.optional=s.decl.optional;
    d.decl.n_row_parts=s.decl.n_row_parts;
    std::copy(std::begin(s.decl.row_parts),std::end(s.decl.row_parts),std::begin(d.decl.row_parts));
    d.logical_bytes=s.logical_bytes;d.shard_lo=s.shard_lo;d.shard_hi=s.shard_hi;d.n_part_spans=s.n_part_spans;
    std::copy(std::begin(s.part_lo),std::end(s.part_lo),std::begin(d.part_lo));
    std::copy(std::begin(s.part_hi),std::end(s.part_hi),std::begin(d.part_hi));
    std::copy(std::begin(s.part_total),std::end(s.part_total),std::begin(d.part_total));
    std::memcpy(&d.enc,&s.enc,sizeof d.enc);d.enc_known=s.enc_known;d.n_sel=s.n_sel;
    std::copy(std::begin(s.sel),std::end(s.sel),std::begin(d.sel));
    std::copy(std::begin(s.sel_rows),std::end(s.sel_rows),std::begin(d.sel_rows));
    std::copy(std::begin(s.sel_cols),std::end(s.sel_cols),std::begin(d.sel_cols));
    d.identity=s.identity;d.widen=s.widen;d.stored_dtype=s.stored_dtype;d.stored_rank=s.stored_rank;
    std::copy(std::begin(s.stored_shape),std::end(s.stored_shape),std::begin(d.stored_shape));
    d.stored_bytes=s.stored_bytes;d.align=s.align;
}
struct Ref {
    uint32_t geometry=0,name=0,source=0;
    int32_t layer=0,expert=0,slot=0,unit=-1,member=-1;
};
struct Slice {uint32_t offset=0,length=0;};
struct Bucket {uint64_t hash=0;uint32_t representative=UINT32_MAX,id=0;};
static_assert(sizeof(Ref)==32&&alignof(Ref)==4&&std::is_trivially_copyable_v<Ref>);
static_assert(sizeof(Slice)==8&&alignof(Slice)==4&&std::is_trivially_copyable_v<Slice>);
static_assert(sizeof(Bucket)==16&&alignof(Bucket)==8&&std::is_trivially_copyable_v<Bucket>);
constexpr uint64_t fixed_allowance=4096; // owner headers/array cookies/allocator bookkeeping; not RSS
uint64_t table_count(uint64_t entries){
    const auto minimum=std::max<uint64_t>(8,mul(entries,2));uint64_t result=8;
    while(result<minimum){require(result<=UINT64_MAX/2,RAD_E_SHAPE,"authority hash-table overflow");result*=2;}
    require(result<=SIZE_MAX/sizeof(Bucket),RAD_E_SHAPE,"authority hash-table size overflow");return result;
}
uint64_t bytes_hash(const void* data,size_t size,uint64_t hash=14695981039346656037ull)noexcept{
    const auto* p=static_cast<const unsigned char*>(data);
    for(size_t i=0;i<size;++i){hash^=p[i];hash*=1099511628211ull;}return hash;
}
uint64_t string_hash(const std::string& value)noexcept{
#ifdef RAD_NATIVE_AUTHORITY_TEST_COLLISIONS
    (void)value;return 0;
#else
    return bytes_hash(value.data(),value.size());
#endif
}
uint64_t geometry_hash(const WeightInfo& w)noexcept{
#ifdef RAD_NATIVE_AUTHORITY_TEST_COLLISIONS
    (void)w;return 0;
#else
    // A hash is only an index accelerator. Every collision uses FULL equality.
    // Deliberately avoid representation/padding hashes of RadWeightDecl.
    uint64_t h=14695981039346656037ull;
    auto scalar=[&](const auto& v){h=bytes_hash(&v,sizeof v,h);};
    scalar(w.decl.dtype);scalar(w.decl.rank);scalar(w.decl.access);scalar(w.decl.shard);
    scalar(w.decl.optional);scalar(w.decl.n_row_parts);
    for(const auto v:w.decl.shape)scalar(v);for(const auto v:w.decl.row_parts)scalar(v);
    scalar(w.logical_bytes);scalar(w.shard_lo);scalar(w.shard_hi);scalar(w.n_part_spans);
    for(const auto v:w.part_lo)scalar(v);for(const auto v:w.part_hi)scalar(v);for(const auto v:w.part_total)scalar(v);
    scalar(w.enc_known);h=bytes_hash(&w.enc,sizeof w.enc,h);scalar(w.n_sel);
    for(const auto v:w.sel)scalar(v);for(const auto v:w.sel_rows)scalar(v);for(const auto v:w.sel_cols)scalar(v);
    h=bytes_hash(w.layout_tag.data(),w.layout_tag.size(),h);
    scalar(w.identity);scalar(w.widen);scalar(w.stored_dtype);scalar(w.stored_rank);
    for(const auto v:w.stored_shape)scalar(v);scalar(w.stored_bytes);scalar(w.align);return h;
#endif
}
const std::string& input_string(std::span<const WeightInfo> weights,uint32_t rep){
    return rep&1u?weights[rep/2].source:weights[rep/2].name;
}
struct Scan {
    std::unique_ptr<Bucket[]> geometry,strings;
    NativeAuthorityStats stats;
};
uint32_t geometry_id(Scan& scan,std::span<const WeightInfo> weights,uint32_t handle,bool insert){
    const auto hash=geometry_hash(weights[handle]);const auto mask=scan.stats.geometry_bucket_count-1;
    for(uint64_t step=0,slot=hash&mask;step<scan.stats.geometry_bucket_count;++step,slot=(slot+1)&mask){
        auto& entry=scan.geometry[size_t(slot)];
        if(entry.representative==UINT32_MAX){
            require(insert,RAD_E_STATE,"authority geometry changed between passes");
            require(scan.stats.geometries<UINT32_MAX,RAD_E_SHAPE,"authority geometry IDs exhausted");
            const auto id=uint32_t(scan.stats.geometries++);entry={hash,handle,id};
            scan.stats.layout_tag_bytes=add(scan.stats.layout_tag_bytes,weights[handle].layout_tag.size());return id;
        }
        if(entry.hash==hash&&native_weight_metadata_equal(weights[handle],weights[entry.representative],false))return entry.id;
    }
    throw Failure{RAD_E_SCRATCH,"authority geometry hash table full"};
}
uint32_t string_id(Scan& scan,std::span<const WeightInfo> weights,uint32_t representative,bool insert){
    const auto& text=input_string(weights,representative);const auto hash=string_hash(text);const auto mask=scan.stats.string_bucket_count-1;
    for(uint64_t step=0,slot=hash&mask;step<scan.stats.string_bucket_count;++step,slot=(slot+1)&mask){
        auto& entry=scan.strings[size_t(slot)];
        if(entry.representative==UINT32_MAX){
            require(insert,RAD_E_STATE,"authority string changed between passes");
            require(scan.stats.strings<UINT32_MAX,RAD_E_SHAPE,"authority string IDs exhausted");
            const auto id=uint32_t(scan.stats.strings++);entry={hash,representative,id};
            scan.stats.name_source_bytes=add(scan.stats.name_source_bytes,text.size());return id;
        }
        if(entry.hash==hash&&text==input_string(weights,entry.representative))return entry.id;
    }
    throw Failure{RAD_E_SCRATCH,"authority string hash table full"};
}
NativeAuthorityStats calculate(NativeAuthorityStats s){
    s.reference_bytes=mul(s.weights,sizeof(Ref));s.geometry_bytes=mul(s.geometries,sizeof(WeightInfo));
    s.slice_bytes=mul(add(s.strings,s.geometries),sizeof(Slice));
    s.arena_bytes=add(s.name_source_bytes,s.layout_tag_bytes);
    require(s.arena_bytes<=UINT32_MAX,RAD_E_SHAPE,"authority arena offset overflow");
    s.immutable_bytes=add(fixed_allowance,add(add(s.reference_bytes,s.geometry_bytes),add(s.slice_bytes,s.arena_bytes)));
    s.construction_peak_bytes=add(s.immutable_bytes,s.scratch_bytes);
    s.reference_size=sizeof(Ref);s.slice_size=sizeof(Slice);s.bucket_size=sizeof(Bucket);s.geometry_size=sizeof(WeightInfo);
    s.admissible=s.construction_peak_bytes<=s.metadata_budget;return s;
}
Scan scan_inputs(std::span<const WeightInfo> weights,std::span<const int32_t> unit_map,
                 std::span<const MoveUnit> units,uint64_t budget){
    require(!weights.empty()&&weights.size()<=262144&&unit_map.size()==weights.size()&&
        unit_map[0]==-1&&units.size()<=65536,RAD_E_SHAPE,"authority count/map cap");
    require(budget&&budget<=(uint64_t(128)<<20),RAD_E_INVAL,"authority metadata budget invalid");
    for(size_t i=0;i<weights.size();++i){const auto& w=weights[i];
        require(w.name.size()<=512&&w.source.size()<=512&&w.layout_tag.size()<=512,RAD_E_SHAPE,"authority string cap");
        if(!i)continue;
        const auto unit=unit_map[i];require(unit>=0&&size_t(unit)<units.size(),RAD_E_SHAPE,"authority expected unit outside map");
        const auto& members=units[size_t(unit)].weights;
        require(std::count(members.begin(),members.end(),rad_weight(i))==1,RAD_E_SHAPE,"authority exact member mapping");
    }
    Scan scan;auto& s=scan.stats;s.weights=weights.size();s.metadata_budget=budget;
    s.geometry_bucket_count=table_count(weights.size());s.string_bucket_count=table_count(mul(weights.size(),2));
    s.scratch_bytes=add(fixed_allowance,mul(add(s.geometry_bucket_count,s.string_bucket_count),sizeof(Bucket)));
    require(add(s.scratch_bytes,fixed_allowance)<=budget,RAD_E_SCRATCH,"authority scratch reservation cap before allocation");
    scan.geometry=std::make_unique<Bucket[]>(size_t(s.geometry_bucket_count));
    scan.strings=std::make_unique<Bucket[]>(size_t(s.string_bucket_count));
    for(uint32_t i=0;i<weights.size();++i){geometry_id(scan,weights,i,true);string_id(scan,weights,2*i,true);string_id(scan,weights,2*i+1,true);}
    s=calculate(s);return scan;
}
void explain(std::string* why,const char* message)noexcept{if(why)try{*why=message;}catch(...){}}
template<class Fn> int operation(std::string* why,Fn&& fn){
    try{fn();return RAD_OK;}catch(const Failure& f){explain(why,f.message);return f.status;}
    catch(const std::bad_alloc&){explain(why,"authority allocation failed");return RAD_E_NOMEM;}
    catch(...){explain(why,"authority construction exception");return RAD_E_STATE;}
}
} // namespace
bool native_weight_metadata_equal(const WeightInfo& a,const WeightInfo& b,bool names)noexcept{
    return (!names||(a.name==b.name&&a.source==b.source&&a.decl.group.layer==b.decl.group.layer&&
        a.decl.group.expert==b.decl.group.expert&&a.decl.group.slot==b.decl.group.slot))&&
        geometry_fields(a,b)&&a.layout_tag==b.layout_tag;
}
struct NativeWeightAuthority::Impl {
    NativeAuthorityStats stats;
    std::unique_ptr<Ref[]> refs;
    std::unique_ptr<WeightInfo[]> geometries;
    std::unique_ptr<Slice[]> strings,tags;
    std::unique_ptr<char[]> arena;
    bool equal(Slice slice,const std::string& text)const noexcept{
        if(text.size()!=slice.length)return false;
        return !slice.length||!std::memcmp(arena.get()+slice.offset,text.data(),slice.length);
    }
    bool matches(size_t index,const WeightInfo& w)const noexcept{
        if(index>=stats.weights)return false;const auto& r=refs[index];
        return r.geometry<stats.geometries&&r.name<stats.strings&&r.source<stats.strings&&
            r.layer==w.decl.group.layer&&r.expert==w.decl.group.expert&&r.slot==w.decl.group.slot&&
            equal(strings[r.name],w.name)&&equal(strings[r.source],w.source)&&
            geometry_fields(w,geometries[r.geometry])&&equal(tags[r.geometry],w.layout_tag);
    }
};
NativeWeightAuthority::NativeWeightAuthority(std::unique_ptr<Impl> p):p_(std::move(p)){}
NativeWeightAuthority::~NativeWeightAuthority()=default;
size_t NativeWeightAuthority::size()const noexcept{return size_t(p_->stats.weights);}
bool NativeWeightAuthority::matches(rad_weight w,const WeightInfo& live)const noexcept{return p_->matches(w,live);}
int32_t NativeWeightAuthority::unit_for_weight(rad_weight w)const noexcept{return w<p_->stats.weights?p_->refs[w].unit:-1;}
int32_t NativeWeightAuthority::member_index(rad_weight w)const noexcept{return w<p_->stats.weights?p_->refs[w].member:-1;}
int64_t NativeWeightAuthority::stored_bytes(rad_weight w)const noexcept{return w<p_->stats.weights?p_->geometries[p_->refs[w].geometry].stored_bytes:-1;}
const NativeAuthorityStats& NativeWeightAuthority::stats()const noexcept{return p_->stats;}
int NativeWeightAuthority::inspect(std::span<const WeightInfo> weights,std::span<const int32_t> map,
    std::span<const MoveUnit> units,uint64_t budget,NativeAuthorityStats* out,std::string* why){
    return operation(why,[&]{require(out,RAD_E_INVAL,"authority quote output null");const auto scan=scan_inputs(weights,map,units,budget);*out=scan.stats;});
}
int NativeWeightAuthority::capture(std::span<const WeightInfo> weights,std::span<const int32_t> map,
    std::span<const MoveUnit> units,uint64_t budget,std::unique_ptr<NativeWeightAuthority>* out,std::string* why){
    return operation(why,[&]{
        require(out,RAD_E_INVAL,"authority output null");auto scan=scan_inputs(weights,map,units,budget);
        require(scan.stats.admissible,RAD_E_SCRATCH,"authority immutable/construction reservation cap before allocation");
        auto p=std::make_unique<Impl>();p->stats=scan.stats;
        p->refs=std::make_unique<Ref[]>(size_t(p->stats.weights));
        p->geometries=std::make_unique<WeightInfo[]>(size_t(p->stats.geometries));
        p->strings=std::make_unique<Slice[]>(size_t(p->stats.strings));p->tags=std::make_unique<Slice[]>(size_t(p->stats.geometries));
        if(p->stats.arena_bytes)p->arena=std::make_unique<char[]>(size_t(p->stats.arena_bytes));
        uint64_t cursor=0;
        auto copy=[&](const std::string& text){
            require(cursor<=p->stats.arena_bytes&&text.size()<=p->stats.arena_bytes-cursor,RAD_E_STATE,"authority arena changed during copy");
            const Slice slice{uint32_t(cursor),uint32_t(text.size())};
            if(!text.empty())std::memcpy(p->arena.get()+cursor,text.data(),text.size());cursor+=text.size();return slice;
        };
        for(size_t i=0;i<scan.stats.string_bucket_count;++i){const auto& e=scan.strings[i];
            if(e.representative!=UINT32_MAX)p->strings[e.id]=copy(input_string(weights,e.representative));}
        for(size_t i=0;i<scan.stats.geometry_bucket_count;++i){const auto& e=scan.geometry[i];
            if(e.representative==UINT32_MAX)continue;
            copy_geometry(p->geometries[e.id],weights[e.representative]);p->tags[e.id]=copy(weights[e.representative].layout_tag);
        }
        const auto empty_capacity=std::string{}.capacity();p->stats.template_strings_empty=true;
        for(size_t i=0;i<p->stats.geometries;++i){const auto& g=p->geometries[i];
            for(const auto* text:{&g.name,&g.source,&g.layout_tag}){
                p->stats.template_strings_empty=p->stats.template_strings_empty&&text->empty();
                if(text->capacity()>empty_capacity)p->stats.template_string_capacity_excess=
                    add(p->stats.template_string_capacity_excess,text->capacity()-empty_capacity);
            }
            require(!g.decl.source&&!g.decl.planes&&!g.ptr&&!g.host_ptr,RAD_E_STATE,"authority template borrowed ignored pointers");
        }
        require(p->stats.template_strings_empty&&!p->stats.template_string_capacity_excess,
            RAD_E_STATE,"authority template retained ignored string storage");
        require(cursor==p->stats.arena_bytes,RAD_E_STATE,"authority exact arena extent changed");
        for(uint32_t w=0;w<weights.size();++w){const auto& live=weights[w];auto& ref=p->refs[w];
            ref.geometry=geometry_id(scan,weights,w,false);ref.name=string_id(scan,weights,2*w,false);ref.source=string_id(scan,weights,2*w+1,false);
            ref.layer=live.decl.group.layer;ref.expert=live.decl.group.expert;ref.slot=live.decl.group.slot;ref.unit=map[w];
            if(w){require(ref.unit>=0&&size_t(ref.unit)<units.size(),RAD_E_STATE,"authority unit changed during copy");
                const auto& members=units[size_t(ref.unit)].weights;const auto at=std::find(members.begin(),members.end(),rad_weight(w));
                require(at!=members.end()&&size_t(at-members.begin())<=INT32_MAX,RAD_E_STATE,"authority member changed during copy");ref.member=int32_t(at-members.begin());}
        }
        // Exclusive graph construction remains required. This recheck prevents
        // accidental incomplete snapshots; it does not legalize data races.
        for(size_t w=0;w<weights.size();++w){const auto& ref=p->refs[w];
            require(p->matches(w,weights[w])&&ref.unit==map[w],RAD_E_STATE,"authority current metadata changed before publication");
            if(w)require(ref.unit>=0&&size_t(ref.unit)<units.size()&&ref.member>=0&&size_t(ref.member)<units[size_t(ref.unit)].weights.size()&&
                units[size_t(ref.unit)].weights[size_t(ref.member)]==rad_weight(w)&&
                std::count(units[size_t(ref.unit)].weights.begin(),units[size_t(ref.unit)].weights.end(),rad_weight(w))==1,
                RAD_E_STATE,"authority current member changed before publication");
        }
        auto ready=std::unique_ptr<NativeWeightAuthority>(new NativeWeightAuthority(std::move(p)));
        *out=std::move(ready);
    });
}
} // namespace rad

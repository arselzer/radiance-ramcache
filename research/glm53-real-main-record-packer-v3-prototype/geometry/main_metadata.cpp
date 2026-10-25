#include "main_metadata.h"
#include "format/encoding.h"
#include "format/share.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>
#include <type_traits>
namespace glm53::main_metadata {
namespace {
constexpr int first_layer=3,layers=42,experts=288,world=4,records=layers*experts;
struct Failure {int status;const char* why;};
void need(bool value,int status,const char* why){if(!value)throw Failure{status,why};}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,RAD_E_SHAPE,"metadata addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,RAD_E_SHAPE,"metadata multiplication overflow");return a*b;}
uint64_t align(uint64_t n,uint64_t a){need(a&&!(a&(a-1)),RAD_E_SHAPE,"alignment is not power of two");return add(n,a-1)/a*a;}
void explain(std::string* why,const char* message)noexcept{if(why)try{*why=message;}catch(...){}}
std::string_view bounded(const char* p){need(p,RAD_E_INVAL,"null actual name-map string");
    const auto n=::strnlen(p,513);need(n&&n<=512,RAD_E_SHAPE,"actual name-map string exceeds512");return {p,n};}
void bounded(const std::string& s){need(!s.empty()&&s.size()<=512&&s.find('\0')==std::string::npos,RAD_E_SHAPE,"actual weight name/source is not bounded exact text");}
struct MapRef {std::string_view name;const RadNameMap* map;};
const RadNameMap& find_map(const std::vector<MapRef>& maps,const std::string& name){
    const auto i=std::lower_bound(maps.begin(),maps.end(),std::string_view(name),[](const MapRef& a,std::string_view b){return a.name<b;});
    need(i!=maps.end()&&i->name==name&&(i+1==maps.end()||(i+1)->name!=name),RAD_E_INVAL,"actual source name-map missing or duplicate");return *i->map;
}
std::array<char,160> raw_name(int layer,int expert,int projection){
    std::array<char,160> s{};const char* p=projection==0?"gate":projection==1?"up":"down";
    const int n=std::snprintf(s.data(),s.size(),"model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight",layer,expert,p);
    need(n>0&&size_t(n)<s.size(),RAD_E_SHAPE,"raw name bound");return s;
}
std::array<char,128> logical_name(int layer,int expert,int member){
    std::array<char,128> s{};const int n=std::snprintf(s.data(),s.size(),"blk.%d.ffn_%s_exps.%d.%s",layer,member<2?"gate_up":"down",expert,member%2?"scale":"weight");
    need(n>0&&size_t(n)<s.size(),RAD_E_SHAPE,"logical name bound");return s;
}
void check_raw(rad::Checkpoint& ck,const rad::SrcTensor* code,const rad::SrcTensor* scale,
               const char* name,int projection){
    const std::string scale_name=std::string(name)+"_scale";
    const auto* raw=ck.find(name);const auto* raw_scale=ck.find(scale_name);
    need(raw&&raw_scale&&raw->name==name&&raw_scale->name==scale_name&&raw->data&&raw_scale->data,
         RAD_E_INVAL,"actual opened raw source identity/span missing");
    const int64_t rows=projection==2?4096:2048,cols=projection==2?1024:2048;
    need(raw->rad_dtype==RAD_U8&&raw_scale->rad_dtype==RAD_U8&&raw->shape==std::vector<int64_t>({rows,cols})&&
         raw_scale->shape==std::vector<int64_t>({rows,cols/16})&&raw->bytes==rows*cols&&raw_scale->bytes==rows*cols/16,
         RAD_E_SHAPE,"actual raw Quark code/scale shape/type/bytes");
    need(code&&scale&&code->data==raw->data&&scale->data==raw_scale->data,
         RAD_E_INVAL,"foreign cached source does not alias the actual opened raw spans");
    const rad::SrcTensor* actual_code=nullptr;const rad::SrcTensor* actual_scale=nullptr;
    const int imported=ck.import_piece(name,&actual_code,&actual_scale,nullptr);
    need(imported==RAD_OK&&actual_code==code&&actual_scale==scale,
         RAD_E_INVAL,"resolved cache uses foreign/stale imported source objects");
    need(code&&scale&&code->name==raw->name&&scale->name==raw_scale->name&&
         code->data==raw->data&&scale->data==raw_scale->data&&code->bytes==raw->bytes&&scale->bytes==raw_scale->bytes&&
         code->rad_dtype==RAD_FP4E2M1&&scale->rad_dtype==RAD_E8M0&&
         code->shape==std::vector<int64_t>({rows,cols*2})&&scale->shape==raw_scale->shape,
         RAD_E_SHAPE,"same opened Quark import object exact names/planes/spans");
}
void source(rad::Checkpoint& ck,const rad::CheckpointSources& cache,const std::vector<MapRef>& maps,
            const rad::WeightInfo& w,int layer,int expert,int member,const rad::CkptWeight** output){
    const auto expected=logical_name(layer,expert,member);
    const auto logical=logical_name(layer,expert,member<2?0:2);
    need(w.name==expected.data()&&w.source==logical.data(),RAD_E_INVAL,"actual stock main member name/source contract");
    const auto& map=find_map(maps,w.source);const auto* resolved=cache.find(w.source);
    const auto mx=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
    const bool gu=member<2;const int parts=gu?2:1;
    need(map.mode==(gu?RAD_MAP_CONCAT:RAD_MAP_COPY)&&map.n_src==parts&&map.concat_dim==0&&
         resolved&&resolved->name==w.source&&resolved->trivial&&!resolved->imported&&!resolved->paro&&!resolved->wnorm&&
         !resolved->block_importer&&resolved->derived.empty()&&resolved->wnorm_scale.empty()&&
         resolved->pieces.size()==size_t(parts)&&resolved->scales.size()==size_t(parts)&&
         resolved->concat_dim==0&&resolved->rank==2&&resolved->shape[0]==4096&&resolved->shape[1]==(gu?4096:2048)&&
         resolved->rows==4096&&resolved->cols==(gu?4096:2048)&&rad_enc_equal(&resolved->enc,&mx),
         RAD_E_SHAPE,"actual logical source map/order/encoding/shape");
    for(int piece=0;piece<parts;++piece){const auto name=raw_name(layer,expert,gu?piece:2);
        need(bounded(map.src[piece])==name.data()&&map.src_index[piece]==-1,
             RAD_E_INVAL,"actual logical map does not name full ordered gate/up/down pieces");
        check_raw(ck,resolved->pieces[size_t(piece)],resolved->scales[size_t(piece)],name.data(),gu?piece:2);
    }
    *output=resolved;
}
RecordLayout inspect(rad::Checkpoint& ck,const rad::CheckpointSources& cache,const rad::Program& p,
                     const rad::Plan& plan,const std::vector<MapRef>& maps,int32_t unit){
    need(unit>=0&&size_t(unit)<plan.units.size(),RAD_E_INVAL,"actual main unit outside Plan");const auto& u=plan.units[size_t(unit)];
    need(u.weights.size()==4&&u.offset.size()==4&&u.access==RAD_ACCESS_CONDITIONAL&&u.layer>=3&&u.layer<=44&&u.expert>=0&&u.expert<experts,
         RAD_E_SHAPE,"actual main unit four-member/group/access contract");
    RecordLayout row;row.world=world;row.rank=p.ctx.rank;row.unit=unit;row.layer=u.layer;row.expert=u.expert;row.stride=u.slot_bytes;
    uint64_t end=0;const auto mx=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
    constexpr int64_t shape[4][2]{{1024,4096},{1024,128},{4096,512},{4096,16}};
    for(size_t m=0;m<4;++m){const auto h=u.weights[m];need(h&&h<p.weights.size()&&plan.unit_of_weight[h]==unit,RAD_E_INVAL,"actual main member unit map");
        const auto& w=p.weights[h];bounded(w.name);bounded(w.source);
        need(w.enc_known&&rad_enc_equal(&w.enc,&mx)&&w.n_sel==1&&w.sel[0]==int(m%2)&&
             w.decl.group.layer==u.layer&&w.decl.group.expert==u.expert&&w.decl.group.slot==int(m/2)&&
             w.decl.access==RAD_ACCESS_CONDITIONAL&&!w.decl.optional&&w.decl.shard==(m<2?RAD_SHARD_ROW:RAD_SHARD_COL)&&
             w.decl.rank==2&&w.decl.shape[0]==shape[m][0]&&w.decl.shape[1]==shape[m][1]&&
             w.decl.dtype==(m%2?RAD_E8M0:RAD_FP4E2M1),RAD_E_SHAPE,"actual main declared planes/group/shard/shape/type");
        need(w.identity&&!w.widen&&w.layout_tag.empty()&&w.stored_rank==2&&w.stored_dtype==w.decl.dtype&&
             w.stored_shape[0]==shape[m][0]&&w.stored_shape[1]==shape[m][1]&&w.sel_rows[0]==shape[m][0]&&w.sel_cols[0]==shape[m][1],
             RAD_E_UNSUPPORTED,"selected main layout is not canonical identity rank2 (no relayout claim)");
        for(size_t d=2;d<RAD_MAX_RANK;++d)need(!w.decl.shape[d]&&!w.stored_shape[d],RAD_E_SHAPE,"main rank2 trailing shape");
        need(w.align>0&&w.align<=4096&&4096%w.align==0&&w.stored_bytes>0&&
             uint64_t(w.stored_bytes)==uint64_t(shape[m][0])*uint64_t(rad_enc_row_bytes(w.stored_dtype,shape[m][1])),
             RAD_E_SHAPE,"actual main canonical byte/alignment contract");
        need(w.n_part_spans==0&&w.shard_lo==(int64_t(p.ctx.rank)*512)/(m==3?32:1)&&
             w.shard_hi==(int64_t(p.ctx.rank+1)*512)/(m==3?32:1),RAD_E_SHAPE,"actual stock TP4 shard span");
        need(w.decl.n_row_parts==(m<2?2:0),RAD_E_SHAPE,"actual main GU row-parts count");
        if(m<2)need(w.decl.row_parts[0]==512&&w.decl.row_parts[1]==512,RAD_E_SHAPE,"actual main gate/up equal local halves");
        for(size_t j=m<2?2:0;j<RAD_MAX_ROW_PARTS;++j)need(!w.decl.row_parts[j],RAD_E_SHAPE,"unexpected actual main row-part tail");
        const rad::CkptWeight* resolved=nullptr;source(ck,cache,maps,w,u.layer,u.expert,int(m),&resolved);
        rad::PlaneShare shares[RAD_ENC_MAX_PLANES];
        const int shared=rad::weight_share(w,resolved->enc,resolved->rank,resolved->shape,world,p.ctx.rank,shares,nullptr);
        need(shared==RAD_OK,RAD_E_SHAPE,"actual weight_share refused main source/shard");const auto& s=shares[0];
        need(s.dtype==w.stored_dtype&&s.rows==shape[m][0]&&s.cols==shape[m][1]&&s.dense_bytes()==w.stored_bytes&&s.rects.size()==(m<2?2u:1u),
             RAD_E_SHAPE,"actual main source rectangles do not cover canonical member");
        auto& member=row.members[m];member.weight=h;member.plane=m%2;member.slices=s.rects.size();
        for(size_t part=0;part<s.rects.size();++part){const auto& q=s.rects[part];auto& physical=member.physical[part];
            const bool gu=m<2;const int projection=gu?int(part):2;const auto* raw=ck.find(raw_name(u.layer,u.expert,projection).data());
            const int64_t expected_row0=gu?int64_t(part)*2048+int64_t(p.ctx.rank)*512:0;
            const int64_t expected_byte0=gu?0:int64_t(p.ctx.rank)*(m%2?16:256);
            const int64_t expected_rows=gu?512:4096,expected_row_bytes=gu?(m%2?128:2048):(m%2?64:1024);
            need(q.row0==expected_row0&&q.rows==expected_rows&&q.byte0==expected_byte0&&q.bytes==(gu?expected_row_bytes:(m%2?16:256))&&s.row_bytes==expected_row_bytes,
                 RAD_E_SHAPE,"actual weight_share differs from ordered TP4 gate/up or DOWN rectangle");
            if(m%2){const auto* scale=ck.find(std::string(raw->name)+"_scale");need(scale,RAD_E_INVAL,"opened scale vanished");raw=scale;}
            physical.projection=Projection(projection);physical.plane=m%2;physical.raw_dtype=raw->rad_dtype;
            physical.raw_shape={raw->shape[0],raw->shape[1]};physical.raw_bytes=raw->bytes;
            physical.row0=q.row0-(gu?int64_t(part)*2048:0);physical.rows=q.rows;physical.byte0=q.byte0;
            physical.row_bytes=s.row_bytes;physical.bytes_per_row=q.bytes;
            need(physical.row0>=0&&physical.row0<=raw->shape[0]&&q.rows<=raw->shape[0]-physical.row0&&q.byte0>=0&&q.bytes<=raw->shape[1]-q.byte0&&
                 raw->bytes==raw->shape[0]*s.row_bytes,RAD_E_SHAPE,"physical raw source span outside its own tensor");
        }
        const auto at=align(end,uint64_t(w.align));need(u.offset[m]>=0&&uint64_t(u.offset[m])==at,RAD_E_SHAPE,"actual main member offset disagrees with alignment");
        end=add(at,uint64_t(w.stored_bytes));row.offsets[m]=u.offset[m];row.bytes[m]=w.stored_bytes;row.dtype[m]=w.stored_dtype;row.shape[m]={w.stored_shape[0],w.stored_shape[1]};
    }
    need(u.bytes>0&&end==uint64_t(u.bytes)&&u.slot_bytes>0&&uint64_t(u.slot_bytes)==align(end,4096)&&
         u.slab_class>=0&&size_t(u.slab_class)<plan.classes.size()&&plan.classes[size_t(u.slab_class)].stride==u.slot_bytes,
         RAD_E_SHAPE,"actual main unit/class bytes/stride");
    return row;
}
void selected_device_layouts(const rad::Program& p,const rad::Plan& plan,std::vector<uint8_t>& claimed){
    uint64_t references=0,bands=0;
    need(p.ops.size()<=50000,RAD_E_SHAPE,"actual op metadata count cap");
    for(const auto& op:p.ops){references=add(references,op.weights.size());bands=add(bands,op.bands.size());}
    need(references<=1000000&&bands<=100000,RAD_E_SHAPE,"actual op weight/band metadata work cap");
    for(const auto& op:p.ops)for(size_t wi=0;wi<op.weights.size();++wi){const auto h=op.weights[wi];
        if(!h)continue;need(h<p.weights.size(),RAD_E_INVAL,"actual op references outside Program weights");const auto& w=p.weights[h];
        if(w.decl.group.expert<0)continue;
        need(plan.unit_of_weight[h]>=0&&!op.bands.empty(),RAD_E_SHAPE,"main member has no actual selected bands");
        const int operand=wi<op.weight_opd.size()&&op.weight_opd[wi]>=0?op.weight_opd[wi]:int(wi);
        for(const auto& band:op.bands){const auto& r=band.dom[RAD_DOMAIN_DEVICE];
            need(r.row&&r.row->info&&!r.instance,RAD_E_UNSUPPORTED,"requires actual selected, uninitialized DEVICE main rows");
            RadTensor planes[RAD_ENC_MAX_PLANES]{};rad::weight_planes(w,planes);RadLayout layout{};
            const auto* info=r.row->info;const int status=info->layout?
                info->layout(r.geom.params(),r.geom.n_params(),operand,&w.enc,w.sel,planes,w.n_sel,&layout):RAD_E_UNSUPPORTED;
            if(status==RAD_LAYOUT_ANY)continue;
            need(status==RAD_E_UNSUPPORTED,RAD_E_UNSUPPORTED,"actual selected DEVICE main kernel requires relayout or refused layout");
            claimed[h]=1;
        }
    }
    for(size_t h=1;h<p.weights.size();++h)if(p.weights[h].decl.group.expert>=0)
        need(claimed[h],RAD_E_UNSUPPORTED,"main member lacks actual canonical DEVICE layout claim");
}
} // namespace
int quote(rad::Checkpoint& ck,const rad::CheckpointSources& cache,const rad::Program& p,const rad::Plan& plan,
          std::span<const int32_t> residents,const Limits& limits,Quote* out,std::string* why)noexcept{
    static_assert(std::is_trivially_copyable_v<RecordLayout>&&std::is_standard_layout_v<RecordLayout>);
    static_assert(std::is_trivially_copyable_v<OwnedLayout>);
    // Expected matching 64-bit Core ABI. Parent compilation establishes these
    // actual POD sizes; the runtime receipt reports sizeof, never a private mirror.
    static_assert(sizeof(PhysicalSlice)==80&&sizeof(MemberLayout)==176&&sizeof(RecordLayout)==880&&sizeof(OwnedLayout)==8);
    try{
        need(out,RAD_E_INVAL,"null metadata quote output");
        need(p.ctx.world_size==world&&p.ctx.rank>=0&&p.ctx.rank<world&&plan.world_size==world&&plan.rank==p.ctx.rank&&
             p.ctx.max_spec==0&&!plan.dynamic&&p.meta.n_layers==45&&p.meta.n_embd==4096&&p.meta.n_expert==experts&&p.meta.n_expert_used==8&&
             ck.meta().arch=="glm5next"&&ck.imports()==rad::Checkpoint::Import::QuarkMx,
             RAD_E_UNSUPPORTED,"requires actual stock main GLM TP4/max_spec0 profile");
        need(limits.max_metadata_bytes&&limits.max_metadata_bytes<=(128u<<20)&&limits.max_program_weights&&limits.max_program_weights<=262144&&
             limits.max_program_units&&limits.max_program_units<=65536&&limits.max_name_maps&&limits.max_name_maps<=262144&&
             p.weights.size()>1&&p.weights.size()<=limits.max_program_weights&&plan.units.size()<=limits.max_program_units&&
             p.name_map.size()<=limits.max_name_maps&&plan.unit_of_weight.size()==p.weights.size()&&plan.unit_of_weight[0]==-1&&residents.size()<=records,
             RAD_E_SHAPE,"actual stock metadata counts/caps/unit map");
        std::array<int32_t,records> unit_of{};unit_of.fill(-1);std::array<uint8_t,records> resident{};
        size_t count=0;
        for(size_t u=0;u<plan.units.size();++u){const auto& unit=plan.units[u];if(unit.expert<0)continue;
            need(unit.layer>=3&&unit.layer<=44&&unit.expert<experts,RAD_E_UNSUPPORTED,"non-main/MTP expert in max_spec0 input");
            need(unit.weights.size()==4&&unit.offset.size()==4,RAD_E_SHAPE,"actual main unit must have four bounded members");
            const size_t k=size_t(unit.layer-first_layer)*experts+size_t(unit.expert);
            need(unit_of[k]<0,RAD_E_INVAL,"duplicate actual main layer/expert unit");unit_of[k]=int32_t(u);++count;
        }
        need(count==records,RAD_E_SHAPE,"actual main GLM requires42x288 distinct expert units");
        for(const auto u:residents){need(u>=0&&size_t(u)<plan.units.size(),RAD_E_INVAL,"resident main unit outside Plan");const auto& unit=plan.units[size_t(u)];
            need(unit.layer>=3&&unit.layer<=44&&unit.expert>=0&&unit.expert<experts,RAD_E_INVAL,"resident is not actual main expert");
            const size_t k=size_t(unit.layer-3)*experts+size_t(unit.expert);need(unit_of[k]==u&&!resident[k],RAD_E_INVAL,"duplicate/foreign resident mask");resident[k]=1;
        }
        // No full WeightInfo copy: validate coverage by existing handles and four-member units.
        for(size_t h=1;h<p.weights.size();++h){const auto& w=p.weights[h];if(w.decl.group.expert<0)continue;
            need(w.decl.group.layer>=3&&w.decl.group.layer<=44&&w.decl.group.expert<experts,RAD_E_SHAPE,"main expert weight group outside closed model");
            const auto u=unit_of[size_t(w.decl.group.layer-3)*experts+size_t(w.decl.group.expert)];
            need(plan.unit_of_weight[h]==u&&std::count(plan.units[size_t(u)].weights.begin(),plan.units[size_t(u)].weights.end(),rad_weight(h))==1,
                 RAD_E_INVAL,"main Program weight missing/duplicated in actual Plan");
        }
        for(const auto& m:p.name_map)(void)bounded(m.declared);
        const uint64_t immutable=add(sizeof(Quote),add(mul(records,sizeof(RecordLayout)),add(mul(records-residents.size(),sizeof(OwnedLayout)),mul(residents.size(),sizeof(int32_t)))));
        const uint64_t scratch=add(sizeof(unit_of)+sizeof(resident)+sizeof(RecordLayout)+sizeof(rad::PlaneShare)*RAD_ENC_MAX_PLANES+4096,
                                   add(mul(p.name_map.size(),sizeof(MapRef)),p.weights.size()));
        const uint64_t peak=add(immutable,scratch);need(peak<=limits.max_metadata_bytes,RAD_E_SCRATCH,"metadata budget refused BEFORE exact arrays/source inspection");
        std::vector<MapRef> maps;maps.reserve(p.name_map.size());for(const auto& m:p.name_map)maps.push_back({bounded(m.declared),&m});
        std::vector<uint8_t> claimed(p.weights.size(),0);
        std::sort(maps.begin(),maps.end(),[](const MapRef& a,const MapRef& b){return a.name<b.name;});
        // First pass proves every real source/layout/member/span without output
        // arrays. Second pass fills exact-size POD arrays; caller is untouched
        // on any OOM, mutation or unsupported layout in either pass.
        for(const auto unit:unit_of)(void)inspect(ck,cache,p,plan,maps,unit);
        selected_device_layouts(p,plan,claimed);
        Quote q;q.world=world;q.rank=p.ctx.rank;q.input_weights=p.weights.size();q.input_units=plan.units.size();q.input_name_maps=p.name_map.size();q.main_units=records;
        q.immutable_requested=immutable;q.scratch_requested=scratch;q.construction_requested=peak;
        q.sizeof_record=sizeof(RecordLayout);q.sizeof_member=sizeof(MemberLayout);q.sizeof_slice=sizeof(PhysicalSlice);q.sizeof_owned=sizeof(OwnedLayout);
        q.records.resize(records);q.owned.resize(records-residents.size());q.resident_main_units.resize(residents.size());size_t owned=0,static_units=0;
        for(size_t k=0;k<records;++k){auto row=inspect(ck,cache,p,plan,maps,unit_of[k]);
            for(size_t m=0;m<4;++m){q.stored_bytes=add(q.stored_bytes,uint64_t(row.bytes[m]));q.physical_rectangles=add(q.physical_rectangles,row.members[m].slices);
                for(size_t s=0;s<row.members[m].slices;++s){const auto& a=row.members[m].physical[s];q.source_selected_bytes=add(q.source_selected_bytes,mul(uint64_t(a.rows),uint64_t(a.bytes_per_row)));}}
            if(resident[k])q.resident_main_units[static_units++]=row.unit;else q.owned[owned++]={row.unit,uint32_t(k)};q.records[k]=row;
        }
        need(owned==q.owned.size()&&static_units==q.resident_main_units.size()&&q.stored_bytes==q.source_selected_bytes,RAD_E_STATE,"metadata final coverage/count/byte check");
        *out=std::move(q);return RAD_OK;
    }catch(const Failure& f){explain(why,f.why);return f.status;}
    catch(const std::bad_alloc&){explain(why,"metadata allocation refused; caller preserved");return RAD_E_NOMEM;}
    catch(...){explain(why,"metadata quote exception; caller preserved");return RAD_E_STATE;}
}
} // namespace glm53::main_metadata

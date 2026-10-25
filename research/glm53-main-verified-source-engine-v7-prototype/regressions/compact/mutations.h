#pragma once
#include "rad_core.h"
#include <functional>
#include <string>
#include <vector>
namespace compact_tests {
struct Mutation {std::string label;std::function<void(rad::WeightInfo&)> apply;bool guarded=true;};
inline std::vector<Mutation> mutations(){
    using rad::WeightInfo;std::vector<Mutation> out;
    auto put=[&](std::string label,std::function<void(WeightInfo&)> f,bool guarded=true){out.push_back({std::move(label),std::move(f),guarded});};
    put("name last byte",[](auto& w){if(w.name.empty())w.name="x";else w.name.back()^=1;});
    put("name embedded NUL suffix",[](auto& w){w.name+=std::string("\0suffix",7);});
    put("source last byte",[](auto& w){if(w.source.empty())w.source="x";else w.source.back()^=1;});
    put("source equal prefix",[](auto& w){if(w.source.empty())w.source="x";else w.source.pop_back();});
    put("group.layer",[](auto& w){++w.decl.group.layer;});put("group.expert",[](auto& w){++w.decl.group.expert;});put("group.slot",[](auto& w){++w.decl.group.slot;});
    put("decl.dtype",[](auto& w){++w.decl.dtype;});put("decl.rank",[](auto& w){++w.decl.rank;});
    put("decl.access",[](auto& w){++w.decl.access;});put("decl.shard",[](auto& w){++w.decl.shard;});
    put("decl.optional",[](auto& w){++w.decl.optional;});put("decl.n_row_parts",[](auto& w){++w.decl.n_row_parts;});
    for(size_t i=0;i<RAD_MAX_RANK;++i){put("decl.shape"+std::to_string(i),[i](auto& w){++w.decl.shape[i];});put("stored_shape"+std::to_string(i),[i](auto& w){++w.stored_shape[i];});}
    for(size_t i=0;i<RAD_MAX_ROW_PARTS;++i){put("row_parts"+std::to_string(i),[i](auto& w){++w.decl.row_parts[i];});put("part_lo"+std::to_string(i),[i](auto& w){++w.part_lo[i];});put("part_hi"+std::to_string(i),[i](auto& w){++w.part_hi[i];});put("part_total"+std::to_string(i),[i](auto& w){++w.part_total[i];});}
    put("logical_bytes",[](auto& w){++w.logical_bytes;});put("shard_lo",[](auto& w){++w.shard_lo;});put("shard_hi",[](auto& w){++w.shard_hi;});put("n_part_spans",[](auto& w){++w.n_part_spans;});
    put("enc_known",[](auto& w){w.enc_known=!w.enc_known;});put("n_sel",[](auto& w){++w.n_sel;});
    for(size_t i=0;i<RAD_ENC_MAX_PLANES;++i){put("sel"+std::to_string(i),[i](auto& w){++w.sel[i];});put("sel_rows"+std::to_string(i),[i](auto& w){++w.sel_rows[i];});put("sel_cols"+std::to_string(i),[i](auto& w){++w.sel_cols[i];});}
    for(size_t i=0;i<sizeof(RadEncoding);++i)put("encoding byte"+std::to_string(i),[i](auto& w){reinterpret_cast<unsigned char*>(&w.enc)[i]^=1;});
    put("layout tag same length",[](auto& w){if(w.layout_tag.empty())w.layout_tag="x";else w.layout_tag.back()^=1;});
    put("layout tag embedded suffix",[](auto& w){w.layout_tag+=std::string("\0tail",5);});
    put("identity",[](auto& w){w.identity=!w.identity;});put("widen",[](auto& w){w.widen=!w.widen;});
    put("stored_dtype",[](auto& w){++w.stored_dtype;});put("stored_rank",[](auto& w){++w.stored_rank;});
    put("stored_bytes",[](auto& w){++w.stored_bytes;});put("align",[](auto& w){++w.align;});
    put("ignored decl.source",[](auto& w){w.decl.source=reinterpret_cast<const char*>(uintptr_t(1));},false);
    put("ignored decl.planes",[](auto& w){w.decl.planes=reinterpret_cast<const char*>(uintptr_t(3));},false);
    put("ignored ptr",[](auto& w){w.ptr=reinterpret_cast<void*>(uintptr_t(5));},false);
    put("ignored host_ptr",[](auto& w){w.host_ptr=reinterpret_cast<void*>(uintptr_t(7));},false);
    put("ignored file_offset",[](auto& w){++w.file_offset;},false);
    put("ignored site",[](auto& w){w.site=rad::Site::Host;},false);
    put("ignored tier",[](auto& w){w.tier=rad::Tier::Mapped;},false);
    put("ignored slab_slot",[](auto& w){w.slab_slot=42;},false);
    put("ignored lay_op",[](auto& w){w.lay_op=73;},false);put("ignored lay_band",[](auto& w){w.lay_band=19;},false);
    put("ignored lay_dom",[](auto& w){w.lay_dom=7;},false);put("ignored lay_operand",[](auto& w){w.lay_operand=5;},false);
    put("ignored first_use",[](auto& w){w.first_use_op=88;},false);put("ignored last_use",[](auto& w){w.last_use_op=99;},false);
    return out;
}
}

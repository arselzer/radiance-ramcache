#pragma once
// Frozen original native_inventory.cpp predicate, independently retained before
// the compact implementation. Source SHA is recorded in SOURCE_RECEIPT.json.
#include "rad_core.h"
#include <algorithm>
namespace compact_baseline {
template<class T,size_t N>bool same(const T(&a)[N],const T(&b)[N]){return std::equal(a,a+N,b);}
inline bool metadata(const rad::WeightInfo& a,const rad::WeightInfo& b,bool names){
    const auto& x=a.decl;const auto& y=b.decl;
    return (!names||(a.name==b.name&&a.source==b.source&&
        x.group.layer==y.group.layer&&x.group.expert==y.group.expert&&x.group.slot==y.group.slot))&&
        x.dtype==y.dtype&&x.rank==y.rank&&same(x.shape,y.shape)&&x.access==y.access&&
        x.shard==y.shard&&x.optional==y.optional&&x.n_row_parts==y.n_row_parts&&
        same(x.row_parts,y.row_parts)&&a.logical_bytes==b.logical_bytes&&
        a.shard_lo==b.shard_lo&&a.shard_hi==b.shard_hi&&a.n_part_spans==b.n_part_spans&&
        same(a.part_lo,b.part_lo)&&same(a.part_hi,b.part_hi)&&same(a.part_total,b.part_total)&&
        a.enc_known==b.enc_known&&rad_enc_equal(&a.enc,&b.enc)&&a.n_sel==b.n_sel&&
        same(a.sel,b.sel)&&same(a.sel_rows,b.sel_rows)&&same(a.sel_cols,b.sel_cols)&&
        a.layout_tag==b.layout_tag&&a.identity==b.identity&&a.widen==b.widen&&
        a.stored_dtype==b.stored_dtype&&a.stored_rank==b.stored_rank&&
        same(a.stored_shape,b.stored_shape)&&a.stored_bytes==b.stored_bytes&&a.align==b.align;
}
}

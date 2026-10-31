#include "ds41_phase2_audit.h"
#include "../rad_internal.h"
#include "../sched/batch.h"
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

namespace rad { namespace ds41_phase2 { namespace audit {
const char* prefix(){static const char*p=std::getenv(kEnv);return p;}
bool requested(){const char*p=prefix();return p&&*p&&std::strcmp(p,"0");}
bool prefix_valid(const char*p){
    if(!p||!*p||!std::strcmp(p,"0"))return true;
    const size_t n=std::strlen(p);if(p[0]!='/'||n>1024)return false;
    for(size_t i=0;i<n;++i)if((unsigned char)p[i]<32||p[i]=='"'||p[i]=='\\')return false;
    return true;
}
Identity make_identity(const RadBatch*b,const StepEntry*entries,std::size_t n,const uint64_t*seqs,std::size_t ns,bool generated){
    Identity id;
    if(generated||!b||b->phase!=RAD_PHASE_PREFILL||b->draft_pass||b->enc||b->n_seq!=1||b->n_seq_decode||
       b->n_tok_decode||b->n_spec||!entries||n!=1||!seqs||ns!=1||b->n_tok<1||b->n_tok>4096)return id;
    const StepEntry&e=entries[0];
    if(!e.is_prefill||e.dev||e.n_spec||e.n_tokens!=b->n_tok||e.seq!=seqs[0]||e.card<0||e.ctx_len<0||
       e.prompt_end<int64_t(e.ctx_len)+e.n_tokens||e.prompt_end>INT32_MAX)return id;
    id.valid=true;id.step=b->step;id.card=e.card;id.sequence=e.seq;id.begin=e.ctx_len;id.rows=e.n_tokens;id.prompt_end=e.prompt_end;
    return id;
}
Trace::Trace(FILE*f,int rank,uint64_t program,uint64_t plan,Limits limits):fp_(f),rank_(rank),program_(program),plan_(plan),limits_(limits){
    if(!f||rank<0||rank>=4||!program||!plan||!limits.bytes||limits.bytes>kMaxBytes||limits.frames<1||
       limits.frames>kMaxFrames||limits.spans<1||limits.spans>kMaxSpans){incomplete("invalid_trace_binding");return;}
    out("{\"type\":\"header\",\"schema\":\"ds41-p2-audit-v1\",\"rank\":%d,\"program\":%llu,\"plan\":%llu,"
        "\"max_bytes\":%llu,\"max_frames\":%d,\"max_spans\":%d,\"performance_eligible\":false}\n",
        rank_,(unsigned long long)program_,(unsigned long long)plan_,(unsigned long long)limits_.bytes,limits_.frames,limits_.spans);
}
Trace::~Trace(){close();}
void Trace::out(const char*fmt,...){
    if(failed_||!fp_)return;
    va_list a;va_start(a,fmt);const int n=std::vfprintf(fp_,fmt,a);va_end(a);
    if(n<0){incomplete("output_io_error");return;}bytes_+=(uint64_t)n;
}
bool Trace::room(uint64_t n){if(failed_)return false;if(n>limits_.bytes||bytes_>limits_.bytes-n){incomplete("output_cap");return false;}return true;}
void Trace::incomplete(const char*why){
    if(failed_)return;
    // Constant reason strings only; reserve closure space before accepting every frame.
    if(fp_)std::fprintf(fp_,"{\"type\":\"incomplete\",\"rank\":%d,\"step\":%d,\"seq\":%llu,\"epoch\":%llu,\"reason\":\"%s\"}\n",
        rank_,piece_.step,(unsigned long long)piece_.sequence,(unsigned long long)epoch_,why);
    RAD_WARN("DSP2AUDIT incomplete rank=%d step=%d seq=%llu epoch=%llu reason=%s performance_eligible=0",
        rank_,piece_.step,(unsigned long long)piece_.sequence,(unsigned long long)epoch_,why);
    failed_=true;span_ok_=false;
}
void Trace::common(const char*type){
    out("{\"type\":\"%s\",\"rank\":%d,\"program\":%llu,\"plan\":%llu,\"step\":%d,\"seq\":%llu,"
        "\"span_epoch\":%llu,\"epoch\":%llu,\"piece\":%d,\"begin\":%lld,\"query_rows\":%d,\"card\":%d",
        type,rank_,(unsigned long long)program_,(unsigned long long)plan_,piece_.step,(unsigned long long)piece_.sequence,
        (unsigned long long)span_epoch_,(unsigned long long)epoch_,pieces_-1,(long long)piece_.begin,piece_.rows,piece_.card);
}
void Trace::begin(int32_t step,int32_t rows,bool p1,const int32_t*row0,int32_t full){
    if(failed_)return;
    if(piece_on_){incomplete("unclosed_piece");return;}
    const Identity id=next_;next_={};
    const bool valid=id.valid&&id.step==step&&id.rows==rows&&id.card>=0&&id.begin>=0&&rows>0&&rows<=4096&&
        id.prompt_end>=id.begin+rows&&id.prompt_end<=INT32_MAX;
    if(!valid&&p1){piece_=id;piece_.step=step;epoch_=0;incomplete("unbound_p1_identity");return;}
    if(!valid||!p1){
        if(span_on_)summary(valid?"non_p1_boundary":"unbound_step_boundary");
        if(room(1024))out("{\"type\":\"excluded\",\"rank\":%d,\"step\":%d,\"identity_valid\":%s,\"seq\":%llu,\"begin\":%lld,"
            "\"query_rows\":%d,\"prompt_end\":%lld,\"reason\":\"%s\"}\n",rank_,step,valid?"true":"false",
            (unsigned long long)id.sequence,(long long)id.begin,rows,(long long)id.prompt_end,valid?"non_p1_pass":"unbound_non_native_step");
        return;
    }
    if(span_on_&&id.sequence==piece_.sequence&&id.card!=piece_.card){incomplete("sequence_card_changed");return;}
    if(span_on_&&(id.sequence!=piece_.sequence||id.begin!=span_end_||id.prompt_end!=piece_.prompt_end||
        pieces_==routes::kMaxPieces||id.begin+rows-span_begin_>routes::kMaxSpanRows))summary("scope_boundary");
    if(!span_on_){
        if(spans_>=limits_.spans){piece_=id;epoch_=0;incomplete("span_cap");return;}
        ++spans_;span_on_=true;span_ok_=true;span_begin_=id.begin;span_end_=id.begin;
        pieces_=0;span_epoch_=0;seen_={};totals_={};
    }
    piece_=id;piece_on_=true;++pieces_;epoch_=0;current_layer_=-1;
    frame_seen_={};p1_layer_={};done_={};copied_={};
    if(full<0||full>routes::kLayers){incomplete("invalid_consumer_full");return;}
    full_=full;
    for(int l=0;l<routes::kLayers;++l){row0_[l]=row0?row0[l]:0;
        if(row0_[l]<0||row0_[l]>rows){incomplete("invalid_consumer_cut");return;}}
}
void Trace::frame(uint64_t generation,uint64_t serial,int layer,int from,const routes::View&v,const UnitMap&map,
                  bool p1,uint64_t extra,bool reused,uint64_t counts_wait_ns){
    if(!collecting())return;
    if(!generation||!serial||generation<=last_epoch_||(epoch_&&generation!=epoch_)||layer<0||layer>=full_||
        frame_seen_[layer]||from!=row0_[layer]){incomplete("stale_or_duplicate_route_binding");return;}
    epoch_=generation;if(!span_epoch_)span_epoch_=generation;
    if(frames_>=limits_.frames){incomplete("frame_cap");return;}
    if(!room(uint64_t(piece_.rows-from)*routes::kTopK*24+65536))return;
    routes::Binding b;b.sequence=piece_.sequence;b.span_epoch=span_epoch_;b.frame_epoch=generation;b.plan_cookie=plan_;
    b.layer=layer;b.rank=rank_;b.piece=pieces_-1;b.begin=piece_.begin;b.query_rows=piece_.rows;b.row_from=from;b.phase=routes::Phase::Prefill;
    const char*why=nullptr;if(!routes::freeze(b,v,&frozen_,&why)){incomplete("invalid_route_snapshot");return;}
    for(int e=0;e<96;++e){
        const Unit&u=map[e];if(u.unit<0||u.native_bytes<=0||u.tier<0||u.tier>4||u.residence==routes::Residence::Unknown){incomplete("invalid_plan_unit_binding");return;}
        for(int j=0;j<e;++j)if(map[j].unit==u.unit){incomplete("aliased_plan_unit");return;}
        Seen&s=seen_[layer][e];if(s.unit>=0&&s.unit!=u.unit){incomplete("plan_unit_changed");return;}s.unit=u.unit;
        if(frozen_.counts()[e]&&u.residence==routes::Residence::Nonresident){
            ++totals_.selected_nonresident;if(!s.selected_mask)++totals_.unique_selected_nonresident;
            s.selected_mask|=uint8_t(1u<<(pieces_-1));
        }
    }
    frame_seen_[layer]=1;p1_layer_[layer]=p1;done_[layer]=!p1;++frames_;current_layer_=layer;
    copied_={};current_map_=map;std::copy_n(frozen_.counts(),96,current_counts_.begin());
    common("route");out(",\"producer_serial\":%llu,\"layer\":%d,\"row_from\":%d,\"p1\":%s,\"extra_readback_bytes\":%llu,\"reused_order\":%s,\"counts_wait_ns\":%llu,\"ids\":[",
        (unsigned long long)serial,layer,from,p1?"true":"false",(unsigned long long)extra,reused?"true":"false",(unsigned long long)counts_wait_ns);
    for(int i=0;i<frozen_.slots();++i)out("%s%d",i?",":"",frozen_.ids()[i]);out("],\"sorted\":[");
    for(int i=0;i<frozen_.slots();++i)out("%s%d",i?",":"",frozen_.sorted()[i]);out("],\"counts\":[");
    for(int e=0;e<96;++e)out("%s%d",e?",":"",frozen_.counts()[e]);out("],\"offsets\":[");
    for(int e=0;e<97;++e)out("%s%d",e?",":"",frozen_.offsets()[e]);out("],\"units\":[");
    for(int e=0;e<96;++e)out("%s[%d,%d,%lld,%d]",e?",":"",map[e].unit,map[e].tier,(long long)map[e].native_bytes,int(map[e].residence));
    out("]}\n");
}
void Trace::copy(const Copy&c){
    if(!collecting())return;
    const int l=current_layer_,e=c.expert;
    if(l<0||!frame_seen_[l]||!p1_layer_[l]||done_[l]||e<0||e>=96||copied_[e]||c.unit!=current_map_[e].unit||
        c.rows!=current_counts_[e]||c.rows<=0||c.slot<0||c.kind<0||c.kind>2||c.gu_bytes<=0||c.dn_bytes<=0||
        c.gu_bytes>12533760||c.dn_bytes>6266880||current_map_[e].residence!=routes::Residence::Nonresident){incomplete("unbound_or_duplicate_copy");return;}
    if(!room(1024))return;copied_[e]=1;
    Seen&s=seen_[l][e];const uint64_t n=uint64_t(c.gu_bytes+c.dn_bytes);
    for(int p=0;p<pieces_-1;++p)if((s.copy_mask&(1u<<p))&&s.gu_by_piece[p]==c.gu_bytes&&
        s.dn_by_piece[p]==c.dn_bytes&&bool(s.packed_by_piece[p])==c.packed){
        totals_.duplicate_same_representation_bytes+=n;break;
    }
    s.gu_by_piece[pieces_-1]=c.gu_bytes;s.dn_by_piece[pieces_-1]=c.dn_bytes;s.packed_by_piece[pieces_-1]=c.packed;
    if(s.copy_mask){if(s.gu!=c.gu_bytes||s.dn!=c.dn_bytes||s.packed!=c.packed)s.mixed=true;}
    else{s.gu=c.gu_bytes;s.dn=c.dn_bytes;s.packed=c.packed;}
    s.copy_mask|=uint8_t(1u<<(pieces_-1));s.minimum=s.minimum?std::min(s.minimum,n):n;s.maximum=std::max(s.maximum,n);
    ++totals_.copies;totals_.copy_bytes+=n;
    common("copy");out(",\"layer\":%d,\"global_expert\":%d,\"expert\":%d,\"unit\":%d,\"rows\":%d,\"ring_slot\":%d,"
        "\"source_kind\":%d,\"packed\":%s,\"gu_bytes\":%lld,\"dn_bytes\":%lld,\"enqueue_succeeded\":true,\"nonresident\":true}\n",
        l,rank_*96+e,e,c.unit,c.rows,c.slot,c.kind,c.packed?"true":"false",(long long)c.gu_bytes,(long long)c.dn_bytes);
}
void Trace::layer_done(int l,bool ok){
    if(!collecting())return;
    if(l<0||l>=40||l!=current_layer_||!frame_seen_[l]||!p1_layer_[l]||done_[l]||!ok){incomplete("unfinished_p1_layer");return;}
    for(int e=0;e<96;++e)if(current_counts_[e]>0&&current_map_[e].residence==routes::Residence::Nonresident&&!copied_[e]){
        incomplete("selected_nonresident_copy_missing");return;}
    done_[l]=1;
}
void Trace::end(bool ok){
    if(!piece_on_)return;
    if(!failed_){
        if(!ok||!epoch_){incomplete("piece_failed_or_no_actual_generation");}
        else for(int l=0;l<40;++l){
            if(l>=full_||row0_[l]==piece_.rows){
                if(frame_seen_[l]){incomplete("unexpected_cut_layer_route");break;}
                common("cut");out(",\"layer\":%d,\"row_from\":%d,\"reason\":\"%s\"}\n",l,row0_[l],l>=full_?"consumer_full":"key_only");
            }else if(!frame_seen_[l]||!done_[l]){incomplete("missing_actual_layer");break;}
        }
        if(!failed_){last_epoch_=epoch_;span_end_=piece_.begin+piece_.rows;common("piece_end");out(",\"complete\":true,\"coverage_only\":true,\"request_commit_proven\":false}\n");}
    }
    piece_on_=false;
    if(!failed_&&span_end_==piece_.prompt_end)summary("prompt_boundary");
}
void Trace::summary(const char*reason){
    if(!span_on_)return;
    if(piece_on_){incomplete("summary_with_open_piece");return;}
    if(!failed_){
        for(const auto&layer:seen_)for(const Seen&s:layer)if(s.copy_mask){
            totals_.union_min_observed_bytes+=s.minimum;totals_.union_max_observed_bytes+=s.maximum;
            if(s.mixed)++totals_.mixed_representation_units;
        }
        common("span_summary");out(",\"covered_begin\":%lld,\"covered_end\":%lld,\"prompt_end\":%lld,\"pieces\":%d,\"covered_p1_complete\":%s,"
            "\"whole_prompt_covered\":%s,\"reason\":\"%s\",\"copies\":%llu,\"copy_bytes\":%llu,\"duplicate_nonresident_same_representation_bytes\":%llu,"
            "\"union_min_observed_representation_bytes\":%llu,\"union_max_observed_representation_bytes\":%llu,\"mixed_representation_units\":%llu,"
            "\"projected_repeat_bytes_min\":%llu,\"projected_repeat_bytes_max\":%llu,"
            "\"selected_nonresident_occurrences\":%llu,\"unique_selected_nonresident\":%llu,\"projected_only\":true,\"saving_claim\":false}\n",
            (long long)span_begin_,(long long)span_end_,(long long)piece_.prompt_end,pieces_,span_ok_?"true":"false",
            span_begin_==0&&span_end_==piece_.prompt_end?"true":"false",reason,(unsigned long long)totals_.copies,(unsigned long long)totals_.copy_bytes,
            (unsigned long long)totals_.duplicate_same_representation_bytes,(unsigned long long)totals_.union_min_observed_bytes,
            (unsigned long long)totals_.union_max_observed_bytes,(unsigned long long)totals_.mixed_representation_units,
            (unsigned long long)(totals_.copy_bytes-totals_.union_max_observed_bytes),
            (unsigned long long)(totals_.copy_bytes-totals_.union_min_observed_bytes),
            (unsigned long long)totals_.selected_nonresident,(unsigned long long)totals_.unique_selected_nonresident);
        if(std::fflush(fp_))incomplete("output_flush_error");
        if(!failed_)RAD_INFO("DSP2AUDIT span rank=%d seq=%llu span_epoch=%llu begin=%lld end=%lld pieces=%d complete=1 copies=%llu bytes=%llu "
            "duplicate_nonresident_same_representation_bytes=%llu union_min_observed_bytes=%llu union_max_observed_bytes=%llu projected_only=1 saving_claim=0 performance_eligible=0",
            rank_,(unsigned long long)piece_.sequence,(unsigned long long)span_epoch_,(long long)span_begin_,(long long)span_end_,pieces_,
            (unsigned long long)totals_.copies,(unsigned long long)totals_.copy_bytes,(unsigned long long)totals_.duplicate_same_representation_bytes,
            (unsigned long long)totals_.union_min_observed_bytes,(unsigned long long)totals_.union_max_observed_bytes);
    }
    span_on_=false;
}
void Trace::close(const char*reason){
    if(closed_)return;
    if(piece_on_)incomplete("shutdown_with_open_piece");piece_on_=false;summary(reason);
    if(fp_&&std::fflush(fp_)&&!failed_)incomplete("output_flush_error");
    bool complete=!failed_&&frames_>0&&spans_>0;
    if(fp_){
        const int n=std::fprintf(fp_,"{\"type\":\"audit_end\",\"rank\":%d,\"complete\":%s,\"frames\":%d,\"spans\":%d,\"performance_eligible\":false}\n",
            rank_,complete?"true":"false",frames_,spans_);
        if(n<0||std::fflush(fp_)){if(!failed_)incomplete("output_flush_error");complete=false;}
    }
    RAD_INFO("DSP2AUDIT end rank=%d complete=%d frames=%d spans=%d performance_eligible=0 executor=0",rank_,complete,frames_,spans_);
    closed_=true;
}
}}}

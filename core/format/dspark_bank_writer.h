// Private missing-only DSpark conversion policy. No payload copying or module loading here.
#pragma once
#include "dspark_rad_banks.h"
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace rad {
class DsparkBankWriterPlan {
public:
    struct Stats { uint64_t omitted_weights=0, omitted_payload_bytes=0, draft_weights=0, draft_payload_bytes=0; };
    // Both strings describe externally established lineage. This helper does not
    // freshly hash the canonical payload or authenticate checkpoint payloads.
    int begin(const RadFile& parent,const std::string& parent_sha,const std::string& source_identity) {
        if(parent_ || !parent.is_open() || !sha(parent_sha) || source_identity.empty() || source_identity.size()>1024)
            return RAD_E_INVAL;
        for(unsigned char c:source_identity)if(c<32 || c==127)return RAD_E_INVAL;
        const auto integer=[&](const char* key,int64_t want){return DsparkRadBanks::expected_geometry_scalar(parent,key,want);};
        if(std::strcmp(parent.str(parent.header().arch_id),"deepseek41") ||
           !integer("n_layers",40) || !integer("n_embd",5120) || !integer("n_ff",2304) ||
           !integer("n_expert",384) || !integer("n_expert_used",6) || !integer("n_vocab",129280) ||
           !integer("num_nextn_predict_layers",3) || !integer("dspark_block_size",5) ||
           !integer("dspark_n_routed_experts",128) || !integer("dspark_num_experts_per_tok",3) ||
           parent.entry_count()<=0 || parent.entry_count()>65536 || parent.meta_count()>65536)
            return RAD_E_FORMAT;
        // Ordinary metadata is an ordered typed sequence, including canonical repeats.
        for(int64_t i=0;i<parent.meta_count();++i) {
            const auto& kv=parent.meta_at(i);const std::string key=parent.str(kv.key);
            if(key.empty() || key.starts_with("draft_bank.") ||
               (kv.type!=RAD_P_INT && kv.type!=RAD_P_STR && kv.type!=RAD_KV_F64) ||
               (kv.type==RAD_KV_F64 && !std::isfinite(kv.v.f)))return RAD_E_FORMAT;
        }
        for(int64_t i=0;i<parent.entry_count();++i) {
            const auto& e=parent.entry(i);
            if(candidate(parent.str(e.name),nullptr) || e.layer>=40)return RAD_E_FORMAT;
        }
        RAD_TRY(namespace_.bind(parent,nullptr));
        RAD_TRY(DsparkRadBanks::metadata_digest(parent,&metadata_sha_));
        if(!namespace_.same_epoch())return RAD_E_STATE;
        parent_=&parent;parent_sha_=parent_sha;source_identity_=source_identity;
        return RAD_OK;
    }
    // Called on EVERY fully planned logical weight, before the output is opened.
    // A canonical match must be completely compatible; it is omitted, never a
    // reuse-copy job. Missing entries are restricted to native DSpark sources.
    int consider(const RadWriter::Weight& weight,bool* write) {
        if(!parent_ || sealed_ || !write || !namespace_.same_epoch())return RAD_E_STATE;
        if(weight.name.empty() || seen_.contains(weight.name) || seen_.size()>=65536)return RAD_E_FORMAT;
        uint64_t bytes=0;
        if(!plane_bytes(weight,&bytes))return RAD_E_FORMAT;
        const auto* existing=parent_->find(weight.name);
        if(existing) {
            if(!same(*parent_,*existing,weight) || stats_.omitted_payload_bytes>UINT64_MAX-bytes)return RAD_E_FORMAT;
            seen_.insert(weight.name);++stats_.omitted_weights;stats_.omitted_payload_bytes+=bytes;
            *write=false;return RAD_OK;
        }
        int named_layer=-1;
        if(!candidate(weight.name,&named_layer) || (named_layer>=0 && weight.layer!=named_layer) ||
           stats_.draft_payload_bytes>UINT64_MAX-bytes)return RAD_E_FORMAT;
        size_t expert_key=0;const bool expert=weight.expert>=0;
        if(expert) {
            if(weight.layer<40 || weight.layer>=43 || weight.expert>=128 || weight.slot<0 || weight.slot>1 ||
               !native_expert(weight,bytes))return RAD_E_FORMAT;
            expert_key=size_t((weight.layer-40)*256+weight.expert*2+weight.slot);
            if(experts_[expert_key])return RAD_E_FORMAT;
        }
        seen_.insert(weight.name);if(expert)experts_[expert_key]=1;
        ++stats_.draft_weights;stats_.draft_payload_bytes+=bytes;*write=true;return RAD_OK;
    }
    int finish_plan() {
        if(!parent_ || sealed_ || stats_.omitted_weights!=uint64_t(parent_->entry_count()) || stats_.draft_weights<768)
            return RAD_E_FORMAT;
        for(uint8_t present:experts_)if(present!=1)return RAD_E_FORMAT;
        RAD_TRY(check_parent(true));sealed_=true;return RAD_OK;
    }
    int check_parent(bool digest=false)const {
        if(!parent_ || !namespace_.same_epoch())return RAD_E_STATE;
        if(digest) {
            std::string now;RAD_TRY(DsparkRadBanks::metadata_digest(*parent_,&now));
            if(now!=metadata_sha_)return RAD_E_STATE;
        }
        return namespace_.same_epoch()?RAD_OK:RAD_E_STATE;
    }
    int header(RadWriter& writer)const {
        if(!sealed_)return RAD_E_STATE;
        RAD_TRY(check_parent(true));const auto& h=parent_->header();
        writer.set_arch(parent_->str(h.arch_id));writer.set_model_name(parent_->str(h.model_name));
        writer.set_quant(parent_->str(h.quant));
        // Preserve types and floating bit patterns; the donor cannot override
        // ordinary metadata. Tokenizer, auxiliary tokenizer and profile stay out.
        for(int64_t i=0;i<parent_->meta_count();++i) {
            const auto& kv=parent_->meta_at(i);const char* key=parent_->str(kv.key);
            if(kv.type==RAD_P_INT)writer.meta_i(key,kv.v.i);
            else if(kv.type==RAD_KV_F64)writer.meta_f(key,kv.v.f);
            else writer.meta_s(key,parent_->str(kv.v.s));
        }
        writer.meta_i("draft_bank.version",1);writer.meta_s("draft_bank.kind","deepseek41.dspark");
        writer.meta_s("draft_bank.parent_metadata_sha256",metadata_sha_);
        writer.meta_i("draft_bank.parent_file_bytes",parent_->size());
        writer.meta_s("draft_bank.parent_payload_sha256",parent_sha_);
        writer.meta_s("draft_bank.source_identity",source_identity_);return RAD_OK;
    }
    int validate_output(const RadFile& output)const {
        if(!sealed_)return RAD_E_STATE;
        RAD_TRY(check_parent(true));DsparkRadBanks reader;
        RAD_TRY(reader.bind(*parent_,&output));
        if(output.entry_count()!=int64_t(stats_.draft_weights))return RAD_E_FORMAT;
        std::set<int64_t> consumed;
        for(int64_t i=0;i<output.entry_count();++i)consumed.insert(i);
        RAD_TRY(reader.finish_coverage(consumed));return check_parent(true);
    }
    const Stats& stats()const{return stats_;}
    static bool compatible(const RadFile& parent,const RadFileEntry& entry,const RadWriter::Weight& weight) {
        return same(parent,entry,weight);
    }
private:
    static bool sha(const std::string& value) {
        if(value.size()!=64)return false;
        for(char c:value)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
        return true;
    }
    static bool candidate(const std::string& name,int* layer) {
        if(name.starts_with("dspark.") && name.size()>7){if(layer)*layer=-1;return true;}
        if(!name.starts_with("blk."))return false;
        const char* end=name.data()+name.size();int value=0;
        const auto parsed=std::from_chars(name.data()+4,end,value);
        if(parsed.ec!=std::errc{} || parsed.ptr==name.data()+4 || parsed.ptr==end || *parsed.ptr!='.' ||
           parsed.ptr+1==end || value<40 || value>=43)return false;
        if(layer)*layer=value;
        return true;
    }
    static bool plane_bytes(const RadWriter::Weight& w,uint64_t* bytes) {
        if(!rad_enc_valid(&w.enc) || w.rank==0 || w.rank>RAD_MAX_RANK)return false;
        int64_t rows=1;
        for(uint32_t i=0;i<w.rank;++i)if(w.shape[i]<=0)return false;
        for(uint32_t i=w.rank;i<RAD_MAX_RANK;++i)if(w.shape[i]!=0)return false;
        for(uint32_t i=0;i+1<w.rank;++i) {
            if(rows>INT64_MAX/w.shape[i])return false;
            rows*=w.shape[i];
        }
        uint64_t sum=0;
        for(int i=0;i<w.enc.n_planes;++i) {
            const auto& p=w.enc.plane[i];
            const auto ceil=[](uint64_t n,uint64_t block){return block?n/block+(n%block!=0):uint64_t(1);};
            const uint64_t pr=p.kind==RAD_PLANE_TABLE?uint64_t(p.extent[0]):ceil(uint64_t(rows),uint64_t(p.block[0]));
            const uint64_t pc=p.kind==RAD_PLANE_TABLE?uint64_t(p.extent[1]):ceil(uint64_t(w.shape[w.rank-1]),uint64_t(p.block[1]));
            const uint64_t bits=uint64_t(rad_dtype_bits(p.dtype));
            if(!pr || !pc || !bits || pc>(uint64_t(INT64_MAX)-7)/bits)return false;
            const uint64_t row_bytes=(pc*bits+7)/8;
            if(!row_bytes || pr>uint64_t(INT64_MAX)/row_bytes)return false;
            const uint64_t part=pr*row_bytes;
            if(sum>uint64_t(INT64_MAX)-part)return false;
            sum+=part;
        }
        *bytes=sum;return true;
    }
    static bool same(const RadFile& parent,const RadFileEntry& e,const RadWriter::Weight& w) {
        if(!rad_enc_equal(&parent.encoding(e),&w.enc) || w.quantizer!=parent.str(e.quantizer) ||
           w.options!=parent.str(e.options) || e.rank!=w.rank || e.layer!=w.layer || e.expert!=w.expert || e.slot!=w.slot)
            return false;
        for(uint32_t i=0;i<RAD_MAX_RANK;++i)if(e.shape[i]!=w.shape[i])return false;
        return true;
    }
    static bool native_expert(const RadWriter::Weight& w,uint64_t bytes) {
        if(w.rank!=2 || !w.quantizer.empty() || !w.options.empty())return false;
        const int64_t N=w.slot==0?4608:5120,K=w.slot==0?5120:2304;
        if(w.shape[0]!=N || w.shape[1]!=K || bytes!=uint64_t(w.slot==0?12533760:6266880))return false;
        const RadEncoding native=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
        return rad_enc_equal(&native,&w.enc);
    }
    const RadFile* parent_=nullptr;
    DsparkRadBanks namespace_;
    std::string metadata_sha_,parent_sha_,source_identity_;
    std::set<std::string> seen_;
    std::array<uint8_t,768> experts_{};
    Stats stats_{};
    bool sealed_=false;
};
} //namespace rad

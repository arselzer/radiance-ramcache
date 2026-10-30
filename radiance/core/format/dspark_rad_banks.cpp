#include "dspark_rad_banks.h"
#include "place/expert_profile_sha256.h"
#include <atomic>
#include <charconv>
#include <new>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
namespace rad { namespace {
constexpr uint64_t max_metadata=256ull<<20;
constexpr const char* prefix="draft_bank.";
uint64_t generation(){static std::atomic<uint64_t>next{1};auto v=next.load(std::memory_order_relaxed);while(v<UINT64_MAX){if(next.compare_exchange_weak(v,v+1,std::memory_order_relaxed))return v;}return 0;}
int fail(std::string*why,const char*text){if(why)*why=text;return RAD_E_FORMAT;}
bool sha_string(const char*s){if(!s||std::strlen(s)!=64)return false;for(int i=0;i<64;++i)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return false;return true;}
bool epoch_fd(int fd,dspark_bank::Epoch*out){
    struct stat s{};if(fd<0||!out||::fstat(fd,&s)||!S_ISREG(s.st_mode)||s.st_size<=0||s.st_mtim.tv_sec<0||s.st_ctim.tv_sec<0)return false;
    constexpr int64_t ns=1000000000;
    if(uint64_t(s.st_mtim.tv_sec)>uint64_t(INT64_MAX/ns)||uint64_t(s.st_ctim.tv_sec)>uint64_t(INT64_MAX/ns))return false;
    const int64_t mt=int64_t(s.st_mtim.tv_sec)*ns,ct=int64_t(s.st_ctim.tv_sec)*ns;
    if(s.st_mtim.tv_nsec<0||s.st_mtim.tv_nsec>=ns||s.st_ctim.tv_nsec<0||s.st_ctim.tv_nsec>=ns||mt>INT64_MAX-s.st_mtim.tv_nsec||ct>INT64_MAX-s.st_ctim.tv_nsec)return false;
    *out={uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),uint64_t(s.st_nlink),uint32_t(s.st_mode),mt+s.st_mtim.tv_nsec,ct+s.st_ctim.tv_nsec};return out->valid();
}
using Meta=std::vector<const RadFileKV*>;
bool metadata(const RadFile&f,bool donor,Meta*out){
    if(f.meta_count()<0||f.meta_count()>65536)return false;
    for(int64_t i=0;i<f.meta_count();++i){const auto&kv=f.meta_at(i);const std::string k=f.str(kv.key);
        if(k.empty()||(kv.type!=RAD_P_INT&&kv.type!=RAD_P_STR&&kv.type!=RAD_KV_F64)||(kv.type==RAD_KV_F64&&!std::isfinite(kv.v.f)))return false;
        if(k.starts_with(prefix)){if(!donor)return false;continue;}
        out->push_back(&kv);
    }return true;
}
bool same_meta(const RadFile&a,const RadFile&b){
    Meta x,y;if(!metadata(a,false,&x)||!metadata(b,true,&y)||x.size()!=y.size())return false;
    //Canonical metadata is an ordered typed sequence, including repeated keys.
    //RadFile getters select their first matching type; deduplicating would change
    //that contract. Donor ordinary rows must reproduce every original row exactly.
    for(size_t i=0;i<x.size();++i){const auto*av=x[i];const auto*bv=y[i];
        if(std::strcmp(a.str(av->key),b.str(bv->key))||av->type!=bv->type)return false;
        if(av->type==RAD_P_INT){if(av->v.i!=bv->v.i)return false;}
        else if(av->type==RAD_KV_F64){if(std::memcmp(&av->v.f,&bv->v.f,sizeof(double)))return false;}
        else if(std::strcmp(a.str(av->v.s),b.str(bv->v.s)))return false;
    }return true;
}
bool integer(const RadFile&f,const char*k,int64_t want){int64_t v=0;return f.get_i(k,&v)&&v==want;}
bool candidate(const std::string&name,int*layer){
    if(name.starts_with("dspark.")&&name.size()>7){if(layer)*layer=-1;return true;}
    if(!name.starts_with("blk."))return false;const char*beg=name.data()+4;const char*end=name.data()+name.size();int l=0;auto r=std::from_chars(beg,end,l);
    if(r.ec!=std::errc{}||r.ptr==beg||r.ptr==end||*r.ptr!='.'||r.ptr+1==end||l<40||l>=43)return false;
    if(layer)*layer=l;return true;
}
} //namespace
bool DsparkRadBanks::expected_geometry_scalar(const RadFile&file,const char*key,int64_t want){
    if(!file.is_open()||!key||!*key||want<0||file.meta_count()<0||file.meta_count()>65536)return false;
    char expected[32];const auto decimal=std::to_chars(expected,expected+sizeof(expected),want);
    if(decimal.ec!=std::errc{})return false;
    *decimal.ptr='\0';bool found=false;
    for(int64_t i=0;i<file.meta_count();++i){const auto&kv=file.meta_at(i);
        if(std::strcmp(file.str(kv.key),key))continue;
        found=true;
        if(kv.type==RAD_P_INT){if(kv.v.i!=want)return false;}
        else if(kv.type==RAD_P_STR){if(std::strcmp(file.str(kv.v.s),expected))return false;}
        else return false;
    }
    return found;
}
int DsparkRadBanks::metadata_digest(const RadFile&f,std::string*out){
    if(!out||!f.is_open()||f.size()<=0)return RAD_E_INVAL;
    dspark_bank::Epoch a,b;if(!epoch_fd(f.fd_for_metadata_identity(),&a)||a.bytes!=uint64_t(f.size()))return RAD_E_STATE;
    const auto&h=f.header();if(h.file_bytes!=a.bytes||h.data_off<sizeof(RadFileHeader)||h.data_off>h.file_bytes||h.data_off>max_metadata)return RAD_E_FORMAT;
    //The first sidecar contract admits ordinary prefix-table RADs only. Append-format RADs may
    //move tables past the payload; hashing only their original prefix would omit live metadata.
    const auto prefix_span=[&](uint64_t off,uint64_t count,uint64_t width){
        return count==0||(width&&count<=UINT64_MAX/width&&off<=h.data_off&&count*width<=h.data_off-off);};
    if(!prefix_span(h.str_off,h.str_bytes,1)||!prefix_span(h.meta_off,h.meta_count,sizeof(RadFileKV))||
       !prefix_span(h.dir_off,h.dir_count,sizeof(RadFileEntry))||!prefix_span(h.plane_off,h.plane_count,sizeof(RadFilePlane))||
       !prefix_span(h.enc_off,h.enc_count,sizeof(RadEncoding))||!prefix_span(h.vocab_off,h.vocab_bytes,1)||
       !prefix_span(h.prof_off,h.prof_count,sizeof(RadFileProfile))||
       !prefix_span(h.reserved[RAD_HDR_AUX_VOCAB_OFF],h.reserved[RAD_HDR_AUX_VOCAB_BYTES],1))return RAD_E_FORMAT;
    try{ds41_integrity::Sha256 hash;hash.update(f.base(),size_t(h.data_off));
        std::string value=hash.finish();if(!epoch_fd(f.fd_for_metadata_identity(),&b)||a!=b)return RAD_E_STATE;out->swap(value);return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
}
int DsparkRadBanks::bind(const RadFile&main,const RadFile*donor,std::string*reason){
    if(count_)return RAD_E_STATE;
    try {
    files_[0]=files_[1]=nullptr;count_=0;snapshot_={};manifest_={};
    if(!main.is_open())return fail(reason,"canonical RAD is not open");
    dspark_bank::Snapshot ns;ns.generation=generation();ns.count=donor?2:1;if(!ns.generation)return RAD_E_STATE;
    if(!epoch_fd(main.fd_for_metadata_identity(),&ns.epoch[0])||ns.epoch[0].bytes!=uint64_t(main.size()))return RAD_E_STATE;
    if(!donor){files_[0]=&main;count_=1;snapshot_=ns;return ns.valid()?RAD_OK:RAD_E_STATE;}
    if(!donor->is_open()||!epoch_fd(donor->fd_for_metadata_identity(),&ns.epoch[1])||ns.epoch[1].bytes!=uint64_t(donor->size())||!ns.valid())return fail(reason,"draft bank must be a distinct retained ordinary file");
    const auto&mh=main.header();const auto&dh=donor->header();
    if(std::strcmp(main.str(mh.arch_id),"deepseek41")||std::strcmp(main.str(mh.arch_id),donor->str(dh.arch_id))||std::strcmp(main.str(mh.model_name),donor->str(dh.model_name))||std::strcmp(main.str(mh.quant),donor->str(dh.quant))||!same_meta(main,*donor))return fail(reason,"draft bank cannot override canonical model metadata");
    if(!expected_geometry_scalar(main,"num_nextn_predict_layers",3)||!expected_geometry_scalar(main,"dspark_block_size",5)||!expected_geometry_scalar(main,"dspark_n_routed_experts",128)||!expected_geometry_scalar(main,"dspark_num_experts_per_tok",3)||!expected_geometry_scalar(main,"n_layers",40)||!expected_geometry_scalar(main,"n_embd",5120)||!expected_geometry_scalar(main,"n_ff",2304)||!expected_geometry_scalar(main,"n_expert",384)||!expected_geometry_scalar(main,"n_expert_used",6)||!expected_geometry_scalar(main,"n_vocab",129280))return fail(reason,"first bank path is native DS4.1 Flash1017 geometry only");
    if(dh.vocab_bytes||dh.reserved[RAD_HDR_AUX_VOCAB_BYTES]||dh.prof_count)return fail(reason,"draft bank owns no tokenizer or profile");
    constexpr const char*known[]={"draft_bank.version","draft_bank.kind","draft_bank.parent_metadata_sha256","draft_bank.parent_file_bytes","draft_bank.parent_payload_sha256","draft_bank.source_identity"};
    std::set<std::string>seen;
    for(int64_t i=0;i<donor->meta_count();++i){const auto&kv=donor->meta_at(i);const std::string k=donor->str(kv.key);if(!k.starts_with(prefix))continue;
        bool ok=false;for(auto n:known)if(k==n)ok=true;if(!ok||!seen.insert(k).second)return fail(reason,"unknown or duplicate draft-bank manifest key");}
    const char*kind=donor->get_s(known[1]);const char*md=donor->get_s(known[2]);const char*payload=donor->get_s(known[4]);const char*source=donor->get_s(known[5]);int64_t bytes=0;
    if(seen.size()!=6||!integer(*donor,known[0],1)||!kind||std::strcmp(kind,"deepseek41.dspark")||!sha_string(md)||!sha_string(payload)||!source||!*source||std::strlen(source)>1024||!donor->get_i(known[3],&bytes)||bytes!=main.size())return fail(reason,"incomplete draft-bank parent/source manifest");
    for(const char*p=source;*p;++p)if(uint8_t(*p)<32||uint8_t(*p)==127)return fail(reason,"invalid source identity text");
    std::string actual;const int hashed=metadata_digest(main,&actual);if(hashed<0)return hashed;if(actual!=md)return fail(reason,"draft bank belongs to different canonical metadata");
    if(donor->entry_count()<=0||donor->entry_count()>65536)return fail(reason,"bounded nonempty donor directory required");
    for(int64_t i=0;i<donor->entry_count();++i){const auto&e=donor->entry(i);const std::string name=donor->str(e.name);int layer=-1;
        if(main.find(name)||!candidate(name,&layer)||(layer>=0&&e.layer!=layer)||(e.expert>=0&&(e.layer<40||e.layer>=43||e.expert>=128||e.slot<0||e.slot>1)))return fail(reason,"draft bank duplicates or adds a foreign logical source");}
    files_[0]=&main;files_[1]=donor;count_=2;snapshot_=ns;manifest_={actual,payload,source,uint64_t(bytes)};
    if(!same_epoch()){files_[0]=files_[1]=nullptr;count_=0;snapshot_={};return RAD_E_STATE;}return RAD_OK;
    }catch(const std::bad_alloc&){files_[0]=files_[1]=nullptr;count_=0;snapshot_={};manifest_={};return RAD_E_NOMEM;}
}
bool DsparkRadBanks::same_epoch()const{
    auto now=snapshot_;for(uint32_t i=0;i<count_;++i)if(!files_[i]||!epoch_fd(files_[i]->fd_for_metadata_identity(),&now.epoch[i]))return false;
    return count_&&snapshot_.current(now);
}
int DsparkRadBanks::direct_owner_matches(uint32_t bank,int fd)const{
    dspark_bank::Epoch opened;return bank<count_&&same_epoch()&&epoch_fd(fd,&opened)&&opened==snapshot_.epoch[bank]?RAD_OK:RAD_E_STATE;
}
int DsparkRadBanks::find(const std::string&name,Entry*out)const{
    if(!out||!count_)return RAD_E_INVAL;if(!same_epoch())return RAD_E_STATE;Entry got;
    for(uint32_t i=0;i<count_;++i)if(const auto*e=files_[i]->find(name)){if(got.entry)return RAD_E_FORMAT;got={files_[i],e,i,int64_t(e-&files_[i]->entry(0))};}
    if(!got.entry)return RAD_E_NOTFOUND;if(!same_epoch())return RAD_E_STATE;*out=got;return RAD_OK;
}
int DsparkRadBanks::bound(const WeightInfo&w,Entry*out)const{
    if(!out||w.file_id>=count_||w.source_bank_generation!=snapshot_.generation||w.source_entry_index<0)return RAD_E_STATE;
    if(!same_epoch())return RAD_E_STATE;const auto*f=files_[w.file_id];
    if(w.source_entry_index>=f->entry_count())return RAD_E_STATE;const auto&e=f->entry(w.source_entry_index);
    if(w.source!=f->str(e.name)||!same_epoch())return RAD_E_STATE;
    *out={f,&e,w.file_id,w.source_entry_index};return RAD_OK;
}
WeightSourceFn DsparkRadBanks::sources()const{
    return [this](const std::string&name,const Program&,WeightSource*out)->int{if(!out)return RAD_E_INVAL;Entry ref;const int st=find(name,&ref);if(st<0)return st;
        WeightSource value;value.enc=ref.file->encoding(*ref.entry);value.rank=ref.entry->rank;for(int i=0;i<RAD_MAX_RANK;++i)value.shape[i]=ref.entry->shape[i];if(!same_epoch())return RAD_E_STATE;*out=value;return RAD_OK;};
}
int DsparkRadBanks::bind_weights(std::vector<WeightInfo>&weights,std::set<int64_t>*used)const{
    if(!same_epoch())return RAD_E_STATE;
    try{
        std::vector<Entry>refs(weights.size());std::set<int64_t>next=used?*used:std::set<int64_t>{};
        for(size_t i=1;i<weights.size();++i){const auto&w=weights[i];const int st=find(w.source,&refs[i]);
            if(st==RAD_E_NOTFOUND&&w.decl.optional)continue;if(st<0)return st;if(refs[i].bank&&used)next.insert(refs[i].index);}
        if(!same_epoch())return RAD_E_STATE;
        for(size_t i=1;i<weights.size();++i){weights[i].file_id=refs[i].bank;weights[i].source_entry_index=refs[i].index;weights[i].source_bank_generation=refs[i].entry?snapshot_.generation:0;}
        if(used)used->swap(next);return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
}
int DsparkRadBanks::finish_coverage(const std::set<int64_t>&used)const{
    if(!dual()||!same_epoch())return RAD_E_STATE;const auto&d=*files_[1];if(used.size()!=size_t(d.entry_count())||d.group_count()!=3)return RAD_E_FORMAT;
    for(int64_t i=0;i<d.entry_count();++i)if(!used.contains(i))return RAD_E_FORMAT;
    for(int layer=40;layer<43;++layer){const auto*g=d.group_for_layer(layer);if(!g||g->n_expert!=128||g->n_slot!=2)return RAD_E_FORMAT;}
    return RAD_OK; //native shapes, source-byte lineage, fit and learned-draft math remain separate gates.
}
} //namespace rad

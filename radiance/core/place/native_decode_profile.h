/* Private experimental startup placement input. No model-weight mutation or hot-path work.
 * Existing expert-stats/1 JSON + source_identity, sealed offline from the SAME immutable model.
 * Only pure-DECODE trunk counters are used. MIXED cannot be separated into decode experts. */
#pragma once
#include "rad_format.h"
#include "rad_builder.h"
#include "expert_profile_sha256.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <utility>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace rad {
constexpr size_t kNativeDecodeProfileMaxBytes=2u<<20;
constexpr uint64_t kNativeDecodeProfileMaxMetadata=256ull<<20;
struct NativeDecodeProfile {
    std::vector<RadFileProfile> rows;
    uint64_t decode_tokens=0,nonzero_rows=0;
};
struct NativeProfileIdentity {
    uint64_t file_bytes=0,metadata_bytes=0,device=0,inode=0,mtime_ns=0,ctime_ns=0;
    std::string metadata_sha256,model,container;
};
inline bool native_profile_epoch(const struct stat&a,const struct stat&b) {
    return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&
        a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&
        a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}
/* File is the actual already validated RadFile. Only [0,data_off) is hashed, never weights.
 * Descriptor is BORROWED. lstat proves the named file still names the mapped descriptor. */
template<class File>
int native_profile_identity(const File&file,NativeProfileIdentity*out) {
    if(!out||file.size()<=0||!file.base())return RAD_E_INVAL;
    const auto&h=file.header();const int fd=file.fd_for_metadata_identity();
    struct stat a{},b{},named{};
    if(fd<0||::fstat(fd,&a)||!S_ISREG(a.st_mode)||a.st_size!=file.size()||
       ::lstat(file.path(),&named)||!S_ISREG(named.st_mode)||!native_profile_epoch(a,named)||
       h.file_bytes!=uint64_t(file.size())||h.data_off<sizeof(RadFileHeader)||
       h.data_off>h.file_bytes||h.data_off>kNativeDecodeProfileMaxMetadata||
       a.st_mtim.tv_sec<0||a.st_ctim.tv_sec<0)return RAD_E_FORMAT;
    ds41_integrity::Sha256 hash;hash.update(file.base(),size_t(h.data_off));
    if(::fstat(fd,&b)||::lstat(file.path(),&named)||!native_profile_epoch(a,b)||
       !native_profile_epoch(a,named)||!S_ISREG(named.st_mode))return RAD_E_STATE;
    NativeProfileIdentity id;id.file_bytes=h.file_bytes;id.metadata_bytes=h.data_off;
    id.device=uint64_t(a.st_dev);id.inode=uint64_t(a.st_ino);
    id.mtime_ns=uint64_t(a.st_mtim.tv_sec)*1000000000u+uint64_t(a.st_mtim.tv_nsec);
    id.ctime_ns=uint64_t(a.st_ctim.tv_sec)*1000000000u+uint64_t(a.st_ctim.tv_nsec);
    id.metadata_sha256=hash.finish();id.model=file.str(h.model_name);id.container=file.path();
    *out=std::move(id);return RAD_OK;
}
inline int parse_native_decode_profile(std::string_view text,const NativeProfileIdentity&expected,
                                      NativeDecodeProfile*out,std::string*reason) {
    using J=nlohmann::json;
    auto require=[](bool value,const char*why){if(!value)throw std::runtime_error(why);};
    auto integer=[&](const J&v)->uint64_t{
        require(v.is_number_unsigned()||(v.is_number_integer()&&v.get<int64_t>()>=0),"nonnegative integer required");
        return v.get<uint64_t>();
    };
    try{
        require(out&&text.size()<=kNativeDecodeProfileMaxBytes,"bounded profile output/text");
        const auto doc=J::parse(text.begin(),text.end());
        require(doc.is_object()&&doc.at("format")=="radiance-expert-stats/1"&&
            doc.at("final").is_boolean()&&doc.at("final").get<bool>(),"complete expert-stats/1 required");
        require(doc.at("arch")=="deepseek41"&&doc.at("model")==expected.model&&
            doc.at("container")==expected.container&&integer(doc.at("tp"))==4&&
            integer(doc.at("n_trunk_layers"))==40&&integer(doc.at("n_experts"))==384,
            "exact native DS4.1 EP4 source/model geometry required");
        const auto&identity=doc.at("source_identity");
        require(identity.at("format")=="rad-metadata-prefix-sha256/1"&&
            integer(identity.at("file_bytes"))==expected.file_bytes&&
            integer(identity.at("metadata_bytes"))==expected.metadata_bytes&&
            integer(identity.at("device"))==expected.device&&integer(identity.at("inode"))==expected.inode&&
            integer(identity.at("mtime_ns"))==expected.mtime_ns&&integer(identity.at("ctime_ns"))==expected.ctime_ns&&
            expected.metadata_sha256.size()==64&&identity.at("sha256")==expected.metadata_sha256,
            "profile source identity differs from held model generation/metadata");
        const uint64_t layers=integer(doc.at("n_layers"));
        require(layers==40||layers==41,"40 trunk layers and at most one ignored draft head");
        const auto&counts=doc.at("counts").at("decode");
        const auto&observed=doc.at("observed_tokens").at("decode");
        const auto&selections=doc.at("selections").at("decode");
        const auto&hist=doc.at("histograms").at("decode");
        const auto&passes=doc.at("passes").at("decode");
        const auto&experts=doc.at("layer_experts");const auto&top=doc.at("top_k");const auto&mapping=doc.at("map");
        for(const J*array:{&counts,&observed,&selections,&hist,&passes,&experts,&top,&mapping})
            require(array->is_array()&&array->size()==layers,"exact per-layer array shapes");
        const auto&ranks=doc.at("ranks");
        for(const char*kind:{"credited","unmapped","unclassified","dropped"}){
            const auto&row=ranks.at(kind);require(row.is_array()&&row.size()==4,"all four rank quality counters");
            for(const auto&n:row)require(std::strcmp(kind,"credited")==0?integer(n)>0:integer(n)==0,
                                       "lost/unmapped/unclassified routing is not calibration data");
        }
        NativeDecodeProfile value;value.decode_tokens=integer(doc.at("tokens").at("decode"));
        require(value.decode_tokens>0&&value.decode_tokens<=UINT64_MAX/6&&
                integer(doc.at("steps").at("decode"))>0,"positive bounded pure decode coverage");
        value.rows.reserve(40*384);
        for(size_t layer=0;layer<40;++layer){
            require(integer(experts[layer])==384&&integer(top[layer])==6&&mapping[layer]=="container"&&
                counts[layer].is_array()&&counts[layer].size()==384&&
                integer(observed[layer])==value.decode_tokens&&integer(hist[layer])>0&&
                integer(passes[layer])>=integer(hist[layer]),"complete source-mapped trunk layer coverage");
            const uint64_t total=6*value.decode_tokens;uint64_t sum=0;
            for(const auto&n:counts[layer]){const auto c=integer(n);require(c<=value.decode_tokens&&sum<=total-c,"invalid expert count/overflow");sum+=c;}
            require(sum==total&&integer(selections[layer])==total,"decode selection conservation");
            for(size_t expert=0;expert<384;++expert){
                const uint64_t c=integer(counts[layer][expert]);
                RadFileProfile row{};row.layer=int32_t(layer);row.expert=int32_t(expert);
                row.share=float(double(c)/double(total));value.rows.push_back(row);value.nonzero_rows+=c!=0;
            }
        }
        *out=std::move(value);if(reason)reason->clear();return RAD_OK;
    }catch(const std::bad_alloc&){if(reason)*reason="profile allocation failed";return RAD_E_NOMEM;}
    catch(const std::exception&e){if(reason)*reason=e.what();return RAD_E_FORMAT;}
}
inline int read_native_decode_profile(const std::string&path,const NativeProfileIdentity&identity,
                                     NativeDecodeProfile*out,std::string*reason) {
    struct Owned {int fd=-1;~Owned(){if(fd>=0)::close(fd);}}input;
    input.fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    struct stat a{},b{},named{};
    if(input.fd<0||::fstat(input.fd,&a)||!S_ISREG(a.st_mode)||a.st_size<=0||
       uint64_t(a.st_size)>kNativeDecodeProfileMaxBytes)return RAD_E_IO;
    try{
        std::string text(size_t(a.st_size),'\0');size_t at=0;
        while(at<text.size()){
            const auto n=::pread(input.fd,text.data()+at,text.size()-at,off_t(at));
            if(n<0&&errno==EINTR)continue;if(n<=0)return RAD_E_IO;at+=size_t(n);
        }
        if(::fstat(input.fd,&b)||::lstat(path.c_str(),&named)||!S_ISREG(named.st_mode)||
           !native_profile_epoch(a,b)||!native_profile_epoch(a,named))return RAD_E_STATE;
        return parse_native_decode_profile(text,identity,out,reason);
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
}
/* Exact first prototype domain. Group identity is local; SOURCE mapping is a separate,
 * already-tested adapter. No global expert id is inferred from rank number here. */
template<class Weights>
bool native_decode_profile_geometry(const Weights&weights) {
    std::array<std::array<unsigned,96>,40> seen{};
    for(size_t i=1;i<weights.size();++i){const auto&w=weights[i];const auto&g=w.decl.group;
        if(g.expert<0||g.layer<0||g.layer>=40)continue;
        if(g.expert>=96||g.slot<0||g.slot>1||w.decl.access!=RAD_ACCESS_CONDITIONAL||
           w.decl.rank!=2||w.layout_tag!="ggml_mxfp4"||w.n_sel!=2||!w.enc_known||w.enc.n_planes!=2||
           w.enc.plane[0].dtype!=RAD_FP4E2M1||w.enc.plane[1].dtype!=RAD_E8M0)return false;
        const bool gu=g.slot==0;
        if(w.decl.shape[0]!=(gu?4608:5120)||w.decl.shape[1]!=(gu?5120:2304)||
           w.stored_bytes!=(gu?12533760:6266880))return false;
        unsigned&mask=seen[size_t(g.layer)][size_t(g.expert)];const unsigned bit=gu?1u:2u;
        if(mask&bit)return false;mask|=bit;
    }
    for(const auto&layer:seen)for(unsigned mask:layer)if(mask!=3)return false;
    return true;
}
} //namespace rad

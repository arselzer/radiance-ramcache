#pragma once
// Research metadata profile only. Nothing here activates an Engine or creates a Ctx.
#include "rad_core.h"
#include <cstdint>
#include <memory>
#include <string>
#include <array>
#include <utility>
#include <type_traits>
namespace rad {class Checkpoint;}
namespace glm53::engine_core_contract {
inline constexpr int L=8,H=128,F=128,E=4,K=2,T=5,R=2,V=320;
inline constexpr uint64_t MiB=uint64_t(1)<<20;
inline constexpr uint64_t admitted=192*MiB,category_sum=135244800;
inline constexpr const char* arch_id="glm53nativetiny";
enum class Backend:uint32_t {Host=1,Aql=2};
struct VocabQuote {
    uint32_t tokens=0,eog_count=0;
    int32_t bos=-1,eos=-1,eot=-1;
    bool add_bos=false,add_eos=false,bos_terminal=false,eos_terminal=false,im_end_terminal=false;
};
static_assert(std::is_trivially_copyable_v<VocabQuote>&&sizeof(VocabQuote)<=64);
struct SourceQuote {
    uint64_t file_bytes=0,header_bytes=0,payload_bytes=0,fingerprint=0;
    uint32_t physical_tensors=0,vocabulary=0;
    uint32_t artifact_files=0;
    uint64_t sidecar_bytes=0;
    VocabQuote vocab{};
};
// Created only by the bounded pre-open factory. The private unique Checkpoint
// and four before/after-open artifact identities live as long as ANY inspected
// result/seal retains this shared owner. Borrow checkpoint() solely for actual
// Core source resolution; callers MUST NOT reopen/add files/change its metadata.
class SourceCheckpoint {
public:
    ~SourceCheckpoint();
    rad::Checkpoint& checkpoint()const noexcept;
    const std::string& root()const noexcept{return root_;}
    int audit(std::string* why=nullptr)const;
    SourceCheckpoint(const SourceCheckpoint&)=delete;
private:
    struct Stamp {
        uint64_t device=0,inode=0,bytes=0,mode=0;
        int64_t mtime_sec=0,mtime_nsec=0,ctime_sec=0,ctime_nsec=0;
        bool operator==(const Stamp&)const=default;
    };
    std::unique_ptr<rad::Checkpoint> checkpoint_;
    std::string root_;
    std::array<Stamp,4> artifacts_;
    SourceCheckpoint(std::unique_ptr<rad::Checkpoint>,std::string,std::array<Stamp,4>);
    static std::array<Stamp,4> stamps(const std::string&);
    friend int open_source_checkpoint(const std::string&,std::shared_ptr<const SourceCheckpoint>*,std::string*);
};
int open_source_checkpoint(const std::string&,std::shared_ptr<const SourceCheckpoint>*,std::string* why=nullptr);
class ColdSourceSeal;
class BoundSources;
class InspectedSource {
public:
    const SourceQuote& quote()const noexcept{return quote_;}
private:
    SourceQuote quote_;
    std::shared_ptr<const SourceCheckpoint> owner_;
    InspectedSource(const SourceQuote& q,std::shared_ptr<const SourceCheckpoint> owner):quote_(q),owner_(std::move(owner)){}
    friend int inspect_checkpoint_source(const std::shared_ptr<const SourceCheckpoint>&,const std::string&,
        std::unique_ptr<const InspectedSource>*,std::string*);
    friend int seal_source(const InspectedSource&,std::unique_ptr<const ColdSourceSeal>*,std::string*);
};
struct BackendQuote {
    Backend backend=Backend::Host;
    uint64_t stack_bytes=0,guard_bytes=0,host_stream_control_bytes=0;
    uint32_t loader_threads=2,streams=3;
    bool matching_private_size_quote=false;
    bool tiny_loader_retirement_implemented=false,host_stream_raii_implemented=false;
};
struct AdmissionQuote {
    uint64_t categories=category_sum,cap=admitted,backend_stream_bytes=0;
    uint32_t owned_threads=0;
    bool source_admitted=false,backend_admitted=false;
    bool engine_eligible=false; // Deliberately ALWAYS false in this source-only package.
};
// Shape check usable by this separately named architecture. It is NOT an Engine
// zero-head exception; current Engine metadata admission remains untouched.
int admit_declaration_shape(const RadModelMeta*,const RadBuildCtx*)noexcept;
class ColdSourceSeal {
public:
    const SourceQuote& source()const noexcept{return source_;}
    const SourceCheckpoint& owner()const noexcept{return *owner_;}
    bool permits_zero_head_metadata(const RadModelMeta&,const RadBuildCtx&)const noexcept;
    bool engine_eligible()const noexcept{return false;}
private:
    SourceQuote source_;
    std::shared_ptr<const SourceCheckpoint> owner_;
    ColdSourceSeal(const SourceQuote& q,std::shared_ptr<const SourceCheckpoint> owner):source_(q),owner_(std::move(owner)){}
    friend int seal_source(const InspectedSource&,std::unique_ptr<const ColdSourceSeal>*,std::string*);
    friend class BoundSources;
};
// Atomic source-only seal. A caller cannot construct a sealed token with a public
// boolean. This synthetic fixture fingerprint is change detection, NOT auth.
int seal_source(const InspectedSource&,std::unique_ptr<const ColdSourceSeal>*,std::string* why=nullptr);
int quote_admission(const ColdSourceSeal&,const BackendQuote&,AdmissionQuote*,std::string* why=nullptr);
} // namespace glm53::engine_core_contract

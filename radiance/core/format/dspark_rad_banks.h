//Private source resolver. No Engine/CLI enablement in this first source packet.
#pragma once
#include "radfile.h"
#include "dspark_bank_contract.h"
#include "build/rad_build.h"
#include <set>
#include <string>
namespace rad {
class DsparkRadBanks {
public:
    DsparkRadBanks()=default;
    DsparkRadBanks(const DsparkRadBanks&)=delete;
    DsparkRadBanks& operator=(const DsparkRadBanks&)=delete;
    struct Entry {const RadFile* file=nullptr;const RadFileEntry* entry=nullptr;uint32_t bank=0;int64_t index=-1;};
    struct Manifest {std::string parent_metadata_sha256,parent_payload_sha256,source_identity;uint64_t parent_file_bytes=0;};
    //BORROWED owners must outlive this resolver, Programs, mappings and all transfer leases.
    //bind is quiescent and single-use after success; rebind is refused.
    //Dual bind validates metadata/parent-prefix identity; payload lineage is conversion/operator
    //provenance and is NOT rehashed or authenticated by this metadata resolver.
    int bind(const RadFile&canonical,const RadFile*draft,std::string*reason=nullptr);
    bool dual()const{return count_==2;}
    const Manifest& manifest()const{return manifest_;}
    const RadFile* file(uint32_t bank)const{return bank<count_?files_[bank]:nullptr;}
    int find(const std::string&name,Entry*out)const;
    int bound(const WeightInfo&weight,Entry*out)const; //checks immutable owner/generation/source key
    WeightSourceFn sources()const;
    int bind_weights(std::vector<WeightInfo>&weights,std::set<int64_t>*draft_entries=nullptr)const;
    //Call only after ALL rank declarations: no donor entry can be unused/overridden, and the
    //three complete128x2 expert groups are required. Native shape/math is a separate Program gate.
    int finish_coverage(const std::set<int64_t>&consumed)const;
    //Both bank epochs are checked, including a donor mutation during a warm canonical hit.
    bool same_epoch()const;
    int direct_owner_matches(uint32_t bank,int opened_fd)const;
    const dspark_bank::Snapshot& snapshot()const{return snapshot_;}
    static int metadata_digest(const RadFile&file,std::string*out);
    //Ordinary geometry only: every occurrence must be INT==want or exact decimal STR.
    //Manifest version/file-byte fields deliberately remain strictly typed INT.
    static bool expected_geometry_scalar(const RadFile&file,const char*key,int64_t want);
private:
    const RadFile* files_[2]{};
    uint32_t count_=0;
    dspark_bank::Snapshot snapshot_{};
    Manifest manifest_{};
};
} //namespace rad

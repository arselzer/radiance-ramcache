/* Read-only, bounded P1 route/copy evidence. Never an executor or a saving estimate. */
#pragma once
#include "ds41_phase2_routes.h"
#include <array>
#include <cstdio>
#include <cstdint>

struct RadBatch;
namespace rad {struct StepEntry;}
namespace rad { namespace ds41_phase2 { namespace audit {
inline constexpr const char* kEnv="RADIANCE_EXPERIMENTAL_DS41_P2_AUDIT";
inline constexpr uint64_t kMaxBytes=64ull<<20;
inline constexpr int kMaxFrames=512,kMaxSpans=16;
struct Limits {uint64_t bytes=kMaxBytes;int frames=kMaxFrames,spans=kMaxSpans;}; // host cases may only shrink
// Unset/empty/0 is off. An explicit absolute output prefix enables the audit.
const char* prefix();
bool requested();
bool prefix_valid(const char*);
struct Identity {
    bool valid=false;
    int32_t step=-1,card=-1;
    uint64_t sequence=0;
    int64_t begin=0,prompt_end=0;
    int32_t rows=0;
};
Identity make_identity(const RadBatch*,const StepEntry*,std::size_t entries,
                       const uint64_t* sequences,std::size_t sequence_count,bool generated);
struct Unit {
    int32_t unit=-1,tier=-1;
    int64_t native_bytes=0;
    routes::Residence residence=routes::Residence::Unknown;
};
using UnitMap=std::array<Unit,routes::kLocalExperts>;
struct Copy {
    int32_t expert=-1,unit=-1,rows=0,slot=-1,kind=-1;
    int64_t gu_bytes=0,dn_bytes=0;
    bool packed=false;
};
struct Totals {
    uint64_t copies=0,copy_bytes=0,duplicate_same_representation_bytes=0;
    uint64_t union_min_observed_bytes=0,union_max_observed_bytes=0,mixed_representation_units=0;
    uint64_t selected_nonresident=0,unique_selected_nonresident=0;
};
/* FILE is owned by the caller. All retained metadata is fixed-size. Every method is host-only;
 * device readbacks remain the FileStager's separately owned/fenced allocation. File failures,
 * malformed evidence and caps poison this audit, never the engine or its reply. */
class Trace {
public:
    Trace(FILE*,int rank,uint64_t program_cookie,uint64_t plan_cookie,Limits limits={});
    ~Trace();
    Trace(const Trace&)=delete;Trace&operator=(const Trace&)=delete;
    void bind(const Identity& i){next_=i;}
    void begin(int32_t step,int32_t rows,bool p1,const int32_t* row0,int32_t full);
    bool collecting()const{return piece_on_&&!failed_;}
    void frame(uint64_t generation,uint64_t producer_serial,int layer,int row_from,
               const routes::View&,const UnitMap&,bool p1_layer,uint64_t extra_bytes,bool reused_order,uint64_t counts_wait_ns=0);
    void copy(const Copy&);
    void layer_done(int layer,bool ok);
    void end(bool ok);
    void close(const char* reason="shutdown");
    void incomplete(const char* reason);
    bool failed()const{return failed_;}
    const Totals& totals()const{return totals_;}
private:
    struct Seen {
        int32_t unit=-1;uint8_t selected_mask=0,copy_mask=0;
        uint64_t minimum=0,maximum=0;
        int64_t gu=0,dn=0;bool packed=false,mixed=false;
        std::array<int64_t,routes::kMaxPieces> gu_by_piece{},dn_by_piece{};
        std::array<uint8_t,routes::kMaxPieces> packed_by_piece{};
    };
    FILE* fp_=nullptr;int rank_=-1;uint64_t program_=0,plan_=0,bytes_=0;
    Limits limits_{};
    bool failed_=false,piece_on_=false,span_on_=false,span_ok_=true,closed_=false;
    Identity next_{},piece_{};uint64_t epoch_=0,span_epoch_=0,last_epoch_=0;
    int pieces_=0,frames_=0,spans_=0,current_layer_=-1;
    int64_t span_begin_=0,span_end_=0;
    std::array<int32_t,routes::kLayers> row0_{};
    int full_=routes::kLayers;
    std::array<uint8_t,routes::kLayers> frame_seen_{},p1_layer_{},done_{};
    std::array<uint8_t,routes::kLocalExperts> copied_{};
    UnitMap current_map_{};std::array<int32_t,routes::kLocalExperts> current_counts_{};
    std::array<std::array<Seen,routes::kLocalExperts>,routes::kLayers> seen_{};
    routes::Frame frozen_{};Totals totals_{};
    bool room(uint64_t);
    void out(const char*,...);
    void common(const char*);
    void summary(const char*);
};
}}} // namespace rad::ds41_phase2::audit

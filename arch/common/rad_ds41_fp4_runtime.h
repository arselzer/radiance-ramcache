#ifndef RAD_DS41_FP4_RUNTIME_H
#define RAD_DS41_FP4_RUNTIME_H

#include <cstdint>
#include <cstring>
#include "rad_ds41_fp4_classifier.h"

/* Alias-only contract. No old operation gains an operand, parameter or buffer. */
namespace rad_ds41_fp4_runtime {
inline constexpr const char* kEnv = "RADIANCE_EXPERIMENTAL_DS41_FP4_CLASSIFY";
inline constexpr const char* kAuditEnv = "RADIANCE_EXPERIMENTAL_DS41_FP4_CLASSIFY_AUDIT";
inline constexpr const char* kDevice = "ds41.fp4_classifier.audit.device";
inline constexpr const char* kHost = "ds41.fp4_classifier.audit.host";
enum Op : int { Quant = 0, QuantCapture, Norm, NormCapture, Rope, RopeCapture, OpCount };
enum Field : int { Fast = 0, Fallback, CapturedFast, Blocks, Overflow, FieldCount };
struct Slot { Op op; int width; };
inline constexpr Slot kSlots[] = {
    {Quant,128},{Quant,512},{Quant,4096},{QuantCapture,128},{QuantCapture,512},
    {Norm,128},{NormCapture,128},{Rope,128},{Rope,512},{Rope,4096},
    {RopeCapture,128},{RopeCapture,512}
};
inline constexpr int kSlotCount = int(sizeof(kSlots)/sizeof(kSlots[0]));
inline constexpr int kWords = kSlotCount * FieldCount;
inline constexpr int kBytes = kWords * int(sizeof(uint64_t));
inline constexpr int slot(Op op,int64_t width) {
    for(int i=0;i<kSlotCount;++i)if(kSlots[i].op==op&&kSlots[i].width==width)return i;
    return -1;
}
inline constexpr const char* kNames[OpCount] = {
    "quant_roundtrip_fp4_classify", "quant_roundtrip_fp4_classify_kvpack",
    "rmsnorm_rope_roundtrip_fp4_classify", "rmsnorm_rope_roundtrip_fp4_classify_kvpack",
    "rope_rows_roundtrip_fp4_classify", "rope_rows_roundtrip_fp4_classify_kvpack"
};
inline constexpr const char* kRows[OpCount] = {
    "ds41_quant_fp4_classify", "ds41_quant_fp4_classify_kvpack",
    "ds41_norm_rope_fp4_classify", "ds41_norm_rope_fp4_classify_kvpack",
    "ds41_rope_fp4_classify", "ds41_rope_fp4_classify_kvpack"
};
inline bool parse(const char* s, bool* out) {
    if (!out) return false;
    if (!s || !std::strcmp(s, "0")) { *out = false; return true; }
    if (!std::strcmp(s, "1")) { *out = true; return true; }
    return false;
}
/* Integers/enums match only the existing producer's native configurations. Unknown sites use
 * the WHOLE original body before any kernel is issued; this is not a general FP4 replacement. */
inline constexpr bool quant_scope(int64_t n, int64_t block, int fmt, int scale, bool capture) {
    return fmt == 1 && ((n == 512 && block == 16 && scale == 1) ||
           ((n == 128 || (!capture && n == 4096)) && block == 32 && scale == 0));
}
inline bool declare_quant(bool on,int64_t n,int64_t block,const char* fmt,const char* scale,
                           bool capture,int kind) {
    if(!on||!fmt||!scale||std::strcmp(fmt,"fp4_e2m1"))return false;
    const int sc=!std::strcmp(scale,"ue8m0")?0:!std::strcmp(scale,"e4m3")?1:-1;
    return quant_scope(n,block,1,sc,capture)&&(!capture||kind==(n==512?1:2));
}
inline constexpr bool rope_scope(int64_t n, int64_t hd, int64_t half, int64_t off,
                                 int64_t pairs, bool gptj, float sign) {
    return gptj && sign == 1.0f && half == 32 &&
           ((n == 512 && hd == 512 && off == 448 && pairs == 32) ||
            (n == 128 && hd == 128 && off == 64 && pairs == 32) ||
            (n == 4096 && hd == 128 && off == 64 && pairs == 1024));
}
/* Host receipt predicate: no overflowing/wrapped counters are authority, and a capture alias
 * must account for every fast value whose nibble is reused. Zero rows are permitted for aliases
 * not present in a fused graph, but do not prove model coverage. */
inline constexpr bool valid_row(const uint64_t* p, bool capture) {
    return p && !p[Overflow] && p[CapturedFast] == (capture ? p[Fast] : 0) &&
           p[Fast] <= UINT64_MAX - p[Fallback] &&
           ((p[Fast] || p[Fallback]) ? p[Blocks] > 0 : p[Blocks] == 0);
}
}
#endif

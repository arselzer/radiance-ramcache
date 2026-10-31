/* vram_packed.h -- RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED: DeepSeek-V4.1's routed experts held in VRAM in the
 * packed-scale form instead of block_mxfp4.
 *
 * ---- why --------------------------------------------------------------------------------------
 *
 * The RAM cache already holds the CPU's experts in tools/ds41-host-experts' packed form
 * (ds41_host_experts.h, RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE): the release's codes plane unchanged
 * and each block's e8m0 as a 2-bit code from one base a weight, a round of 32 blocks with a scale
 * outside the window stored raw beside. 99.975% of the release's scales are 119..121, so with room for
 * 512 raw rounds a gate_up and 320 a down (all but 7 of the 15360 experts fit) an expert is 18,001,920
 * bytes a slot against block_mxfp4's 18,800,640: x1.0444. This holds the card's static residents the
 * same way, so the same weights pool holds ~31 more experts a rank (712/720 -> ~743/752 in the
 * recommended configuration with the FP8 dense weights; 598/615 -> ~624/642 without), each a CPU expert
 * or a drive miss fewer for the tokens that route to it.
 *
 * ---- how --------------------------------------------------------------------------------------
 *
 *   select  Engine::start sets the machine fact device.ds41_vram_packed (kVramPackFact) before
 *           declare, which makes libr4d's moe_gemm_ggml_mxfp4_gemv_vp row (kVramPackRow) take the
 *           routed GEMMs' decode band. That row declares the gemv row's stored form, block_mxfp4, so
 *           every other tier (the pinned pool, the file tier, the RAM cache and the CPU path, the
 *           expert profile's checks) keeps it untouched: the packed form is VRAM's alone.
 *   plan    A weight is eligible (vram_pack_eligible) when it is a trunk routed expert's ggml_mxfp4
 *           gate_up or down, whole on its rank, read only by moe_gemm_ggml tables whose bands all
 *           resolved gemv_vp, or whose M=1..16 bands resolved gemv_vp and every higher band the
 *           packed-aware P1 twin through its 4096-row bound. Other readers, including the ordinary
 *           LCP twin, keep the block form. The expert fill charges an eligible unit at its packed size,
 *           after reading its weights' scales from the container (vram_pack_scan): a weight with
 *           more raw rounds than its budget stays block_mxfp4 in VRAM and is charged so. Only a
 *           static plan (--deterministic) packs: a movable resident sits in a slot of its class's
 *           stride and the heat engine copies slots as they are.
 *   bank    With a DSpark bank bound (--draft-bank) the trunk's experts still come from the main
 *           container, which is the `file` here, and pack as without one. The bank's weights (file_id
 *           1, the draft layers') never do: Engine::start names kVramPackRow trunk-only
 *           (Registry::set_trunk_only_rows), so the draft's grouped GEMMs keep the plain gemv row and
 *           RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV's path, and vram_pack_eligible refuses a bank weight
 *           whatever reads it. RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM's hot set stays block_mxfp4,
 *           charged at its slots before the fill as without the switch; packed trunk residents free
 *           room for more trunk residents only.
 *   load    The loader packs each such weight from its planes (vram_pack_weight) and checks it
 *           before the upload: every block's e8m0, as the kernels widen it, equals the planes' byte
 *           for byte and the codes plane is unchanged. A mismatch refuses the load.
 *   issue   Ctx marks a packed weight's table entry with bit 0 (weights are 64-byte aligned).
 *           gemv_vp decodes a marked entry from the packed form and any other as block_mxfp4, by the
 *           same arithmetic in the same order, so every output is the block row's bit for bit, with
 *           the live-tail ranges (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL) off or on
 *           (tests/r4d_emu/test_ds41vp.cpp runs both on the host emulator, `tail` the second way;
 *           tools/ds41-host-experts/ds41_vpack_device_check.hip on a card, where ab_window.sh's Vp
 *           arms run only on a libr4d.so it passed).
 *
 * With the switch off nothing here runs: no fact, so the row never matches; no callback, so the
 * planner charges what it did; no packed weight, so no mark and no packing at load.
 */
#pragma once
#include "../rad_core.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

class RadFile;
class Plan;

/* libr4d's R4D_FACT_DS41_VRAM_PACKED and the row it selects (r4d_ds41_mxfp4.h, .inc), restated: the
 * core links no plugin. */
constexpr const char* kVramPackFact = "device.ds41_vram_packed";
constexpr const char* kVramPackRow  = "moe_gemm_ggml_mxfp4_gemv_vp";

/* The switch and the raw rounds a packed gate_up (raw[0]) and down (raw[1]) may hold: the RAM cache's
 * 512 and 320 by default (mover.cpp, kPackedRawDefault), which hold all but 7 of the release's 15360
 * experts. */
struct VramPackConfig {
    bool on = false;
    int  raw[2] = { 512, 320 };
};
/* RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1 and ..._VRAM_PACKED_RAW=<gate_up>,<down>. RAD_E_INVAL, logged,
 * for a value that is neither. */
int vram_pack_config(VramPackConfig* out);

/* One weight as the packed form would hold it. bytes 0: not eligible. */
struct VramPackWeight {
    int64_t  bytes = 0;          /* ds41hx::packed_bytes(N, K, cap) */
    int32_t  cap = -1;           /* raw rounds the form has room for */
    int32_t  N = 0, K = 0;
    uint64_t scales_off = 0;     /* the scales plane in the container */
    int64_t  scales_bytes = 0;
};

/* Per weight of `prog` (indexed by rad_weight, 0 the sentinel), what the packed form takes in VRAM, from
 * the declaration and the container's directory alone. `why`, when set, gets a summary of the routed
 * ops whose weights are not eligible (empty when all are), for the startup log. */
void vram_pack_eligible(const Program& prog, const RadFile& file, const VramPackConfig& cfg,
                        std::vector<VramPackWeight>* out, std::string* why);

/* The raw rounds w's scales need (ds41hx::pack_plan's n_exc), read from the container into `buf` and
 * dropped from the page cache after; -1 when the read or the plan fails. */
int vram_pack_scan(const RadFile& file, const VramPackWeight& w, std::vector<uint8_t>* buf);

/* The planner's question (PlannerInput::vram_pack) answered from vram_pack_eligible's table: the bytes
 * weight `w` takes on the card packed, 0 when it keeps its stored form. Without `scan`, from the table
 * alone; with it, after vram_pack_scan, so 0 also when its scales need more raw rounds than its room
 * (`*raw` says how many, -1 when they could not be read). `cap` gets the room. */
int64_t vram_pack_query(const RadFile& file, const std::vector<VramPackWeight>& table, rad_weight w, bool scan,
                        int32_t* cap, int32_t* raw, std::vector<uint8_t>* buf);

/* Packs one weight's planes (codes [N, K / 2], scales [N, K / 32]) into dst_bytes (packed_bytes(N, K, cap))
 * at dst and checks the result: the scales' round trip through the kernels' widening, and the codes
 * plane byte for byte. RAD_OK; RAD_E_FULL when the scales need more than cap raw rounds; RAD_E_STATE
 * when the check fails; RAD_E_SHAPE for a shape the form cannot hold. `n_exc` gets the raw rounds. */
int vram_pack_weight(const uint8_t* codes, const uint8_t* scales, int N, int K, int cap, uint8_t* dst,
                     int64_t dst_bytes, int* n_exc);

/* With a DSpark bank bound, after the plan: what the bank's side holds, and the check that none of it went
 * the packed way. A draft op is a moe_gemm_ggml reading a bank weight (file_id != 0) or a draft layer's (at
 * or past meta.n_layers). RAD_E_STATE, with `why`, when a draft op's band resolved kVramPackRow or a bank
 * weight is planned packed (Placement::vram_bytes); RAD_OK otherwise. */
struct VramPackBankReport {
    int64_t draft_ops = 0;       /* moe_gemm_ggml ops over the draft's experts */
    int64_t bank_weights = 0;    /* weights bound to the bank */
    int64_t bank_vram = 0;       /* ...of them VRAM residents, block_mxfp4 */
    int64_t trunk_packed = 0;    /* the main container's weights planned packed */
};
int vram_pack_bank_check(const Program& prog, const Plan& plan, VramPackBankReport* out, std::string* why);

}  /* namespace rad */

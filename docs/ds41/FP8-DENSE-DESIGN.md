# FP8-stored dense weights: implementation plan

# FP8-stored dense weights for ds41 (`gemm_nt_bf16f8`): file-by-file plan

**Snapshot.** Line numbers refer to `/home/as/radiance-ds41-claude` at HEAD `a458256f`, plus the uncommitted `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN` edits in `arch/deepseek41/deepseek41.cpp` and `arch/common/rad_block_moe_fp8.h` as of 29 Oct, about 11:25 UTC. The tree moved twice while I was reading it, so if a number has drifted, search for the symbol named next to it. I changed no file in the repository. Everything I verified (CPU only, single thread, each run at most 3 s and 160 MB RSS) is in `/home/as/.claude/jobs/635603c1/tmp/fp8design/plan/` (see the appendix).

## 0. Decisions

| Question | Decision | Why |
|---|---|---|
| Where the FP8 bytes come from | Keep `model.rad` as it is. At load, libr4d's relayout re-encodes each opted-in plain-bf16 weight and checks every element. | Bit identity stays tied to the bytes the base engine reads. No 416 GB reconvert is needed. A relayout may derive values "a reader of the planes could derive", such as a block maximum (abi/rad_abi.h:111-117), provided it never rounds. |
| How the engine opts in | A new op, `gemm_nt_bf16f8`, with exactly `gemm_nt`'s schema. It is declared only at the FP8-sourced call sites, behind `RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1`. | (1) A layout hook sees neither data nor a weight name (rad_abi.h:129-135). (2) A new dtype string on `gemm_nt` cannot be probed, because libref's `ref_gemm_nt` is a SIMPLE row with no constraints (libref/ref_registry.cpp:2469, 2209-2210). (3) A `gemm_*` name is declined automatically for runtime-LoRA targets (core/build/rad_builder.cpp:747-773). (4) A shape-keyed switch would also catch the bf16-sourced layer-20 compressor kv, whose geometry matches attn_kv's. |
| Bands | One FP8 skinny row for every M. An FP8 tiled row comes later, for prefill speed. | Every band of an op must read one stored form (rad_builder.cpp:2197-2208). The tiled bf16 row is byte-identical to m64 by contract (r4d_gemm_bf16_nt_tiled.hip:14-17), so the skinny FP8 row is exact at every M. The A/B runs only M ≤ 16 (tools/ds41-host-experts/ab_window.sh:138). |
| Tuning | The FP8 rows are NOTUNE: SK always comes from `pick_sk`, called with the bf16 rows' own arguments. | Tune-cache rows are keyed per kernel name (core/plugin/select.cpp:331-343). The A/B homes have no `tune/` directory, so the bf16 rows also use the law. |
| Fallback | Per tensor, decided at declare time: prechecks plus an exclusion list fed by the offline scan. At load the relayout refuses a weight; it never degrades it. | The stored size is fixed at declare time (rad_builder.cpp:2242-2245). See §2.4. |
| Planner | No change. | `stored_bytes` → `weight_bytes` → statics → experts (§4). |

**Freed VRAM per card** (TP4, all 570 container entries, which hold the 610 release pieces):
- rank 0: 4520.0 → 2262.2 MiB, which frees 2.205 GiB;
- ranks 1-3: 4220.0 → 2112.1 MiB, which frees 2.059 GiB.

That is about 117-126 more resident MXFP4 expert units per rank. With `RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1` (commit 6154f175) the saving is 225 MiB smaller per rank.

## 1. Stored form, op, rows, opt-in

### 1.1 `libr4d/r4d_bf16f8_layout.h` (new; pure C, included by the host hooks and the kernels)

```c
#pragma once
#include <stdint.h>
#if defined(__HIPCC__) || defined(__HIP__)
#define R4D_BF16F8_HD __host__ __device__ __forceinline__
#else
#define R4D_BF16F8_HD static inline
#endif
enum { R4D_BF16F8_SMIN = 10, R4D_BF16F8_SMAX = 246 };   /* every product is a normal bf16 or +-0 */
R4D_BF16F8_HD int64_t r4d_bf16f8_ldws(int64_t K) { return (K / 32 + 7) / 8 * 8; }
R4D_BF16F8_HD int64_t r4d_bf16f8_scale_off(int64_t N, int64_t K) { return (N * K + 255) / 256 * 256; }
R4D_BF16F8_HD int64_t r4d_bf16f8_bytes(int64_t N, int64_t K) {
    return r4d_bf16f8_scale_off(N, K) + (((N + 31) / 32) * r4d_bf16f8_ldws(K) + 255) / 256 * 256; }
/* byte of element (n,k): r4d_layout.cpp w8_slot (:695-702) at R4D_W8_KPB=32, without w8a16's +128 */
R4D_BF16F8_HD int64_t r4d_bf16f8_slot(int64_t K, int64_t n, int64_t k) {
    const int64_t t = n / 16, r = n % 16, kb = k / 32, kin = k % 32, s = kin / 16, kk = kin % 16;
    const int64_t lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    return ((t * (K / 32) + kb) * 32 + lh * 16 + r) * 16 + s * 8 + e; }
/* E4M3 code b times 2^(s-127) as bf16 bits (b not 0x7F/0xFF, s in [SMIN,SMAX]) */
R4D_BF16F8_HD uint16_t r4d_bf16f8_expand(uint32_t b, uint32_t s) {
    const uint32_t sign = (b & 0x80u) << 8, mag = b & 0x7Fu;
    if (mag == 0) return (uint16_t)sign;
    if ((mag >> 3) == 0) { const uint32_t p = mag >= 4 ? 2u : (mag >= 2 ? 1u : 0u);   /* subnormal m*2^-9 */
        return (uint16_t)(sign | ((s - 9u + p) << 7) | (((mag << (3u - p)) & 7u) << 4)); }
    return (uint16_t)(sign | ((mag << 4) + ((s - 7u) << 7))); }
```

**VRAM byte layout** for a weight of this rank's `[N][K]` (N%16 == 0, K%32 == 0):

- **Codes** start at byte 0 and take N·K bytes, as 512-byte packets.
  - Packet `t·(K/32)+kb` holds the 16-row tile `t` and the 32-K block `kb`. Within a tile row the K blocks are contiguous, so a wave's K slice is one run of (Ks/32)·512 bytes.
  - Lane `l = 16h + r` owns bytes `[16l, 16l+16)`.
  - Byte `8j+e` of a lane holds slot e of k step `2kb+j`, which is `W[16t+r][32kb+16j+8(e>>2)+4h+(e&3)]`. Slots 0-3 sit in the first dword and slots 4-7 in the second, which is what `v_cvt_pk_f32_fp8` selects as its lo and hi byte pairs.
  - This is the bf16 fragment slot order (libr4d/r4d_gdn_wmma.h:2-6).
- **Scales** start at `round_up(N·K, 256)`.
  - They form `ceil(N/32)` rows with a pitch of `ldws = round_up(K/32, 8)` bytes.
  - Byte `[nb][kb]` is the block's E8M0 exponent s, meaning 2^(s-127), with s in [10, 246]. Padding bytes are 0.
- **Total** is `r4d_bf16f8_bytes(N,K)`, with alignment 256.
- **Example:** q_a [1280,5120] takes 6,560,000 bytes, which is 1.001 bytes per element.

### 1.2 `libr4d/r4d_args.h:263`
After `R4D_DT_M2XFP4_FRAG`, add `R4D_DT_BF16F8 = RAD_DT_PLUGIN_BASE + 10` (bf16 weight stored as E4M3 codes plus E8M0 32×32 exponents, read by gemm_nt_bf16f8). The layout tag is `"r4d.bf16f8.e4m3.e8m0b32.frag32"`.

### 1.3 `libr4d/r4d_rows.cpp`

- **Constraints**, after `cGemmNtTiled` (:4712-4714):
  ```cpp
  static const RadConstraint cGemmNtBf16F8[] = { C_LE("M", 65536), C_DIV("N", 16), C_DIV("K", 32), C_IN("dtype", "bf16") };
  static const RadConstraint cGemmNtBf16F8Tiled[] = { C_GE("M", 65), C_LE("M", 65536), C_GE("N", 128), C_DIV("N", 16), C_DIV("K", 64), C_IN("dtype", "bf16") };   // commit 9
  ```
- **Schema**, after `SCHEMA("gemm_nt", ...)` (:3264-3265). It reuses `pGemmNt`/`oGemmNt` (:757-767):
  `SCHEMA("gemm_nt_bf16f8", pGemmNt, oGemmNt, "gemm_nt over a bf16 weight every 32x32 block of which is E4M3 codes times one power of two: gemm_nt's bytes; the kernel stores the codes and an E8M0 exponent a block, and a weight not exactly so is refused at load.")`
- **Row**, after `gemm_bf16_nt_tiled` (:7698-7707). The field order follows rad_abi.h:435-476:
  ```cpp
  { "gemm_bf16f8_nt_m64", "gemm_nt_bf16f8", "gemm",
    "gemm_bf16_nt_m64 over E4M3 codes + E8M0 32x32 exponents expanded in registers to the same bf16 operand",
    "any M; N % 16, K % 32; split by gemm_bf16_nt_m64's law", "bf16 A/C, fp32 accumulate. BYTE-IDENTICAL to gemm_bf16_nt_m64",
    RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY, ROW(cGemmNtBf16F8), NOTUNE,
    nullptr, nullptr, r4d_rad_gemm_bf16f8_m64, nullptr, r4d_rad_layout_bf16f8, r4d_rad_relayout_bf16f8,
    r4d_shape_gemm_nt, r4d_rad_unrelayout_bf16f8 },
  ```
- **Chaining:** add `"gemm_bf16f8_nt_m64"` to `kChained`, next to the bf16 rows (:10124). It has an r4dchain form for every instantiation. Concurrency already defaults to on (:10084-10096).

### 1.4 `libr4d/r4d_plugin.h`
Add `R4D_HOOKS(bf16f8)` to the list at :615-624, and add `int r4d_rad_gemm_bf16f8_m64(const RadArgs*, RadStream);` (later also `..._tiled`).

### 1.5 `libr4d/CMakeLists.txt:136`
Add `"r4d_gemm_bf16f8_nt_m64|"` after `"r4d_gemm_bf16_nt_tiled|"`. The plugin flags already force `-ffp-contract=off` (cmake/RadAddPlugin.cmake:155-165).

### 1.6 Opt-in: env knob, gate and per-weight marker

**`arch/deepseek41/deepseek41.cpp`**

1. **Constant:** `static constexpr const char* kDenseF8 = "gemm_nt_bf16f8";`
2. **Model fields** next to `wty` (:478): `bool f8_on = false; std::set<rad_weight> f8; std::set<std::string> f8_skip; int64_t f8_codes = 0;`
3. **`dense_f8_gate(b, meta, &m.f8_skip)`**, called in `declare()` right after `DS_TRY(geometry(m, ctx))` (declare is at :1877). It returns on only when all of these hold:
   - `getenv("RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8")` is exactly `"1"` (the same style as the ENGRAM_SPLIT knob, :916);
   - the container metadata reads `imported_quantization_config.quant_method == "fp8"`, `.scale_fmt == "ue8m0"` and `.weight_block_size == "32 32"` via `rad_meta_gets`. model.rad has exactly these (rad-info lines 40-44);
   - `rad_op_declared(b, kDenseF8)`.

   It parses `RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE` (comma-separated declared weight names) into `f8_skip`. If the knob is set but the gate fails, it prints a `rad_note`.
4. **`lw8(...)`**: the same signature as `lw()` (:613-640). It calls `lw()` and then marks the weight only if all of these hold:
   - `f8_on` is set;
   - the weight is not in `m.wty` (not GGUF);
   - its name is not in `f8_skip`;
   - `weight_enc` reports plain bf16 with one plane, matching LinearFP8's test (arch/common/rad_fp8.h:1115-1116);
   - `n % 16 == 0` and `k % 32 == 0`;
   - for a split weight (`world > 1 && shard != NONE`), `lo % 32 == 0` and `hi % 32 == 0`.

   Marking means `m.f8.insert(w); m.f8_codes += n*k;`.
5. **Call sites switched to `lw8`** (trunk only):
   - engram kv: :1055 (split) and :1057 (replicated);
   - q_a :1206, q_b :1208, kv :1215;
   - indexer q_b :1309;
   - wo_a groups :1384;
   - wo_b :1389.

   **Left on `lw`**:
   - compressor kv :1263, which is bf16-sourced and has attn_kv's shape;
   - indexer attn_k :1276 and indexer proj :1310;
   - token_embd/output :1916/:1918;
   - the DSpark stage and fc sites :1551-1568 and :1730 (v1).
6. **`linear()`** (:642-653). Move `fused()` (:690-698) above it or forward-declare it. Insert before the `gemm_nt` declaration:
   ```cpp
   if (m.f8.count(w))   // gemm_nt's bytes from the weight's E4M3+E8M0 stored form (libr4d)
       if (const rad_op h = fused(b, kDenseF8, { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k),
                                                 RAD_STR("dtype", m.g.dtype) }, { w }, { x }, { y }))
           return h;   // unresolved only for a runtime-LoRA target, whose declined op owns no weight
   ```
   This covers every reader of a marked weight, including all 8 linear() consumers at :1113/1120/1210/1212/1217/1311/1386/1391.
7. **PREFETCH_ATTN** (uncommitted, :1393-1396). The byte counts are computed as bf16 (`n*k*2`). For a marked weight they must be `n*k`, the codes; use a helper `wbytes(m, w, n, k)`. Otherwise `r4d_rad_prefetch` reads twice the stored size, because it cannot clamp a plugin dtype (libr4d/r4d_prefetch.hip:65-72; `rad_dtype_bits` returns 0 for plugin dtypes, abi/rad_types.h:184-188).
8. **MoE config** for the trunk (:1422-1442): set `s.cfg.shared_bf16_op = (m.f8_on && neither "blk.%d.ffn_{gate_up,down}_shexp.weight" is in f8_skip) ? kDenseF8 : nullptr;`. Do not set it for the drafter's config (:1642).
9. **Summary note** at the end of `declare()` on rank 0: `rad_note("deepseek41: %zu dense weights read by gemm_nt_bf16f8, %.2f GiB of codes")`.

**`arch/common/rad_fp8.h`** (LinearFP8 :711)

Add a field next to `stored`/`access` (:784-788): `const char* bf16_op = nullptr;`, with the comment "set before declare(): the op the bf16 branch declares instead of gemm_nt where served, same schema and bytes; null = gemm_nt".

In the bf16 branch (:1156-1193), before the `gemm_nt` RAD_OP at :1187, declare `bf16_op` with identical params and `RAD_WEIGHTS(w)` when all of these hold:
- `bf16_op` is set and `plain` is true;
- `N%16 == 0` and `K%32 == 0`;
- `span_lo%32 == 0` and `span_hi%32 == 0`;
- every row part is a multiple of 32;
- `rad_op_declared(b, bf16_op)`.

If the op resolves, keep it. Otherwise fall through to gemm_nt. `step()` (:1567+) issues `op` unchanged. GLM and Qwen never set the field.

**`arch/common/rad_block_moe_fp8.h`**
- Add `Config::const char* shared_bf16_op = nullptr;`, next to `shared_ungated` (:194).
- Before `sh_gate_up.declare` (:1471), set `sh_gate_up.bf16_op = sh_down.bf16_op = c.shared_bf16_op;`.

At TP4 the shared slices are [0,768) [768,1280) [1280,1792) [1792,2304) (deepseek41.cpp:752-760). They are 256-aligned, so the gate|up seam inside the stacked tensor also falls on a multiple of 32.

## 2. Load-time conversion (libr4d/r4d_layout.cpp, after the w8 section :671-755)

### 2.1 Layout hook `r4d_rad_layout_bf16f8`
- `operand != 1` (gemm_nt's `b`) → RAD_E_UNSUPPORTED.
- N and K come from `r4d_pgeti_or` (as `wgeom` does, :98-106).
- An encoding that is not `plain` bf16 with one plane → RAD_E_DTYPE. That makes the row a non-candidate (select.cpp:222-241).
- Any of these → RAD_E_SHAPE: `n != 1`, `!selects(enc,sel,n,"codes")`, `!dims(pl[0],N,K)` (libr4d/r4d_enc.h:78,92), `N%16`, `K%32`.
- Otherwise `fill(out, tag, R4D_DT_BF16F8, N, K, r4d_bf16f8_bytes(N,K))`, which sets align 256 (:111-123).

### 2.2 Relayout `r4d_rad_relayout_bf16f8`
The loader passes this rank's dense `[N][K]` bf16 plane (rad_abi.h:131-134; engine_bringup.cpp:2544-2580). The relayout checks that `dst_bytes == L.bytes`, then for every 32×32 block (the last block may have 16 rows):

1. **Derive the exponent from the block maximum**, as mxfp4's `b_ref` is derived: `s = max over nonzero x of (E(x) − 8 + (mant(x) > 0x60))`. Use 127 for an all-zero block, and clamp up to 10. If s > 246, refuse. This is the smallest s with max|x| ≤ 448·2^(s−127). Since a larger s only loses low bits, the minimum is the only s that can succeed.
2. **Encode each element.** Let `d = (x & 0x7FFF) − ((s−7) << 7)`.
   - If `(d & 0xF) == 0 && 0x80 ≤ d ≤ 0x7E0`, the code is `d >> 4` (a normal code from 0x08 to 0x7E; 0x7F/NaN can never occur).
   - Else if `(x & 0x7FFF) == 0`, the code is 0 (signed zero kept).
   - Else match one of the 7 subnormal images `r4d_bf16f8_expand(m, s)` for m = 1..7.
   - In every case OR in the sign bit.
3. **Verify every element:** `r4d_bf16f8_expand(code, s) == x`, otherwise fail. This check is the definition of exactness, so bf16 subnormals, NaN and Inf are all refused.
4. **Write.** The scale goes to `sc[(n0/32)·ldws + kb]`. Codes go to the tile packet `q + (((n0+i)/16)·(K/32) + kb)·512 + kPerm[r][j]`, where `kPerm[r][j] = (((j%16)%8)/4·16 + r)·16 + (j/16)·8 + ((j%16)/8)·4 + j%4` is a precomputed 16×32 table.

On failure the relayout prints `[r4d] bf16f8: (n,k) = 0x%04x is not an E4M3 code times 2^(s-127)` and returns RAD_E_FORMAT. The loader then fails, naming the weight and the tag (engine_bringup.cpp:2581-2586).

This output-ordered form produced the same bytes as the closed-form `r4d_bf16f8_slot` reference on real tensors (scratch `relayout_fast.cpp`).

### 2.3 Unrelayout `r4d_rad_unrelayout_bf16f8`
It writes `pl[0]` as `r4d_bf16f8_expand(code[slot(n,k)], scale[n/32][k/32])`, for tools (rad_abi.h:161-171).

### 2.4 Fallback policy
**Per-block fallback cannot be expressed.** The hook runs at declare time with `data == null` (rad_abi.h:129-135), and its byte count fixes `stored_bytes` and so placement (rad_builder.cpp:2242-2245; planner.cpp:89-97). A load-time BF16 fallback would need bytes that were never reserved.

**Per-tensor fallback works like this:**
- **(a) Declare-time prechecks** keep a weight on `gemm_nt`: not plain bf16, misaligned slice, GGUF, drafter.
- **(b) `RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE`** keeps named tensors on gemm_nt BF16. It is populated from the offline scan's report.
- **(c) The load-time check is the safety net.** It refuses; it never stores a wrong number.

If the scan ever finds failures, a "bandwidth-only" variant is possible: allocate a bf16-sized blob and mark fallback blocks with scale byte 0xFF (E8M0 NaN) plus raw bf16 halves. That gives exact per-block fallback but saves no VRAM. Do not build it unless needed.

### 2.5 Cost
- **Single-thread speed:** 5.5-5.9 ns/element for the naive reference and 3.0-6.8 ns/element for the output-ordered form, measured on the busy box.
- **Total work:** about 2.2G elements per rank, so 8.8G for 4 ranks, roughly 25-50 CPU-s. The RelayoutPool has `min(16, hw−2)` workers (engine_bringup.cpp:3320-3323), so that is about 2-3 s of wall time, overlapped with a ~9 s container read. Not yet measured in the engine.
- **Upload path:** dense weights move from direct DMA to staging, then relayout, then upload (engine_bringup.cpp:2549-2592; slots are 4 MiB, and anything above 256 MiB goes to a heap buffer, :2353-2360). The engram FP8 blob, at 157 MB, uses the reused `b.out` buffer.

## 3. Kernel: `libr4d/r4d_gemm_bf16f8_nt_m64.hip` (new)

### 3.1 Shared source, ISA-identical (commit 1)
Move the following from `r4d_gemm_bf16_nt_m64_body` (:54-132) into macros in a new `libr4d/r4d_gemm_m64_shared.h`:
- `R4D_M64_WAVE_SETUP()` = lines 59-69 (`red`, tid/wave/lane, col_local/ksplit, nt/n0/mt0, Ks/ksteps/kbase, acc=0);
- `R4D_M64_A_OFFSETS()` = :74-77;
- `R4D_M64_REDUCE_STORE()` = :111-130.

The bf16 body keeps `woff` (:73) between the first two macros. Inside the macros, `#pragma unroll` becomes `_Pragma("unroll")`.

**Verified:** I compiled the original and the refactored file for gfx1201 with the build's own flags. All 8 kernels compile to identical instructions; only the `__hip_cuid_*` symbol differs, which the real build pins with `-cuid`. By contrast, two function-based versions changed the address arithmetic (118 and 447 diff lines), so use macros.

### 3.2 Body (compiled with the build flags; scratch `r4d_gemm_bf16f8_nt_m64.hip`)

```cpp
template <bool HI> __device__ __forceinline__ unsigned f8pair(unsigned q, float f) {
  const r4d_f2v v = __builtin_amdgcn_cvt_pk_f32_fp8(q, HI);                 // exact, subnormals and +-0 too
  return __builtin_amdgcn_perm(__float_as_uint(v.y * f), __float_as_uint(v.x * f), 0x07060302u); }  // high halves
__device__ __forceinline__ v8bf f8frag(unsigned d0, unsigned d1, float f) {  // == fragU's 4 VGPRs
  r4d_u4v r; r.x = f8pair<false>(d0, f); r.y = f8pair<true>(d0, f); r.z = f8pair<false>(d1, f); r.w = f8pair<true>(d1, f);
  return __builtin_bit_cast(v8bf, r); }
__device__ __forceinline__ float e8m0(unsigned s) { return __uint_as_float(s << 23); }

template <int MT, bool CHAIN, int KU, bool SALIGN>     // KU 16 at MT=1, 8 at MT 2..4; SALIGN = (K/SK) % 128 == 0
__device__ __forceinline__ void r4d_gemm_bf16f8_nt_m64_body(const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Wb, __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK,
    int LDA, int LDC, long long soff, int ldws) {
  if (CHAIN) r4d_chain_wait();
  R4D_M64_WAVE_SETUP();
  R4D_M64_A_OFFSETS();
  if (n0 < N) {
    r4d_gu2 pa[MT]; for i: pa[i] = (r4d_gu2)(A + aoff[i]);                       // fragU's A, unchanged
    const int ntu = __builtin_amdgcn_readfirstlane(nt), kb0 = __builtin_amdgcn_readfirstlane(kbase) >> 5;
    r4d_gq4 pw = (r4d_gq4)Wb + ((size_t)ntu * (K >> 5) + kb0) * WAVE32 + lane;  // 16 B a lane, 512 B a wave
    r4d_gb1 ps = (r4d_gb1)Wb + soff + (size_t)(ntu >> 1) * ldws + kb0;          // wave-uniform scale row
    int s = 0;
    if (!SALIGN && (kbase & 16)) { q = pw[0]; f = e8m0(rfl(ps[0])); b = f8frag(q.z, q.w, f);     // peel: 2nd half
      for i: { acc[i] = mma(fragU(pa[i], 0), b, acc[i]); pa[i] += 4; }  pw += 32; ps += 1; s = 1; }
    for (; s + KU <= ksteps; s += KU) {
      r4d_u4v q[KU/2]; for j: q[j] = pw[j * 32];                                 // KU/2 x global_load_b128, imm offsets
      unsigned sw[KU/8]; for j: sw[j] = rfl(SALIGN ? ((r4d_gu1)ps)[j] : bytes ps[4j..4j+3]);   // s_load_b64 when SALIGN
      for u < KU: { f = e8m0((sw[u>>3] >> (8*((u>>1)&3))) & 0xFF);
                    b = (u & 1) ? f8frag(q[u>>1].z, q[u>>1].w, f) : f8frag(q[u>>1].x, q[u>>1].y, f);
                    for i: acc[i] = mma(fragU(pa[i], u), b, acc[i]); }               // same builtin, A=src0, W=src1
      pw += (KU/2) * 32; ps += KU/2; for i: pa[i] += KU * 4; }
    for (; s < ksteps; s += 2) {                                                 // whole blocks, then a half block
      q = pw[0]; f = e8m0(rfl(ps[0]));
      b0 = f8frag(q.x, q.y, f); for i: acc[i] = mma(fragU(pa[i], 0), b0, acc[i]);
      if (s + 1 < ksteps) { b1 = f8frag(q.z, q.w, f); for i: acc[i] = mma(fragU(pa[i], 1), b1, acc[i]); }
      for i: pa[i] += 8; pw += 32; ps += 1; }
  }
  R4D_M64_REDUCE_STORE();
  if (CHAIN) r4d_chain_done();
}
```

The launch is `gemm_m64_launch`'s geometry, verbatim (r4d_gemm_bf16_nt_m64.hip:159-185): grid `(ceil(ntiles/WV), ceil(mtiles/MB))`, block `WV·SK·32`, smem `WV·SK·256` floats, with the same checks plus `N%16` and `K%32`. It passes `soff = r4d_bf16f8_scale_off(N,K)` and `ldws`. The MB switch picks `MT=1 → KU 16` and `MT 2..4 → KU 8`, × SALIGN × chain, which is 16 kernels, all `__launch_bounds__(1024)`.

**Entry `r4d_rad_gemm_bf16f8_m64`:**
- Calls `gemm_ops(a, &g, 0, 0, /*pitched=*/1)`, the same handling as the bf16 path, including wo_a's `bcol` A/C views.
- If `b->dtype == RAD_BF16` (a cold tool's logical form), it calls `r4d_rad_gemm_bf16_m64(a, s)`.
- If `b->dtype != R4D_DT_BF16F8`, it returns RAD_E_DTYPE.
- It applies the bf16 row's own M, `lda%4` and 32-bit-offset checks (:196-197).
- `cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16)` is literally :199.
- It then launches with `c.wv, c.sk, c.mb` and the `chain` param.

### 3.3 What stays identical and what changes
**Identical:**
- the wave geometry and K slicing;
- A fragments (rowGo/fragU, LDA);
- the `mma(fragU(A), B, acc)` builtin and operand order (r4d_gdn_wmma.h:98-100);
- one chain from `(v8f)0` per (tile, split, row tile) in ascending k steps;
- the SK (same `cfg_for` call; NOTUNE gives wv = 1, sk = 0, mb = 0 exactly as tGemmSkinny's defaults at r4d_rows.cpp:5180-5186 via `r4d_cfg_gemm` :10194-10200);
- the LDS split-order sum from 0.f and the RNE `(__hip_bfloat16)` store (the same macro source);
- grid, block and LDS.

**Changed:** the B registers (proven equal), the unroll KU (it changes the unroll, never the order), two extra kernel args, and the dtype check. MB and WV are bit-neutral under BR_FLAT (r4d_gemm_args.h:129-132).

### 3.4 Register budget and instruction mix (gfx1201, clang 21, measured from the compiled .s)

| | VGPR | SGPR | Spills |
|---|---|---|---|
| MT=1/KU16/SALIGN | 104 (chain) / 109 | 29-34 | none |
| MT=1, not SALIGN | 116-117 | | none |
| MT 2-4/KU8 | 102-121 | ≤40 | none, no scratch |
| bf16 m64, for comparison | 78-79 (MT=1), 93-96 (MT 2-4) | | |

One 16-step round at MT=1 is 346 instructions:
- 8 `global_load_b128` (W: 4 KiB a wave, the same bytes in flight as bf16's KU=8 round);
- 32 `global_load_b64` (A);
- 1 `s_load_b64` (8 scale bytes);
- 16 `v_wmma`;
- 64 `v_cvt_pk_f32_fp8` (e32 + e64 op_sel);
- 128 f32 multiplies, issued as 58 VOPD `v_dual_mul_f32` + 12 single;
- 64 `v_perm_b32`.

That is about 21.6 instructions per k step, against about 9 for bf16. Compute stays well under the memory time at M ≤ 16. The latency-bound small shapes (kv: 128 waves) will gain less. NT loads and 8 KiB per wave are later bit-neutral knobs.

### 3.5 Alignment and access
- The blob is 256-aligned (`gemm_ops` checks 16 B, r4d_gemm_args.h:369).
- Each lane reads one aligned 16-byte chunk per block, and a wave request is 512 contiguous bytes. A round is one base plus immediate offsets 0..3584.
- A 16-row × 16-K step always lies inside one 32×32 block, so there is one wave-uniform scale per step.
- SALIGN holds for all 610 shapes. Every Ks is a multiple of 128 (Ks ∈ {2560, 1280, 2048, 768, 512, 6144}), so no shape needs the peel or the byte-wise scale path. Those paths are still tested.

### 3.6 Prefill tiled row (commit 9; not needed at `--max-num-batched-tokens 16`)
The new `gemm_bf16f8_nt_tiled` (prio 55, `cGemmNtBf16F8Tiled`, tGemmBf16Tiled `tile` only, same three hooks) copies or templates `r4d_gemm_bf16_nt_tiled.hip`. Only the W staging changes:
- Thread `tid` owns fp8 chunks `c = tid + 256i` (BN/64 chunks per K tile), where `l = c%32`, `j = (c/32)%2`, `tt = c/64`.
- It loads 16 B from packet `(min(n0/16+tt, N/16−1), 2kt+j)`, plus its scale byte, into registers. These replace `rw[]`, which held the next tile prefetched (:98-119).
- At the LDS store it expands each chunk into 4 `ds_store_b64` at `sW[r·72 + 32j + 16jj + 4h (+8)]`.

The resulting LDS words equal the bf16 staging's: my emulation found 0 differing words at BN 128/64, including a clamped tile. fragP, WMMA, `left/tot` split and store are unchanged (:120-166). The extra cost is about 60 VALU per thread per K tile, against 32 WMMA per wave. If templating the bf16 kernel changes its ISA, copy it instead.

## 4. Planner: no code change
The VRAM flows through the existing chain:
1. The hook's `L.bytes` becomes `stored_bytes` (rad_builder.cpp:2244-2245).
2. `weight_bytes` uses it (planner.cpp:89-97).
3. Dense units (`grp_layer`, `expert = -1`) are statics (planner.cpp:168-270).
4. `elastic = claimable − statics` (core/mem/vram_budget.cpp:143). Under `--deterministic`, used by the A/B (ab_window.sh:139), the no-lender branch hands it to experts: `exp = min(ceiling, elastic − kv_one)` (:170-175).
5. `weights = statics + exp` (:180), so the pool stays 15.31 GiB and only its split moves.
6. `room0 = vram_budget − fixed` (planner.cpp:794) fills more residents.

**Things to check in the window log:**
- static vram drops by about 2.06-2.21 GiB;
- "planned units VRAM=" (engine_bringup.cpp:2081-2085) rises by about 117-126 per rank.

**Notes:**
- Rank 0's extra ~300 MiB of statics is the uneven shared-expert split: gate_up +200 MiB, down +100 MiB.
- Dense units stay non-file-eligible at TP4, because their sharded members fail both file-tier tests (core/rad_core.h:253-256, 555-560; planner.cpp:405-409).
- If the cache lends (not `--deterministic`), the KV cache takes up to its ceiling first (vram_budget.cpp:156-161), which is tiny at 512 context.

## 5. What stays BF16, and how it is routed
- **bf16-sourced linears:** compressor kv (:1263), indexer attn_k (:1276), indexer proj (:1310), token_embd/output, router (f32), ratio-2 compressors (`gemm_nt_f32`) and the DSpark bank (v1). They stay on plain `lw()`, so linear() emits `gemm_nt`. Marking has to be per call site, because the layer-20 compressor kv has attn_kv's exact `[512,5120]` geometry.
- **M > 64:** the same op. Band [65,T] resolves to the FP8 skinny row (exact, slower) until commit 9 adds the FP8 tiled row. No band of a marked weight can resolve to a bf16 row, because the bf16 rows serve a different op, so mixed stored forms cannot occur; the builder would fail loudly anyway (rad_builder.cpp:2197-2208).
- **Fused consumers:** none read these weights.
  - r4d_fuse.cpp has no `gemm_nt` step.
  - The only gemm_nt sites in ds41 and MoeFP8 are linear() (:650), LinearFP8 (:1187), the f32 router (rad_block_moe_fp8.h:831) and the ungated-shared gate (:1500, which ds41 replaces with `fill`).
  - Any future reader must accept the same tag.
- **`prefetch`** (RAD_LAYOUT_ANY, r4d_rows.cpp:8753-8761) reads stored bytes as they are. It needs the byte-count fix (§1.6 item 7, commit 4).
- **Runtime-LoRA targets:** the builder declines `gemm_*` fused ops and clears their weights, so linear() falls through to gemm_nt over the bf16 store.
- **Host backend or libref-only builds:** the op is not declared, so the weights stay bf16.
- **Oracle and kbench:** there is no libref row, so they skip the row by name. The unrelayout is provided for tools.
- **`engine_probe` dumps** (engine_probe.cpp:195-199) would show the FP8 bytes as bf16. This only affects debug output.

## 6. Tests (details in test_plan)
- **CPU unit tests:** the codec exhaustive check, kernel-address emulation at every legal SK, refusals, round trip and a golden hash.
- **CPU scan** of all 570 entries / 610 pieces through the real hook.
- **CPU ISA diff** for commit 1.
- **GPU byte-compare:** one-hot exhaustive coverage of every code and scale, every ds41 shape × M × SK × chain, and real tensors.
- **r4d_selftest case.**
- **Engine A/B** with a reply-identity check.

## 7. Commits, in order

| # | Commit | Files | Est. |
|---|---|---|---|
| 1 | libr4d: share gemm_bf16_nt_m64's wave setup, A offsets and split-K reduce/store as macros; ISA-identical | new r4d_gemm_m64_shared.h; r4d_gemm_bf16_nt_m64.hip (:58-77, :109-130) | 1 h |
| 2 | libr4d: bf16f8 stored form and hooks, with CPU tests | new r4d_bf16f8_layout.h; r4d_args.h:263; r4d_layout.cpp (after :755); r4d_plugin.h:615-624; new tests/bf16f8_layout_test.cpp + tests/CMakeLists.txt:50 | 4 h |
| 3 | tools: CPU exactness scan of the container through the hook. Run it outside a GPU window and expect 570/570 exact. | new tools/ds41-host-experts/ds41_dense_fp8_scan.cpp, Makefile target, README | 2.5 h |
| 4 | libr4d: kernel, op `gemm_nt_bf16f8`, row, chain, docs, selftest case | new r4d_gemm_bf16f8_nt_m64.hip; CMakeLists.txt:136; r4d_rows.cpp (:4714, :3265, :7707, :10124); r4d_plugin.h; docs/OPS.md:723-728 (row + paragraph; required by tools/ops_doc_check.sh's UNDOCUMENTED rule); r4d_selftest.cpp | 5 h |
| 5 | libr4d: prefetch clamps an R4D_DT_BF16F8 weight to `r4d_bf16f8_bytes(shape)` | r4d_prefetch.hip:65-72 | 0.5 h |
| 6 | tools: GPU device check (byte compare vs gemm_bf16_nt_m64). Run it in a GPU window under both locks. | new tools/ds41-host-experts/ds41_bf16f8_device_check.hip | 4 h |
| 7 | arch/common: optional bf16-op override for LinearFP8 and the MoeFP8 shared arm (default null, so no change for GLM or Qwen) | rad_fp8.h:784-788, :1184-1192; rad_block_moe_fp8.h:194, :1471 | 1.5 h |
| 8 | deepseek41: `RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8` gate, lw8 marker, linear() branch, shared arm, prefetch byte counts, note; README knob | deepseek41.cpp (:478, :613-653, :690, the call sites in §1.6 item 5, :1393-1396, :1440, :1877+); tools/ds41-host-experts/README.md | 3 h |
| 9 | Engine A/B (best config ± knob; also with PREFETCH_ATTN and ENGRAM_SPLIT); NIGHT-LOG entry | launch_window.sh arms | 2 h |
| 10 | libr4d: `gemm_bf16f8_nt_tiled` prefill row + device-check cases at M ≥ 65 vs gemm_bf16_nt_tiled | new r4d_gemm_bf16f8_nt_tiled.hip (or templated tiled.hip), rows | 4.5 h |
| 11 | Optional core hardening: a `rad_decl_op` that resolves nowhere owns no weight, as the LoRA decline already does, so a probe can never mark a weight non-device in `resolve_sites` | rad_builder.cpp:820-835 + planner test | 1 h |

Later, if wanted: DSpark bank weights, once their provenance is checked; NT loads, 8 KiB per wave and a 64-K packet (all bit-neutral).

## Appendix: scratch evidence (`/home/as/.claude/jobs/635603c1/tmp/fp8design/plan/`)
- `cc.sh`: device-only gfx1201 compile with libr4d's exact flags (from /home/as/ds41-claude-build/aql/compile_commands.json).
- `base_m64.s` vs `refac5_m64.s` + `r4d_gemm_m64_shared.h`: the ISA-identity proof for commit 1.
- `r4d_bf16f8_layout.h`, `r4d_gemm_bf16f8_nt_m64.hip`, `f8_m64.s`, `f8_loop.txt`: the compiled kernel, register table and loop mix.
- `bf16f8_ref.cpp`:
  - the exhaustive codec check;
  - scale optimality;
  - kernel-address emulation over every legal SK × KU for 16×96, 48×96, 64×160, 32×512, 48×1280, 128×5120, 80×768 and 1024×576;
  - real tensors blk.0.attn_kv, blk.3.attn_output_a.g5 and blk.39.ffn_down_shexp: all exact, 0 mismatches.
- `relayout_fast.cpp`: the output-ordered relayout, byte-equal to the reference, with timings.
- `tiled_stage_check.py`: the FP8 tiled staging writes the same LDS words as bf16 staging.
- `entry_off.py`: looks up a container entry.

## Exactness argument

Claim: for every opted-in weight W (this rank's bf16 [N][K]) and every activation A, gemm_nt_bf16f8 writes the same bytes as gemm_nt, that is, libr4d's gemm_bf16_nt_m16 at M ≤ 16, m64 at 17-64, and tiled above 64.

1) The stored form is a lossless re-encoding.
- The relayout writes, for each element, an E4M3 code c and its 32×32 block's E8M0 exponent s, so that r4d_bf16f8_expand(c, s) == W[n][k] bit for bit. This includes ±0 (0x00/0x80) and E4M3 subnormals.
- Every element is checked. If any check fails, the whole weight is refused with RAD_E_FORMAT and the load fails (engine_bringup.cpp:2581-2586). No value is ever rounded.
- s is the block-maximum exponent, as mxfp4's b_ref is a derived maximum, which is the kind of value rad_abi.h:111-117 allows a relayout to derive. It is the smallest s with max|x| ≤ 448·2^(s−127). A larger s can only lose low bits, so if the minimal s fails, no s works.
- CPU check: for every s in [10,246] and all 65,536 bf16 patterns, the encoder accepts exactly the 60,198 images of the 254 non-NaN codes and none of the other 15,471,834 patterns, with 0 disagreements. On random weights built from known (codes, s), the derived s never exceeds the generating one.
- Real container tensors (blk.0.attn_kv, blk.3.attn_output_a.g5, blk.39.ffn_down_shexp) re-encode exactly. An earlier sample of 23,104 blocks across 8 weights found 0 inexact blocks. The full 570-entry scan is commit 3.

2) The kernel hands WMMA the same B registers as fragU.
- (a) Addressing. At k step u of the wave owning tile t and split j, lane l reads the code of W[16t+(l&15)][kbase+16u+8(e>>2)+4(l>>4)+(e&3)] for slot e, with slots 0-3 in dword 0 and 4-7 in dword 1. That is fragU's slot map (r4d_gdn_wmma.h:2-6; r4d_gemm_bf16_nt_m64.hip:41-46). The stored order is w8's (r4d_layout.cpp:695-702).
  - A CPU emulation of the kernel's exact pointer arithmetic covered pw/ps setup, the peel when kbase%32 == 16, the aligned and byte-wise scale paths, KU 16 and 8, and the half-block remainder. It ran over every legal SK of 8 synthetic shapes and pick_sk of 3 real tensors, with 0 mismatches.
  - A 16-row × 16-K step always lies inside one 32×32 block (n0 is 16-aligned and kbase is 16-aligned), so one wave-uniform scale per step is correct.
- (b) Value.
  - v_cvt_pk_f32_fp8 decodes every non-NaN E4M3 byte exactly. The result is a normal f32 with |v| ≥ 2^-9, or ±0 (r4d_fp8_wmma.h:92-127).
  - as_float(s<<23) is a normal power of two for s in [10,246]. The product is exact, and it equals the container's bf16 value, which the relayout required to be normal or ±0. So denormal flushing and rounding mode cannot apply.
  - The product has at most 4 significant bits, so its low 16 bits are 0 and its high half is the bf16 pattern. v_perm(.., 0x07060302) packs {slot 2i in the low half, slot 2i+1 in the high half}, which is the lm_bf16_pair / f2bf2 idiom (r4d_gdn_wmma.h:20-24).
  - Exhaustively emulated on CPU for 254 codes × 237 scales. The same pairs must also pass the GPU one-hot test on the deployed clang 21.1.8 / HIP 7.1 before any A/B.
  - The plugin build forces -ffp-contract=off and no fast-math (cmake/RadAddPlugin.cmake:155-165).

3) Everything else is the bf16 kernel.
- Literally the same source:
  - wave geometry and K slicing;
  - A fragments (rowGo/fragU over the same LDA, including wo_a's pitched bcol views);
  - the builtin __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 with A as src0 and W as src1;
  - one dependent chain from (v8f)0 per (tile, split, row tile) in ascending k steps (KU only changes the unroll);
  - the LDS reduction over splits in split order from 0.f, and the RNE (__hip_bfloat16) store.
- The shared macros were proven ISA-identical for the bf16 kernel: all 8 instantiations compile to the same instructions, and only the __hip_cuid symbol differs.
- SK comes from the identical call cfg_for(a, K, M, 16, BR_FLAT, &c, ceil(N/16)) (r4d_gemm_bf16_nt_m64.hip:199). With NOTUNE, r4d_cfg_gemm falls back to tGemmSkinny's defaults (wv 1, sk 0 = pick_sk, mb 0), which are exactly what the untuned bf16 rows get (r4d_rows.cpp:5180-5186, 10185-10200).
- MB and WV cannot change bits under BR_FLAT (r4d_gemm_args.h:129-132). The only WV-dependent input is pick_sk's cap = 1024/(wv·32), and wv = 1 on both sides.
- The m16 row is a pure forward to m64 (r4d_gemm_bf16_nt_m16.hip:14-16).

4) Bands.
- At M > 64, gemm_nt resolves to gemm_bf16_nt_tiled, which is byte-identical to m64 by contract: the same per-slice chains from 0, tot += acc in split order, the same cfg_for (r4d_gemm_bf16_nt_tiled.hip:14-17, 136-147, 219-222). So the FP8 skinny row matches at every M.
- The later FP8 tiled row stages the same bf16 words into LDS (emulation: 0 differing words at BN 128/64, including a clamped tile) and keeps fragP, WMMA, split and store verbatim.
- One stored form per weight across every band and every reading op is enforced by the builder (rad_builder.cpp:2197-2208).

5) End to end.
- Every dense GEMM output is byte-identical, and every other op is unchanged.
- The extra VRAM only adds resident experts. The CPU and card expert paths are already bit-exact; raising rank 0's residents from 374 to 556 kept every reply identical (/home/as/ds41-decode-review-20261028/NIGHT-LOG.md:78).
- The A/B's reply-identity check is the final gate.

Preconditions:
- (i) The bf16 rows also take SK from the law: no tune-cache rows for gemm_bf16_nt_{m16,m64,tiled} at these geometries and no RADIANCE_TUNE gemm_nt pin. This holds today, because the A/B homes have no tune/ directory and ab_window.sh sets no RADIANCE_TUNE.
- (ii) The HIP toolchain is unchanged.
- (iii) Only FP8-sourced, plain-bf16 weights whose slices start on 32-row/32-column boundaries are marked. Every TP4 split boundary is a multiple of 256.

## Test plan

Already done in scratch (CPU only; no repository changes):
- Exhaustive codec check: 15.5M (bf16, s) cases, 0 disagreements.
- Hardware-path emulation equals the integer expansion for all 254 codes × 237 scales.
- Kernel-address emulation: 0 mismatches at every legal SK × KU for 8 synthetic shapes, including peel and remainder cases, and on 3 real model.rad tensors (all exactly re-encoded).
- The output-ordered relayout is byte-equal to the closed-form reference.
- ISA-identity proof for the macro refactor of the bf16 m64 kernel.
- The FP8 kernel compiles for gfx1201 with 102-121 VGPRs and no spills.
- FP8 tiled staging emulation: 0 differing LDS words.

A. CPU unit tests (ctest; no ROCm). New tests/bf16f8_layout_test.cpp, added to RAD_TESTS_WITH_R4D_HOST (tests/CMakeLists.txt:50, which links r4d_layout.cpp at :187-188).
1. Exhaustive codec. For s in [10,246] and all 65,536 bf16 patterns, the relayout's encoder accepts x exactly when x is r4d_bf16f8_expand(c, s) for some non-NaN c, and expand(enc(x)) == x. Expect 60,198 accepted.
2. Emulated hardware path. Exact f32 decode × 2^(s−127), take the high 16 bits, and require them to equal r4d_bf16f8_expand, with zero low bits, for 254 × 237 pairs.
3. Kernel-address emulation (port bf16f8_ref.cpp's emulate()). Shapes 16×96, 48×96, 64×160, 32×512, 48×1280, 128×5120, 80×768 and 1024×576, at every legal SK in {1,2,4,5,8,10,16,20} and KU in {16, 8}, must give 0 mismatches against fragU's slot map. These shapes include the peel (K=96 with SK=2; K=160 with SK=10), the byte-wise scale path, N%32 == 16, and the half-block tail.
4. Refusals.
   - Layout hook: RAD_E_DTYPE for a non-plain or non-bf16 encoding; RAD_E_SHAPE for N%16, K%32 or wrong plane dims; RAD_E_UNSUPPORTED for operand ≠ 1.
   - Relayout: RAD_E_FORMAT for 0x3F81 (too many mantissa bits), 0x0001 (a bf16 subnormal), 0x7FC0 (NaN), 0x7F80 (Inf), and blocks that need s > 246; RAD_E_SHAPE for dst_bytes ≠ L.bytes.
5. Round trip: relayout then unrelayout gives the input bit for bit, including ±0 and subnormal codes.
6. Golden FNV-1a hash of the stored bytes of a fixed seeded weight, to pin the layout (as relayout_parity_test.cpp:365-367 does).
7. Byte count: r4d_bf16f8_bytes for the 9 ds41 shapes equals the table in plan §0, for example q_a = 6,560,000.

B. CPU-only exactness scan of all 570 entries / 610 pieces (commit 3; run outside any GPU window, under no locks, because it reads about 8.4 GB from the NVMe that the A/B's FileStager also uses). Tool: tools/ds41-host-experts/ds41_dense_fp8_scan.cpp, host only, built by the Makefile, compiling libr4d/r4d_layout.cpp the way the tests do.
- It parses the model.rad directory (abi/rad_format.h, as rad_dir.py does) and selects the entries named blk.N.{attn_q_a, attn_q_b, attn_kv, attn_output_a.gK, attn_output_b, ffn_gate_up_shexp, ffn_down_shexp, indexer.attn_q_b, engram_kv}.weight.
- It cross-checks that count (570) against the conversion's sources.jsonl census: 570 entries, 610 fp8_e4m3+e8m0 pieces, all "cast dtype=bf16".
- It preads each entry in 1024-row bands (whole 32-row blocks, bounded memory, ionice -c3, posix_fadvise DONTNEED). For each band it calls the real r4d_rad_layout_bf16f8 and r4d_rad_relayout_bf16f8 on a plain-bf16 [rows, K] tensor.
- It uses 4 threads. CPU time is about 25-50 s; I/O is bound by the drive.
- Per entry it reports: blocks, refused blocks with the first failure (n, k, bits, s), subnormal codes, +0/−0 counts, the s histogram, and a byte-count check.
- Globally it asserts that every TP2-4 split boundary is a multiple of 32 in both dimensions: q_b rows r·8192, wo_b columns r·2048, the shared 768/512 splits and the gate|up seam, and the engram split rows r·6400.
- Outputs: a JSON summary, plus the EXCLUDE list for RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE.
- Pass: 570/570 exact and an empty list.
- Optional audit: decode the release .weight/.scale from /srv/bulk and compare bitwise against the container's bf16. This confirms the cast; it is not needed for correctness.

C. CPU ISA checks (commits 1 and 10). Compile the base and refactored kernels device-only for gfx1201 with the build's flags (scratch cc.sh). Pass: no instruction differs, only the __hip_cuid symbol. If the tiled kernel's bf16 instantiation is not ISA-identical after templating, keep a separate file instead.

D. GPU byte-compare (commit 6). tools/ds41-host-experts/ds41_bf16f8_device_check.hip follows the ds41_mxfp4_device_check.hip pattern: dlopen the built libr4d.so, find the rows by name through rad_kernel_count/rad_kernel_at, and call their launch and hook pointers. Run it in a GPU window under both research locks (compiler, then heavy-runtime) with launch_window.sh or gpu_window.sh, on card 0.
1. Exhaustive one-hot.
   - Weight: W [256 × 1024], with 256 blocks whose scales sweep 10..246 and whose 1024 elements cycle through all 254 non-NaN codes. That is every (code, s) pair.
   - Activation: A is 16 one-hot rows, row m = 1.0 at k = 16q + m. 64 launches cover all of K, so every output equals one weight element.
   - The FP8 row's output (blob from its own relayout hook) must equal gemm_bf16_nt_m64's output on the bf16 W byte for byte, and both must equal the host-expected W bits (−0 may become +0 identically in both).
   - Repeat at every legal SK (pass the "sk" param; r4d_cfg_gemm reads it from RadArgs).
2. Shapes.
   - The per-rank shapes: 1280×5120, 8192×1280, 512×5120, 1024×4096 (pitched: lda 8192 and ldc 2048 via column-slice views), 5120×2048, 1536×5120, 1024×5120, 5120×768, 5120×512, 4096×1280, 25600×6144, 6400×6144.
   - M in {1,2,3,4,5,7,8,15,16,17,31,32,33,48,63,64,65,100,256}; chain in {0,1} (an unmatched record runs without waiting); SK = the law plus every other legal value; mb in {0,1,2,4}.
   - Random FP8-exact W (seeded) and random bf16 A, compared byte for byte against gemm_bf16_nt_m64. At M ≥ 65, also compare against gemm_bf16_nt_tiled, and against gemm_bf16f8_nt_tiled once it exists.
3. Real weights. One model.rad tensor per class, rank 0 and rank 3 slices (pread), relaid by the library's hook, at M = 1 and 16, compared byte for byte.
4. Odd shapes: N=16, K=96, SK=2 (peel); K=160, SK=10 (byte-wise scales); N=48; M not a multiple of 16.
5. Time. µs and GB/s per shape at M = 1 for bf16 vs FP8, 200 reps, rotating 8 weight copies so the 64 MB on-die cache stays cold.

Pass for 1-4: 0 differing bytes; the tool exits 1 on any difference. Add the same comparison at one shape as an r4d_selftest case for the row (r4d_selftest.cpp gemm_case pattern :874-896).

E. Engine A/B (commit 9; GPU window under both locks; /home/as/ds41-claude-build/work/launch_window.sh; never from a build shell).
- Arms:
  - the best configuration;
  - the same + RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1;
  - the same + PREFETCH_ATTN=1 (byte-count interplay);
  - the same + ENGRAM_SPLIT=1.
  Arm names must be unique.
- Checks:
  - (a) ab_sum.py shows every one of the 8 held-out replies identical to the base engine;
  - (b) the startup log has the rad_note "N dense weights read by gemm_nt_bf16f8", static vram about 2.06-2.21 GiB lower, and "planned units VRAM=" about 117-126 higher per rank;
  - (c) load-time delta, expected about +2-4 s from the relayout;
  - (d) decode tok/s and TTFT against the base. Research estimate: +10-14%; about 5% from bytes, the rest from residents. Unmeasured.
- Optionally run one arm per config with R4D_GEMMCFG=1 and diff the [gemmcfg] lines.

## Risks

1. Tune parity (bit risk).
   - The FP8 rows are NOTUNE and always use pick_sk. If a $RADIANCE_HOME/tune/<machine>.tune row or a RADIANCE_TUNE "gemm_nt:sk=..." pin gives gemm_bf16_nt_m16/m64 another SK, the knob-on and knob-off runs differ.
   - Tune keys are per kernel name and pins are per op (select.cpp:331-369).
   - Today: none exist in the A/B homes. Document this as a precondition, and in the production home check that the tune file has no gemm_bf16_nt rows at these geometries before enabling the knob.
2. Strict load. A single non-representable block fails the whole load, because there is no per-block fallback within the fixed stored size. Mitigation:
   - run the 570-entry scan first, through the same hook code;
   - put any failing tensors in RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE.
   The release format makes failures unlikely: 0 in 23k sampled blocks plus 3 full tensors.
3. PREFETCH_ATTN interplay (uncommitted, deepseek41.cpp:1393-1396).
   - Its byte counts are bf16 sizes. r4d_rad_prefetch cannot clamp a plugin dtype (rad_dtype_bits returns 0), so it would read twice the FP8 stored size past the weight: bandwidth waste and on-die cache pollution, and a possible page fault at the end of the pool.
   - Fix both sides (commits 5 and 8).
   - Also, FP8 roughly halves what the prefetch saves on those weights, so measure the two together.
4. Unresolved-probe hazard.
   - core/place/planner.cpp:145-164 marks every weight of an op that resolved nowhere as neither device- nor host-readable, and unresolved ops keep their weights (rad_builder.cpp:820-835).
   - linear() and LinearFP8 therefore precheck everything the FP8 row's constraints and hook test (plain bf16, N%16, K%32, 32-aligned spans). An FP8 declare can then fail only through the runtime-LoRA decline, which clears the op's weights (rad_builder.cpp:756-773).
   - The optional core commit 11 removes the hazard for every fused() probe.
5. Performance uncertainty.
   - About 2.4× the instructions per k step (21.6 vs 9). Small latency-bound GEMMs (kv: 128 waves; down: 320 waves × 32-48 steps) may gain little.
   - Launch overhead is unchanged. The decode gain (≈5% from bytes) and the residency gain are estimates.
   - Knobs to try (bit-neutral): NT loads, 8 KiB per wave, double-buffered W, a 64-K packet.
6. Prefill without the tiled FP8 row. Correct, but slow above 64 rows (the skinny kernel re-reads A per N tile). It matters only with --max-num-batched-tokens > 64; the A/B uses 16.
7. Load time and CPU.
   - Unvectorized relayout runs at 3-7 ns/element on a busy box: about 25-50 CPU-s for 4 ranks on the 16-worker pool, alongside the existing relayouts. Measure it.
   - Dense weights also move from direct DMA to host staging and upload.
8. Toolchain and hardware. v_cvt_pk_f32_fp8 and VOPD codegen are proven only on clang 21.1.8 / HIP 7.1 for gfx1201. Do not update HIP. The GPU one-hot test must pass before any A/B.
9. Tooling side effects.
   - docs/OPS.md must gain a row, or ops_doc_check fails with UNDOCUMENTED.
   - rad-kbench and the op oracle skip the row (no libref twin).
   - engine_probe dumps show the FP8 bytes as bf16.
   - opshapes has no size for the new op (only matters if someone tunes it).
10. Moving tree.
   - The line anchors come from a458256f plus uncommitted PREFETCH_ATTN edits, and parallel sessions keep committing (engram split 6154f175, DSpark beside a458256f).
   - ENGRAM_SPLIT reduces the saving by 225 MiB per rank.
   - The DSpark drafter's weights are excluded in v1. If they are added later, their wkv has two reader ops (query and context pass); linear()'s handle-keyed marker covers both.
11. TP1 or other worlds. At world 1 the replicated FP8 weights become eligible for the file tier's per-read relayout (rad_core.h:555-560, ≤ 32 MiB). That is not reachable for ds41 on these cards, but note the CPU cost if a unit is ever filed. Non-32-aligned spans at other TP counts are simply not marked.
12. ABI wording. The relayout derives a per-block exponent, which fits "derives only what a reader could derive" only because every element is verified. Add one sentence to rad_abi.h:111-117 (and spec §4.3) saying a narrower re-encoding is allowed iff it is verified element by element and refused otherwise.

Estimated effort: 29 hours

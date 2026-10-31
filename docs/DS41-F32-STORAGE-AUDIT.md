# DS41 redundant-F32 storage audit, 31 October 2026

Source snapshot `b089fde0` (native numerical graph remains the current0f62
cohort). Source inspection and sizing only: no build, test, model/weight read,
emulator, server or GPU operation. Index-score storage/fusion belongs to a
separate lane and is deliberately excluded here.

**Finding: no additional large VRAM F32 plane is proven to contain only
BF16-rounded floats. No runtime code is proposed.** The large already-rounded
planes are already BF16, while the remaining F32 producers store unrounded
projection sums, gates, probabilities or trigonometric values. A BF16 input or
weight does not make its accumulated F32 output BF16-rounded.

## Coverage and prior work

Checked actual DS41 declarations/issue sites, common MoE/router/FP8 helpers,
MXFP4 output stores, CPU/P1 paths, mHC and compressor kernels, head/sampler,
collective allocation/payload contracts, prefetch and relevant private scratch.
Reconciled HANDOFF/NEXT/LONG-CONTEXT, IMPL-ROUTER, FP8-IMPL-REPORT,
PREFILL-DESIGN/200-DESIGN/CRITIQUE/ENGINE-SURVEY and PHASE1-PLAN. Raw historical
captures, all old patches and every unrelated architecture were not re-read.

Already implemented/tried, not new opportunities:

- `Rb` stores eligible router/compressor WEIGHTS as high BF16 halves and
  expands by shift before the same FP32 arithmetic. The archive's checked
  40-router/three-compressor result is180MiB/card; it says nothing about
  rounding their activation outputs. Both `gemm_nt_f32` and `..._f32bf16`
  write raw FP32 sums.
- Dense FP8 weight storage, packed CPU/VRAM expert scales, compressed/index
  FP4 storage and the FP8-window candidate preserve representations at other
  already-established roundtrip boundaries. They are separate from a new
  F32 activation narrowing.
- P1 already replays GU/SwiGLU waves using BF16 GU/FF/down planes. Span work
  removes redundant replay coverage while preserving untouched rows, not a
  dtype conversion. Its stretched lifetimes remain necessary.
- Old layer-major retention estimates include an80KiB/F32 stream estimate.
  Actual current stream declarations are BF16,40KiB/row per stream plane.
  Reducing a historical F32 estimate is not an additional current saving.
  Layer-major/recompute work is still a separate implementation; approximate
  128-row replay is not eligible for the byte-exact base.

## Actual current buffer inventory

Values below are raw declared payload per TP4 rank, native D5120, FF2304,
top-k6, hc4, one sequence, no speculation, max_ctx32768. C is max_tok; other
sequence budgets use the shown formulas. These are NOT additive arena peaks:
transients overlay by liveness, and the scratch region is a maximum shared by
ops. No buffer-plan execution was performed, so no new net arena saving is
asserted.

| Plane/family | C2048 | C4096 | Writer/reader closure |
|---|---:|---:|---|
| `comp_stage`, `[2*(C+S),1024]` F32 |16.0078125MiB|32.0078125MiB|`qsa_work` copies original raw projections/tail state; `softmax_pool` consumes their full FP32 scores/values before BF16 output. Not redundant.|
| Two persistent RoPE planes at32K |16MiB|16MiB|Original `rope_table` stores FP32 sin/cos/YaRN; scalar/fused readers consume those exact floats. Separate backing design only.|
| `comp_kv_score`, `[C,1024]` F32 |8MiB|16MiB|Raw `gemm_nt_f32`/`..._f32bf16`; feeds stage/pool and raw tail store. No BF16 store boundary.|
| mHC split scratch, `C*20*25` floats |3.90625MiB|7.8125MiB|Raw chunk dot/square partials; finish adds in chunk order then gates/Sinkhorn. Not redundant; may share with larger scratch.|
| `router_logits`, `[C,384]` F32 |3MiB|6MiB|Raw replicated projection or raw split/gather; routing reads full precision. Excluded precision change.|
| Actual router `SplitF32Proj.part`, `[C,96]` |0.75MiB|1.5MiB|Raw projection slice, byte-gathered to router_logits. Output dtype is F32 despite parameter spelling `dtype=bf16` for inputs.|
| mHC pre-attn/pre-FF/post/comb, `C*(3*4+16)` floats total |0.21875MiB|0.4375MiB|Initial fills0/1 are not the only writers: `mhc_mix` writes raw sigmoid/exp/division/Sinkhorn. `mhc_post`/collapse and fused chains read full floats.|
| expert route weights, `[C,6]` F32 |48KiB|96KiB|Raw chosen score×normalizer; gather/fused gather uses FP32. Not redundant.|

The compressor does not currently declare a `SplitF32Proj` scratch plane;
extrapolating the router helper to it would invent a nonexistent allocation.
Scores/LSE/rawF32 compressor tails remain excluded as requested.

The final head logits use `R*32320*4 = R*129280` bytes/rank, where
`R=max(min(C,S*(max_spec+1)),max_out_rows)`. Ordinary single-sequence serving
has R1,126.25KiB; scored/KLD requests can enlarge R up to the configured row
bound (252.5MiB atR2048,505MiB atR4096). They are still ineligible:
`logits_gemm_kernel` writes the final raw FP32 sum, explicitly because BF16
mantissas are too short for sampling cuts. Sampler penalties/transforms may
also rewrite logits in FP32. Candidate values retain raw logits; logprob
summaries contain FP32 normalizers/probabilities. A large scored-head buffer
is not a hidden widened-BF16 plane.

## Important false positives and all-path checks

The following large native planes are already BF16 in the actual DS41
declarations, regardless of generic helper comments:

| Plane | C2048 | C4096 |
|---|---:|---:|
| Expert GU, `[C*6,4608]` |108MiB|216MiB|
| Expert FF, `[C*6,2304]` |54MiB|108MiB|
| Expert down, `[C*6,5120]` |120MiB|240MiB|
| One residual stream plane, `[C,4*5120]` |80MiB|160MiB|
| Full Engram KV intermediate, `[C,5*5120]` |100MiB|200MiB|

MXFP4 gemv/tall/Vp/tiled producers use a uint16 destination and `mx_put`
performs the original BF16 rounding. CPU experts produce uint16 GU/FF/down;
P1 stager explicitly admits BF16 operands and replays those same captured
buffers. Quant/fused norm/RoPE/attention and shared/routed gather output BF16
at their existing stores. There is no F32 copy to halve in these lifetimes.
`Geom::w_bf16=true` suppresses ActFP8 q/s/q8/s8 declarations in native DS41;
generic F32 activation-scale planes are not current allocations.

DSpark uses the same already-BF16 expert/stream pattern. Its F32 head logits,
unary candidates, pair/gather logits and observer fields contain true FP32
values. ID halves of pair records are integer bit patterns, not rounded
floats; global token IDs exceed16bits. A new split ID encoding would need a
separate contract and is tiny at the fixed block/candidate dimensions.

Exact BF16 all-reduce and Engram BF16 gather payloads already use2-byte
elements. FP32 accumulation is local ordered arithmetic, not an extra global
F32 shadow. F32 gathers for raw projections/candidates preserve their raw
values. Existing AR/AG caps/shared-family work targets their allocations;
it is not a new F32-half opportunity.

The MXFP4 GEMV's LDS activation tile widens BF16 to F32, so its low halves ARE
redundant for that narrow tile. However LDS is not retained VRAM and the tile
is not a persistent arena allocation. The old Phase3 design already proposes
BF16 LDS staging for larger row tiles. Converting it might change occupancy
and load cost, but cannot reclaim the requested VRAM and is not implemented
here. FP32 weight/dot tiles and partial accumulators are not covered by this
observation. The host SiLU table also stores unrounded `g/(1+exp(-g))`, not a
BF16-rounded float table, and lives in host memory.

## Decision

No new closed, valuable high16-storage candidate remains after excluding
scores, already-BF16 activations and required FP32 precision. No source kernel,
dtype, ABI, graph or lifetime changes were made. Keep effort on the independent
score-plane lane, exact packed caches/window, collective scratch caps and
measured P1/bank work. A future candidate needs an explicit final uint16/shift
store boundary on EVERY writer, every consumer/fusion/CPU fallback accounted
for, and an actual liveness/peak proof before any claimed recovered residency.

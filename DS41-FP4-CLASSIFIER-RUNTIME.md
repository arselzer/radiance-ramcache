# Default-off FP4 classifier runtime prototype

SOURCE ONLY. No build, test, import, syntax check, emulator, GPU or model run was performed by the
author. This branch starts at serving53 (`53fe48b0`) plus the standalone classifier helper6c67
cherry (`eda176be`). It does not modify serving53, the isolated KV free-order fix77f, published
defaults, sealed runners or prior evidence.

Root's actual four-card standalone component receipt in
`work/autonomous-20261101/fp4-component-run1/window/complete.json` proves the pointwise classifier
and its original plain native producer comparisons. It does **not** qualify these new aliases,
the cloned fused/N128 kernels, capture/store composition, audit lifetime, readers or serving.

## Construction and admission

`RADIANCE_EXPERIMENTAL_DS41_FP4_CLASSIFY=1` selects six new operation names and exports. Unset/0
preserve the original operation schemas, parameter order, operands, HIP functions and dispatcher
bodies. No public ABI header/structure is changed. Empty/other values refuse. The architecture's
initial feature scope is the native40-layer D5120 TP4 model, head_dim512/index_dim128/index_heads32
and rot64. Context/cache budgets are not fixed and no full-cache assumption is introduced.

| Alias family | Newly admitted site | Capture |
|---|---|---|
| `quant_roundtrip_fp4_classify` | n512/B16/E4M3; n128/B32/UE8M0; n4096/B32/UE8M0 | no |
| `quant_roundtrip_fp4_classify_kvpack` | n512/format1 or n128/format2, same blocks/scales | original nibble+scale |
| `rmsnorm_rope_roundtrip_fp4_classify` | n128, GPT-J rot64/off64, one head; F32 or BF16 gain | no |
| `rmsnorm_rope_roundtrip_fp4_classify_kvpack` | same n128, format2 | original nibble+scale |
| `rope_rows_roundtrip_fp4_classify` | latent n512/head512/off448; index n128/head128/off64; query n4096/32 heads128/off64 | no |
| `rope_rows_roundtrip_fp4_classify_kvpack` | latent512/format1 or index128/format2 | original nibble+scale |

Declaration predicates run before alias selection. Unknown widths, scales, blocks, rotations,
inverse forms, capture kinds or row ranges keep the **original operation name/schema**, preserving
its resolver chain. This feature does not register libref aliases. The global opt-in admission
refuses unconverted FP4 producers rather than claiming unsupported model configurations work.
The new direct-API dispatchers additionally call the whole original R4D body for an out-of-scope
site, before any launch; this is an extra numerical fallback, not inherited libref resolution.
Unknown parameter names still follow each unchanged schema's normal refusal rules.

Startup admission checks every actual band for the exact new row/export, one loaded libr4d DSO
origin/order, no tuning axes/choices and no host fallback. Actual malformed tensor views refuse.
The supported alias value count is bounded by INT_MAX rows and4096 columns, so each local/block
integer tally fits U64. Unsupported dimensions never partially issue RoPE before falling back.

The only arithmetic replacement is the final nearest-magnitude lookup on the **same**
`ds_clamp(v / scale,-6,6)` result. Finite |x|<=6 uses the standalone integer-bit classifier;
everything else calls the literal old `ds_e2m1_rt`, including NaNs and the old huge rounded ties.
All amax butterflies, floors, scale conversion, division, norm reductions, RoPE expressions,
intermediate BF16 boundaries, final multiplication and BF16 store helpers remain the original
expressions. The optional captured nibble reuses `Result.code`; fallback obtains the original code
from the old q value. No BF16 reencoding or FP8/WINDOW change occurs. N128 still uses the existing
default-off flag/forced checker forms and the old trailing-zero-wave additions.

## Actual-use audit and ownership

`RADIANCE_EXPERIMENTAL_DS41_FP4_CLASSIFY_AUDIT=1` requires CLASSIFY1. It adds two planner-owned
PERSIST I64[60] buffers:480B device counters and480B mapped-host readback. Each occupies a512B
planned region, with fixed context handles/bool recorded in the startup log. No new retained
payload or per-step allocation exists. Classify0 and audit0 allocate neither buffer. Audit0
aliases pass an absent optional final operand and compile Audit=false kernels with no counter
reductions, shared counter storage, atomics or reads. All original kernels remain untouched.

Each of12 alias+width slots owns five words: fast values, old-loop fallback values, fast
captured nibbles, nonempty workgroups, and overflow latch. Per-thread counts fold locally once at
the end of the kernel; lane0 totals fold per workgroup, followed by checked CAS additions. There
are no per-value global atomics. N128's inactive trailing waves reach the audit barrier with zero
counts, so partial eight-row blocks cannot strand a barrier. Overflow never wraps into authority.

Ctx validates ownership/shape/domain/alignment and zeroes the device buffer once, before either
lane may issue. Persistent arena offsets survive levels, lending and graph replay. Shutdown sets
the owning device, uses its existing main/second-lane drains, then copies exactly480B once into
the owned host mirror and waits for that copy before either arena is released. Failure, overflow
or capture mismatch emits FAILED and no complete receipt. Repeated shutdown cannot reread or
report. Twelve rank/op/width lines plus one complete line bind actual work to this process/context;
these are not host dispatch/startup assertions. There is no per-token D2H/synchronization.

Audit1's INOUT dependency and extra LDS/atomics may affect diagnostic overlap. Price all audit
overhead, match audit1 across functional comparisons, and never treat audit1 timing as quiet
serving performance. All counters cover the rank/op context lifetime, not an individual request.

## Required root validation

All execution belongs to root, serialized by the load rules. Host targets added here:

- `ds41_fp4_runtime_test`:5 cases for strict configuration, native/unknown admission, counter
  contracts, every-band DSO/export/role checks, tuning/host fallback refusals.
- `ds41_fp4_runtime_arch_test`:5 cases for full FLAG0 graph/issue equality, enabled alias
  normalization against every unchanged issue, fixed audit planes and exact full audit views,
  unavailable/bad flags and original-name unsupported declarations. Existing fixture-specific
  grouped-view exemptions remain narrow; other lint is checked after alias normalization.
- `ds41_fp4_audit_test`:3 cases, HOST backend and zero skips required: real arena ownership,
  startup zero/once-only teardown receipt, missing/duplicate/malformed planes, overflow/mismatch
  and disabled path. Also rerun existing runtime/build/arena, DS41 architecture and packed tests.
- `ds41_fp4_classifier_test`:the existing5 strict pointwise cases, unchanged. Root must also run a
  cross-source FLAG0 graph/issue driver against literal53; unset-vs0 tests alone are insufficient
  to certify a compiler/cohort change.

`make fp4-runtime-check` builds `ds41_fp4_runtime_device_check`, invoked with the **same full
libr4d path**. Run separately on every physical card, with actual canonical BDF receipt, in each
unpinned/exact/LCP profile that will be admitted. Both homes must be home-sync and all five
plugins/core/checkers must be one new coherent build. Ordinary lock-held QFN stop/restore/answer
and forced0/KFD-empty prerequisites remain mandatory.

The new checker requires2241 cases,4482 changing-payload graph calls,2016 actual unchanged packed
store comparisons,12 nonzero fast/fallback per-alias/width receipts, captured-fast equality and five
refusals/overflow cases. It covers14 heights1..4096, all BF16 bit images through exhaustive large
planes, ties/neighbors, finite/nonfinite/signzero/subnormals, strided positions, invalid positions,
pitched BF16/capture output, F32/BF16 gain, old/new workgroup and forced N128, normal dispatch,
Audit0/1, persistent replay totals and full byte/padding/guard comparisons. Unknown n256 direct
calls must match the original producer with zero audit work. Each completed comparison is raw
uint16/U8 equality; float equality is never authority. No timings are reported.

The unchanged codec/packed paths/reader gates must still run fresh. Full packed-store equality
does not by itself prove a reader path. Existing LCP/P1/balance/SiLU/CPUchain/Rb/Vp gates also stay
mandatory for each proposed model profile. Inspect the generated code separately for classifier
and Audit=false behavior before any performance interpretation; source intent is not codegen.

Only then may root prepare an instrumented same-cohort model gate: original prompt sets/caps,
full3397 direct-old-base KLD, complete reply bytes/counts/actual IDs/logical Kh,96GiB real misses,
concurrent N4/N8 and prefix/cancel/drain/replay. Every rank must show actual nonzero fast work for
every selected alias/width slot, no overflow, nonempty block totals and capture_fast=fast for each
selected capture slot; unselected slots may be zero, with graph/issue proof. Require both
query and latent/key path coverage, not one aggregate fast count. Audit0 quiet comparisons require
an actual instrumented functional qualification first and independent reverse-order repetition.
No source in this branch writes a passing receipt or authorizes default enable/adoption.

## Short source review and remaining risks

Reviewed: old HIP functions remain literal; new schemas alone gain an optional operand; disabled
architecture declarations/issues are retained; capture uses the same q and scale; partial N128
blocks cannot bypass an audit barrier; fixed PERSIST counters cannot alias lent transient memory;
teardown consumes ownership before fallible readback; unproved declarations keep original names.

Still unexecuted: compilation/host cases, actual alias/fused/capture NaN payload bytes, all-card
graph replay, audit drain/readback behavior, model path composition and generated-code cost. The
classifier can increase register pressure or lose to the existing loop after compilation. The
counter oracle does not establish speed. Current53 and77f behavior, defaults and evidence remain
unchanged until root validates a coherent candidate.

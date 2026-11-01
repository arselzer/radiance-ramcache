# Native FP8 conversion aliases — SOURCE ONLY

This prototype is **unbuilt and unexecuted**. The four-card `d89a410d` standalone component gate
passed all original assertions on Nov1, including 12,480 full-product exception pairs per card.
That receipt does not qualify these new aliases, their fused compiler context, graph capture,
observer, serving replies or performance. Both flags below remain OFF.

The base is `53fe48b0758765a91589e7d6d30ad0d4b7e86870`; this branch also retains the independent
ea3/52ef/d89 standalone source history. The runtime commit can be cherry-picked independently:
it does not include any tool header in serving code. No public ABI changes or toolchain updates.

## Closed source scope

`RADIANCE_EXPERIMENTAL_DS41_FP8_CONVERT=1` chooses NEW aliases only for:

| Alias | Native geometry | Ordinary no-draft trunk use |
|---|---|---|
| `quant_roundtrip_fp8_convert` | BF16, width512 or8192, group32, FP8 E4M3, UE8M0 | Normally absent: fused producers already resolve |
| `rope_rows_roundtrip_fp8_convert` | BF16 width8192, 16 heads×512, GPTJ rot64/off448/inverse1, group32/UE8M0 | Forty output-RoPE sites per rank |

Output RoPE is the smallest useful actual trunk extension. It keeps the original 256-thread
column grid, row walk, pair shuffle, BF16 RoPE round, XOR amax fold, scale expression, divide,
clamp, F32 product and BF16 store sequence. Only the software E4M3 encoder/decoder changes on the
proven finite domain. There are statically8192×40=327,680 conversion pairs/output-token/rank at
these output sites. This is a work count, **not elapsed cost or an expected speedup**.

The512-wide fused norm/RoPE window producer, original-code WINDOW capture, FP4 producers, cache
readers and all other sites keep their whole old operations. The wrapper does not force any old
fused producer to decline. DSpark's native plain512/8192 sites are eligible if separately enabled;
the initial useful serving gate should have DSpark OFF and prove actual trunk coverage.

Unknown static geometry delegates to the whole original declaration/resolver chain. Direct new
exports also delegate unsupported geometry to the whole old body before any candidate launch.
Unknown format/invalid shape remains refused under the old contract. New aliases are admitted
only on actual gfx1201/wave32 native TP4. A gfx1201 compilation without the conversion builtins
fails compilation instead of generating a software-only apparently hardware alias.

FLAG0 preserves old operation names, graph structure, operand order, original HIP bodies, exports,
schemas and resolver chain. New schema/row/shape/export entries are additive. Old helper signatures
are untouched; the wrappers are separate so the parallel FP4 helper's appended parameters merge
additively. Source text preservation is not a substitute for cross-build reply/byte proof.

## Exact conversion and NaN constraints

Hardware conversion requires finite `v` and positive NORMAL finite `scale`. Divide and the old
`ds_clamp(v/scale,-448,448)` stay unchanged. All NaN/Inf values and exceptional scales keep the
original software conversions. Encoding duplicates the same float into the installed RNE packed
FP8 instruction and selects its low byte. Decoding uses the corresponding OCP E4M3 instruction.
No q8/group128 reciprocal/fminmax helper is reused. No re-quantization, new scale or quantization.

The private product is the SAME gfx1201 `v_mul_f32_e32 scale,decoded` source order admitted for
d89. It is not a NaN mask/rule or numeric equality relaxation. The original standalone ea3 and
52ef failures remain preserved: source-equivalent C++ multiplied the operands in reversed ISA
order and changed NaN payloads. These new runtime kernels **still need independent actual ISA
admission** against their actual original quant and fused-RoPE kernels. The d89 receipt cannot
authorize a new compiler context. No NaN tolerance or reference rewrite is allowed.

## Startup and actual-use authority

The flags accept only unset/`0`/`1`; `_AUDIT=1` requires the main flag. Architecture parsing occurs
at declare, not per token. Native model geometry and availability of both NEW schemas are checked.
Before load/CTX, core inspects every actual selected band: exact operation/row name, exact exported
launch address and DSO origin, same plugin order, no tunables/choices/host fallback or unexpected
init/fini/layout/scratch/replaces hook. Every rank must contain the40 native output aliases.
Actual device props must identify gfx1201/wave32. A missing or unknown alias fails startup.

Per-rank enabled-only admission marker:

```
ds41 fp8 conversion admitted: rank=R ops=N trunk_rope=40 n=8192 group=32 original_arithmetic=1 capture=unchanged model_qualified=0
```

`RADIANCE_EXPERIMENTAL_DS41_FP8_CONVERT_AUDIT=1` adds planner-owned PERSIST I64[12] device+host
planes:96B payload each, with ordinary planner alignment/accounting. Three slots distinguish
plain512, plain8192 and fused output8192. Four U64 fields are fast, fallback, workgroups, overflow.
The fixed context handles/live state are charged by `sizeof(Ctx)`; startup logs their aligned16B
bound. There is no extra per-value atomic, token readback, TLS allocation or hidden cache payload.
Audit=false compiles out its LDS, integer reductions, atomics and counter reads entirely.

Audit=true folds per-thread integer counts AFTER the original stores, then performs checked
workgroup-level atomic adds. Overflow latches rejection and never wraps into accepted evidence.
The planes initialize once at startup and persist across captured replays. After main and lane1
are drained, shutdown reads the96B plane once and validates all slots before reporting:

```
ds41 fp8 conversion audit: rank=R op=rope_rows_roundtrip_fp8_convert n=8192 fast=F fallback=S blocks=B overflow=0 readbacks=1 scope=rank_op_width lifetime=cohort
ds41 fp8 conversion audit complete: rank=R rows=3 bytes=96 readbacks=1 drained=1
```

Zero plain rows are legitimate inactivity. They do not prove plain serving coverage. Every rank
must have positive output-RoPE fast/workgroup counts before a model gate can claim actual use.
Any `audit FAILED`, absent completion or overflow is fatal to qualification. Audit-on timing is
diagnostic; it includes the observer costs and cannot forecast audit-off performance.

## Prepared proof sources and root-only execution order

No command below has been executed by the source agent. Parent must serialize under build.lock;
all card work uses both research locks and the normal QFN stop/restore+new-answer window.

1. Coherently rebuild core/allfive plugins and the new checker using the exact serving O3,
   contract-off/no-fast-math flags. Retain full source/core/DSO/ABI/compiler/header/checker hashes.
   Run `ds41_fp8_runtime_test` (five host cases) and `ds41_fp8_runtime_arch_test` (four architecture
   cases) with `-LE gpu`; retain ordinary existing gates, tiny configs and variable192/96GiB paths.
   Architecture cases compare complete unset/0 graphs and issues on four ranks, direct old/new
   graph normalization, N1/N4/N8/M17, fixed audit ownership and missing-alias refusal. The known
   backend-supported grouped output-A pitched view has a fixture-local exact shape/offset/stride
   exemption on BOTH graphs; all other lint remains fatal.
   The NEW host-only `tools/ds41-host-experts/fp8-runtime/literal_source_check.py` takes explicit
   `--original` literal53 elem source (SHA d7e6a869...) and `--candidate` new elem source, plus a
   fresh `--output`. It compares20 complete original function signatures/bodies including both
   original kernel/dispatch/export paths, records both full source hashes, and never rebaselines
   the sealed d89 checker. The old d89 literal checker correctly refuses a changed whole-file
   SHA even when changes are additive; keep that old checker/evidence untouched.
2. Manually inspect the actual new DSO's gfx1201 ISA for original quant/RoPE, candidate audit0/1
   and sideband original/candidate instances. Bind binary SHA, extracted image SHA/offset, actual
   objdump command/log and scale/decoded register provenance. Require old/candidate product source
   order equal, no modifiers/MODE writes/FMA, same old RoPE instruction/arithmetic order, genuine
   hardware conversions and independent raw-product comparisons. If any compiler context does not
   match, HOLD; do not change the reference oracle to agree.
3. Fresh four-card original d89 pointwise component proof remains mandatory (all12,480 exception
   pairs, BF16×UE grid and eight actual native producer profiles). Then build target
   `fp8-runtime-check` and run `ds41_fp8_runtime_device_check LIBR4D.so DEVICE` independently on
   physical cards0..3 with actual BDF witnesses. Recommended fresh processes also cover existing
   unpinned, EXACT_TALL and LCP4096 environment pins; the candidate leaves their dispatch alone.
4. The new checker requires168 cases/336 changed-input graph replays, each comparing the actual
   old public export against the actual alias, and separate unregistered original/candidate
   sideband kernels. It covers widths512/8192, both operations, M1/2/8/17/128/2048/4096, four
   profiles including all65536 BF16 source patterns at adequate rows, NaN/Inf/signed-zero and
   normal data, pitched matrices, strided positions, negative/out-of-table positions, exceptional
   RoPE tables, audit OFF/ON, persistent counts, all padding/redzones and unchanged tables/positions.
   Strict code bytes, F32 scale/product words and BF16 bytes must match. Two malformed-audit/stride
   refusals and one unknown-width whole-old fallback are required. Every slot must demonstrate
   both fast and fallback work. No numeric NaN equality. Final exact marker:

```
PASS ds41_fp8_runtime_exact cases=168 replays=336 products=P codes=P scales=P bad=0 guards=0 refusals=2 whole_old=1 scope=component serving_proof=0 integrated=0 performance_eligible=0
```

The old public exports expose BF16 only; raw F32/code/scale sidebands come from UNREGISTERED
instrumented instances calling the unchanged original helper, not from reading hidden old state.
Their BF16 outputs must ALSO equal the old public export. The manual ISA link above is necessary
to relate raw-product reference semantics to that actual old compiler context. Do not describe a
sideband-only PASS as an actual original-export raw-F32 output proof.

5. Only after every prerequisite: independent ordinary/alias long3397 KLD direct to the approved
   old plain record, unchanged long/prefix sets and N1/N4/N8 replies/counts/actual IDs/Kh,96GiB
   spill/cancel/prefix lifecycle. Hold every unrelated new switch OFF and keep Rb/Vp/supported
   prefill profiles matched. Actual40 output sites on all ranks, no errors/unmeasured arms, clean
   ordinary server exits and normal same-container QFN restore+new answer are mandatory.
6. Performance requires a subsequent quiet matched/reversed pair with audit/Kh OFF, full process
   costs included. There is no performance forecast from static work counts or component success.

WINDOW/KV packed capture composition is intentionally unchanged numerically but not newly qualified
by this feature. No cache budget hardcoding, automatic enablement, new policy or adoption.

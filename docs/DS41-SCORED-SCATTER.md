# Exact native M1 sqrtsoftplus selection plus scatter

Source only off389bf676. This lane has run no build, syntax check, test, emulator, model,
benchmark, GPU or service command. Default off; no speed or byte-proof claim before root gates.

`RADIANCE_EXPERIMENTAL_DS41_SCORED_SCATTER=1` declares one optional native target M1 op per
trunk layer. Unset/empty/0 keeps the original declaration and issue order; only1 is accepted.
The Model parses the setting at declaration, never during issue. Initial scope is40×D5120,
64 attention heads/D512,FF2304,hc4, TP4/384 experts/96 local/top6/shared-ungated, no speculation
or TBO. Route trace and Phase2 audit are explicitly refused until both old and new selector
issue handles have a complete diagnostic mapping. No bank/draft graph is admitted. Cache budgets
remain variable; this allocates no new cache or host workspace.

Projection is unchanged. M1 issues the existing GlmRouter projection, then the dedicated
`ds41_scored_topk_scatter` op. Every other height issues the old scored selector and scatter.
The first implementation does not fuse a projection or shared gate, widen a GEMM, change a
norm/reduction partition, quantize any value, change routing or modify acceptance/RNG.

One256-thread workgroup owns the complete result. Its first wave copies the original
OWN12 sqrtsoftplus scorer's exp/log1p, threshold-at20, sqrt, correction-rank keys, top6 rounds,
chosen-score retrieval and selected-order F32 sum/normalization. The old kernel source stays
unchanged. Its NaN/−Inf exclusion, lower-ID ties, signed-zero rank canonicalization and token0
fallback when all keys are excluded are retained, including repeated token0 selections. This
is not the existing GLM sigmoid fusion. A block barrier then enters the same
`r4d_moesort::scatter_small_place<8,INV>` as the original standalone small scatter. All ranks
retain their own96-expert slice, stable original slot order, exact offsets/counts, untouched
unused sorted tail and optional inverse map. There is no grid-wide counter/barrier or spin.

Operand order is logits,bias,expert_ids,weights,sorted_tok,expert_offset,expert_count,optional
token2slot. F32 scorer rows/pitches and strided bias remain supported atM1; sorted/count/offset/
inverse are exact contiguous native vectors. Invalid shape/dtype/world/base/score, aliasing of
an output with a touched input/output range, overflow and protected-expert requests refuse
before dispatch. FileStager's count witness recognizes only the exact new op, retaining every
existing rank/row/stream/native shape check and normal issue-before-count-copy ordering.

Before load_weights/Ctx/warmup, core requires exactly40 unique correction bindings and native
M1 ops, exact operand roles, every actual selected device band named
`ds41_scored_topk_scatter_m1` from one loaded libr4d, and each launch pointer at the exact
`r4d_rad_ds41_scored_topk_scatter` symbol. Missing/miswired/tuned/host/unknown paths refuse.
The opt-in on a different architecture or plugin set refuses. Old operators remain independent
reference exports. No public ABI layout is changed.

Expected per-rank startup marker:

```
ds41 scored scatter admitted: rank=R ops=40 M=1 experts=384 local=96 topk=6 projection=unchanged
```

After a successful normal fused issue, once per rank/process:

```
[r4d] ds41 scored scatter active: rank=R M=1 path=sqrtsoftplus+stable_scatter inverse=0|1
```

Off has neither marker. `r4d_rad_ds41_scored_scatter_paths()` exposes the successful normal
wrapper's rank bitmap for the independent card checker; it is not an engine scheduler control.

## Prepared proofs and root integration queue

`ds41_scored_scatter_test` has native parse/scope and all-rank actual export/role/geometry
admission cases, including partial/duplicate/bad correction bindings, wrong symbol/library,
tuned/missing/host bands and sigmoid/incorrect dtype refusals. `ds41_scored_scatter_arch_test`
records the full native graph: empty/unset/0 fingerprint equality; all four ranks retain the
old complete M2/M17 issue hashes and projection; M1 replaces exactly40 old selectors/sorts;
buffer lifetime/extent lint remains active. Both use `RAD_TEST_MAIN()` and are unexecuted.

Root builds core and all five serving plugins coherently under the existing build lock/nice
rules, then exact host targets with `-LE gpu`; affected existing targets include build_test,
runtime_test, p1_stager_test, ds41_shape_probe_arch_test, ds41_dspark_defer_arch_test and the
host-only native mapped mover. Do not mix a newly built architecture with the old kernel home.

`make scored-scatter-check O=ABS_DIR` in tools/ds41-host-experts builds the standalone HIP
source with the existing no-fast-math/fp-contract-off flags. Root alone runs it inside the
normal two-lock stopped-QFN window, all four cards visible:

```
CHECK --inside-guarded-window HOME/kernels/libr4d.so
```

The guard flag is a required claim, not permission to bypass the actual stopped-QFN/locks/
fresh-cohort checks in the window controller. The tool loads original scorer and original
scatter/inverse plus the actual new exports from the same library, validates exact symbol starts,
and compares complete bytes including IDs,F32 weights,sorted slots,offsets/counts,inverse,
padding/unused tails and outer guards. Input buffers must remain untouched. Prepared coverage:
808 cases/card=3232, both normalization modes and inverse on/off, ties, all-qNaN/sNaN/±Inf,
signed zeros/subnormals, overflow,20±1ulp, every expert-partition edge and wrap, pitched rows/
strided bias, all65536 widened-BF16 logit and bias patterns;48 shape/alias/dtype/scope refusals
must leave every output untouched. Positive paths=15 proves the actual wrapper on every rank.

Required exact final markers (only printed after successful comparisons):

```
PASS ds41_scored_scatter rank=R cases=808 refusals=12 all_bf16_logits=65536 all_bf16_bias=65536 bad=0
PASS ds41_scored_scatter_exact ranks=4 cases=3232 refusals=48 paths=15 bytes=... bad=0
```

No model-enable harness part or fresh-receipt preparer is added here. Root must first add a
strict independent per-arm0/1 knob/reset and same-window SHA-bound card receipt, actual four-rank
startup/dispatch checks and KLD-reference refusal, preserving all existing feature guards.
Then plainM16/new independent longKLD3397 must match the approved old plain record; unchanged
long/harness complete replies/counts/prompt IDs/logical hashes and ordinary single-stream heldout
decode must match controls. M1 serving must actually issue the new path; a tall-prefill-only
KLD run can prove fallback but cannot substitute for that dispatch proof. Include N4/N8 old-height
fallback, cancellation/prefix reuse and96GiB functional spill before adoption/compositions.

Removing one dispatch/layer is40/token at eligibleM1; actual latency saving is unmeasured.
LDS/barrier/register pressure and the extra declaration's liveness can erase the saving. Keep
the original graph as reference and use paired uninstrumented timing after every byte gate.

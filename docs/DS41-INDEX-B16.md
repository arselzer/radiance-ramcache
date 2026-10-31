# Lossless16-bit index score plane

Source only; default off. No build, test, emulator or GPU execution by this lane.
`RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_BF16=1` stores the high16bits of the
already-BF16-rounded f32 score, then reconstructs the exact original f32 bits by
integer shift before candidate/top-k readers. There is no additional rounding,
quantization, dot change, new accumulation partition or selection change.

The current scalar and WMMA writers both finish with `ds_round_bf16(acc)`;
unreachable/unmapped columns store−Inf. All65536 possible high16-bit patterns,
including signed zero, subnormals, infinities and NaN payloads, expand exactly.
The new `index_score_b16`, `index_candidates_b16`, `index_topk_b16` aliases keep
old f32 kernels and shims intact as an independent reference. New copies differ
only in storage accesses/dtype checks and new-alias diagnostic counters. The
original merge kernel still consumes the unchanged64-bit candidate records.

The original irows quotient/floor, run count, row_part indexing, runtime R/tail
extents, causal reach, grid/HS and WMMA/fallback choices are preserved. Compact
index floor is an independent flag: B16 never spends the saved bytes on more
rows. At32K,2047×32769 scores shrink from255.883 to127.941MiB/card. This is a
source buffer-size saving, not a measured process allocation or speed gain;
arena aliasing/other peaks and changed expert residency must be measured.

## Admission and composition

The architecture changes only the plane dtype and the three explicit op aliases.
Flag0/unset keeps the old graph, kernels and logging. Malformed values refuse.
Before loading weights, Ctx init or warmup, core proves every selected band has
the exact named B16 op/kernel identity, no tuning, no host fallback, and one
actual loaded libr4d DSO/plugin order. A missing/old/unknown writer or reader
refuses. The plane must be a unique positive BF16 rank2 `index_score` buffer.
Existing LCP/compact-index admission remains in force.

Future KV_PACKED composition explicitly refuses until it has a separately
proven B16-output packed-cache writer alias. The existing packed f32 writer
must never receive the shortened score plane. No public ABI layout changes.

Each rank prints one startup line:
`ds41 index b16 admitted: rank=R rows=... columns=... bytes=... legacy_bytes=...
same_runs=1 writers=... candidates=... topk=...`.
After successful normal launch, new aliases print one marker per rank/role:
`ds41 index b16 active: rank=R role=score|candidates|topk`.
All three roles on all four ranks are required model evidence; enabled startup
alone is insufficient. Off arms must contain neither marker family.

## Source checks prepared for root

Host targets: `ds41_index_b16_test`, `ds41_shape_probe_arch_test`, existing
`ds41_index_memory_test`, `ref_test`, `engine_test`, `runtime_test`, `build_test`.
New metadata cases cover all bands/DSO identity/tuning/old+packed refusal, every
score bit pattern, exact half-size with identical rows/runs, sizing graphs and
malformed flags. Run only under build.lock with `-LE gpu` and exact ctest names.

Build standalone card source with `make index-b16-check O=...` under the same
build lock. Invoke ONLY inside a stopped/restored production window, with all
four cards visible, in three fresh processes (set unrelated flags explicitly0):

```
RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=0 RADIANCE_EXPERIMENTAL_DS41_LCP=0 CHECK --inside-guarded-window LIB unpinned 4
RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1 RADIANCE_EXPERIMENTAL_DS41_LCP=0 CHECK --inside-guarded-window LIB exact 4
RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=0 RADIANCE_EXPERIMENTAL_DS41_LCP=4096 CHECK --inside-guarded-window LIB lcp4096 4
```

The checker requires original and new exports from the SAME actual LIB. On each
card/profile,12 scorer cases compare expanded B16 to every original f32 output
bit, plus12 selection cases compare keep bits and IDs with/without keep. It
checks input/outer/pitched guards, row_part tail/unmapped/future columns,
all65536 score bit patterns and a native128-wide dataset containing every BF16
input bit pattern. Cases include HS1/2/4/8, unpinned WMMA1/2, H33 and pitched-q
fallback, M49/2048/4096, width8192 and split thresholds24576/24577, M16/17,
allNaN/all−Inf and tie-heavy rows. Actual new-alias path bits must include
1023 unpinned or975 pinned, with no WMMA bit under pins. Literal final success:
`PASS index_b16 coverage: profile=P ranks=4 score_cases=48 select_cases=48
all_score_bits=65536 all_input_bits=65536 max_M=4096 max_C=65536 ... bad=0`.
No success marker is printed following a failure.

Require a fresh checker/library/engine/arch/libref SHA receipt and actual perrank
markers before an enabled arm. Then paired longKLD/Kh+prefix/N4/N8 at192GiB,
actual32K prompts and one96GiB spill regression, with all replies/counts/hash
rows exact. Compare off/on rows/run receipts; use the same configured context,
compact flag and existing numeric pins in both arms. No performance claim
until end-to-end measurements pass.

## Harness source

New arm part `Ib16`; every other arm appends the flag0 explicitly. Set
`DEVICE_CHECKS` to include `index_b16`, `INDEX_B16_COHORT=<manifest>` and
`INDEX_B16_LOCK_HOLDER_PID=<existing research lock holder>`. The window generates
a fresh process/time nonce after stopping QFN, runs all three checker profiles,
then admits only arms using the checked engine/architecture/libref/library.
`INDEX_B16_GATE` may name the snapshotted helper; `INDEX_B16_RECEIPT` defaults to
the window output directory. Copy `ds41_index_b16_gate.py` beside the snapshotted
ab_window.sh (or use its explicit absolute helper path). Missing helpers/receipts
refuse enabled arms and do not affect ordinary arms.

Before the window, prepare explicit frozen authority (no GPU/library load):

```
python3 -I ds41_index_b16_gate.py prepare --source-commit FULL40SHA --engine BIN --architecture HOME/architectures/deepseek41.so --libref HOME/kernels/libref.so --library HOME/kernels/libr4d.so --checker CHECK --out NEW-cohort.json
```

The helper hashes those five artifacts before/after every card profile; no
authority is inferred from printed logs. It checks the actual stopped production
container, both occupied research locks, live holder PID and four-card mask
before launching the checker. Every profile's exact complete coverage/log hash
is retained in the fresh-window receipt. Replaced libraries, stale windows,
missing/duplicate ranks/roles, incorrect half-size ledgers and log tampering
refuse. The same gate covers serving and KLD completion; Ib16 cannot become its
own KLD reference. Existing feature gates remain in force.

Prepared fake host targets `ds41_index_b16_gate_test` and
`ab_window_index_b16_test` cover these refusals, outside-window/QFN-running/free
lock/card-mask rejection before checker invocation, per-arm0/1 reset despite
global enable and KLD-reference refusal. They have not been executed.

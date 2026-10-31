# Default-off direct-PV adjacent-pair reconstruction

Source prepared on `claude/ds41-kv-pair`, rooted at
`5fb651c173b98da4bc805951defc8c6475d72bb3`. No builds, syntax checks, tests, GPU work or
performance measurements were performed during preparation. The motivation/source audit is
`work/autonomous-20261031/kv-reader-cost/PACKED-READER-COST.md` in the parent work area.
The existing packed model window is FUNCTIONAL_ONLY and is not an adoption receipt for this.

`RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_PAIR=1` requires explicit `KV_PACKED=1`,
`KV_PACKED_FAST=1`, and `WINDOW_PACKED=0` (full environment prefix for each is
`RADIANCE_EXPERIMENTAL_DS41_`). Architecture admission is native TP4, 16 attention heads/rank,
D512. Unset/0 is off; malformed strings and missing dependencies fail startup.

Only a normal M1/H16 packed-fast attention operation selects a separate compile-time
`HG1/NT1024/DPT2/PVD=true` kernel. This is the form the old selector already chooses there.
If that form refuses LDS, the old dispatch/refusal sequence remains. Other heights and forms,
index, QK, producer/store, window and WMMA loads keep their existing implementations. No op
schema, argument layout, page representation, cache allocation, retained shadow or budget is added.
The optional new kernel has the same key/lane mapping, P4 pipeline, float accumulators, key order,
multiply/add expressions, folds and output rounding. Its only replacement is the direct-PV
two-value reconstruction inside the existing load helper.

For even columns, `rad_ds41_kv_pair::decode_row2` loads one group scale and one payload byte;
normal E4 coefficient/power or Pow2 source/half-FTZ preparation is shared before invoking the
unchanged integer value reconstruction twice. Scale0 reads the two literal BF16 patterns and
one selector byte once. It returns the same low/high uint16 word the old helper assembled.
Odd/group-crossing generic requests retain the original scalar decode semantics. Byte loads
make no new row alignment assumption. Invalid bounds/metadata reject before writing through
the checked helper; the device's trapping wrapper fails closed. Null/zero-key guards precede
physical row access exactly as in the old packed load, including speculative padded PV reloads.

The new normal branch logs, at most once per rank, only after its dedicated kernel enqueue
returns RAD_OK:

```
[r4d] ds41 KV_PACKED_PAIR active rank=0 normal=1 M=1 H=16 HG=1 NT=1024 DPT=2 PVD=1 row_bytes=288 decoder=integer-pair
```

This proves the normal dedicated kernel was selected/enqueued, not that its completion or byte
gate passed. Completion and actual selected compressed data are checked by the oracle/model
receipts. The checker-only forced export does not print this marker; a model gate must find it
in that model server's own log for each active rank, not copy an oracle's receipt.

PAIR1-only fixed, cache-line-separated host counters also record successful normal enqueues,
classifying the actual new kernel, old M>1 fallback and other old fallback. Clean plugin teardown
prints one `KV_PACKED_PAIR totals` row per touched rank with `pair_calls`,
`multirow_fallback_calls`, `other_fallback_calls`, `max_M`,
`scope=all_normal_dispatch_phases` and `host_successful_enqueues=1`. These can include prefill
and warmup; they are not decode-only GPU completion or performance counters. The first successful
M>1 old dispatch logs `KV_PACKED_PAIR fallback ... reason=multirow dispatch=original`.
PAIR0 does not construct or update this counter metadata. N4/N8 gates additionally require actual
`rows_unspec>steps_decode>0`, zero mixed steps and observed parallel owners; neither the requested
N nor a prefill fallback counter alone proves concurrent decode. M16 KLD is fallback/mixed-tail
coverage and may legitimately include M1 tail work.

There is no new registered alias/ABI capability. As with FAST, a stale old plugin can ignore an
environment variable. Therefore the new unregistered export
`r4d_rad_ds41_attn_sparse_sink_kvpair_check`, coherent core/all-five-plugin/source hashes and
fresh same-library oracle receipts are mandatory admission to a model measurement. The window
must refuse missing exports, missing normal M1 markers or a mismatched plugin cohort. Flags
alone are insufficient. The export's form0 forces the old direct-PV kernel; form1 forces the new
kernel and requires M1/H16/packed-fast. Every other form is refused.

## Prepared checks (all unexecuted)

Root alone performs the coherent build and serialized checks under AGENT-LOAD-RULES. New host
target `ds41_kv_pair_test` contains five cases: all 256 scale bytes/all 256 nibble pairs for both
formats; every literal uint16 pattern selected as both patterns, with odd selector cases;
unaligned mixed/zero/half-FTZ rows at every starting column; invalid geometry/metadata/no-partial
writes; strict flag/dependency/native-shape eligibility. The existing architecture test target
`ds41_work_rows_arch_test` adds two cases proving off/on graph and issue identity plus startup
refusals (missing FAST/PACKED, malformed values, WINDOW_PACKED and non-TP4). Shape-probe fixture
environment isolation includes the new flag. Existing old targets/output are retained.

In `tools/ds41-host-experts`, the new build-only Make target is `kv-pair-check` (set `O` to a
parent-owned build directory). It produces `ds41_kv_pair_device_check` and
`ds41_kv_pair_paths_check`. Build follows the normal serialized parent workflow; no Make target
executes these binaries. Once built, the following are **GPU commands, only inside the ordinary
stopped/restored-QFN window with both research locks and the persistent physical build-lock owner**:

```
ds41_kv_pair_device_check
RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=1 RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_PAIR=1 ds41_kv_pair_paths_check /absolute/coherent/libr4d.so
```

The guarded all-four-card runner uses `HIP_VISIBLE_DEVICES=<one selected backend ordinal>` and
calls `ds41_kv_pair_device_check --one-card-proof` in a fresh process for each selection. That mode
requires one API-visible device, validates inverse PCI lookup and prints its actual
`DS41_PAIR_CARD_ID ... bdf=...`. Four unique actual BDFs must equal the explicitly frozen TP4 map;
the same isolated environment is retained for that card's existing/new oracle subprocesses.

Expected byte-only codec line has exact counts plus positive actual successful even/odd calls,
positive checked refusals and zero bad values:

```
PASS ds41 pair codec vs reference: normal=131072 literal=131072 rows=256 metadata=1 even=... odd=... rejected=... bad=0
```

The path checker has 12 cases (ratios1/2, windows3/16/128, topk40/512) with M1/H16, real selected
compressed pages, literal NaN/zero groups, a selected zero-initialized page, absent/remapped
pages, out-of-table selections, non-multiple-32 tails and output guards. It compares every output
and guard byte from ordinary BF16, the old forced packed kernel, the new forced packed kernel,
and the new normal dispatch. Its 36 invalid-shape/format/form calls must leave outputs intact.

```
PASS ds41 pair directPV reader: cases=12 normal=12 forced=12 old=12 refused=36 bad=0 M=1 H=16 HG=1 NT=1024 DPT=2 PVD=1
```

Keep the independent existing codec/producer/store/fused/all-form reader oracle gates, including
the explicit 1/254 half-FTZ contract, literal NaN payloads and all old scalar/decode/WMMA forms.
Run independent profiles in fresh processes where cached options require it. Do not substitute
the new small checker for the old full producer/dispatch coverage.

Before performance work, inspect actual generated old/new served kernel ISA, VGPRs, scratch
spills, loads and wait scheduling. The compiler may already CSE the scalar pair, or the new
expression may be worse. No source-level savings, occupancy or speed claim is made.

Only then compare ordinary, existing packed-fast+PAIR0 and packed-fast+PAIR1 on identical frozen
original prompts/caps with all other knobs, home-sync, placement and source-cohort receipts
explicit. Require byte-identical KLD/logits/replies/counts/Kh, actual normal M1 marker per rank,
real required urgent work, complete request/TTFT/decode timing and normal QFN restore. Before
adoption retain 96 GiB, concurrency and prefix/cancel/reuse gates. This feature does not assume a
full host cache, change small-cache spill paths, or establish large filled-context correctness.

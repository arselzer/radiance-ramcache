# Original Engram: focused implementation follow-up, 4 November 2026

**Research only; lower priority than current performance windows.** No code/model changes, builds, tests,
imports, bulk reads, weight downloads or GPU work occurred in this follow-up. The only write is this note.
Source inspected: decode-impl `32d033e1119609843efbb09541c9b9959258766f`. The existing IQ4 hybrid stays the
default and byte-reference model. Original Engram is an additional model cohort requiring a distinct baseline;
it must not be presented as preserving IQ4 replies or as independently qualifying the entire official runner.

## What has closed since the first plan

- [Nonexpert audit attempt2](/home/as/ds41-claude-build/work/native-precision-audit/run-root-attempt2/summary.json)
  is **PASS: 1,119/1,119 entries, 16,834,990,528 destination bytes, zero mismatches, inexact values or refused
  specials**, authenticated original sources/unchanged epochs. Summary SHA256
  `fbdd78cff376cb40b68194322d6b714b352767cef0897372c02998467d192220`. This closes the declared import-value
  question for those weights, including the two F32 gate products. It excludes Engram tables and whole-runner semantics.
- [Fresh control replay](/home/as/ds41-claude-build/work/native-control-audit/fresh-root-attempt1/summary.json)
  is **PASS: all seven current integer planes, 129,384 signed words/517,952 bytes, zero differences**, including a
  genuine fresh original-source/real-tokenizer/PCG run with regular input copies. Summary SHA256
  `98cd88ead8de0ae9cc0afe2d27e9f84b88444c4609649e81557146ccb1c5464e`. Reuse these constants and receipts;
  do not repeat the normalizer/RNG investigation or call them unverified merely because older notes do.
- The exact lookup body is already local and pinned: official revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`,
  [model.py:296](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/reference/inference/model.py:296),
  whole-file SHA256 `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.
  Lines317–320 perform **FP8 lookup + E8M0 lookup → F32 expansion/multiply → BF16 cast**; line321 masks
  unowned rows to positive zero, then the official TP path sums partial rows. No new source fetch is required.
  Earlier [CPU official fixtures](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/DS41_OFFICIAL_ORACLE_ACQUISITION_PLAN.txt:97)
  explicitly stubbed embed/wkv, so they do **not** qualify this lookup.

The missing work is original table payloads plus the native host gather/source policy. The prior
[restoration plan](NATIVE-ENGRAM-RESTORE-PLAN-20261104.md), asset hashes and old IQ4 AVX2 component history
remain applicable. IQ4 component speedups did not test original per-group FP8/E8M0.

## Smallest implementation slice

| Seam | Concrete change and refusal boundary |
|---|---|
| [ref gather](/home/as/radiance-ds41-claude/libref/ref_gemm.cpp:587), [AVX gather](/home/as/radiance-ds41-claude/libavx/avx_gemm.cpp:1235) | Add one explicit native branch before the current integer-code refusal: `RAD_F8E4M3` codes, `RAD_E8M0` scales, group32, width256, no codebook, BF16 output, multiplier exactly1. Other native/mixed forms refuse before output writes. Existing IQ4/integer branches keep their behavior and resolver priority. |
| Arithmetic | Decode the original code and its own group exponent, reproduce the official **F32 multiply then BF16 cast**, preserving signed zero and supported exceptional semantics. E8M0 byte0 is2^-127, not zero. Do not multiply an already rounded BF16 scale, add the integer path's second multiply, or assume host FTZ/DAZ is harmless. An integer-bit implementation or128KiB code×exponent→BF16 LUT is possible **after** exhaustive proof; the nonexpert audit's rational decoder alone is not that original-lookup execution proof. |
| [schemas/cases](/home/as/radiance-ds41-claude/libref/ref_shapes.cpp:2732), ref/AVX registries, [hand fixtures](/home/as/radiance-ds41-claude/libavx/avx_check_hand.cpp:1319) | Retain default I8 fixture generation. Add explicitly typed native fixtures so generic checks cannot silently exercise I8 instead. Verify actual native branch selection and both scale/code operands. |
| [native source reader](/home/as/radiance-ds41-claude/core/format/checkpoint.cpp:2946), [writer](/home/as/radiance-ds41-claude/tools/rad_convert.cpp:1556) | Already recognize F8/E8M0 `[rows,256]`/`[rows,8]`, group32 and direct raw-plane copy. Add fixtures and exact role validation; no new quantizer or BF16 table expansion is needed. The current no-rule path passes an entire mapped source plane to `write_plane`; use bounded4MiB raw copy/hash calls for a paced restore rather than assuming that call already enforces a4MiB window. Preserve native expert codes/scales and all audited nonexpert choices. |
| [architecture policy](/home/as/radiance-ds41-claude/arch/deepseek41/deepseek41.cpp:225), [converter policy](/home/as/radiance-ds41-claude/tools/rad_convert.cpp:659) | Introduce a separate explicit native policy. Do not relabel native planes as ISTA or broaden the six-role GGUF whitelist. The native policy must bind exact revision, two code/scale pairs, official controls, dimensions and source ancestry. The old ISTA policy remains independently selectable. |

**Prefer native per-layer tables for the first cohort.** Current `local_engram` is coupled to ISTA; separate
the layout choice from table precision/source. Reuse the existing two per-layer handles/names and the now-proved
**local** offsets, mapping their neural planes to `layers.{1,14}.engram.embed.{weight,scale}`. Generate the small
control artifact from unchanged official bodies, serialized as the native policy's explicitly validated layer arrays;
no donor neural supplement is needed. This avoids changing the hash layout while restoring the table values.

The existing release path instead joins both tables at
[architecture:2893](/home/as/radiance-ds41-claude/arch/deepseek41/deepseek41.cpp:2893), requiring second-layer
offsets rebased by384,006,168. It remains a viable alternative with a seam/ID proof. **Do not feed the PR266
global-offset artifact into the local-table path, or reuse local offsets unchanged with the joined table.**
`--reuse` matches encoding/options/shape, not source values
([converter:1344](/home/as/radiance-ds41-claude/tools/rad_convert.cpp:1344)); a global-offset conversion must
explicitly invalidate changed control sources, or it could copy same-shaped old local controls.

## File-backed access and variable RAM

Use `--ngram-placement disk` for the first native qualification. The existing mapped-tier loader publishes
each original plane without a whole-table load
([bringup:3372](/home/as/radiance-ds41-claude/core/engine_bringup.cpp:3372)); `auto` already has a host/file-backed
fallback when a whole RAM copy cannot fit. No188.8GiB pin, VRAM copy, new minimum RAM or R9700 dependency is needed.
Expert-cache size stays configurable, including128GB hosts and changing pressure.192GiB of expert cache plus
188.8GiB of Engram cannot both be pinned on the current256GB machine.

Keep `embed_lookup_planes`, so existing
[prefetch planning](/home/as/radiance-ds41-claude/core/runtime/issue.cpp:1356) discovers **both** the256-byte code
row and8-byte scale row. Existing [row prefetch](/home/as/radiance-ds41-claude/core/runtime/rowprefetch.h:1) merges
page runs, distinguishes cold/warm gathers and joins its helper before IDs may change. Prove this with native
views; do not duplicate I/O or keep detached ID pointers across steps. The host gather writes only requested BF16
rows, retaining the existing host/device handoff and ascending-rank target fold. Official TP byte parity is a
separate gate; preserving the current engine's qualified fold does not establish official-runner equivalence.

Use≤4MiB bands for fixture/extraction/verification work and a128MiB explicit working-buffer cap, excluding the
already pinned backend runtime. Stream raw code/scale bytes and hashes; never allocate a full table. For model
prefill, keep the existing bounded chunk size and gather output; decode scratch is one256-element row plus
bounded ID/output batches. Do not expand the complete tables to BF16 (about366.2GiB).

Possible later exact I/O improvements, each separate from the first implementation:

- Native scale planes total **6,144,182,800 bytes (~5.72GiB)**. An optional, separately budgeted scale-only
  RAM tier could remove scale-page faults while leaving code rows file-backed. Current `--ngram-placement ram`
  selects whole tables, so scale-only placement is **new work**, not an existing flag. Require a byte cap,
  reclaimable/mapped fallback and normal ownership checks; do not take6GiB unconditionally from expert cache.
- If measured faults matter, adapt the existing direct-row cache to raw **paired**264-byte rows. Bind both
  plane owners/offsets/format/model generation; publish only when both reads succeed; preserve in-flight readers
  during eviction/shrink/error/teardown. Budget keys, slots, pinned I/O and in-flight buffers as well as payload.
  [The existing cache](/home/as/radiance-ds41-claude/libavx/avx_ngram.cpp:283) has a fixed2^19 rows and
  [process-lifetime ownership](/home/as/radiance-ds41-claude/libavx/avx_ngram.cpp:380); do not copy those limits
  into a new supposedly dynamic cache. A mapped fallback must work at zero explicit cache budget.

## Disk/conversion choices

Original47/48 remain absent as full local objects; retained selected tails contain q/k/wkv only. The two
original objects need203,073,077,576 bytes; their embedding planes need202,758,032,400 bytes. Exact filenames,
public revision URLs and whole-object SHA256s are in the prior plan. No fetch is part of this task.

| Option | Storage/I/O consequences and readiness |
|---|---|
| New compact native `.rad` using existing converter | About508,383,016,656 bytes (473.47GiB), excluding small alignment changes. Retaining full47/48 beside old+new models needs ~711.46GB additional disk before scratch/readback. `--reuse` saves conversion work but currently **copies** unchanged planes in bounded bands; it is not a reflink/zero-I/O promise. This is the simplest supported writer path. |
| Future clone-and-append restore helper | Existing [RadWriter::begin_append](/home/as/radiance-ds41-claude/core/format/radfile.cpp:719) and [add_existing](/home/as/radiance-ds41-claude/core/format/radfile.cpp:767) can carry unchanged entries at their old offsets. A NEW helper could take a mandatory reflink clone, omit the two IQ4 entries from its new directory, append the two native pairs, and write new native metadata/header last. Logical size ~618.98GB; physical new payload ~202.76GB plus metadata **if** actual reflink sharing succeeds. With retained originals, additional physical storage ~405.83GB. This is source-backed feasibility, **not implemented/qualified**. Existing `rad-convert --in-place` explicitly refuses changed encodings and cannot do this replacement. Never append to the baseline itself; require distinct inode/nlink1, exclusive clone ownership, unchanged carried entry/group/offset bytes, abort/header recovery and final full-SHA/readback. If reflink is unavailable, price a real copy or choose the compact conversion; do not silently fall back. |
| Raw-original sidecar/overlay | Could avoid duplicating table planes but needs typed multi-source/metadata/lifetime support. The existing source-bank path is DSpark-specific, not a generic Engram overlay. Do not repurpose its CLI or bypass its coverage/epoch guards. Defer behind the native gather. |

Both full-stream authentication and conversion/full readback are heavy I/O, scheduled by ROOT between
performance windows. Retain only embedding ranges during a complete authenticated stream if storage requires it;
this saves the small tails, not the202.8GB tables. Partial HTTP ranges alone do not inherit full-object SHA proof.

## Minimal next experiment and gates

1. **No-weight HOST vertical slice first.** Extract the unchanged official `ParallelEngramEmbedding` lookup
   body into a CPU-only fixture without importing model.py/TileLang. A256-row×256-column synthetic table can
   cover every code×exponent combination: all256 raw FP8 codes per row, each row one E8M0 byte. Compare all65,536
   BF16 results/raw bits to the new ref and AVX branches, including signed zeros/subnormals/overflow/NaNs and
   supported FTZ/DAZ modes. Record unsupported CPU embedding/cast paths as refusals, not a substitute oracle.
   Use strided outputs, padding, repeated/negative/out-of-range IDs, TP ownership, page boundaries, truncated
   reads and helper-drain/cancellation fixtures; keep existing IQ4 fixtures unchanged. No full model is needed.
2. **Tiny converter/placement fixture.** Two deliberately different native layers with local offsets, exact
   code/scale byte hashes and mismatched native/IQ4 negative cases. Confirm both planes remain mapped,
   group32/schema/native branch actually selected, and prefetch addresses stay in each owned extent. Test a
   joined-layout alternative only if that layout is selected. This decides policy/serializer before huge assets.
3. **Only after these pass**, schedule authorized47/48 acquisition, storage choice and a new immutable
   native cohort. Readback must preserve every original code/exponent and unchanged expert/nonexpert plane;
   reuse the existing seven-control proof, checking artifact/view identity. A signed-zero/NaN ambiguity or
   any changed non-Engram weight blocks qualification. No tolerance-based gate.
4. **Separate protected model window,5–11 arms.** Include old-IQ4 N1/N8 regressions; native reference/AVX
   N1/N8 and long-prefill pairs with native full state/replies/counts; a smaller-RAM/miss case and accepted
   cancel→recovery. Split lifecycle into a second window if needed. Use the same coherent engine/plugin cohort
   and explicit model-bound references. Different native versus IQ4 replies are a new-baseline difference,
   never a PASS against the old model. Present the concrete native result for baseline approval before any default change.

Then measure native performance with quiet matched ABBA. Native rows are264 versus144 IQ4 bytes, but both
formats generally touch separate4KiB code/scale pages per cold random row; **1.833× logical bytes is not1.833×
latency or disk amplification**. Larger code pages hurt cache residency, smaller scale rows help scale residency,
and four ranks share the filesystem page cache while repeating decode work. Watch row faults/read amplification,
host gather/joins, NVMe contention, cache pressure and startup/full-conversion I/O before investing in SIMD or
an explicit cache. No native Engram model TPS result or speedup forecast is established.

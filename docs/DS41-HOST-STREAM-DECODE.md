# Native one-row decode-to-MAC prototype

Source prepared off389bf676. Default off, unbuilt, untested and unmeasured. This is the narrow
first experiment from `/home/as/ds41-decode-review-20261028/CPU-KERNEL-NEXT.md`, not an FMA
experiment or an adopted configuration. Root owns all serialized builds, assembly checks,
host/card gates and model windows. Query-free work is separate.

`RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_STREAM_DECODE=1` is read once when the existing CPU
expert pool starts. Unset/empty/0 selects the original path; an invalid nonempty value refuses
the pool as other strict pool options do. It applies only to a **single token row** of a native
planar or packed matrix, for full32-block rounds passing the original `fast_round` predicate
(all scales113..127). Every general round, partial tail, ggml-block call and multirow call uses
the original code. No format, ABI, cache budget, expert placement, job/slot ownership, SwiGLU,
target pipeline, data movement or GPU arithmetic changes.

## Exact operations

The new `kv_pair16_consume` uses the existing paired byte transpose, nibble LUT, saturating
scale adjustment and exact FP16 bits. It widens those bits with F16C and immediately performs
the old separate `mul_ps` then `add_ps` on the existing halved/transposed activations. For each
of the four independent8-lane accumulators the order remains blockL,L+32,… and value0..31.
In particular native transpose callbackj is consumed as4j,4j+1,4j+2,4j+3. The old store sequence
is4j,4j+2,4j+1,4j+3; blindly consuming in that order would reassociate the sum. Finishing pair0/1
before pair2/3 changes no dependency between independent accumulators. The original32-lane
fold and BF16 conversion remain unchanged. Compilation still requires `-ffp-contract=off`
and `-fno-fast-math`.

Packed scale selection uses the original `PkRounds` cursor, widening and raw-exception handling.
This prototype does not implement direct packed-scale register plumbing, change eligibility,
assume raw rounds are rare, or add FMA/new partial sums. Exceptional values outside the fast
domain keep the exact old general path. Exceptional activations inside it retain the old half,
FP16 decode, multiply/add and NaN behavior; correctness means bytes equal the old path, including
that behavior, rather than a claim about a different algorithm on arbitrary NaNs/subnormals.

No new TLS buffer is retained: the old planar and packed single-row wrappers select the new
consumer after building activations in their existing TLS storage. Its maximum at servedGU
K5120 remains5×1024floats=20KiB per existing native wrapper. The lean/Sc2 route uses the existing
worker layout and key. No job-wide activation copy or new cache reservation is introduced.

Successful actual use logs once per pool, thread safely:

```
DS41HX_STREAM first_use rows=1 native_fast_round=1 arithmetic=mul_add fallback=unchanged
```

The startup option line alone is insufficient. The marker is emitted only after at least one
full fast round actually executed the direct consumer. With the flag off there is no extra log
or atomic update. With it on, later qualifying tiles perform only a relaxed first-use check.
The shared process pool covers all four EP ranks; this is a per-pool use marker, not four rank
counters. The callable helpers return whether any actual direct full round ran, independently
of output data or requested configuration.

## Source-prepared gates

Targets: `radiance`, `ds41_hx_stream_test`, `host_expert_pool_test`. CMake auto-discovers the new
test source. Its six cases byte-compare the old entry against both the direct-input and cached
lanes entry points for planar/packed layouts. They cover:

- K32/224/256/992/1024/1056/1280/2016/2048/2304/5120; nonsquareN24, whole and interior row ranges,
  empty ranges, all-tail calls, full rounds and partial-group tails; output/input/slot redzones.
- All256 scale bytes, boundary113/127 versus112/128, raw first/middle/last rounds and tails,
  packed cursors starting at a nonzero row, mixed fast/general rounds, all16 code nibbles.
- All65536 BF16 activation patterns, signed zero, all subnormal classes, infinities, signaling
  and quiet NaN payloads, finite cancellation inputs; exact uint16 equality (no tolerance).
- Each of the four FTZ/DAZ settings, with the original control state restored after the case;
  original ggml fallback; full4608×5120 gate/up→unchanged SwiGLU→5120×2304 down bytes, stage by
  stage. The host chain table is intentionally the existing host table for old/new identity;
  a real model/card gate still needs the captured card table.

Focused ctest aliases `host_expert_pool_test_stream` and `host_expert_pool_test_stream_lean`
run the existing packed-job and many-row cases at flag1, lean0/1. Packed single-row output must
equal the original scalar reference; block/multirow behavior stays checked, and the test
requires `stream_decode_used()==true`. Existing base/lean/wake-off aliases explicitly set0.
The six-case oracle and both focused pool aliases must exit0 with no skip/failure; the pool
logs must include the actual-use marker. No prepared test has been run by this lane.

Root's serialized host command after coherent build is the existing load-rule form, e.g.:

```
flock /home/as/ds41-claude-build/work/build.lock nice -n19 \
  ctest --test-dir BUILD -LE gpu -R '^(ds41_hx_stream_test|host_expert_pool_test_stream|host_expert_pool_test_stream_lean)$' --output-on-failure
```

Then the root gates, in order:

1. Compare actual optimized code generation of old fast round versus `gemv_stream_xt` on the
   target compiler. Establish whether oldW16 scratch already vanished, count stack/spills and
   reloads, confirm no FMA or extra partial sums and no register-pressure regression. Direct
   consumption lengthens decode/accumulator live ranges; source alone proves no speed gain.
   Include flag0 old-wrapper code, since their optional internal stream dispatch shares storage.
2. A separately bound micro driver must call `gemv_avx2_stream` / `_lanes` directly and compare
   full output bytes before timing. Test hot and DRAM-sized distinct experts, GU/DN and a full
   chain, packed/planar and lean/no-lean, then40-thread pool-shaped work. Avoid an L2-only claim
   when current192GiB serving is bandwidth limited; retain smaller-cache/spill functional gates.
   Existing `ds41_host_experts_test rbench` still calls the OLD entry and cannot prove this path.
3. Inside the normal stopped-QFN, both-lock GPU window, run an actual stream host→GPU chain
   oracle with card-captured SwiGLU table and fresh source/library hashes. Existing
   `ds41_mxfp4_device_check ... check` still calls the OLD host entry; its PASS alone cannot be
   used as the stream receipt. A new driver/mode or actual CPU-vs-card per-miss verification
   must bind the new entry and require the first-use marker. No GPU checker mode was changed.
4. Explicit flag0/1 model arms, all other knobs identical and current prompt sets unchanged,
   full texts/counts/Kh/KLD as appropriate versus same-cohort base and approved base. Require
   the actual-use marker in1 and its absence in0, TP4, current256GB/192GiB plus96GiB functional
   spill, and N1 plus representative concurrent traffic because priorities remain total
   concurrent decode first. Include all dispatch/marking costs in timing. No new global arm
   parser token or recommended configuration was added by this lane.

If assembly shows equivalent scratch elimination already, or register spills/time worsen,
leave the flag off and record the result instead of widening arithmetic or retrying old
unconditional FMA/retile proposals under this name.

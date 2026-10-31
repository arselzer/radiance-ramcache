# Compact index-score row floor

Status: source only, default off. No build, test, device or serving result is established by
this patch. This changes scratch sizing inside the existing row-run implementation; it does
not stream KV, change cache pages, reduce index arithmetic, or establish 500K–1M serving.

`RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_COMPACT=1` permits a 64-row floor for the existing
256 MiB score-plane target, only beside a valid LCP chunk. The quotient still includes the
existing guard column, and run balancing, `row_part`, causal reach, candidate masks and top-k
remain the existing operators. Unset/empty/0 preserves the 256-row floor. A requested switch
without valid LCP, including EXACT_TALL alone, keeps that legacy floor. Other switch values
are declaration errors.

For C2048 and a ratio-1 source, source-derived plane sizes are:

| Maximum context | Legacy rows/bytes | Compact rows/bytes |
|---|---:|---:|
| 32,768 |2047 /268312572|2047 /268312572|
| 524,288 |256 /536871936|127 /266338812|
| 1,048,576 |256 /1073742848|64 /268435712|

The 1Mi plane is 256 MiB + 256 B; the 512Ki plane is 254 MiB + 508 B. The 64-row minimum remains a floor,
so still larger contexts can exceed the target. Nothing is fixed to a RAM or expert-cache
budget. Activation lifetime packing determines the eventual process allocation; these are
plane extents, not an additive measured VRAM saving.

A smaller balanced run can cross 48 rows, the sparse launcher's WMMA threshold. LCP pins
`index_score_body` to scalar at every height. Its HS1/2/4/8 variants keep each head's 128 FMAs,
BF16 dot/weight/product rounding and ascending-head fp32 fold; only independent work is split.
The packed `index_score_kvpack` alias uses that same body and only reconstructs original BF16
cache bits before it. Candidate/top-k operators select each row independently with the same
score/tie/position ordering, including top-k's exact small-row split/merge.

The environment is not sufficient authority. After ordinary LCP admission proves the routed
twin reaches the chunk, compact activation additionally requires every affected device band
to resolve to the known index-score, candidate and top-k rows, with no axes or tuned point,
the same `libr4d` registry identity/order, and the same actual loaded DSO base as the admitted
LCP/P1 twin (`dladdr` of launch functions). Unknown/overridden/tuned/missing-origin activation
is a startup refusal, not a silent compact fallback. With the switch off this extra guard
does not run. KernelRow stores a plugin name/order, not a Plugin pointer; actual module bases
provide the additional origin check. No ABI or kernel is changed.

Prepared host sources cover metadata declarations at 32Ki/512Ki/1Mi without allocating those
planes, sizing probes, actual recorded run/input/output extents and tails; pure origin/axes
admission positives and refusals; and reference-op byte composition of scores, causal masks,
candidate bits and top-k across two sequences, missing pages, ties, pitches and 64/127/256-row
floors. These do not replace card scalar/packed oracle and model byte/logit/hash gates.

RoPE remains separate: two full persistent F32 tables at N×64 columns cost 512 B per maximum
position, 512 MiB/card at 1Mi context and 256 MiB at 512Ki. Paging/rebasing them is deferred. Likewise,
row tiling bounds scratch while the three early producer index layers still score all
reachable history, retaining quadratic long-prefill work. Prior archive sources already
describe row runs, bit masks, exact top-k and LCP pins; this patch does not reintroduce them as
new optimizations. Older 128GB/P0 timings are not forecasts for the current 256GB deployment.

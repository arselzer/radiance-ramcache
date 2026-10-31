# Exact score+selection fusion feasibility

Source audit only. No fused kernel implemented or measured. Current B16 plane
is the smaller first implementation; this design must retain every original
per-score FMA/BF16 rounding, head fold and stable selection rule.

Archive reconciliation: all31 earlier reports/full NIGHT-LOG were covered by
the prior audit, then index-related history/current source was rechecked here.
Row runs, bitpacked masks, split/merge top-k and scalar-tall pinning are already
implemented. No prior full score-plane elimination trial was found. Old P0
timings/other engines are not current256GB/32K performance evidence.

At32K the early index layers2/8/14 have ratio2 and16385 columns. They remain
below existing split_topk's≥4×8192-column condition at≤16 rows. Early-only fusion
would retain the shared plane for layer20/later layers and still need two
launches; it recovers no score arena. Scoring also reads256 index-key bytes and
performs4096 FMAs per valid score, so scratch traffic savings are not a model
speed forecast. The three early scans remain quadratic in long prefill.

## Candidate full mapping

One optional pinned-scalar prefill op, two internal launches, keeps current row
runs. Launch1 assigns one256-thread workgroup to a row×8192-column tile, uses
unchanged scalar per-score expressions, keeps32-bit score keys in32KiB LDS,
then emits its exact top512 `(key<<32 | global_position)` in ascending position
order plus count. Launch2 reuses existing ordered global merge. Every global
winner is among its own tile's first512 under score-desc/index-asc ordering;
concatenating tiles in position order preserves global tie handling.

Keep old decode initially: large tiles collapse its scoring parallelism from
hundreds of HS workgroups to2 nonempty tiles at a16K row. A1024-thread fused
selector also risks spilling the scorer's128-float per-thread key state. A
256-thread integer selector can zero-pad8 wave totals into the current32-wave
scan, and scan2048 histogram bins in four ordered512-bin slabs. Histogram
counts/prefixes are integer and bounded; output compaction visits ascending
positions just as before. This mapping still needs independent byte oracles
and actual occupancy/time measurements.

Layer20 needs TWO independent outputs: global top512 over all positions, and
block maxima→top2048 blocks of8 with newest block pinned. Its own top512 is
NOT restricted by the mask it creates. Emit block-max keys with the same
NaN-ignore/−Inf exclusion; override the newest block to+Inf even if its page
is unmapped. Eight-position blocks align within8192-column tiles. Launch2 can
perform the unchanged block-key selection as well as candidate merge. Later
layers24/28/32/36 consume that existing mask and may skip excluded scores,
since no other reader consumes them. This does not accelerate the early dense
index scans or candidate-layer scoring asymptotically.

Suggested op operands retain q/w/cache/table/positions/cu plus sel, optional
input keep/output keep, and real output counts[R,nchunks]. Counts' live shape
provides chunk width without a fake metadata-only out-of-bounds view. Internal
scratch holds uint64 candidates[R,nchunks,512] and optional block maxima. Round
column coverage to a tile only beyond causal reach; unused tiles/counts must
be freshly zeroed. Use resolved R for row_part*R, not the shorter last live
output's rows. Keep captured-pass keys and changing live widths/tails explicit.

NaNs produce no top-k key, signed-zero keys use current canonicalization,
−Inf is absent and+Inf is valid. Base R4D outputs are the oracle for exceptional
selection (do not assume host partial_sort on NaNs defines the same order).
Preserve missing-page handling, negative positions, consumer row cuts,
multi-sequence cu offsets, keep padding and ascending selected IDs/−1 fills.

At32K the existing2047×32769 f32 plane is255.883MiB/card. Full fusion's worst
run needs about40.02MiB records/counts plus31.99MiB block maxima. Retaining a
64-row old decode plane adds8MiB: about80MiB before arena aliasing, or roughly
176MiB recovered. Early-only fusion saves none of that arena. B16 storage alone
gets127.94MiB with unchanged scorer parallelism and is therefore prioritized.
These are source workspace bounds, not measured process/residency or speed.

Gate every score key/keep bit/selected ID against the original paths across
all HS/tile/run/tie/NaN/page/capture boundaries before any model arm. Initial
fusion scope must pin scalar arithmetic and refuse unknown/packed writers
until separately proved. Root owns all serialized build/card/model windows.

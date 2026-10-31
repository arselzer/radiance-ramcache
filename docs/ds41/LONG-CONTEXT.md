# DS41 long context and exact packed cache: source audit, Oct31

Status at15:25 UTC: source sizing/design, not500K–1M serving validation. The research server runs
only in guarded windows. Its context limit varies per arm:1024 for concurrent decode,8192 for
long prompts,32768 for harness turns. There is no32K model/engine hard limit established here.

The frozen local1017 checkpoint config declares max_position_embeddings=1,048,576 and YaRN16×
over65,536. It lives under
`/home/as/radiance-setup/research/deepseek-v41-1017-20261024/metadata/deepseek-ai--DeepSeek-V4.1-Flash-1017/config.json`.
The official [model config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)
and [model report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md)
also describe native1M context. That is model support, not a local throughput result.

## Current storage and source-derived room

Full-history cache sources are layers2/8/14 with ratio2, and layer20 with ratio1. Later index
query layers24/28/32/36 reuse layer20's cache. Each current compressed row is512 BF16 values
(1024B); each index row128 BF16 values(256B). Shared cached rows per input token are
3/2+1=2.5, so full-history payload is exactly3200B/token/card. TP4 currently replicates it;
do not divide that figure by four.

The current dataflow already applies FP4 rounding before storing those BF16 values:

- Main compressed rows: all512 columns, including RoPE, E2M1 with E4M3 scales per16.
- Index rows: all128 columns, E2M1 with power-of-two scales per32.
- Window rows: FP8 E4M3 with power-of-two scales per32, then BF16. FP4 windows would add loss.
- Compressor tails: rawF32. Engram token state and window snapshots also remain exact.

For one active sequence at C2048, window allocation is138 pages×640KiB=86.25MiB/card, independent
of maximum context. This includes a live-chunk allowance; the old6.88MiB window figure was for
M16, not P1. At C4096 it is266 pages=166.25MiB, an80MiB increment. Aligned raw state is about49KiB.

| Allocation, per card | 32Ki context | 512Ki | 1Mi |
|---|---:|---:|---:|
| Full BF16 history payload |100MiB|1600MiB|3200MiB|
| Rounded KV pool, C2048 |187MiB|1687MiB|3287MiB|
| Two persistent RoPE tables |16MiB|256MiB|512MiB|
| Existing index-score plane |255.883MiB|512.001MiB|1024.001MiB|
| Candidate bitmask |1.008MiB|16.008MiB|32.008MiB|
| Device block tables |0.502MiB|8.002MiB|16.002MiB|

RoPE uses2×positions×64×4B=512B/position. Tables, score buffers and masks have distinct lifetimes;
the activation planner can overlay some buffers, so this table is not an additive process
allocation measurement. Source estimates suggest one BF161M sequence could fit16GiB cards by
displacing roughly4.3GiB/card of expert residents. Actual startup fit, latency and numeric gates
have not been run. Eight simultaneous1M BF16 full histories alone exceed25GiB/card and cannot
fit those cards. More host RAM does not directly enlarge VRAM.

## The890B/token native target

Native main rows use256B FP4 codes+32B scales=288B; index rows64B codes+4B scales=68B. Thus
2.5×(288+68)=890B/input token per replica:890MiB at1,048,576 tokens versus3200MiB expanded BF16.
This saves2310MiB/card of full-history payload if realized. Page alignment, bounded windows,
tails, metadata and transient work remain additional costs. The lightweight official Python
reference fake-quantizes and retains expanded floating tensors; it does not establish physical
packed storage. Optimized cache formats are defined in
[DeepInfer cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/cache_common.py)
and [FlashMLA](https://github.com/deepseek-ai/FlashMLA).

The old archived FP4 test accepts +0==-0 and packers drop zero signs. It is not byte proof for
this task. Current Radiance retains signed zero; NaN/Inf, scale255 and minimum-scale FTZ need
explicit treatment. Re-quantizing BF16 has no established idempotence guarantee.

New source commits9f9c7ebe(codec),56c5fa3b(page storage),daf7d0a1(every actual KV operand guard)
and89cec0a9(producers/readers) capture ORIGINAL codes/scales. Normal decoding reproduces BF16
bits before existing arithmetic. Scale0 uses two captured BF16 patterns+selector bits inside
the existing payload; original index Inf outputs need at most two patterns. Stored254 represents
the source-scale1 half-product FTZ variant. No permanent BF16 shadow or worst-case RAW pool is
allocated. A generic unrepresentable group returns Raw; a packed active cache must prove zero
Raw for the closed producer domain before model testing. All these new sources await integration
and host/card/model gates. Faster integer decoding is a separate optional source lane.

## Existing mechanisms versus new work

Already implemented: index row runs targeting256MiB, bitpacked candidate masks, row/column
top-k splitting, later candidate cuts and idle host prefix tiers (Pcr can store replicated
groups once). Do not propose these again as missing features. Active host-authoritative KV
streaming is different: native DS41 has no streamed store/resolve path; --kv-resident is ignored
for it and LCP refuses streamed groups.

The score plane still has a256-row minimum. At1Mi context, its width is max_ctx/rmin+1=1,048,577
and the minimum forces1GiB+1024B. A new source lane will keep the existing budget quotient but
allow a64-row floor only under proven LCP scalar dispatch:64 rows cost256MiB+256B at1Mi;
the quotient selects127 rows near254MiB at512Ki. It will verify actual known rows, no tuned
selection and the same loaded module as the admitted LCP twin. Unpinned/unknown cases retain the
legacy path or refuse explicit unsupported activation. This is not implemented/adopted yet.

Row tiling bounds scratch, not arithmetic. The scalar indexer scores all reachable history;
candidate masking occurs later. The three early producer index layers therefore retain
quadratic long-prefill work.512 selected attention positions do not eliminate that scan.
Smaller score planes do not establish practical1M prefill latency. Lazy/rebased exact RoPE
tables and column tiling with exact global tie/NaN/top-k merging remain separate possible work.

## Growing physical backing

Today KVManager::carve_into backs all carved bytes at startup. Block assignment grows inside
that backing. Existing KV loans lend already-backed FULL stripes to experts and recall them;
they do not map/unmap, and general lending is disabled under deterministic placement.

Pool/VMem already reserve stable VA and support granule commit/decommit. New source policy
2167bd20 and transaction77ff9ce9 plan/pin every held granule, coordinate rank publication,
rollback failed mappings and poison unsafe recovery.29319f85 separately fixes a failed-unmap
ledger path. The policy host test passed; transaction/backend cases are source-only so far.
No engine caller currently grows KV. Safe live wiring must park/drain CPU and all GPU/copy
work BEFORE scheduler allocation/adoption, publish no unbacked ID, and retain prefix/COW/window/
raw-state holders. A rounded physical granule must be charged in full. Expert-slab expansion is
additional work; physically freeing KV alone does not increase expert residency.

Next gates: coherent full ABI rebuild; strict codec/producer/fused/scalar/fast/WMMA bytes; page,
prefix/COW/snapshot/rollback/cancel; short-model logits/replies/hashes;96GiB spill; increasing
context across window/chunk/resume and65,536-position boundaries. Current model testing has
exercised about9.7K harness history, not a full32K,500K or1M request.

Archive coverage remains all31Markdown reports, full NIGHT-LOG and engine README. Raw historical
captures and every commit were not exhaustively re-read. Earlier128GB/2133 measurements and
watchdog-contaminated P0 timings are not forecasts for the current256GB/2666 system.

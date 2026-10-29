# DeepSeek-V4.1 host experts

DeepSeek-V4.1 Flash on four RX 9070 XT cards keeps most of its routed experts out of VRAM: in a
RAM cache, or on the NVMe drive. This directory holds the CPU kernels and tools behind the engine's
host-expert path: the CPU computes every routed expert the cards do not hold, bit-exact with the
card's MXFP4 GEMV, so every reply stays byte-identical to the base engine's.

On the loop's eight held-out prompts (64 tokens, greedy, TP4) the path takes decode from 5.03 to
about 17 tok/s and the time to first token summed over the prompts from 64.0 to about 15.5 s. Every
step was measured as an A/B window; the log is `/home/as/ds41-decode-review-20261028/NIGHT-LOG.md`.

## What runs where

- `core/place/host_expert_pool.*`: one process-wide pool of CPU workers, shared by the four rank
  threads, running the kernels in `ds41_host_experts.*`.
- `core/place/stager.*` (`FileStager::hx_*`): at each routed layer's routing sync, the experts the
  RAM cache holds go to the pool; missed ones are read from the container's planes and computed
  too, then relaid into the cache in the background. The card gets aliases for them, and their rows
  are copied over the card's before the gather.
- `core/place/plane_mirror.*`: an optional second copy of the planes on another drive.
- `core/place/router_predictor.*`: the next-layer predictor (shadow statistics; read-ahead is off).
- `core/place/readahead.*`: read-ahead in pieces (`..._PREFETCH=pieces`, off): the predicted misses
  of layer L+2 read at layer L in small pieces that give way to demand reads.

## Configuration

Everything is opt-in and off by default. The measured best configuration:

```
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=40                 # pool threads (SMT siblings help)
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=<silu table captured on this card>
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=all         # every RAM-cached expert on the CPU
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=16          # misses on the CPU in passes up to 16 tokens
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES=2     # prefill misses read while card ops issue
RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS=1
RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1
RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL=1
RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW=1               # pin the cache's slabs at startup
RADIANCE_EXPERIMENTAL_DS41_MXFP4_PACK_SSE2=1
RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=1             # VRAM file buffers to residents
RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM=1                    # fill the cache from the profile
RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1                  # engram_kv's rows split over the ranks
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1                 # next layer's attention weights into the on-die cache
RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1                  # the RAM cache in a CPU-only packed form (x1.044 slots)
RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=2           # profile-hot cached units get two second chances
RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1                     # FP8-sourced dense weights read from E4M3 codes
RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces                 # read-ahead of predicted misses in 512 KB pieces
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=4

radiance ... --tp 4 --placement expert_tiered --weights-disk-tier \
  --expert-ram-cache-total-mib 98304 --host-pool-mib 256 --embedding-placement ram \
  --max-num-batched-tokens 16 --expert-profile <decode profile> \
  --expert-plane-mirror <dir of the checkpoint's model-*.safetensors> \
  --expert-plane-mirror-decode-share 0.13                              # the mirror is optional, see below
```

With the planner's routed-only staging reserve (`bba1fda6`) the pinned pool needs only 256 MiB a rank,
which frees ~6.4 GB. Given to the cache instead (`--expert-ram-cache-total-mib 104448`, the RAM the
earlier 96 GiB + 1900 MiB configuration used): decode +4.3% and TTFT −6% in two runs of three, misses −9%, but with ~8 GB free the third lost 15% when the box ran short of memory (other jobs, the n-gram table's page cache).

The plane mirror is optional and off unless `--expert-plane-mirror` names a directory; nothing in
the engine or the scripts assumes where the checkpoint is. `ab_window.sh` takes it as
`MIRROR=<dir>` for its `R` arms and refuses those arms without it.

| Setting | Effect measured | Notes |
|---|---|---|
| `..._HOST_EXPERTS=N`, `_POLICY=all` | cached experts on the CPU: 5.63 → 5.96 tok/s | `greedy` splits with the copy link (no better); `verify` computes both and compares rows |
| `..._MISSES=16` | misses on the CPU: 5.96 → 8.89 tok/s | needs the mapped RAM cache |
| `--embedding-placement ram` | +1.5%, TTFT −2.4% | the engine's default (the A/B script had forced `vram`): the token embedding pinned in host RAM (1.26 GB), its 316 MiB a rank of VRAM to ~17 more resident experts |
| `--host-pool-mib 256`, cache 96 GiB | +9% (vs the 4 GiB pool) | the pinned pool's RAM is worth more as cache; 256 MiB since the planner fix (was 1900); 96 GiB leaves ~14 GiB free (102 GiB: +4.3% in two runs of three, ~8 GiB free, sensitive to other jobs) |
| `..._NATIVE_CACHE_PREGROW=1` | request 2 TTFT 13.7 → 5.0 s | slabs pinned at startup (+~30 s startup) |
| `--expert-plane-mirror DIR` | prefill TTFT −7 to −11% | `--expert-plane-mirror-share` (0.18) in passes of `--expert-plane-mirror-min-rows` (2) tokens or more; `--expert-plane-mirror-decode-share` (0) in decode, where 0.10 measured +3.6% (three runs), 0.14 −1% against 0.10 and 0.18 −4%; with the kept readers 0.13 +0.6% against 0.10 (two runs); turns itself off if a read fails or a shard changes; `RADIANCE_EXPERIMENTAL_DS41_PLANE_MIRROR_FAULT=N` tests that |
| `..._FILE_BUFFER_UNITS=1` | +9.5% (176 more VRAM experts a rank) | refused unless the CPU takes every routed unit of every pass (all, MISSES=16, passes ≤ 16 tokens) |
| `..._CACHE_WARM=1` | first request 7.1 → 11.2 tok/s, TTFT 7.9 → 4.6 s | +~14 s startup; pins the slabs first |
| `..._ASYNC_MISSES=2` | TTFT −25% | decode passes keep the direct path (no gain there) |
| `..._HOST_EXPERTS_INSERT_THREADS` (4) | | background relayout+insert threads |
| `..._PREFETCH=shadow\|on`, `_PREFETCH_K` | read-ahead: top 6 −14%, top 2 ±0 | off; shadow prints precision by prediction rank |
| `..._PREFETCH=pieces`, `_PREFETCH_K` (4) | +3.6% on top of FP8 (top 4 and 6 alike; NVMe busy 44 → 51%) | layer L's router input predicts layer L+2 (posted after the CPU work is submitted; the prediction is handed straight to the read-ahead). Each rank's predicted uncached units are read in `_PREFETCH_PIECE_KB` (512) pieces by `_PREFETCH_QD` (4) readers over the process, none started while any rank's demand read is in flight; a unit goes to its layer only whole, a partly read one is finished at once while the layer waits, and the rest are cancelled and never inserted. At most `_PREFETCH_LIVE` (8) units a rank; `_PREFETCH_FINISH=0` reads a partly read unit afresh instead of finishing it. Arm parts `PpK`, `PqN`, `PbN`, `Pf0`; the shutdown report's "read-ahead in pieces", "read-ahead readers" and "router predictor" lines |
| `..._ACCESS_TRACE=<dir>` | | per-rank access traces for `cache_sim.py` |
| miss reads on kept readers, 2 MiB pages | +4.6%, TTFT −3.6% | on by default; `RADIANCE_EXPERIMENTAL_DS41_READ_POOL=0` starts threads a call and `..._PLANE_HUGEPAGES=0` uses 4 KiB pages, as before (A/B switches) |
| missed units relaid straight into a reserved cache slot | +1.1% | on by default with `..._NATIVE_CACHE_DIRECT_FILL=1`; `RADIANCE_EXPERIMENTAL_DS41_INSERT_DIRECT=0` relays via scratch and copies, as before |
| `..._ENGRAM_SPLIT=1` | +3.5%, TTFT −2.6% | in the deepseek41 plugin (build it from this branch; the dev tree's has no switch): each rank holds and computes 6,400 of engram_kv's 25,600 rows and an all_gather rebuilds the row; the replicated 315 MB GEMV (1.4 ms a token on every card) becomes a quarter, and the ~450 MiB a card it frees holds 25 more experts a rank |
| `..._PREFETCH_ATTN=1` | +2.8% | in the deepseek41 plugin: while the card waits for a layer's CPU rows, lane 1 reads the next layer's q_a, kv, q_b and wo_a (~56 MB a card) into the 64 MB on-die cache (libr4d `prefetch`, read-only); decode passes only, and not ahead of an Engram layer. That stays its behaviour unless one of the three modifiers below is set |
| `..._PREFETCH_MIB=N`, `..._PREFETCH_SET=k1,k2,..`, `..._PREFETCH_ENGRAM=1` | not yet measured | modifiers of `..._PREFETCH_ATTN=1` (deepseek41 plugin, `arch/deepseek41/ds41_prefetch_plan.h`). `MIB`: a budget of N MiB an issue point, filled in use order (`ekv` engram_kv, `comp` the compressor, `qa`, `kv`, `iq` the indexer's q_b, `qb`, `oa` wo_a's groups, `ob` wo_b, `rt` the f32 router, `sgu`/`sdn` the shared expert's gate\|up and down), the read that crosses it cut to what is left: at bf16 56 MiB is the four reads above plus 2.5 MiB of wo_b; with `..._DENSE_FP8=1`, whose E4M3 codes halve the reads (the four alone are then 26.75 MiB), the whole list through the shared expert (rank 0 55.5 MiB). `SET`: the reads named, in that order (also `iw`, `ik`: the indexer's small projections; `ma`, `mf`: the connections' mixes), whole unless a budget is set. `ENGRAM`: an Engram layer (1, 14) is read too, its engram_kv by the layer before when `ekv` is listed, the rest right behind its Engram GEMM. Each rank prints its plan at startup and its counts at shutdown (`rank R: on-die reads`). Arm parts `PmN`, `Pl<k1+k2+..>`, `Pe` |
| `..._CACHE_SECOND_CHANCE=N` | N=1 +3.9%, N=2 +5.1% (TTFT −1.5% / +1.2%) | a cached unit the decode profile calls hot (`..._CACHE_PROTECT_SHARE`, default 6 of the sealed profile's 4326 selections a layer: ~1150 units a rank) moves to the front up to N times between uses before an insert may evict it. The held-out prompts share task families with the profile; replayed on the 16 chat prompts too, critical decode misses fell 7.9% (N=1) |
| `..._PACKED_CACHE=1` | +3.2% alone, TTFT −5.5%; with `..._CACHE_SECOND_CHANCE=2` 15.52 / 15.48 tok/s (+8.2%) | the RAM cache holds each expert's codes plane plus its e8m0 scales packed to 2 bits from one base a weight (raw rounds beside, budget 512 / 320, `..._PACKED_CACHE_RAW`): x1.0444 slots and 4.4% fewer bytes a CPU hit; 7 of 15,360 experts never fit and stay misses. Refused unless the CPU takes every routed unit (all, M16, B1); the card's copy path never reads or fills it |

## The packed RAM cache

With the file buffers capped (`..._FILE_BUFFER_UNITS`) the card never reads the RAM cache: the CPU
computes every expert in it. So the cache need not hold block_mxfp4 (17 bytes per 32 weights). With
`RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1` each weight keeps the release's codes plane and stores its
e8m0 scales as 2-bit codes from one base per weight (gate_up, `[w1; w3]`, is one weight): every block
of a 32-block round (the kernel's unit; down's 8-block row tail is a round of its own) in base .. base
+ 3, or the round is raw and its 32 bytes are kept beside (`ds41_host_experts.h`, the packed form).
The kernels widen a round's codes into the 32 bytes the scales plane holds there and run the planar
round code unchanged, so the bits are the planar kernel's, which are the card's.

A full scan of the release's scales (all 40 x 384 experts, 17 GB of e8m0 planes) found 120 (58.7%) and
121 (41.2%) nearly everywhere: 0.0046% of gate_up rounds and 0.0107% of down rounds are raw, and
92% of gate_ups and 86% of downs have none. The tail is in layers 37 to 39 (a down with 3331 raw
rounds of 15360). A slot is fixed per class; the default holds 512 raw rounds a gate_up and 320 a
down: 11,999,296 + 6,001,984 bytes for a pair of 12,533,760 + 6,266,880, so **x1.0444 the pairs**
in the same RAM (~1424 instead of 1364 a rank at 24 GiB). Seven experts do not fit (L37 E123, L38
E250, L39 E007/E087/E188/E313/E382: 0.042% of the decode profile's selections); they are never
cached and stay misses, computed from their planes. 256/256 would give x1.0451 with 15 left out
(0.087%), 600/3400 x1.0378 with none. `..._PACKED_CACHE_RAW=<gate_up>,<down>` sets other budgets.
The scan, budget and profile scripts are in `/home/as/ds41-claude-build/work/packed/run2/`.

Every path that fills the cache writes the packed form: the misses' background inserts and the
warm-up (both `Mover::insert_unit_planes`, which counts a unit's raw rounds first, then packs each
weight straight into a reserved slot: its codes plane copied, its scales packed). Before a slot is
published its scales are widened back as the kernels widen them and compared with the planes byte
for byte; a weight that fails is never published (counted). The cache carries a layout tag: only
the CPU path's `borrow_packed`/`reserve_packed` reach a packed slot, and every block_mxfp4 call (the
card's copy path: `borrow`, `borrow_readonly`, `reserve_fill`, `insert`) is turned away as a miss or
bypass and counted, so the card's path reads the container instead and never stores block bytes in
a packed slot. The shutdown report prints the packed cache's outcomes.

Kernel, 40 threads in the engine pool's shape (64-row tiles, 32 experts streamed from DRAM, the
research sample's real scales; `ds41_host_experts_test pkbench 32 40 24`, medians of seven):

| layout | 40 threads, ms an expert | 24 threads |
|---|---|---|
| block_mxfp4 | 0.416 | 0.387 |
| planes | 0.387 (−6.9%) | 0.354 (−8.6%) |
| packed | 0.381 (−8.5%) | 0.324 (−16.3%) |
| `..._DENSE_FP8=1` | +12.7% (15.05 → 16.96 tok/s), TTFT −10%, VRAM experts +105–114 a rank, misses −12% | in the deepseek41 plugin and libr4d of the fp8 branch: the trunk's dense linears the release ships as E4M3 codes with an E8M0 exponent a 32x32 block (q_a, q_b, kv, wo_a's groups, wo_b, the indexer's q_b, engram_kv and the shared expert; 330 weights a rank, 570 container entries) are read by `gemm_nt_bf16f8`, which stores those codes and exponents (1.001 bytes a value) and writes `gemm_nt`'s bytes. VRAM per rank: rank 0 4520 → 2262 MiB of these weights, ranks 1-3 4220 → 2112 MiB (−2.21 / −2.06 GiB; 225 MiB less with `..._ENGRAM_SPLIT=1`, 450 / 300 MiB less without the shared expert), which go to resident experts. The load re-encodes each weight and refuses one that is not exactly so (`ds41_dense_fp8_scan` checks ahead; all 570 are). Requires: no tune rows or `RADIANCE_TUNE` pin for the bf16 GEMMs (the op always takes the split law), the libr4d row (`gemm_bf16f8_nt_m64`), and the checkpoint's `imported_quantization_config` (fp8, ue8m0, 32x32); otherwise a note and everything stays bf16. `..._DENSE_FP8_EXCLUDE=name,*suffix` keeps weights bf16 (`*_shexp.weight`: the shared expert). Startup logs `deepseek41: rank R: N dense weights read by gemm_nt_bf16f8 ...` per rank |

`ds41_host_experts_test packed` holds packed to planar bit for bit: the 27 sampled experts, a raw
round at every block position (down's tail included), widened rounds on the general path (126 ..
129, 110 .. 113, the subnormal scales), NaN rows, and the packing API's refusals.

## Tools

- `ab_window.sh OUT`: waits for 30 s of QFN idleness, stops the QFN container, runs each arm of
  `$ARMS` as its own server with the eight held-out prompts, compares the replies with the first
  arm's, and always restarts QFN and checks it answers. Hold both research locks first. Run it from
  a snapshot (`/home/as/ds41-claude-build/work/launch_window.sh`), never edit it while it runs, and
  never launch it from a shell that hides the cards (`ROCR_VISIBLE_DEVICES`). An arm is
  `POLICY[-part...]`; the parts are listed in the script (threads `tN`, cache `mMIB`, misses `MR`,
  pool `pMIB`, pregrow `G`, mirror `R`/`RNN`, file buffer cap `BN`, warm `W`, async `A`/`AN`, fault
  `ZN`, predictor `PsK`/`PoK`, FP8-stored dense weights `F8` / `F8n` (shared expert bf16), ...);
  an arm with a part the script does not know is skipped. An `F8` arm is refused under
  `RADIANCE_TUNE` or with a tune cache in its home, and is not measured unless its server log
  names the weights read by `gemm_nt_bf16f8`; every other arm runs with the knob off.
  `PROMPTS`, `AB_REQUESTS`, `AB_MAX_TOKENS`, `AB_CONCURRENCY` and `MAX_SEQS` change the workload.
- `ab_requests.py`: sends a receipt's requests (token ids, `chat_text` framed like the held-out
  prompts, or `messages`) and compares arms.
- `cache_sim.py`: replays an access trace against LRU, SLRU, 2Q, aged LFU and Belady.
- `ds41_mxfp4_device_check.hip`: the kernels against the card (chain check, exhaustive SwiGLU,
  GEMV timing against two libr4d builds).
- `ds41_bf16f8_device_check.hip` (`make device-check O=<dir>`): `gemm_nt_bf16f8` on the card
  against `gemm_nt`, from a built libr4d.so: the weight fragments the kernel expands for all 254
  E4M3 codes at all 237 exponents (`frag`, which sees the sign of a zero), one-hot GEMMs over
  every (code, exponent) pair, signed zeros and cancelling pairs through the slices that start
  mid-block (`zeros`), ds41's per-rank shapes at every split and row count, container tensors and a
  timing table. Exit 0 when every output bit is equal. Run it in a GPU window under both locks:
  `HIP_VISIBLE_DEVICES=0 ds41_bf16f8_device_check <libr4d.so> all <model.rad>`.
- `ds41_dense_fp8_scan.cpp` (`make scan O=<dir>`, host only): the exactness scan behind
  `RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8`. It runs libr4d's own bf16f8 layout hooks over every
  dense weight deepseek41 would store as E4M3 codes (the 570 container entries of q_a, q_b, kv,
  wo_a's groups, wo_b, the shared expert, the indexer's q_b and engram_kv) in 1024-row bands, checks
  the inverse gives every band back, cross-checks the conversion's `sources.jsonl` (`--census`) and
  the tensor-parallel cuts at 2..4 ranks, and prints the `..._DENSE_FP8_EXCLUDE` list of any entry
  the load would refuse. Read-only, idle I/O priority, and it pauses while `ab_window.sh` runs. On
  `ds41-native-mxfp4-v3` (2026-10-29): 570 / 570 entries exact, 6,676,480 blocks, 0 refused
  (970,434 subnormal codes, 35,738 +0, 36,130 -0, block exponents 109..121), 610 E4M3 pieces in
  the census, 57 cuts all on 32; 13.7 GB in 17 s with 4 threads.
- `ds41_host_experts_test.cpp`: host tests of the kernels against the scalar reference.

## Correctness

Each row the CPU computes uses the card's arithmetic: the same lane and block order, butterfly,
BF16 rounding and no fused multiply-add. A job of several token rows decodes each round of codes
once for all of them (`gemv_avx2_rows`). Each row's sums still see the same operations in the same
order, so the bits do not change, and an extra row costs 0.37–0.46 of a row instead of 0.52–0.62
(40 threads, experts streamed from DRAM). `POLICY=verify` compared 31,670 rows with the card's (and
28,413 with the plane mirror's bytes) with no difference. Every A/B arm's replies are compared with
the base engine's; the final configuration also matched it on 16 chat-framed prompts, on the eight
prompts three times on one server, and on 256-token replies.

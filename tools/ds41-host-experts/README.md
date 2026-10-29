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
| `RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1` with `--max-num-seqs N` | aggregate decode 16.2 → 20.1 tok/s at N=4 (+24%), 20.6 at N=8; per stream ~1/N | concurrent requests' decode rows run in one step and a prefill chunk runs in a step of its own (the 16-token budget otherwise admits nothing beside a decode row, so N>1 served requests one after another). Every reply identical to running alone. `--max-model-len` must cover N sequences (512 holds four ~110-token requests). Arm parts Pf, NK, CK, LK |
| `..._ACCESS_TRACE=<dir>` | | per-rank access traces for `cache_sim.py` |
| miss reads on kept readers, 2 MiB pages | +4.6%, TTFT −3.6% | on by default; `RADIANCE_EXPERIMENTAL_DS41_READ_POOL=0` starts threads a call and `..._PLANE_HUGEPAGES=0` uses 4 KiB pages, as before (A/B switches) |
| missed units relaid straight into a reserved cache slot | +1.1% | on by default with `..._NATIVE_CACHE_DIRECT_FILL=1`; `RADIANCE_EXPERIMENTAL_DS41_INSERT_DIRECT=0` relays via scratch and copies, as before |
| `..._ENGRAM_SPLIT=1` | +3.5%, TTFT −2.6% | in the deepseek41 plugin (build it from this branch; the dev tree's has no switch): each rank holds and computes 6,400 of engram_kv's 25,600 rows and an all_gather rebuilds the row; the replicated 315 MB GEMV (1.4 ms a token on every card) becomes a quarter, and the ~450 MiB a card it frees holds 25 more experts a rank |
| `..._PREFETCH_ATTN=1` | +2.8% | in the deepseek41 plugin: while the card waits for a layer's CPU rows, lane 1 reads the next layer's q_a, kv, q_b and wo_a (~56 MB a card) into the 64 MB on-die cache (libr4d `prefetch`, read-only); decode passes only, and not ahead of an Engram layer |
| `..._CACHE_SECOND_CHANCE=N` | N=1 +3.9%, N=2 +5.1% (TTFT −1.5% / +1.2%) | a cached unit the decode profile calls hot (`..._CACHE_PROTECT_SHARE`, default 6 of the sealed profile's 4326 selections a layer: ~1150 units a rank) moves to the front up to N times between uses before an insert may evict it. The held-out prompts share task families with the profile; replayed on the 16 chat prompts too, critical decode misses fell 7.9% (N=1) |
| `..._PACKED_CACHE=1` | +3.2% alone, TTFT −5.5%; with `..._CACHE_SECOND_CHANCE=2` 15.52 / 15.48 tok/s (+8.2%) | the RAM cache holds each expert's codes plane plus its e8m0 scales packed to 2 bits from one base a weight (raw rounds beside, budget 512 / 320, `..._PACKED_CACHE_RAW`): x1.0444 slots and 4.4% fewer bytes a CPU hit; 7 of 15,360 experts never fit and stay misses. Refused unless the CPU takes every routed unit (all, M16, B1); the card's copy path never reads or fills it |
| `RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1` | not yet measured on the engine | the RAM cache in a CPU-only packed form: x1.044 the pairs, kernel −8.5% a cached expert (below); refused unless the CPU takes every routed unit (all, M16, B1); arm part `K` |
| `RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1` (`=2`) | not yet measured on the engine (estimate +1.2 to +3.5%) | in decode passes (in every pass issued live), a device launch whose first dispatch reads nothing the dispatches in flight on its queue write, and writes nothing they read or write, goes without the AQL barrier bit (`core/runtime/live_chain.h`): what `haz_mark` does for recorded passes, which a staged pass never is. Exact: only start order changes. Anything on the queue the issue path did not write (events, waits, copies, table patches) orders the next launch; collectives, kernels without `rad_kernel_concurrent` and host-visible launches never go unordered. Shutdown line `live chain: ...`; arm parts `Ov` / `Ov2` |
| `RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE=N` | instrument | one trunk decode pass in N on rank 0: every launch (layer, op, kernel, queue and packets, `would` / `why` / `clash`, its dependences on earlier launches, the buffers it reads and writes) into `<prefix>.ops.csv`, every kernel's start and end on card 0 into `<prefix>.stamps.csv` (`..._CHAIN_TRACE_OUT`, default `/tmp/radiance-chain-<pid>`); `chain_report.py` joins them. Without `..._CHAIN_OVERLAP` it changes nothing (the dump-only arm). Arm part `Ct` / `CtN` (N 16) |
| `RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1` | not yet measured | a routed layer's card ops (gate_up, swiglu, down, the shared expert, the ffn mix, the attention prefetch) are queued before the issuing thread waits for the routing counts: with the file buffers capped every file-tier unit is the CPU's, so the card's aliases do not depend on the routing. The counts are waited for, the CPU's share chosen and submitted and a decode pass's misses read at the layer's gather, which then waits for the CPU rows as before. The card no longer idles from its counts copy through the host's wake-up, choice and miss reads. Refused unless the CPU takes every routed unit (all, M16, B1); arm part `Ei`; shutdown line `host experts, early issue` |
| `RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1` | not yet measured | the counts row and the pool's token rows, sorted order and offsets reach the host in one ranged copy instead of four dispatches (each behind the last, each with a system-scope release), from a range list kept on the card per routed layer and pass size and rewritten in stream order when an address changes; needs host experts; arm part `Fl`; shutdown line `host experts, fetch lists` |

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
  prompts, `chat_turns` conversations in the same framing, or `messages`) and compares arms
  (arms that sent another receipt are compared among themselves); `--ttft DIR ARM...` prints each
  request's time to first token and cached prompt tokens (see Prefix caching).
- `make_prefix_receipt.py OUT MPL-2.0 core/mem/prefix.cpp` and `make_prefix_receipt.py --long OUT
  long-receipt.json`: the prefix-cache receipts in `/home/as/ds41-claude-build/work/extra-prompts/`
  (`pfxchat-receipt.json`, part `Upfxchat`; `pfxlong-receipt.json`, part `Upfxlong`).
- `cache_sim.py`: replays an access trace against LRU, SLRU, 2Q, aged LFU and Belady.
- `chain_report.py PREFIX`: reads a `Ct` arm's `PREFIX.ops.csv` and `PREFIX.stamps.csv`: per layer
  kind the dispatches before the routing-counts sync, how many went (or would go) unordered and why
  the rest waited, the card's busy time and gaps, the critical chain, and the fusion candidates
  (dependent pairs whose second launch is short and reads only the first's output).
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

## Prefix caching

The A/Bs run with `--no-prefix-cache`. The engine's prefix cache serves DeepSeek-V4.1 as it is
(read from the code; not yet run on the cards):
- The full groups (`kv_compressed_r2`/`kv_index_r2`, 2-token blocks, layers 2-19; `..._r1`, 1-token
  blocks, layers 20-39) are cached, in 2-token entries; a hit adopts their blocks on every rank (one
  block table serves all four).
- Everything else a position carries is state, so a hit resumes only at a snapshot: the window
  (`kv_window`, 128 tokens: 9 pages behind the position), the compressor's f32 tail
  (`kv_comp_tail_r2`) and the Engram token history (`kv_engram_tokens`), copied out after the chunk
  that ends there and restored on every rank before the hit's first step. The indexer's score
  plane and candidate list are rebuilt each step from the cached index keys.
- Under `--prefix-resume exact` (the default) snapshots sit where a cold prefill starts a chunk: at
  `--max-num-batched-tokens 16` the grid is every 16 tokens, whatever `--checkpoint-interval`, so a
  hit recomputes the cold run's own 16-token chunks. Snapshots are taken at every interval boundary
  and where a prompt's last chunk starts (its tail; past the first interval only).
- The CPU expert path, the file-buffer cap (`B1`) and the RAM cache see passes only: a hit's first
  chunk is an ordinary 16-token pass at a later position.

A snapshot costs no VRAM: 5.65 MiB a rank of pinned host memory (24 KiB of compressor tail, 24 B of
Engram history, 9 window pages of 640 KiB), 22.6 MiB a slot for the four ranks, one D2H copy per
rank after the chunk that ends there (~0.3 ms) and one H2D copy before a hit's first step.
`--checkpoint-slots auto` gives a sequence the geometric-backoff ladder at `--max-model-len` plus
one: at 4096 tokens 11 slots at an interval of 16 (249 MiB), 7 at 256 (158 MiB), 3 at 2048. The
cached blocks live in the KV pool the VRAM split already sizes for one `--max-model-len` sequence
(~3.1 KiB a token a card: 13 MiB at 4096), which holds finished conversations between requests at
no further cost; nothing moves from the experts.

What blocked it in the measured configuration: `--expert-ram-cache-total-mib 98304` with
`..._NATIVE_CACHE_PREGROW=1` (part G). Startup keeps `--host-headroom-mib` plus the whole cache
budget free in every host check, and after the load the pregrown slabs are already gone from
MemAvailable (12.3-14 GiB left in the 14:54 window), so the checkpoint store gets no room:
`--checkpoint-slots auto` resolves to 0 ("checkpoint slots: auto finds no room ...", which now also
names the switch) and every hit prefills from the start (an explicit count is refused). And the
default interval of 2048 leaves a prompt of fewer than 2,049 tokens without any snapshot. The
switches:

| Setting | Effect | Arm part |
|---|---|---|
| `--no-prefix-cache` dropped, `--prefix-resume exact` (the default) named | the cache on | `Pc` (also sets the next row) |
| `RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1` | the host checks after the load stop reserving what the cache's slabs already hold (`host headroom:` startup line); unchanged before the load | `Pc` (`Pc0`: the cache on without it, the control) |
| `--checkpoint-interval N` | snapshots every N tokens (a multiple of 16); a prompt needs more than N tokens for its tail snapshot | `PciN` |
| `--checkpoint-slots N` | the store's size (auto otherwise) | `PcsN` |
| `RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=1` | a finished sequence publishes only the prefill chunks it computed as an idle cold prefill cuts them (alone in their step, from one 16-token grid position to the next); without it a hit can adopt blocks the turn before last wrote in its short last chunk or in decode, under a snapshot computed from a full-chunk recompute of those positions, which is bit-identical only if every kernel rounds a row alike at every pass height up to 16 (a decode step's 1, a short chunk's, a full chunk's 16). `/stats` `publish_held_tokens` | `Pcx` |

The exactness hazard `Pcx` removes: an exact hit recomputes the cold run's chunks from its snapshot
but adopts every cached block below it, and `insert()` keeps the bytes an entry was first published
with. A third turn therefore adopts blocks the first turn wrote in its short last chunk (and, when
the client sends the model's reply back, in its 1-row decode steps) beneath a snapshot the second
turn computed from its own 16-row recompute of those positions. That is bit-identical to a cold run
only if every op rounds a row alike at 1 to 16 rows (the prefill design's audit says so; no window
has tested it). `Pc` without `Pcx` is that test.

The test receipts (`make_prefix_receipt.py`; token counts from the official tokenizer, which gives
the long receipt's measured 82/272/933/1,890/3,086 exactly), each request capped at 48 tokens:

| Receipt (part) | Requests | Prompt tokens, cold | With the cache (modelled) |
|---|---|---|---|
| `pfxchat` (`Upfxchat`) | F1-F2 (19 tokens, repeat), A1-A5 (1,207-1,221: one license text, four questions, A1 again), B1-B4 (651-991: a four-turn chat with fixed replies), D1 (B3 edited), C1-C3 (338-470: a story whose turns send the model's replies back) | 11,447 | 2,855 at `Pci16`, 3,415 at `Pci256` |
| `pfxlong` (`Upfxlong`) | L1 (= `long-3500`, 3,086), L2 (repeat), L3 (cut to 2,655), L4-L5 (3,163-3,232: L1, its reply, more questions) | 15,222 | 3,366 |

What each request should take from the cache (`cache_n` in the `--ttft` table), from an offline
model of the scheduler's exact rule, the retention ladder and the slot store:

| Request | `Pci16` | `Pci256` | Adopts (without `Pcx`) |
|---|---|---|---|
| F2 | 16 | 0 | - |
| A2, A3, A4 / A5 | 1,184 / 1,216 | 1,024 / 1,216 | - |
| B2 / B3 / D1 / B4 | 640 / 752 / 832 / 848 | 640 / 752 / 768 / 848 | B3, D1, B4: 10-18 tokens of a short last chunk |
| C2 / C3 | 336 / 400 | 336 / 400 | C3: 2 tokens of C1's short last chunk, 46 of its decode steps |
| L2 / L3 / L4 / L5 | 3,072 / 2,560 / 3,072 / 3,152 | same | L5: 14 of L1's short last chunk, 30-46 of its decode steps |

Every reply of a `Pc` arm must equal the same arm's without `Pc` on the same receipt (put that arm
first): `ab_requests.py --compare` and `--ttft` (texts `same`; `pdif` marks a prompt that differs
because an earlier reply differed and was sent back, the earlier `DIFF` is the one to read). The
`--ttft` table also lists repeats within an arm (F2 = F1, A5 = A1, L2 = L1: want `same`), replies
sent back with their own token count against the generated count (equal means the next turn's
chain runs through the decode steps' blocks), and the `/stats` prefix counters.

## Correctness

Each row the CPU computes uses the card's arithmetic: the same lane and block order, butterfly,
BF16 rounding and no fused multiply-add. A job of several token rows decodes each round of codes
once for all of them (`gemv_avx2_rows`). Each row's sums still see the same operations in the same
order, so the bits do not change, and an extra row costs 0.37–0.46 of a row instead of 0.52–0.62
(40 threads, experts streamed from DRAM). `POLICY=verify` compared 31,670 rows with the card's (and
28,413 with the plane mirror's bytes) with no difference. Every A/B arm's replies are compared with
the base engine's; the final configuration also matched it on 16 chat-framed prompts, on the eight
prompts three times on one server, and on 256-token replies.

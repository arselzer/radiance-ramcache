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
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB=56                 # ... and the rest of the next layer in use order, 56 MiB an issue point
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM=1               # ... the Engram layers too, behind their GEMM
RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1                  # the RAM cache in a CPU-only packed form (x1.044 slots)
RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=2           # profile-hot cached units get two second chances
RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1                     # FP8-sourced dense weights read from E4M3 codes
RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces                 # read-ahead of predicted misses in 512 KB pieces
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=4
RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1                    # routed card ops queued before the counts sync

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
| `RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1` | not yet measured on the engine (expected on the eight held-out prompts: heldout-05, their one end-of-generation stop, ends about one pass sooner, ~0.06 s warm and ~0.29 s with its EOS token's experts uncached as in the 29 Oct trace; the other seven, which end at max_tokens, a few ms later; decode tok/s +0.1 to +0.9%) | Every step is followed by a step issued ahead (`Scheduler::step_next`) before the host has read its tokens, so the step that samples a reply's end-of-generation token gets one more forward pass, for the finished request (max_tokens ends are predicted and get none). On the CPU path issuing a pass waits for the card at every routed layer, so `issue_ahead` returns only once most of that pass has run, and the commit that ends the reply comes after it: the reply, its `predicted_ms` and the next request all wait it out (29 Oct trace: 204-207 of its 240 selections off the card, 103-105 misses). Nothing reads that pass: its token is dropped, the prefix cache publishes a finished sequence up to its last token's position and the pass writes the one after, and its blocks are freed at its commit. With the switch, a step of one sequence that its own token can end (not `ignore_eos`) has its ids read back first (`Engine::peek_outcome`: collect_step's rules without its effects), and `step_next` issues nothing behind a step whose outcome ends every request it carries (commit's own end conditions, `Scheduler::outcome_ends_every_request`). Exact: every pass that runs ran before, and the commits are the same. The cost is the host's turnaround between two such passes, while the card waits: the next step's build, advance and the ranks' release (tens of µs a token). Steps of two or more sequences keep the step issued ahead. Arm part `Ss`; the shutdown's `post-stop passes:` line counts the passes run after a reply's last token and, with the switch, the steps read back, the wait for their ids, the turnaround and the passes withheld |
| miss reads on kept readers, 2 MiB pages | +4.6%, TTFT −3.6% | on by default; `RADIANCE_EXPERIMENTAL_DS41_READ_POOL=0` starts threads a call and `..._PLANE_HUGEPAGES=0` uses 4 KiB pages, as before (A/B switches) |
| missed units relaid straight into a reserved cache slot | +1.1% | on by default with `..._NATIVE_CACHE_DIRECT_FILL=1`; `RADIANCE_EXPERIMENTAL_DS41_INSERT_DIRECT=0` relays via scratch and copies, as before |
| `..._ENGRAM_SPLIT=1` | +3.5%, TTFT −2.6% | in the deepseek41 plugin (build it from this branch; the dev tree's has no switch): each rank holds and computes 6,400 of engram_kv's 25,600 rows and an all_gather rebuilds the row; the replicated 315 MB GEMV (1.4 ms a token on every card) becomes a quarter, and the ~450 MiB a card it frees holds 25 more experts a rank |
| `..._PREFETCH_ATTN=1` | +2.8% | in the deepseek41 plugin: while the card waits for a layer's CPU rows, lane 1 reads the next layer's q_a, kv, q_b and wo_a (~56 MB a card) into the 64 MB on-die cache (libr4d `prefetch`, read-only); decode passes only, and not ahead of an Engram layer. That stays its behaviour unless one of the three modifiers below is set |
| `..._PREFETCH_MIB=N`, `..._PREFETCH_SET=k1,k2,..`, `..._PREFETCH_ENGRAM=1` | MIB=56: 17.65 / 17.29 / 17.58; MIB=56 + ENGRAM: 17.65 / 17.73 / 17.36 tok/s against 16.53 (a dip) / 17.51 / 17.24 (+1.0 to +1.4%), every reply identical; ENGRAM overlaps the Engram all_gather, so it was also checked on the 16 chat prompts with 256-token replies: 16/16 identical, 15.84 against 15.74 tok/s. Adopted (MIB=56 + ENGRAM) | modifiers of `..._PREFETCH_ATTN=1` (deepseek41 plugin, `arch/deepseek41/ds41_prefetch_plan.h`). `MIB`: a budget of N MiB an issue point, filled in use order (`ekv` engram_kv, `comp` the compressor, `qa`, `kv`, `iq` the indexer's q_b, `qb`, `oa` wo_a's groups, `ob` wo_b, `rt` the f32 router, `sgu`/`sdn` the shared expert's gate\|up and down), the read that crosses it cut to what is left: at bf16 56 MiB is the four reads above plus 2.5 MiB of wo_b; with `..._DENSE_FP8=1`, whose E4M3 codes halve the reads (the four alone are then 26.75 MiB), the whole list through the shared expert (rank 0 55.5 MiB). `SET`: the reads named, in that order (also `iw`, `ik`: the indexer's small projections; `ma`, `mf`: the connections' mixes), whole unless a budget is set. `ENGRAM`: an Engram layer (1, 14) is read too, its engram_kv by the layer before when `ekv` is listed, the rest right behind its Engram GEMM. Each rank prints its plan at startup and its counts at shutdown (`rank R: on-die reads`). Arm parts `PmN`, `Pl<k1+k2+..>`, `Pe` |
| `..._CACHE_SECOND_CHANCE=N` | N=1 +3.9%, N=2 +5.1% (TTFT −1.5% / +1.2%) | a cached unit the decode profile calls hot (`..._CACHE_PROTECT_SHARE`, default 6 of the sealed profile's 4326 selections a layer: ~1150 units a rank) moves to the front up to N times between uses before an insert may evict it. The held-out prompts share task families with the profile; replayed on the 16 chat prompts too, critical decode misses fell 7.9% (N=1) |
| `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1` | not yet measured on the engine | the cards' static expert residents in the same packed form (below): 18,001,920 instead of 18,800,640 bytes a unit, so x1.0444 the residents (with F8, 712 / 720 a rank should become ~743 / 752); needs a `libr4d.so` with the row `moe_gemm_ggml_mxfp4_gemv_vp` in the run home, `--deterministic` and passes of at most 16 tokens; `..._VRAM_PACKED_RAW=<gate_up>,<down>` (512,320) sets the raw rounds of room; arm part `Vp`, which runs only on a `libr4d.so` the device check passed (`VPACK_CHECK`, below) |
| `..._PACKED_CACHE=1` | +3.2% alone, TTFT −5.5%; with `..._CACHE_SECOND_CHANCE=2` 15.52 / 15.48 tok/s (+8.2%) | the RAM cache holds each expert's codes plane plus its e8m0 scales packed to 2 bits from one base a weight (raw rounds beside, budget 512 / 320, `..._PACKED_CACHE_RAW`): x1.0444 slots and 4.4% fewer bytes a CPU hit; 7 of 15,360 experts never fit and stay misses. Refused unless the CPU takes every routed unit (all, M16, B1); the card's copy path never reads or fills it |
| `RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1` (`=2`) | neutral: 16.85 / 16.65 against 17.07 / 16.36 tok/s (47% of launches went unordered; the card's chain is not the critical path) | in decode passes (in every pass issued live), a device launch whose first dispatch reads nothing the dispatches in flight on its queue write, and writes nothing they read or write, goes without the AQL barrier bit (`core/runtime/live_chain.h`): what `haz_mark` does for recorded passes, which a staged pass never is. Exact: only start order changes. Anything on the queue the issue path did not write (events, waits, copies, table patches) orders the next launch; collectives, kernels without `rad_kernel_concurrent` and host-visible launches never go unordered. Shutdown line `live chain: ...`; arm parts `Ov` / `Ov2` |
| `RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE=N` | instrument | one trunk decode pass in N on rank 0: every launch (layer, op, kernel, queue and packets, `would` / `why` / `clash`, its dependences on earlier launches, the buffers it reads and writes) into `<prefix>.ops.csv`, every kernel's start and end on card 0 into `<prefix>.stamps.csv` (`..._CHAIN_TRACE_OUT`, default `/tmp/radiance-chain-<pid>`); `chain_report.py` joins them. Without `..._CHAIN_OVERLAP` it changes nothing (the dump-only arm). Arm part `Ct` / `CtN` (N 16) |
| `RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1` | +1.9% over five interleaved pairs, each won (17.40 / 16.95 / 17.22 / 17.39 / 17.59 against 17.07 / 16.36 / 17.07 / 17.34 / 17.11 tok/s), replies identical | a routed layer's card ops (gate_up, swiglu, down, the shared expert, the ffn mix, the attention prefetch) are queued before the issuing thread waits for the routing counts: with the file buffers capped every file-tier unit is the CPU's, so the card's aliases do not depend on the routing. The counts are waited for, the CPU's share chosen and submitted and a decode pass's misses read at the layer's gather, which then waits for the CPU rows as before. The card no longer idles from its counts copy through the host's wake-up, choice and miss reads. Refused unless the CPU takes every routed unit (all, M16, B1); arm part `Ei`; shutdown line `host experts, early issue` |
| `RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1` | neutral: 16.36 / 17.19 against 17.07 / 16.36 tok/s | the counts row and the pool's token rows, sorted order and offsets reach the host in one ranged copy instead of four dispatches (each behind the last, each with a system-scope release), from a range list kept on the card per routed layer and pass size and rewritten in stream order when an address changes; needs host experts; arm part `Fl`; shutdown line `host experts, fetch lists` |
| `RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=N` (2-4, designed for 3) | neutral (17.19 / 17.61 / 17.20 against 16.53 / 17.51 / 17.24 tok/s): N=3 cuts a lone miss's mirror leg 2.59 → 2.43 ms (copy 0.217 → 0.102 ms), but the NVMe leg (~2.75 ms) lands last in 95% of calls either way, so the miss still takes ~2.8 ms | a decode pass that misses one unit reads the plane mirror's decode tail (2.4 MB at 0.13) in N pieces on N readers, so the SATA stripe's preads and the copies out of the bounce buffers overlap; the same bytes (pieces start on the container's 4 KiB grid, so a failed one falls back to the container as a whole tail does). Needs the decode mirror (`R`, `RdNN`); arm part `MsN`. The read-ahead in pieces (`PpK`) reads the container only, so its pieces are not split; the whole-unit read-ahead (`PoK`/`PtK`) reads a unit as a lone decode miss and is split too. Gate: the shutdown report's "one-token miss reads, one unit" line; drop it if the mirror lands last in under 25% of calls and its copy takes under 0.15 ms |
| `RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1` | neutral (17.48 / 17.15 / 17.27 against 16.53 / 17.51 / 17.24 tok/s; with Ms3 17.53 / 17.35 / 17.61): every one-unit miss posted at the choice, no change in the caller's wait | a one-token pass's missed planes start on the kept readers as soon as `hx_choose` knows them, instead of after `hx_submit` (its pool wake, 28-32 µs a layer) and the copy path's staging; the issuing thread then waits in `hx_submit_misses_now` (`NativeReadWorkers::post`/`join`, which hold the readers' call mutex from post to join as `run` does). The same reads: the set `hx_submit_misses_now` reads, so with the read-ahead in pieces (`PpK`) never a unit read ahead whole nor one the piece readers are finishing (whose slot `hx_ra_settle` may give fresh planes); the post is a demand read for the piece readers from the post on, and is finished before the urgent units are settled. Also at an early-issued layer's gather (`Ei`). A post that finds the readers on another thread's call (the whole-unit read-ahead's, `PoK`/`PtK`, or the async misses' worker, `A`) does not wait: it posts nothing and the layer reads as before (the mover's "kept plane readers" line counts them). Arm part `Mp`; the "readers" line shows the readers' start and the caller's return, and the EP decision's own `posting the misses' reads` line its time (not in "choose") |
| `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1` | not yet measured on the engine | in the deepseek41 plugin and libr4d (build both from this branch): the 40 routers ([384, 5120], 300 MiB a card replicated) and the kv\|gate projections of the 3 compressors with a ratio above 1 ([1024, 5120], layers 2, 8, 14) are f32 in `gemm_nt_f32` but ship bf16. The container holds the routers widened to f32 (`scan_low_halves.py`: 0 of 78.6M low halves nonzero), which libr4d's `gemm_nt_f32bf16` stores as their bf16 halves, and the compressors as bf16, which the loader had widened and which are now read as they are: 180 MiB a card back to the experts (~+10 a rank; 150 MiB of it the routers'). The kernel widens each by a shift in the same tiles, lanes and order, so every logit is bit-identical. The load refuses an f32 weight with a nonzero low half rather than rounding it; startup prints a `ROUTER_BF16:` line on rank 0 counting the weights held at two bytes an element. The on-die reads (`PmN`, `Pl..`) take both at their bf16 bytes (`rt` 3.75 MiB, `comp` 10), and the router predictor (`PpK` and the others) reads the router's f32 plane from the container as before. Gate: `r4d_selftest --perff32 384 5120 1` at or under 14 us for the halves (19.7 us f32 in the engine); the same run byte-compares and times tile forms 1-8, and `..._ROUTER_BF16_FORM=N` makes the halves take form N at passes of up to 16 rows without a rebuild (forms are byte-identical). Arm parts `Rb`, `RbN`: `ab_window.sh` measures them only when `SELFTEST_F32BF16=<r4d_selftest>`'s card checks (the check case, then every form byte-compared and timed at 1 and 16 rows at both shapes) passed in the same window on the arm's own libr4d.so |
| `RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=<prefix>` | instrument, exact; overhead not yet measured (rank 0: three more copies, ~1.6 KB, a routed layer behind its counts row's, which the CPU's choice does not wait for; every rank: one cache lock a decode layer) | the routing trace for the lossy track (`core/place/route_trace.h` has the format; below). Arm part `Rt` (files under `rtrace-<arm>/`); shutdown lines `route trace: ...` per rank (picks by status) and for the files (records, bytes, drops) |
| `RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1` | not yet measured (model, layer-passes the cut leaves at 16-row passes: TTFT −2% at 82 prompt tokens, −3.5% at 272, −10% at 933, −20% at 1,890, −30% at 3,086, −41% at ~7K; DS41Engine realised 0.8-0.95 of its model) | the exact consumer-suffix skip (`core/sched/consumer_skip.h`, schedule and proof; the deepseek41 plugin's `cut_declare` / `step()`). Layers 21..39 see earlier rows only through layer 20's compressed and index keys and their own 128-row windows, so a prompt's early pure prefill passes run layers 0..F-1 in full, layer F's window-key ops alone (the full layer's own ops, so the stored bytes are the same; layer 20 also its compressor and key stores), and skip the rest, F ≥ 20 rising by 8 passes a layer to 40 at the prompt's last pass. Every computed row and stored key is the base engine's, and no row reads a key the request did not write; the next step's windows and the tail checkpoint's are complete (with the prefix cache on, a prompt is cut for its tail checkpoint and no snapshot below it is written). Nothing is read, staged, predicted or computed for a skipped layer (file stager, read-ahead, CPU path, next-layer prefetch). Refused, with the reason, for speculation, KL or embedding modes, media, R-SWA, concurrent sequences with chunks wider than the 16-token quantum (unless `..._SCHED_PREFILL_ALONE=1`), and a model the derivation does not cover. Startup lines `consumer skip: on` (scheduler) and `consumer suffix declared` (plugin); shutdown lines `consumer skip: ... layer-passes run in full, ... key ops alone, ... skipped` (engine) and `rank R: consumer skip: ...` (plugin, the same counts). Needs a radiance and a deepseek41 from `claude/ds41-cskip` (run home `home-cskip`); arm part `Cs` |
| `RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1` | instrument | the exactness evidence for the consumer skip, usable without it (`core/engine_hash.cpp`): at the end of each prompt's prefill, before its sampler and before the next step, every rank hashes each KV group the prompt wrote (whole blocks of a full group, the last window of the windowed one, the state slots), each layer apart, and the last row's logits; lines `prefill hash: seq S, N tokens (...), rank R: <group>[layers]=<hash> ... logits=<hash>`, the window group by layer, and `prefill hash: seq S, N tokens: all ranks <digest>`. Steps are not issued ahead while it is on, so time prompts in arms without it. Arm part `Kh`; the window compares the Kh arms' hashes with the first Kh arm's (`ab_requests.py --hashes DIR ARM...`) |

The shutdown report counts the miss reads of one-token passes whatever the switches (two lines a
rank, one for calls of one missed unit and one for more): how many calls had a mirror leg and in how
many the mirror landed last (and by how much), the legs' mean times from the dispatch and the
mirror's copy out of its bounce buffer (the longest piece's and all of them), when the first read,
the first container read and the last read began, and how long after the last read landed the
issuing thread was back (a `done_` wake when a reader, not the caller, finished last). The plane
mirror's line gives its copy time too, and the mover's "kept plane readers" line how many calls were
posted.

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

## Packed VRAM residents

The cards hold 712 / 720 static expert residents a rank in block_mxfp4 in the recommended configuration
(with F8; 598 / 615 without). With `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1` they take the packed form the
RAM cache uses (the codes plane and 2-bit e8m0 codes from one base a weight, raw rounds beside, room for
512 / 320 raw rounds): 18,001,920 bytes a unit instead of 18,800,640, so the same weights pool holds
x1.0444 the residents. From the room each rank left (712 units and 3.90 MiB free on rank 0, 720 and
10.62 MiB on the others) that is 743 / 752, 31 / 32 more a rank, one fewer for each ~10-15 MB that
weights over their raw-round budget (charged block_mxfp4) take back (`core/place/vram_packed.h` has the
whole design).

- **Selection.** The engine sets the machine fact `device.ds41_vram_packed` before declare, which lets
  libr4d's `moe_gemm_ggml_mxfp4_gemv_vp` row take the routed GEMMs' decode band (priority 59 over the gemv
  row's 58; without the fact the row never matches). The row declares the gemv row's stored form, so the
  pinned pool, the file tier, the RAM cache, the CPU path and the expert profile's checks all keep
  block_mxfp4: only the VRAM copies change.
- **Planning.** A weight is eligible when it is a trunk routed expert's `ggml_mxfp4` gate_up or down,
  whole on its rank, read only by `moe_gemm_ggml` ops every band of which resolved the gemv_vp row (so
  `--max-num-batched-tokens` at most 16). The expert fill charges a unit at its packed size after reading
  its weights' scales from the container (only the units it considers for VRAM: 1.1 MB a unit, about
  0.8 GB a rank with F8); a weight over its raw-round budget stays block_mxfp4 in VRAM and is charged so.
  Only a static plan (`--deterministic`) packs. The startup log prints per rank what was packed and how
  long the scan took; the planner report (`--debug-placement`) has a `vram packed` line.
- **Loading.** The loader packs each such weight from its planes and checks it before the upload: every
  block's e8m0 as the kernels widen it equals the planes' byte for byte, and the codes plane is unchanged.
  Any mismatch refuses the load.
- **Issue.** A packed weight's table entry carries bit 0 (weights are 64-byte aligned). gemv_vp decodes a
  marked entry from the packed form and any other (the pinned pool's, staged and aliased experts) as
  block_mxfp4; a packed block is one 128-bit load of its 16 code bytes and its scale from its round,
  then the block path's arithmetic in its order (`Mx<MX_PK>::dec`, `mx_unit`), so every output is the
  plain row's bit for bit. A marked entry whose header is not a packed weight's gives NaN rows (loud,
  never quietly wrong).
- **Checks.** `tests/r4d_emu/test_ds41vp.cpp` runs both rows on the host SIMT emulator over the kernel's
  every path (whole K and chunks, one and two rows a wave, 1 to 16 tokens, partial rounds of 1, 8, 9 and
  31 blocks, raw rounds anywhere, edge scales and activations, the research sample's real scales at the
  served shapes, a rank's 96-entry table) and compares every value; `r4d_emu_ds41vp_tail` runs it again
  with the live-tail ranges on (`RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1`), where the cases at their
  geometry take the tail branch (gemv_vp's kernel, then the dropped slots zeroed). `place_test` holds
  `vram_pack_eligible` to what it may pack (a tiled or kernel-less band, another op or operand position,
  a draft head, a weight stored as read, half a weight, a form no smaller: none) and the scan and the
  planner's query to the container's scales (raw rounds, over the room, unreadable); `runtime_test` holds
  the table mark (bit 0 on packed residents' entries only, kept through patches).
  `ds41_vpack_device_check.hip` compares both rows on a card (`check 48`; `check 48 tail` with the
  live-tail ranges on, the dropped slots' zeros compared too) and times them (`time`: plain over block,
  gemv_vp over packed, and gemv_vp over unmarked block entries, what the new kernel costs the experts it
  does not pack). Build it like the other harness and give the window `VPACK_CHECK=<binary>`: before the
  arms, once for each `libr4d.so` a Vp arm serves (by md5), ab_window.sh runs `check 48` and `time` on
  card 0 (and `check 48 tail` when the window's environment has the live-tail ranges on), and a Vp arm
  runs only when each printed PASS and exited 0 on its library; otherwise it is skipped. By hand on a free
  card: `ds41_vpack_device_check LIBR4D.so check 48` (PASS on exit 0) and `... time`.
- **Shutdown report.** The mover's report has a `VRAM packed experts` line: residents and weights
  packed, their bytes against block_mxfp4's, weights packed and checked at load, weights over budget
  kept block, the scan.

## The lossy track: routing trace and quality harness

Exact past ~18-19 tok/s means more RAM, a second drive, or giving up exactness on misses
(`/home/as/ds41-decode-review-20261028/RESEARCH-3-LOSSY.md`). Before any lossy change, two measuring
tools, both exact (every reply stays the base engine's):

**The routing trace** (`RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=<prefix>`, arm part `Rt`). Per
EP-routed layer of every pass of up to 16 tokens: the six picks in the router's round order with
their weights (the card's) and biased keys (score + correction bias, recomputed on the host in f32
from the logits), rank 0's 384 router logits on a one-token pass, the layer's bias once, and for
every pick what the rank owning it did: on the card (V), RAM hit (H), miss read while the layer
waited (M), read ahead whole (A) or in part (P), the card's copy path (C). Every rank also records
which of its file-tier experts the RAM cache held at the choice (one lock, no lease or LRU touch:
`Mover::units_cached`) and, once, every local expert's tier; the step loop records which request
and position each pass carried and where each request ended, so the post-stop pass (issued ahead
of a request that had just stopped) is flagged. Rank 0 copies its layer's ids, weights and logits
to pinned memory right behind the counts row's copy, under an event of its own that the CPU's
choice does not wait for; a background thread
(nice 10) writes `<prefix>.rank<r>.bin` and `<prefix>.steps.bin`. Rank 0 writes ~1.7 KB a decode
layer (~66 KB a token), the others ~60 bytes.

`rtrace.py report PREFIX ARM.json` (ARM.json: the arm's `ab_requests.py` output, for the requests'
names, families and measured decode time) prints the assessment's gates over decode tokens,
post-stop passes apart: the weight share a miss carries (and the mass drop-all removes a token); the
weight-threshold drop gate (go if 50% or more of misses carry a share below tau 0.1); the
cache-aware routing gate (go if 40% or more of misses have a held substitute -- on the card, or in
the owner's RAM cache at that moment -- with a key within delta 5% of the 6th key, kill below 20%;
`--delta` sweeps it); the misses' slots (protect-top-2: 60% at slot 2 or below); all by prompt
family and by layer; and a replay estimate of decode tok/s for drop-all, drop below tau, a cap of 6
drops a token and the protect-top-k arms (drop only misses at slot 2 or below, 4 or below) from the
assessment's time model (T = 24.7 + 2.54 x waited misses + 0.133 x CPU hits ms a token, and the
constrained 31 / 2.28 / 0.108), each also with its fixed part refitted to the arm's measured time,
and a third model fitted on the trace's own token times (rank 0's pass starts). `--json OUT` keeps
the tables. A miss there is M or P; A counts as a hit. Single-sequence decode passes only (C1 arms).

**The quality harness** (arm part `Lq`; `ab_requests.py --lp / --force / --kl`). After the arm's
timed requests (unchanged), the window's first `Lq` arm of a prompt set records the reference:
free-running greedy replies with each position's logprob and top 20 from the raw logits (the
sampler's `sample_logprobs`, before any stage), into `lpref-<set>.json` (`LQ_REF=<file>` names an
existing one instead). Every `Lq` arm then decodes teacher-forced along it: each prompt with a GBNF
grammar of the reference's token ids (`<[id]> <[id]> ...`, so tokenisation cannot drift), so every
position is a decode pass over the reference's own prefix (KV a lossy arm wrote stays in the cache,
unlike a re-sync that re-prefills); a reply that stopped on its own is forced up to its stop token,
which the finished grammar leaves to the model among the end-of-generation ids. Each forced run
first probes the grammar: one prompt's first position forced to its second-likeliest token, which
must come back (an exact arm forced along its own reference would match it even unforced). The
window ends with `ab_requests.py --kl-dir`: per forced file, the positions that followed the
reference, the bit-identical ones, and KL(ref || arm) over the reference's top 20 plus a tail
bucket, top-1 agreement and |d logprob| of the reference token, overall, without position 0 (a
prefill pass's) and by prompt family (the receipt's `domain`, else the name's prefix), and the
probe's outcome. A grammar disables the step issued ahead (`Scheduler::step_next`), so forced runs
are slower and have no post-stop pass; logits do not change. The exact arm against itself must give
every position bit-identical: KL 0, top-1 100%.

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
  prompts, or `messages`) and compares arms; `--lp`, `--force`, `--kl`, `--kl-dir`: the quality
  harness above (`ab_requests_test.py` runs it against a fake server with a toy model: an exact
  model forced along its own reference is bit-identical, a perturbed one gives KL > 0).
- `rtrace.py report|dump|synth`: reads the routing trace (above); `rtrace_test.py` checks it on a
  synthetic trace with known answers, and on the C++ writer's own file when
  `RTRACE_CPP_PREFIX=<dir>/rt` (made by `RTRACE_DUMP=<dir> route_trace_test`).
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
- `ds41_vpack_device_check.hip`: libr4d's `moe_gemm_ggml_mxfp4_gemv_vp` over packed experts against the
  plain gemv row over block_mxfp4, byte for byte, and both timed (packed VRAM residents, above).
- `ds41_host_experts_test.cpp`: host tests of the kernels against the scalar reference.
- `scan_low_halves.py model.rad`: whether a container's f32 routers (and compressor kv|gate
  projections) are bf16-valued, every low half zero, as `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16`
  requires (its load refuses the weight otherwise). ~300 MiB of reads; outside a window.

## Correctness

Each row the CPU computes uses the card's arithmetic: the same lane and block order, butterfly,
BF16 rounding and no fused multiply-add. A job of several token rows decodes each round of codes
once for all of them (`gemv_avx2_rows`). Each row's sums still see the same operations in the same
order, so the bits do not change, and an extra row costs 0.37–0.46 of a row instead of 0.52–0.62
(40 threads, experts streamed from DRAM). `POLICY=verify` compared 31,670 rows with the card's (and
28,413 with the plane mirror's bytes) with no difference. Every A/B arm's replies are compared with
the base engine's; the final configuration also matched it on 16 chat-framed prompts, on the eight
prompts three times on one server, and on 256-token replies.

## Many token rows of one expert

`gemv_avx2_many` computes up to 32 token rows of one matrix in one call, for prefill passes whose
rows meet in one expert. A tile's output rows go in blocks of 16; each round of each row is decoded
once into f32 weights (w = kv · 2^(e−128), exact), and each token row adds x · w value by value in
register blocks of two output rows by four token rows. Those are the card's two operations a value,
in its order, so every row equals `gemv_avx2` on that row alone: `ds41_host_experts_test check`
holds that at 1–33 rows in every layout (block_mxfp4, planes, packed) of every case (the release's
scales, the edges with NaN and Inf rows, partial rounds) and over output rows [8, N); `packed` adds
nine rows on every packed case.

`RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=N` (1–32; default 4, the behaviour so far) lets a
pool job carry up to N token rows of one expert; with N above 4, jobs of four or more rows run
`gemv_avx2_many` and smaller ones `gemv_avx2_few` (at four rows the many-row kernel is 10–18%
faster, at three about even, at one or two slower). Arm part `MrN`. The startup line names the
limit; each rank's shutdown report counts the jobs that ran the many-row kernel. A 16-token pass
gives an expert at most 16 rows, so `Mr16` and `Mr32` act alike there.

In the pool's shape (`ds41_host_experts_test rbench 32 40 packed`: 40 threads, 64-row tiles, 32
experts with the sample's scales streamed from DRAM and queued at once; medians of seven), ms an
expert:

| token rows | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| jobs of up to four (`gemv_avx2_few`, today) | 0.37 | 0.34 | 0.49 | 0.99 | 1.91 | 3.80 |
| one job (`gemv_avx2_many`) | 0.38 | 0.35 | 0.40 | 0.60 | 1.01 | 1.85 |

A further row costs 0.050–0.053 ms with the many-row kernel (from 4 to 32 rows) against
0.115–0.118 ms in jobs of four; planes and block_mxfp4 measure within 6% of packed (16 → 32 rows:
0.054 and 0.052 ms a row). That is under the prefill critique's 0.06 ms gate, so 1,024-token chunks
can stay on the CPU, whose floor becomes about 8 ms a token (158 non-resident rows a token × 0.052
ms). One thread with the matrix in L3 (`rhot`): 0.100 cycles a row-weight at 32 rows against 0.189
for `gemv_avx2_few`, where Zen 3's two multiply and two add pipes allow 0.0625.

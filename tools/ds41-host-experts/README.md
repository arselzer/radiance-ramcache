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
  of layer L+2 read at layer L in small pieces that give way to demand reads; one-token passes, and
  with `..._PREFETCH_MULTIROW=N` decode passes of 2..N rows (every row predicted, the union read).

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
RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1                 # prefill: layers 21-39 only on rows that reach the output
RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT=512          # with the prefix cache: the snapshots within 512 tokens below a cut prompt's target whole (the default when unset; all: every one, N: those within N tokens, 0: none -- see the table)
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=16          # prefill: CPU jobs of up to 16 rows of one expert, many-row kernel
RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1                   # routers and compressor kv|gate held as bf16 halves (needs the libr4d whose f32bf16 card checks passed)
RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1                   # VRAM residents in the packed-scale form, ~31 more a card (needs the libr4d whose gemv_vp card check passed)
RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1                # no pass after a reply's end-of-generation token
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1               # a partly read read-ahead unit's rest read whole in its layer's own read
RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB=256            # background inserts in 256 KiB pieces, waiting for an idle pool
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int8             # the read-ahead predictor scores in int8, its best 12 again in bf16
RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=7                    # up to 7 units a rank the packed cache refuses keep their planes
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1            # pool workers sleep on a futex, woken without a lock

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
| `RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1` with `--max-num-batched-tokens 32` (or 48) | byte-identical (--kld-record over 1,512 positions; card sweep up to 32 rows) but alone slower: TTFT 99.5 → 108.4 s at 3,086 tokens (+9%), held-out +4.5%; to be tried with 32-row CPU jobs (MAX_ROWS=32) | prefill passes of up to 32 (48) tokens, every reply, logit and KV entry the 16-token chunking's. Four kinds of op change arithmetic with the pass height: the routed MoE GEMM above 16 tokens (the gemv row's band ends there; the tiled row sums in another order), the sparse attention and the indexer score above 48 rows (WMMA forms), and every norm above 32 rows (`r4d_act_threads` halves the width, which regroups the sum of squares). Under the switch libr4d adds `moe_gemm_ggml_mxfp4_gemv_tall`, the gemv one priority below the gemv row with its bound on M raised to the tallest pass libr4d keeps exact (`libr4d/r4d_ds41_tall.h`), and keeps the sparse ops on their scalar forms at every height; the core raises the counts contract to 48 tokens and the CPU path to the step (its pinned rows, sorted order, output rows and fetch lists), and lets the file-buffer cap take passes of the step with the misses on the CPU in all of them (`..._MISSES` at least the step). At startup `FileStager::tall_init` refuses a step whose passes would change bits: above 48 tokens, a routed MXFP4 GEMM band off the gemv rows (a libr4d without the twin, or a step past its bound), or any other op on two kernels other than gemm_nt's m16 / m64; rank 0 then logs `exact tall passes: passes of up to N tokens ...`. Arm parts `Xt` and `BTN` (e.g. `...-M32-...-Xt-BT32`); the byte gate is a pair of `Kld` arms (below, Tools). Tests: `ds41_tall_test`, `hx_fetch_list_test` (16 / 32 / 48-row tables), `r4d_emu_ds41tall` and `r4d_emu_ds41tall_on` (the twin's rows = the gemv's in passes of 16; the fused norms at heights 1..48), `ds41_mxfp4_device_check LIBR4D tall` and `sweep` / `sweep0` on a card |
| `RADIANCE_EXPERIMENTAL_DS41_LCP=C` with `--max-num-batched-tokens C` (C = 1024) | not yet measured (model: 3,086 tokens in ~37 s alone, ~27 s with the per-row consumer cut, against 99.5 s at 16-token passes and 69 s with `Cs`; decode about −1.3% at C = 1024 for the permanent arena) | large-chunk prefill: chunks of C tokens with every row, logit and KV entry the 16-token chunking's. libr4d keeps the exact-tall pins at every height and raises its twin `moe_gemm_ggml_mxfp4_gemv_tall` to 2048 rows (`libr4d/r4d_ds41_tall.h`); `FileStager::lcp_init` admits a step only when every routed MXFP4 band is on the gemv rows with the twin reaching C, every other op is on one kernel or on a known byte-identical pair (gemm_nt m16 / m64 / tiled, the exact all-reduce's one- and two-shot), `--max-num-batched-tokens` is C, and nothing outside what was proven is on (EXACT_TALL other than 1, speculation, `..._FETCH_LIST`, `..._VRAM_PACKED`); `hx_init` requires the CPU to take every routed unit (`..._POLICY=all`, `..._MISSES` ≥ C, `B1`); `Scheduler::lcp_init` requires chunks of C (the checkpoint interval a multiple of C, or a span) and a step a chunk alone (`--max-num-seqs 1` or `..._PREFILL_ALONE`), and every rank's stager to have taken it. Misses stream to the pool as each unit lands (at most `..._LCP_PLANES` read at once), jobs are cut evenly and submitted longest first, and the RAM cache admits only what the prompt's last `..._LCP_ADMIT` tokens (128) select, as the chunks reach them (earlier rows borrow read-only; at most `..._LCP_INSERTQ` admitted misses queued). With `Cs` and the plugin's declared 16-row grain the consumer skip cuts each chunk per row (`RadBatch::consumer_row0`). Rank 0 logs `large-chunk prefill: passes of up to C tokens ...`. Arm parts `LcN`, `AdN` / `Ada`, `LpN`, `LiN` (below) |
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
| `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1` (`=2`) | not yet measured (estimate: the per-unit wait 2.4–2.7 → ~2.1 ms with 1 and ~1.9 ms with 2, on ~2.4 such units a token: +0.5 to +1.4%; the drive-cache verifier priced the container-only form at about 0 and the mirror's share at +0.4–0.5%) | a partly read unit its layer needs (`PpK`, ~6.5 MB of its ~18.8 read ahead) is finished by the layer's own read instead of urgent 512 KiB pieces (~20 reader round trips a unit, 4.9 GB/s): `hx_ra_take` stops it (`PieceReader::stop`: no further piece starts), and the call that reads the layer's other misses on the mover's kept readers reads the rest of each extent, one large read an extent's rest at the demand read's depth (`Mover::read_unit_planes` with `PlanesIn`), while the pieces in flight land; `hx_ra_settle` then waits for those (a failed piece: the unit read again whole). The same bytes at the same offsets: the planes are the container's either way. `=2` gives the plane mirror its decode share of the rest (`Mover::rest_mirror_tail`: the decode tail scaled by the share of the unit left, at most the w3 codes past the pieces), so a barely started unit is finished too instead of read afresh; it needs the decode mirror (`R`, `RdNN`). With `_MISS_POST` the rest is posted with the other misses. Needs `_PREFETCH_FINISH` (on by default). Arm parts `Pu` (1), `Pu2` (2), pinned to 0 in every other arm; every rank names it at startup ("reads the rest of a partly read unit its layer needs ... PREFETCH_URGENT=N)"), and the shutdown report's "read-ahead in pieces, rests on the kept readers" line counts the rests and their mirror MB, "read-ahead in pieces, partly read units" (in every `PpK` arm) the per-unit wait from the layer's choice to the unit whole, and the same wait on the units every arm finishes (started at least the mirror's decode share when the layer took them: `=2` also finishes barely started units, ~16-18 MB rests, which the others read afresh; the gate: that figure must fall), the rests line the units whose rest the mirror read part of (a count, which a Pu2 arm must show above 0), the readers' line the units stopped, and "one-token miss reads, with a read-ahead unit's rest" these calls' legs apart from the plain misses' |
| `..._PREFETCH_SCORE=int8\|int4`, `_PREFETCH_RESCORE` (12), `_PREFETCH_SCORE_CHECK=N`, `_PREFETCH_RECORD=<path>` | not yet measured on the engine (estimates: the predictor's ~597 us a prediction to ~300 us with int8 and ~150-200 us with int4, +0.3-0.8% decode). Offline (`router_score_eval` on the container's routers, 3,648 synthetic inputs, iid and aligned): int8 with 12 rescored gave bf16's own top 4, in its order, every time (bf16's picks all within int8's best 8); int4 98.6 / 99.5% at 12, 99.3 / 99.8% at 16, 99.9 / 100% at 24. One thread, quiet box, tables cold: bf16 228 us a prediction, int8 133 + 12 us rescoring, int4 100 + 11 us. No trace keeps real router inputs yet: a `Pv` arm records them for the tool, and a `PkN` arm checks against bf16 live | the read-ahead's predictor (`router_predictor.h`, arithmetic in `core/place/router_score.h`) scores every expert from a quantized copy of the routers first -- one scale an expert (int8 `max\|w\|/127`; int4 `w ~ s (n - 8)` with the clip of `max\|w\|` that gives the row the least squared error), the input quantized once to int16, integer dots (`vpmaddwd`, exact) -- and only its best `_PREFETCH_RESCORE` again in bf16; the best k of those are the prediction, so it is bf16's own whenever bf16's best k are among the rescored. A prediction streams 1.97 / 0.98 MB instead of 3.93 MB from DRAM during the pool's bursts (79 / 39 MB of RAM beside the bf16 copy, 2 MiB pages; quantized at load, ~0.2 / ~0.25 s). Only reads ahead change, never a reply. `_PREFETCH_SCORE_CHECK=N`: one prediction in N is also scored in bf16 alone after it was handed over and compared (a precision arm, not a timed one). Startup: the predictor's line says `scored in int8\|int4 (...), the best 12 scored again in bf16`; shutdown: `router predictor scoring` (the quantized pass and the rescoring a prediction) and `router predictor check against bf16 alone` (the same set, the same order, bf16's picks by their rank in the quantized scores, those past the rescored). `_PREFETCH_RECORD=<path>` (any predictor mode): every posted router input is appended to the file after its prediction was handed over (64-byte header `RADPRDR1`, then u32 pass, i32 layer, 5120 bf16 a record, ~190 MB an arm of the held-out prompts) for `router_score_eval` (Tools) to score offline in bf16, int8 and int4; shutdown line `router predictor record`. Arm parts `Pi8`, `Pi4`, `Pi8rN` / `Pi4rN` (rescore N), `PkN` (with Pi), `Pv` (the record) |
| `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=1` | not yet measured (estimate +0.1-0.2%; rank 0's EP-decision `choose` 8.1-8.9 us should fall to ~4.3 us, the `predictor post` part showing the post's 4-6 us after the submit) | with `..._EARLY_ISSUE=1` and the read-ahead in pieces: at an early-issued layer's gather rank 0 posts the read-ahead's router input after the layer's CPU work is submitted, as `start_ep_routed` does, instead of before `hx_choose`, where the post (its copy of the input and the predictor's wake) held back the pool's start. Refused without Ei and pieces. Startup line `rank 0 posts the read-ahead's router input once a layer's CPU work is submitted`; arm part `Pw` |
| `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=N` (2-16) | not yet measured | the read-ahead in pieces in decode passes of 2..N rows too: DSpark's verify passes (2 rows at cap 1, up to 6) and several sequences' decode rows, which otherwise all go without it (window C's `Ds1`/`Ds2` arms: 0 inputs posted, 0 pieces read). Rank 0 posts every row's router input at the one-token pass's points; the predictor reads layer L+2's router once for all the rows (each row's logits bit for bit the one-row kernel's) and hands every rank the union of the rows' top K (every row's best before any row's second), planned under the same live cap, with the same yields to demand reads, finishing rule (the plane mirror's share for the pass's rows) and epoch check. One-token passes go as without it; prefill chunks, mixed passes and the drafter's own passes never read ahead. Reads only. Refused unless the read-ahead in pieces runs (`..._PREFETCH=pieces`, `..._MISSES` ≥ N). The on-die reads (`Pa`/`PmN`/`Pe`: decode passes of up to 16 rows) and early issue (`Ei`) already covered verify passes (window C's Ds1 arm: on-die reads in all 269 verify passes, 11,640 early layers). Arm part `PmrN` (`Pmr`: 6), with a `PpK` part; startup line `read-ahead: decode passes of 2-N rows read ahead too`; shutdown lines `read-ahead in pieces by pass size` (each rank's planned / used / dropped by the pass's rows) and `router predictor, multi-row decode passes` (the scoring a row) |
| `RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1` with `--max-num-seqs N` | aggregate decode 16.2 → 20.1 tok/s at N=4 (+24%), 20.6 at N=8; per stream ~1/N | concurrent requests' decode rows run in one step and a prefill chunk runs in a step of its own (the 16-token budget otherwise admits nothing beside a decode row, so N>1 served requests one after another). Every reply identical to running alone. `--max-model-len` must cover N sequences (512 holds four ~110-token requests). Arm parts Pf, NK, CK, LK |
| `..._ACCESS_TRACE=<dir>` | | per-rank access traces for `cache_sim.py` |
| `RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1` | alone 17.40 / 17.80 / 17.35 against 17.37 / 17.65 / 17.69 tok/s (−0.3%), replies identical; with Rb and Vp 17.86 / 17.99 / 17.96 (+2.1%); being separated | Every step is followed by a step issued ahead (`Scheduler::step_next`) before the host has read its tokens, so the step that samples a reply's end-of-generation token gets one more forward pass, for the finished request (max_tokens ends are predicted and get none). On the CPU path issuing a pass waits for the card at every routed layer, so `issue_ahead` returns only once most of that pass has run, and the commit that ends the reply comes after it: the reply, its `predicted_ms` and the next request all wait it out (29 Oct trace: 204-207 of its 240 selections off the card, 103-105 misses). Nothing reads that pass: its token is dropped, the prefix cache publishes a finished sequence up to its last token's position and the pass writes the one after, and its blocks are freed at its commit. With the switch, a step of one sequence that its own token can end (not `ignore_eos`) has its ids read back first (`Engine::peek_outcome`: collect_step's rules without its effects), and `step_next` issues nothing behind a step whose outcome ends every request it carries (commit's own end conditions, `Scheduler::outcome_ends_every_request`). Exact: every pass that runs ran before, and the commits are the same. The cost is the host's turnaround between two such passes, while the card waits: the next step's build, advance and the ranks' release (tens of µs a token). Steps of two or more sequences keep the step issued ahead. Arm part `Ss`; the shutdown's `post-stop passes:` line counts the passes run after a reply's last token and, with the switch, the steps read back, the wait for their ids, the turnaround and the passes withheld |
| miss reads on kept readers, 2 MiB pages | +4.6%, TTFT −3.6% | on by default; `RADIANCE_EXPERIMENTAL_DS41_READ_POOL=0` starts threads a call and `..._PLANE_HUGEPAGES=0` uses 4 KiB pages, as before (A/B switches) |
| missed units relaid straight into a reserved cache slot | +1.1% | on by default with `..._NATIVE_CACHE_DIRECT_FILL=1`; `RADIANCE_EXPERIMENTAL_DS41_INSERT_DIRECT=0` relays via scratch and copies, as before |
| `RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=1`; `..._INSERT_SLICE_KIB=N` with `..._INSERT_WAIT_US` (4000) | not yet measured (reviews: non-temporal stores alone +0.15 to +0.6%, with the pieces and waits +0.2 to +1.5%, central ~0.5-0.8%) | the background inserts of missed units into the packed RAM cache (`..._PACKED_CACHE=1`; elsewhere a warning, and they copy as before). Each of ~9.6 misses a decode token relays ~18 MB beside the pool's DRAM-bound bursts; glibc's `memcpy` streams only from 24 MiB, so its cached stores read every line of the slot first (~53 MB of traffic an insert, ~35 with streaming stores) and leave the slot in the L3 the pool reads through. `INSERT_NT` writes each weight's codes plane (all but ~0.3 MB of a unit) by streaming stores (`ds41hx::copy_nt`, an `sfence` at the end of each piece); `INSERT_SLICE_KIB=N` (implies it) does so in pieces of N KiB, and an insert from a one-token pass sleeps before each piece while the pool has submitted jobs unfinished (`HostExpertPool::work` / `wait_work`: not while it holds only the reservations of misses whose planes are still being read, ~2.8 ms a miss layer with the pool idle after its hits, which a wait on `backlog()` slept through), at most `INSERT_WAIT_US` an insert, after which it copies on; prefill's and the warm-up's inserts never wait. The job that finishes the pool's submitted work wakes those sleepers after its ticket counts down rather than before (the critical-path review's pitfall, which `INSERT_YIELD_US`'s `wait_idle` still has), so the lock and broadcast stay off the waiting rank's path. Exact: every slot byte is what one `memcpy` writes (`ds41_host_experts_test check`'s packed copy case; `native_mapped_copy_mover_test`), the scales' round trip still gates the publication, and the slot is published whole after the last piece's `sfence`. Startup line `background inserts write a missed unit's codes by non-temporal stores ...` (rank 0); shutdown line per rank `host experts' background inserts:` (units, ms a unit from the queue to published, one-token passes' apart; non-temporal pieces and MB; pieces that waited, ms asleep, inserts out of their wait; MB and ms copied while the pool had submitted jobs unfinished; pieces begun while it held only reservations). Arm parts `In`, `Is` / `IsN` (256), `IwN`; an Is arm counts only on a radiance whose startup line says `while the pool has submitted jobs unfinished` |
| `RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES=1` | not yet measured (review: +0.2 to +0.5%) | the per-tile costs the cpu-dram review kept (its helper threads dropped): a worker keeps the x lanes it laid out for its last tile and runs its next tile on them when that tile is of the same job and projection (`ds41hx::few_lanes` / `gemv_avx2_few_lanes`, `gemv_avx2_few`'s two halves: ~1.3 µs a tile at K 5120, ~0.8 at 2304), and a claim looks only at the slots holding a job (a bitmap submit sets and a job's last tile clears) instead of all 128 slot headers. Exact: the same lanes and the same kernel code, so the same bits (`ds41_host_experts_test check`; `host_expert_pool_test_lean` runs every pool case with it). Startup line `host experts: lean tiles:`; shutdown line `host expert pool, lean tiles:` (tiles claimed, slots looked at a claim, lanes laid out and reused). Arm part `Ph` |
| `..._ENGRAM_SPLIT=1` | +3.5%, TTFT −2.6% | in the deepseek41 plugin (build it from this branch; the dev tree's has no switch): each rank holds and computes 6,400 of engram_kv's 25,600 rows and an all_gather rebuilds the row; the replicated 315 MB GEMV (1.4 ms a token on every card) becomes a quarter, and the ~450 MiB a card it frees holds 25 more experts a rank |
| `..._PREFETCH_ATTN=1` | +2.8% | in the deepseek41 plugin: while the card waits for a layer's CPU rows, lane 1 reads the next layer's q_a, kv, q_b and wo_a (~56 MB a card) into the 64 MB on-die cache (libr4d `prefetch`, read-only); decode passes only, and not ahead of an Engram layer. That stays its behaviour unless one of the three modifiers below is set |
| `..._PREFETCH_MIB=N`, `..._PREFETCH_SET=k1,k2,..`, `..._PREFETCH_ENGRAM=1` | MIB=56: 17.65 / 17.29 / 17.58; MIB=56 + ENGRAM: 17.65 / 17.73 / 17.36 tok/s against 16.53 (a dip) / 17.51 / 17.24 (+1.0 to +1.4%), every reply identical; ENGRAM overlaps the Engram all_gather, so it was also checked on the 16 chat prompts with 256-token replies: 16/16 identical, 15.84 against 15.74 tok/s. Adopted (MIB=56 + ENGRAM) | modifiers of `..._PREFETCH_ATTN=1` (deepseek41 plugin, `arch/deepseek41/ds41_prefetch_plan.h`). `MIB`: a budget of N MiB an issue point, filled in use order (`ekv` engram_kv, `comp` the compressor, `qa`, `kv`, `iq` the indexer's q_b, `qb`, `oa` wo_a's groups, `ob` wo_b, `rt` the f32 router, `sgu`/`sdn` the shared expert's gate\|up and down), the read that crosses it cut to what is left: at bf16 56 MiB is the four reads above plus 2.5 MiB of wo_b; with `..._DENSE_FP8=1`, whose E4M3 codes halve the reads (the four alone are then 26.75 MiB), the whole list through the shared expert (rank 0 55.5 MiB). `SET`: the reads named, in that order (also `iw`, `ik`: the indexer's small projections; `ma`, `mf`: the connections' mixes), whole unless a budget is set. `ENGRAM`: an Engram layer (1, 14) is read too, its engram_kv by the layer before when `ekv` is listed, the rest right behind its Engram GEMM. Each rank prints its plan at startup and its counts at shutdown (`rank R: on-die reads`). Arm parts `PmN`, `Pl<k1+k2+..>`, `Pe` |
| `..._CACHE_SECOND_CHANCE=N` | N=1 +3.9%, N=2 +5.1% (TTFT −1.5% / +1.2%) | a cached unit the decode profile calls hot (`..._CACHE_PROTECT_SHARE`, default 6 of the sealed profile's 4326 selections a layer: ~1150 units a rank) moves to the front up to N times between uses before an insert may evict it. The held-out prompts share task families with the profile; replayed on the 16 chat prompts too, critical decode misses fell 7.9% (N=1) |
| `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1` | alone 17.56 / 17.60 / 17.58 against 17.37 / 17.65 / 17.69 tok/s (+0.1%), TTFT −3%, replies identical; the gemv_vp kernel is 27-51% slower a call than the plain row (register spills), being rewritten | the cards' static expert residents in the same packed form (below): 18,001,920 instead of 18,800,640 bytes a unit, so x1.0444 the residents (with F8, 712 / 720 a rank should become ~743 / 752); needs a `libr4d.so` with the row `moe_gemm_ggml_mxfp4_gemv_vp` in the run home, `--deterministic` and passes of at most 16 tokens; `..._VRAM_PACKED_RAW=<gate_up>,<down>` (512,320) sets the raw rounds of room; arm part `Vp`, which runs only on a `libr4d.so` the device check passed (`VPACK_CHECK`, below) |
| `..._PACKED_CACHE=1` | +3.2% alone, TTFT −5.5%; with `..._CACHE_SECOND_CHANCE=2` 15.52 / 15.48 tok/s (+8.2%) | the RAM cache holds each expert's codes plane plus its e8m0 scales packed to 2 bits from one base a weight (raw rounds beside, budget 512 / 320, `..._PACKED_CACHE_RAW`): x1.0444 slots and 4.4% fewer bytes a CPU hit; 7 of 15,360 experts never fit and stay misses. Refused unless the CPU takes every routed unit (all, M16, B1); the card's copy path never reads or fills it |
| `RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=N` | not yet measured (estimate +0.3-0.5% decode: the 7 never-packable experts' ~0.1 misses a token, ~2.7 ms each, gone; refused inserts down to one per distinct unit) | with the packed cache (`..._PACKED_CACHE=1`) and misses on the CPU: the planes of up to N units a rank whose inserts the packed cache refuses over its raw-round room (7 of 15,360 experts; 1-2 a rank in the held-out runs) are kept after their first miss (the insert thread, `core/place/hx_side_store.h`) and handed to their later misses as a unit read ahead whole is: no drive read, the same bytes, the container's epoch checked first; never inserted again, and the read-ahead treats them as cached. ~18 MiB of planes a unit held (2 MiB-rounded buffers); the store only ever holds units the cache refused, 1-2 a rank in the held-out runs (~36 MiB a rank), and N bounds it (N=7: every never-packable expert even if one rank had them all). Refused without the packed cache or the CPU's misses. Startup line `side store: the planes of up to N units a rank ...`; shutdown `host experts' side store: K of at most N units held (...); M misses handed their planes without a read`, and the mover's `unit inserts refused over a raw-round budget` count should equal the distinct units. Arm part `NsN` |
| `RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1` (`=2`) | neutral: 16.85 / 16.65 against 17.07 / 16.36 tok/s (47% of launches went unordered; the card's chain is not the critical path) | in decode passes (in every pass issued live), a device launch whose first dispatch reads nothing the dispatches in flight on its queue write, and writes nothing they read or write, goes without the AQL barrier bit (`core/runtime/live_chain.h`): what `haz_mark` does for recorded passes, which a staged pass never is. Exact: only start order changes. Anything on the queue the issue path did not write (events, waits, copies, table patches) orders the next launch; collectives, kernels without `rad_kernel_concurrent` and host-visible launches never go unordered. Shutdown line `live chain: ...`; arm parts `Ov` / `Ov2` |
| `RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE=N` | instrument | one trunk decode pass in N on rank 0: every launch (layer, op, kernel, queue and packets, `would` / `why` / `clash`, its dependences on earlier launches, the buffers it reads and writes) into `<prefix>.ops.csv`, every kernel's start and end on card 0 into `<prefix>.stamps.csv` (`..._CHAIN_TRACE_OUT`, default `/tmp/radiance-chain-<pid>`); `chain_report.py` joins them. Without `..._CHAIN_OVERLAP` it changes nothing (the dump-only arm). Arm part `Ct` / `CtN` (N 16) |
| `RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1` | +1.9% over five interleaved pairs, each won (17.40 / 16.95 / 17.22 / 17.39 / 17.59 against 17.07 / 16.36 / 17.07 / 17.34 / 17.11 tok/s), replies identical | a routed layer's card ops (gate_up, swiglu, down, the shared expert, the ffn mix, the attention prefetch) are queued before the issuing thread waits for the routing counts: with the file buffers capped every file-tier unit is the CPU's, so the card's aliases do not depend on the routing. The counts are waited for, the CPU's share chosen and submitted and a decode pass's misses read at the layer's gather, which then waits for the CPU rows as before. The card no longer idles from its counts copy through the host's wake-up, choice and miss reads. Refused unless the CPU takes every routed unit (all, M16, B1); arm part `Ei`; shutdown line `host experts, early issue` |
| `RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1` | neutral: 16.36 / 17.19 against 17.07 / 16.36 tok/s | the counts row and the pool's token rows, sorted order and offsets reach the host in one ranged copy instead of four dispatches (each behind the last, each with a system-scope release), from a range list kept on the card per routed layer and pass size and rewritten in stream order when an address changes; needs host experts; arm part `Fl`; shutdown line `host experts, fetch lists` |
| `RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=N` (2-4, designed for 3) | neutral (17.19 / 17.61 / 17.20 against 16.53 / 17.51 / 17.24 tok/s): N=3 cuts a lone miss's mirror leg 2.59 → 2.43 ms (copy 0.217 → 0.102 ms), but the NVMe leg (~2.75 ms) lands last in 95% of calls either way, so the miss still takes ~2.8 ms | a decode pass that misses one unit reads the plane mirror's decode tail (2.4 MB at 0.13) in N pieces on N readers, so the SATA stripe's preads and the copies out of the bounce buffers overlap; the same bytes (pieces start on the container's 4 KiB grid, so a failed one falls back to the container as a whole tail does). Needs the decode mirror (`R`, `RdNN`); arm part `MsN`. The read-ahead in pieces (`PpK`) reads the container only, so its pieces are not split; the whole-unit read-ahead (`PoK`/`PtK`) reads a unit as a lone decode miss and is split too. Gate: the shutdown report's "one-token miss reads, one unit" line; drop it if the mirror lands last in under 25% of calls and its copy takes under 0.15 ms |
| `RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1` | neutral (17.48 / 17.15 / 17.27 against 16.53 / 17.51 / 17.24 tok/s; with Ms3 17.53 / 17.35 / 17.61): every one-unit miss posted at the choice, no change in the caller's wait | a one-token pass's missed planes start on the kept readers as soon as `hx_choose` knows them, instead of after `hx_submit` (its pool wake, 28-32 µs a layer) and the copy path's staging; the issuing thread then waits in `hx_submit_misses_now` (`NativeReadWorkers::post`/`join`, which hold the readers' call mutex from post to join as `run` does). The same reads: the set `hx_submit_misses_now` reads, so with the read-ahead in pieces (`PpK`) never a unit read ahead whole nor one the piece readers are finishing (whose slot `hx_ra_settle` may give fresh planes); the post is a demand read for the piece readers from the post on, and is finished before the urgent units are settled. Also at an early-issued layer's gather (`Ei`). A post that finds the readers on another thread's call (the whole-unit read-ahead's, `PoK`/`PtK`, or the async misses' worker, `A`) does not wait: it posts nothing and the layer reads as before (the mover's "kept plane readers" line counts them). Arm part `Mp`; the "readers" line shows the readers' start and the caller's return, and the EP decision's own `posting the misses' reads` line its time (not in "choose") |
| `RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1` | not yet measured (estimate +0.3 to +1.5%: the first CPU rows ~10-35 us sooner a burst. The submit itself need not fall: in the review's bench at 40 workers it rose, 44 → 77-87 us, as the woken workers start at once and compete with the waker through its `FUTEX_WAKE` loop, and without `Mp` a miss layer's reads start after it. Judge Fw on the pool's first- and 24th-tile times and the gather waits, the miss layers' among them, not on the submit) | the CPU pool's idle workers sleep on a futex word (the 32-bit work generation) instead of `idle_mu_` and a condition variable, and a submit wakes them with one `FUTEX_WAKE` and no lock (`core/place/host_expert_pool.h`). Through the condition variable the waker holds `idle_mu_` across the broadcast (23-27 us of the 28-36 us decode decision), every woken worker takes the mutex again in turn, and the other ranks' submits queue on it meanwhile. The C2 exit (30 us) and the kernel's wake loop remain, so judge it on the pool's "first tile" counters, not the submit alone. Exact: only when workers wake changes. Arm part `Fw`; startup line `idle workers sleep on a futex`, shutdown line `host expert pool:` (`futex wake`) |
| `RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=N` | not yet measured (estimate +1 to +3% with `FUTEX`, ~+1.5% central; the awake cost ~24 workers x ~50-100 us a layer, against MWAITX's 40 x 1.5 ms) | up to N sleeping workers (about 24) woken shortly before a one-token layer's CPU work, then back to sleep after `_PREWAKE_SPIN_US` (100) if none came: (1) every rank records a card-local, host-waited event right before each routed layer's router GEMV (the gemm reading `blk.L.glm_router.weight`, found at startup), and at the layer's decision its issuing thread waits for that event before the counts, then calls the pre-wake (the pool takes the first rank's call of the layer: one call a window); (2) in a miss read the issuing thread, done with its own share (the mirror's tail), waits for the kept readers with a deadline `_PREWAKE_LEAD_US` (60; 0 turns this trigger off) before the read's predicted landing (the median of the last 32 reads of as many units, from their dispatch; armed from the eighth), and pre-wakes there if the reads are still out. Not while read-ahead units are being finished (their wait follows the read). Exact: the same jobs, rows and bytes. Spins of 2.5 ms (-8.5%) and MWAITX (-7.2%) lost more than the wake was worth by keeping all 40 workers runnable, so N is a subset, the window short and the shutdown line counts what it cost: `host expert pool:` (router / miss pre-wakes: calls, workers woken, those that took a tile and those that slept again, their spin; the first tile and the 24th of a submit's first job; idle spin in all) and per rank `host experts, pre-wake:` (router event to counts, the calls' time, miss-read pre-wakes fired and the lead they gave); every host-experts arm also prints per rank `host experts, one-token passes:` (decode gathers' waits for the CPU rows, layers with and without a miss apart: the number the switch should cut by 20 us or more, with the router waits flat). Arm part `WpN` (1 .. the arm's threads), with `WlN` the miss-read trigger's lead (`_PREWAKE_LEAD_US`, 60 in every other arm; `Wl0` the router trigger alone); startup lines `pre-wake of up to N workers ... at its router and 60 us before a miss read's predicted end` (`Wl0`: `... and not in miss reads`) and `pre-wake: an event before K of 40 routed layers' router GEMVs`. The per-rank line also counts the router calls whose layer had its counts in when the call returned (the call held the decision back); the miss trigger's 1 us timer slack covers that wait alone |
| `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1` | alone 17.84 / 17.92 / 17.26 against 17.37 / 17.65 / 17.69 tok/s (+0.6%, noisy), replies identical, card checks passed; with Vp and Ss +2.1% every round | in the deepseek41 plugin and libr4d (build both from this branch): the 40 routers ([384, 5120], 300 MiB a card replicated) and the kv\|gate projections of the 3 compressors with a ratio above 1 ([1024, 5120], layers 2, 8, 14) are f32 in `gemm_nt_f32` but ship bf16. The container holds the routers widened to f32 (`scan_low_halves.py`: 0 of 78.6M low halves nonzero), which libr4d's `gemm_nt_f32bf16` stores as their bf16 halves, and the compressors as bf16, which the loader had widened and which are now read as they are: 180 MiB a card back to the experts (~+10 a rank; 150 MiB of it the routers'). The kernel widens each by a shift in the same tiles, lanes and order, so every logit is bit-identical. The load refuses an f32 weight with a nonzero low half rather than rounding it; startup prints a `ROUTER_BF16:` line on rank 0 counting the weights held at two bytes an element. The on-die reads (`PmN`, `Pl..`) take both at their bf16 bytes (`rt` 3.75 MiB, `comp` 10), and the router predictor (`PpK` and the others) reads the router's f32 plane from the container as before. Gate: `r4d_selftest --perff32 384 5120 1` at or under 14 us for the halves (19.7 us f32 in the engine); the same run byte-compares and times tile forms 1-8, and `..._ROUTER_BF16_FORM=N` makes the halves take form N at passes of up to 16 rows without a rebuild (forms are byte-identical). Arm parts `Rb`, `RbN`: `ab_window.sh` measures them only when `SELFTEST_F32BF16=<r4d_selftest>`'s card checks (the check case, then every form byte-compared and timed at 1 and 16 rows at both shapes) passed in the same window on the arm's own libr4d.so |
| `RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=<prefix>` | instrument, exact; overhead not yet measured (rank 0: three more copies, ~1.6 KB, a routed layer behind its counts row's, which the CPU's choice does not wait for; every rank: one cache lock a decode layer) | the routing trace for the lossy track (`core/place/route_trace.h` has the format; below). Arm part `Rt` (files under `rtrace-<arm>/`); shutdown lines `route trace: ...` per rank (picks by status) and for the files (records, bytes, drops) |
| `RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1` | TTFT 99.2 → 69.1 s at 3,086 prompt tokens (−30%), 68.0 → 50.3 s at 1,890 (−26%), 31.1 → 27.6 s at 933, 224 → 125 s at 6,817 (−44%); every prefill's KV-group and last-row-logit hashes identical to the base (5/5 and 1/1, all cut), replies identical, decode unchanged (window ab-20261029T200958Z). Adopted | the exact consumer-suffix skip (`core/sched/consumer_skip.h`, schedule and proof; the deepseek41 plugin's `cut_declare` / `step()`). Layers 21..39 see earlier rows only through layer 20's compressed and index keys and their own 128-row windows, so a prompt's early pure prefill passes run layers 0..F-1 in full, layer F's window-key ops alone (the full layer's own ops, so the stored bytes are the same; layer 20 also its compressor and key stores), and skip the rest, F ≥ 20 rising by 8 passes a layer to 40 at the prompt's last pass. Every computed row and stored key is the base engine's, and no row reads a key the request did not write; the next step's windows and the tail checkpoint's are complete (with the prefix cache on, a prompt is cut for the deepest checkpoint its idle prefill writes -- its tail, else the last interval boundary, else a shared head -- and the snapshots below it are made whole or not written: `..._CONSUMER_SKIP_CKPT`, next row; none is counted as dropped without the cache). Refused also when the ranks' declarations differ or KV streams (`--kv-resident`); a step that writes a checkpoint is never cut. Nothing is read, staged, predicted or computed for a skipped layer (file stager, read-ahead, CPU path, next-layer prefetch). Refused, with the reason, for speculation, KL or embedding modes, media, R-SWA, concurrent sequences with chunks wider than the 16-token quantum (unless `..._SCHED_PREFILL_ALONE=1`), and a model the derivation does not cover. Startup lines `consumer skip: on` (scheduler) and `consumer suffix declared` (plugin); shutdown lines `consumer skip: ... layer-passes run in full, ... key ops alone, ... skipped` (engine) and `rank R: consumer skip: ...` (plugin, the same counts). Needs a radiance and a deepseek41 from `claude/ds41-cskip` (run home `home-cskip`); arm part `Cs` |
| `RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT=512` (default with the prefix cache when unset), `=all`, `=N`, `=0` | not yet measured on the cards (claude/ds41-csckpt). Predicted from the schedule, layer-passes run in full against the base engine's, idle 16-token prefill, exact resume (nearest is the same but where its fewer snapshots move the target): at `--checkpoint-interval` 16 and 256 `all` keeps 97.5% and 96.3% at every length (the cut saves 2.5-4%), at 2048 `all` 82.3% at 3,086 tokens, 78.9-78.8% from 4,096 to 32,768; `=512` 77.6-77.8% at 3,086, 60.3-60.4% at 8,192, 52.6% at 32,768 at any interval; `=0` (the cut alone) 70.2%, 57.5%, 51.9%; 1,221 tokens 87.3% alone, 91.2-90.2% at `=256`, 96.4-97.6% `all` (table below) | which snapshots below a cut prompt's target are written, each holding the base engine's bytes (`core/sched/consumer_skip.h`, "complete checkpoints", with the proof). Without them a later prompt that shares only an earlier prefix of a cut one finds no snapshot (window ab-20261029T232708Z, `Cs-Pc-Pci256`: chat turns A2-A4 31 s instead of 5.2-5.8 s, `pfxL-3` 56 s instead of 3 s, `hit_cuts no_snapshot`). Key ops alone do not make a snapshot whole: a layer's window key at a row is computed from that layer's input there, layer L-1's output, so layer L-1 must run in full on that row, reading its own window W-1 rows further back, and so on down. A whole snapshot at P costs its own dependency cone: layer L in 21..38 in full on the 128 (39-L) rows before P, layer 39's keys on the last 128, layer 20 in full on the 2,432 before it; 1,520 16-row layer-passes for a lone one far from the target (20% of a 3,086-token prompt), about 17 a pass of distance near the target, none within a pass of it. A pass runs F = max(the target's F, the first whole checkpoint ahead's), still one number a pass, so the plugin is unchanged; checkpoints every I tokens overlap their cones for every layer L with 128 (39-L) >= I, which then runs in full throughout. `all`: every snapshot the uncut prefill writes (the base engine's prefix-cache contents, hit for hit); `N`: those within N tokens below the target and the request's shared head (keep point), the rest not written (unset or not a token count: `512`, the default, which keeps most of the skip and every hit of both receipts); `0`: none, as before. A step that writes a snapshot below the target is cut only if it is the one planned and stores layer 39's keys (else it runs every layer: `run in full off the plan`, stays 0). Startup: `consumer skip: on ...` names the policy, and `consumer skip: complete checkpoints (..._CKPT=all|N) -- ...` (with the cost of a lone one at the deployed shape) unless 0; shutdown: `checkpoints below a prompt's target: W written whole, D not written; their cones raised P passes by X layer-passes in full (the key ops alone stay one a cut pass), K snapshot passes cut at the last layer's keys, 0 run in full off the plan`. Arm parts: none (= `512`, the default) / `Ck` (= `all`), `CkN`, `Ck0`, with `Cs` and `Pc`; a Cs Pc arm must log its policy line or it is not measured. Tests: `consumer_skip_test` (any set of snapshots on random grids, the scheduler's sets at intervals 16, 256, 2048 in both modes, negative controls: the target alone, key ops alone, a cone a pass short or for narrower passes), `sched_test` (the real scheduler: every snapshot of the uncut run written and whole, per-pass F, bands, a shared head, a resumed prefill, the variable), `ds41_consumer_skip_arch_test` (the plugin on a non-monotone F) |
| `RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1` | instrument | the exactness evidence for the consumer skip, usable without it (`core/engine_hash.cpp`): at the end of each prompt's prefill, before its sampler and before the next step, every rank hashes each KV group the prompt wrote (whole blocks of a full group, the last window of the windowed one, the state slots), each layer apart, and the last row's logits; lines `prefill hash: seq S, N tokens (...), rank R: <group>[layers]=<hash> ... logits=<hash>`, the window group by layer, and `prefill hash: seq S, N tokens: all ranks <digest>`. Steps are not issued ahead while it is on, so time prompts in arms without it. Arm part `Kh`; the window compares the Kh arms' hashes with the first Kh arm's (`ab_requests.py --hashes DIR ARM...`) |

The shutdown report counts the miss reads of one-token passes whatever the switches (two lines a
rank, one for calls of one missed unit and one for more): how many calls had a mirror leg and in how
many the mirror landed last (and by how much), the legs' mean times from the dispatch and the
mirror's copy out of its bounce buffer (the longest piece's and all of them), when the first read,
the first container read and the last read began, and how long after the last read landed the
issuing thread was back (a `done_` wake when a reader, not the caller, finished last). The plane
mirror's line gives its copy time too, and the mover's "kept plane readers" line how many calls were
posted.

## DSpark's draft GEMVs (`RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV`)

DSpark's drafter (`Ds` arms) has three MoE stages (layers 40-42) whose 128 experts are 32 a rank,
top-3, read in place from pinned host memory (`--draft-bank-expert-placement host`). A query pass
is 5 rows, 15 slots a stage, about 4 of them a rank's own. The decode GEMV (libr4d
`r4d_ds41_mxfp4.hip`) did that work twice over:

- `mx_range` gives the rank's last expert every slot routed to another rank (`[off[31], T)`); the
  rows come out zero, but its weights are read for them, at gate_up once a slot.
- Since 02439b8f an expert with more rows than the LDS holds whole (one gate_up row at K 5120,
  three down rows at K 2304) takes them in passes of that many, each a read of its weights: right
  for VRAM, where a re-read costs less than restaging the activations, wrong over the link.

Measured in the cap-1 profile (ab-20261029T154954Z; the per-op tables of all four ranks): draft
gate_up 6.1-6.7 ms a call on ranks 0 and 1 and 11.2-11.8 ms on ranks 2 and 3 (whose cards share
root complex 00), down 0.5-1.2 ms; 22 ms of draft GEMVs a step on ranks 0 and 1, 38 ms on 2 and 3,
which the first two wait for in the stages' all-reduces (5.0-6.5 ms each). The draft costs about
45 ms a step in all (Ds0-o 12.53, Ds1-o 10.81 tok/s).

`=1` takes the live tail for the draft's shape (only the trunk's 96 / top-6 had it, under
`..._LIVE_TAIL`; the dropped rows are zeroed by the tail kernel instead) and gives each live expert
one pass over its weights, K restaged in chunks a row group where its rows do not fit whole (the
chunked path, which keeps each lane's blocks and their order). `=2` is the live tail alone. Exact:
each output is its weight row dotted with its activation row in the same lane and block order, and
the dropped rows are zero either way. The trunk's GEMMs are not touched. libr4d prints a line at
the first draft call and its counts at exit (`libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV ...:
N draft moe_gemm_ggml calls (gate_up, down; query rows; tiled), M draft-shaped calls left on the old
path`). Arm parts `Dk` / `Dkt` (with a `Ds` part). `tests/r4d_emu/test_ds41mx.cpp` holds every mode
to a host restatement of the decode sequence with the work counted; on a card,
`ds41_mxfp4_device_check LIBR4D.so draft [trials]` holds them to the host experts with the bank's
experts in pinned host memory and in VRAM, then times each mode at a 5-row query pass
(`DEVICE_CHECKS=draft` in a window, with `DRAFT_CHECK_BIN` and `DRAFT_CHECK_LIB`). Not yet measured
on the card.

The replies cannot show a wrong draft GEMV (the target verifies every draft), so `ab_window.sh`
holds a `Dk`/`Dkt` arm to what its name says: it runs only on the `libr4d.so` (sha256) whose draft
check passed in the same window; it is measured only if libr4d's first-call line is in its log and
the exit line leaves 0 draft-shaped calls on the old path (otherwise its results are kept as
`<arm>.unmeasured.json`); every other arm runs with `..._DRAFT_GEMV=0` and
`..._DSPARK_SHORT_BLOCK=0` and must show neither line; a `DbN` arm needs the engine's "runs its short
block of N row(s)" startup line, and no other arm may have it. At the window's end each `Dk` arm's
per-request `draft_n` and `draft_n_accepted` (and texts) must equal those of the same arm without
`Dk` (`ab_requests.py --draft-stats`): the exact drafter proposes the same tokens. On a table the
packed VRAM residents' `gemv_vp` row serves (`Vp`), the draft mode does not apply (its kernel has no
one-pass form) and such calls are counted as left on the old path.

`RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=N` (2-4; the deepseek41 plugin) declares a short query
block of N rows, [anchor, noise x (N - 1)], proposing N tokens (`rad_drafter_short_block`), which the
core runs from `--draft-short-from` decoding sequences on; the verify cap (`DsK`) now also accepts
`--draft-short-from 1` when the drafter has one. At N = K + 1 the three draft stages route 3 (K + 1)
slots instead of 15. The block's attention is non-causal, so its proposals are not the full
block's first ones: the replies stay the target's, acceptance may move (the drafter was trained at
5, and a short block has never been measured). Its own ops where the block length is a parameter:
the offsets placing every row at the block's last position, each stage's attention window (`window`
context keys before the anchor plus the block's), and the walk. Arm part `DbN` (with a `Ds` part);
off, nothing of it is declared. Run in ab-20261030T070039Z with `Dk` on the best settings' DSpark
form (`...-Pm56-Pe-Rm7-Ds1-Dk-Db2`): every reply identical; acceptance 216/268 against 217/268 for
the full block on the held-out prompts (384/548 on chat); a drafting step 108.5 ms against 113.9
(63.7 drafting off), so the draft costs ~0.2 of a plain step instead of ~0.3.

## DSpark's acceptance gate (`RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE`)

A drafting step pays when 1 + a > rho + d (a: drafts accepted; rho + d: the drafting step's cost
in plain steps). On the best settings' DSpark form rho + d is 1.78 at cap 1 with the full block
(ab-20261029T220909Z: 110.5 against 62.1 ms a step held-out, 126.4 against 71.4 chat) and 1.70
with `Db2` (ab-20261030T070039Z), while acceptance per reply runs from 0.27 to 1.00. The gate
(`core/sched/draft_gate.h`, decided in `Scheduler::block_begin`) keeps each sequence's last W
verifies of head drafts; while the mean drafts accepted a verify is under the threshold the
sequence drafts nothing: its query pass is skipped, the step issued ahead verifies no drafts for it
(nor does the prompt-lookup fallback stand in), and the context pass, which fills the drafter's
window cache, still runs. One step in R of a closed run drafts anyway, a probe whose verify enters
the mean. A sequence drafts from its start until W verifies are in. The mean lags one verify (the
open step's acceptance is still on the card), and nothing waits on the card.

`=1` takes the defaults, threshold 0.78 (the full block's break-even at cap 1), W 8, R 8;
`=T,W,R` sets them (T in drafts accepted a verify: at a deeper cap use that cap's rho + d - 1;
`Db2` breaks even at 0.70). Refused at startup when unreadable, with compute OFF (`Ds0`), and
without a block drafter. Startup line `DSpark acceptance gate: on (...)`; at shutdown a line with
every sequence's counts (steps asked, drafted while warming up / open / as probes, gated, closed
runs and reopenings, drafts verified and accepted, the probes'), a line of the decode steps of one
sequence by kind (verify + draft, plain, a probe's draft, its verify) with their mean wall time and
the measured break-even (verify + draft over plain, less one) to set T from, and a line for each of
the last 256 sequences. `/stats` carries the same under `draft_gate`. Only how many drafts a step
verifies changes, so replies stay identical. Arm part `Dg` or `DgTT[wW][rR]` (threshold 0.TT, or
T.TT for three digits; W and R default to 8), with a `Ds` part other than `Ds0`.

Replay of the gate on the measured per-reply acceptance (i.i.d. outcomes within a reply,
`gate_sim2.py`-style, against drafting off with the bank): with `Db2` always-on gives +5.4%
held-out and -0.4% chat; the gate at 0.70/8/8 +5.1% and +2.0% (+5.2% and +2.7% on 256-token
replies), the per-reply oracle +6.2% and +5.0%. With the full block (rho + d 1.78) the gate at the
defaults +1.6% held-out and -0.3% chat against +1.0% and -2.8% always on.

## The drafter's dense linears in FP8 (`RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8`)

The release ships the DSpark stages' dense linears as the trunk's: `mtp.{0,1,2}.attn.{wq_a, wq_b,
wkv, wo_a, wo_b}`, `ffn.shared_experts.{w1, w2, w3}` and `mtp.0.main_proj` are F8_E4M3 with
F8_E8M0 scales one per 32x32 block (model-00044..46's headers), and the draft bank holds them cast
to bf16 (`blk.40-42.*`, `dspark.fc.weight`). With `=1` (and `..._DENSE_FP8=1`, which it extends)
the deepseek41 plugin marks them as lw8() marks the trunk's and gives the drafter's shared expert
`gemm_nt_bf16f8`; the load re-encodes each and refuses one that is not exactly so
(`ds41_dense_fp8_scan` now covers a bank's `dspark.fc.weight` too; over the bank of
ds41-dspark-bank-20261028T140400Z: 43 / 43 entries exact, 551,424 blocks, none refused, every
tensor-parallel cut on 32). The op writes gemm_nt's bytes, so
the drafter proposes the same tokens; the stored form halves those weights, 25 a rank (fc and, per
stage, q_a, q_b, kv, wo_b, two wo_a groups and the shared pair): 438 MiB of bf16 become 219 MiB of
codes on rank 0 and 415.5 become 207.8 on ranks 1-3 (`ds41_dspark_fp8_arch_test`), about 11 more
resident experts a rank, and the draft's and the context pass's reads of them halve. Startup line per rank `deepseek41: rank R: DSpark
drafter: N dense weights read by gemm_nt_bf16f8 (...)`, apart from the trunk's. Arm part `Df` (with
`F8` and a `Ds` part); at the window's end each `Df` arm's `draft_n`/`draft_n_accepted` must equal
the same arm's without `Df`.

## The draft bank's hot set in VRAM (`RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM`)

The pinned bank costs the best arm ~4.5% before it drafts anything: `--draft-bank-expert-placement
host` takes 1.68 GiB a rank of pinned pool (p2048 instead of p256) and the RAM cache drops to 90 GiB.
`RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=N` (1-32) with `..._DRAFT_HOTSET=<file>` replaces that
placement (the flag stays `host`, the bank's manual contract; core/place/draft_vram.h):

- each rank keeps the first N experts of each of its three stage lists in the hot-set file as
  static VRAM residents, charged to the weights pool before the expert fill (`Planner::
  place_draft_bank_vram`): the trunk gives up 3N residents a rank (a pair is a trunk expert's size);
- the other draft experts are parked: mapped tier with no class, slot or pool, never read; each is
  published at its stage's first resident's weight of the same slot on its card, so every pointer the
  grouped GEMM is handed names card memory (`draft_vram_aliases`);
- nothing of the bank is pinned, so the arm runs at B's p256 and m98304;
- after the load every rank's three router correction planes (the bank ships them zero) are read
  back, written with -inf at every expert no rank keeps, and read back again (`draft_vram_mask_planes`).
  `router_scored_topk` never takes a -inf key, so the router takes the best three kept experts,
  weighted by their unbiased scores renormalised over the three. A stage keeping fewer than three in
  all is refused (with two kept, libref refuses the launch and libr4d's third pick is a masked
  expert's id; `r4d_emu_ds41mask`).

The target verifies every draft: the mask moves acceptance and speed, never a reply byte. With
`DRAFT_GEMV=2` (`Dkt`) the live tail serves the VRAM weights and a parked expert is never read; even
without it the aliases keep every read inside card memory. Every rank tallies its draft routing
histograms (the ones the core already copies back) and prints at exit the selections on its resident
experts and on its parked ones, which must be none (`draft VRAM: rank R at exit: ... (none: the mask
held)`).

The hot set comes from a capture: `--expert-stats` in an unmasked DSpark arm (part `Xd`, with a `Ds`
part) counts each stage's selections by model-wide expert from the router's own histograms, every
query pass's five rows. `draft_hotset.py write OUT.json xstats-*.json [--top N]` sums captures and
ranks each rank's 32 experts of each stage by count (rank r holds 32r..32r+31; the engine checks each
listed id against the container's directory and refuses one the rank does not hold); `show`, `cover`
(a capture's selections the first N keep: held-out coverage when the capture is another prompt set's),
`summary` and `check` (a capture taken under the mask selects only the kept experts). On the 2-prompt
capture of 28 Oct (37 query passes) N = 8 keeps 88.6% of the selections in sample (stage 40 78.9%, 41
88.8%, 42 98.2%), N = 16 98.9%.

Arm part `DvN` (with a `Ds` part). Its hot set is the first set of `DRAFT_HOTSET_<set>_db<K>`,
`DRAFT_HOTSET_<set>`, `DRAFT_HOTSET_db<K>`, `DRAFT_HOTSET`, where `<set>` is its `U` part's name
(`heldout` without one) and `K` its `DbK` part's: so each prompt set can be scored on the other's
capture, and a short-block arm on a capture taken with the short block. A `Dv` arm is measured only if its log has the
switch's line, all four ranks' "draft VRAM final placement ... hot=N/N/N" and masked-plane lines, and
at exit all four "(none: the mask held)"; every other arm runs with `..._DRAFT_VRAM=0` and must have
no "draft VRAM" line. At the window's end `ab_requests.py --acceptance` prints each DSpark arm's drafts,
acceptance and its ratio r to the first arm with the same `Ds` and `Db` parts in its prompt set (put the
pinned-bank arm first), tokens a step and decode tokens/s.

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
weight straight into a reserved slot: its codes plane copied, its scales packed; with
`..._INSERT_NT` / `..._INSERT_SLICE_KIB` the codes plane by streaming stores, in pieces, the same
bytes). Before a slot is
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
  prompts, `chat_turns` conversations in the same framing, or `messages`) and compares arms (each
  request by name against the first arm that sent it; only arms with no request in common, such as
  a U part's receipt beside the held-out prompts, are compared apart); `--ttft DIR ARM...` prints each
  request's time to first token and cached prompt tokens (see Prefix caching); `--lp`, `--force`,
  `--kl`, `--kl-dir`: the quality harness above (`ab_requests_test.py` runs it against a fake
  server with a toy model: an exact model forced along its own reference is bit-identical, a
  perturbed one gives KL > 0).
- `rtrace.py report|dump|synth`: reads the routing trace (above); `rtrace_test.py` checks it on a
  synthetic trace with known answers, and on the C++ writer's own file when
  `RTRACE_CPP_PREFIX=<dir>/rt` (made by `RTRACE_DUMP=<dir> route_trace_test`).
- `make_prefix_receipt.py OUT MPL-2.0 core/mem/prefix.cpp` and `make_prefix_receipt.py --long OUT
  long-receipt.json`: the prefix-cache receipts in `/home/as/ds41-claude-build/work/extra-prompts/`
  (`pfxchat-receipt.json`, part `Upfxchat`; `pfxlong-receipt.json`, part `Upfxlong`).
- `cache_sim.py`: replays an access trace against LRU, SLRU, 2Q, aged LFU and Belady.
- `router_score_eval MODEL.rad [RECORD] [--synth N] [--k K] [--max N] [--time N]` (`make router-eval
  O=<dir>`): the read-ahead predictor's int8 and int4 scoring (`..._PREFETCH_SCORE`) against its bf16
  scoring, offline, on the container's own routers (read as `scan_low_halves.py` reads them, ~320 MB:
  outside a GPU window, under `ionice -c3`): for the router inputs a `Pv` arm recorded
  (`pred-<arm>.bin`, `..._PREFETCH_RECORD`), or `--synth N` inputs a layer, how often each rescore
  count (4-32) gives bf16's set and order, and where bf16's picks rank among the quantized keys;
  `--time N`: one thread's time a prediction for each scoring with the tables cold.
- `chain_report.py PREFIX`: reads a `Ct` arm's `PREFIX.ops.csv` and `PREFIX.stamps.csv`: per layer
  kind the dispatches before the routing-counts sync, how many went (or would go) unordered and why
  the rest waited, the card's busy time and gaps, the critical chain, and the fusion candidates
  (dependent pairs whose second launch is short and reads only the first's output).
- `ds41_mxfp4_device_check.hip` (`make mxfp4-check O=<dir>`): the kernels against the card (chain
  check, exhaustive SwiGLU, GEMV timing against two libr4d builds). `tall [T...]` checks
  `RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL`: the twin row in the plugin's table and its bound, its
  gate_up and down at 17..96 tokens against the gemv over the same tokens in passes of 16 and the
  host kernels, and the sparse attention and index score at 17..100 rows against the same rows
  alone and 16 at a time. `sweep [trials]` (with the switch) and `sweep0` (without) launch every
  other op whose launcher reads the pass height (rmsnorm at 5120 / 1280 / 512 / 128, the fused
  collapse, post-collapse and norm-rope-roundtrip norms, mhc_mix split and fused, the skinny bf16
  GEMM, the lm_head) at heights 1..48 and hold each row to the same row launched alone and in
  launches of 16: up to the reach (the twin's bound with the switch, 32 without) every row must keep
  its bytes; above it only a norm may differ, and the count is printed. Exit 0 when nothing failed.
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

Run `Upfxlong` at `Lm8192` (`Upfxchat` at `Lm4096`). At `Lm4096` the KV pool holds 4,284 tokens, and
the trim that keeps a quarter of it free (counting the closing request's own blocks) evicts exactly
L1's short last chunk and decode blocks at L3's close: L4 recomputes them and L5 adopts L4's full
chunks, so the arms with and without `Pcx` test the same thing. At `Lm8192` (~8,568 tokens) nothing
in the receipt is trimmed (`/stats` `prefix_evictions` 0) and the values above hold; the auto
snapshot slots are 12 at `Pci16` and 8 at `Pci256`. `PciN`, `PcsN` and `Pcx` need `Pc` or `Pc0` (an
arm with one and neither is skipped), and every arm without `Pc` / `Pcx` runs with
`..._CACHE_HEADROOM_BACKED=0` / `..._PREFIX_EXACT_PUBLISH=0`.

Every reply of a `Pc` arm must equal the same arm's without `Pc` on the same receipt (put that arm
first): `ab_requests.py --compare` and `--ttft` (texts `same`; `pdif` marks a prompt that differs
because an earlier reply differed and was sent back, the earlier `DIFF` is the one to read). The
`--ttft` table also lists repeats within an arm (F2 = F1, A5 = A1, L2 = L1: want `same`), replies
sent back with their own token count against the generated count (equal means the next turn's
chain runs through the decode steps' blocks), and the `/stats` prefix counters.

### With the consumer skip (`Cs`)

Cut, a prompt's layers 21..39 never run on most of its early rows, so a snapshot below its cut
target holds their windows incomplete, and until claude/ds41-csckpt none was written: in window
ab-20261029T232708Z `Cs-Pc-Pci256` took A2-A4 at 31 s instead of 5.2-5.8 s and L3 at 56 s instead of
3.2 s (`hit_cuts no_snapshot`). `RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT` (the table above)
now makes them whole, byte for byte the base engine's (`core/sched/consumer_skip.h`, "complete
checkpoints"). That is not a matter of key ops: layer L's key at a row is computed from layer L's
input there, which is layer L-1's output, so a snapshot at P costs its own dependency cone (layer
L in 21..38 in full on the 128 (39-L) rows before it, layer 39's keys on the last 128, layer 20 in
full on 2,432), 1,520 16-row layer-passes alone far from the target, about 17 a pass of distance
near it. Snapshots every I tokens share their cones for each layer with 128 (39-L) >= I, which
then runs in full throughout, so `all` takes back most of the cut at small intervals; the default
(unset) is therefore a band of 512, which keeps most of the cut and every hit of both receipts. Layer-passes
in full against the base engine's, idle 16-token prefill, exact resume (`consumer_skip_test`,
`CSCKPT_TABLE=1`, prints every length and both modes):

| Interval | Prompt | Base layer-passes | `=0` (cut alone) | `=256` | `=512` | `=1024` | `all` |
|---|---|---|---|---|---|---|---|
| 16 | 1,221 | 3,080 | 87.3% | 91.2% | 94.3% | 97.4% | 97.6% |
| 16 | 3,086 | 7,720 | 70.2% | 73.9% | 77.8% | 85.1% | 97.5% |
| 16 | 8,192 | 20,480 | 57.5% | 58.9% | 60.4% | 63.4% | 97.5% |
| 16 | 32,768 | 81,920 | 51.9% | 52.2% | 52.6% | 53.3% | 97.5% |
| 256 | 1,221 | 3,080 | 87.3% | 90.2% | 93.3% | 96.4% | 96.4% |
| 256 | 3,086 | 7,720 | 70.2% | 73.8% | 77.6% | 84.7% | 96.3% |
| 256 | 8,192 | 20,480 | 57.5% | 58.9% | 60.3% | 63.2% | 96.3% |
| 256 | 32,768 | 81,920 | 51.9% | 52.2% | 52.6% | 53.3% | 96.3% |
| 2048 | 3,086 | 7,720 | 70.2% | 70.2% | 70.2% | 82.3% | 82.3% |
| 2048 | 8,192 | 20,480 | 57.5% | 57.5% | 57.5% | 57.5% | 78.8% |
| 2048 | 32,768 | 81,920 | 51.9% | 51.9% | 51.9% | 51.9% | 78.8% |

The receipts' hits (the tables above) under each policy, from the snapshots each request writes:
`all` keeps every one; `=512` keeps every one at `Pci16` and `Pci256` (the deepest a hit uses below a
target is L3's 2,560 under L1's 3,072, A2-A4's 1,024 under A1's 1,216 at `Pci256`, 1,184 at
`Pci16`); `=256` loses L3 (2,560 is 512 below 3,072); `=32` keeps the `pfxchat` hits at `Pci16`,
loses A2-A4 and D1 at `Pci256` and L3 at either; `=0` loses A2-A4, D1 (768/832, it takes B2's 752)
and L3, as the window showed. Turns that extend the previous prompt (B, C, L4, L5) hit at its target under any policy.
Run `Cs` arms against the same arms without `Cs` (and `Ck0`/`CkN` beside them): replies `same`,
`cache_n` as above, and TTFT; `Kh` on the cold arms holds the cut prefill's state to the base's.

## Correctness

Each row the CPU computes uses the card's arithmetic: the same lane and block order, butterfly,
BF16 rounding and no fused multiply-add. A job of several token rows decodes each round of codes
once for all of them (`gemv_avx2_rows`). Each row's sums still see the same operations in the same
order, so the bits do not change, and an extra row costs 0.37–0.46 of a row instead of 0.52–0.62
(40 threads, experts streamed from DRAM). `POLICY=verify` compared 31,670 rows with the card's (and
28,413 with the plane mirror's bytes) with no difference. Every A/B arm's replies are compared with
the base engine's; the final configuration also matched it on 16 chat-framed prompts, on the eight
prompts three times on one server, and on 256-token replies.

Exact tall passes (`RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1`) keep this: a pass of up to 32 (48)
tokens gives every row the bits it gets in today's 16-token chunking, because every op either runs
the decode band's arithmetic at every height or is checked to (`ds41_mxfp4_device_check tall` and
`sweep`, `r4d_emu_ds41tall`), and the server refuses any step whose bands say otherwise. The A/B's
byte gate: two `Kld` arms of the same configuration (e.g. `...-Lm2048-Kld` and
`...-M32-...-Lm2048-Xt-BT32-Kld`; the plain one first: an `Xt` arm is never taken as the reference,
and `KLD_TIMEOUT`, 900 s, bounds a run) record every position of a corpus with `--kld-record` (its f16
log-probabilities, its top 16 in f32, and the next token's NLL and the whole row's entropy in f32, so
one changed logit anywhere in a row shows), and the second must equal the first byte for byte
(`ab_requests.py --kld-compare`).

## Large-chunk prefill

`RADIANCE_EXPERIMENTAL_DS41_LCP=C` (C a multiple of 16 in 64..2048) runs a lone prompt's prefill in
chunks of C tokens, each giving every row the bits of the same tokens in 16-token passes. It needs
`--max-num-batched-tokens C`, the CPU path on every routed unit (`..._POLICY=all`, `..._MISSES` ≥ C,
`..._FILE_BUFFER_UNITS`), chunks of C (the default checkpoint interval 2048 serves C up to 2048;
above it, `--checkpoint-interval C`; with the prefix cache and a smaller interval, `--prefill-span
C/I`), and a step a chunk alone; the server refuses anything else at startup with the reason (also
speculation, KV streaming, `..._FETCH_LIST`, `..._VRAM_PACKED` and `..._EXACT_TALL` other than 1),
and refuses a libr4d.so whose twin does not reach C (so a run home with an old plugin cannot mix
sums).
Under it libr4d's `ds41_mxfp4_device_check LIBR4D lcp [C]` holds the twin, the sparse ops (also as
the suffix launches of the per-row cut) and every other height-dependent op (gemm_nt by the row the
table resolves, the router's f32 projection and its 96-column split, the norms, mhc_mix, the head) to
their rows launched alone and 16 at a time, up to 2048 rows; `r4d_emu_ds41tall_lcp` holds the twin to
256 tokens, the norms to 1024 rows, gemm_nt's tiled row and the f32 split on the host.

The CPU path at C rows: jobs of up to `..._HOST_EXPERTS_MAX_ROWS` (now up to 64: two many-row calls
on one weight tile; set it to 32 or 64, since at the default 4 a chunk's experts take about twice the
CPU time a row, and rank 0 warns) cut evenly, hits submitted longest first, 256 pool slots, a submitter waiting for a
slot outside the submit lock; a chunk's misses are laid out and their jobs built before the reads, and
each unit's jobs go to the pool as its last extent lands (`Mover::read_unit_planes`' `landed`), at most
`..._LCP_PLANES` (48) units read at once. The RAM cache admits a chunk's units only where the prompt's
last `..._LCP_ADMIT` tokens (128; 0 none; `all` every unit) select them, decided as the chunks reach
those rows (`RadBatch::lcp_admit_row`); other units are leased without an LRU touch
(`borrow_packed_readonly`) and their misses computed and let go; at most `..._LCP_INSERTQ` (16) admitted
misses wait for the insert threads. With `Cs` and the plugin's 16-row grain, each chunk runs the
16-row schedule in each of its 16-row groups: layer L in full on the chunk's rows from
`consumer_row0[L]`, its key ops from the layer below's (`core/sched/consumer_skip.h`,
`consumer_row_starts`; the same (layer, row) set as 16-row passes).

Arm parts: `LcN` (needs `MN`, `B1` and an `Mr` part, `Mr32` or `Mr64`; never with `Xt` or `BT`; never
the Kld reference), `AdN` / `Ada`, `LpN`, `LiN`, `Mr` up to 64. `lcp` in `DEVICE_CHECKS` (with
`TALL_CHECK` and `TALL_LIBR4D` named, `LCP_CHECK_C` the chunk, 1024) runs `lcp C`, `sweep` under LCP and
`sweep0` on card 0 (logs `devcheck-lcp-*.log`), and the window measures an Lc arm only on the libr4d.so
(sha256) those passed on, at a chunk of at most `LCP_CHECK_C`. The byte gates are those of exact tall
passes: a plain `Kld` arm first, then `...-M1024-...-Lc1024-Kld` (identical records), and `Kh` arms
with and without `Cs` (identical prefill hashes). Leave `Vp` out of an Lc arm (refused) and `Rb` too
(`gemm_nt_f32bf16` is not in the sweep).

## Many token rows of one expert

`gemv_avx2_many` computes up to 32 token rows of one matrix in one call (more in calls of 32), for
prefill passes whose rows meet in one expert. A tile's output rows go in blocks of 16; each round of each row is decoded
once into f32 weights (w = kv · 2^(e−128), exact), and each token row adds x · w value by value in
register blocks of two output rows by four token rows. Those are the card's two operations a value,
in its order, so every row equals `gemv_avx2` on that row alone: `ds41_host_experts_test check`
holds that at 1–64 rows in every layout (block_mxfp4, planes, packed) of every case (the release's
scales, the edges with NaN and Inf rows, partial rounds) and over output rows [8, N); `packed` adds
nine rows on every packed case.

`RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=N` (1–64; default 4, the behaviour so far) lets a
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

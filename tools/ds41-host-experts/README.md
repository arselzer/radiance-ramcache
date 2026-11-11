# DeepSeek-V4.1 host experts

DeepSeek-V4.1 Flash on four RX 9070 XT cards keeps most of its routed experts out of VRAM: in a
RAM cache, or on the NVMe drive. This directory holds the CPU kernels and tools behind the engine's
host-expert path: the CPU computes every routed expert the cards do not hold, bit-exact with the
card's MXFP4 GEMV, so every reply stays byte-identical to the base engine's.

The model behind the measurements is a hybrid conversion of the official
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):
routed experts retain the release's native FP4 E2M1 codes and E8M0 scales; other neural weights
come from that checkpoint with BF16/F32 conversions; the two Engram embedding tables use
ISTA's IQ4_NL conversion. The supplemental IQ2_XS file supplies integer Engram controls only,
not IQ2 expert weights. This is not a wholly untouched official-precision checkpoint. The
exactness claims below compare optimizations with this same imported base engine; they do not
establish equivalence to an independent official runner. The numerical audit passed all 1,119
other neural tensors against their declared decode/cast mappings (zero mismatches), and all seven
integer Engram controls matched the official rules. The two IQ4_NL Engram tables remain additional
quantization; the audit does not turn this into an untouched official-precision checkpoint.

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

Host-expert optimizations are opt-in and off by default. The measured best configuration:

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
RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=6             # two VRAM file buffers of 6 units: Phase 1's 12-slot ring (1 without P1)
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
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=32          # prefill: CPU jobs of up to 32 rows of one expert, many-row kernel
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=2048         # misses computed on the CPU in passes of up to 2048 tokens (>= the chunk with a lent arena)
RADIANCE_EXPERIMENTAL_DS41_LCP=2048                         # large-chunk prefill: chunks of 2048 tokens (with --max-num-batched-tokens 2048)
RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=all                    # recommended 192 GiB cache: retain every unit a chunk uses (use 128 with smaller/changing budgets)
RADIANCE_EXPERIMENTAL_DS41_LCP_INSERTQ=64                   # bounded insertion queue; zero drops in the 6.8K/48K qualification
RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE=1                    # routed gemv: work split by row passes, all-dropped passes skipped (exact)
RADIANCE_EXPERIMENTAL_DS41_P1=1                             # prefill Phase 1: layer passes of >= 320 rows (..._P1_MIN_ROWS) compute every routed expert on the cards (needs the libr4d whose P1 card check passed)
RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=64                 # the activation arena's idle top holds ~42 decode residents a card between long prefills (needs the deepseek41 plugin from this tree and ..._EARLY_ISSUE=1)
RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1                    # static VRAM experts in the packed-scale form, x1.044 as many a card (gemv_vp and the P1 row read it; needs --deterministic and the libr4d whose vpack card check passed)
RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1                    # the 40 routers and 3 compressor kv|gate projections read as bf16, 180 MiB a card back to the experts (the load refuses one with a nonzero f32 low half; the deepseek41 plugin and libr4d from one build, whose gemm_nt_f32bf16 card check passed)
RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1                # no pass after a reply's end-of-generation token
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1               # a partly read read-ahead unit's rest read whole in its layer's own read
RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB=256            # background inserts in 256 KiB pieces, waiting for an idle pool
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int8             # the read-ahead predictor scores in int8, its best 12 again in bf16
RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=7                    # up to 7 units a rank the packed cache refuses keep their planes
RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1            # pool workers sleep on a futex, woken without a lock
RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=26000               # optional, only with a fifth card of the rank cards' architecture listed after them in ROCR_VISIBLE_DEVICES: 26000 MiB of its VRAM hold the hottest experts the CPU computes (see "The aux expert worker")

radiance ... --tp 4 --placement expert_tiered --weights-disk-tier --deterministic \
  --expert-ram-cache-total-mib 196608 --host-pool-mib 256 --embedding-placement ram \   # 192 GiB on a 256 GB box (98304 on 128 GB)
  --max-num-batched-tokens 2048 --expert-profile <decode profile> \
  --expert-plane-mirror <dir of the checkpoint's model-*.safetensors> \
  --expert-plane-mirror-decode-share 0.13                              # the mirror is optional, see below
```

**For agent harnesses (multi-turn sessions)** add the prefix cache with end-of-request snapshots on a 16-token grid:
`--prefix-resume exact` (the default), `RADIANCE_EXPERIMENTAL_PREFIX_EORS=1` and
`RADIANCE_EXPERIMENTAL_DS41_RESUME_GRID=16`. A turn then resumes within 15 tokens of the previous reply's end
instead of at the last multiple of the 2,048-token chunk; ds41 prefill is partition-invariant under the switches
above for passes on the 16-token quantum, and the switch refuses configurations where that is not established (a
tuned kernel point per M band, no large-chunk prefill). Measured on 13 harness turns of up to 9.7K tokens
(ab-20261031T090154Z): prefill hashes identical to the cold run on 13 of 13 turns, every reply identical; summed
time to first token 72.7 s, against 82.5 s on the chunk grid and 135.1 s cold; short follow-ups 0.3-0.6 s instead
of 3.3-3.7 s; decode unchanged. Add `RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD=0` (Phase 1's read-ahead off, part `Qa0`)
for these turns: on the same 13 turns with the switches above at `--max-model-len 32768` (ab-20261104T081812Z, A-B-B-A,
prefill hashes on) the summed time to first token was 63.48 and 61.28 s against 72.03 s (and 89.13 s in the window's
first long-prefill arm), 12-15% less; follow-up turns gained most (3.5 to 1.7 s, 6.8-7.0 to 4.1-4.3 s), the long first
turn lost 7-10% (14.6-14.9 to 15.7-16.4 s); decode unchanged; prefill hashes 13/13 and replies 52/52 identical.

With admit-all on the tested 192 GiB cache, a later warmed 6,817-token screen after an N8 wave
measured 12.8833 to 12.3527 s TTFT with Qa0 (-4.12%; control drift -0.84%). All 75 original
responses/counts, three full cold hashes and eleven joined exits passed. Post-long N8 decode
was neutral (-0.19%); the cold 48K comparison was inconclusive because its controls drifted
-11.96%. This extends the warmed-prompt evidence; it does not change the general default of eight.
See [November 5 results](../../docs/DS41-RESULTS-20261105.md) for scopes and the final receipt.

**DSML chat replies**: the ds41 plugin declares upstream PR 266's reply format, so
`/v1/chat/completions` returns typed DSML calls as `message.tool_calls` and thinking as
`message.reasoning_content`. The format header was already present from the upstream-main sync;
only its registration in the ds41 plugin changed. It applies by default to chat reply parsing.
The benchmark chat and harness sets frame their own prompts and use raw completions. Across
`ab-20261031T111018Z` and `ab-20261031T115207Z`, registration left all 16 chat and 13 harness replies byte-identical,
with identical prompt IDs, generated token counts and harness prefill hashes; the base arm matched
the merged base on 8/8 held-out replies. A thinking-on tool request parsed both fields,
and all 116 generated token IDs matched the same rendered prompt through raw `/completion`.
The research container carries no chat template: pass `--override-chat-template <DS41 template>`
to serve chat. The window used `work/dsml/ds41-chat.jinja`, a text-only verification adapter whose
tool-request prompt equals the local official V4.1 encoder byte for byte. The template and offline
check are saved with the window evidence. The first window stopped at a 501 before the live request;
the follow-up supplied this explicit template and reused the unchanged libr4d's passed card gates
by SHA256. No model/container metadata was modified.

**For concurrent requests** (agent harnesses) add `RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1` and `--max-num-seqs N`
(with `--max-model-len` sized so N sequences' KV fit, e.g. 1024 at N = 4-8): large-chunk prefill refuses steps that
mix a prompt's chunk with other sequences' decode rows, and the window arms at 4 and 8 streams below ran that way.
Where short prompts arrive in bursts (several sent at once) and the prefix cache is off, also add
`RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=256`: whole prompts waiting together then share one prefill-only step instead
of one a step. It needs prefill alone, large-chunk prefill (`..._LCP`), `--max-num-seqs` above 1 and
`--no-prefix-cache`, and is refused with the reason otherwise, so the agent-harness setup above (prefix cache on) does
not get it. In one window (ab-20261103T230607Z, the 8 held-out prompts sent 4 or 8 at a time) every reply was
identical to the engine without it, and so was every prefill hash at 8 streams; the total at 4 / 8 streams rose
8.3-12.9% / 8.5-9.3%, and at 8 streams the first reply's first token came about 1.9 s later and the last one's
1.2-1.4 s earlier (the `..._BATCH_ADMIT` row of the table below). Prompts arriving apart and prompts longer than 256
tokens are admitted as before and gain nothing.
In the paragraphs from here to the two-batch overlap one, every 4- and 8-stream arm also ran
`..._PREFETCH_MULTIROW=N` (parts `Pmr4` / `Pmr8`) with `--max-model-len 1024` (`L1024`), sending the 8 held-out
requests N at a time. A total at N streams is N times the mean of the requests' `predicted_per_second`
(`ab_requests.py --compare` prints it per request); a one-stream rate is that tool's decode line.

**2,048-token chunks with the lent arena** (the block above without `..._VRAM_PACKED` and `..._ROUTER_BF16`;
ab-20261031T011849Z, T022949Z, T030023Z, before the upstream sync: prefill hashes and every reply identical): time to
first token 5.0 / 7.2 / 17.2 s for prompts of 1,890 / 3,086 / 6,817 tokens, against 7.5 / 8.6 / 26.7 s at 1,024-token
chunks (the 26.7 s from ab-20261030T231526Z; shorter prompts unchanged); held-out decode 29.45 against 29.91 tok/s
(−1.5%); 4 and 8 streams 37.2 / 41.5 against 38.0 / 41.2 tok/s in total (−1.9% at 4, a tie at 8); short replies right
after a long prompt decode ~6% slower until the lent residents are reloaded (~1-1.5 s). For decode-heaviest use keep
`..._LCP=1024`, `--max-num-batched-tokens 1024`, `..._MISSES=1024` and no lent arena. Both chunk sizes include the
compressor tail sized to the step (+0.6% decode at 1,024). On the stable build (runtime e184c6b7, libr4d 501b98e2,
four RX 9070 XT) the same configuration decoded 37.27 / 40.84 tok/s in total at 4 / 8 streams (ab-20261103T090558Z,
one run each; 37.20-37.54 / 41.24-41.53 on earlier builds in ab-20261031T030023Z, T100852Z and T123034Z) and 28.68
tok/s at one stream (ab-20261103T110235Z, one run; 27.83 and 24.41 in ab-20261103T114720Z and T090558Z, each the first
arm after production stopped, with a slow first or stalled request); it took 1.89 / 3.86 / 4.24 / 5.39 / 7.39 s to
first token at 82 / 272 / 933 / 1,890 / 3,086 tokens (ab-20261103T090558Z, one run with prefill hashes on) and 17.22 s
at 6,817 tokens sent alone after a start (ab-20261103T114720Z, the second of two runs; 17.39 s in
ab-20261103T110235Z). Times depend on the prompts before them: the `..._P1_AHEAD` note below took
8.71-8.92 / 11.83-11.85 s at 3,086 / 6,817 tokens on the same build (with `Rb`/`Vp`, the default read-ahead), with 82,
3,086 and 6,817 tokens sent in turn to one server, the 6,817-token prompt's first chunk reading 178 drive units
instead of 279. Every held-out reply was identical to the base engine. The decode-only M16 setup with
`..._VRAM_PACKED` and `..._ROUTER_BF16` measured 38.7 / 42.9 at 4 / 8 streams (ab-20261030T231526Z, before the
upstream sync; not re-measured).

**Recommended admission with a 192 GiB expert cache** (four RX 9070 XT, C2 at 2,048-token chunks): the block above sets
`RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=all` and `RADIANCE_EXPERIMENTAL_DS41_LCP_INSERTQ=64`. Every expert unit a
chunk selects can then enter the existing bounded RAM cache, instead of only units selected by its last 128 rows.
Later chunks reuse those bytes rather than reading them again. Weights, routing and arithmetic are unchanged.
On the unchanged stable runtime f84b23db / libr4d 92bd7719, uncached prefills use a fixed
131,072-token reservation; the prefix-cached follow-up uses 32,768:

- A fresh 6,817-token prompt took 13.24 s to first token against 17.16 s with default admission,
  22.8% less time in a quiet single-pair screen (ab-20261104T115653Z; the separate startup control
  and hash-instrumented arms are retained).
- Quiet 47,917-token prompts in OFF/ON/ON/OFF order took 100.68 / 62.04 / 61.53 / 99.83 s, a 38.4% reduction by
  pair means (ab-20261104T121517Z). Each was followed on the same server by the original eight held-out requests at
  concurrency eight: mean total decode 46.67 versus 47.16 tok/s, wave time 14.08 versus 14.00 s, no measured regression.
  The two admit-all runs measured 46.50 and 47.83 tok/s, using eight times the mean server-reported
  per-request decode rate, consistently with the overview table.
- Quiet 99,987-token prompts in OFF/ON/ON/OFF order measured 240.4594 versus 151.3235 s TTFT by pair means,
  37.069% less (ab-20261105T160034Z). Server-reported prefill throughput averaged 416.43 versus 660.97 tok/s
  (+58.72%; OFF endpoint drift 7.60%). The subsequent original N8 wave's canonical total decode was
  47.3029 versus 47.9545 tok/s, while wave wall time was neutral at 13.9895 versus 14.0133 s;
  this is a post-prefill guard, without a broad decode-gain claim.
- After a fresh held-out wave at concurrency eight, the 6,817-token prompt took 13.33 / 11.74 / 12.02 / 12.82 s in
  the same OFF/ON/ON/OFF order, 9.1% less time by pair means. All warming requests and observer snapshots are retained.
- All 80 replies and original prompt/generated counts across the earlier 6.8K/48K windows matched their qualified references;
  all five captured full prefill digests matched too. Every admit-all rank reported zero insert-cap drops. Some
  experts were evicted, which is why the post-prefill decode guard is required.
- The further 100K/prefix window passed all 84 original replies/counts, 14 full prefill hashes and nine
  joined server exits. Its quiet 13-turn prefix-cached pair reduced summed TTFT from 68.9714 to 56.7298 s;
  this single pair is exploratory. The separate hash-instrumented ON arm is timing-ineligible.

This is the recommended default for the tested 192 GiB configuration. When reducing or changing its RAM-cache
budget, also set `..._LCP_ADMIT=128` until separately measured; admit-all can thrash a smaller cache. The new
100K/prefix evidence extends the existing recommendation on four XT cards and a 192 GiB cache: 100K prompts
at a fixed 131K reservation, prefix-cached turns at a fixed 32K reservation. It does not qualify
smaller/dynamic caches or the optional worker combination. The engine's
general fallback remains `128`, preserving existing configurations that do not select this profile.

**4,096-token chunks** (opt-in, exact; for one stream prefilling uncached prompts well above 2,048 tokens, not for
concurrent decode): `..._LCP=4096` with `--max-num-batched-tokens 4096`, `..._MISSES=4096` and
`--checkpoint-interval 4096` (Phase 1 refuses an interval below the chunk; the default is 2048); window parts `M4096`
`Lc4096` `Pci4096`, card checks with `LCP_CHECK_C=4096`. With `..._VRAM_PACKED` and `..._ROUTER_BF16` on the stable
build (ab-20261103T090558Z; against 2,048-token chunks without them unless noted): every held-out reply at 1, 4 and 8
streams identical to the base engine (24 of 24), prefill hashes identical to plain 16-token passes on the five long
prompts (against the plain arm of ab-20261103T110235Z), logits identical to them at 3,397 of 3,397 positions over 20
documents. A 3,086-token prompt took 6.00 against 7.39 s (7.41 with `Rb`/`Vp`, ab-20261103T110235Z): one 3,086-row
Phase 1 chunk instead of two chunks (1,040 + 2,046 tokens; Phase 1 passes of 1,040 + 2,032 rows), 401 drive units
against 587 (5.69 against 7.33 s in ab-20261031T162629Z, both without `Rb`/`Vp`); prompts of up to 2,048 tokens gained
nothing (1,890: 5.53 against 5.39 s, and 5.09 with `Rb`/`Vp` in ab-20261103T110235Z). Costs: the activation arena
doubles (1.78 GiB against 910 MiB a card at `--max-model-len 1024`), leaving 590 / 598 static residents a card against
665 / 674 at 2,048 with `Rb`/`Vp` (627 / 635 without), which the lent arena's 64 slots (the cap) do not make up: 4 / 8
streams total 36.27 / 39.63 tok/s, −4.5% / −5.1% against 2,048-token chunks with `Rb`/`Vp` and −2.7% / −3.0% without
(one run each; its 8-stream arm ran in a 10-minute interval with about 5x the direct page reclaim, `sar -B`, of the
interval that held its 4-stream arm and the 2,048-token arms at 4 and 8 streams). The replies after the long prompts
decoded 5.8% slower (23.56 against 25.00 tok/s, one run each); the first request after a start was slow (82 tokens
4.88 against 1.89 s; 3.48 against 1.91 s in ab-20261031T162629Z), and the 933-token prompt took 9.53 against 4.24 s,
its first Phase 1 chunk reading the same 383 drive units at 0.76 against 1.70 GB/s (the slow first chunk noted under
`..._VRAM_PACKED` below, which 2,048-token chunks show too). These prefill times are single runs with prefill hashes
on, each after the 2,048 arm. The 6,817-token gain (12.69 against 17.12 s, ab-20261031T162629Z, without `Rb`/`Vp`) is
not re-measured on the stable build, whose 2,048-token chunks take 17.22-17.58 s for that prompt alone after a start,
with or without `Rb`/`Vp` (ab-20261103T114720Z, the three runs after one with the slow first chunk). With the prefix
cache, ab-20261031T162629Z's 13 harness turns took 64.96 s summed time to first token against 72.70-73.84 s at 2,048
in ab-20261031T100852Z, T123034Z and T182505Z (11-12% less) but decoded at 24.20 against 25.81-25.95 tok/s (6-7%
slower): different builds, without `Rb`/`Vp`, not re-measured. Above 2,048 rows the stable build's exactness evidence
is that one 3,086-row chunk and the card checks (the logit corpus stops at 1,886 tokens); without `Rb`/`Vp`,
ab-20261031T162629Z also matched 2,048-token chunks on the long prompts (at 96 GiB too) and the 6,817-token prompt,
and ab-20261031T100852Z's 13 harness turns.

**Phase 1 prefill** (`..._P1=1` with the 6-unit file buffers; ab-20261030T231526Z, every reply and prefill hash
identical to the large-chunk path): time to first token 1.9 / 3.7 / 4.2 / 7.4 / 8.7 s for prompts of 82 / 272 / 933 /
1,890 / 3,086 tokens, against 1.9 / 3.7 / 10.5 / 18.5 / 28.0 s without it (355 tok/s at 3,086 tokens); 6,817 tokens
in 26.7 s against 51.2 s. Layer passes below 320 rows keep the large-chunk path, so short prompts are unchanged.
Decode stays within ~1% (29.68 against 29.88 tok/s on the held-out prompts). `..._P1_LEND` (decode residents in the
ring's idle slots) measured nothing more and stays off. 2,048-token chunks are faster still for long prompts
(6.6 s at 3,086 tokens in ab-20261030T201710Z) but their activation arena costs ~50 VRAM residents a card.

`RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1` (part `Vp`) and `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1` (part `Rb`) are
in the block above since the stable build (runtime e184c6b7, libr4d 501b98e2, four RX 9070 XT) passed their gates
under Phase 1 at 2,048-token chunks (ab-20261103T090558Z, T110235Z, T114720Z). Under Phase 1 the packer packs (gemv_vp
serves passes of up to 16 tokens, the P1 row decodes the packed form above them; large-chunk prefill without Phase 1
still refuses `Vp`), and `Rb` gives 180 MiB a card back, so a card holds 665 / 674 static residents (rank 0 / the
others) against 627 / 635 without them at `--max-model-len 1024` (658 / 667 and 620 / 628 at 8192, 642 / 650 and
604 / 613 at 32768). Against the block without them: 4 / 8 streams 37.98 / 41.77 against 37.27 / 40.84 tok/s in total
(+1.9% / +2.3%, one run each, the pairs in opposite orders; the same RAM-cache misses and 3.3% fewer rows sent to the
CPU; ab-20261103T090558Z), above every run without them on the earlier builds too (the 2,048-token paragraph); one
stream, one run each: 28.36 against 28.68 tok/s (−1.1%, ab-20261103T110235Z) and 29.29 against 27.83
(ab-20261103T114720Z, where the arm without them ran first after production stopped and its first request was slow);
13 harness turns with the prefix cache in 78.29 against 79.02 s summed time to first token, decoding at 25.00 against
24.46 tok/s (ab-20261103T110235Z, one run each); the 6,817-token prompt in 21.01 / 17.58 / 17.22 / 17.22 s without,
with, with and without them, in that order (ab-20261103T114720Z; the 21.01 s run had the slow first chunk below).
Exact: all 32 held-out replies at 1, 4 and 8 streams identical to the base engine; prefill hashes identical to plain
16-token passes on the five long prompts and the 6,817-token one (8c583a997ef6f1db in all seven runs of it,
ab-20261103T110235Z and T114720Z); 13 of 13 harness turns identical to the block without them (prefill hashes,
replies, 60,272 cached tokens); logits identical to plain 16-token passes at 3,397 of 3,397 positions over 20
documents (ab-20261103T110235Z against ab-20261103T090558Z's plain record). These stable-build gates ran at 192 GiB;
at 96 GiB both switches were checked only on an earlier build (ab-20261031T144623Z: prefill hashes on the five long
prompts identical to plain 16-token passes). `Rb` needs the deepseek41 plugin and libr4d from one build, `Vp` a static
plan (`--deterministic`) to pack, and each its card check passed on the served libr4d (`SELFTEST_F32BF16` for `Rb`,
`VPACK_CHECK` for `Vp`; `ab_window.sh` measures their arms only on a libr4d checked in the same window). A server's
first Phase 1 chunk sometimes read the drive at about half rate, cause unknown: in the first 6,817-token run above
(without them; the same 279 drive units at 0.97 against 1.59-1.77 GB/s in the three runs after it), in the long-prompt
run at 4,096-token chunks (above; the same 383 units at 0.76 GB/s, while the 2,048-token run just before it read them
at 1.70) and in both logit-corpus arms (with both switches; 399 units at 0.87 GB/s at 2,048-token chunks in
ab-20261103T110235Z, 0.68 at 4,096 in ab-20261103T090558Z). Time long prompts over repeats. ab-20261103T110235Z's
6,817-token run with both (25.36 s) overlapped a git push on the box and is not used. A decode-only server without
`..._LCP` (and so without Phase 1) keeps both; there `Vp` packs only with `--max-num-batched-tokens` at most 16.

Two-batch overlap of concurrent decode steps (`RADIANCE_EXPERIMENTAL_SCHED_TBO=1`, opt-in, exact) loses on 256 GB
too: 37.4 against 38.7 tok/s in total at 4 streams and 40.7 against 42.9 at 8 (ab-20261030T231526Z). Leave it off.

**For contexts beyond 32K** keep the KV reservation fixed: with the block above (`--deterministic`) the pool is reserved
at startup for one sequence of `--max-model-len` tokens on every card and paid for in VRAM expert residents (startup's
`kv:` lines). Use `--max-model-len 131072` (part `Lm131072`) for long-context harness use; the 4,096-token
chunks above (`..._LCP=4096` with its companions, parts `M4096` `Lc4096` `Pci4096`) when long prompts dominate: they
prefill 1.34-1.5x faster at 48K-100K tokens, at the cost of the 4- and 8-stream decode deficit described there; and
`L1024` or another small `--max-model-len` for decode-heavy concurrency. Growing KV is not the route: it refuses Phase
1, the lent arena and more than one sequence, funds its full maximum at startup anyway, and its re-commit corruption is
still open (`/home/as/ds41-decode-review-20261028/LONGCTX-FIFTHCARD-20261103.md`, 3.3-3.4). Measured on four RX 9070
XT with the block above, one run each (ab-20261103T230607Z, the 524,288 row ab-20261104T003545Z; claude/ds41-batch-admit
f4fe2ff1 and abe09503, libr4d f1b1507a in both):

| `--max-model-len` | KV a card | rank 0: static residents + lent slots | 1 stream, held-out prompts |
|---|---|---|---|
| 512 (no `Lm` part) | 21.6 MiB | 667 + 41 | 29.11 tok/s |
| 65,536 | 287 MiB | 635 + 51 | 29.50 |
| 131,072 | 487 MiB | 621 + 48 | 29.35 |
| 262,144 | 887 MiB | 594 + 40 | 28.66 |
| 524,288 | 1.65 GiB | 525 + 40 | 27.73, against 29.73 at 512 in its window (−6.7%) |

The rest is ab-20261104T003545Z:
- **At 8 streams** with batch admission (`Ba`, `Pmr8`, the held-out prompts sent 8 at a time) `Lm131072` totalled
  44.91 / 45.12 against 46.04 / 45.67 tok/s at `L1024` (−1.7%, two runs each).
- **Exact.** At `Lm131072` with prefill hashes on, the 6,817-token prompt gave its reference prefill hash
  (8c583a997ef6f1db) and reply; the 32,768-token fill prompt and its 82-token follow-up (`Ufill32k`) gave the hashes
  71eaab9a74338840 / 112530d7d4d5b367 and both replies of the reference run with plain 16-token passes (M16) on the
  stable e184 build; at 47,917 and 99,987 tokens (`Ulong48k`, `Ulong100k`; no 16-token reference exists above 32K)
  2,048- and 4,096-token chunks gave the same prefill hash (87d485202785b186, 53cf0ecb89f99dad) and reply. All 48
  held-out replies (at 512, 1,024, 131,072 and 524,288) equalled the base engine's. The card check `lsparse`
  (`DEVICE_CHECKS=lsparse`, with the `TALL_LIBR4D` / `TALL_CHECK` pair): index_score, index_candidates, index_topk and
  attn_sparse_sink on a 256-row chunk ending at index widths 32,769-524,289 gave 0 differing values against the same
  rows launched alone, 16 at a time and as suffixes (top-k also against the host). It is logged and gates no arm.
- **Prefill** at `Lm131072` (prefill hashes on): 47,917 tokens 115.1 s (416 tok/s) at 2,048-token chunks against 76.6 s
  (626 tok/s) at 4,096; 99,987 tokens 221.7 s (451 tok/s) against 165.1 s (606 tok/s), and 224.5 s at 2,048 with the
  hashes off; the 32,768-token fill prompt (a 6,817-token text repeated) 64.2 s.
- **Decode at a long context**: after the 99,987-token prompt, prefill hashes off, 20.9 tok/s against 29.35 at
  `Lm131072` on short prompts in the table, but that gap is the text, not the context. Profiled in ab-20261104T025134Z
  (part `Ct4`, card 0's kernel times for one decode pass in 4): a 1,899-token head of the same text (`Ulong2k`) decoded at
  52.60 ms a step against 53.43 ms after the 99,987 tokens (+1.6%), while the 8 held-out prompts took 37.67 ms; the long
  text sends more experts to the CPU (22.4-22.8 gathers with CPU rows a step against 19.4). The one cost that grows with
  the context is index_score, 11-13 to 329 us a call on its 8 layers (+2.5 ms of card time a step, mostly hidden behind
  the CPU's rows). Arms with prefill hashes decode slower still (12-14 tok/s), an instrumentation cost.
- The prefix cache did not run at these lengths; idle conversations must fit three quarters of the pool (**Long
  contexts** under Prefix caching). Receipts: `make_longctx_receipts.py` (parts `Ufill32k`, `Ulong48k`, `Ulong100k`;
  a receipt's `max_context_tokens` skips an arm with a shorter `--max-model-len` and lengthens its request timeout).

**On a box with a fifth card of the rank cards' architecture** (optional; window F1 ran an R9700, gfx1201 like the
four RX 9070 XT) add the block's last line, `RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB>`, and list the extra card
after the rank cards in `ROCR_VISIBLE_DEVICES`: its VRAM then holds the hottest experts the CPU path computes and serves
their rows in passes of up to 16 tokens, while the CPU computes the rest. Exact (every row it serves is the CPU path's
bytes), and in its one window (ab-20261104T051616Z: the block above with batch admission, the 8 held-out prompts, 26000
MiB of the R9700 beside whisper-server) +24.1% in total at 8 streams (A-B-B-A), +18.1% at 4 and +19.9% at one, every
reply identical to the base engine. The extra card must have no other compute client (BoltzGen idle); without such a
card the switch turns itself off with the reason and the server runs as with four cards. Details, switches and
caveats: "The aux expert worker" below.

**On 256 GB of RAM (8x 32 GB DDR4-2666) with a 192 GiB expert cache** (ab-20261030T194..., replies identical): decode **31.2 tok/s** a stream (19.9 with 96 GiB on 128 GB of DDR4-2133; 20.9 with 96 GiB on the new RAM, so the cache size, not the memory speed, is the gain: rank 0's misses fall 2,073 → 361 on the held-out prompts); **38.7 tok/s in all at 4 streams**; prefill 3,086 tokens in 28.1 s (110 tok/s) and 6,817 in 51.7 s (132 tok/s). 192 and 208 GiB measure the same.

**Drafter quantization pilot (8 November):** the separately built prototype retains every
drafter expert and tests IQ2_XXS gate/up with MXFP4 or IQ2_S down weights in RAM and VRAM.
The [root README](../../README.md#drafter-quantization-pilot-8-november) contains the five-arm
acceptance/decode table; the [conversion recipe](../../docs/DS41-DRAFTER-QUANTIZATION-20261108.md)
records the source, GPU commands and validation. All 80 pilot replies/counts matched base.
These are provisional results; the target checkpoint, installed engine and defaults are unchanged.

**Earlier separately built DSpark candidate (7 November):** runtime `fbc0f812`, ABI33,
fixed 131,072-token context, one stream, 192 GiB expert cache on the 256 GB host. Both sides use
the recommended 2,048-token C2/P1 profile. The stable engine/configuration above are unchanged.

Both sides below use a **2,048 MiB pinned HOST pool per GPU**. The plain 256 MiB pool
was not included in this window. These gains are relative to the tested plain baseline;
comparison with the best plain configuration awaits the four-condition pool/bank/drafting test.

| Original workload | Plain decode, tok/s | DSpark, tok/s | Change | Scope |
|---|---:|---:|---:|---|
| CODE16, cap256 | 24.81 | 28.59 | +15.23% | Matched OFF/ON/ON/OFF bracket |
| CHAT16, cap64 | 24.55 | 25.23 | +2.76% | Exploratory single pair |

All 96 original replies and both token counts matched; native/origin/policy gates, joined stops
and production restoration passed. Rates divide generated tokens by decode time and average
same-treatment arms. The ordinary held-out/concurrent table and earlier 32K prototype use different
workloads/settings. No concurrent, auxiliary or prefix-cache DSpark promotion follows.

The P1 partial-wave-flush candidate (Qwf) separately passed full 192/96 GiB model/activity gates.
Both quiet timing screens failed the overall 5% control-drift gate; Qwf stays OFF. Qwf-OFF means:

| Expert cache | Original prompt | Prefill, tok/s | TTFT, s | Decode, tok/s |
|---|---|---:|---:|---:|
| 192 GiB | 6,817 tokens | 431.09 | 15.857 | 26.50 |
| 96 GiB | 6,817 tokens | 191.79 | 35.568 | 11.34 |
| 192 GiB | 99,987 tokens | 682.68 | 146.523 | 21.68 |
| 96 GiB | 99,987 tokens | 653.92 | 153.293 | 12.66 |

These are diagnostic reference points from separate windows, not a qualified cache-size effect.
Both used the 256 GB host, fixed 131K/N1/2K chunks/admit-all; decode is generated-token-weighted and
startup is separate. A 96 GiB cache test here does not validate the whole current configuration on
128 GB. See [November 7 results](../../docs/DS41-RESULTS-20261107.md) for OFF/ON data and drift;
the [November 5 report](../../docs/DS41-RESULTS-20261105.md) retains the earlier 32K prototype.

**Earlier single-stream DSpark measurements and configuration.** Exact (every draft is verified; replies identical in
every window). On 256 GB with the 192 GiB cache, against the configuration above with packed VRAM
residents on both sides (ab-20261030T205440Z): code prompts **+12.5%** (29.79 vs 26.48 tok/s, 16 coding
tasks), held-out prompts **+9.3%** (34.15 vs 31.24), chat +1.2% (26.92 vs 26.60). On 128 GB it won only
on code (+9.2%; held-out a tie, chat -1.3%): there the experts the drafter displaces from VRAM fell to the
drive, while the bigger cache keeps them in RAM. The bank also serves beside concurrent requests
(--max-num-seqs above 1, with RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1): a sequence drafts only while it
decodes alone, and every step of two or more decoding sequences is a plain batch (with
RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=1 those steps run no drafter pass; the one left alone catches its
drafter up from a per-sequence tap history). Measured in ab-20261030T234957Z, replies identical: one stream
unchanged (code 29.80 / 29.74 with the deferred pass, against 26.54 tok/s plain), four streams 37.1 / 36.9
against 38.8 tok/s in total (-4.5%: the bank's VRAM, nothing drafts while several decode). So the bank pays
where one request decodes alone more than about a quarter of the time; otherwise plain decode gives 38.8 tok/s
at 4 streams and 42.2-42.9 at 8. Verifying two
drafted tokens a step (`RADIANCE_EXPERIMENTAL_DS41_DSPARK_VERIFY_CAP=2`) adds 3.3% on code (30.73 vs
29.75 tok/s, 80% of second drafts accepted) but loses 12% on chat (43%); three lose on both
(ab-20261030T213623Z). Each verified row costs ~21 ms a step, the RAM reads of its experts, so keep 1
unless the load is code alone.
```
RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1                # the 3-stage DSpark drafter from its bank
RADIANCE_EXPERIMENTAL_DS41_DSPARK_VERIFY_CAP=1              # one drafted token a step
RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=2             # 2-row query block (with --draft-short-from 1)
RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=2                     # draft expert GEMVs on the live tail
RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=24                    # 24 hot draft experts a stage and card in VRAM, router masked
RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET=<hot set file>      # draft_hotset.py write (ranked from an --expert-stats capture)
RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8=1                     # the drafter's dense linears from E4M3 codes
RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=6              # read-ahead in 2-row verify passes too

radiance ... (as above) --draft-bank <draft.rad> --draft-bank-expert-placement host \
  --num-speculative-tokens 5 --spec-ngram 0 --draft-short-from 1 --expert-plane-mirror-min-rows 7
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
| `RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1` with `--spec-ngram K` | measured once (prompt lookup alone, ab-20261029T083552Z: 11.72 / 11.76 tok/s against 11.92; not recommended) | the plugin leaves its DSpark drafter out, so the core drafts by prompt lookup alone. Upstream refuses `--spec-ngram` beside a declared draft head; without the switch the plugin refuses `--spec-ngram K` for K other than the drafter's block (5). Replaces private ABI 32's `RadBuildCtx::draft_head_off`, dropped in the 2026-10-31 upstream sync (RadBuildCtx is now upstream's). Arm parts `SgK` / `SgaK` set it |
| `RADIANCE_EXPERIMENTAL_DS41_LCP=C` with `--max-num-batched-tokens C` (C = 1024) | measured (ab-20261030T105535Z, ab-20261030T144409Z): exact everywhere (Kld identical at C=256/512/1024, prefill hashes 5/5); alone it was 2.3-3.4x slower than 16-token passes (3,086 tokens 202.5 s with Cs), 84% of it the twin's dead-slot passes; with `..._MOE_BALANCE=1` 3,086 tokens take 31.7 s (97 tok/s; 16-token Cs+Mr16 64.5 s, +Mb1 59.8 s), 1,890 tokens 28.7 s, 933 tokens 14.3 s, 272 tokens 11.4 s (16-token 10.5 s); a 1024-row layer-pass is then 355 ms, 279 ms of it waiting for miss reads (the drive) | large-chunk prefill: chunks of C tokens with every row, logit and KV entry the 16-token chunking's. libr4d keeps the exact-tall pins at every height and raises its twin `moe_gemm_ggml_mxfp4_gemv_tall` to 2048 rows (`libr4d/r4d_ds41_tall.h`); `FileStager::lcp_init` admits a step only when every routed MXFP4 band is on the gemv rows with the twin reaching C, every other op is on one kernel or on a known byte-identical pair (gemm_nt m16 / m64 / tiled, the exact all-reduce's one- and two-shot), `--max-num-batched-tokens` is C, and nothing outside what was proven is on (EXACT_TALL other than 1, speculation, `..._FETCH_LIST`, `..._VRAM_PACKED`); `hx_init` requires the CPU to take every routed unit (`..._POLICY=all`, `..._MISSES` ≥ C, `B1`); `Scheduler::lcp_init` requires chunks of C (the checkpoint interval a multiple of C, or a span) and a step a chunk alone (`--max-num-seqs 1` or `..._PREFILL_ALONE`), and every rank's stager to have taken it. Misses stream to the pool as each unit lands (at most `..._LCP_PLANES` read at once), jobs are cut evenly and submitted longest first, and the RAM cache admits only what the prompt's last `..._LCP_ADMIT` tokens (128) select, as the chunks reach them (earlier rows borrow read-only; at most `..._LCP_INSERTQ` admitted misses queued). With `Cs` and the plugin's declared 16-row grain the consumer skip cuts each chunk per row (`RadBatch::consumer_row0`). Rank 0 logs `large-chunk prefill: passes of up to C tokens ...`. Arm parts `LcN`, `AdN` / `Ada`, `LpN`, `LiN` (below) |
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
| `RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=R` (32-256) beside `..._PREFILL_ALONE=1`, `..._LCP` and `--max-num-seqs` above 1, prefix cache off | exact; for a burst of short prompts arriving together +8.3-12.9% in total at 4 streams and +8.5-9.3% at 8 (window ab-20261103T230607Z, four RX 9070 XT, the recommended configuration with prefill alone, `L1024` and `PmrN`, the 8 held-out prompts sent N at a time; one window, below the route-trace replay's estimate of +13-17% / +12-18%). Every reply identical to the base engine (112 of 112: 8 prompts in 14 arms, `ab_requests.py --compare` exit 0), every prefill hash at 8 streams (8 of 8). A-B-B-A at 4 streams: 37.65 / 40.79 / 42.64 / 37.78 tok/s in total (+8.3% and +12.9% against the arm without it beside each), the wave's wall time 16.62 / 16.00 / 15.59 / 16.66 s, prefill steps 8 → 4 / 3, TTFT min / mean / max 0.49 / 1.64 / 2.97 and 0.47 / 1.67 / 3.03 s without it against 0.47 / 1.82 / 2.69 and 1.32 / 1.96 / 2.45 s with it; at 8 streams 41.79 / 45.68 / 45.36 / 41.79 tok/s (+9.3% and +8.5%), wall time 15.37 / 14.11 / 14.22 / 15.26 s, prefill steps 8 → 2, TTFT 0.96 / 3.29 / 5.45 and 1.01 / 3.18 / 5.29 s without against 2.86 / 3.17 / 4.09 and 2.88 / 3.19 / 4.13 s with it (the first token later, the last earlier); the pair with prefill hashes on (`Kh`) 42.75 against 46.31 tok/s. Prompts arriving apart and prompts longer than R are admitted as before. Recommended for such bursts ("For concurrent requests" above) | prompts waiting together are admitted into one prefill-only step instead of one a step, in arrival order, each whole (its prompt one chunk from position 0, the chunk it takes alone), while their rows fit in R (held to the chunk and, under Phase 1, below its threshold: at most 256 rows stays on the activation arena's smallest level, so the lent arena stays lent and the card path is never armed); decode rows wait as under prefill alone. A prompt of at most R rows that the rows left cannot take ends the batch (none behind it is taken ahead of it) and opens the next step's, which the prompts behind it may join. Kept out, each in a step of its own as before: a prompt longer than R, a request back from preemption, media, scored or embedding inputs, an adapter or plugin words. Only step composition changes: under LCP's admission every op gives a row the same bytes at any launch height (the `lcp` card check holds each row launched alone, 16 at a time and whole), and the sequence-aware ops (sparse attention, compressor work list, indexer, Engram ids) read each sequence's own rows, block tables and state through `cu_seqlens`; a shared step runs every layer on every row (the consumer skip cuts only a prompt alone in its step) and admits every unit into the RAM cache. Refused with the reason without prefill alone or LCP, at one sequence, with the prefix cache, KLD modes, `--prefill-span` or R-SWA. Startup line `batch admission: on (RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=R)`, shutdown line `batch admission: N steps admitted P whole prompts together`. Gates: replies and Kh against the arm without it at N4 / N8 (`ab_requests.py --hashes` matches prompts that arrived in another order; a prompt run in a shared step is labelled `not cut: shared step` in the hash lines and not counted as cut), steps_prefill. Arm part `Ba` / `BaN` (needs `Pf`, `Lc` and `NK` above 1; skipped beside `Pc` or `Kld`; `claude/ds41-batch-admit`); tests `sched_test` `batch_admission_*`, `ab_window_ba_test.sh` |
| `RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB>` (1024-1048576) with `..._HOST_EXPERTS_POLICY=all`, `..._PACKED_CACHE=1`, `..._VRAM_PACKED=1`, `..._FILE_BUFFER_UNITS` set, `--deterministic`, an extra card of the ranks' ISA visible after the rank cards (`ROCR_VISIBLE_DEVICES=<ranks' UUIDs>,<its UUID>`) | exact; +24.1% / +18.1% / +19.9% in total at 8 / 4 / 1 streams (window ab-20261104T051616Z, four RX 9070 XT and an R9700 holding 1,513 units in 26000 MiB, the recommended configuration with batch admission, the 8 held-out prompts; one window). Every reply identical to the base engine (104 of 104), the verify arm's rows 0 differ on every rank, prefill hashes 8 of 8, Kld 3,397 of 3,397. Optional, for a box with such a card ("The aux expert worker" below) | the aux expert worker (`core/place/aux_expert_worker.h`, `work/aux/SPEC.md`; experimental, default off: with the switch unset or 0 no other variable is read and nothing changes). The extra card (by capability, or `..._AUX_DEVICE=<BDF or GPU-UUID>`) holds the hottest CPU-path units it fits (per-rank quotas from the sealed profile's global order, loaded from the packed RAM cache and read back) and computes their rows for routed layer passes of up to 16 rows with the cards' own decode kernels (gemv_vp gate_up, swiglu_clamped, gemv_vp down) on a queue per rank; the CPU computes the rest as before. A job late at max(its start + `..._AUX_DEADLINE_US` (2000), the pool's end + 50 us), failed, or forced (`..._AUX_FORCE_LATE=N`, every Nth) has its units computed by the CPU from the kept leases, and the lane backs off `..._AUX_BACKOFF` (64) layers. `..._AUX_POLICY=verify` computes every row on the CPU and compares the worker's. Pinned host memory keeps mapping to the rank cards only (a pin set), the extra card's faults stay its own, and `..._AUX_HEADROOM_MIB` (1024) must stay free beside its other tenants. Startup lines `aux experts: device ...` and `aux experts: on: N units ...` (or `aux experts: off: <why>`), shutdown lines `aux experts: rank R: ...`. Refused with `..._SCHED_TBO=1`, `..._ROUTE_TRACE` or a dynamic plan. Arm parts `Ax` / `AxN`, `Axv` / `AxvN`, `Ax0`, `AxfN` (`claude/ds41-aux-worker`; an `Ax` / `Axv` part beside `Kld` only on the 16-token base, e.g. `...-M16-...-B1-...-Rb-Vp-Ax-Kld`: KL mode is prefill only and the worker takes passes of up to 16 rows, so an `Lc` or `BT` step above 16 would leave it no row; its run is gated like a serving arm's, at exit too); tests `aux_discovery_test`, `aux_unitset_test`, `aux_compact_test`, `aux_stager_test`, `r4d_emu_ds41aux`, `ab_window_aux_test.sh` |
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
| `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1` | alone 17.56 / 17.60 / 17.58 against 17.37 / 17.65 / 17.69 tok/s (+0.1%), TTFT −3%, replies identical; those runs used the first gemv_vp kernel, 27-51% slower a call than the plain row (register spills). claude/ds41-vpk2's default kernel (r2cz: the raw rounds' bytes through the scalar unit, one rolled block loop, and a pass of dropped slots alone storing its zeros) is 2.3-5.3x faster a call than the plain row (gate_up 28 / 54 / 73 us against 148 / 169 / 177 at 1 / 2 / 3 live experts; down 20 / 30 / 43 against 105 / 105 / 99), mostly from the dropped-slot skip (5 of a decode call's 6 passes with the live tail off); r2cf, the kernel without the skip, is 0.69-0.77 of plain. Every bit identical (card `check 48` and `check 48 tail`, 0 of 6,087,680 values differ; dropped slots 0 non-zero). `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL=old|r2|r2cf|r2cz` picks a kernel for an A/B (an unknown value refuses every gemv_vp call); not yet measured end to end | the cards' static expert residents in the same packed form (below): 18,001,920 instead of 18,800,640 bytes a unit, so x1.0444 the residents (with F8, 712 / 720 a rank should become ~743 / 752); needs a `libr4d.so` with the row `moe_gemm_ggml_mxfp4_gemv_vp` in the run home, `--deterministic` and passes of at most 16 tokens, or any pass under `..._P1=1`; `..._VRAM_PACKED_RAW=<gate_up>,<down>` (512,320) sets the raw rounds of room; arm part `Vp`, which runs only on a `libr4d.so` the device check passed (`VPACK_CHECK`, below) |
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

The cards hold 712 / 720 static expert residents a rank in block_mxfp4 in the decode-only M16 configuration
(with F8; 598 / 615 without). With `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1` they take the packed form the
RAM cache uses (the codes plane and 2-bit e8m0 codes from one base a weight, raw rounds beside, room for
512 / 320 raw rounds): 18,001,920 bytes a unit instead of 18,800,640, so the same weights pool holds
x1.0444 the residents. From the room each rank left (712 units and 3.90 MiB free on rank 0, 720 and
10.62 MiB on the others) that is 743 / 752, 31 / 32 more a rank, one fewer for each ~10-15 MB that
weights over their raw-round budget (charged block_mxfp4) take back (`core/place/vram_packed.h` has the
whole design). The Lc2048/P1/lent-arena block holds 627 / 635 block or 665 / 674 packed (with `Rb`) at
`--max-model-len 1024` (ab-20261103T090558Z).

- **Selection.** The engine sets the machine fact `device.ds41_vram_packed` before declare, which lets
  libr4d's `moe_gemm_ggml_mxfp4_gemv_vp` row take the routed GEMMs' decode band (priority 59 over the gemv
  row's 58; without the fact the row never matches). The row declares the gemv row's stored form, so the
  pinned pool, the file tier, the RAM cache, the CPU path and the expert profile's checks all keep
  block_mxfp4: only the VRAM copies change.
- **Planning.** A weight is eligible when it is a trunk routed expert's `ggml_mxfp4` gate_up or down,
  whole on its rank, read only by `moe_gemm_ggml` ops every band of which resolved the gemv_vp row, or
  whose bands of up to 16 rows resolved it and every higher band the packed-aware P1 row through its
  4096-row bound (so passes of at most 16 tokens, or any pass under `..._P1=1`). The expert fill charges
  a unit at its packed size after reading its weights' scales from the container (only the units it
  considers for VRAM: 1.1 MB a unit, about 0.8 GB a rank with F8); a weight over its raw-round budget
  stays block_mxfp4 in VRAM and is charged so. Only a static plan (`--deterministic`) packs. The startup
  log prints per rank what was packed and how long the scan took; the planner report
  (`--debug-placement`) has a `vram packed` line.
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

## The placement profile (`--expert-profile`, `profile_blend.py`)

The profile's `decode` counts decide which experts the planner keeps on the cards and, after them, which the RAM
cache holds and which stay on the drive (192 GiB of cache hold about nine tenths of the off-card experts on the
256 GB box). Prompts of different kinds route to different experts, so a profile recorded on one family places
another badly. `profile_blend.py show` on the sealed 721-token profile and on captures of three prompt sets, the
share of each set's decode selections on a profile's hottest 715 units a rank (a card's residents in the current
plan):

| selections of \ placed by | sealed | `broad6414` | the set's own capture |
|---|---|---|---|
| chat, 16 prompts | 44.5% | 50.6% | 64.8% |
| whole-file edits, 8 prompts | 29.9% | 39.8% | 80.3% |
| agent harness, 13 requests | 49.0% | 51.0% | 75.7% |

`broad6414` is the sealed profile x6 (721 tokens) + 8 chat prompts x4 (422) + 8 code prompts x1 (1,939) + the long
prompts x4 (75). Measured against the sealed profile on the same build, every reply the base engine's (placement
moves residency, never a byte):

| set | sealed | `broad6414` | |
|---|---|---|---|
| code, 8 streams, 16 prompts x 256 tokens (total tok/s) | 34.69, 35.77 | 42.84, 43.44 | +22.5%; the eight prompts not in the profile +16.9% |
| whole-file edits, 8 streams, none in the profile | 23.32, 22.81 | 27.12, 27.04 | +17.4% |
| chat, 8 streams, 16 prompts x 64 tokens | 30.93, 31.15 | 35.31, 34.71 | +12.8%; the eight prompts not in the profile +3.6% |
| agent harness, one stream, none in the profile | decode 25.30 tok/s, TTFT sum 64.5 s | 27.08 tok/s, 49.7 s | decode +7.0%, TTFT -23%, wall -16% |
| held-out, 8 streams (the sealed profile's own families) | 33.90, 32.54 | 34.11, 33.93 | no loss |

(Windows ab-20261110T131419Z and ab-20261110T152719Z, four RX 9070 XT, 192 GiB cache, 256 GB DDR4-2666.) The
harness's prefill gains because the profile also picks the drive-only units: its card-path chunks read 2,537-2,711
units a rank from the drive under the sealed profile and 1,231-1,380 under `broad6414`.

Prompts that are in no profile and in no capture (ab-20261110T202631Z, 8 streams, total tok/s, every reply the
base engine's), with `agent2` = `broad6414` + the edit capture + 2 x the harness capture (13,116 tokens):

| unseen set | sealed | `broad6414` | `agent2` |
|---|---|---|---|
| whole-file edits (Rust, Java, TypeScript, Python), 8 prompts x 384 tokens | 24.04 | 27.43 (+14.1%) | 32.55 (+35.4%) |
| code, 16 prompts x 256 tokens | 36.92 | 44.14 (+19.6%) | 48.07 (+30.2%) |
| chat, 16 prompts x 64 tokens | 31.20 | 34.26 (+9.8%) | 31.50 (+1.0%) |

The agent harness under `agent2`, whose capture is in that profile: decode 25.34 -> 29.57 tok/s (+16.7%), TTFT sum
65.1 -> 47.8 s, wall 114.2 -> 90.0 s. With the fifth card's online unit set (below) the unseen chat set gives 41.19
tok/s under the sealed profile and 45.63 under `broad6414`.

**A profile for your own traffic.** Serve representative requests with `--expert-stats FILE` (no device work; the
file is written every 30 s and at shutdown; `ab_window.sh`'s `Xst` part does it for an arm), then

```
profile_blend.py show  SEALED.json CAPTURE.json ...          # tokens of each, and the overlap table above
profile_blend.py write OUT.json 6:SEALED.json 2:CAPTURE.json ...
radiance ... --expert-profile OUT.json                       # ab_window.sh: part Xq<name>, EXTRA_PROFILES/<name>.json
```

The first file is a sealed decode profile of this container (its source identity is kept); every file's decode
counts are taken times its whole-number weight. Weigh families by what will be served, not by how many tokens a
capture happens to hold, and keep the sealed profile in the mix: the prompts a profile was recorded on gain more
than new prompts of the same kind. A capture must be pure decode steps (no mixed steps, no draft passes) of the
same container; `write` refuses anything else and checks its result against the engine's parser rules.

## The aux expert worker (a fifth card)

`RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB>` (experimental, opt-in, off by default; `core/place/aux_expert_worker.h`,
`tools/ds41-host-experts/ds41_aux.h`) is for a box with a card of the rank cards' architecture beyond the `--tp` rank
cards (gfx1201 here: an R9700, or another RX 9070 XT). That card holds the hottest units the CPU path computes, in the packed
RAM cache's form, and computes their rows in routed layer passes of up to 16 rows. At each such pass a rank leases the
worker's units among the experts it hands the CPU, compacts the pass's sorted order to them and queues one job on its
own queue on the extra card: the token rows and that order in a pinned record mapped to that card only, libr4d's
decode-band kernels as the rank cards run them (gemv_vp gate_up, swiglu_clamped, gemv_vp down), the rows back into
pinned memory. The CPU pool computes every other unit as before, and the rank places the worker's rows in the CPU
path's output before the unchanged copy to its own card. Taller passes and drafter layers are declined. With the switch
unset, empty or 0 nothing of it runs and no other variable is read; on a box without such a card it turns itself off
and the server runs as with four cards.

**Exact.** The worker computes only rows the CPU path computes, with the code object the rank cards run, on weights
that are the RAM cache's packed slots, copied and read back identical. Each output value depends only on its weight
row, its activation row and K, so a launch over the compacted order gives the full launch's bytes, which are the CPU's
(the card check `aux` holds both, and gemv_vp to the plain gemv, on the extra card and on a rank card). A job not done by
its deadline, or failed, has its units computed by the CPU from the leases kept for it, so the output holds the CPU
path's bytes whatever the timing.

**Switches** (with the first on, a malformed value of any of them refuses startup):
- `RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB>` (1024-1048576): the worker's VRAM on the extra card (layer tables, the
  lanes' workspaces and unit slots), reserved at startup and held, so a tenant arriving during the load cannot take it.
  Window F1 ran 26000 on a 32 GB card.
- `..._AUX_DEVICE=<BDF or GPU-UUID>` (empty: by capability) restricts discovery to that card.
- `..._AUX_POLICY=verify` (default `serve`): the CPU computes every row as without the worker, the worker computes its
  units too, and the rows are compared (shutdown: `verify: V rows compared, D differ`).
- `..._AUX_DEADLINE_US` (2000, 100-1000000): a job still pending at max(its submit + D, the pool's end + 50 us) is late,
  and its rows are the CPU's.
- `..._AUX_HEADROOM_MIB` (1024, 256-65536): free VRAM the extra card must keep beside the reservation, checked before and
  after allocating it and after the warm-up.
- `..._AUX_BACKOFF` (64, 0-65536): layer jobs a lane declines after a deadline miss, once the late job completed.
- `..._AUX_FORCE_LATE=N` (0): every Nth job of a lane abandoned as if late, to exercise the fallback on real data.
- It needs `..._HOST_EXPERTS` with `..._POLICY=all`, `..._FILE_BUFFER_UNITS`, `..._PACKED_CACHE=1`, `..._VRAM_PACKED=1`
  and a static plan (`--deterministic`), all in the recommended block, and is refused with the reason under
  `RADIANCE_EXPERIMENTAL_SCHED_TBO=1`, `..._ROUTE_TRACE` or a dynamic plan.

**Discovery** (once, right after the devices are chosen; the decision is `ds41aux::decide`). The candidates are the
visible devices beyond the `--tp` ranks' cards, so list the rank cards first in `ROCR_VISIBLE_DEVICES` and the extra
card after them; with none it is off. Rank cards of mixed products while a card of their architecture is left over turn
it off with a warning (the accident where the extra card became rank 0). `..._AUX_DEVICE` keeps only its card (off when
it names a rank's card or no card). A candidate needs the ranks' ISA, must not be integrated and must have cap + headroom
free; among those the worker takes one whose PCI root complex no rank card shares, then the one with more free VRAM,
then the lowest BDF. No product name, ordinal, BDF or memory size is built in; the startup line names the chosen card,
its root port and every rank card sharing its complex. From the moment the switch parses on, pinned host memory maps to
the rank cards only (a pin set), so a card left out maps nothing and the extra card's mappings are its own; a fault on
it is scoped to it (its lanes retire and the CPU computes their rows), which no window has exercised, and a fault the
driver escalates to a reset still takes the server down. The unit set follows the sealed decode profile's global order
(profile share, then expert, layer and rank), cut into per-rank quotas at what the cap holds; each rank loads its units
from the packed RAM cache on its own thread through its own queue (a unit that fails to load is replaced by the rank's
next), every slot is read back and compared, and a warm-up job on each lane must give the CPU pool's rows before the
ranks bind. Any failure turns it off with the reason (`aux experts: off: <why>`), and the server runs as with four
cards.

**Sharing the extra card.** The engine checks the extra card's free VRAM against cap + headroom at startup and does not
watch its other tenants afterwards. The window harness is stricter: an `Ax` / `Ax0` arm and the aux card checks start
only while the extra card's KFD clients all match `AUX_ALLOW_CLIENTS` (`^whisper`, the box's permanent tenant), together
hold at most `AUX_FOREIGN_MAX_MIB` (6144) and leave cap + headroom + 512 MiB free (`AUX_CHECK_FREE_MIB`, 6144, for the
checks), and a client arriving during an arm sets its results aside. So BoltzGen, which uses the same card, must be
idle: any other client leaves the worker off in the harness. Serving beside other compute tenants needs a sharing policy
first.

**Window parts and card checks.** `Ax` / `AxN` (the cap `AUX_MIB`, 26000, or N MiB), `Axv` / `AxvN` (verify), `Ax0` (the
extra card visible, the switch 0: a fifth visible card's own cost) and `AxfN` (beside `Ax` / `Axv`: every Nth job forced
late, 4, with the backoff 0). The harness finds the extra card by capability in the KFD topology (`aux_find`;
`AUX_UUID` / `AUX_BDF` pin it) and appends it to the window's mask of the four rank cards' UUIDs, so ranks 0-3 stay the
rank cards. `DEVICE_CHECKS=aux`, with `AUX_CHECK` (`ds41_mxfp4_device_check`, `TALL_CHECK` by default) and `VPACK_CHECK`,
runs after production stopped: `aux` on card 0 and on the extra card (compacted gemv_vp and gemv rows against the full
launch's at M1/M4/M8/M16; the two cards' outputs must be byte-identical), on the extra card `swiglu` (its SiLU table
recaptured and compared with `SILU_TABLE`) and `vpack check 48`, then `auxrt` (a job's round trip through HIP, logged
only). `Ax` / `Axv` arms run only on the libr4d those passed on, and a serve `Ax` arm only after an `Axv` arm of the
window measured on it. Every arm states the six switches and the startup line must state them back; at exit every rank
needs layer jobs and 0 launch failures (`Axv`: 0 rows differ; `AxfN`: a forced miss); the extra card's free VRAM is
sampled every 2 s (`AUX MARGIN LOW` under 1024 MiB). An `Ax` part runs beside `Kld` only on the 16-token base (no `Lc`,
`BT` at most 16): KL mode is prefill only, and taller passes would leave the worker no row.

**Window F1** (ab-20261104T051616Z, one window): four RX 9070 XT and an R9700 (32 GB, gfx1201; its root complex shared
with rank 3's card), the recommended block (the configuration above with `Lc2048`, `P1`, the lent arena, `Rb`, `Vp` and
`Cs`; batch admission and `L1024` `Pmr4` / `Pmr8` at 4 and 8 streams), the 8 held-out prompts; the worker at 26000 MiB,
deadline 2000 us, backoff 64, headroom 1024 MiB.
- Card checks passed on card 0 and on the R9700: compacted gemv / gemv_vp rows equal to the full launch's at M1-16
  (80 cases on each card) and the two cards' outputs byte-identical; on the R9700 the SiLU table recaptured identical to
  the rank cards' and `vpack check 48` passed. `auxrt`: median round trips of 134-906 us for 1-32 rows.
- The worker held 1,513 units (25.37 GiB: rank 0 396, rank 1 348, rank 2 377, rank 3 392), 39% of the CPU path's
  calibration mass (profile share 5.13 of 13.20), loaded in 2.0-2.1 s from the packed RAM cache and read back identical,
  beside whisper-server's 4.24 GB; at least 2,208 MiB stayed free on the R9700 in every arm.
- Exact: the verify arm compared every worker row with the CPU's (3,288-3,753 rows a rank) and none differed on any
  rank; every reply equalled the base engine's (104 of 104: the 8 prompts in 13 arms, the forced-lateness arm
  included); the prefill hashes at 8 streams equalled the in-window reference without the worker (8 of 8); Kld on the
  16-token base was identical at 3,397 of 3,397 positions (the worker computed 15-19% of the CPU path's rows there).
- Speed, in total (N times the mean of the requests' `predicted_per_second`):

  | streams | without | with the worker | |
  |---|---|---|---|
  | 8, A-B-B-A | 46.40 and 46.87 tok/s; wave 14.28 and 14.17 s | 57.57 and 58.19 tok/s; wave 11.91 and 11.95 s | +24.1% |
  | 4 | 43.59 | 51.46 | +18.1% (one pair) |
  | 1 | 28.30 | 33.92 | +19.9% (the arm without was the window's first) |

  At 8 streams the worker served 28-31% of the rows the CPU path met in passes of up to 16 rows, with mean round trips
  of 1.3-2.0 ms a job (0.73-0.99 ms at 4 streams, 0.23-0.25 ms at one) hidden behind the CPU's own rows: the ranks
  waited 34-39 ms beyond them in a wave of about 12 s. Deadline misses: 1 and 0 at 8 streams, 4 at 4 and 7 at one
  (1-6 rows a rank recomputed on the CPU); 0 launch failures. With prefill hashes on, 8 streams gave 57.32 against
  46.20 tok/s.
- The R9700 visible with the worker off (`Ax0`): 43.14 against 43.59 tok/s at 4 streams (noise); serving after 78 s
  against 76 s; the R9700's used VRAM +4.4 MB.
- Every 4th job forced late (`Axf`, backoff 0): 584-648 forced misses a rank, 813-954 rows a rank recomputed on the CPU,
  replies identical, 49.58 tok/s at 4 streams.

**Window F2** (ab-20261104T064211Z, the stable build of this branch, libr4d 92bd7719 with its own `DEVICE_CHECKS=aux` pass
and an `Axv` arm, 0 rows differ): every reply identical with the worker on and off (8 held-out = the base engine's, 16
chat prompts, 13 harness turns).
- **Chat prompts** (the 16 of `Uchat`, 8 streams with batch admission, A-B-B-A): 38.90 and 38.39 tok/s without, 45.82
  and 46.48 with the worker (+19.4%); wave 30.79 and 29.82 s against 27.26 and 25.77 s.
- **Harness turns** (the 13 of `Uharness` with `Pc-Pce-Pcg16` at `Lm32768`, one stream, A-B-B-A): decode 25.31 and
  26.08 tok/s without, 29.42 and 29.55 with it (+14.7%); summed time to first token 76.35 / 72.20 against 71.96 /
  72.89 s (unchanged: prefill passes of more than 16 rows stay on their usual path); all 13 turns 125.38 / 119.74
  against 114.00 / 114.65 s.

Caveats: two windows; the unit set is static, from the decode profile, so traffic that routes differently gains less
(chat +19% against +24% on the held-out prompts at 8 streams); BoltzGen idle (the extra card had no compute client
but whisper-server).

Startup lines `aux experts: device D (BDF, name, arch): ...` and `aux experts: on: N units ...` (or
`aux experts: off: <why>`); shutdown lines `aux experts: rank R: ...` (layer jobs, units and rows on the extra card of
the rows the CPU path met, declines by reason, round trip mean and max, deadline misses with the forced ones and the
rows the CPU recomputed, launch failures, verify counts) and
`aux experts: device D (BDF): N units held; ...; lanes drained at exit`. Tests: `aux_discovery_test`, `aux_unitset_test`,
`aux_compact_test`, `aux_stager_test` (switch-off identity, the deadline fallback, the pin set on the host backend),
`r4d_emu_ds41aux` and `r4d_emu_ds41aux_tail` (the compacted launches on the emulator), `ab_window_aux_test.sh`.

### Sharing the card, and the online unit set (experimental, not yet measured on a card)

Branch `claude/ds41-aux-dynamic` (work/aux-dyn/SPEC.md). Both are off unless asked for; unset, the worker above is
unchanged. They share one control thread (core/place/aux_control.h), one slot pool and one step-boundary hand-off.

**Use the card only when it is free.** `RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE=allow` (the box's policy: serve while every
other process on the card matches `..._SHARE_ALLOW`, default `^whisper`, and together they hold at most
`..._SHARE_FOREIGN_MIB`, 6144) or `exclusive` (no other process at all). The server watches the card's KFD clients and
VRAM counters through sysfs every `..._SHARE_POLL_MS` (100): a process the policy refuses releases the card at the first
poll that sees it (any process holding 1 MiB there, `..._SHARE_MIN_MIB`, is a client: ROCr's own setup takes ~4.4 MB),
the card's free VRAM under `..._SHARE_FLOOR_MIB` (1024) for 2 polls does too. A release shuts the ranks out of the card
within one layer (their layers go to the CPU path, as declined layers do), waits for every job on the card, and frees
the 26 GB arena; the replies are the CPU path's throughout. Once the card has been free for `..._SHARE_REACQUIRE_S`
(60; doubled after a release within 10 minutes of an acquire, up to 16x) the server reserves the arena again, warms
each lane against the CPU's rows and refills the unit set at `..._SHARE_FILL_MBPS` (3000), serving from the first unit
of each rank. A server that finds the card busy at startup starts released and acquires later.

**Manual control.** `RADIANCE_EXPERIMENTAL_DS41_AUX_CONTROL=<path>` (works with any policy, `always` too): write `off`,
`on` or `auto` into `<path>` (best through a temp file and `mv`); `<path>.state` says what the server does
(`starting`, `serving`, `draining`, `released`, `loading`, `dead freed`, `dead held <MiB>`, `stopped`, with the time,
the server's pid and start time, the mode and why). For a job that needs the card's memory:

```
tools/ds41-host-experts/aux_yield.sh <path> [timeout s, 60] -- python3 train_fig.py ...
```

holds the card (a flocked holder under `<path>.d/`) for the command's life, starts the command only once the state file
says the arena is freed (`released`, `dead freed`, `stopped`, or no server), and refuses (exit 3, 4) when the card stays
held (`dead held`: until the server restarts) or the timeout passes. Without it, a job that allocates more than the
card's free VRAM within the release's fraction of a second can fail its first allocation.

**The online unit set.** `RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE=1`: the set starts as the sealed one and follows the
traffic: each rank counts the rows its passes of at most 16 rows route to each candidate unit, the control thread keeps
decayed scores (half-life 512 tokens, the sealed profile as a 256-token prior) and swaps up to 2 units a token
(`..._ONLINE_RATE` 200 per 100 tokens) within each rank's quota, copying each new unit from its RAM-cache slot into one
of 16 spare slots on a copy queue of its own (read back on a sample), flipping both units at the next step boundary.
Replay of the Oct 29 traces under the engine's exact rule (work/aux-dyn/replay/q5_engine_rule.py; plan L1024, 1,513
slots): absorption of CPU-path decode rows 40.1 / 41.9% held-out / chat at 8 streams against the static set's 30.1 /
21.6%, 36.1 / 38.0% at one stream. Expected decode gain (the measured elasticity, net of the copies): +4..+8%
held-out, +17..+21% chat at 8 streams. `..._ONLINE_DRY=1` runs the copies without flipping (their cost alone).

Startup lines `aux experts: sharing: policy ...`, `aux experts: online set: ...` (or `aux experts: released at start:
<why>; ...`); transitions `aux experts: releasing (...)`, `released in T ms (...)`, `acquiring (...)`,
`serving again in T s`, `full set in T s`; shutdown lines `aux experts: sharing: R releases (...), A acquires ...` and
`aux experts: online: D decisions, S swaps (...), R read back (0 differ), ...`. Harness parts `AxS` / `AxSx`, `AxO` /
`AxON` / `AxOd`, `AxT` / `AxTN` (the control file toggled every N/2 s), `AxF` / `AxFN` (a simulated tenant, aux_tenant).
Tests: `aux_share_test`, `aux_online_test` (the C++ planner against the replay, swap for swap), `aux_fence_test` (the
control thread, worker and lanes on the host backend: jobs never read a slot or table entry being rewritten, through
swaps, releases and reacquires), the `aux_stager_test` additions, `ab_window_aux_test.sh` §11; the card check
`ds41_mxfp4_device_check ... auxswap` (window only).

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

### Harness-grade prefix reuse

An agent harness sends the whole conversation every turn: the system prompt, every turn so far
with the model's reply sent back, and the new tool output. Cold, every turn pays prefill on all of
it (31-45 tokens/s here). The prefix cache above resumes a turn at the previous *prompt's* last
snapshot, so it still prefills the previous reply and whatever follows the last chunk start. The
switches below (claude/ds41-eors) make a turn prefill only what it added, exactly.

**End-of-request snapshots** (`RADIANCE_EXPERIMENTAL_PREFIX_EORS=1`, part `Pce`). A decode step
whose row completes a position on the resume grid past its prompt snapshots the state there
(reserved when the step is planned, saved after it, as a prefill chunk's is), a prompt ending on the
grid snapshots its end, and every prompt its tail (the start of its last chunk) however short. So
for a request that computed n positions (its prompt and every output token but the last) the
resumable positions are the grid positions g with prompt < g <= n, the prompt's end when on the
grid, its tail, and the prompt ladder below. Under the consumer skip (`Cs`, in every core arm) a
prefill writes no snapshot below its prompt's cut target, where the last layers' windows would be
incomplete; the target is the deepest snapshot its idle prefill writes, the prompt's end when on the
grid and its tail otherwise. So there the resumable positions are the target and the grid positions
past the prompt (decode rows run every layer): not the prompt ladder below the target, nor the tail
of a prompt ending on the grid. Either way the next turn resumes at grid_floor(n), at most 15 tokens
before the end of the reply at `--max-num-batched-tokens 16` (31 under `Xt-BT32`, whose grid is 32).
A step that carries a decode snapshot is planned after its predecessor's commit, not issued ahead
(its key hashes the token that step samples): one host turnaround every 16 tokens, plus a 5.65 MiB
copy a rank (~0.3 ms). The trim after a publish runs before the request frees its blocks, which are
safe from any trim then; with `Pce` it also keeps the few entries the request recomputed rather than
adopted (between the snapshot it resumed from and the end of its hit: up to seven 2-token entries),
which a plain trim takes once the live sequences hold more than three quarters of the pool, cutting
the chain below the new snapshot (`trim_spared` counts them). An idle conversation has no such
protection: other sessions' trims and admissions take it least recently used first, its end first,
and its end-of-request snapshot with it; the host tier (`PcmN`) is what keeps its KV under pressure.
Refused, with the reason in the log, without `--prefix-resume exact`, with speculation, with `Pcx`
(which keeps out the decode blocks such a snapshot stands on), on a model with no state to snapshot,
and on one whose recurrent state is scanned in chunks (a gated delta net: qwen35, QFN), where a
decoded row is not the row a prefill computes. A model whose only state is a sliding window takes it
only with `RADIANCE_EXPERIMENTAL_PREFIX_EORS_WINDOW=1` (nothing yet shows (A) below for one).
DeepSeek-V4.1 is neither: its linear state (the compressor's tail, the Engram tokens) has a state
chunk of 1, and its window rides in the snapshot beside it. Counters: `/stats` `eors` (reserved,
committed, ahead_withheld, hits, hit_tokens, hit_deepest) and `snapshot_store`, and the shutdown
line `end-of-request snapshots: N reserved, M stored (X written, ...), K step(s) not issued ahead
for one; H admission(s) resumed from one, T tokens reused (deepest D); snapshot store: ...`.

Why it is exact (the full argument is at `Scheduler::init`, "end-of-request snapshots"): the next
turn resumes on the grid, so from there it computes the cold run's own chunks from the same state;
that state is the cold run's because every row below it is, by induction, given (A) a row computed
alone (decode) has the bits of the same row in a prefill pass of up to `--max-num-batched-tokens`
rows, (B) deterministic arithmetic, (C) a snapshot restoring the whole per-sequence state on every
rank. (A) is what the exact-tall device sweep checks op by op (each row against the same row alone
and in launches of 16, up to the reach), and what the `Kh` arms below check end to end: the prefill
hashes of every KV group and the last row's logits at the end of each prompt of the resumed turns
must equal the cold arm's. On libref the e2e test
`a_turn_resumed_at_an_end_of_request_snapshot_returns_what_a_cold_run_does` holds it for LFM2 (a
convolution's state: with the switch the next turn resumes at the reply's last grid position, 192,
without it at nothing) and Gemma 4 (no state to snapshot: the switch is refused and the hit adopts
the decode-written blocks at 192 either way); every reply and probability is a cold server's.
qwen35's gated delta net decodes by another recurrence than its chunked prefill, so (A) fails
there, as it would on any such model: with the switch taken, a turn resumed at its decode snapshot
gave other probabilities than a cold run (review of 30 Oct). The switch is now refused there, and
the same e2e case holds the refusal: qwen35's next turn, with the switch asked or not, resumes
nowhere and answers as a cold server. On the cards the first evidence came from today's cache:
in window ab-20261029T232708Z `Pc-Pci256` on `Upfxchat` answered 15/15 as the cold arm, C3 among
them, which adopts 46 of C1's decode-step blocks beneath C2's recompute snapshot (token level; the
`Kh` arms are the byte level). Under the consumer skip (`Cs`) the snapshots are at or past each
prompt's cut target and decode rows run every layer; under `Xt` (A) is the tall checks, but no
window has yet shown `Pce` with `Xt` end to end, so `ab_window.sh` skips an arm with both; the
window variable `PCE_XT=1` runs them for that evidence (a `Kh` pair at `Xt-BT32`, cold and `Pce`,
whose `--hashes` must agree).

**The snapshot store for harness sessions.** `--checkpoint-slots auto` sizes the store for the
running sequences (7 slots of 22.6 MiB at `Lm32768`, interval 2048, one sequence: one session's
ladder). A harness keeps more conversations than it runs: `RADIANCE_EXPERIMENTAL_CHECKPOINT_STORE_MIB=N`
(part `PcbN`) sizes it as a byte budget instead (`Pcb1024`: 45 slots, about six sessions at 7;
QFN's production store is 40 slots of 113 MiB, 4.42 GiB). `RADIANCE_EXPERIMENTAL_PREFIX_SESSION_LRU=1`
(part `Pcl`) makes a full store give up the least recently used conversation's snapshots, its
shallowest first and its tip last, before any of a conversation used since (otherwise the least
recently used snapshot of any conversation goes, which can be a busy one's anchor). The KV blocks of
idle conversations can move to the host tier, `--prefix-cache-mib N` (part `PcmN`), restored over
PCIe on the next turn; with `RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED=1` (part `Pcr`, a deepseek41
plugin from this branch) the compressed and index caches are declared replicated, which they are
(every rank's `Kh` hashes of them agree), so a host-tier entry holds one copy, 3,200 B a token
instead of 12,800 at TP4.

Host memory, measured: with the 96 GiB RAM cache pregrown (`m98304-G`) MemAvailable runs at 12.0-12.7
GiB at its lowest during serving (12.6-16.8 GiB mean; the 29 Oct 20:09 and 22:09 windows' `.mem`
files, QFN stopped). Under `Pc` startup's checks keep 8.02 GiB free after the load ("host headroom:
... the RAM cache's slabs already hold 95.98 GiB of its 96.00 GiB budget, so the checks from here on
keep 8.02 GiB free": `--host-headroom-mib` 8192 and the 16 MiB the cache has yet to grow), so about
4-5 GiB can still be pinned; the auto store at `Lm8192` took 5 slots (113 MiB) there. `Pcb1024`
(45 slots) alone, or `Pcb512` with `Pcm2048`, keeps MemAvailable above 9 GiB, well clear of the
window's memory guard (3 GiB); more needs a smaller `--host-headroom-mib`.

**Long contexts.** The KV a token costs a card: `kv_index_r2` 384 B, `kv_compressed_r2` 1,536 B,
`kv_index_r1` 256 B, `kv_compressed_r1` 1,024 B, 3,200 B in all (the startup table: 32.70 MiB at
`Lm8448`, 19.42 MiB at `Lm4096`), plus the window's 6.88 MiB a sequence. `--max-model-len` also
sizes two persistent rope tables on every card, `rope_window` and `rope_compressed`, [max_ctx, 64]
f32 each (`deepseek41.cpp` `buffers`): 512 B more a context token, 2 MiB at `Lm4096`, 16 MiB at
`Lm32768`, 32 MiB at `Lm65536`; and the indexer's score plane in the step arena, [16, max_ctx] f32
at `M16`, up to 64 B more. The VRAM split gives the pool one `--max-model-len` sequence and the
experts what is left, so against `Lm4096` a card gives up 101.5 MiB at `Lm32768` (87.5 MiB more
pool, 107 MiB in all, and 14 MiB more rope tables) and 217.5 MiB at `Lm65536` (187.5 MiB more pool,
207 MiB in all, and 30), plus at most 2 and 4 MiB of score plane: about 5.7 expert units of 17.9 MiB
a card (23 over the four) and 12.2 (49). By the sealed decode profile the units given up are the
coldest of the ~710-719 a card holds, about 9 selections each in 721 decode tokens: 0.28 (32K) to
0.61 (64K) of a token's 240 expert selections move from the card to the CPU path, about 0.35-0.75%
more CPU work, so an estimated 0.25-0.6% of decode (measure it: `Lm4096` against `Lm32768`/`Lm65536`
on the held-out prompts). The conversations kept idle must fit three quarters of the pool together
to keep their chains whole (each publish trims the pool to a quarter free, the least recently used
conversation's end first), or live in the host tier; one larger than that alone stays whole through
its own turns with `Pce`, not through another session's. Parts: `Lm32768`, `Lm65536` (the `LmN`
part). Measured up to 524,288 since: **For contexts beyond 32K** under Configuration.

**The harness receipt** (`make_prefix_receipt.py --harness`, part `Uharness`; `Uharness1` is its
first session alone): two agent sessions interleaved turn by turn, each turn the whole
conversation with the model's own replies sent back (up to 128 tokens each) and a system prompt
(`chat_system`, right after BOS): hxA, a code-review agent (2.9K-token system prompt, five tool
outputs of repository source, 7 turns, 9.4K tokens at the last), and hxB, a licence agent (2.3K
system prompt, two MIT licences, consumer_skip.h and geometry.cpp, 6 turns, 9.8K). Cold: about 79K
prompt tokens over the 13 turns; with `Pce` about 19K. Run it at `Lm32768` (both sessions under
three quarters of the pool). `ab_requests.py --ttft` prints per turn the TTFT, the tokens the cache
supplied and the text against the first (cold) arm, then per session the sums and the cached share.

**Reading a `Pce` window.** Two results gate it, both printed at the end of the window:
`ab_requests.py --hashes` must report every prompt identical against the cold arm (the prefill
hashes of every KV group and the last row's logits at the end of each prompt: 13 of 13 on
`Uharness`), and `--ttft` must show every text `same`. The rest is advisory, because it moves with
the replies' lengths and with how each reply re-tokenizes when it is sent back: the shutdown line's
`H admission(s) resumed from one` (11 on `Uharness` when every later turn reaches its predecessor's
last snapshot) and, in `--ttft`, each later turn's cached tokens equal to its predecessor's last grid
position (the grid floor of that request's prompt and reply, less its last token). A reply whose text
re-tokenizes to other ids than the generated ones (the `sent back` lines list them; on 29 Oct the
23:27 window had a reply come back as 40 tokens for 41 generated) parts the chain early, and that
turn resumes, still exactly, at the deepest snapshot before the parting, with fewer tokens cached.
A hash that differs is (A) failing for some op on the cards: if the `Pc` arm without `Pce` (which
adopts decode-written cache blocks too, from the third turn on) agrees, suspect the decode-written
window or compressor-tail state in the snapshot; if it differs too, the decode-written blocks.

## Correctness

Each row the CPU computes uses the card's arithmetic: the same lane and block order, butterfly,
BF16 rounding and no fused multiply-add. A job of several token rows decodes each round of codes
once for all of them (`gemv_avx2_rows`). Each row's sums still see the same operations in the same
order, so the bits do not change, and an extra row costs 0.37–0.46 of a row instead of 0.52–0.62
(40 threads, experts streamed from DRAM). `POLICY=verify` compared 31,670 rows with the card's (and
28,413 with the plane mirror's bytes) with no difference. Every A/B arm's replies are compared with
the base engine's; the final configuration also matched it on 16 chat-framed prompts, on the eight
prompts three times on one server, and on 256-token replies.

Since the upstream sync (upstream main `b6acbd69`, v1.15.0 + 12, merged on `claude/ds41-sync`) the base engine
itself rounds differently: upstream's standalone rmsnorm sums over a fixed 256-lane partition, so decode steps and
chunks of up to ~72 rows round ds41's 5120/1280/512-wide norms differently from before. The reference is the
merged base engine: its held-out replies (`ab-20261031T104524Z/base.json`) differ from the pre-sync base's in 3 of 8
(mid-reply, equally fluent), and every arm of `ab-20261031T100852Z` (decode-only, recommended prefill settings at
1 / 4 / 8 streams) matched it, 32 of 32. The same plain `Kld` arm agrees with the pre-sync build's record on the
top token at 98.9% of 1,512 positions (KL mean 0.012, median 0.00015). The swiglu and chain checks, the silu
table recapture and a `POLICY=verify` arm (29,668 rows, 0 differ) passed on the merged libr4d. Every plugin must
come from this tree (the loader checks an ABI layout fingerprint), and prompt lookup alone on ds41 needs
`RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1` (upstream refuses `--spec-ngram` beside a declared draft head).

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

`RADIANCE_EXPERIMENTAL_DS41_LCP=C` (C a multiple of 16 in 64..4096) runs a lone prompt's prefill in
chunks of C tokens, each giving every row the bits of the same tokens in 16-token passes. It needs
`--max-num-batched-tokens C`, the CPU path on every routed unit (`..._POLICY=all`, `..._MISSES` ≥ C,
`..._FILE_BUFFER_UNITS`), chunks of C (the default checkpoint interval 2048 serves C up to 2048;
above it, `--checkpoint-interval C`; with the prefix cache and a smaller interval, `--prefill-span
C/I`), and a step a chunk alone; the server refuses anything else at startup with the reason (also
speculation but prompt lookup alone in the decode band, below; KV streaming, `..._FETCH_LIST`,
`..._VRAM_PACKED` without `..._P1=1` and
`..._EXACT_TALL` other than 1), and refuses a libr4d.so whose twin does not reach C (so a run home
with an old plugin cannot mix sums).

Prompt-lookup speculation beside it (and beside `..._P1=1` and `..._CONSUMER_SKIP=1` on top; arm parts
`SgK` / `SgaK`; `core/place/ep_route_counts.h`): taken where no draft head is declared
(`..._LOOKUP_ALONE=1` on a container with the DSpark stages) and a step of decode rows fits the decode
band, `--max-num-seqs` x (1 + `--num-speculative-tokens`) ≤ 16, the rows the routed GEMM's gemv /
gemv_vp and the lm_head's gemv serve. A verify step is then a decode pass of at most 16 rows of the
kernels the band checks already hold to a 16-token pass's bits; it is never cut, never arms the card
path and takes no admission row. Startup says so ("prompt-lookup speculation admitted": rank 0's file
stager, the scheduler, the card path, the consumer skip; the plugin names the consumer suffix it declared
beside it), and at shutdown every rank counts its
verify passes and the passes outside the band, which must be 0 (`ab_window.sh` measures an Sg arm with
an Lc part only with those lines and that count). A draft head (DSpark), more decode rows a step, and
end-of-request snapshots under any speculation stay refused. Measured (ab-20261110T202631Z, one stream,
eight whole-file edits of 250-374 reply tokens, every reply the base engine's, 0 passes outside the band
on every rank): per-stream decode 22.30 tok/s without, 31.25 with `Sg8` (+40%; 3.70 tokens a step, 94.3%
of the drafted tokens accepted, a verify step 119 ms against 45 ms), 33.13 with `Sg15` (+49%; 4.51 tokens a
step, 90.9%); on the held-out prompts, which repeat nothing, `Sg8` drafts twice and changes no reply.
Verify passes have no read-ahead yet.
Under it libr4d's `ds41_mxfp4_device_check LIBR4D lcp [C]` holds the twin, the sparse ops (also as
the suffix launches of the per-row cut) and every other height-dependent op (gemm_nt by the row the
table resolves, the router's f32 projection and its 96-column split, the norms, mhc_mix, the head) to
their rows launched alone and 16 at a time, up to 2048 rows (C above 2048); `r4d_emu_ds41tall_lcp`
holds the twin to 256 tokens, the norms to 1024 rows, gemm_nt's tiled row and the f32 split on the host.

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
with and without `Cs` (identical prefill hashes). Leave `Vp` out of an Lc arm without `P1` (refused:
gemv_vp serves 16 rows; under `P1` the P1 row decodes the packed form). `Rb` arms run only after
`SELFTEST_F32BF16` passed in the window (its tall cases reach M 4096); the `lcp` sweep does not cover
`gemm_nt_f32bf16`.

### Balanced routed gemv (`RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE`)

Window H (`ab-20261030T105535Z`) measured large-chunk prefill exact but 3-4x slower than 16-token
passes: on rank 0 of the `Lc1024-Cs` Ulong arm, 388 s of its ~463 s of prefill was the issuing thread
waiting for the card at the routers' counts (1.19 s a layer-pass, against 4.3 ms a 16-token
layer-pass), 72 s waiting for misses still being read and 2.4 s for the pool. The card's time is the
routed GEMM's twin: the plain gemv kernel gives each live expert the same number of work items, and
`mx_range` hands an EP rank's last local expert every slot routed to the other ranks (`[off[96], T)`:
about 4,600-5,500 of a 1,024-token chunk's 6,144 slots), one weight pass each at gate_up's XLM 1, on
the 4-6 blocks that expert gets. Phase 0's `m3-twin` fits that to the percent: (T − local) × rgpi row
groups at 5.1 us each on a lone block is 1.05 s / 1.24 s / 2.56 s for L25 rank 0 / L25 rank 2 /
uniform at M 1024 (measured 1.09 / 1.25 / 2.59 s of gate_up) and 60 ms at M 64 (61 ms). The same tail
is most of a 16-token pass's card time (about 72 dropped slots a pass).

Under `..._MOE_BALANCE=1` (every plain gemv call: the decode band and the twin) or `=2` (the twin above
16 tokens only) `mx_gemv_kernel` cuts each live expert's row passes into chunks of P passes up to the
live end `off[ne]` (P sized so the chunks fill the grid once) and the passes past it into chunks of 64 P,
and a pass whose slots are all dropped writes its zeros without reading weights. Every pass is the plain
loop's (the same rows from the expert's first row in steps of XLM) and every output one wave's sum in the
same lane and block order, so the bytes are the plain kernel's; `r4d_emu_ds41bal` holds that on the
host (gemv and twin, XLM 1 / 3 / 8, six routings including tail-only and no local slot) and
`r4d_emu_ds41tall_bal` reruns the twin-against-16-token-passes check with the switch on;
`ds41_mxfp4_device_check LIBR4D bal [C] [ROUTES]` holds it on a card at ds41's shapes and times both
(the checker export `r4d_rad_moe_gemm_mxfp4_gemv_bal` forces the switch per call). Not for gemv_vp
(`Vp`), the live tail (`..._LIVE_TAIL`) or the draft bank's one-pass mode.

Arm parts: `Mb1` / `Mb2`; `bal` in `DEVICE_CHECKS` (with `TALL_CHECK`, `TALL_LIBR4D`, `LCP_CHECK_C`, and
optionally `BAL_ROUTES`, a `p0_tables.py routes` file) runs the card check (`devcheck-bal.log`) and gates
the Mb arms on its libr4d.so. `Lf` sets `RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE=1`: at exit each rank logs
`pass profile, rank R, <height>: ...` for passes of 1, 2-16, 17-256 and 257+ tokens -- routed layer-passes,
rows, a layer-pass's wall time (counts to counts), its wait for the card at the counts, for misses at the
gather, for the pool at the gather, misses and read waves a layer-pass.

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

## Card experts in large chunks (P1)

Prefill Phase 1 (`RADIANCE_EXPERIMENTAL_DS41_P1=1` beside `RADIANCE_EXPERIMENTAL_DS41_LCP=<C>`; the
contract with libr4d is `ds41_p1.h`, the plan `PHASE1-PLAN.md` in the review directory, the engine side
`core/place/card_expert_stager.h`). A routed layer of a pure prefill pass that runs on at least the
threshold's rows (`..._P1_MIN_ROWS`, 320; under the consumer skip a deep layer can run on fewer than its
pass) gives no routed expert to the CPU; a shorter layer pass takes the CPU path exactly as the adopted
large-chunk path does (a P1 layer streams nearly every routed expert of the layer to the cards whatever
its rows, which does not pay below ~300 rows):

- **VRAM residents** run in the plugin's own expert ops (gate_up, the clamped SwiGLU, down).
- **Every other touched expert** is published to those ops with the skip mark (`kMarkSkip`, bit 1), so
  the P1 twin row (`moe_gemm_ggml_mxfp4_gemv_tall_p1`) leaves its rows alone.
- **The ring.** Each such expert is copied, on the card's DMA lane, into a slot of a VRAM ring: the file
  stager's two buffers seen as whole 18,800,640-byte slots (`FILE_BUFFER_UNITS=N` gives 2N slots; B6 is
  12 slots, 215 MiB a card). A RAM-cache unit is copied from its pinned slab slot (borrowed read-only,
  packed or block form); a unit the plan pinned in the host pool (a static plan's, `--host-pool-mib`
  256 leaves three a rank) from its pool slot, published skipped for its layer like the others instead of
  being read across the link once a row pass; a drive miss is read into host planes by the kept readers (with the plane
  mirror's prefill share), packed by a packer thread a rank into a pinned ready slot in the packed RAM
  cache's form (or block form where the packer refuses it, or without the packed cache), and copied from
  there.
- **Waves.** Right after the layer's down projection is issued, the stager launches the same three ops
  again in waves of `..._P1_WAVE` ring units (`OpStager::launched` captured their arguments), each with
  its own expert table: the wave's units name their ring slots (`kMarkPacked` where packed), every other
  entry carries `kMarkSkip`. The same kernels and arguments, so the decode band's bits. The SwiGLU
  relaunch covers all of the layer's slots each wave, which is idempotent. The waves run right behind
  the down projection, before any later op is issued, and under the switch the builder stretches the
  lives of every buffer the gate_up and SwiGLU touch (the MoE input rows, the gate_up rows, the sorted
  order) to the down projection, so the arena never gives their bytes to the down projection's output
  or to anything that starts there (`Builder::finish_buffer_liveness`; a packing change only, which
  costs the arena some bytes; off, nothing changes).
- **Order and safety.** Units take slots in the order their copies are queued, and waves launch the
  oldest queued units first; a copy waits on its slot's free event (recorded behind the down projection
  of the wave that read the slot's previous unit), a wave on its units' copy events, and a lease or ready
  slot is let go only after the host saw its copy's event pass.
- **Read-ahead.** After a layer's demand reads, the next layer's likely misses (file-tier units neither in
  the slab nor in the RAM cache, by the sealed profile's share) are read, `..._P1_AHEAD` a rank; the next
  layer takes the ones it selects, the rest are dropped.
- **Admission.** The RAM cache stays read-only during a prompt, except the units the prompt's last rows
  select (`..._LCP_ADMIT`, 128): those hits get the LRU touch, those misses go to the insert threads after
  packing. The sorted order comes with the counts from the addresses the layer's gate_up last read; a
  layer never seen before admits nothing.
- **Chunks** are aligned to the prompt's end (`..._P1_ALIGN=end`): the first is cut so the last is full,
  every chunk ending on the 16-token grid (and the quantum's): 3,086 tokens at C 2048 are 1,040 + 2,046,
  6,817 are 688 + 2,048 + 2,048 + 2,033. The interval's snapshots are written where a chunk ends on the
  interval, and the prompt's tail snapshot as before. Start-aligned under `--prefix-resume exact` and with
  `..._P1_ALIGN=start`, with a startup line saying so.
- **The lent ring** (`..._P1_LEND=K`). Outside the card path's passes the ring's slots past the two the
  file buffers' alias bases name (B1's share) hold the rank's K hottest file-tier units (by the profile)
  as decode residents, published at their slots and computed by the plugin's own ops, never aliased or
  handed to the CPU; B6 with K 10 gives decode B1's residents. A pass that arms the card path takes them
  back first (published as file-tier units again, which the card path or the CPU path takes like any
  other), and its first ring copy waits on their last reader (an event recorded at the end of every pass
  that had residents). `..._P1_GRACE_MS` after the card path's last pass they are reloaded in the
  background, from the RAM cache where it holds the card's block form, else read from the drive and relaid
  into one of two pinned buffers by a reload thread (`Mover::relay_unit_block`), copied into their slots on
  the DMA lane behind the card path's last ring reader, and published at a later pass's start once the host
  has seen the copy pass; until then, or after a failed reload, they take the CPU path. Residency changes
  only at a pass's start, before the weight tables are brought up to date, and a copy a card-path pass
  overtakes is never published. Off by default.
- **The lent arena** (`..._P1_LEND_ARENA=K`; measured with chunks of 1,024, 2,048 and 4,096). The activation arena is sized for the
  chunk and its buffers are live across the whole graph, so it is about the sum of its buffers: 935 MiB a
  card at 2,048 rows, 468 at 1,024 (`expert_down` 60 KiB a row, `expert_gate_up` 54, `engram_kv` 50, the
  two hyper-connection streams 40 each, `expert_ff` 27, ...; all but ~22 MiB of persistent buffers scale
  with the rows). A step of at most 256 rows uses ~13% of it. Under the switch the deepseek41 plugin
  declares again at smaller steps (`RadArchProbe::shape_probe_ok`, said only beside it; the sizing
  declare writes scratch and keeps the real declare's op list), the core binds the 256-row arena level
  (`Engine::probe_arena_levels`, `plan_level`) and hands each rank's file stager the arena's top above
  that level's buffers and the 1 MiB guard; up to K whole slots there hold the next hottest file-tier
  units after the lent ring's, exactly as the lent ring's slots do. A step at the full plan (above 256
  rows: every card-path pass, and the CPU path's passes of 257-319 rows) takes them all back at its
  begin_pass, before its first op; the compute stream first waits on any reload copy still queued
  (`LendHost::fence`), and at its end the mover's stream waits on an event behind it, so a later reload
  lands behind its last reader. The grace runs from the last such step. Residency moves, bytes do not:
  replies stay byte-identical. At Lc2048 the slots that fit a card grow with `--max-model-len`: 41 at 512
  and 1024, 42 at 4096, 44 at 8192 and 53 at 32768 on the stable build, the same with and without `Rb`/`Vp`
  where both ran (ab-20261103T090558Z, T110235Z, T114720Z); before the upstream sync, at 4096, 41-42 at Lc2048
  (the 39 residents Lc2048 costs against Lc1024) and 17 at Lc1024 (ab-20261031T011849Z, T022949Z). At Lc4096
  the cap of 64 fits at 512-8192 (ab-20261103T090558Z). The plugin must come from a tree with the change
  (arch/deepseek41). Off by default.
- **The lent slots follow live decode routing** (`..._P1_LEND_HEAT=S`; not yet measured on the cards). The
  lent ring's and the lent arena's units are chosen once, at startup, by the calibration profile's share,
  and live traffic routes differently. Under the switch each rank keeps, by file-tier unit, a decayed count
  of the decode rows routed to it (`p1::LendHeat`; counted where the file stager learns a routed trunk
  layer's counts, in pure trunk decode passes only: no prefill row, no draft pass; it halves every
  `..._P1_LEND_HEAT_HALF_ROWS` decode rows, 100, by a growing scale, nothing swept). Between passes, at the
  begin of a pass that does not take the slots back and only while the lender is settled (every lent slot
  holds its resident; nothing wanted, in flight, cancelled or failed, the grace over), up to S of the
  coldest residents give their slots to the hottest lendable units off the card whose heat is above
  `..._P1_LEND_HEAT_MARGIN_PCT` percent (150) of the resident's plus half a row, so equal heat never swaps;
  nothing moves before one half-life of rows has been counted. A retarget (`RingLender::retarget`) first has
  the mover's stream wait on the event the compute stream recorded at the end of the pass before (the same
  hand-over an armed pass's first ring copy gets, `Mover::mover_wait`; that pass's end is behind every
  kernel that could read the resident from its slot, and every slot copy goes through that one stream),
  then publishes the resident as a file-tier unit again, as a take-back does, before the pass's weight
  tables are brought up to date, then names the slot's new unit and lets the existing reload path load it,
  copy it and publish it at a later pass's begin; until then both units take the CPU path. It does nothing,
  and counts why, when the pass takes the slots back or they are lent or in the grace, when a slot is busy
  or failed, when a draft bank's loan holds ring residents on the rank, when the event was not recorded at
  the end of the pass right before, or when a chosen unit can no longer hold a slot (not a whole
  gate_up-and-down pair off the card that the CPU path can read, or one the aux expert worker holds). A
  pass that takes the slots back in the middle of a retarget cancels it as it does any reload, and the slot
  comes back with its new unit after the grace. Which experts a row routes to and every unit's bytes are
  unchanged: residency moves, bytes do not. Needs `..._P1_LEND`, `..._P1_LEND_ARENA` or `..._P1_LEND_SLAB`
  (the lent slab, whose slots it governs with the others: "Adaptive residency" below). Off by default (0),
  and then nothing of it exists.
- **Unchanged.** Decode, every pass of up to 16 tokens and every layer pass below the threshold run the
  CPU path as before.

For long prompts on the measured 256 GB host with a 192 GiB expert cache and four RX 9070 XTs,
set `RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD=0` alongside the P1 configuration above. The independent
stable-build check on 3 November used fresh Qa8/Qa0 hash arms, then quiet Qa8/Qa0/Qa8 controls:
3,086-token TTFT fell from 8.71–8.92 s to 7.65 s (12–14%), and 6,817-token TTFT from 11.83–11.85 s
to 10.82 s (8.5–8.7%). All 15 replies and all six full prefill hash checks matched. This disables
speculative P1 reads; demand reads, weights and arithmetic stay unchanged. The 82-token prompt was
1.8–3.5% slower, so the existing default remains 8. These measurements cover the stated cache and
no-prefix long-prompt profile; other cache budgets and broader lifecycle tests have separate scopes.
The raw result is `work/autonomous-20261101/stable_e184_Qa_B/summary.json`; the report is
`PREFILL-QA0-20261103.md` in the review archive. No engine or plugin rebuild is needed for this
existing switch.

Switches (each a whole number; anything else refuses to start): `..._P1_WAVE` (2; 1-16),
`..._P1_INFLIGHT` (drive units read at once a rank, 16), `..._P1_READY` (pinned ready slots a rank, 6,
18.8 MB each), `..._P1_AHEAD` (8; 0 off), `..._P1_VERIFY` (0: after each layer's waves, up to N
expert-rows of its RAM-cache units are recomputed by the CPU pool from the cache's bytes, borrowed again
read-only, and compared with the card's down rows; the pass waits for the readback, so it is for exactness
arms only; drive units are not sampled, their pinned slot being reused by then; the stats line counts rows
compared and differing, and samples skipped because the unit was evicted since its copy),
`..._P1_ALIGN` (`end` or `start`), `..._P1_MIN_ROWS` (the threshold, 320; 17-4096: 17 is the card path
for every pass above 16 tokens, as before the threshold), `..._P1_LEND` (lent ring slots, 0 off; up to
64, at most the ring's slots less two), `..._P1_LEND_ARENA` (lent arena slots, 0 off; up to 64, as many
as fit), `..._P1_GRACE_MS` (1000; 0-60000), `..._P1_LEND_HEAT` (lent slots a rank handed to a hotter unit a
pass, 0 off; 1-8), `..._P1_LEND_HEAT_HALF_ROWS` (the heat's half-life in decode rows, 100, or 2000 beside the
lent slab; 1-100000000), `..._P1_LEND_HEAT_MARGIN_PCT` (the hysteresis, 150; 100-1000), `..._P1_LEND_SLAB`
(the lent slab's slots a card, 0 off; 1-256) and `..._P1_LEND_AUDIT` (the lent slots' read-back audit, 0 off;
1): "Adaptive residency" below.

Refused at startup, with the reason: an invalid switch value; P1 without an admitted LCP chunk; a
routed band of 17 rows or more not on the P1 row (a libr4d with LCP's twin is named as such); a
band of up to 16 rows not on the gemv or gemv_vp; `FILE_BUFFER_UNITS` unset or below 2 (fewer than 4 ring
slots); the CPU path not taking the layer passes below the threshold (host experts with `..._POLICY=all`,
`..._MISSES` of min(C, threshold - 1) or more: 319 at the default, 16 at 17); speculation other than the
prompt lookup large-chunk prefill took (lookup alone, at most 16 decode rows a step; so no DSpark
bank serves beside P1 or its lent ring); `..._FETCH_LIST`; the route trace; the DSpark head tail; a routed
layer whose gate_up, single `swiglu_clamped` and down cannot be named; a routed unit off the card neither on
the file tier nor pinned in the host pool of a static plan (`--deterministic`) as a block_mxfp4 gate_up and
down pair; `..._P1_LEND` with no slot or unit to lend; `..._P1_LEND_ARENA` with no arena level (a plugin
without the change), a dynamic plan, or no slot or unit to lend; with either the CPU path not taking every
pass that does not arm P1 (`..._MISSES` below C); `..._P1_LEND_HEAT` or `..._P1_LEND_AUDIT` with no lent
slot (ring, arena or slab); `..._P1_LEND_SLAB` unless this rank's plan set exactly that many slots aside (a
static plan with the file tier, the switch in the environment when the plan was made); a checkpoint interval below C. Under P1, LCP admits
`..._VRAM_PACKED=1` (the P1 row decodes the packed form; gemv_vp serves 16 rows or fewer).

Log lines: rank 0's `ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of
N to C rows ...` at startup (the threshold, the CPU path's bound, ring, tunables, pooled units), with a
lent ring `ds41 p1: lent ring: K of the ring's S slots ...`, with a lent arena `activation arena: steps of up
to 256 tokens take E of A; the L above them and a 1 MiB guard go to prefill Phase 1's lent arena` and
`ds41 p1: lent arena: N slots (..._P1_LEND_ARENA=K; F fit) ...`, with the heat `ds41 p1: lent heat: on
(..._P1_LEND_HEAT=S; half-life H decode rows, ...; margin M%, ...) ...`; the scheduler's
`ds41 p1: a lone prompt's chunks are aligned to ...`; rank 0's `ds41 p1 chunk: rows R, layers L, wall S s
(issue), rank 0 drive units D, G GB/s, MemAvailable M MiB` after each card-path chunk; each rank's
`ds41 p1 stats, rank R: ...` at shutdown (chunks, layers, units RAM / drive, read-ahead posted / used /
dropped, waves, ring copies and DMA GB, drive units, GB and busy % of the chunks' wall time, the issuing
thread's waits, the packer's ms a unit, admitted units, leases at pass end (must be 0), with a lent ring
its residents, bursts that took them back, returns, reloads by source, failures and overtaken reloads,
with the heat `lent heat: N retargets (...), W residents withdrawn for them, D passes declined (lent or in
the grace, a slot busy, draft bank, candidate, no fence)`,
touched experts split into VRAM residents / slab / ring); the large-chunk line names the threshold and the
CPU path's bound, and each chunk line the layers left below the threshold.

Window parts (`ab_window.sh`): `P1` (needs `LcN`, `B2`..`B8` and an `M` part of min(N, threshold - 1)
or more; the arm's `Mr` part serves the CPU path's passes), `QwN`, `QiN`, `QrN`, `QaN`, `QvN`, `Qs`,
`QmN` (the threshold), `QlK` (the lent ring, K at most 2B - 2), `QeK` (the lent arena, 1-64), `QhMS` (their
grace, the hold; needs `Ql` or `Qe`; with either the `M` part is N or more), `LhS` (the lent slots follow
live decode routing, `..._P1_LEND_HEAT=S`, 1-8; needs `Qe` or `Ls`; every arm states the switch, 0 without
the part, and its two tunables, empty but for the half-life in an `Lhh` arm), `LsN`, `La` and `LhhR` (the
lent slab, the read-back audit and the heat's half-life: "Adaptive residency" below); a P1 arm
is measured only if the startup line states its threshold and chunk, a `Qe` arm only with its lent-arena
line, a `Ql` arm only with its lent-ring
line, an `Lh` arm only with its `lent heat: on (..._P1_LEND_HEAT=S;` line and any other arm only without one; `DEVICE_CHECKS` gains `p1` (`P1_CHECK_M`, 4096, and `P1_ROUTES`),
and a P1 arm runs only on the libr4d.so its `p1: PASS` was on; `LCP_CHECK_C=2048` for Lc2048 arms. The
prompt receipts `Up1len` (five documents of ~1K, ~2K, ~5K, ~8K and ~16K tokens, `max_tokens` 16, from
license texts the long receipts do not use) and `Up1reply` (the ~2K and ~5K ones at 256 tokens) are made
by `p1_receipts.py`; their token counts are estimates (4.9 characters a token) until the window's first
arm logs them.

Host tests (cards hidden): `p1_stager_test` runs the ring's scheduling on a simulated card (two streams
in random order, events with record generations, slots that a copy in flight leaves garbage): every
unit in exactly one wave, RAM units first by rows, table parity by layer, sources let go only after
their copies, and the ring stress (2 to 12 slots, waves of 1 to 3, 440 random runs of 12 layers), also
with pooled units beside RAM and drive ones (never let go, block form only); the host-pool source (a pooled
unit's two weights at its pool slot whatever their order; skip marks on and off around a layer); the
threshold over the end-aligned chunks of every length to 6,000 under the consumer cut (a pass arms only
from the threshold, a layer is the card path's exactly at its own rows, every other layer within the CPU
path's bound, decode, mixed, draft and speculative passes never); the lent ring's state machine on a
simulated host (lend, return after the grace, back-to-back prompts, failed loads and copies left on the
CPU path, loads and copies a card-path pass overtakes never published, decode between chunks, 200 random
runs, every copy still queued fenced when a pass takes the slots back), the lent arena (its slots whole,
aligned and inside the arena's top; a simulated card's two streams in random order, where no reload copy
ever lands inside a full step's activations and every resident read finds its unit, 300 runs, while the
same runs without the fence or the end's hand-over find the race), the lent slots' heat (a retarget of a
settled lender fences, withdraws, renames, reloads, copies and publishes in that order; refused while lent,
in the grace, busy or failed, and when the host cannot fence, with nothing changed; a card-path pass in the
middle of one cancels it and the slot comes back with its new unit; 1,000 retargets with random landings
and card-path passes between them; the decay by half-lives of decode rows; no swap at equal heat or at
exactly the margin; the slots following a shift in routing, each once, and resting while it is steady;
which passes and layers the file stager counts; and on the simulated card's two streams, decode lagging
the host, no copy for a slot's next unit ever under a queued reader of the unit before, 300 runs, while the
same runs without the retarget's fence find the race) and the mover's ring residents
(published at their slots, never aliased or borrowed, file-tier
again once taken back); the tunables; the end-aligned chunks for every length to 20,000; the admission
against the sorted order.
`p1_chunk_test` runs `Scheduler::plan_chunk`'s end alignment on the scheduler fakes, with and without a
checkpoint interval.

P1 twin (package K). Under `RADIANCE_EXPERIMENTAL_DS41_P1=1` beside `LCP`, libr4d registers
`moe_gemm_ggml_mxfp4_gemv_tall_p1` (M ≤ 4096) in place of `_gemv_tall`. It runs gemv_vp's vp2 kernel at
every height over a weight table. Bit 0 marks a packed-scale entry; bit 1 (skip) means no rows written and
no bytes read. Experts own the live-tail ranges, every call zero-fills `[off[96], T)`, and work is cut into
items by row passes, so one expert with 2,000 rows does not serialise a launch. gate_up stages two
activation rows a pass and down three, on 512 blocks. Each output is one wave's sum in the gemv's lane and
block order, so the bytes equal 16-token passes. Gate: `ds41_mxfp4_device_check LIB p1 4096 ROUTES` prints
`p1: PASS` (rows, host rows, skip, tail, splits, waves of 1-3). On a 9070 XT with real routing at M
1024-2048 it costs about 16.4 µs an expert-row in block form and 18.8 packed, against 2,238 for the served
LCP twin and 30 for lcpfast's balanced twin.

## Adaptive residency: the lent slab, the heat over every lent slot, the read-back audit

Not yet measured on the cards. Three switches, each off by default, and with all three off no computed
value, no plan and no pass changes. With them on, a unit's bytes and the experts a row routes to are what
they were; only where a unit is computed moves.

**What it is.** Which experts a card holds is fixed at startup from the calibration profile, but for the
lent arena's 41-53 slots, which every large prefill pass takes back. Replays of routing traces put the
CPU-side reads a card could save by following live routing at about 13% with 41 slots, 21-27% with 128 and
26-47% with all of them. The lent slab gives the lender a large region of its own, the heat policy moves it
with the traffic, and the audit checks every byte it moves.

| Switch | What it does |
|---|---|
| `RADIANCE_EXPERIMENTAL_DS41_P1_LEND_SLAB=N` (1-256 a card; needs `..._P1=1`, `--deterministic`, the file tier) | N slots of VRAM that no pass writes, paid for by the coldest static residents, holding the hottest units off the card from bring-up on |
| `RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=S` (1-8), `..._P1_LEND_HEAT_HALF_ROWS` (1-100000000; 100, or 2000 beside the slab), `..._P1_LEND_HEAT_MARGIN_PCT` (100-1000; 150) | every lent slot (ring, arena and slab) follows a decayed count of live decode rows, at most S slots a rank a pass |
| `RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT=1` (needs a lent slot) | every lent slot read back from the card and compared before its unit is published and before its resident is withdrawn; not for a timed arm |

**The slab's plan** (`Planner::assign_expert_tiered`, `Plan::lend_slab_vram`, `lend_slab_units`). The plan
without the switch is made first. Its static residents then give way from the cold end of the fill's order,
N first and one at a time after, until N slots of 18,800,640 bytes and a 4 KiB pad fit the same weights pool
beside what stays; nothing else becomes resident. With block-form residents that is N residents (N + 1 where
the pool's last free bytes do not hold the pad). With packed residents (`..._VRAM_PACKED`, the recommended
configuration) one frees 18,001,920 bytes and a lent slot holds the block form, so N slots cost about
N x 1.044 residents: 67 for 64, 134 for 128, 268 for 256, give or take one for the pool's slack. The slots'
units are the N hottest file-tier units of the plan that results: the residents given up, hottest first,
less what the host pool takes of them (it holds the first units off the card in any plan; 3 a card in the
recommended configuration), whose former units come next. So with the heat off the cards compute what they
computed, less the few units the packed form's difference costs, which are the hottest candidates for the
lent arena's slots. Packed lent slots would close that gap and are not built: the weight tables mark a
packed entry by its weight's plan, not by where it is published. The mover reserves exactly what the plan
charged; `Plan::report` names it (`lent slab`).

**At bring-up and in passes.** The slab's slots are the lender's last, and kept (`RingLender`): each rank
loads its N units through the lender's own path before the first pass (`CardExpertPath::lend_prime`: from
the RAM cache where it holds the unit, else read from the drive and relaid, 12-26 ms a unit in the lent
arena's measured reloads, so up to about 7 s a card for 256, one card after another; at most 120 s a rank,
the rest comes in at the passes). A pass that takes the ring's and the arena's slots back leaves the slab's: its
residents stay published through it, the plugin's own ops read them at any height (block form, unmarked: the
decode band's row at up to 16 rows, the P1 twin above), there is no grace, and a load or copy in flight is
neither cancelled nor fenced. A kept slot's reload begins only while the other slots are not lent out, so an
armed pass has the drive and the DMA lane to itself. Like the other lent slots, the slab makes the CPU path
take every pass that does not arm the card path (`..._MISSES` of C or more).

**The heat** is the policy above ("The lent slots follow live decode routing"), over all lent slots: the
same counts, margin, swap budget, settled-state rule and fence. Beside the slab its half-life defaults to
2,000 decode rows (a region this large should follow the workload, not the last few replies), and it can be
set up to 100,000,000. Nothing moves before one half-life of decode rows has been counted, so a long
half-life also starts late. And it moves a slot only while every lent slot holds its resident: after each
pass that takes the arena's slots back it rests through the grace and until they are reloaded (about two
seconds with 41 of them), the slab's slots with the rest, so under prompts that arrive faster than that
it acts seldom; the stats line's declined passes say how often.

**Reloads from the packed RAM cache** (`CardExpertPath::load_from`, `Mover::relay_packed_unit_block`,
`ds41hx::packed_scales`). A lent slot holds the card's block form (block_mxfp4), the packed RAM cache
(`..._PACKED_CACHE=1`) the CPU kernels' packed form, so a lease cannot be the copy's source as a block-form
cache's is. But the packed form is lossless: it keeps the release's codes plane as it is and every block's
e8m0 as a 2-bit code against one base or, in a raw round, as it is (the cache's insert checks the round
trip). So where the packed cache holds the unit, the lender borrows its lease (read-only, no LRU move), the
reload thread widens the scales back to the release's plane (`ds41hx::packed_scales`, the kernels' own
widening) and relays the two planes into the reload buffer with the same relayout as a drive reload's, and
the lease goes once that has landed; the copy, the audit and the publish are a drive reload's. No drive
read, and the slot's bytes are the drive reload's byte for byte (`p1_lend_reload_from_the_packed_ram_cache_
gives_the_drive_reloads_slot_bytes`). A unit the cache does not hold (or whose lease is not a packed unit's)
is read from the drive as before. Bring-up, returns after the grace and retargets all take this path. The
stats line's `lent reloads: R from the RAM cache (P relaid from the packed form at M ms a unit), D from the
drive` part counts them (R includes block-form leases).

**The read-back audit** (`RingLender::set_audit`, `LendHost::audit`, `LendAudit`). Twice a residency the
slot's two weights are copied back from the card to pinned host memory on the DMA lane, and the issuing
thread waits for them:

- at publish, once the host has seen the slot's copy pass and before its unit is published: byte for byte
  against the bytes that were copied, which the lender still holds. The slot has no reader yet and no
  writer but the copy. One 64-bit digest a 4 KiB page of what was read is kept (37 KiB a slot);
- at withdrawal, before a resident is published as a file-tier unit again (a pass that takes the slots
  back, or a retarget): behind the event the compute stream recorded at the end of the pass before, the
  residents' last reader, and before anything can write the slot next (the host waits before it issues the
  pass or queues the slot's next copy); its digests against the ones kept;
- at shutdown, every slot that still holds a resident, as at a withdrawal: a slab resident no retarget
  moved is never withdrawn by a pass, so this is where its residency is closed.

A slot that differs, or cannot be read back, is given up: the error line `ds41 p1: lent audit: rank R <ring|
arena|slab> slot S (lent slot K), unit U: ...` names the first differing byte's offset (at publish, with
both bytes) or the first differing page's (at withdrawal), the unit is not published (or is withdrawn) and
takes the CPU path, nothing is loaded into the slot again, and the rank's heat policy makes no further
retarget (`Decline::Audit`). Everything else goes on, audited and counted. Off, none of it exists: no
buffer, no event, no call. On, each read-back costs a copy of 18 MB off the card and a pass over it, at
every publish and every withdrawal, on the issuing thread between passes: the first passes after each
large prefill pay for the arena's slots coming back, so an audited arm proves bytes, not speed.

Log lines: rank 0's `ds41 p1: lent slab: N slots (..._P1_LEND_SLAB=N; V of VRAM a card that no pass writes,
for which the plan gave up G static residents, coldest first) hold rank 0's N hottest file-tier units ...
(P loaded in T s, ...)` and `ds41 p1: lent audit: on (..._P1_LEND_AUDIT=1): ...` at startup; a warning for
any rank whose slab was not full after bring-up; in each rank's `ds41 p1 stats, rank R:` line at shutdown
`lent slab: N slots, R residents now (P at bring-up)` (the `lent ring:` part before it counts the ring's and
the arena's slots alone), `lent audit: A slots read back (D differ, U not read) at M ms a slot, L slots
lost, retargets stopped|not stopped`, and with the heat `heat half-life H decode rows, C counted`.

Window parts (`ab_window.sh`): `LsN` (the slab, 1-256; needs `P1` and an `M` part of the chunk or more),
`La` (the audit; needs `Ql`, `Qe` or `Ls`), `LhhR` (the half-life; needs `Lh`), and `LhS` now beside `Qe` or
`Ls`. Every arm states `..._P1_LEND_SLAB` and `..._P1_LEND_AUDIT` (0 without the part) and the half-life
(empty without `Lhh`). At startup an `Ls` arm is measured only with its `lent slab: N slots
(..._P1_LEND_SLAB=N;` line, an `La` arm only with `lent audit: on`, an `Lh` arm only if its line states the
half-life it asks for (`Lhh`'s, 2000 beside `Ls`, else 100), and an arm without `Ls` or `La` only without
their lines. At shutdown (`lend_exit_gate`, on all four ranks' stats lines) an `Ls` arm is measured only if
every slab held all N units at bring-up, an `La` arm only if every rank read slots back and none differed
or could not be read, an `Lhh` arm only if every rank ran with that half-life and some rank counted that
many decode rows. An arm that fails one is set aside as `<arm>.unmeasured.json` with the reason; for a
differing slot the reason carries the rank, its counts and the audit's own error line.

Host tests (cards hidden): `place_test` `the_lent_slab_is_paid_for_by_the_coldest_residents_and_holds_the_
hottest_units_off_the_card` (the slab's plan against the plan without it: block-form and packed residents,
a host pool with room, a dynamic plan, more slots than residents); `p1_stager_test` `p1_lend_slab_*` (the
tunables; kept slots through bring-up, take-backs, the grace and failures; a random stress beside kept
slots), `p1_lend_heat_governs_arena_and_slab_slots_alike_and_rests_through_a_take_back`, and `p1_lend_audit_*`
on a host with the slots' bytes whose device can corrupt them (a copy that lands with a byte flipped is
caught before its unit is published, a byte written under a resident at its take-back, its retarget or at
shutdown, each with the unit withdrawn, the slot given up and the policy stopped; nothing is read back with
the switch off; random runs); `ab_window_lh_test.sh` (the three parts' startup lines, `lend_exit_gate`, the
arm loop).

What a window has to settle before any of this is recommended: that an `Ls` arm with the heat off answers
byte for byte as the arm without it and at its speed (it holds about N x 0.044 fewer units a card with
packed residents, and bring-up reads N units a card from the drive); that armed prefill passes run as fast
with residents published in the slab; that `Ls` with `Lh` gains on the prompt families it is meant for and
loses nothing on the others; and, with `La`, that no slot ever differs.

## October31 follow-up experiments (256 GB RAM, 192 GiB cache)

The follow-up controls preserve the merged base replies. Since ab-20261103T090558Z / T110235Z / T114720Z the recommendation is Lc2048/P1/lent arena with `Rb` and `Vp` (above, gated at 192 GiB); the other new switches remain opt-in.

- The frozen DSpark verification policy (`Dpp`, `RADIANCE_EXPERIMENTAL_DS41_DSPARK_PREFIX_POLICY=v1:0.937915564:0.97501117`) improved weighted decode on the original code/chat sets by 4.7% / 9.9% against fixed 2 / fixed 1, including the query/observer/wait and required policy telemetry. Held-out gains were 3.4% / 7.7%. All 128 non-reference replies/counts equal the plain controls. These are single-stream 192 GiB results; the same frozen policy's 96 GiB and concurrent-solo composition require their own gates. No family-specific threshold or refit was used. Evidence: ab-20261031T165338Z.
- Optional Lc4096 / Pci4096 passed long corpus, logical KV/logit hashes, prefix turns and 96 GiB spill. The 3,086-token TTFT changed 7.33→5.69s; 6,817 changed 17.12→12.69s (~537tok/s). The short/long five-prompt TTFT sum was neutral and the first tiny prompt was slower. This establishes a long-prompt option, not a general concurrency recommendation. Its first model window keepsRb / Vp / N128 off. Evidence: ab-20261031T162629Z. With `Rb` and `Vp` on the stable build, see **4,096-token chunks** above (ab-20261103T090558Z).
- P1+staticVp now has actual packed residents and passes the long/spill gate. Reverse-order N4/N8 pairs changed 37.67→38.40 and 40.66→41.75tok/s; the previous N8 pair was neutral, so the small gain is noisy. Evidence: ab-20261031T134615Z and ab-20261031T144623Z (the former's Ulong leg was invalid; use the latter's corrected long/96GiB results).
- AR/Engram gather row slicing passed fresh four-card byte oracles and 3,397-position corpus comparisons independently and together. Combined Ar1280 / Ag1024 changed long TTFT 17.68→16.96s, prefix-turn sum 73.84→72.80s, and N8 total 41.08→40.64tok/s. It is not recommended generally on this single mixed-performance pair. Evidence: ab-20261031T182505Z.
- DSpark bank lending restored every borrowed bank byte/alias/mask and resumed a real solo query after actual N4 execution at 192 / 96 GiB, with all replies/counts equal. The 1,792-token mixed workload was slower than a static bank: 56.62→60.29s and 79.01→82.91s. Full cold lifecycle was also slower. Keep it experimental; packed fills, transition subphase timing and longer paid bursts are separate follow-ups. Evidence: ab-20261031T180730Z. The earlier ab-20261031T175623Z client attempt is invalid and is retained as failed evidence.
- P1 SwiGLU span and n128 passed their independent 3,397-position corpus gates and all 42 long/prefix serving requests with logical hashes equal. Long TTFT reference/span/n128 was 17.22 / 17.02 / 17.17 s; prefix sums 73.42 / 72.87 / 72.79 s. These small single-pair differences do not establish a repeatable performance gain. Both remain opt-in. Evidence: ab-20261031T185127Z.

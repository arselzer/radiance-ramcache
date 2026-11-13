# radiance-ramcache: DeepSeek-V4.1-Flash with native MXFP4 experts on four RX 9070 XT

This is a fork of [radiance](https://codeberg.org/StillDeadcode/radiance) tuned for one job: serving
DeepSeek-V4.1-Flash (ds41) with its native MXFP4 routed experts on a workstation with four 16 GB Radeon RX 9070 XT cards (gfx1201, tensor parallel 4)
and an AMD EPYC 7443 (Zen 3 "Milan", 24 cores) with 256 GB of DDR4-2666. The routed experts (about
289 GB) do not fit in the cards' 64 GB, so the hottest experts stay in VRAM, most of the rest sit in a
192 GiB packed RAM cache whose experts the CPU computes, and the remainder is read from NVMe. Every
optimization is exact: replies stay byte-identical to the base engine's, which each measurement window
checks by comparing replies, prefill hashes and logits. [RAMCACHE.md](RAMCACHE.md) lists what the fork
adds; [tools/ds41-host-experts/README.md](tools/ds41-host-experts/README.md) has the configuration,
the switches and every measurement. Below the line is upstream's README, unchanged.

**Batch jobs** (hundreds of short prompts at once): see "Batch jobs: quickstart" in
[tools/ds41-host-experts/README.md](tools/ds41-host-experts/README.md#batch-jobs-quickstart) for the one-command
launcher (`serve-batch.sh`), a client and the measured throughput (256 questions in 329 s on four cards).

**Checkpoint precision:** the model used for these results preserves the official 1017 release's
MXFP4 expert codes/scales, but uses ISTA IQ4_NL Engram tables and BF16/F32 conversions of other
official neural weights. It is a hybrid, not an entirely untouched official-precision checkpoint.
The IQ2_XS supplement supplies integer Engram controls only, not IQ2 neural weights. The numerical
audit passed all 1,119 other neural tensors against their declared decode/cast mappings, and all
seven integer Engram controls matched the official rules. Exactness checks compare with this same
imported baseline; they do not establish equivalence to an independent official runner. Restoring
official FP8 Engram requires a new qualified baseline.

Accepted serving measurements (November 5, 2026): the recommended configuration on the `ds41-stable`
branch (radiance 1.15 base). On the cards, `main` (radiance 1.16) matched that line with identical replies and logits at
the same speed, before batch admission and the expert worker were merged; those two are built and
host-tested here. Each row comes from one or two A/B windows.

| Workload | Four RX 9070 XT (tok/s) | Plus optional R9700 expert worker (tok/s) |
|---|---:|---:|
| Decode, 1 stream | 28-30 | 33.9 |
| Decode, 4 streams | 41-44 | 51.5 |
| Decode, 8 streams | 45-47 | 57.6-58.2 |
| Decode, 8 streams, chat prompts | 38.4-38.9 | 45.8-46.5 |
| Decode, agent-harness turns with prefix cache | 25-26 | 29.4-29.6 |
| Prefill, 6,817-token prompt | 515 | — |
| Prefill, 47,917-token prompt | 776 | — |
| Prefill, 99,987-token prompt | 661 | — |

Decode rates are totals across the active streams (N times the mean per-request rate); concurrent
rows use batch admission. Prefill uses the server's measured prompt-processing rate, rounded to
whole tokens per second; the two longer prompts' values are means of two runs. It excludes parsing,
tokenization and admission wait. All prefill rows use admit-all with the tested 192 GiB cache;
`—` means unmeasured with the worker.

- **Prefill latency**: admit-all takes 13.24 s to first token at 6,817 tokens, 22.8% less than the
  fresh control (one quiet pair), and 61.79 s at 47,917 tokens, 38.4% less (OFF/ON/ON/OFF means).
  All 80 replies/counts and five full prefill hashes matched, with no post-prefill N8 decode regression.
  A further OFF/ON/ON/OFF window at 99,987 tokens measured 151.32 s against 240.46 s, 37.069% less;
  all 84 original replies/counts, 14 full prefill hashes and nine joined server exits passed.
  Subsequent eight-stream wave wall time stayed neutral (13.9895 to 14.0133 s).
  Default admission previously measured 48K/100K at
  115/222 s, or 77/165 s with 4,096-token chunks. Smaller cache budgets retain default admission.
- **Agent-harness latency**: 13 prefix-cached turns total 61-63 s to first token with Phase 1
  read-ahead off (72 s with it on). The optional worker reduces total turn time by about 5-7%.
  A separate admit-all pair at the fixed 32K reservation reduced summed TTFT from 68.9714 to
  56.7298 s; that single quiet pair is exploratory.
- **Context**: a fixed 131,072-token reservation is qualified through 100K-token prompts (about
  0-2% decode cost at short context); a 524,288-token reservation starts and costs 7% at short context.
- **Notes**: the expert worker is optional and turns itself off without a fifth card of the
  same architecture. Branches: `main` (this one), `ds41-stable` (the measured line), `ds41` (a curated
  evidence snapshot) and `ds41-research`.

## Matched resident-drafter screen (9 November)

A separately built prototype compares the best small-pool plain profile with
resident DSpark: four RX 9070 XT cards, one stream, 192 GiB cache, fixed 131K
context and a 256 MiB HOST pool per GPU on both sides.

| Original workload | Plain decode, tok/s | Resident DSpark, tok/s | Change | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| Held-out8, cap64 | 28.38 | 31.21 | +9.99% | 88.64% |
| CODE16, cap256 | 24.69 | 29.04 | +17.63% | 93.82% |

Rates exclude prefill and use generated tokens divided by total decode time.
All 48 full replies and prompt/generated counts matched. This is one fresh pair
per corpus, without repeats or a drift estimate; broader qualification still
blocks adoption. These workloads differ from short CODE4 screens and the older
decode-only profiles. The [dated report](docs/DS41-MATCHED-DECODE-20261109.md)
also records the rejected bank-lending and pageable-cache screens. Defaults and
the qualified serving/cache-budget tables are unchanged.


## Adaptive decode and prefix reuse (9 November)

Further prototype screens keep the same four cards, 192 GiB cache, 256 MiB HOST
pool/GPU and fixed131K context. The latest native-down / IQ2_S-down / native-down
brackets use all384 resident drafter experts:

| Original workload | Candidate | Decode, tok/s | Gain vs matched control mean | Control drift |
| --- | --- | ---: | ---: | ---: |
| CODE16/cap256 | Adaptive three-row, IQ2_S down | 30.20 | +0.58% | +0.22% |
| Held-out8/cap64 | Third-token gate, IQ2_S down | 32.10 | +0.72% | +0.57% |

All72 complete replies/counts matched. The smaller bank frees227.625MiB/GPU and
actually fits13 additional target expert units/card. These are small provisional
gains with one candidate run per corpus; [controls, acceptance and limits](docs/DS41-DN-S-ADAPTIVE-20261109.md)
remain explicit. Earlier native-down adaptive depth improved CODE3.23–3.32%
against its two-row controls; the third gate added no measurable CODE gain.

A separate unchanged13-turn prefix-cache pair reduced summed first-token latency
31.4%. With prefix caching enabled, a read-ahead8/0/8 bracket reduced it8.7% in
that experiment (101.524→92.676→101.523s), with flat decode. These are latency
measurements, not prompt-processing tok/s. Fresh state/lifetime and broader model
gates still precede adoption. See the [prefix and adaptive measurements](docs/DS41-ADAPTIVE-PREFIX-20261109.md).
Qualified serving settings, cold-prefill throughput and96GiB-cache evidence remain unchanged.


## Drafter quantization pilot (8 November)

The separately built DSpark prototype quantizes **only the drafter**; target weights and
arithmetic stay unchanged. All 384 drafter experts remain. Four RX 9070 XTs, 192 GiB RAM cache,
fixed 131K context, one stream, the same controller policy, and 16 CODE requests per arm:

| Drafter gate/up | Drafter down | Placement | HOST pool/GPU, MiB | Draft acceptance | Decode, tok/s |
|---|---|---|---:|---:|---:|
| MXFP4 | MXFP4 | RAM | 2,048 | 94.37% | 27.54 |
| IQ2_XXS | MXFP4 | RAM | 1,387 | 93.82% | 28.65 |
| IQ2_XXS | MXFP4 | GPU | 256 | 93.82% | 28.83 |
| IQ2_XXS | IQ2_S | RAM | 1,159 | 93.53% | 29.00 |
| IQ2_XXS | IQ2_S | GPU | 256 | 93.53% | 29.10 |

All 80 complete replies and token counts matched the base engine. Acceptance is accepted/proposed
draft tokens; decode excludes prefill. The full GPU expert bank reserves 1,130.625 MiB/card with
MXFP4 down weights, or 903.0 MiB with IQ2_S down weights, saving 227.625 MiB/card. Dense weights,
KV and transient buffers are additional. These are **single-order provisional configuration
comparisons**, with different pool sizes and target-expert budgets; differences below 1% need
repetition. Remaining state/hash and broader model gates block adoption. See the
[quantization recipe and evidence](docs/DS41-DRAFTER-QUANTIZATION-20261108.md) for the source bank,
GPU conversion commands, uniform importance weights and validation.

## Earlier measurements (7 November)

These separately built ABI33 candidates leave this checkout's serving configuration and defaults
unchanged. Four RX 9070 XTs are used; the optional worker is off.

### Matched decode and pinned-host pool

A matched held-out test found no decode slowdown between the accepted and current builds
at the same pool size. Four RX 9070 XTs, one stream, 192 GiB cache and fixed 131K context:

| Engine/profile | Pinned HOST pool per GPU, MiB | Decode, tok/s |
|---|---:|---:|
| Accepted ABI32, common settings | 256 | 29.07 |
| Current ABI33, common settings | 256 | 29.27 |
| Current ABI33, common settings | 2,048 | 26.07 |
| Current ABI33, admit-all + insert queue 64 | 2,048 | 26.96 |

All 64 original replies/counts matched and all four paired drift screens passed. The larger
pool shifted 100 target experts per GPU into GPU-readable pinned memory; VRAM residency
stayed constant. The published plain recommendation already uses 256 MiB. The HOST DSpark
bank has a separate larger-pool requirement. Concurrent/prefill follow-ups remain pending.
Rates use generated tokens divided by decode time, averaged across paired arms; the
[replay report](docs/DS41-POOL-REPLAY-20261107.md) gives the controls and both estimators.

### BetterBench quick screen

A local 48-request BetterBench screen used the separately built plain ABI33 prototype, four
RX 9070 XTs, 192 GiB cache, fixed 131K context and up to eight sequences. DSpark and the worker
were off. These are client measurements with 3 chat/code samples and 2 samples per prefill depth:

| Workload | Median rate, tok/s | Median client TTFT, s |
|---|---:|---:|
| Code, streaming decode | 28.28 | 0.928 |
| Chat, streaming decode | 25.91 | 1.189 |
| Prefill, actual ~5.9K tokens | 622.89 | 9.533 |
| Prefill, actual ~23.4K tokens | 755.56 | 31.028 |
| Prefill, actual ~47.1K tokens | 742.61 | 63.476 |

Eight-request waves at concurrency 1/4/8 delivered 23.54/30.37/33.66 tok/s **end to end**,
including prefill and scheduling. SSE decode excludes the first content update; prefill divides
actual prompt tokens by client TTFT. These differ from the server metrics above. This small
screen is a baseline, without a full BetterBench score or new quality claim. The
[screen report](docs/DS41-BETTERBENCH-SCREEN-20261107.md) gives all samples, settings and a separate
matched workload control; installed stable/defaults are unchanged.

DSpark in the recommended C2/P1 profile, fixed 131,072-token context, one stream and a 192 GiB
expert cache on the 256 GB host:

Both sides below use a **2,048 MiB pinned HOST pool per GPU**. The plain 256 MiB pool
was not included in this window. These gains are relative to the tested plain baseline;
comparison with the best plain configuration awaits the four-condition pool/bank/drafting test.

| Original workload | Plain decode, tok/s | DSpark, tok/s | Change | Scope |
|---|---:|---:|---:|---|
| CODE16, cap256 | 24.81 | 28.59 | +15.23% | Matched OFF/ON/ON/OFF bracket |
| CHAT16, cap64 | 24.55 | 25.23 | +2.76% | Exploratory single pair |

All 96 replies and original prompt/generated counts matched. Rates divide generated tokens by
decode time, then average same-treatment arms. These workloads differ from the historical
held-out/concurrent serving measurements; they are not a new absolute-speed record.

The separate P1 partial-wave-flush experiment passed full correctness/activity checks, but both
quiet timing screens exceeded the 5% control-drift limit. Qwf stays OFF. These are descriptive
Qwf-OFF reference means from separate windows, not a matched cache-size comparison:

| Expert cache | Original prompt | Prefill, tok/s | TTFT, s | Decode, tok/s |
|---|---|---:|---:|---:|
| 192 GiB | 6,817 tokens | 431.09 | 15.857 | 26.50 |
| 96 GiB | 6,817 tokens | 191.79 | 35.568 | 11.34 |
| 192 GiB | 99,987 tokens | 682.68 | 146.523 | 21.68 |
| 96 GiB | 99,987 tokens | 653.92 | 153.293 | 12.66 |

Both budgets ran on the 256 GB host with fixed 131K context, one stream, 2,048-token chunks and
admit-all. Decode is generated-token-weighted; startup is separate. A 96 GiB cache run on this host
does not validate the entire current configuration on a 128 GB machine: the larger host leaves
different room for OS page cache, staging and temporary allocations. See the
[dated results](docs/DS41-RESULTS-20261107.md) for full OFF/ON rates, drift and exact scope.

---

# radiance

An inference server for AMD GPUs, in C++ and HIP: language models, and image, video, speech and
music generation. Its HTTP API speaks OpenAI's chat, completions, Responses, images, audio and video
APIs, Anthropic's Messages API and llama-server's native one, so their clients and SDKs work
unchanged. Model architectures, kernel libraries and quantisers are `.so` plugins; the ones in this
tree are built and loaded the same way a third party's are ([`docs/PLUGIN.md`](docs/PLUGIN.md)).

The bundled device kernels (`libr4d`) target AMD RDNA4 (`gfx1200`, `gfx1201`). The serve flags
below are for two 32 GB cards (Radeon AI PRO R9700). The engine starts on any AMD GPU family;
another card needs a kernel library for it. A machine with no GPU builds the host backend, whose
AVX-512/AVX2 kernels (`libavx`) serve every model here: the small ones (EmbeddingGemma 2,
Qwen3-ASR, the 0.8B-3B models) at interactive speed on a desktop CPU, the 27B and larger far more
slowly than a card.

To support the development of radiance: [Patreon](https://patreon.com/StillDeadCode).

## Models

| model | container | get it | compose file |
|---|---|---|---|
| Qwen3.8-Flash-Next: 4-bit experts, int8 trunk, MTP, vision | `qwen3.8-next-flash-fp8-iq4r-moe.rad` (114 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe) | `flashnext.yaml` |
| Qwen3.8-27B FP8, DFlash2 drafter, vision | `qwen3.8-27b-fp8.rad` (29 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-fp8) | `qwen3.8-27b-fp8.yaml` |
| Qwen3.8-27B MXFP4 (AMD Quark AWQ), DFlash2 drafter, vision | `qwen3.8-27b-mxfp4.rad` (19 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-mxfp4) | `qwen3.8-27b-mxfp4.yaml` |
| Qwen3.6-35B-A3B FP8, MTP | `qwen3.6-35b-a3b-fp8.rad` (35 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-fp8) | `qwen3.6-35b-a3b.yaml` |
| Ornith-1.5-35B-A3B FP8, MTP, vision | `ornith-1.5-35b-a3b-fp8.rad` (35 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-35b-a3b-fp8) | `ornith-1.5-35b-a3b.yaml` |
| Ornith-1.5-35B-A3B w4a8: 4-bit experts (GPTQ), MTP, vision, fits one card | `ornith-1.5-35b-a3b-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-35b-a3b-w4a8) | `ornith-1.5-35b-a3b-w4a8.yaml` |
| Ornith-1.5-122B FP8, MTP, vision, four cards | `ornith-1.5-122b-fp8.rad` (118 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-122b-fp8) | `ornith-1.5-122b.yaml` |
| Ornith-1.5-122B FP8, DFlash2 drafter, vision, four cards | `ornith-1.5-122b-fp8-dflash2.rad` (116 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-122b-fp8-dflash2) | `ornith-1.5-122b-df2.yaml` |
| Ornith-1.5-122B w4a8: 4-bit experts (GPTQ), MTP, vision, four cards | `ornith-1.5-122b-w4a8.rad` (65 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-122b-w4a8) | `ornith-1.5-122b-w4a8.yaml` |
| Ornith-1.5-122B w4a8: 4-bit experts (GPTQ), DFlash2 drafter, vision, four cards | `ornith-1.5-122b-w4a8-dflash2.rad` (64 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-122b-w4a8-dflash2) | `ornith-1.5-122b-w4a8-df2.yaml` |
| Sage-A1.5 FP8, MTP, vision | `sage-a1.5-fp8.rad` (35 GiB) | [download](https://huggingface.co/Aijuus/sage-a1.5-fp8) | `sage-a1.5.yaml`, `sage-a1.5-tp1.yaml` |
| Sage-A1.5 FP8, DFlash2 drafter, vision | `sage-a1.5-fp8-df2.rad` (34 GiB) | [download](https://huggingface.co/Aijuus/sage-a1.5-fp8-df2) | `sage-a1.5-df2.yaml` |
| MiniCPM5-2B FP8, DSpark drafter | `minicpm5-2b-fp8.rad` (3 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-fp8) | `minicpm5-2b.yaml` |
| MiniCPM5-2B bf16, DSpark drafter | `minicpm5-2b-bf16.rad` (5 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-bf16) | `minicpm5-2b-bf16.yaml` |
| Gemma 4 31B: GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-w4a8) | `gemma4-31b-w4a8.yaml` |
| Gemma 4 31B heretic-ara (refusals removed): GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-heretic-ara-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-heretic-ara-w4a8) | `gemma4-31b-heretic-ara-w4a8.yaml` |
| Gemma 4 31B: int4 trunk, int8 activations and lm_head, DFlash drafter, vision | `gemma4-31b-dense-w4a8-df.rad` (22 GiB) | [download](https://huggingface.co/Ashlyn4202025/gemma4-31b-w4a8-dflash) | `gemma4-31b-w4a8-df.yaml` |
| Gemma 4 31B Q8: int8 trunk and lm_head (Q8_0's grid at group 128), int8 DFlash drafter, vision | `gemma4-31b-dense-q8-df.rad` (33 GiB) | [convert](#convert-a-checkpoint) | `gemma4-31b-q8-df.yaml` |
| Gemma 4 31B on four cards: int4 trunk, int8 activations and lm_head, RedHatAI's DFlash2 drafter, vision | `gemma4-31b-dense-w4a8-df2-qkv.rad` (23 GiB) | [download](https://huggingface.co/Ashlyn4202025/gemma4-31b-w4a8-dflash2) | `gemma4-31b-w4a8-df2-tp4.yaml` |
| Gemma 4 31B on four or two cards: int4 trunk, int8 activations and lm_head, a block-16 DFlash2 drafter (15 drafts a step, verified as a tree), vision | `gemma4-31b-dense-w4a8-df2b16-qkv.rad` (23 GiB) | [download](https://huggingface.co/Ashlyn4202025/gemma4-31b-w4a8-dflash2-b16) | `gemma4-31b-w4a8-df2b16-tp4.yaml`, `gemma4-31b-w4a8-df2b16-tp2.yaml` |
| NVIDIA-Nemotron-3.5-Bolt-E4B bf16 (Gemma 4 E4B), NVIDIA's DFlash drafter, vision, audio | `nemotron-3.5-bolt-e4b-bf16.rad` (17 GiB) | [download](https://huggingface.co/keptn/nemotron-3.5-bolt-e4b-bf16) | `nemotron-3.5-bolt-e4b-bf16.yaml` |
| NVIDIA-Nemotron-3.5-Bolt-E4B int8 (i8a8, Gemma 4 E4B), NVIDIA's DFlash drafter, vision, audio | `nemotron-3.5-bolt-e4b-i8g.rad` (13 GiB) | [download](https://huggingface.co/keptn/nemotron-3.5-bolt-e4b-i8g) | `nemotron-3.5-bolt-e4b-i8g.yaml` |
| NVIDIA-Nemotron-3.5-Bolt-E4B 4-bit (int4 feed-forward, GPTQ, Gemma 4 E4B), NVIDIA's DFlash drafter, vision, audio | `nemotron-3.5-bolt-e4b-a4.rad` (12 GiB) | [download](https://huggingface.co/keptn/nemotron-3.5-bolt-e4b-a4) | `nemotron-3.5-bolt-e4b-a4.yaml` |
| Qwen-Image 2.1 FP8: text to image and image edits | `qwen-image-2.1-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-fp8) | `qwen-image-2.1.yaml` |
| Qwen-Image 2.1 Turbo FP8: the same in 8 steps | `qwen-image-2.1-turbo-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-turbo-fp8) | `qwen-image-2.1-turbo.yaml` |
| Krea 2 Turbo FP8: text to image, one card | `krea-2-turbo-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/krea-2-turbo-radiance-fp8) | `krea-2-turbo.yaml` |
| Krea 2 Raw FP8: the same, undistilled | `krea-2-raw-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/krea-2-raw-radiance-fp8) | `krea-2-raw.yaml` |
| FLUX.3 [klein] FP8: text to image and edits from up to ten pictures, one card | `flux-3-klein-fp8.rad` (9.3 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-fp8) | `flux-3-klein.yaml` |
| FLUX.3 [klein] base FP8 | `flux-3-klein-base-fp8.rad` (9.3 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-base-fp8) | `flux-3-klein.yaml` |
| LTX-2.5 distilled FP8: video with its stereo soundtrack from text or a first frame, one card | `ltx-2.5-distilled-fp8.rad` (37 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-distilled-fp8) | `ltx-2.5.yaml` |
| LTX-2.5 base FP8 | `ltx-2.5-fp8.rad` (37 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-fp8) | `ltx-2.5.yaml` |
| MiniMax-H3 FP8: video with its stereo soundtrack from text | `minimax-h3-fp8.rad` (60 GiB) | [download](https://huggingface.co/StillDeadcode/minimax-h3-fp8) | `minimax-h3.yaml` |
| FastH3 8-Step V2 FP8: the same in eight steps | `fasth3-8-step-v2-fp8.rad` (62 GiB) | [download](https://huggingface.co/StillDeadcode/fasth3-8-step-v2-fp8) | `fasth3.yaml` |
| MiniMax-Music3 bf16: songs from lyrics and a description | `minimax-music3-bf16.rad` (20 GiB) | [download](https://huggingface.co/StillDeadcode/minimax-music3-bf16) | `minimax-music3.yaml` |
| Unlimited-OCR bf16: document parsing with the DeepEncoder tower | `unlimited-ocr-bf16.rad` (6.2 GiB) | [download](https://huggingface.co/Aijuus/unlimited-ocr-bf16) | `unlimited-ocr-bf16.yaml` |
| Breeze-TTS-2 bf16: speech, whole, streamed or live, one card | `breeze-tts2-bf16.rad` (5.5 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-breeze-tts2-bf16) | `breeze-tts2.yaml` |
| Fish Audio drama-3: speech with speaker tags and inline direction, voices cloned from reference clips, whole, streamed or live; block-FP8 experts, two cards | `drama3-fp8.rad` (36 GiB) | [convert](#convert-a-checkpoint) | `drama3.yaml` |
| Qwen3.8-27B ParoQuant int4, int5 or MXFP4 (learned rotations), DFlash2 drafter, vision | `qwen3.8-27b-paro-int4.rad` (19 GiB; int5 22 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-paro.yaml` |
| TranslateGemma 27B MXFP4 (GPTQ), translation; optional DFlash2 drafter | `translategemma-27b-mxfp4-gptq.rad` (17 GiB) | [convert](#convert-a-checkpoint) | `translategemma-27b.yaml` |
| GLM-5.3-Flash: MXFP4 experts and block-FP8 attention (AMD Quark), MTP, vision, eight cards | `glm53-flash-mxfp4.rad` (172 GiB) | [convert](#convert-a-checkpoint) | `glm53-flash.yaml` |
| Qwen3.8-27B bf16 | `qwen3.8-27b-bf16.rad` (50 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-bf16.yaml` |
| Qwen3.8-Flash-Next bf16, a disk-streamed reference | `qwen3.8-flash-next-bf16.rad` (335 GiB) | [convert](#convert-a-checkpoint) | `flashnext-bf16.yaml` |
| Stable Diffusion XL bf16 and its finetunes (Illustrious, NoobAI, Pony): text to image, one card | `illustrious-xl-v2.0-bf16.rad` (6.4 GiB), from any single-file SDXL checkpoint | [convert](#convert-a-checkpoint) | `sdxl.yaml` |
| EmbeddingGemma 2 bf16: text, image, video and audio embeddings, one card | `embeddinggemma2-bf16.rad` (1.4 GiB) | [convert](#convert-a-checkpoint) | `embeddinggemma2.yaml` |
| nomic-embed-text-v2-moe bf16: text embeddings, one card | `nomic-embed-v2-moe-bf16.rad` (959 MB) | [convert](#convert-a-checkpoint) | - |
| Nemotron 3.5 Embed 1B BF16: text and image embeddings, one card | the checkpoint as it ships | [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) | — |
| Nemotron 3.5 Embed 8B BF16: text and image embeddings, one card | the checkpoint as it ships | [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | — |
| Nemotron 3.5 Rerank 8B BF16: text/image document relevance, one card | the checkpoint as it ships | [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | — |
| Anima Aesthetic FP8: anime and illustration, text to image, one card (Base and Turbo convert the same way) | `anima-aesthetic-fp8.rad` (3.0 GiB) | [convert](#convert-a-checkpoint) | `anima.yaml` |
| Qwen3-ASR 1.7B bf16: speech to text, `/v1/audio/transcriptions` and live over `/v1/realtime`, 30 languages, one card | the checkpoint as it ships (4.1 GiB) | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | — |

A container holds everything the server needs: the weights, the tokeniser, the chat template and
any drafter. Each download link's repository has the recipe and the exact command that made the
file. Download one into your models directory (the transfer resumes if interrupted):

```sh
mkdir -p /srv/models/rad && cd /srv/models/rad
curl -LC - -O https://huggingface.co/StillDeadcode/minicpm5-2b-fp8/resolve/main/minicpm5-2b-fp8.rad
```

The Gemma 4 container is keptn42's: GPTQ calibrated in transformers, with an int8 lm_head; its
repository has the recipe and the calibration step. To convert Google's bf16 checkpoint yourself,
`data/recipes/gemma4-31b-dense-bf16-w4a8.recipe` quantises without calibration (round-to-nearest
with a scale search, bf16 lm_head); see [Convert a checkpoint](#convert-a-checkpoint). On two R9700s
that container decodes at 103 tok/s on prose and 141 on code, against 112 and 150 for keptn42's,
and accepts 0.46 of its drafts on prose against 0.50.

TranslateGemma 27B is converted from Google's bf16 checkpoint in three steps: an int8 container to
calibrate in, its Gram matrices over translation prompts, then MXFP4 with GPTQ
(`data/recipes/translategemma-27b-*.recipe`, the steps in docs/ARCHITECTURES.md). On two R9700s it
decodes 56 tok/s a stream (205 at four) and scores mean KL 0.042 against the bf16 model on held-out
translations; with a DFlash2 drafter, 152 a stream. Its chat template takes structured turns
naming the two languages and refuses the plain turn the server renders at startup, so
`translategemma-27b.yaml` serves a plain Gemma 3 template (`deploy/compose/gemma3.jinja`) and a
client sends on `/v1/completions` the prompt the checkpoint's template renders.

GLM-5.3-Flash is converted from AMD's Quark export (`amd/GLM-5.3-Flash-Quark-MXFP4`) with
`data/recipes/glm53-mxfp4.recipe`, which keeps every weight as stored except the latent
attention's small kv_b, decoded to bf16. At 172 GiB it takes eight R9700s (`--tp 8`), every expert
resident, and its own MTP layer drafts. On eight R9700s it prefills 3958 tok/s at 6k tokens,
decodes 166 tok/s on one stream and 426 over eight, and 506 under eight streams of agent turns.
Pictures go through its own vision tower, up to 4096 tokens an image at the default encoder pass
(`--mm-max-patches 32000` serves the checkpoint's 8000); the KV cache is bf16. Its template knows
the reasoning efforts `low` and
`high` and renders any other as its maximum, so `glm53-flash.yaml` maps `medium` and `xhigh`
to `high`.

The ParoQuant containers keep the trunk exactly as the ParoQuant releases ship it, with the learned
Givens rotations it was trained in. Convert int4 from `z-lab/Qwen3.8-27B-PARO`, int5 from
`Launch80/Qwen3.8-27B-PARO-int5` and MXFP4 from `MisterSnrub/Qwen3.8-27B-PARO-MXFP4-v2`, each with
`data/recipes/q38-27b-paro-mxfp4-df2.recipe` and the DFlash2 drafter. A ParoQuant container made
before 1.5.0 is refused at startup; convert it again with the old one as `--reuse`, which copies
every tensor that did not change.

The Qwen-Image containers serve OpenAI's images API (`/v1/images/generations`, `/v1/images/edits`):
pictures from a prompt, and edits of up to ten pictures with an optional mask, 2048x2048 by default.
On two R9700s a 2048x2048 picture takes 61 s at the base model's 40 steps and 13 s with Turbo (56 and
12 s with `--tp-wire wht6`); 1024x1024 takes 6 s and 2.6 s. Qwen-Image 2.1 is under Qwen's research
license, which is non-commercial.

The MiniMax-H3 containers serve OpenAI's video jobs (`/v1/videos`): a video with its stereo
soundtrack from a prompt, 5 to 14 s, up to 1344x768. On two R9700s a 5.17 s clip at 1344x768 takes
about 68 s with FastH3 and 8.8 minutes at MiniMax-H3's 50 steps; both read about 39 GiB of pinned
host memory. With LBH-123-AI's latent upscaler in the container, `"latent_upscale": 2` runs the first
steps at half the size and refines at the full size: MiniMax-H3's clip then takes 3.4 minutes
(docs/GUIDE.md §5.16). A ComfyUI node, `integrations/comfyui/ComfyUI-Radiance-MiniMax`, renders
through `/v1/videos` from a ComfyUI graph as a client of the server. MiniMax-H3's license, which
FastH3 carries, does not cover the EU, the UK, the Republic of Korea or the USA.

MiniMax-Music3 and Breeze-TTS-2 answer `/v1/audio/speech`, whole or streamed as they are made: a song
from lyrics and a description, 44.1 kHz stereo, made at about the speed it plays on two cards; and
speech in a voice the request describes, made in 0.6 of its playing time on one card, its first
streamed audio 0.21 s after the request. Breeze-TTS-2 also speaks text while it is being written,
over the `/v1/audio/speech/stream` WebSocket; it is licensed for research and non-commercial use
only. Fish Audio's drama-3 speaks a script whose speakers and direction are written in it
(`<|speaker:1|>`, `[laughs]`), each speaker in the model's voice or one cloned from a reference
clip and its transcript, at 0.3 of its playing time on two cards, its first streamed audio
0.23 s after the request; under the Fish Audio Community License a service that lets others
generate speech with it needs a licence from Fish Audio, and imitating a real person's voice needs
their consent.

## Run with Docker

The host needs Docker and an `amdgpu` driver that supports ROCm 10.1; it does not need ROCm. The
card works best with Resizable BAR on (Above 4G Decoding and Resizable BAR in the firmware setup);
without it the engine keeps kernel arguments in pinned host memory, and every dispatch reads them
over the link. The image
is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance), with its kernels
built for `gfx1201`. Pull it, clone this repository for its compose files, then start a model from
one of them:

```sh
docker pull stilldeadcode/radiance
git clone https://codeberg.org/StillDeadcode/radiance.git
cp -r radiance/deploy/compose ~/radiance-compose && cd ~/radiance-compose
cp .env.example .env                         # set RADIANCE_MODELS, RADIANCE_API_KEY, the GIDs
docker compose -f minicpm5-2b.yaml up -d     # or any file from the table
docker compose -f minicpm5-2b.yaml logs -f
docker compose -f minicpm5-2b.yaml down
```

Every model listens on the same port and most use both cards, so run one at a time. Running without
compose, and the reason for each `docker run` option, is in [`docs/DOCKER.md`](docs/DOCKER.md).

To build the image yourself, for example for other cards, run `docker/build.sh -t 'gfx1200;gfx1201'`
in the clone (it needs `docker buildx` and tests the build as it goes). Then set
`RADIANCE_IMAGE=radiance:latest` in `.env`.

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+ and Ninja. For the device kernels you also need
ROCm 10.1 or 7.2 (`hipcc`; set `ROCM_PATH` if it is not `/opt/rocm`). Clone the repository and
build in it:

```sh
git clone https://codeberg.org/StillDeadcode/radiance.git && cd radiance
./build.sh                                   # configure, build for the cards present, run the fast tests
./build.sh --no-hip                          # host backend only
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS='gfx1201'   # build for cards this machine lacks
```

To install, name the prefix when the build directory is first configured
(`-DCMAKE_INSTALL_PREFIX=/opt/radiance`), build, then run `cmake --install build`. The default
plugin path, `<prefix>/share/radiance`, is compiled in from that first configure, so
`cmake --install build --prefix DIR` alone gives a tree that looks for its plugins elsewhere. The
prefix gets `bin/`, `include/radiance/`, `lib/cmake/radiance/`, `libexec/radiance/`,
`share/radiance/` and `share/doc/radiance/`.

With ROCm 10.1 the build turns the compiler's VOPD pairing (dual-issue instruction pairs) off: its
LLVM 24 pairs in an order that follows heap addresses, so two builds of the same source can differ.
The Docker image builds its compiler with the fix and keeps the pairing, which is 1.4-3.5% faster
on the fp8 prefill GEMMs; a compiler that has the same fix is declared with
`-DRAD_HIP_VOPD_REPEATABLE=ON`. ROCm 7.2 is unaffected.

## Run without Docker

```sh
export RADIANCE_HOME=$PWD/build/radiance_home   # the plugins; an install finds its own
build/bin/radiance --model /srv/models/rad/minicpm5-2b-fp8.rad --port 8000 <flags>
build/bin/radiance --model StillDeadcode/minicpm5-2b-fp8 --port 8000 <flags>
```

`--model` also takes a Hugging Face repo id, as vLLM does. The server fetches the repo with `curl`
and `jq` (`scripts/hfget.sh`) into `~/.cache/radiance/hf` (`--download-dir` puts it elsewhere),
serves the container it holds, and checks it for changes at every start; a start that cannot reach
the hub serves the copy it has. The Docker image carries both tools; in a container,
`--download-dir` must name a writable mount, since the compose files mount `/models` read-only.

These rows match the compose files, minus Flash-Next's `--prefix-cache-mib`; the three- and
four-rank rows have no compose file. The other models' flags are in their compose files.

| model | flags |
|---|---|
| Qwen3.8-Flash-Next | `--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-Flash-Next, four ranks | `--tp 4 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-Flash-Next, three ranks (the shipped container as it is, [`docs/TP3.md`](docs/TP3.md)) | `--tp 3 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 8192 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-27B FP8 and MXFP4, Qwen3.6-35B-A3B | `--tp 2 --max-num-seqs 32 --max-model-len 200000 --kv-cache-dtype fp8` |
| Qwen3.8-27B bf16 | `--tp 2 --max-num-seqs 8 --max-model-len 200000 --kv-cache-dtype fp8` |
| MiniCPM5-2B FP8 and bf16 | `--tp 2 --max-num-seqs 32 --max-model-len 131072 --kv-cache-dtype fp8 --gpu-headroom-mib 160 --max-num-batched-tokens 512` |
| Qwen3.8-Flash-Next bf16 | `--tp 2 --max-model-len 32768 --placement expert_tiered --weights-disk-tier --host-pool-mib 12288 --gpu-headroom-mib 768 --deterministic --max-num-seqs 32 --max-num-batched-tokens 32768 --num-speculative-tokens 0` |
| Qwen-Image 2.1 and 2.1 Turbo | `--tp 2 --tp-wire wht6` |
| MiniMax-H3 and FastH3 | `--tp 2 --vision-tower-placement ram` |
| MiniMax-Music3 | `--tp 2` |
| Breeze-TTS-2 | `--tp 1` |
| Fish Audio drama-3 | `--tp 2` |

Every model here runs at up to 32 sequences a step. The Flash-Next rows (except the bf16 reference)
and the 27B in bf16 keep 8: on a hybrid model each sequence holds its own recurrent state, and 32 of
them take that memory from the attention cache and, on Flash-Next, from the expert slab. See
[`docs/GUIDE.md`](docs/GUIDE.md) §4.4.

At `--tp 4`, decode and verify steps carry the exact all-reduce in the gated residual's write, with
the MoE gather in front of it; a prefill chunk runs the unfused chain. That chain's exact
all-reduce is one-shot up to 1 MiB a message and two-shot above, which sends half the bytes on a
prefill chunk; both produce the same bits. The four-rank row runs exact. At three and four ranks
`--tp-wire wht6` selects a different lossy wire, tiered int8 over the two-shot, above
`--tp-wire-min-kb`. Eight ranks have no lossy wire, so `--tp 8 --tp-wire wht6` stops at startup
naming the all-reduce.

Four ranks is a capacity configuration: its per-step rate against `--tp 2` on the same cards has
not been measured, and its tiering flags are the two-rank recipe carried over. As at two ranks, each
rank owns a slice of every expert's width: each parity's 128-column blocks are cut contiguously over
the ranks and the odd block alternates with the parity, so every rank's share is whole blocks and
every block has one owner. The resident expert bytes per rank are smaller, so the elastic window
has a different shape: read the boot log's `elastic` line and re-derive `--host-pool-mib` before
trusting it.

`--host-pool-mib` is pinned on every card, so the four-rank row pins 4 × 12 GiB = 48 GiB of RAM
that nothing else can use and the kernel cannot swap out. The amdgpu driver caps what all processes
together pin at the machine's RAM less 1/64 and 1.5 GiB; the pool is locked user memory, which
`ttm.pages_limit` does not count. A 64 GiB machine serves `flashnext-tp1.yaml` with 49 GiB pinned
([`docs/GUIDE.md`](docs/GUIDE.md) §7).

Add `--api-key KEY` to require `Authorization: Bearer KEY` (or `X-Api-Key: KEY`). `--debug-graph`
prints every op and the kernel it resolved to, then exits without serving. A chat request:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

An image model answers OpenAI's images API instead, with base64 pictures:

```sh
curl -s localhost:8000/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a lighthouse on a cliff at sunset, oil painting","size":"1024x1024"}' \
  | python3 -c 'import sys,json,base64; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`http://localhost:8000/` is the dashboard. [`docs/GUIDE.md`](docs/GUIDE.md) covers the API, memory
budgets, long context, speculative decoding and operating a server.

## Convert a checkpoint

`rad-convert` turns a Hugging Face checkpoint into a container, quantising what a recipe in
`data/recipes/` names. Download checkpoints with `scripts/hfget.sh <org/repo> <dir>`.

```sh
build/bin/rad-convert /srv/models/hf/Qwen/Qwen3.8-27B -o qwen3.8-27b-bf16.rad \
    --recipe data/recipes/q38-27b-bf16.recipe
build/bin/rad-convert /srv/models/hf/Qwen/Qwen3.8-Flash-Next -o qwen3.8-flash-next-bf16.rad --max-ctx 4096
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-w4a8.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/google/gemma-4-31B-it-assistant
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-dense-w4a8-df.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8-df.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/z-lab/gemma-4-31B-it-DFlash
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-dense-q8-df.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-q8-df.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/z-lab/gemma-4-31B-it-DFlash
build/bin/rad-convert /srv/models/hf/z-lab/Qwen3.8-27B-PARO -o qwen3.8-27b-paro-int4.rad \
    --recipe data/recipes/q38-27b-paro-mxfp4-df2.recipe \
    --draft-model /srv/models/hf/z-lab/Qwen3.8-27B-DFlash2
build/bin/rad-convert /srv/models/hf/ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ -o swift15-27b-w4a16.rad \
    --recipe data/recipes/q38-27b-w4a16.recipe
# Any single-file SDXL checkpoint (SDXL 1.0, Illustrious, NoobAI, Pony, a civitai finetune), with
# CLIP's tokenizer.json from openai/clip-vit-large-patch14
build/bin/rad-convert /srv/models/hf/OnomaAIResearch/Illustrious-XL-v2.0/Illustrious-XL-v2.0.safetensors \
    -o illustrious-xl-v2.0-bf16.rad --recipe data/recipes/sdxl-bf16.recipe \
    --tokenizer /srv/models/hf/openai/clip-vit-large-patch14/tokenizer.json
```

With Docker, use `docker run --rm -v /srv/models:/models --entrypoint rad-convert stilldeadcode/radiance ...`.
In the image the recipes are in `/opt/radiance/share/radiance/recipes/`. A checkpoint served at its
own precision needs no container: `radiance --model /srv/models/hf/Qwen/Qwen3.8-27B-FP8`. A release
someone else quantised is read as it ships when the engine serves its form (block-scaled FP8,
Quark's MXFP4), written at its own numbers when the kernels read them in another spelling (an int4
W4A16 release, compressed-tensors' AWQ and GPTQ or AutoRound's, on the 4-bit GEMMs with
`data/recipes/q38-27b-w4a16.recipe`), and converted from its exact dequantisation when no kernel
reads it: NVFP4 (compressed-tensors' and NVIDIA ModelOpt's) to the MXFP4 kernels, an AutoRound
Flash-Next's int4 block projections to int8. [docs/GUIDE.md](docs/GUIDE.md) names the recipes of record
with their `rad-convert` commands.

## Test

```sh
ctest --test-dir build -L fast -j8         # the inner loop, under 15 s; build.sh runs it
ctest --test-dir build -LE 'gpu|oot' -j8   # everything that runs without a card (about 5 minutes)
ctest --test-dir build -L gpu              # the card tests; they need the cards to themselves
ctest --test-dir build -L oot              # rebuilds every bundled plugin out of tree against an install
```

`CMakePresets.json` names the same runs (`ctest --preset fast`, `host`, `gpu`, `full`, `asan`), and
[`docs/GUIDE.md`](docs/GUIDE.md) §1.3 describes the tiers and the test binaries' options.

## Documentation

| | |
|---|---|
| [`docs/GUIDE.md`](docs/GUIDE.md) | installing, converting, serving, the HTTP API, operating a server |
| [`docs/TOOLS.md`](docs/TOOLS.md) | every flag of every tool, and the recipe syntax |
| [`docs/DOCKER.md`](docs/DOCKER.md) | the image and the compose files |
| [`docs/PREPARING_YOUR_RIG.md`](docs/PREPARING_YOUR_RIG.md) | a two-R9700 machine from firmware setup to a serving model |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | writing a kernel library, architecture or quantiser plugin, including one for another AMD card |
| [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) | the supported model families and their geometry |
| [`docs/TP3.md`](docs/TP3.md) | any world size: the attention world and uneven delta-net heads; three ranks |
| [`docs/OPS.md`](docs/OPS.md), [`docs/KERNELS.md`](docs/KERNELS.md) | the op vocabulary, and what `rad-kbench` last measured |
| [`spec.md`](spec.md) | the design |

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from. The Docker image also holds third-party libraries under their own licenses, GPL and LGPL
among them; `/usr/share/doc/THIRD-PARTY` in the image lists each one with its version and source.

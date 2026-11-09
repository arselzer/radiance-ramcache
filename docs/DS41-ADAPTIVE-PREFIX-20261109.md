# Adaptive decode and prefix-cache screens, 9 November 2026

These are separately built four-RX-9070-XT prototypes: one stream, 192 GiB
expert cache, 256 MiB pinned HOST pool per GPU, fixed131072 context and2048-token
prefill chunks. All384 drafter experts remain; gate/up is IQ2_XXS and down is
MXFP4. Target weights, arithmetic and routing are unchanged. Exactness is against
the imported checkpoint baseline described in the README. No serving default
or installed runtime changed.

## Decode: separate code and held-out results

Each condition starts a fresh server without warmup. Decode is generated tokens
divided by summed server decode time, excluding prefill. CODE16/cap256 generated
3995 tokens per condition; held-out8/cap64 generated485. All104 full replies and
prompt/generated counts matched each other and the historical base references.

| Workload | Two-row query | Adaptive three-row proxy | Proxy plus third-token gate |
| --- | ---: | ---: | ---: |
| CODE16/cap256, decode tok/s | 29.11 | 30.08 | 30.09 |
| Accepted/proposed drafts | 2246/2394 | 2444/2627 | 2420/2573 |
| Held-out8/cap64, decode tok/s | 31.38 | 31.33 | 31.81 |
| Accepted/proposed drafts | 234/264 | 251/296 | 249/285 |

The proxy uses a prospectively fixed.999 gate on the existing first-two signal.
The new third-token gate additionally requires third candidate-set share>.95.
CODE improves3.32% for the proxy versus the initial two-row control; the repeat
control was29.29tok/s(+.608%). A previous full CODE bracket independently found
3.23% with.077% control drift. The third gate adds only.056% versus the proxy:
no incremental CODE improvement is established.

Held-out third gating improves1.37% versus its two-row control and1.53% versus
the proxy, but lacks a held-out repeat. It is provisional, not a universal policy.
An observation-only control kept every proxy query/action/acceptance and target
placement identical; its measured speed differed by.033%, inside CODE drift.
Conditional third-position acceptance improves while queries rise1570→1591 on
CODE. Cleaner acceptance alone does not imply better delivered throughput.

The actual new observer components ran on all four cards before models. Full
applicable target-byte/state, seeded, cancellation, long-context and numerical
gates remain before adoption; reply equality alone does not replace them.

## Prefix reuse: lower time to first token

The original13-turn agent harness and its128-token caps remained unchanged.
Both tests below used the same immutable prefix-refill prototype and two-row
resident drafter. These are separate experiments, not one cumulative timing run.

| Experiment | Summed first-token latency, s | Total client request time, s |
| --- | ---: | ---: |
| Prefix cache OFF | 148.393 | 194.921 |
| Prefix cache ON | 101.797 | 147.960 |
| Read-ahead8 with prefix cache ON | 101.524 | 147.720 |
| Read-ahead0 with prefix cache ON | 92.676 | 138.869 |
| Read-ahead8 repeat | 101.523 | 147.904 |

The first pair saves31.4% summed first-token latency and24.1% request time;
it has no repeat. The read-ahead8/0/8 bracket saves8.7% first-token latency and
6.0–6.1% request time with read-ahead disabled. Repeated-control first-token
drift was−.0008%. Decode was flat, so no decode gain is attributed to it.

All26 replies/counts/prompt-ID lists matched in the cache pair; all39 matched
in the read-ahead bracket. Each arm produced1266 tokens with601/670 accepted
drafts and reused47104 of78128 prompt tokens when caching was enabled. About71%
of read-ahead8's posted work was dropped; this is not a physical-drive-read share.
First-token latency includes parsing/admission and should not be relabelled as
pure prompt-processing throughput. The README retains separately measured cold
prefill tok/s and96 GiB cache evidence.

The current2048-token checkpoint spacing leaves13266 tokens between attention
reuse potential and actual linear-state reuse. Finer checkpoints are a future
cost experiment; this is not a promise to recover every token cheaply. Current
restore, giveback, eviction/reuse, tap recovery and cancellation state boundaries
still require fresh qualification before normal use.

## Work stopped after screening

The isolated one-row CPU decode-to-MAC fusion passed all1206 original numeric
cases/4542120 BF16 words plus actual few-entry/GGML/FP-status checks, but its paid
packed-weight timings lost. Large parallel endpoint drift is preserved. Actual
disassembly showed substantial stack/register pressure. It was parked before
Engine integration or GPU qualification. Existing rejected pageable-cache and
bank-lending designs remain parked as described in the matched-decode report.

Next: measure the smaller IQ2_S-down bank together with adaptive depth, and
qualify the measured prefix/read-ahead latency gains. Their earlier independent
benefits are not assumed additive. RAM cache budgets remain configurable.

Local evidence: third-share-workloads-screen-full8-root1,
c2-prefix-refill-screen-root1, c2-prefix-ahead-screen-root1 and
host-m1-decode-mac-root1 under work/gpu-resume-kv-j1-20261106. GPU results were
classified by strict readers after owned-session retirement and same-image
normal production restoration. These screens do not replace qualified serving
defaults, concurrency evidence or the96 GiB-cache table.

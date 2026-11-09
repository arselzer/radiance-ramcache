# Smaller drafter bank with adaptive depth, 9 November 2026

Six fresh servers compared the existing IQ2_XXS gate/up plus MXFP4 down drafter
with IQ2_S down weights. All384 drafter experts remain. Target weights, routes
and arithmetic are unchanged; exactness is against the imported checkpoint
baseline described in the README. This is a prototype screen, not adoption.

Four RX9070XT, one stream,192GiB cache,256MiB HOST pool/GPU, fixed131K context,
2048-token chunks/P1/admit-all. Original CODE16/cap256 uses adaptive.999 depth;
held-out8/cap64 additionally uses the third-token candidate-share gate>.95.
Policies, corpora, caps and fresh-server order were fixed before measuring.

| Workload | Native-down first | IQ2_S down | Native-down repeat | S vs control mean |
| --- | ---: | ---: | ---: | ---: |
| CODE16/cap256, decode tok/s | 29.99 | 30.20 | 30.06 | +0.58% |
| Held-out8/cap64, decode tok/s | 31.78 | 32.10 | 31.96 | +0.72% |

Decode excludes prefill and divides generated tokens by summed server decode
time. Each CODE arm produced3995 tokens; each held-out485. All72 complete replies
and prompt/generated counts matched their controls and historical base references.
The saved responses have no prompt IDs, so no new token-ID certificate is claimed.
Repeats reuse the same corpus and are not additional independent prompts.

Native control drift was+.225% for CODE and+.566% held-out. The S observation
exceeded both controls in each bracket, but one S run per corpus is not a
confidence interval. These are small provisional gains; do not add their
percentages to earlier windows or describe a new universal default.

| Workload | Native accepted/proposed | S accepted/proposed |
| --- | ---: | ---: |
| CODE | 2444/2627 (93.03%) | 2437/2616 (93.16%) |
| Held-out | 249/285 (87.37%) | 253/293 (86.35%) |

S reserves903.0 versus1130.625MiB/card, saving227.625MiB. Actual target GPU units
increased499/509/509/509→512/522/522/522:13 more per card. All96 draft pairs/card
were loaded with zero parked; bank charging preceded target placement. Both
native controls had identical placement. This comparison includes the resulting
target-capacity change and is not an isolated quantized-kernel benchmark.

Whole-run CPU handed-row counters fell705915→700080 on CODE and97920→97084
held-out, about.83%/.85%. They include prefill and changed policy outcomes;
overlapping gather waits do not establish a causal bandwidth split. They are
consistent with the modest size of the measured gain. Better acceptance alone
does not determine throughput.

All-four old-observer/third-tail component cases ran before models. Strict
owned-session/global-GPU retirement and same-image normal production restoration
passed before classification. Full applicable target RAW/state, seeded,
cancellation, long-context and numerical gates remain before adoption.
Serving defaults and configurable RAM-cache budgets are unchanged. The qualified
serving, cold-prefill and96GiB-cache evidence remains in the README.

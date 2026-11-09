# Matched decode screen, 9 November 2026

This separately built prototype compares the plain engine with the full resident
GU-IQ2_XXS/MXFP4-down DSpark bank. Target weights and arithmetic are unchanged.
The imported checkpoint's precision limitations remain as described in the
README; this is exactness against that baseline, not a new official-runner claim.

Four RX 9070 XT cards, one stream, 192 GiB expert cache, 256 MiB pinned HOST pool
per GPU, fixed 131,072-token context, 2,048-token prefill chunks, P1/admit-all,
insert queue 64 and arena lending 64. Both modes use the same host-pool budget.
The drafter's additional VRAM reservation is paid by its placement; target-expert
residency is not artificially held equal. The full bank retains all 384 experts,
96 gate/up and down pairs per rank. Its existing adaptive policy selects zero,
one or two proposals from two-row draft queries.

| Original corpus | Plain decode, tok/s | Resident DSpark, tok/s | Change | Accepted/proposed drafts |
| --- | ---: | ---: | ---: | ---: |
| Held-out8, cap64 | 28.38 | 31.21 | +9.99% | 234/264 (88.64%) |
| CODE16, cap256 | 24.69 | 29.04 | +17.63% | 2,246/2,394 (93.82%) |

Decode is total generated tokens divided by total server decode time, excluding
prefill. Each condition uses a fresh server with no warmup: 485 output tokens for
held-out and 3,995 for CODE. There is one pair per corpus, no repeat or measured
control drift. All 48 full replies and prompt/generated counts matched between
conditions and the available original historical references. No token-ID claim
is inferred from older results without that field.

These are preliminary matched results, not an adopted configuration or a new
absolute-speed record. Remaining numerical/state, seed, cancellation and long
context gates precede adoption. Serving defaults are unchanged. The full CODE
result does not indicate a regression from the older resident GU's 28.48 tok/s
mean; its roughly 2% cross-run difference is not an established gain. Historical
"CODE128" named 128 responses across eight arms, not a 128-token cap. Short
CODE4/cap128 screens and older held-out/decode-only profiles remain separate.

Two subsequent experiments were rejected promptly:

- A compressed-bank loan during eight-stream bursts delivered 31.19 versus 31.40
  tok/s for the loaded control, while the actual bankless configuration delivered
  34.80. These are aggregate end-to-end rates including prefill/fill, not the
  decode metric above. Two fills and verified restores cost about 9.9 seconds.
  The loan roles disabled P1 lending and had different target reservations;
  this rejects that implementation/profile, not every possible bank-sharing design.
- Pageable expert-cache backing cut startup 78.05→64.03 seconds and left short
  CODE decode flat at 27.32→27.34 tok/s, but slowed 3,086-token prefill 329.65→172.14
  tok/s and first token 9.365→17.932 seconds. All paired replies/counts matched,
  and actual pageable P1 copies were exercised. Ordinary registered backing stays
  preferred for inference. This is one pair without a drift estimate.

The local evidence windows are `matched-workloads-root1`,
`cohort-N8-screen-root1` and `pageable-cache-screen-root1`, under
`work/gpu-resume-kv-j1-20261106`. Their strict readers classified results only
after session retirement and normal production restoration. These prototype
results do not replace the qualified serving or 96 GiB-cache tables in the README.

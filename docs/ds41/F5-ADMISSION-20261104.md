# F5 admission experiments — 2026-11-04

## A: 6,817-token screen, complete

Window `ab-20261104T115653Z`, Nov4 11:56–12:12 UTC. Unchanged stable engine `f84b23db`, coherent `aql-stable/home-stable` (both homes), libr4d `92bd7719`; four RX9070XTs, 256GB physical RAM, 192GiB expert-cache budget, C2/Lc2048/P1/lent arena, fixed128K context. Original long8k receipt and all16 generated tokens retained in every arm. No default or engine change. Tool source `8a2a0fe4`; optional standalone checker source `df68735c`, binary `6d13948d`.

| Arm | TTFT (s) | Rank0 chunk drive units |
|---|---:|---|
| Ad128 + Kh startup control | 21.597 | 279/296/311/584 |
| Ad128 + Kh | 19.965 | 279/296/311/584 |
| Admit all / insertq64 + Kh | 16.984 | 279/167/138/367 |
| Admit all / insertq64, quiet | 13.244 | 279/167/138/367 |
| Ad128, quiet | 17.163 | 279/296/311/584 |

External reference check PASS: **5/5 replies, original names/order, prompt and generated counts; all3 full prefill digests match qualified CR reference** (`8c583a997ef6f1db`). Raw source/results and strict checker: `work/f5/A-check.log`, `A-summary.json`, original window directory. Startup control is retained separately, not pooled with the matched baseline. Kh instrumented and quiet comparisons are separate.

Quiet TTFT falls 22.84% (17.163→13.244s); this is a fresh single-pair screen, not adoption. Rank0 chunk drive units fall1470→951; all ranks report zero insert-cap drops. Some cache evictions occur (Kh admit-all rank0=48, rank1=8), so post-prefill decode is a required gate. All16-token tails remain recorded; they do not qualify steady decode performance.

## Validation throughput improvement, qualified

Explicit `--qualify` removes benchmark repetition while retaining BAL byte comparisons/guards and all P1 exact cases and former timing-only shapes. BAL still compares2,762,440,704 output values/redzones at the identical nine heights through2048; observed elapsed time9m24s(full F4)→2m28s(qualify F5A). P1 retains52 cases/49,546 rows/371 host rows/813 NaN pairs, zero bad, and synchronizes all160 projection/shape launches. This is test latency improvement, not model TPS. Ordinary benchmark mode remains available. Helper and plugin artifact identities are separate; all served plugins are the unchanged coherent stable cohort.

## B: 48K and same-process guards, complete

Window `ab-20261104T121517Z`,12:15–12:49 UTC, same unchanged stable runtime/plugins. All75 original replies, names/order, prompt/generated counts and both full48K prefill digests match qualified references. Every phase's complete observer snapshots are joined, request deltas exactly1/8, no hidden warm requests. Each admit-all arm has four explicit zero insert-drop reports.

| Quiet ABBA metric | OFF r1 / r2 | ON r1 / r2 | Mean effect |
|---|---|---|---|
| 47,917-token TTFT |100.679 /99.829s|62.044 /61.535s|−38.37%|
| Subsequent N8 server decode rate |46.512 /46.833 tok/s|46.497 /47.827 tok/s|+1.05%, no measured regression|
| Subsequent N8 wave wall |14.065 /14.089s|14.071 /13.924s|−0.56%|
| 6,817 TTFT after original held-out N8 wave |13.325 /12.822s|11.742 /12.016s|−9.13%|

Separate48K Kh pair:109.207→65.447s, both full `87d485202785b186`. Quiet timings above use only fresh matched controls. The startup control is retained separately. Together A/B qualify80 original replies and5 full prefill digests. Production restored at12:48:48 and returned normal ` Paris. The capital` at12:49:39 UTC.

Rate clarification: use the server's `timings.predicted_per_second`, matching the existing comparison tool: means46.672→47.162tok/s. Earlier live notes used raw `predicted_n / predicted_ms` (47.483→47.981); those raw measurements remain in B-summary.json. Both yield about+1%, but canonical rates are the user-facing comparison. Files: `work/f5/B-check-attempt1.log`, `B-summary.json`, `B-server-rates.json`, `B_PASS`.

Adopted in the RECOMMENDED CURRENT192GiB configuration at user request (README0d8102a9: LCP_ADMIT=all, LCP_INSERTQ=64). The general engine fallback stays128; other budgets retain explicit choices. Source tools8a2a0fe4 and README d65bccce/0d8102a9 merged into decode-impl; stable compiled runtime and all plugins unchanged. Keep Ad128 and variable cache budgets; prior small-cache replay found thrashing. 100K and prefix-cached harness extensions are still unmeasured with admit-all. F5C is source-ready/conditional but deferred behind the user's two-batch-overlap/DSpark priority.

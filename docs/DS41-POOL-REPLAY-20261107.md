# Matched decode and pinned-host-pool replay — 7 November 2026

The current ABI33 C2 build reproduced the accepted ABI32 build's held-out decode speed
with the same 256 MiB pinned-host pool. Enlarging that pool changed placement and reduced
single-stream decode throughput. All 64 original replies and prompt/generated counts matched.

| Engine and configuration | Pinned pool per GPU, MiB | Weighted decode, tok/s |
|---|---:|---:|
| Accepted ABI32, common profile | 256 | 29.0653 |
| Current ABI33, common profile | 256 | 29.2731 |
| Current ABI33, common profile | 2,048 | 26.0670 |
| Current ABI33, admit-all + insert queue 64 | 2,048 | 26.9638 |

The common-profile binary comparison is +0.715%; increasing the current build's pool costs
10.952%. The larger-pool admission/queue bundle is +3.440% relative to that common profile.
Those are paired screening results, without an assertion of statistical significance.

The pool is a pinned HOST placement tier per rank, separate from the total 192 GiB pageable
expert RAM cache. Planned target VRAM counts stayed 621/630/630/630; pinned target experts
increased from 3 to 103 per rank and file-tier experts decreased by 100. This moves those
units from CPU-eligible file-tier service into GPU-readable host memory. The logs establish
placement and timing; they do not measure exclusive CPU/PCIe critical-path contributions.

## Method and qualifications

Four RX 9070 XTs, TP4, one sequence/client, fixed 131,072-token context, 2,048-token maximum
prefill chunk, 40 CPU expert workers, total 192 GiB expert cache. All eight original held-out
integer prompts and cap64 were retained on each of eight fresh servers. Common settings
include P1/Lc2048/Qe64/Rb/Vp/Cs; the two middle servers add Ada/Li64. DSpark, the auxiliary
worker, prefix caching and numerical observers were off.

Order: accepted256/current256/current2048/current2048+AdaLi64, then reverse. Each group
has its own endpoint/repeat drift screen. All four groups pass the predeclared 5% bound for
both estimators. Weighted drifts are +1.146%, +0.100%, +4.269% and -3.138%, respectively.
The primary table averages each group's two sum(tokens)/sum(decode-time) rates. The alternative
mean-per-request estimator gives 29.0873/29.2477/26.5923/27.1253 tok/s and the same conclusion.
Startup, prompt processing and client wave wall time are separate fields in the portable data.

Accepted engine SHA256 starts ed9a6be3, with library 92bd7719; the current engine starts
70f126a1, from fbc0f812. Each complete immutable engine/home/checker tuple ran its own full
fresh native matrix before any model request. Native or ABI receipts were not transferred
between builds. The exact earlier f1b1507a historical tuple was unavailable: this is the
preserved accepted stable92 comparison. Eight client/server lifetimes joined with status0
and no forced cleanup. Both research locks, explicit four-card selection, all-five-GPU
foreign-client checks, owned service pause and production QFN stop/restore were enforced.
QFN answered its normal completion before result analysis.

## Practical consequence and next controls

The published plain configuration already uses a 256 MiB pool. The full HOST DSpark expert
bank needs 1,721.25 MiB per rank from the pool, so its separately qualified configuration
uses a larger allocation. The planner reserves that bank before filling target HOST slots.
Bank-loaded/drafting-off and drafting-on comparisons are therefore needed to distinguish
placement costs from speculative decoding benefits. Current CODE16/cap256 measurements
also use a different workload from these held-out/cap64 requests.

The next pool comparison covers N4/N8; prefill and whole-machine 128 GB conclusions require
their own scopes. Exact hot-draft caching with HOST fallback and CPU service of eligible
pinned target experts remain implementation candidates. This report changes no runtime,
installed serving configuration or default.

[Portable paired metrics and original hashes](pool-replay-20261107.json) retain all eight
arm rates, both drift estimators, actual placement/KV observations and runtime identities.

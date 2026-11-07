# BetterBench quick screen and matched workloads — 7 November

The run completed48 requests,
including12 warmups and36 measured requests. This is the predeclared quick
screen, with3 measured chat/code samples each,2 per prefill depth and8 per
concurrency level. The426-request full suite and standard combined score
remain unmeasured here;378 requests/six categories were omitted. These
responses have no new semantic-quality or per-case reply-equivalence proof.

Four RX9070XT/256GB host,192GiB expert cache, fixed131072 context,
2048-token chunks/P1/admit-all, plain prototype fbc0/engine70f126a1,
DSpark/prefix OFF. BetterBench used server N8/Pf/Ba; the separate matched
control used N1/C1. Installed stable f84b23db/libr4d92bd7719 was not a
comparison arm. Existing imported-model/Engram precision caveats still apply.
No installation/default/adoption change follows from this screen.

BetterBench0.6.1 used temperature0.7/top_p0.95/top_k20/seed20261023 with
original nonce/caps/timeouts. All48 requests completed and joined, with
fresh native gates and normal production restoration. Prefix cached_tokens
were0; this does not assert empty expert-RAM or OS caches. Samples are too
small for tail-percentile claims (the benchmark flags p99 undersampling).

| Measured prompt | Actual prompt → generated tokens | Client TTFT, s | SSE decode, tok/s |
|---|---:|---:|---:|
| chat-followup |93 →200|1.434|23.25|
| chat-clarify |78 →160|1.189|25.91|
| chat-medium |84 →400|1.130|26.73|
| code-gen-medium |61 →198|0.928|24.35|
| code-debug-medium |58 →392|0.962|28.28|
| code-sql-short |61 →237|0.879|29.21|

These are the3 measured samples/category after3 warmups/category.
SSE decode is `(completion_tokens−1)/(last_content−first_content)` on the
client. Updates may batch multiple tokens. End-to-end single-request rate
instead divides completion tokens by complete request wall time.

| Requested prefill depth | Actual tokens, two measured samples | Client TTFT, s | Tokens/client TTFT, tok/s |
|---|---|---|---|
|8000|5903 /5973|9.529 /9.537|619.49 /626.30|
|32000|23424 /23445|31.637 /30.419|740.39 /770.73|
|64000|47262 /47011|63.867 /63.084|740.00 /745.21|

Each depth had2 separate warmups. Target depth is not actual tokenizer
length. This prefill estimate includes first-token/network latency; it is
different from the existing table's server prompt-processing throughput.

| Concurrency | Measured requests | Complete wave wall, s | Aggregate end-to-end, tok/s |
|---:|---:|---:|---:|
|1|8|121.742|23.54|
|4|8|93.469|30.37|
|8|8|87.006|33.66|

Aggregate rate is total completion tokens/full wave wall, including prefill
and scheduling. It is not eight times a per-request decode rate. Startup
is excluded from these request/wave metrics.

The separate exact H/C/C/H control completed48 original calls on four fresh
N1 servers, same fbc0 cohort, with HELDOUT8/cap64 and CODE16/cap256:

| Workload | Actual prompt/output totals per arm | Generated-token-weighted decode, tok/s |
|---|---|---:|
|HELDOUT8|312 /485|26.7383|
|CODE16|503 /3995|24.6898|

Each entry averages its two arm rates: sum(generated tokens)/sum(server
decode seconds), including the first generated token in the declared count.
This differs from both SSE and end-to-end metrics. Weighted endpoint drift
was4.0247%; CODE repeat spread−0.5817%, within5%. All replies/counts matched
their own references. Different prompts AND caps prevent a causal content,
routing or engine-speedup conclusion; this does not replace the older stable
held-out/concurrent table or the separate matched DSpark gain measurement.

The first BetterBench post59686870 failed its reader's warmup-order check.
Recovery SOURCEe6212549 corrected only its expectation of the benchmark runner's measured-index reset
to zero, then reparsed the same48 requests. Original failure remains; no
request/cap/setting/result or GPU/model/service run was changed or repeated.
Recovered status is QUICK_SCREEN_NOT_FULL_BETTERBENCH_COMPLETE; matched
status EXACT_48_CALL_MATCHED_WORKLOAD_SCREEN_PASS. Neither adopts a change.

[Portable selected-field measurements](betterbench-screen-20261107.json) retain the original
result hashes, settings, sample counts and failed-reader recovery history.

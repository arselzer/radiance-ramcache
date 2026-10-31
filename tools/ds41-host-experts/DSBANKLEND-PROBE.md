# Mixed-traffic bank-lending probe

Source only; no syntax check, unit case, client, model or service command was executed in this
lane. `ds41_bank_lend_probe.py` is separate from `ab_requests.py` and all existing prompt sets.
It submits exactly one long anchor and three bursts, with no warmup. Root owns integration,
serialized checks, coherent builds and the two-lock QFN-stop/restore window.

The frozen workload uses `ab_requests.framed` and the server's existing `/tokenize` content form.
It sends framed token arrays through streaming `/v1/completions`, temperature0/seed1, with
1024 anchor tokens and256 per burst. Each prompt asks for substantially more output than its
cap; supported `ignore_eos=true` makes the cap deterministic. `AB_MAX_TOKENS` is refused during
preparation. This probe does not use chat-completions parsing or modify harness prompt sets.

A fresh server must report zero admitted/running/waiting/in-flight requests and an empty request
table. The client observes the anchor decoding alone with at least16 generated tokens, then
starts exactly three bursts. It requires a `/stats` snapshot in which all four requests are
actually decoding with output, not merely queued/prefilling. All bursts must finish before the
anchor. The anchor must then decode alone for at least32 more observed tokens with remaining
capacity; bank arms must expose a positive outstanding draft in that phase. An early anchor,
missed concurrency, timeout, foreign request, error, restart or incomplete drain is INVALID.
Polling defaults to100ms and240s with explicit upper bounds; generation connections time out
and invalid workloads close their streams rather than leaving a hidden run behind.

HTTP completion IDs are random `cmpl-*`; they are not scheduler sequence IDs. Four distinct frozen
(prompt token count,max_tokens) pairs bind names bijectively to numeric `/stats` IDs. The receipt
records both identities, preserving anchor/burst order across runs. Changed IDs, counter
regression and ambiguous or foreign table rows fail. Prompt arrays are frozen once and compared
exactly between arms. Generated token IDs are not exposed by this OpenAI streaming endpoint;
the gate compares complete UTF-8 reply bytes and prompt/generated counts. It never fabricates
HTTP-ID-to-sequence conversion or claims a generated-token-ID oracle.

## Launcher contract

Only after its own validated lock acquisition and production stop, the launcher writes a guard:

```json
{"production_container":"radiance-flashnext","production_stopped":true,
 "both_research_locks":true,"lock_holder_pid":713246}
```

The PID is the current live lock holder, not a fixed value. The client requires `--window-guard`,
checks the holder is alive, opens BOTH exact research lock files read-only and requires their
nonblocking exclusive lock attempts to contend, and uses read-only `docker inspect` to require
production is stopped. It checks again at the end, preserving identical attestations. There is
no service stop/start or GPU API in this client. The launcher must keep both locks and retain its
normal restore-and-answer check after all clients finish; the client does not replace that.

Within that guarded window, prepare the workload once against the first fresh reference:

```
python3 tools/ds41-host-experts/ds41_bank_lend_probe.py prepare URL bank-workload.json --window-guard guard.json
python3 tools/ds41-host-experts/ds41_bank_lend_probe.py run URL bank-workload.json nobank.json --role no-bank --window-guard guard.json
```

Restart the research server with identical sampling, placement/cache budget and prompt policy
for each role; do not warm it. Run `static` with the same DSpark bank and existing Dx, lender off;
run `lend` with that bank/Dx plus the new lender. C owns the exact Engine flags and harness:
`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND=1`, explicit positive
`..._BANK_LEND_BACKUP_MIB`, optional `..._BANK_LEND_GRACE_MS` and
`..._BANK_LEND_COOLDOWN_MS` (both default1000). Existing Dx is required because suppressing queries
alone does not defer bank context work. Initial generic-loan/Qe/Ql/TBO conflicts remain refused.
Root should run current192GiB and separate96GiB functional cohorts; never compare different
cache budgets as reference and candidate for a speed claim.
Use at least four scheduler slots and an explicit max-model-len of at least the frozen anchor's
prompt token count plus1024 (2048 is sufficient for the supplied prompts). The usual512 default
cannot admit this probe; a refusal is INVALID rather than evidence about lending.

After process shutdown/complete logs and production restore, validate offline:

```
python3 tools/ds41-host-experts/ds41_bank_lend_probe.py validate --no-bank nobank.json --no-bank-log nobank.server.log --static static.json --static-log static.server.log --lend lend.json --lend-log lend.server.log --output bank-summary.json
```

All four texts, generated counts, full prompt arrays and total prompt counts must match both
references. Proposal IDs need not equal a no-bank run. Static-bank/lender-off versus lender-on
proposal/capture binding is a separate optional gate with the existing observer tooling.

## Activity proof and cost

The strict validator consumes the agreed `DSBANKLEND` key=value schema. It requires ordered
lend begin/ready and restore begin/ready for each loan epoch, all-rank deferred bank context/query
work, one parked/drained row per action/rank, actual copied target bytes and lent slots on all
four ranks, and all-rank restore byte verification, aliases/masks and matching expected hashes.
The selected restore must bind the actual surviving anchor sequence, then a successful real
post-restore query must emit `resume ready=1 query_rows>0` after READY. A terminal idle restore
with seq0 and no query is allowed as additional cleanup, but cannot establish anchor resume.
A READY flag or a concurrent plateau alone does not pass. Startup backup capture is separate
from a loan epoch. Missing, duplicated, zero-byte, out-of-order or partial-rank proofs fail.

Receipts preserve server `startup_s`, monotonic submission/stream times, the full bounded stats
trace, complete workload-through-idle-drain time and their summed load/workload/drain time.
Bank copy, target fill, strict restore verification, hysteresis and resumed drafting happen
inside that measured workload duration. Summary throughput uses all1792 generated tokens over
load/workload/drain, not a hand-picked concurrent plateau. The outer launcher's QFN stop/restore
and exact process-lifecycle times must remain beside this artifact; they are not inferred from
`startup_s` or reported as measured by the client. Any separate cancel experiment must use the
actual supported streaming socket disconnect (`server.cpp` on_close calls cancel_all), never an
invented cancel endpoint. The default exactness/performance receipt completes all four requests;
it deliberately does not mix variable partial cancellation counts into that comparison.

Prepared `ds41_bank_lend_probe_test.py` fake HTTP/state source cases cover the four-request
sequence, guard failure, binding/counter errors, real concurrency/solo-tail requirements, SSE
bytes/errors, restore/hash/alias/mask/drain failures and complete three-reference comparison.
Root must run these through the build lock; no fake case is a performance measurement.

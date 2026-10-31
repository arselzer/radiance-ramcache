# Extract matched quiet DSpark costs

`dspark_cost_extract.py` validates the measurement logs and writes the existing
`ds41-proposal-costs-v1` input for `dspark_prob_analyze.py`. It performs no fitting, threshold
selection, request generation, engine work or network access. The preparation agent has not run
it or its tests. The integration owner runs them under build.lock/nice19 after the active window.

```sh
python3 -I tools/ds41-host-experts/dspark_cost_extract.py \
  --manifest /path/dme-cost-manifest.json --out /path/matched-costs.json
```

The manifest is mandatory. Logs establish evidence that an instrument ran; they cannot establish
that its binary, libraries, hot set, cache budget, request mapping and deployment settings are the
ones the operator intended. Those are explicit operator assertions with retained provenance.
The helper checks their consistency, hashes the logs/results/manifest/frozen metadata, and checks
each referenced original hotset file's actual bytes. It does not derive config authority from an
arm's name or a startup line, and does not hash the currently installed library as a substitute
for the historical library that executed a window.

## Manifest and training scope

Use this shape, replacing all example paths/hashes/assertions with the measured sources. Full
40-character engine commits and lowercase64-character SHA256 values are required. Paths are
absolute or relative to the manifest. Include all coherent plugin hashes, at least `libr4d.so`
and `deepseek41.so`. No memory/cache capacity is hardcoded by this tool.

```json
{
  "schema": "ds41-matched-cost-manifest-v1",
  "measurement_profile": "actual commit,256GB hardware,current192GiB cache,Db2 profile and window",
  "same_drafter_observer_wait": true,
  "authority_assertion": "Operator checked source/build snapshot, coherent plugins, common settings and matching early waits.",
  "profile": {
    "id": "window-common-profile",
    "hardware": "actual hardware and startup cache/pool settings",
    "engine_commit": "<40 lowercase hex characters>",
    "common_config_sha256": "<64 lowercase hex characters>",
    "libraries": {"libr4d.so": "<sha256>", "deepseek41.so": "<sha256>"}
  },
  "family_profiles": {
    "code": {
      "id": "code-original-db2-hotset",
      "config_sha256": "<sha256 of full normalized code config INCLUDING code hotset>",
      "hotset": {"path": "/path/original-hot-both-db2.json", "sha256": "<sha256>"}
    },
    "chat": {
      "id": "chat-original-db2-hotset",
      "config_sha256": "<sha256 of full normalized chat config INCLUDING chat hotset>",
      "hotset": {"path": "/path/original-hot-heldout-db2.json", "sha256": "<sha256>"}
    }
  },
  "frozen_split": {
    "metadata": "/path/metadata-frozen.json",
    "sha256": "<sha256>",
    "seed": "ds41-prob-v1",
    "heldout_fraction": 0.25
  },
  "runs": [
    {
      "run": "code-C0",
      "family": "code",
      "k": 0,
      "scope": "cost_train",
      "logging": "quiet",
      "log_level": "info",
      "arm": "<complete actual arm containing bare Ds, Db2 and Dme0>",
      "log": "/path/code-C0.server.log",
      "results": "/path/code-C0.json",
      "profile_id": "window-common-profile",
      "engine_commit": "<same full commit>",
      "common_config_sha256": "<same common config SHA>",
      "libraries": {"libr4d.so": "<same SHA>", "deepseek41.so": "<same SHA>"},
      "family_profile_id": "code-original-db2-hotset",
      "config_sha256": "<same full code config SHA>",
      "hotset": {"path": "/path/original-hot-both-db2.json", "sha256": "<same SHA>"},
      "provenance_assertion": "Actual snapshot/libraries/settings/hotset checked against the window receipt.",
      "request_mapping_assertion": "Fresh server,concurrency1,seq1..N follows these receipt rows; no hidden generation/warmups.",
      "requests": [{"seq": 1, "frozen_run": "code2", "frozen_seq": 1}]
    }
  ]
}
```

Add quiet C1/C2 runs for code, then the three chat runs, with the complete TRAIN-only request
lists. `frozen_run/frozen_seq` refer to the ORIGINAL frozen metadata, while `seq` is the new
measurement server's explicit seq1..N mapping. The helper uses the same split function as the
analyzer; it does not make a new split. For the current plan, code13/chat12 training receipts
already exist; their provenance is recorded in `work/autonomous-20261031/dme-cost/training-receipts.json`.
Do not include held-out requests in quiet cost runs. ab_requests results must contain exactly
the declared names/families in order, no failed/missing responses, concurrency1 and no contradictory
available stats IDs. Quiet logs lack per-request IDs, so the no-hidden-generation mapping remains
an explicit assertion rather than something result order proves.

Normalize only the deliberate control differences: k, prompt/result identities, and audit
logging. Keep the full FAMILY config's original hotset path/hash and every query/placement/cache
setting. Code and chat intentionally use different original Db2 hotsets; config equality is
required ACROSS k WITHIN each family. `common_config_sha256` identifies the shared non-hotset
settings; engine/library/hardware authority stays common. Do not erase hotsets to force one
global full-config digest. The helper does not invent a canonical normalization or compute these
config digests from arbitrary launch text; the operator supplies the reviewed config source.

Each family's fixed controls must use the SAME ordered training prompt/group inventory.
Independent repeats are allowed. Means aggregate integer `cycle_ns` / total steady n over repeats,
not an unweighted mean of arm means. All k must be present; measured means must be positive and
nondecreasing to satisfy the existing cost-file contract. A violation refuses extraction for
investigation/repeat; values are never sorted, clamped, fitted or replaced with historical costs.

## Log checks and optional verbose audits

Each cost run must have exactly one intended Db2 early startup, one invalid-signal-aware totals
line and exactly three costs lines. Only the intended k may accumulate costs. Query, observer and
wait counts must be equal and positive; query_rows must equal2*queries, including fixed0.
Steady samples, cycle_ns and waits must be positive, wait cost must lie inside its cycle/global
wait total, emitted must equal n+accepted, and accepted cannot exceed k*n. The helper checks
rounded printed means against the integer counters and bounds transition/end/cancel/unpublished/
invalid-signal exclusions. Duplicate/unknown early markers, old verify caps, compute-OFF,
adaptive/gated/load-controlled arms and conflicting measurement scopes are refused.

Quiet cost runs require info logging and no Dpc startup/events/cycles or debug proposal-ID lines.
This prices the SAME observer/query/wait without per-proposal JSON/debug overhead. Startup and
summary strings must agree with this profile. A positive steady counter does not make a run an
achieved optimization; these are instrument costs, with the early-wait overlap loss included.

Optional verbose runs use `scope:"audit_only"`, `logging:"capture"`, `log_level:"info"|"debug"`
and a Dpc part beside the intended Dme part. They may include the full chat set, including held-out
groups, because they NEVER enter Ck. Their profile/hotset authority still must match the family.
The helper requires full terminal proposal/outcome pairing and validates each cycle's unique
same-sequence epoch/next-epoch, monotone verifying step, actual anchor advance and accepted prefix,
invalid-depth masks and reach/verified outcomes. Steady cycle sums must reproduce all integer
summary counts/costs/yields exactly; invalid query/cycle counts must agree with trace masks.

The ordinary late-readback Ds2 reference is outside this extractor's matched run scopes. Keep
the existing reply/proposal-ID comparisons against it as a separate gate; never place its cost
in C2. Per-depth unreached/capped/cancelled outcomes stay distinct in the reused strict proposal
validator. Neither an ID join nor an anchor join independently proves the engine's live ctx check;
the instrument's host tests/runtime checks and operator window gate supply that evidence.

The output retains every run's sources, hashes, family profile, request/group mapping, exclusion
counts and optional audit, with `fit_performed:false`. Feed it to the analyzer only after review,
keeping the frozen train/held-out split and exactly one held-out evaluation. Historical0.49/0.70
are acceptance break-evens, not candidate-softmax thresholds. This helper never selects thresholds.

Prepared tests: `dspark_cost_extract_test.py` covers integer weighted aggregation and schema
compatibility; scope/counter/mean/yield/exclusion refusals; explicit provenance, training split and
result mapping; distinct family hotsets and changed bytes; nonmonotone means; cycle/ID/epoch/outcome
joins; and verbose audits excluded from quiet costs. No test has been run by the preparation agent.

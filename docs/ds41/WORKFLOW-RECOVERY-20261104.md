# Claude workflow recovery — 2026-11-04

Recovery snapshot: 11:24 UTC; preservation update 11:32 UTC. This report records recovered local workflow outputs and the status of their review gates. It does not claim a new GPU result. Recovery itself ran no builds, tests, project code imports/execution, GPU work or services. Preservation used Python's standard library only.

The prefill/speculative-decoding research finished, including both skeptical reviews. The aux dynamic-sharing/online-set implementation finished, but **all three post-implementation reviewers stopped at the weekly limit**. Its workflow-level `completed` status is not a review pass. New source inspection during takeover has already confirmed startup defects that the existing host tests missed; repairs and new verification belong in the subsequent handoff/window record.

## Where the results actually live

The UI task IDs and persistent workflow run IDs differ:

| Work | UI task | Persistent run | Actual outcome |
|---|---|---|---|
| Prefill and speculative-decoding research | `wof0gb00v` | `wf_eb5c5a1d-715` | Four investigations, two skeptical reviews and synthesis all finished |
| Aux dynamic sharing and online expert set | `wtndo48we` | `wf_a17d99b4-84b` | Spec, spec review and implementation finished; all three implementation reviews failed at the limit; no verification/fix phase ran |

The local session directory is:

`/home/as/.claude/projects/-home-as-radiance-ramcache/635603c1-406d-403f-ad2a-36bc7f42c610`

Paths below beginning `SESSION/` mean that exact directory. The two authoritative small metadata files are:

- `SESSION/workflows/wf_eb5c5a1d-715.json`
- `SESSION/workflows/wf_a17d99b4-84b.json`

They contain `workflowProgress` states, agent IDs, logs, and final result objects. The corresponding scripts are:

- `SESSION/workflows/scripts/ds41-prefill-specdec-research-wf_eb5c5a1d-715.js`
- `SESSION/workflows/scripts/ds41-aux-dynamic-online-wf_a17d99b4-84b.js`

Every agent's transcript is `SESSION/subagents/workflows/<run>/agent-<agentId>.jsonl`. Complete structured research and spec-review outputs remain embedded there. They can be read without replaying a workflow:

```sh
jq -r 'select(.type == "assistant") | .message.content[]? |
  select(.type == "tool_use" and .name == "StructuredOutput") | .input' \
  /absolute/path/to/agent-ID.jsonl
```

The recovery inspected the two workflow metadata files and selected per-agent outputs, rather than scanning the whole session transcript. It did not copy prompts, private data, or full transcripts into this report.

## Private preservation

The recovered replay files and selected structured outputs are now preserved at:

`/home/as/ds41-decode-review-20261028/workflow-recovered-20261104/`

This is **private local evidence, not public material**. Every directory is mode `0700`; every file is mode `0600`. The preservation pass verified those modes and SHA-256 equality for each copied replay file. No project code was executed.

- `specdec-research/`: all 15 files from `/tmp/specdec-research/`, including the replay scripts, dataset and existing bytecode files. The temporary originals remain untouched.
- `workflow-outputs/wf_eb5c5a1d-715/`: six JSON files containing only the `StructuredOutput` input objects from the four investigators and two skeptics.
- `workflow-outputs/wf_a17d99b4-84b/`: one JSON file containing only the completed spec review's `StructuredOutput` input. The three failed implementation reviewers emitted no structured output to preserve. Other agents' plain-text summaries were not copied as transcripts.
- `manifest.json`: original absolute paths, source sizes and SHA-256s, copied/extracted output paths, sizes and SHA-256s, agent/status mapping, and transcript line numbers for each extraction. Metadata and transcript hashes establish provenance; their full files and prompts were not copied.
- `manifest.sha256`: checksum of the manifest, `4d4ec9fdb3d86d51bdf3b90dafbf52e77699593707442970b4f3a2087f36d0b5`.

The preserved dataset and outputs retain their private status; this preservation does not authorize public publication of the archive.

## Research: completed and usable

Run `wf_eb5c5a1d-715` reports seven agents, 592 tool calls, duration 5,542,629 ms, final timestamp `2026-11-04T11:12:00.463Z`, and no failure logs. Every agent below is explicitly `done`; the script's filtering of failed results did not omit an investigator or skeptic in this run.

| Stage | Agent ID | Recovered output |
|---|---|---|
| Prefill anatomy | `a29d45d61313ad7a2` | Full report and candidate list |
| Other prefill levers | `a0f73fcd14f89554d` | Full report and candidate list |
| Speculative-decoding anatomy | `a047267f6f7316fa4` | Full report and candidate list |
| Speculative-decoding methods | `a1bbc068684756379` | Full report, candidate list and CPU-replay paths |
| Prefill skeptic | `ab5988016b6099594` | Verdicts, corrected estimates and missed levers |
| Speculative-decoding skeptic | `a5c06ab3b4d0e6299` | Verdicts, corrected estimates and seven missed considerations |
| Synthesis | `ad7bf3bb20d691cdc` | Final report and summary |

The durable final report is [PREFILL-SPECDEC-20261104.md](PREFILL-SPECDEC-20261104.md), 640 lines, based on source `f84b23db` and identified raw window logs. The workflow completed research, not implementation or new performance validation. Its forecasts remain estimates.

Useful recovered conclusions, with the skeptics' corrections retained:

- `LCP_ADMIT=all` plus sufficient insertion-queue capacity is the first prefill experiment. Long prompts repeatedly read units while cache slots remain empty. The final report estimates 48K at 2K chunks from about 115 s to 62–68 s, and 100K from about 222 s to 130–140 s. These are cold-server estimates; warm-server gains are much smaller. Admission drops and subsequent decode residency must be measured.
- Cache filling at startup is a separate, already implemented candidate. It costs about 2–3 s at startup, with estimated first-prompt 6,817-token time around 10–10.8 s. It needs an actual prefill window; prior cache-fill measurements were decode measurements.
- Layer-major Phase 2 is not the first implementation priority at the current cache size. Admission captures much of its drive-read saving cheaply. Existing warm chunks establish a compute floor of 55–59 microseconds per row-layer at 2K and about 52 at 4K; optimistic drive-only time models understated that floor.
- The next structural compute ideas are a per-unit CPU/card split for few-row experts, packed P1 execution at block-layout speed, exact row-batched MoE, and balancing streamed experts across ranks. The final report includes these skeptical additions and the busiest-rank correction.
- The zero-code DSpark-plus-worker experiment in the decode-only profile comes before a C2 integration. The corrected estimate is single-stream code +17–24%, chat +3–10%, relative to the same worker profile. The pair has not been measured together.
- Prompt lookup currently disables issue-ahead globally; this tax can outweigh its small gains on ds41's own sets. With that fixed, the report estimates only 0–2% decode gain on current harness/code sets. Larger edit-heavy proxy gains are not ds41 measurements.
- The report retained the need to price query-only DSpark overhead, calibrate same-sequence verification with the worker, investigate snapshots within a verify pass, remeasure the draft-bank tax with the worker, and gate at the intended temperature/thinking settings. No omitted blocking skeptic objection was found in the recovered structured outputs.

The methods investigator's replay material remains at **`/tmp/specdec-research/`** and is now copied to the private durable `workflow-recovered-20261104/specdec-research/` directory above. Files include `simulate.py`, `hybrid.py`, `ds_bucket.py`, `transcripts.py`, `dspark_capture.py`, `recover.py`, `tok_check.py`, `runlen.py`, `armstats.py`, `build_dataset.py`, `dataset.json` and `transcript_list.txt`. Recovery did not execute the scripts or relocate their originals.

The companion route-replay material is already durable at `/home/as/ds41-decode-review-20261028/aux-followups-20261104/`, including `traces.pkl` and the `q1_*`, `q2_*`, `q3_capacity.py`, `gains.py` and `validate.py` scripts.

## Aux: implementation recovered, review unfinished

Run `wf_a17d99b4-84b` reports six agents, 580 tool calls, duration 10,811,385 ms and final timestamp `2026-11-04T11:12:58.425Z`.

| Stage | Agent ID | Actual state |
|---|---|---|
| Spec | `af993405ec96fd268` | `done` |
| Spec review | `a0ca6462ad816926b` | `done`, 14 concrete findings |
| Implement | `ae4d7df2e83335e2e` | `done` |
| Exactness review | `a96a769f883af4ef1` | `error`: weekly limit |
| Lifecycle/concurrency review | `af339887b99a621b6` | `error`: weekly limit |
| Sharing/harness review | `af48183ddac7d5929` | `error`: weekly limit |
| Finding verification | — | Did not run |
| Confirmed-finding fix | — | Did not run |

The workflow logs explicitly contain all three failed-review lines, followed by `review: 0 findings, 0 confirmed`. The result object has `confirmed: []` and `fix: null`. No failed reviewer emitted a `StructuredOutput` verdict. The script maps absent findings to an empty array, filters them out before verification, and returns successfully. Thus **zero findings means no completed implementation review output, not zero defects**.

Recovered implementation at the time of the limit:

- Branch `claude/ds41-aux-dynamic`, base `e1f037e9`, worktree `/home/as/ds41-claude-build/wt-aux-dynamic`.
- Commits `df342389` (sharing/controller and online set), `501bd9a4` (device-check/harness/documentation work), and `e9aafb1a` (two implementation fixes). This describes Claude's snapshot; takeover repairs may advance the worktree afterward.
- Built homes/binaries: `/home/as/ds41-claude-build/aql-aux-dyn`, `home-aux-dyn`, and `dc-aux-dyn`.
- The implementation report and `/home/as/ds41-claude-build/work/aux-dyn/README-WINDOW.md` retain artifact SHA-256s. They apply to the original `e9aafb1a` build and must be refreshed after repairs.
- Spec and review responses: `/home/as/ds41-claude-build/work/aux-dyn/SPEC.md`, with pre-review version `SPEC.md.pre-review`. Its section R records the implementer's response to the 14 spec findings. The original complete findings remain in agent `a0ca6462ad816926b`'s structured output.
- Host validation logs exist at `work/aux-dyn/test-final2.out`, `light-m1c.out` and `dc-final.out`. The final test log records the per-test passes; the implementation report claims 32 selected host tests passed. The light log records all listed shell tests passing, including 537 aux-harness checks and the request tests. These are recovered results, not tests rerun during recovery.
- `work/aux-dyn/replay/q5_engine_rule.py`, `run_q5.sh` and `q5.out` retain the implementation-policy replay. The host planner fixture compares the replay rule with the C++ planner.
- `work/aux-dyn/chainAD1.sh`, `arms-ad1.txt`, `ad1-dryrun.sh` and `ad1-dryrun.out` exist. The planned chain has 14 arms: online and toggle verification gates first, held-out/chat N8 A-B-B-A, copy-only traffic pricing, harness comparison, and a released-start/foreign-tenant arm. It was **written but not launched** by this workflow.

The spec review addressed material hazards: `dead held` versus `dead freed`, startup control before the long model load, holder liveness via flock, early tenant detection, read-back of table entries, full-size zeroed hit buffers, bounded stop/drain paths, and separate online/toggle verification gates. The implementer's claim that these were addressed is not a substitute for the missing implementation review.

### Valuable partial-review results and takeover audit

1. **Released-start views: recovered concern, then source-confirmed on takeover.** Exactness reviewer `a96a769f883af4ef1` noted that `Worker::plan()` creates fresh `RankView{}` objects, while only `aux_load` sizes `resident` and `slot`. It explicitly noticed that `aux_fence_test` always starts with a loaded set. With a busy/off startup, the loader is skipped; subsequent acquire can copy units but boundary publication silently rejects indices outside the empty arrays, leaving the worker unable to offload. The remark is at line 147 of that agent's transcript. The takeover source auditor confirmed the gap in `core/place/aux_expert_worker.cpp`, `core/engine_aux.cpp` and `core/place/aux_control.cpp`. A repair and a true released-start host regression were requested; no pass is asserted here.

2. **Startup arena observation race: confirmed by takeover source audit.** `core/engine_aux.cpp` computed `released = !aux_->has_arena()` before taking the startup claim lock. The controller's `poll_share` can release the reserved arena and write `arena_` concurrently. This is an unsafe observation and can select the wrong startup path. It is a new source-confirmed finding from takeover, not a completed Claude reviewer verdict. The aux lane is repairing it with the released-start coverage.

3. **Startup mode-string race: confirmed by takeover source audit.** `poll_share` writes `mode_` and `mode_src_` before `start_mu_`, while `attach` holds that lock and publishes the strings. The shared string access is a C++ data race. This also needs the takeover repair and verification record.

4. **Worker status observations: confirmed by takeover source audit; repair prepared, not verified.** Startup/control-thread release or reacquire changes arena presence and free-memory status while other threads may report the worker's status; online flips also change its held-unit count. Plain shared status fields (or observing the owned arena pointer directly) permit C++ data races. The prepared change in `core/place/aux_expert_worker.h` and `.cpp` publishes atomic status snapshots for arena presence, free-memory status and held-unit count, while keeping the actual pointers under the startup handoff or closed/drained gate. The snapshot fields do not replace ownership/fencing. No new test or device pass for this repair is asserted here.

5. **Online-only bad-table handling: confirmed by takeover source audit; repair prepared, not verified.** In online-only mode there is no sharing gate. If the swap queue's full-table read-back differs, draining/freeing the arena from the controller can race rank threads that still own and submit to their lanes. The prepared `core/place/aux_control.cpp` change marks the worker dead-held, requests the next boundary to clear its views so subsequent work uses the CPU, and retains the arena until teardown after the ranks join. It does not attempt a concurrent free. Sharing-gated operation retains its separate drain/free path. Host coverage and a GPU transition/error gate remain unverified at this snapshot.

6. **State-file temporary-name collision: partial suspicion, not confirmed.** Lifecycle reviewer `af339887b99a621b6` noted two possible callers of `publish()` using the same temporary file (transcript line 121). The takeover source auditor found startup/watch publications serialized by `start_mu_` and teardown joining the thread; the specific collision scenario was not confirmed. The separate mode-string race above is real.

7. **AQL pointer-locate cache after cross-thread arena reallocation: interrupted question, subsequently dismissed by source audit.** The same reviewer stopped while checking this (transcript line 166). The takeover source auditor found global `loc_forget` generation invalidation in `a_free`. No bug was established by this lead. Actual release/reacquire GPU behavior still requires a window.

8. **Sharing/harness reviewer: unknown verdict.** Agent `af48183ddac7d5929` performed many source/log reads, including tenant phases, traps, AxF timeline, gates and coverage, but left no visible concrete finding or structured verdict before the limit. Silence is not approval; this review remains unfinished.

The implementer had already found and repaired a separate slot-lifetime hazard: an abandoned late job may still read its old slot, so a unit with that old slot pending-free must not be recopied and have its table entry rewritten early. The recovered report describes this fix; cross-queue/device visibility and full transitions remain hardware gates.

## Stale task state and precise stop boundary

Claude's job state is `/home/as/.claude/jobs/635603c1/state.json`; its timeline is `/home/as/.claude/jobs/635603c1/timeline.jsonl`. At line 2360 the root job was finding the admission and DSpark arm specifications; line 2361 records `blocked` at `2026-11-04T11:12:19.615Z`, with weekly-limit reset Nov 9, 21:00 UTC. The aux reviewers hit the same limit around this boundary. The research synthesis had already finished.

The remaining task files are under `/home/as/.claude/tasks/635603c1-406d-403f-ad2a-36bc7f42c610/`:

| File | Recorded state | Interpretation on recovery |
|---|---|---|
| `76.json` | `in_progress` | Prefill roadmap, partly stale after later measurements/research |
| `90.json` | `pending` | Carry/planes-pool stall follow-up; research now says admission may subsume its benefit |
| `91.json` | `pending` | Explicitly parked growing-KV repro; not an actively running experiment |
| `93.json` | `in_progress` | Aux implementation exists; review and hardware gates remain |
| `94.json` | `pending` | Stale: research itself is complete; its experiments remain to run |

## Continuation gates

1. Preserve the completed research and its replay inputs. Use its corrected, current-cache plan rather than restarting research or treating old estimates as measurements.
2. Finish the aux implementation review and the confirmed startup repairs. Update host coverage for a genuinely released start. Refresh affected artifacts and hashes under the load rules.
3. Audit the old chain's production stop/restore/version assumptions against the current handoff before using it. The saved script is a template, not authority to operate production.
4. Run admission/cache-fill calibration and aux hardware gates only within a properly stopped/restored production GPU window holding both research locks. Verify unchanged replies before accepting any timing.
5. Record the new tests, exactness results, performance and any merge/publication separately in HANDOFF/NIGHT-LOG. This recovery snapshot does not predeclare those outcomes.

# chain: implementation report

No GPU run was possible, so both switches are built and pass the host tests but have never run on a card, and nothing is measured. The fusion candidates in item (3) come from reading the issue order and the profiled op table, not from the instrument, which has not been run yet.

**Branch and commits**: `claude/ds41-chain` in `/home/as/ds41-claude-build/wt-chain`, based on dc83cecc, not pushed.
- `d675ea55` Live issue path: hazard-checked overlap and the chain trace (opt-in)
- `55e81d4b` Live chain: tags unique per context, a launch never beside its own op

**Switches and arm parts** (both off by default; with both off, behaviour is identical and the backend only updates two counters per queue):
- **`RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1`**, part `Ov` (decode passes); **`=2`**, part `Ov2` (every pass issued live, prefill too).
  - It does for live passes what `haz_mark` does for recorded ones: a launch's first dispatch drops the barrier bit when it reads nothing the dispatches in flight on its queue write, and writes nothing they read or write.
  - Only the order kernels start in changes, never what they compute.
  - Every packet is tagged with the launch that wrote it. Anything the issue path did not write (events, waits, copies, CPU-row blits, table patches, link holds) makes the next launch keep its bit.
  - These never go unordered: collectives, kernels without `rad_kernel_concurrent`, host-visible launches, a second launch of an op that is still in flight, a dispatch carrying a ride, system-scope fences, chained launches, and anything past 6 unordered packets in a row.
  - The check is in `core/runtime/live_chain.{h,cpp}` and `Ctx::live_before`/`live_after` in `issue.cpp`; the backend side is in `aql.cpp`.
- **`RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE=N`** (output prefix from `_TRACE_OUT`), parts `Ct` (N=16) / `CtN`:
  - Rank 0, one trunk decode pass in N.
  - `chain-<arm>.ops.csv`: one line per launch with layer, op, kernel, queue/packets, unordered/would/why/clash, its dependences on earlier launches, and the buffers it reads and writes.
  - `chain-<arm>.stamps.csv`: card-0 start and end per kernel.
  - Read both with `tools/ds41-host-experts/chain_report.py <dir>/chain-<arm> [--layer N]`.
  - Without `Ov` it is the dump-only arm: no packet changes, but there is host overhead, so it is not a timing arm.

**A/B arms** (with `MIRROR` set as for the best arm):
`B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2`
`ARMS="$B-Ct16 $B $B-Ov $B-r2 $B-Ov-r2 $B-r3 $B-Ov-r3 $B-Ov2 $B-Ov-Ct16"`
- `Ct16` comes first because it exercises the new backend path without clearing any bit.
- `Ov2` is for TTFT.
- The last arm gives card timestamps with the overlap on.

`launch_window.sh` copies `aql/bin/radiance` and the main tree's `ab_window.sh`, which does not know `Ov`/`Ct` and now skips arms with unknown parts. Either merge this branch and rebuild `aql`, or point a copy of the launcher at `/home/as/ds41-claude-build/aql-chain/bin/radiance` and this worktree's `ab_window.sh`. Merging should only conflict in `ab_window.sh` and the README, because the FP8 merge changed those since dc83cecc; none of the core files I touched have moved.

**What to look for in the server log:**
- **At startup (rank 0):** `live chain: independent dispatches overlap in decode passes`. A `W live chain: ... are off` line means the switches did nothing.
- **At shutdown, one line per rank (new counters):** `live chain (...): P passes issued live (D decode); L device launches checked, F with nothing in flight to wait for, U first dispatches went unordered (R permits the backend kept ordered); the others kept the bit: <foreign / clash / opaque / host-visible / after / run cap>; a decode pass: N launches, F free, U unordered`.
- **Expected values:**
  - With `Ov`, U should be about F and R about 0.
  - With `Ct` alone, U is 0.
  - F per decode pass should be roughly 300 or more: the offline model of these rules frees 6, 9 and 17 launches per plain, indexer and source layer before the counts sync, and the unit test reproduces the six for a plain layer.
- **Replies:** 8/8 identical (`ab_sum.py`).
- **If it hangs:** a "no queue has advanced" dump would indicate an ordering problem.

**Built artifacts**: `/home/as/ds41-claude-build/aql-chain/bin/radiance` (md5 16f651d8…), built from the committed tree. `libr4d.so` and `deepseek41.so` are unchanged; use `home2` as for the best arm.

**Tests** (host backend, cards hidden): `live_chain_test` (new, 14 cases), `host_expert_pool_test`, `place_test` and `mapped_slab_pregrow_test` all pass, and the build adds no warnings in the files I touched.

**Fusion candidates (3)**, exact only under the conditions given, ordered by estimated saving:
1. **Router GEMV + sqrtsoftplus top-k + scatter in one launch**, with an arrival counter per op instance: 2 launches and 2 gaps off every layer's chain, about 8–12 µs a layer. Top-k + scatter alone saves about half that.
2. **`q_rope` in `q_b`'s epilogue**: the rope applied to the bf16-rounded outputs with fp-contract off, about 4–6 µs a layer.
3. **`o_f` (inverse rope + fp8 round trip) in the attention epilogue**: about 4–6 µs a layer.
4. **`q_an` into `q_a`** through a last-block finisher that keeps rmsnorm's reduction order: about 4–6 µs a layer.
5. **`mhc_mix` as one launch**: about 3 µs per mix; `mix_a` is already hidden by `Ov`.
6. **`kv_st`→`kv_f`, `k_st`→`k_f`, `l_st`→`l_f`**: exact, but off the critical chain once `Ov` runs them alongside the query chain.

Not exact as they stand: one GEMV over `q_a`+`kv` (the wider N picks a different split and changes kv's bits) or over both `o_a` groups, unless each segment keeps its own split. Together, candidates 1–4 are an estimated 0.5–1.2 ms a token on top of the overlap; a `Ct` arm should rank them.

**Unfinished or uncertain:**
- Exactness is argued (only start order changes) and modelled in the unit test, not yet measured.
- The gain is unmeasured. The offline model's ceiling is about 2.8 ms a token (~4%); the research estimate is +1.2 to +3.5%.
- Either switch adds about 0.1–0.3 µs of host time per launch.
- `Ov2` is untested in prefill.
- None of the fusions are implemented; they are kernel work in libr4d.

# chain: review

I found nothing in the two commits that can change an output bit, hang the card or crash the engine. With both switches off, the engine sends exactly the same packets as before. The arm parts set the switches, but only in this branch's `ab_window.sh`. Two problems in how the A/B would be launched must be fixed first, or the window measures nothing.

I did not build or run anything: a window was pending and then running. This is a reading of `d675ea55` and `55e81d4b` against `dc83cecc` in `/home/as/ds41-claude-build/wt-chain`. The binary `/home/as/ds41-claude-build/aql-chain/bin/radiance` has md5 `16f651d8…`, and its object files are newer than the committed sources.

**Must fix before a GPU A/B**

1. **[High] The standard launcher would skip every new arm.** `work/launch_window.sh:15` copies the main tree's `ab_window.sh` and `/home/as/ds41-claude-build/aql/bin/radiance`.
   - The main tree's script (`ea27e5c9`) now skips any arm with an unknown part (its line 332, `skip=1`).
   - `aql/bin/radiance` has no live chain, so only the `$B` arms would run.
   - Fix: either merge and rebuild `aql`, or point a copy of the launcher at `aql-chain/bin/radiance` and this branch's `ab_window.sh`.
   - Merging conflicts only in `ab_window.sh` and the README. Put `Ov`, `Ov2` and `Ct|Ct[0-9]*` above main's `*) … skip=1` line. Keep main's second grep line and add `live chain` to the first. In the README, drop the branch's old `PACKED_CACHE … not yet measured` row.
   - `core/engine.cpp` has also changed in main (+62 lines), contrary to the report, but it merges cleanly.

2. **[Medium] The run home does not match the binary.** `home2` now holds the FP8-merge builds, identical to `home4`: `libr4d.so` md5 `fdb77e8d…` and `deepseek41.so` `2b86f304…`, both written at 13:57. `aql-chain/bin/radiance` is the `dc83cecc` engine.
   - `abi/` did not change, so it probably loads with F8 off, but this pairing has never been run. The report's "unchanged; use home2" is not accurate.
   - The recommended configuration is now `…-Sc2-F8` (~17 tok/s, NIGHT-LOG 13:58–14:14), which changes the dense GEMVs, so the gain should be measured on it.
   - Recommendation: merge into `claude/ds41-decode-impl`, rebuild, and set `B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8` with the FP8 home.
   - The new `gemm_bf16f8_nt_m64` kernel finishes split-K inside each block and keeps no state between launches, so it is safe to overlap.

**Code findings (none blocks the A/B)**

3. **[Low] A blit copy or fill can take the permit.** The comment at `core/device/aql.cpp:1103` says a fill the launch writes first is not covered by the check. But `live_permits` (`aql.cpp:1111-1116`) cannot tell the op's kernel from a device-local blit fill or copy, so a launch that writes one first gets the permit for it.
   - The only path today is libr4d's first-use counter setup (`split_ctr`, `mxd_out_ctr`, `mxgg_out_ctr` → `hipMemset`, `aql.cpp:4302`) on lane 0. It fills fresh memory and drains right after, so it is harmless.
   - Fix: clear `t_live.permit` in `a_memset_async`, `a_memcpy_async`, `blit_copy` and `hipMemset`.

4. **[Low] The trace-only arm overstates what `Ov` frees.** Without the overlap every launch keeps its bit, so only the previous launch counts as in flight (`live_chain.cpp:131-135`). On the test's plain layer that counts 8 free launches (adds `q_an` and `q_b`) against 6 under `Ov`. Judge the gain from the `Ov` arms' "a decode pass" figures, not from `Ct16`. Optionally, have the bookkeeping simulate the permit when the overlap is off.

5. **[Low] Some expected values in the report need care.**
   - "U ≈ F" holds only per decode pass: F also counts free launches in prefill under `Ov`=1 and launches that wrote no packet.
   - Each routed layer adds one "host-visible" launch. The scatter writes the routing counts through a raw pointer into VRAM (`route_dev_`), and any raw write is marked host-visible (`issue.cpp:1243`). This is safe and is not host memory access.
   - With `--profile-ops`, the event records inside the window (`issue.cpp:1045`) keep every launch ordered, so R ≈ F.

6. **[Low] `chain_report.py:33` can miss the layer's end.** It does not list `router_topk_scatter` or `moe_scatter_inv` as the end of a layer's stretch, so a build that uses either gets an empty report. DS41 issues `moe_scatter` today, so it works for now.

7. **[Low] The core of the exactness argument has no unit test.**
   - `live_ranges` (`issue.cpp:1208-1256`), which turns operands into byte ranges, is untested, and so is the backend half (`live_permits`, `live_note`, `a_live_*`).
   - The test's `FakeQueue` copies the backend's rule, so the two can drift apart.
   - Suggested cases for `live_ranges`:
     - weights skipped only when passed as weights;
     - a buffer in a weight slot counted as a read;
     - read-write operands give both a read and a write;
     - the weight table counted as its pointer array;
     - raw writes marked host-visible;
     - shapeless, dtype-less or negative-stride operands, non-concurrent rows and collectives treated as unknown;
     - scratch counted as a range;
     - omitted trailing optional operands.
   - The switch from a live pass to a played tape is also untested.

**Checked and found sound**
- Every packet writer updates the run count and the launch tag: `aql.cpp:1142` (barrier), `aql.cpp:1246` (dispatch), `aql.cpp:1380` (played packets). So anything the issue path did not write is caught.
- A packet written outside a window always keeps the barrier bit. A copy, event, patch or CPU-row blit after an unordered launch therefore waits for it.
- Events, drains, gates and link holds are barrier packets that wait for everything before them, and dispatches carrying a ride are refused the permit. So host waits, lane joins and the CPU-row and stager handshakes still cover unordered work.
- The run cap of 6 counts every writer, so the existing reserve margin of 8 still guarantees argument slots are free. A played tape's first packet after a live pass is ordered.
- The in-flight bookkeeping is correct for single- and multi-packet launches, a barrier inside a launch, foreign packets interleaved, reused slots and unknown queues.
- `live_ranges` follows `haz_note` or is stricter. Its one relaxation, allowing omitted trailing optional operands, matches the ABI's rule that an absent operand is null.
- None of the libr4d kernels that DS41 decode lets overlap keep hidden state that two launches could share:
  - fp8a8 counters are keyed by stream and output; mxfp4 counters by device and output.
  - The router's counter belongs to its op, and the same-op rule covers it too.
  - `mhc_mix` keeps its partials in scratch.
  - The `hc_*` kernels, which have stream-keyed counters, never overlap.
- Lock order introduces no cycle.
- With both switches off, no window opens; the backend only does two stores per packet and two thread-local reads per dispatch.
- The arm parts pass the switches through: `ab_window.sh:288-293` → environment → server (line 131) → `live_chain.cpp:153-156`.

**A/B to run after fixing 1 and 2**

`ARMS="$B-Ct16 $B $B-Ov $B-r2 $B-Ov-r2 $B-r3 $B-Ov-r3 $B-Ov2 $B-Ov-Ct16"`, with `B` the current best and `MIRROR` set.

What to look for:
- **Startup:** `live chain: independent dispatches overlap in decode passes`, and no `W live chain … are off`.
- **Shutdown, per decode pass:** U ≈ F, R ≈ 0, and "foreign" not dominating.
- **Errors:** no "AQL queue faulted" and no "no queue has advanced".
- **Replies:** 8/8 identical in every arm (`ab_sum.py`).
- **Trace:** run `chain_report.py <out>/chain-<arm>` on both `Ct16` arms.

Before adopting `Ov`, run one longer soak (256-token replies, or 16 requests three times), because a hazard the check missed would show up only intermittently.

# early: implementation report

I implemented both parts as separate opt-in switches and committed them, but neither has run on the cards yet, so there is no speed number. Expect small effects: the ranked plan rated issuing the routed card ops early at ≤0.5% (it was on its dropped list) and the single-copy fetch at +0.2–0.7%. With both switches off, behaviour is unchanged.

**Branch and commit:** `claude/ds41-early` in `/home/as/ds41-claude-build/wt-early`, one commit on top of dc83cecc: **ebf8a00e** "Routed layers: card ops ahead of the counts sync; one D2H dispatch (opt-in)".

**Switches and arm parts**
1. `RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1`, part **`Ei`**. It needs `all`, `M16` and `B1`; otherwise the server refuses to start.
   - Before: at each routed layer's gate_up, the issuing thread waited for the counts, chose and submitted the CPU's share, and in decode read the misses from the drive. Only then did gate_up, swiglu, down, the shared expert, the ffn mix and the `Pa` prefetch reach the card, which sat idle meanwhile.
   - Now: the counts and row copies are queued, all of the layer's file-tier units are aliased, and the thread returns without waiting. The wait, the choice, the submit and the miss reads happen at the layer's gather (new `hx_early_decide`, `core/place/stager.cpp`); the copy of CPU rows over the card's rows is unchanged.
   - It is exact: the card computes the same bytes from the same aliases, and the CPU rows overwrite the stray rows before the gather reads them.
   - It applies from a layer's second routed pass on; the first pass keeps the existing deferred path.
2. `RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1`, part **`Fl`**. It needs host experts.
   - The counts row, token rows, sorted order and offsets reach the host in one `rad_memcpy_ranges_async` instead of four separate dispatches.
   - The range lists live on the card, one per layer and pass size (about 300 KiB a rank, so the copy needs no extra cache flush). A list is rewritten in stream order whenever an address changes, so an operand move re-arms it at the next fetch.

**A/B to run, all arms in one window on the same binary:**
```
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2
ARMS="$B $B-Ei $B-Fl $B-Ei-Fl $B-r2 $B-Ei-r2 $B-Fl-r2 $B-Ei-Fl-r2 $B-r3 $B-Ei-r3 $B-Fl-r3 $B-Ei-Fl-r3"   # plus MIRROR=<source-copies dir>
```
- `launch_window.sh` has to take `/home/as/ds41-claude-build/aql-early/bin/radiance` for every arm (the base arms too) and this branch's `tools/ds41-host-experts/ab_window.sh`.
- The main repo's script doesn't know `Ei`/`Fl`: it only logs "unknown part" and runs the arm without the switch.
- The gains are under 1%, so the three repeats are needed.
- Replies must be identical (`ab_sum.py`). Watch the TTFT sums as well as decode tok/s.

**In the server log**
- At startup (rank 0): `host experts: early issue: ...` and `host experts: fetch lists: ...`.
- At shutdown, per rank:
  - `host experts, early issue: N layers ... waited X ms for the counts at their gathers (Y us a layer) ...; R found their counts already in ...; F fetched their rows again ...`. Expect F = 0. A large R means the thread was often behind the card, where `Ei` can lose time: issuing the card ops first delays the CPU's start.
  - `host experts, fetch lists: C ... one ranged copy, A range lists written, P ... separate copies`. Expect P = 0 and A ≤ 640.
- Compare `EP staging time, one-token passes`, `EP decision ... us a layer` (should stay about 35 µs) and the gather waits against the base arms. With `Ei`, the router wait is measured at the gather instead.
- Errors to grep for: `none of its gathers followed`, `no gate_up seen`, `selected units off the CPU`, `reads rows the CPU path cannot`.

**Built artifacts:** `/home/as/ds41-claude-build/aql-early/bin/radiance` (md5 052d90841a5ad1149e46e534005f18bf), built from this commit. libr4d and the deepseek41 plugin are unchanged, so keep the home2 plugins (`H`).

**Tests:** `host_expert_pool_test`, `place_test` (112 cases), `mapped_slab_pregrow_test` and a new `tests/hx_fetch_list_test.cpp` (3 cases) all pass on the host backend. The new test checks that the single range-list copy (`core/place/hx_fetch_list.h`) lands the same bytes as the four copies, including sources on both sides of the base, at every pass size and both row pitches. It also checks that addresses the copy kernel can't take fall back to the four copies, and that rewriting a list in stream order doesn't change what an earlier copy reads.

**Not covered yet**
- No unit test reaches the early-issue path itself in the stager (it needs the DS41 EP4 setup, the pool and the mapped cache); the first GPU window is its first run.
- The range-copy kernel has not run on a card with sources and destinations spread over several allocations. Look for P = 0 and no GPU faults.
- `Ei` gains only where the card's routed work outlasts the CPU's after the counts arrive; in most decode layers the CPU rows are the long pole. In decode miss layers the card now computes during the drive reads.
- A requested M1 prefetch trace keeps the old path.

# early: review

**Review of `claude/ds41-early` (ebf8a00e against dc83cecc)**

Nothing in the code has to be fixed before a GPU A/B. One thing must be handled in the A/B setup first (finding 1). I found no way for an output bit to change, no race, deadlock or lifetime bug, and the path with both switches off is identical to before. Both arm parts really set their switches.

**Must handle before the window**

1. **High (setup, not code): the launcher would run the wrong script and binary.** `/home/as/ds41-claude-build/work/launch_window.sh` lines 10 and 15 copy the main repo's `ab_window.sh` and `/home/as/ds41-claude-build/aql/bin/radiance`.
   - The main repo's script doesn't know `Ei` or `Fl`. It logs "unknown part" and runs those arms as base arms, so the A/B would silently compare identical configurations.
   - `RADIANCE_BIN=...` can be overridden by a trailing VAR=value argument (env takes the last assignment), but the script cannot.
   - Use a launcher that snapshots `wt-early/tools/ds41-host-experts/ab_window.sh` and `aql-early/bin/radiance` (md5 052d90841a5ad1149e46e534005f18bf).
   - In every `Ei` or `Fl` arm's server log, check the rank-0 startup lines `host experts: early issue:` and `host experts: fetch lists:`. They must be absent from the base arms, and window.log must contain no "unknown part".

**Recommended, not blocking**

2. **Medium (risk): the single-copy fetch (`Fl`) has never run on a card.**
   - `tests/hx_fetch_list_test.cpp:68-69` writes card memory directly from the host (`fill(arena.p…)`), so the test only works on the host backend. It would crash on a card.
   - It is also not in the list of tests run on a card (`tests/CMakeLists.txt:268`).
   - On a card, the copy call (`aql.cpp:2403-2434`) only translates the two base addresses. Every other address depends on the card seeing the same address as the host.
   - I checked that this holds for all the buffers involved: the pinned host copies (`counts_host_`, `hx_x_`, `hx_idx_` are pinned the same way, `aql.cpp:3775`), the card's counts row (`route_dev_`) and the card buffers. The kernel takes signed 64-bit offsets, so negative offsets work.
   - Fix: have the test upload its source data with `rad_memcpy_async`, add it to the card test list, and run it once on card 0 through `gpu_cmd_window.sh`. If skipped, a failure would show as a GPU fault or crash in the `Fl` arms, or as a reply mismatch.

3. **Low (robustness): a refused copy fails every later step.** At `stager.cpp:1413`, if the ranged copy is refused (for example `RAD_E_UNSUPPORTED`), the step aborts. The list stays marked as written, so every later pass fails the same way. Falling back to the four separate copies (reset the list's key, count it in P) would be safer. I don't expect it to trigger, since both base addresses are 16-byte aligned card or pinned memory.

**Minor**
- `stager.cpp:1398-1401`: before writing any list, the code waits for the last list write of any slot, although each layer and pass size has its own staging slot. This is conservative and harmless, because the counts wait already passed that write.
- `stager.cpp:2254-2256`: when an `Ei` layer's gather never comes, the message says "CPU rows reached no gather", but that layer's CPU share was never chosen. Cosmetic.
- `stager.cpp:963`: a bad `Ei` or `Fl` setting is refused only after the cache warm (`W`, `stager.cpp:1153`) has run, so the refusal comes tens of seconds late. If the file stager's setup returns early because no layer has file-tier units, the switches are silently ignored. Neither affects the A/B configuration.
- `stager.cpp:1997-2014`: with `Ei`, a gate_up operand move becomes a refetch, where the base path treats it as an error. Production logs show "0 gate_up operand moves" (ab-20261029T124059Z), so expect F = 0.

**Exactness: what I traced by hand**
- **Early issue (`Ei`):**
  - `alias_file_units` publishes every unit at the same address, whatever the order (`mover.cpp:859`). So aliasing all of the layer's file-tier units gives the same table the base path gets from the unselected units plus the CPU's units.
  - With the buffers capped, the CPU takes every selected unit; `hx_early_decide` refuses anything else (`stager.cpp:1530`).
  - The aliases are published in `before_op` with a residency resync, and the weight tables are patched in stream order (`issue.cpp:298-318`).
  - The copies to the host are queued right before the gate_up launches, with nothing writing those buffers in between.
  - After an operand move, the refetch goes behind the first copies and the counts event is recorded again. Each record gets a fresh signal (`aql.cpp:1746-1777`), so the host waits for the latest.
  - `hx_early_decide` waits for the counts before choosing and submitting. `hx_finish` runs in the same `operands()` call, before the gather launches.
  - None of the ops between gate_up and gather read the down rows: shared expert, ffn mix, swiglu, down, shared down, attention prefetch tail (`rad_block_moe_fp8.h:1797-1882`).
  - Reusing the host buffers is safe: the previous layer's pool jobs and worker reads are waited for before the next start, the predictor copies its input (`router_predictor.cpp:140`), and `hx_submit` still waits for the previous layer's row copies before the CPU rows buffer is reused.
- **Fetch list (`Fl`):** the ranges are byte-for-byte the four copies (96 counts, T rows of 5120 bf16, T×6 sorted entries, 97 offsets). Destinations don't overlap. Each list is written in stream order before its copy reads it.
- **Deadlock:** no new host-card dependency. The counts copy comes before everything `Ei` issues early, there is no collective between gate_up and gather, and the pool wait is unchanged.
- **Switches off:** every new branch is gated on `hx_early_li_ >= 0`, `hx_early_`, `hx_fl_` or a counter, and `hx_init_fetch` returns OK with both flags false.
- **Arm parts:** in this branch's script the `Ei` and `Fl` cases come before `*)` and no earlier pattern catches them.

**What I checked of the build and tests**
- `aql-early` is configured the same way as `build_hx.sh`.
- The binary (13:51) is newer than every changed source (13:48), the tree is clean, and the binary contains the new switch strings.
- I ran `hx_fetch_list_test` on the host backend: 3 of 3 cases pass (230 checks). I did not re-run the other tests. They don't reach the new paths, and the implementer reported them passing.

**A/B**
- Run the implementer's 12 arms in one window on the aql-early binary (all arms), with `MIRROR=` set.
- Check that replies are identical with `ab_requests.py --compare` / `ab_sum.py`.
- Shutdown counters should show F = 0, P = 0 and A ≤ 640 per rank.
- In the `Ei` arms, the counts wait is measured at the gather, and "ms waiting at gathers" will grow because the pool is submitted right before its wait. Neither is a regression signal.
- Judge on tok/s, the TTFT sum, and the share of layers whose counts were already in (R out of N). A large share means the issuing thread was behind the card, which is where `Ei` loses.
- Grep the server logs for: `none of its gathers followed`, `no gate_up seen`, `selected units off the CPU`, `reads rows the CPU path cannot`, GPU faults.

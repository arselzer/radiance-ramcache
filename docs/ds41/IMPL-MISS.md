# miss: implementation report

Both miss-path follow-ups and the counters are built, tested and committed. Nothing has been measured on the GPUs: I didn't use them.

The main branch moved to 54ff6dde while I worked. Its `ab_window.sh` now skips any arm with a part it doesn't know, so the A/B must use this branch's `ab_window.sh`. A launcher that does that is described below.

**Branch and commits:** `claude/ds41-miss`, worktree `/home/as/ds41-claude-build/wt-miss`, on top of dc83cecc. Author is Alexander Selzer <aselzer3@gmail.com>, each message ends with the Co-Authored-By line, nothing pushed.
- `cd021458` NativeReadWorkers: post/join. `run()` itself is unchanged.
- `d9b143d0` PlaneMirror: `read()` reports how long its copy out of the bounce buffer takes, and a new `split()` cuts a tail into pieces.
- `176ae353` The engine side: the counters, the mirror split and the early miss reads.

**Switches** (all exact: the same bytes are read into the same places):

| What | Switch | Arm part |
|---|---|---|
| (a) A decode pass that misses one unit reads the mirror's decode tail in N pieces on N readers | `RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=N` (2–4, designed for 3) | `Ms3` (needs `R` and `Rd13`) |
| (b) A one-token pass's miss reads start on the kept readers right after `hx_choose`, before `hx_submit` wakes the pool; `hx_submit_misses_now` waits for them | `RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1` | `Mp` |

- **(a) details:** each piece starts on the container's 4 KiB grid, so a piece that fails is re-read from the container exactly as a whole tail would be.
- **(b) details:**
  - `post()` holds the readers' call mutex until the matching `join()`, just as `run()` holds it for a whole call.
  - A guard in `start_ep_routed` and in `hx_drain` waits for any posted read on error paths.
  - It is skipped where misses are read on the async worker, i.e. prefill passes with A2.
- **One thing that is not behind a switch:** the counters are always on. They cost a few clock reads per miss read and add report lines; reads and results are unchanged. Gating them is a small change if you want strict "identical with the switch off".

**A/B arms.** Launch with `/home/as/ds41-claude-build/aql-miss/launch_window_miss.sh LOG "ARMS" MIRROR=/srv/bulk/source-copies/ds41-native-mxfp4/d8b2e3c20f5f4b6d5044d37f20b8d53f260f83e466f72259572be76d7e756bf3-main`.
- It follows `work/launch_window.sh`, but snapshots this branch's `ab_window.sh` and my `radiance`.
- Output goes to `wt-miss/tools/ds41-host-experts/build/ab-<ts>`.
- It uses `home2` unchanged.

With B = `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2`, run interleaved, three repeats each:
`B B-Ms3 B-Mp B-Ms3-Mp B-r2 B-Ms3-r2 B-Mp-r2 B-Ms3-Mp-r2 B-r3 B-Ms3-r3 B-Mp-r3 B-Ms3-Mp-r3`.

- The B arms are the control on my binary. They should match the 13:26 window's B (15.45–15.49 tok/s).
- Expected gains are small: (a) +0.6–1.9%, (b) +0.3–0.6%. Arm-to-arm noise is ±0.5–1%, hence three repeats.
- Every reply must match the base engine (`work/ab_sum.py`).
- Optional end-to-end exactness check: `verify-t40-f-M16-H-p256-m98304-G-R-W-A2-Rd13-Er-Es-Pa-Ms3-Mp` should show "0 differ". I haven't checked that this combination starts.

**What to look for in each server log** (per rank):
- **Miss-read summary line.** `host experts' one-token miss reads, one unit: …` (and the same for two or more units) gives:
  - the calls, how many were posted early, how many had the tail in pieces;
  - how often the mirror landed last, and by how much;
  - the mirror and container legs' times;
  - the mirror's copy time;
  - when the caller got back.
- **Gate for (a):** read this line in the B arms. Drop (a) if the mirror lands last in under 25% of one-unit calls and its copy takes under 0.15 ms. With `Ms3` the copy should fall to about a third.
- **Readers line.** `…, readers:` gives when the first read, the first container read and the last read began, and how long after the last read landed the caller got back.
- **Gate for (b):** in B, the first container read's start plus the caller's return should exceed 40 µs.
- **What changes with `Mp`:**
  - `misses on the CPU: … ms reading their planes` (time the issuing thread is blocked) should drop.
  - The "choose" figure in `EP decision` now includes the post.
  - `kept plane readers: … (P posted …)` should show P > 0.
- **Mirror copy time.** The `plane mirror (all ranks)` line now ends with the copy time.

**Built artifacts:**
- `/home/as/ds41-claude-build/aql-miss/bin/radiance` (md5 d94e1c070adcb6fa9fa3c2dc4ae0f699), built from HEAD.
- No plugin changes: use `home2` as is.

**Tests:** all pass.
- In `aql-miss`: `native_read_workers_test` (new, 7 cases), `plane_mirror_test` (2 new cases), `host_expert_pool_test`, `place_test`, `mapped_slab_pregrow_test`.
- I also made a CPU-only build, `/home/as/ds41-claude-build/aql-miss-host`, because `native_mapped_copy_mover_test` refuses to run in a GPU build. All 32 cases pass, including a new one checking that the early and the normal reads both land the container's bytes.

**Unfinished or uncertain:**
- **Split not tested end to end in the engine.** The piece arithmetic and reading a tail in pieces are tested at the mirror level only. A mover-level test would need a mirror whose file names match the test fixture.
- **(b) may gain nothing.** Posting early starts the reads about 30–40 µs sooner. But the issuing thread now always waits for a reader to wake it, which is roughly a 10–30 µs wake and could cancel most of the gain. The readers line will show this.
- **Merge conflicts.** Merging onto 54ff6dde conflicts only in `README.md` and `ab_window.sh`. Keep both sides, and put the `Ms[0-9]` and `Mp` case lines before the `*)` line. `stager.cpp` merges cleanly.

# miss: review

**Review of claude/ds41-miss (worktree /home/as/ds41-claude-build/wt-miss, cd021458, d9b143d0, 176ae353 on top of dc83cecc)**

I found no code defect that needs fixing before the GPU A/B. I found no way for an output bit to change, and no deadlock or lifetime bug in the arms planned. With both switches off the engine behaves as before; only the always-on counters differ (finding 4). Two launch-day steps are required (M1, M2). One merge problem is serious, but it only matters at merge time, not for this A/B (finding 1).

I did not re-run the tests. A window (ab-20261029T135754Z) was running for the whole review, and the load rules forbid work using more than 2 threads while one runs. Everything below comes from reading the code and checking files.

### Must do before the GPU A/B (no code change)

- **M1. Launch only after the current window ends.** Check that `pgrep -f 'ab_[w]indow.sh'` returns nothing first.
  - `aql-miss/launch_window_miss.sh` has no guard against a running window. `work/launch_window.sh` has the same weakness.
  - If started now: its `ab_window.sh` would wait up to 10 min for QFN idleness, give up (exit 3 or 4, depending on QFN's state), and then its wrapper deletes `WINDOW_PENDING` while the other window is still running.
- **M2. Replace the suggested exactness arm and run it first.**
  - The implementer's `verify-…-p256-m98304-…` has no `B1`. Without it, the card's copy path stages units through a 256 MiB pinned pool; `p256` is only safe in B because `B1` removes that path. It may not start, and the implementer never checked.
  - Use the form of earlier verify arms that ran: `verify-f-M1-H-R-Rd13-Ms3-Mp`. It runs both switches on decode misses and compares every CPU row with the card.
  - It must show `0 differ`. Read that from `$name.server.log`, not `window.log` (see finding 6).
  - Optional: `verify-f-M1-H-R-Rd13-Ms3-Z40` checks that a failed piece falls back to the container. Z must be above 32, because `mirror_init` makes 8 sampling reads per rank through `PlaneMirror::read` (mover.cpp:1219). A lower Z disables the mirror at startup instead.

**Arms to run:** prepend `verify-f-M1-H-R-Rd13-Ms3-Mp` to the implementer's interleaved list: `B B-Ms3 B-Mp B-Ms3-Mp`, repeated as `-r2` and `-r3`, where B = `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2`.

**Checks per arm before trusting its tok/s:**
- Every Ms3 arm shows `N with the mirror's tail in pieces` with N > 0 on every rank.
- Every Mp arm shows `posted at the choice` > 0, and the rank-0 startup line for MISS_POST.
- Every arm shows texts 8/8 in `ab_sum.py`.

### Findings (most severe first)

1. **Medium (merge only, not this A/B): the merge note is out of date.** Main moved to ea27e5c9 at 14:14:52Z (read-ahead in pieces merged).
   - A read-only `git merge-tree` trial merge conflicts in `core/place/mover.cpp` (`read_unit_planes` now built on `lay_out_weight`), `core/place/stager.cpp`, `tests/native_mapped_copy_mover_test.cpp`, `README.md` and `ab_window.sh`. "stager.cpp merges cleanly" is no longer true.
   - Three problems the merge must handle in main's new `hx_submit_misses_now`:
     - **(a) Use after release.** `hx_post_misses` (stager.cpp:1536-1545) must skip `hx_miss_ra_` ("urgent") units. `hx_ra_settle` replaces `hx_miss_[i]` and returns the old planes to the free list, so a posted reader could write into planes already handed to another unit.
     - **(b) Self-deadlock.** `finish_unit_planes` must come before `hx_ra_settle`. That function calls `read_unit_planes`, which calls `plane_workers_.run()` on the thread that still holds the call mutex.
     - **(c) Demand signal.** `hx_demand_reads` no longer exists on main. Post and join must call `PieceReader` `demand_enter`/`demand_leave` instead.
   - Main's `start_ep_routed` also has its own `t_posted` and `d_post_ns`, so the names clash.
   - Re-verify after the merge: one verify arm plus a B vs B-Ms3-Mp pair.
2. **Low (speed, not reachable in B): `post()` waits for the readers instead of giving up** (native_read_workers.h:88).
   - If another thread is using the readers (whole-unit read-ahead under Po/Pt at stager.cpp:1682, or hx_mw_), `hx_post_misses` (stager.cpp:784) stops the issuing thread before `hx_submit`. The pool's cached-unit jobs are then delayed by up to a whole read (~2 ms); before this change that wait came after `hx_submit`.
   - Fix: `try_lock` and return false, so the caller reads the normal way.
3. **Low: the split also applies to the read-ahead.** `hx_pf_loop` calls `read_unit_planes(&raw, 1)` with the default rows=1 (stager.cpp:1682), so mover.cpp:984 splits its tail too: extra background SATA reads, not counted. Not in B. Restrict the split to demand reads, or document it.
4. **Low (deviation from the "identical with the switch off" rule): the counters are always on.**
   - Locations: mover.cpp:953/958, plane_mirror.cpp:190-202, stager.cpp:1496, engine.cpp:142-172, mover.cpp:2818-2822.
   - Outputs are identical; log lines differ; cost is under 1 µs per call.
   - Fine for this A/B: every arm carries them and B needs them for both gates. Decide at merge whether to gate them.
5. **Low (reading the numbers): some figures change meaning under the switches and must not be compared across arms.**
   - The `plane mirror (all ranks) … ms a read, … copying` line (mover.cpp:2839) is per piece, so Ms3 cuts it to about a third whether or not the reads overlap. For the copy gate use the "one-token miss reads, one unit" line ("its copy" is the longest piece per call) and its "mirror leg".
   - With Mp, the mover's `ms a batch weight` (mover.cpp:2812-2816, fed at :1061) and the stager's `ms reading their planes` count only the wait at the join, so they drop by construction. Judge Mp by tok/s and the per-call figures instead.
6. **Low: the verify result drops out of `window.log`.** `ab_window.sh:162` keeps only `tail -12` of the `host experts` lines. The 2-4 new lines per rank plus "kept plane readers" push most ranks' `verify: … differ` lines out. Read each server log.
7. **Low: Ms has no startup line.** `RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT` is only read at the first decode miss (mover.cpp:939-945). An Ms arm that silently does nothing (no `Rd`, or the mirror turned off) only shows at shutdown as a pieces count of 0.
8. **Low (robustness, not reachable today):**
   - `post()` locks the call mutex by hand, then takes a lock on `state_mutex_` (:88, :101). An exception there would leave the call mutex held forever.
   - `join()` relies on `posted_`, a plain bool, and does not check which thread calls it (:118, :227). A join from any thread other than the poster is undefined behaviour.
   - Any future call to `read_unit_planes` on the issuing thread between post and finish would deadlock against itself.
   - Today `hx_posted_` is set and cleared within one `start_ep_routed` call on one thread (stager.cpp:776-787, 1476-1481, 1549-1553), so none of this can happen. Optional hardening: release-on-success locking, and record the poster's thread to warn on mismatch.
9. **Low (expected speed cost): with Mp the caller always waits to be woken.** `post` gives every read (up to 7) to the readers (mover.cpp:1121), so each posted call needs a wake-up at the join (native_read_workers.h:126). In the normal path the caller often finishes the mirror read itself and needs none. The "caller back … after the last read landed" figure will show the cost. If it cancels the gain, a short spin in `join()` before sleeping is the next step.
10. **Info (test gaps):**
    - The split and a failed piece's fallback are tested only at the mirror level (plane_mirror_test.cpp:119-173). The mover test (native_mapped_copy_mover_test.cpp:969-1005) has no mirror. I checked the mover's piece arithmetic by hand (mover.cpp:1033-1040, fallback :1057) and it is right:
      - pieces start at a0+cut+k×4 KiB (tail rounded to 4 KiB at :1210, offsets checked at :1196-1197);
      - every container read takes the aligned pread path, so the shared, non-thread-safe bounce buffer in directio.cpp:141-157 is never used by concurrent pieces;
      - each fallback's buffer size equals its aligned length.
    - The verify arm in M2 is the end-to-end substitute.
    - The error-path join in `start_ep_routed` has no test.
    - Small flake risk: native_read_workers_test.cpp:66-75 assumes the joining caller wins 1 of 12 items within the reader's ~36 ms.
11. **Info: B on this binary will not quite match the 13:26 numbers.** This branch lacks e8560766 (vectorized swiglu, on by default in main). The 13:26 B (15.45–15.49) used it; this binary corresponds to that window's scalar Xs arm (15.48 / 14.22 / 14.99). The comparison within one window is unaffected; don't read a small offset as a binary problem.
12. **Info: three repeats probably can't resolve Mp.** Its expected +0.3–0.6% is below what three repeats can show: noise is ±0.5–1%, and the 13:26 window had ~8% outside dips. The 2×2 design gives 6 runs against 6 for each switch. Judge Mp mainly by the counters, keep it only if tok/s is no worse, and add repeats if it's close.

### Checked and found correct

- **Exactness:**
  - The pieces cover the same tail [cut, end) exactly, with boundaries on whole pages (plane_mirror.cpp:155-168), and they read the same bytes.
  - Mp posts the same set the normal path reads (stager.cpp:1538-1540 matches :1470-1472). `hx_miss_` and `hx_miss_ready_` are only written in `hx_choose`, before the post. The miss jobs are built only after the finish (:1497).
- **Switch-off path:** the same single mirror read at the front, the same read order, the same run(8, ≤8) and thread fallback, and the same early return, mirror fallback and counters. `hx_join_posted` does nothing.
- **Concurrency:**
  - post/join follow run()'s protocol.
  - Between post and finish the issuing thread touches nothing that uses `plane_workers_` (`hx_submit`, `hx_pf_target`, `stage_file_units` via `relay_workers_`, `alias_file_units`, mover stats).
  - Readers touch only `plane_call_`, the planes, `file_` (pread) and the mirror; `mirror_ref_` is set at startup only.
  - Every exit after a post joins it, including error returns and exceptions; `hx_drain` joins first.
  - Each rank has its own Mover, and there is no lock-order inversion.
- **Arm parts:** `Ms[0-9]` sets `RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=N` (read at mover.cpp:941) and `Mp` sets `RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1` (stager.cpp:1111). Both reach the server through `env "$@"`, and no earlier case pattern catches them. B has R and Rd13, so Ms3 takes effect (2,441,216 B tail → 198/199/199 pages); A2 keeps one-token passes on the issuing thread, so Mp takes effect.
- **Commits and artifact:**
  - Author, trailers and branch are correct, nothing is pushed, and the worktree is clean.
  - `/home/as/ds41-claude-build/aql-miss/bin/radiance` (md5 d94e1c070adcb6fa9fa3c2dc4ae0f699) was built at 13:45:41, after the last source edit at 13:38:46, with build_hx.sh's cmake options.
- **Launcher inputs:**
  - The snapshot's `ab_requests.py` is identical to main's.
  - home2's plugins are unchanged since the 13:26 window.
  - The silu table and the mirror directory (48 shards) exist.

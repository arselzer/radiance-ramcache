# ram: implementation report

No GPU A/B has been run (GPU use wasn't allowed), so the speed and memory gains below are estimates. All tests pass, and nothing in the computation changes, so replies should stay byte-identical. Each change has its own off-by-default switch.

**Branch and commits:** `claude/ds41-ram` in `/home/as/ds41-claude-build/wt-ram`, on top of `dc83cecc`, not pushed.
- `5fa2228f`: leaves the file tier's idle staging out under the file-buffer cap. Switch `RADIANCE_EXPERIMENTAL_DS41_SKIP_FILE_STAGING=1`, arm part `Ns`.
- `dae55236`: trims and bounds the misses' plane-buffer free list. Switch `RADIANCE_EXPERIMENTAL_DS41_PLANES_KEEP=N`, arm part `PkN`.

**What each does**
- **`Ns`:**
  - The planner reserves nothing for the 4 stage slots and 4 × 32 MiB read windows a rank (`Plan::file_stage_skipped`), and the mover allocates neither.
  - It refuses to start without `B1`, or when a non-routed unit could reach the file tier. The mover and `FileStager::hx_init` refuse such a plan again.
  - If any unit would still need the copy path, `stage_file_units` refuses rather than read through a missing window.
  - It frees no RAM on its own: the pool keeps the room for pinned experts. Drop `--host-pool-mib` by 199 (`p57` instead of `p256`). Any value from 54 to 71 keeps today's placement (3 pinned experts a rank).
- **`PkN`:**
  - After the warm-up it keeps N plane buffers a rank and frees the rest. From then on, a buffer handed back while N are already waiting is freed.
  - With the switch on, the buffers are their own memory mappings, so freeing one returns the memory to the system.
  - N=8 covers decode: the current trace has at most 4 misses per one-token layer pass a rank, and at most 10 in a 16-token pass.
- **Shutdown report (new lines, printed in every arm):**
  - Per rank, the free list's counts: made, kept, peak during and after the warm-up, freed.
  - Once per process: RssAnon, VmRSS, pinned and card-mapped host memory, and MemAvailable.

**Expected memory effect at the same cache size**
- **Pinned pool:** 572 → 373 MiB a rank, −796 MiB in all.
- **Plane buffers:** about 16 → 8 a rank at 18 MiB each, −576 MiB in all. The 16 is the research's estimate from the heap mappings (about 65 buffers, 1.15 GiB); the best arm's new shutdown line gives the real count.
- **Together:** about +1.34 GiB MemAvailable while serving (≈13.8 → 15.2 GiB in the `.mem` trace).
- **The `PkN` mechanism, measured off-engine:** in a standalone program, freeing 8 of 16 buffer pairs of this size dropped RssAnon by 144 MiB.

**Larger cache this allows:** m98304 → **m99584** (+1280 MiB, about 18 more expert pairs a rank, +1.3%). Expected effect: misses about −1.3% and decode +0.4 to +0.7%, which is within noise, hence the repeats. The exact allowance is 796 + 72 × (N − 8) MiB, where N is the per-rank count the best arm prints.

**Arms, interleaved: the best arm and the RAM-to-cache arm three times each, plus one memory-only control**
```
all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2
all-t40-f-M16-H-p57-m99584-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns-Pk8
all-t40-f-M16-H-p57-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns-Pk8
all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-r2
all-t40-f-M16-H-p57-m99584-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns-Pk8-r2
all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-r3
all-t40-f-M16-H-p57-m99584-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns-Pk8-r3
```
`launch_window.sh` copies `ab_window.sh` from the main repo and the binary from `aql`, so the `Ns`/`Pk` parts would only be logged as unknown. Use a variant:
```
sed -e 's#cp -p "$T/ab_window.sh"#cp -p /home/as/ds41-claude-build/wt-ram/tools/ds41-host-experts/ab_window.sh#' \
    -e 's#/home/as/ds41-claude-build/aql/bin/radiance#/home/as/ds41-claude-build/aql-ram/bin/radiance#' \
    /home/as/ds41-claude-build/work/launch_window.sh > /tmp/launch_window_ram.sh
bash /tmp/launch_window_ram.sh ram "$ARMS" MIRROR=/srv/bulk/source-copies/ds41-native-mxfp4/d8b2e3c20f5f4b6d5044d37f20b8d53f260f83e466f72259572be76d7e756bf3-main
```

**What to look for in the server log**
- **Rank 0 at startup:** `placement: the file tier's staging left out ... 199.72 MiB a rank ... lowered by 199`.
- **Pools:** `host_pool 373.00 MiB budget`; the best arm shows 572.00.
- **Same placement:** `planned units VRAM=598 pinned=3 SSD=3239` on rank 0 and `615/3/3222` on ranks 1–3, as in the best arm.
- **Per rank:** `keeps 8 of its misses' plane buffers (144.00 MiB) after the warm-up, which held W at once`.
- **Startup report:** "pinned by this process" about 0.78 GiB lower than the best arm's 98.62 GiB in the control arm.
- **At shutdown:** `memory at shutdown: RssAnon ...` and `host experts' plane buffers: ... at most W out at once during the warm-up and P since; ... F given back past the bound of 8`.
  - The best arm shows bound −1; its free-list count sets the allowance.
  - P ≤ 8 confirms decode never needed more than 8.
  - F counts fresh buffers allocated in prefill. If F is large and TTFT is worse, try `Pk12`.
- **Misses:** `routed experts were not cached` and the mapped cache's miss count should be about 1–2% lower in the m99584 arm.
- **Memory trace:** MemAvailable in `.mem` should be about +1.3 GiB in the control arm, and about equal to the best arm in the m99584 arm.
- **Replies:** 8/8 identical texts in every arm.

**Built artifacts:** `/home/as/ds41-claude-build/aql-ram/bin/radiance` (md5 `5decbe44…`). `libr4d` and `deepseek41` are unchanged, so the existing `home` and `home2` work.

**Tests**
- **HIP build:** `place_test` 116/116 (4 new), `host_expert_pool_test` 6/6, `mapped_slab_pregrow_test` 3/3, `plane_mirror_test` 3/3.
- **Host-only build** (`/home/as/ds41-claude-build/aql-ram-host`): `native_mapped_copy_mover_test` 31/31, `native_file_cache_mover_test` 4/4 and `place_test` 116/116. These two mover tests refuse to run on a HIP build.
- **Commit 1 on its own:** also builds and passes.

**Uncertain or unfinished**
- **Baseline buffer count:** the ~16 a rank comes from an older configuration.
- **Prefill cost:** each buffer freed past the bound costs an unmap plus 18 MiB of page faults later, and the unmap interrupts the pool's cores briefly.
- **Why the buffers are mapped:** in a fresh process glibc already returns this memory on free, but the live server's buffers sat in the heap, where freeing doesn't return it. Mapping them explicitly makes the trim work either way.
- **Optional, untested:** `p0` would also move the 3 pinned experts a rank to the CPU path. That changes the placement.
- **Merging:** expect textual conflicts in the `ab_window.sh` parts list and the README table, where other branches add parts.
- **Planner purity:** the planner now reads an environment switch. `..._FILE_BUFFER_UNITS` already did, despite the planner's "pure" note.

# ram: review

I found no way for an output bit to change, and no race, deadlock or lifetime bug in `claude/ds41-ram` (`5fa2228f`, `dae55236` on `dc83cecc`). Nothing has to be fixed for exactness before the GPU A/B. Findings 1–3 should be dealt with first: findings 2 and 3 are cheap and make the window interpretable, and finding 1 must be fixed before `Pk` is adopted. I modified no files in the worktree; my only additions are two measurement scripts in the job scratch directory.

**What I checked**
- **Exactness:** `Ns` only moves the embedding and the 3 pinned experts to other offsets in the host pool, still aligned. `Pk` only changes where the misses' landing buffers live; every byte the CPU reads from them is read again from the container on each use.
- **Switch off:** behaviour is the same as `dc83cecc` apart from new log lines. Every new branch in the planner, `Mover::init`, `stage_file_units` and `hx_init` is gated on `file_stage_skipped`, which is 0. With the bound at −1 the free list is the old LIFO list, and buffers are still `aligned_alloc` and `std::free`.
- **Arm parts:**
  - `Ns` (ab_window.sh:285) sets `SKIP_FILE_STAGING=1`, which is exactly what the planner checks for.
  - `PkN` (:286) sets `PLANES_KEEP=N`, and the code treats any value above 0 as on.
  - No earlier case pattern catches either part, and main's `N[0-9]*` and `Pp`/`Pq`/`Pb`/`Pf` parts don't clash with them after a merge.
- **`Ns` in the best arm is safe:**
  - Under the cap, `start` and `start_routed` refuse, and policy=all leaves nothing for the card to read (`kHxUnits` is 96, so every local expert fits). So `stage_file_units` only ever gets lists with nothing to read, and its new early exit matches the old "nothing staged" exit.
  - The stage slots only serve `Plan::prefetch` for non-routed units, which the switch refuses. `promote_to`, the direct fill and the plane mirror never use the slots or the windows, and the mover's teardown works with zero windows.
- **Placement:** the best arm's log (ab-20261029T135754Z) shows a 572 MiB pool (256 + 316 embedding) and 3 pinned experts (53.79 MiB). `p57` gives 373 MiB, leaving 57.38 MiB for experts, so still 3 pinned; any value from 54 to 71 gives 3. Under `--deterministic` the card's memory split doesn't depend on the host pool, and each rank runs the planner once.
- **Tests (host-only build, no GPU):**
  - The 7 related `place_test` cases pass.
  - The full `place_test` passes 116/116 both with and without `PLANES_KEEP=8`.
  - `native_mapped_copy_mover_test` passes 31/31 with `PLANES_KEEP=8`, which runs a mapped buffer through a real miss read and its unmap.
  - I didn't re-run the HIP build's tests because another GPU window was running.
- **Binary:** `aql-ram/bin/radiance` (md5 `5decbe4443dce9b968ba2d9ee9bbdc1b`) is newer than every changed file and contains the new log strings. The plugins in `home2` are the FP8-era ones (13:57); the FP8 report says they load in this tree's radiance, and the 13:57 best-arm baseline ran on them. The implementer's `sed` launcher rewrite is correct: only line 15 changes.

**Findings, most severe first**

1. **Medium: `PkN` makes fresh huge-page buffers on the miss-read path.** Locations: stager.h:258-266 (`give` frees past the bound), mover.cpp:63-75 (each new buffer is a new huge-page mapping), stager.cpp:1520.
   - After the warm-up, every burst above N makes new buffers. They are faulted in during the miss read and unmapped again when they come back. The unbounded list paid this once, at warm-up.
   - I measured a first touch of an 18 MiB huge-page mapping against touching it a second time:

     | Conditions | First touch | Second touch |
     |---|---|---|
     | ds41 server serving, 4.7 GB free | 11–331 ms | 1.2–2.3 ms |
     | F8-Pp4 arm serving, 9.6 GB free | 2.2–5.7 ms | 0.9–3.0 ms |
     | Idle box | 1.3 ms | 0.8 ms |
     | Without huge pages | about 8.5 ms | about 1 ms |

   - In the first case the 8 runs hit 216 compaction stalls, and all 72 huge-page faults fell back to 4 KiB pages. The box's huge-page defrag setting is madvise, so these faults compact memory directly.
   - Decode misses are sparse (1,278 of 19,240 one-token layer passes read the drive on rank 0), so the cost lands mostly on prefill bursts and TTFT. On a fragmented box one burst can cost hundreds of ms.
   - Fix before adopting `Pk`: trim once after the warm-up and don't bound afterwards (or trim only between requests), set N at or above the serving peak, or make the overflow buffers without huge pages.

2. **Medium (A/B design): the proposed cache arm bundles `Ns` with `Pk8`.** A `Pk` regression would hide `Ns`'s gain, and the m99584 allowance assumes about 16 buffers a rank, which nobody has measured.
   - Miss totals are deterministic: 10440, 10439, 10442, 10439 and 10440 over 5 best-arm runs. Meanwhile tok/s varies from 14.98 to 15.49 and the TTFT sum from 16.6 to 17.9 s. Judge the cache arms by miss totals, not tok/s.
   - Arms to run instead, interleaved: `B C1 C2 M B-r2 C2-r2 M-r2`

     | Name | Arm | What it shows |
     |---|---|---|
     | B | `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2` | baseline, and N from its shutdown line |
     | C1 | `all-t40-f-M16-H-p57-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns` | `Ns` alone, same cache |
     | C2 | `all-t40-f-M16-H-p57-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns-Pk8` | what `Pk8` costs |
     | M | `all-t40-f-M16-H-p57-m99072-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Ns` | `Ns`'s 796 MiB given to the cache |

   - Run the implementer's `p57-m99584-…-Ns-Pk8` arm only after C2 shows no TTFT loss. Its allowance is then 796 + 72 × (N − 8) MiB, with N from B.

3. **Medium (counters): the shutdown counters can't tell where `Pk`'s churn happened** (engine.cpp:172, stager.h:283-300).
   - The "given back past the bound" count includes decode as well as prefill, so it isn't "fresh buffers allocated in prefill" as the report says.
   - The number of buffers made after the warm-up isn't printed.
   - The peak isn't split by pass type, so "P ≤ 8 confirms decode" only holds if prefill also stayed at or below 8, which it likely won't.
   - Add a per-rank count of buffers made after the warm-up and a peak, each split into one-token and multi-token passes (`pass_tok_` is available in `hx_choose`).

4. **Low (only at merge time): main has moved on (`ea27e5c9` and later).**
   - Main's `Mover::lay_out_weight` (main mover.cpp:1021) still calls `std::free(p->buf[k])`. The merge must change it to `free_plane_buffer`, or it frees a mapped pointer when `Pk` is on.
   - Main's piecewise read-ahead holds more plane buffers during decode (main stager.cpp:2008 and :2126). `Pp4` is now in the recommended configuration, so N=8 must be re-derived with it on.
   - A read-only trial merge conflicts in engine.cpp, mover.cpp/.h, stager.cpp/.h, README and ab_window.sh.
5. **Low:** engine_bringup.cpp:1815-1819 still subtracts the full staging reserve from `--host-pool-mib` when it computes the expert floor (0 with p57). This has no effect here because the floor is unused under `--deterministic`.
6. **Low:** the planner's "lowered by 199" hint (planner.cpp:1329) rounds down, leaving 0.72 MiB more for experts. That gives the same placement on DS41 but isn't guaranteed in general.
7. **Low:** ab_window.sh:162 keeps only the last 12 "host experts" lines. The 4 new per-rank lines push other ranks' lines out of window.log, so read the server logs.
8. **Low (test gaps):**
   - The mapped-buffer mode is never run by ctest, because the mode is fixed per process and `place_test` runs without the variable set. A ctest entry that runs `place_test` with `PLANES_KEEP=8` would cover it; I ran it and it passes.
   - The planner's refusal when a non-routed unit still reaches the file tier under `Ns` (planner.cpp:1355-1365) has no test. `Mover::init` checks for it again.
9. **Info:** the new log lines (engine.cpp:101-112 and 169-178) also print in switch-off arms. That's acceptable, since the task asks for counters.

**What to look for in the server logs**
- **C1:**
  - Rank 0 prints "placement: the file tier's staging left out … 199.72 MiB … lowered by 199".
  - The pool line reads `host_pool 373.00 MiB budget`, and the placement lines read `planned units VRAM=598 pinned=3 SSD=3239` (rank 0) and `615/3/3222` (ranks 1–3).
  - "host pool 369.41 MiB" in the weights-by-tier line, and "pinned by this process" about 97.84 GiB.
  - The summed "routed experts were not cached" count is within ±5 of B's.
- **B:** the "host experts' plane buffers" line gives N (the free-list count, bound −1), plus the warm-up peak and the peak since.
- **C2:** the TTFT sum and decode tok/s against B, the "given back past the bound of 8" count, and the peak since the warm-up.
- **M:** misses about 0.8% below B (around 10,360), and MemAvailable in `.mem` at least as high as B's.
- **Every arm:** 8/8 identical texts in `ab_sum.py`.

Files are in `/home/as/.claude/jobs/635603c1/tmp/review-ram`:
- thp_fault.py
- thp_fault2.py

# l3: implementation report

**l3 (CCD-L3 staging, plan #8): the gate failed, so I stopped before any engine change. The best result at the design point saved 12.1% of the burst time, against the 20% the plan requires.**

Nothing was committed or built. Branch `claude/ds41-l3` is still at dc83cecc and the worktree is clean. There is no switch, no `ab_window.sh` arm part, no arms to A/B and no shutdown-report counters. I never created `/home/as/ds41-claude-build/aql-l3`, so there are no artifacts to copy.

**How the gate was run.** It is `research/ccdl3/burst4` with 40 workers, 1400 us gap, 200 us spin and a 70-expert ring. The plan mentions "the three commands in that directory", but no such list exists, so I picked three: the top-2 design point (E3 S1), a coverage bound (E3 S2) and a heavy layer (E6 S2). The clean runs were 15:33–15:37 UTC: no window pending, the production server idle, each run under the build lock, and no compiler seen in any 0.5 s sample. Burst medians (time saved against base):

| Command | base | conf | stage | stage-ov (interleaved) |
|---|---|---|---|---|
| `burst4 40 3 1 1400 2500 200 70` (design point, f=1/3) | 651.6 us | 649.7 | 572.1 (12.2%) | **572.5 (12.1%)** |
| `burst4 40 3 2 1400 2500 200 70` (f=2/3) | 650.3 | 646.0 | 509.5 (21.7%) | **561.6 (13.6%)** |
| `burst4 40 6 2 1400 1500 200 70` (heavy layer) | 1221.8 | 1220.9 | 1100.8 (9.9%) | **1163.3 (4.8%)** |

- **Earlier run at 13:46:** builds started around it and its tails were long (p90 about 3.5 ms), but the median result agrees: 760.0 to 685.1 us, 9.9%.
- **The real kernel (packed form of the K arm), `burst5 40 3 1 1400 1600 200 70 1 2`:**
  - Base 622.0 us; stage 549.2 (11.7%); stage-ov 546.1 (12.2%).
  - With floating workers and 2 dedicated stager threads per CCD (the version that avoids pinning): 555.9 (10.6%) and 551.8 (11.3%).
  - In that version the stagers' own read time had a p90 of 1.0–1.2 ms, against 219 us when the woken workers stage.
- **One CCD (`l3gap2 1 6 27 47 100 1`):** cold 463.9 us against staged 347–352 us, about 24% less. The mechanism works on one CCD, as the plan said.

**Why it fails.** With 40 workers on 4 CCDs the base burst already streams at 87 GB/s. With a third of the tiles in L3, the workers left on the other tiles don't keep DRAM saturated, so staging a third of the bytes saves 12% rather than about 30%. The staged-first arm reaches 21.7% only when two of three experts are staged, a coverage the predictor can't reach (top-2 covers about 0.31 of hit bytes).

**What it would have given in the engine:**
- 12% of each burst in the miss-free layers (70% of decode layers, 2.78 hits a burst on the 11:55 trace) is at most about 2 ms of a 64.5 ms token, roughly 3% gross.
- Against that come the predictor's measured cost (about +11 us a layer on rank 0, roughly 0.7%), the reader threads' CPU and DRAM use in the gap, and the fact that only part of each burst is exposed.
- That leaves probably about 1% net: inside one arm's ±1% noise, for a large change.

**Uncertain or worth knowing:**
- The bench leaves the gap between bursts idle. The engine's gap carries miss DMA and inserts, which could only make the gain smaller.
- From 13:48 to 15:28 back-to-back windows and other agents' builds left no quiet gap. I asked main for one; once the gate ran I told it the gap was no longer needed.
- The `pgrep -f 'ab_[w]indow.sh'` check in the load rules matches any shell whose command text mentions the script. Main's PID 759214 matched its own loop from about 12:58 and made the check true for every agent. I used argv-based detection instead and told main; that process is gone now.

Files are in `/home/as/.claude/jobs/635603c1/tmp/research/ccdl3-gate/`:
- `gate6.out`: the clean results.
- `gate.run1.out`: the disturbed first run.
- `burst5.cpp` (and its binary `burst5`): burst4 plus the packed form and the floating-worker arms.
- `quietrun6.sh`, `winbusy.sh`, `cpubusy.sh`: a quiet-box runner that never holds the build lock through a window. Other CPU benches could reuse it.
- `trace_bursts.py`: hits per decode layer from an access trace.
- `l3stage-draft-unbuilt.patch`: an unbuilt, untested draft of the engine side (stager reader threads, per-CCD tile claims in the pool), kept for reference only.

# l3: review

**l3 review: there are no commits to review. Nothing needs fixing before a GPU A/B, and l3 should not get an A/B at all. The kill at the gate holds up.**

**Branch state (checked)**
- `claude/ds41-l3` HEAD is dc83cecc. Its reflog has one entry ("branch: Created from dc83cecc"). `git log dc83cecc..HEAD` and `git diff dc83cecc` are both empty.
- The worktree `/home/as/ds41-claude-build/wt-l3` is clean, with no ignored files and no stash. `/home/as/ds41-claude-build/aql-l3` does not exist.
- No L3-staging code (`L3_STAGE`, `L3Stage`, `l3_stage`, `DS41_L3`) exists in the main tree or in any `wt-*`. The draft patch was never applied. The main tree has no uncommitted changes (checked read-only).
- No gate processes are left running, and PID 759214 is gone.
- So no output bit can change, there are no races or lifetime bugs, and the off path is unchanged. There is no switch, arm part, test, counter or artifact to check.

**Gate numbers and conditions (checked against `/home/as/.claude/jobs/635603c1/tmp/research/ccdl3-gate/gate6.out`)**
- Every figure in the report matches: 651.6→572.5 us (12.1%), 650.3→561.6 (13.6%) and 509.5 (21.7%), 1221.8→1163.3 (4.8%), burst5 11.7/12.2/10.6/11.3%, l3gap2 463.9→347–352. The disturbed earlier run gave 760.0→685.1 (9.9%).
- If "20% faster" means a speedup ratio instead of time saved, the design point is 13.8% and f=2/3 interleaved is 15.8%. It fails under either reading.
- The runs were clean:
  - Each run took the build lock and saw no compiler in any sample; p90 was at most 1.05× the median.
  - The last run ended at 15:37:08. `WINDOW_PENDING` was created at 15:37:50 (from `stat`), when window ab-20261029T153750Z started.
  - `launch_window.sh:21-33` keeps that flag and the build lock for the whole window.
- The bench matches the engine:
  - The CPU-to-CCD mapping (`burst4.cpp:81-86`) agrees with sysfs (CCD c is CPUs 6c..6c+5 and 24+6c..24+6c+5).
  - 40 workers matches `t40`, and the 200 us spin is the pool default (`core/place/host_expert_pool.cpp:64`).
  - The packed raw caps 512/320 match `core/place/mover.cpp:56`, and the compile flags match `core/CMakeLists.txt:32-33`.
- Every known bias in the bench favours staging, so 12.1% overstates what the engine would get:
  - The gap is idle.
  - Staging finishes before the gap starts (`burst4.cpp:195, 212-214`).
  - Prediction is perfect (`burst4.cpp:202`).
  - CCD confinement costs nothing in the bench but −3.5% in the engine.
- Why reordering can't reach the bar, from the gate's own numbers:
  - The f=2/3 staged-first arm (509.5 us) implies tiles held in L3 run at about 130 GB/s with 40 workers, only about 1.5× the 86.6 GB/s DRAM rate. The kernel is close to compute-bound.
  - So any order that runs the two kinds one after the other saves at most 18.8 MB × (1/86.6 − 1/130) ≈ 70 us, about 11% (measured 12.2%).
  - Reaching 20% needs L3 and DRAM work to truly overlap. In the interleaved arm, DRAM fell to about 66 GB/s (37.6 MB / 572.5 us) instead of holding about 87.
- The plan's own figures agree. It maps the one-CCD 24% to about +2.5% (`RESEARCH-2-RANKED-PLAN.md:275, 282`), so the measured 12% is worth about +1.2% before the predictor's cost (about +11 us a layer on rank 0, `RESEARCH-2-RANKED-PLAN.md:218`). That is inside one arm's ±1% noise.

**Findings (none block anything)**
1. **Low, wording.** The report compares the dedicated stagers' p90 (1.0–1.2 ms) with the woken workers' median (219 us). The woken workers' p90 is 246 us, and the dedicated stagers' medians are 360 and 379 us. The conclusion doesn't change.
2. **Low, process.** `quietrun6.sh:20` checks the production server's `/stats` only before the lock wait (up to 30 min, line 31), not just before the run. Inside the lock, lines 36-44 check only windows, compilers and CPU load. This is harmless here because the arms alternate burst by burst (`burst4.cpp:220`) and the spreads are tight. Anyone reusing the script for other benches should re-check `/stats` right before the run.
3. **Low, process; for main.** Load rule 1's `pgrep -f 'ab_[w]indow.sh'` (`AGENT-LOAD-RULES.md:9`) matches any process whose command line names `ab_window.sh`, including Bash-tool `bash -c` wrappers. A long-running loop of that kind keeps the rule "busy" for everyone, itself included. Windows started by the launcher keep `WINDOW_PENDING` and the build lock for their whole run, so the flag test alone is enough for them.
   - `winbusy.sh:13` skips `-c` shells. A window started outside the launcher would be caught only by its radiance check (lines 18-19).
4. **Info.** The plan's "the three commands in that directory" (`RESEARCH-2-RANKED-PLAN.md:286`) is indeed unclear. `research/ccdl3/` runs burst4 only as the 7-worker smoke test (`run3.sh`, `run4.sh`), and `run1.sh`'s three commands are for l3gap. The three settings the implementer chose are a reasonable reading, and the design point decides the gate on its own.
5. **Info.** `l3stage-draft-unbuilt.patch` would need a full review if anyone revives it. Its off path is not identical to before:
   - `Slot` gains `dclaim[8]`, and `start()` initialises it whether staging is on or off.
   - `submit()`'s else branch writes `s.job.staged = false` even with staging off.
   - `run_one_l3` reads `staged_live_` and the worker's domain once, outside its retry loop.

**Not checked or still open**
- I did not re-run `trace_bursts.py` (the claims of 70% miss-free layers and 2.78 hits a burst) because a GPU window was running. I only read files. The 70% figure matches the plan (`RESEARCH-2-RANKED-PLAN.md:279`).
- Nobody priced the plan's alternative (copy the predicted expert into the 1-unit VRAM file buffer, `RESEARCH-2-RANKED-PLAN.md:288`). In the best configuration it conflicts with `K` (the RAM cache holds a CPU-only packed form) and with `all`+`M16`+`B1` (the CPU takes every routed unit). Its other open question, what an H2D copy does to all-reduce latency, needs a GPU window. That decision belongs to main.

# pfx: implementation report

PFX ITEM REPORT: claude/ds41-pfx (worktree /home/as/ds41-claude-build/wt-pfx)

Status: the work is built and all tests pass. It has **not** been measured on the cards, because GPU use was not allowed. Never pushed.

**Commits** (first-parent order, on top of dc83cecc)
- 5378696d: deepseek41: the on-die reads by a byte budget in use order, and at the Engram layers. Adds `arch/deepseek41/ds41_prefetch_plan.h`, `tests/ds41_prefetch_plan_test.cpp`, README and ab_window parts.
- 8a4b7f5b: tests: `ds41_prefetch_arch_test`. It runs deepseek41's declare and step on `arch_fakes.h` with the real TP4 geometry. `arch_fakes.h` gains `RadBuilder::kv_block_of`; when it is unset, nothing changes.
- 6447b43f: merges claude/ds41-decode-impl at 446016ed (FP8 dense, read-ahead in pieces, concurrency) into this branch.
  - The FP8 branch's old four-read block is dropped.
  - Its `wbytes()` is kept, with a GGUF case added that returns 0.
  - ab_window.sh keeps both log greps.
  - The newer main (9c8dccb9) also merges into this branch without conflicts (checked, not included).
- a351885a: under `DENSE_FP8`, the shared expert's reads are counted at their E4M3 code size (`LinearFP8::bf16_alt`). Adds the FP8 test case and FP8 sizes in the docs.

**Switches** (all off by default; `PREFETCH_ATTN=1` alone declares and issues exactly 16be66c5's reads, the same ops, bytes, order and points)
- `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1`: unchanged master switch.
- `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB=N`: a budget of N MiB per issue point.
  - It is filled in use order: ekv, comp, qa, kv, iq, qb, oa, ob, rt, sgu, sdn.
  - The read that crosses the budget is cut to whole 4 KiB, and dropped if under 1 MiB.
- `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET=k1,k2,..`: an explicit list and order. Extra names: iw, ik, ma, mf.
- `RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM=1`: Engram layers 1 and 14 are read too.
  - Their engram_kv is read by the layer before, if `ekv` is in the list.
  - Everything else is issued right after the Engram GEMM, before the all_gather.

**ab_window.sh arm parts** (each implies Pa)
- `PmN`: the budget. It is not `Pb`, because `Pb` is already the read-ahead's piece size in main.
- `Pl<k1+k2+..>`: the explicit list.
- `Pe`: the Engram layers.

**A/B arms** (all arms use this plugin)
- Use `H3` instead of `H` and pass `RADIANCE_HOME_DIR3=/home/as/ds41-claude-build/home-pfx`. That run home has home2's kernels (FP8 libr4d) and this plugin.
- `launch_window.sh` copies ab_window.sh from the main tree, so merge this branch into main first, or the new parts are only logged as unknown.
- With B = `all-t40-f-M16-H3-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4`, run interleaved:
  - B, B-Pe, B-Pm56, B-Pm56-Pe, B-Pm48-Pe, B-Pm40-Pe
  - then the same six with `-r2`
  - then `-r3` for B and the best arm.
- Optional without F8 (the task's stated best arm, with H3): `...-Sc2` against `...-Sc2-Pm56-Pe`. At bf16, Pm56 is about the same as the default reads.

**What to look for in the server log** (rank 0's lines are also copied to window.log)
- Startup line per rank, `rank R: on-die reads (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN): ...`. Printed by the plan printer for the real geometry:
  - B: `qa,kv,qb,oa whole (the default); 37 layers read ahead, 5-5 reads, 26.8 MiB a layer`.
  - B-Pm56-Pe: `8-10 reads, 55.6 MiB a layer (55.5-56.0); ... rt 36, sgu 36 (5 cut), sdn 31; Engram layers ... 1 37.5 + 55.5, 14 37.5 + 56.0`.
  - The same plan in bf16 (no F8): 55.9 MiB a layer, mostly the four reads plus about 2.5 MiB of wo_b.
- Shutdown line, `rank R: on-die reads issued live in N decode steps: X reads at Y layer tails, Z GiB (M MiB a step, ...); behind the Engram GEMMs ...`. Expect about 990 MiB a step for B and about 2244 for B-Pm56-Pe.
  - It is printed from a new `rad_plugin_close`. Passes played back from a recording are not counted; host-expert passes always run live.
- Decode tok/s and identity: `work/ab_sum.py`. Every reply should be identical, since the reads only read.
- `EP staging time ... waiting for the card at the routers`: this should fall if the reads help (Pa took it from 26.5-29.1 to 25.3-28.5 s).

**Built artifacts**
- radiance: /home/as/ds41-claude-build/aql-pfx/bin/radiance. It has no core change from this item; it is main at 446016ed.
- deepseek41.so: /home/as/ds41-claude-build/aql-pfx/radiance_home/architectures/deepseek41.so (md5 9cce4e7f…). It is also copied into /home/as/ds41-claude-build/home-pfx.
- libr4d.so: unchanged by this item. Use home2's, which has `gemm_nt_bf16f8`.

**Tests** (all pass in /home/as/ds41-claude-build/aql-pfx)
- New: `ds41_prefetch_plan_test` (8 cases) and `ds41_prefetch_arch_test` (8 cases).
- Existing: host_expert_pool_test, place_test, mapped_slab_pregrow_test, and arch_test and glm5next_test (rebuilt because `arch_fakes.h` changed).

**Uncertain or unfinished**
- The gain is unknown. My rough upper bound with F8 at Pm56 is about +2% (about 1 ms a token). It could be zero or negative if:
  - the shared-expert reads are evicted by the routed experts' gate_up stream before they are used (the router read is the safest part);
  - 56 MiB plus other traffic overflows the 64 MB cache (Pm48 and Pm40 test this);
  - the reads behind the Engram GEMM, or reads past a short CPU wait, compete with the next attention for VRAM bandwidth.
- The step lint is not run in the arch test: the fake setup reads the experts as block fp8 and wo_a reads column slices. Neither comes from this change.
- Scratch files (build script, plan printer, `plans.txt`) are in /home/as/.claude/jobs/635603c1/tmp/research/pfx/.

# pfx: review

**PFX REVIEW: claude/ds41-pfx at a351885a, against dc83cecc**

**Verdict:** Nothing in the code needs fixing before a GPU A/B. With the switches off, the issued ops are the same as before. The new reads only read, so they cannot change an output bit by themselves. The one exactness risk is timing beside a collective (finding 1). The A/B does need three setup steps first, listed next.

**Must do before the A/B (setup, not code)**
- **Get the new parts into the script the window runs.** `/home/as/ds41-claude-build/work/launch_window.sh:14-15` copies `ab_window.sh` from the main tree (`/home/as/radiance-ds41-claude`). Without a merge there, every `Pm`/`Pl`/`Pe` arm is **skipped**, not only logged (`/home/as/ds41-claude-build/wt-pfx/tools/ds41-host-experts/ab_window.sh:349` sets `skip=1`). The implementer says the merge into main has no conflicts.
- **Pass the run home and mirror.** Add `RADIANCE_HOME_DIR3=/home/as/ds41-claude-build/home-pfx` (and `MIRROR=` for the `R` part) as extra arguments to launch_window.sh.
- **Recheck the plugin against main's binary at window time.** The window uses main's `aql/bin/radiance`. Main is now at a566c066; `abi/`, `arch/` and `libr4d/` have not changed since 446016ed, so the pfx plugin matches it today. If they change before the window, rebuild `deepseek41.so` from main plus pfx.

**Findings, most severe first**
1. **MEDIUM (exactness, watch in the A/B): `Pe` reads run beside every Engram all_gather** (`arch/deepseek41/deepseek41.cpp:2649-2656`).
   - The decode all_gather is about 6,400 elements a row, far below libr4d's full-fence threshold of 131,072, so it has no full fence.
   - Prefetch reads beside prefill all-reduces once changed a fresh server's first reply (`arch/common/rad_arch.h:629-632`).
   - `Pa` overlaps decode collectives only incidentally. `Pe` overlaps the Engram all_gather by construction, twice a token.
   - Do not adopt `Pe` unless `ab_sum.py` shows every reply identical. For the winning arm, also re-run the 16 chat prompts and 256-token replies, as the 11:33 entry in NIGHT-LOG.md did for `Pa`.
2. **LOW (budget accounting, only once wt-router merges):** the compressor and router sizes are hard-coded as f32 (`deepseek41.cpp:1149-1150` and `1163-1165`).
   - Under wt-router's `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16` (c86286cf), the ratio-2 compressor is declared bf16 and the router is stored as bf16 halves.
   - There is no over-read: the weight template carries the stored dtype (`core/runtime/ctx.cpp:306`) and `r4d_rad_prefetch` clamps the read.
   - But the budget charges 20 MiB for a 10 MiB compressor read and 7.5 MiB for a 3.75 MiB router read, so later reads get dropped. Size these from the stored dtype when merging the two items.
3. **LOW (claim accuracy):** "ATTN=1 alone declares exactly 16be66c5's reads" is not quite right.
   - 446016ed declared 200 prefetch ops, five per layer, including layers 0, 1 and 14 whose reads are never issued, placed between `o_b` and `ar_attn`.
   - pfx declares 185, at the end of each layer (the test checks 185 at `tests/ds41_prefetch_arch_test.cpp:235`).
   - What is issued is identical: same ops, bytes, order, points and lanes. The buffer plan is unchanged too: the packer sorts by size, then start (`core/build/rad_bufplan.cpp`), and ops that name no buffer only shift indices. Harmless.
4. **LOW (docs):**
   - The report says an unknown part is "only logged"; the arm is in fact skipped (`ab_window.sh:349`).
   - Commit 8a4b7f5b's message says "the step lint is clean", but the test runs no step lint (`tests/ds41_prefetch_arch_test.cpp:474`).
5. **LOW (A/B guard):** unlike F8, nothing checks the server log for the P parts. A malformed or inherited `PREFETCH_*` variable shows only as "ignored:" in the startup line, and the arm still runs. Check rank 0's startup line in window.log for every arm.
6. **LOW (docs):** `ekv` listed without a budget reads all of engram_kv: 75 MiB split or 300 MiB replicated in bf16. Both exceed the 64 MB cache, so the read only evicts. It makes sense only with `Pm` or F8.
7. **TRIVIAL:**
   - `pf_note`: with Engram layers but no `prefetch` op declared, it prints "0 layers read ahead" instead of "none declared" (`deepseek41.cpp:2082-2086`, `:2118`).
   - The shutdown line's "at N layers" counts layer-steps (`:2995`).
   - `PREFETCH_ENGRAM=""` is reported as malformed, while empty MIB or SET values count as unset (`ds41_prefetch_plan.h:108`).
8. **TEST GAP (LOW):**
   - No test checks that, with the switch on, the non-prefetch issue sequence and lanes match the switch-off run. That is the exactness argument.
   - The GGUF-trunk early return (`deepseek41.cpp:1124`) is untested.

**Checked and correct**
- **Switch off:** `pf_layer` returns at once and both read lists stay empty. `pf_issue` returns on an empty list. The tail callback does nothing, as before. `rad_plugin_close` prints nothing.
- **Weight sizes:** every computed size matches its weight's declaration (q_a, kv, q_b, wo_a, wo_b, compressor at ratio 2 and 1, indexer, mixes, router, shared slice, engram_kv). Weights are 256-byte aligned. The bf16f8 clamp holds.
- **Lanes:**
  - Lane 0 never waits on the reads: weightless ops do not set `lane1_pending_` and ds41 declares no lane-1 window (`core/runtime/issue.cpp:596-617`).
  - Every path returns to lane 0, and `Pe` issues from lane 0.
  - Chain mode is never on for ds41: the plugin sets no `chain_pays` and the A/B runs with no speculation (`core/engine_bringup.cpp:1160`). So the lane-0 path runs only in tests.
- **Lifetimes and counters:** `rad_plugin_close` is exported (checked with `nm`). It runs from `Registry::close()` after `ranks_.clear()` (`core/engine.cpp:266-270`) and before `dlclose`, so the models are still alive. Only the rank's own thread writes the counters. Passes the stager arms are always issued live, so they are counted.
- **Arm parts:** `Pm<N>`, `Pl<names>` and `Pe` each set `PREFETCH_ATTN=1` plus their own variable. No earlier pattern shadows them, and no other worktree defines these names.
- **Build and artifacts:**
  - CMake options match build_hx.sh's.
  - The two new tests pass when I re-ran them (8 and 8 cases), and ctest at 15:27 passed all seven listed tests.
  - `deepseek41.so` md5 is 9cce4e7f986f72ca4511939f3cc219ab in both `aql-pfx/radiance_home/architectures/` and `home-pfx/architectures/`. home-pfx's libr4d.so is the same file as home2's (fdb77e8d…).
  - The .so was built at 15:10, before the a351885a commit at 15:28. The test binary from that build already contains a351885a's FP8 test case, so the build included those changes. I did not rebuild to compare.
- **Branch:** not pushed; authorship and Co-Authored-By trailers are correct.

**A/B arms**

B = `all-t40-f-M16-H3-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4`

Run interleaved:
- **Round 1:** B, B-Pm56-Pe, B-Pm56, B-Pm40-Pe, and a control arm `BH` (the same arm with `H` instead of `H3`) to confirm the pfx plugin with `Pa` alone matches main's plugin.
- **Round 2:** the same four with `-r2`.
- **Round 3:** B-r3 and the best arm with `-r3`.

Repeat pairs in NIGHT-LOG differ by up to 0.9% (15.11 / 14.98), so use at least three repeats for the arms that matter. `B-Pe` alone should gain 0.2% or less, which is within the noise; drop it and B-Pm48-Pe if time is short.

**What to look for in the server log (rank 0 is also in window.log)**
- **Startup line:**
  - B: `qa,kv,qb,oa whole (the default); 37 layers read ahead, 5-5 reads, 26.8 MiB a layer`.
  - B-Pm56-Pe: `a budget of 56.0 MiB ... 8-10 reads, 55.6 MiB a layer (55.5-56.0) ... Engram layers (MiB ahead of the GEMM + behind it) 1 37.5 + 55.5, 14 37.5 + 56.0`.
  - There must be no "ignored:".
  - The `BH` control prints no such line.
- **Shutdown line:**
  - Decode steps should be about the number of decode tokens.
  - Rank 0: about 990 MiB a step for B and about 2,244 for Pm56-Pe.
  - "Behind the Engram GEMMs" should be above 0 only in `Pe` arms, at 2 points per step.
- **Other lines:**
  - The F8 line must be present.
  - "EP staging time ... waiting for the card at the routers" should fall.
  - "ms waiting at gathers" should not rise.
  - No `error` or `ERR` lines.
- **Identity:** `work/ab_sum.py` must show every reply identical, especially in the `Pe` arms.

**Notes**
- A GPU window was running during the review. I did no builds and only ran the two new test binaries (0.47 s of CPU on one thread, below the load-rule thresholds).
- New worktrees appeared while I worked: wt-pfx2, wt-pfk and wt-tall. I did not look into them.

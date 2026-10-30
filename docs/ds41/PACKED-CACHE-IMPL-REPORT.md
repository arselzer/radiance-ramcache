The packed RAM cache passed the scan gate and is built, tested and committed. It is opt-in and has not been measured on the GPUs yet. No GPUs, services or protected trees were touched. A capacity-only gain is +2.5–4.5% decode; the kernel speedup below is on top of that, so the A/B may show more.

**Worktree:** `/home/as/ds41-claude-build/wt-packed`, branch `claude/ds41-packed-cache`, not pushed. The main branch moved 3 commits during the work (now at 16be66c5). I rebased my branch onto it, so the main branch can now fast-forward to mine. The only conflicts were in README.md and ab_window.sh; I kept both sides. Everything was rebuilt and retested after the rebase.

**Commits:**
- b037ba7d ds41 host experts: a packed form of a weight's scales, and kernels that widen it
- 1781a99d Mapped slab cache: class sizes set at configure
- b313a19e Mapped file cache: a packed layout only the host expert path reaches
- b70f4522 Host expert pool: packed jobs
- a9e08868 ds41 host experts: the RAM cache packed for the CPU only (opt-in)
- 678f1782 Test the packed RAM cache through the actual HOST mover

The previous attempt's uncommitted kernel edits were sound, so I kept them. I split packing into a plan step (counts raw rounds, writes nothing) and a pack step.

**Scan (gate passed):** all 40×384 experts, 17 GB of scale tensors, read in 124 s. It pauses whenever a window runs. Files are in `/home/as/ds41-claude-build/work/packed/run2/` (`summary.json`, `scan_full.npz`, `budget.py`, `choose_budget.py`, `over_budget.py`).
- **Distribution:** e8m0 is 120 (58.7%) or 121 (41.2%) almost everywhere. Raw rounds (rounds outside the 2-bit window) are 0.0046% of gate_up rounds and 0.0107% of down rounds. 92% of gate_ups and 86% of downs have none.
- **Tail:** the outliers are in layers 37–39, up to 3331 raw rounds in one down. The 27-expert sample missed this tail.
- **Layout chosen:** 2-bit codes from one base per weight (gate_up = [w1; w3] is one weight), with raw rounds stored beside.
- **Fixed budget:** 512 raw rounds for gate_up and 320 for down. Slots are 11,999,296 + 6,001,984 bytes, so the cache holds **×1.0444** the pairs (about 1424 instead of 1364 per rank).
- **Never cached:** 7 of 15,360 experts don't fit, so 99.954% do. They are L37 E123, L38 E250 and L39 E007/E087/E188/E313/E382, together 0.042% of the decode profile's selections. They stay misses, computed from their planes.
- **Alternatives:** 256/256 gives ×1.0451 with 15 experts left out; 600/3400 gives ×1.0378 with none. The 1-bit layouts are worse: ×1.013 with raw rounds only, ×1.036 with a 2-bit patch.

**Kernel tests and bench:**
- **Tests:** packed is bit-identical to planar in every case. That covers all 27 sampled real experts, a raw round at every block position (including down's 8-block tail), the general path (windows 126–129, 110–113 and subnormal scales), NaN rows, a whole expert forward, and rows 2–5. The base choice matches a brute-force search, and the API refusals behave as specified. The existing checks also pass with packed added.
- **Bench** (pool-shaped: 64-row tiles, 32 experts streamed from DRAM, real scales, medians of 7 runs), ms per expert:

| layout | 40 threads | 24 threads |
|---|---|---|
| block_mxfp4 | 0.416 | 0.387 |
| packed | 0.381 (−8.5%) | 0.324 (−16.3%) |
| planes | 0.387 (−6.9%) | 0.354 (−8.6%) |

  The rows are identical in all three layouts.

**Engine** (opt-in `RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1`; budgets can be changed with `..._PACKED_CACHE_RAW=<gate_up>,<down>`):
- **Refusals:** the mover refuses the knob without the mapped slab cache and a file tier, and the stager (`hx_init`) refuses it unless `hx_capped_` holds.
- **Slab cache:** class sizes are set at configure time; the default is the old block sizes.
- **Filling:** the miss inserts and the warm-up both go through `insert_unit_planes`. It checks both weights of a unit against the budget first, then copies the codes and packs the scales straight into a reserved slot. Each weight's scales are compared byte for byte against the source before it is published; a failure is counted and never published.
- **Hit path:** the pool gets packed jobs and refuses slots that don't match the weight.
- **Layout tag:** every block_mxfp4 call from the card's copy path is turned away as a miss or bypass and counted, so the card path reads the container instead.
- **Card-path fills:** the card path's direct-fill reads do not fill the packed cache; they bypass it on purpose, because with the cap that path never runs ("0 dispatches, 0 staged" in the best-config log).
- **Default off:** the code paths are unchanged.

**Unit tests:** all pass on both builds after the rebase.
- GPU build `/home/as/ds41-claude-build/aql-packed`: host_expert_pool_test 6/6 (with a new packed-jobs case), mapped_slab_pregrow_test 3/3 (new runtime sizes case), place_test 112/112. The 3 dspark_two_bank_actual_HOST_* cases also passed here.
- HOST-only build `/home/as/ds41-claude-build/aql-packed-host`: native_mapped_copy_mover_test 31/31 and place_test 112/112. I added this build because native_mapped_copy_mover_test refuses a GPU binary. It includes a new layout-tag case and a Mover-level case with real MXFP4 planes: packed insert and borrow, the over-budget unit, and the card path publishing correct block bytes without touching the cache.

**GPU A/B to run:**
- **Binary:** use the snapshot `/home/as/ds41-claude-build/work/packed/radiance-a9e08868` (sha256 ed8791fb…) for both arms. Pass it to `launch_window.sh` as `RADIANCE_BIN=…`; your VAR=val arguments come after its own setting, so they override it.
- **Script — must do first:** `launch_window.sh` copies ab_window.sh from `/home/as/radiance-ds41-claude`, which has no `K` part. A `K` arm would only log "unknown part K" and silently run unpacked. Fast-forward `claude/ds41-decode-impl` to `claude/ds41-packed-cache` (or pick the ab_window.sh hunk from a9e08868) before launching.
- **Arms:** your current best arm against the same arm with `-K`, interleaved with `-r2` repeats. For example `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er` and `…-Er-K`, or the newer `…-H3-…-Es-Pa` variant, with `MIRROR` set as usual. No extra RAM is needed: same cache budget.
- **What to check in the K arm's log:**
  - the startup line "native mapped cache: packed for the CPU only"
  - the warm-up count (about 1424 instead of 1364)
  - the shutdown line "packed RAM cache": 0 round-trip failures, at most 7 distinct over-budget units, and 0 block calls turned away
  - replies identical to base

**Not done:**
- The GPU A/B itself.
- I did not build each intermediate commit separately; only the final tree, before and after the rebase.

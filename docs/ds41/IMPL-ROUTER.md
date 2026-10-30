# router: implementation report

The router change is built and committed, and all the tests it touches pass. Nothing ran on a card: the speed gate and the decode gain are still unmeasured.

**Branch and commits**
Branch `claude/ds41-router` in `/home/as/ds41-claude-build/wt-router`, on top of dc83cecc:
- `6d72789d` — libr4d: new op `gemm_nt_f32bf16`, which is `gemm_nt_f32` over an f32 weight stored as its bf16 halves.
  - The tile kernel now takes the stored weight type as a template parameter. A bf16 half is widened back by a shift before the same fmaf, in the same lane, step and order, so every output bit is unchanged.
  - At load the kernel row stores an f32 plane as row-major bf16. It checks every element and refuses the weight at the first nonzero low half instead of rounding it. A bf16 plane is read as stored.
  - The op is documented in `docs/OPS.md`.
- `c86286cf` — ds41 switch, plus small supporting changes:
  - `GlmRouter::Config::bf16_halves` and the split router's op name. GLM never sets it and declares exactly what it did before.
  - The router predictor can read a relaid router's f32 plane, so `Ps`/`Po`/`Pt` arms keep working with the switch on.
  - `ab_window.sh` parts, README row, `scan_low_halves.py`.

**What I checked on the real container** (`scan_low_halves.py`): all 40 routers are f32 [384, 5120] with 0 of 78.6M low halves nonzero, so the load will not refuse them. The three compressor kv|gate weights (layers 2, 8, 14) are already bf16 in the container; the loader had been widening them, and with the switch they are declared bf16 and read as stored. Together this saves 180 MiB a card (150 of it the routers'), roughly +10 VRAM experts a rank.

**Switch and arm parts**
- `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1` — arm part `Rb`. Trunk only; the DSpark drafter's routers stay f32.
- `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16_FORM=N` (1–8) — arm part `RbN`. Picks the bf16 GEMV's tile form at passes of up to 16 rows, without a rebuild. All forms give identical bytes.
- Every arm now states the knob (0 when off). An `Rb` arm without its startup line in the server log is not measured, and neither is a plain arm that has one.
- `SELFTEST_F32BF16=<r4d_selftest>` runs the card-0 byte-compare and timing sweep before the arms.

**A/B against the best arm**
`launch_window.sh` copies `ab_window.sh` from the main tree, which has no `Rb` part and skips arms with unknown parts. So first merge my branch, or at least its `ab_window.sh`. Then, with `B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2`:
```
/home/as/ds41-claude-build/work/launch_window.sh window_router "$B $B-Rb $B-r2 $B-Rb-r2 $B-r3 $B-Rb-r3" \
  RADIANCE_BIN=/home/as/ds41-claude-build/aql-router/bin/radiance \
  RADIANCE_HOME_DIR2=/home/as/ds41-claude-build/home-router \
  SELFTEST_F32BF16=/home/as/ds41-claude-build/aql-router/bin/r4d_selftest \
  MIRROR=/srv/bulk/source-copies/ds41-native-mxfp4/d8b2e3c20f5f4b6d5044d37f20b8d53f260f83e466f72259572be76d7e756bf3-main
```
- Both arms use the same binary and home.
- Optionally add `$B-o $B-Rb-o` (`--profile-ops`) to get per-call times for `gemm_nt_f32` against `gemm_nt_f32bf16`.
- If the sweep shows another form beating the rule at 384×5120, add `$B-RbN`.
- Expected gain is about +0.8 to 1.2%, under the ±0.5–1% run-to-run noise for one pair, hence three pairs.
- Your main tree now has FP8, read-ahead and other merges, so the current best arm differs. After merging, rerun the same pair on top of it.

**What to look for in the server log**
- **Startup line (Rb arms):** `radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16: 40 routers and 3 compressor kv|gate projections read as bf16 (gemm_nt_f32bf16), 180.0 MiB less a card`. The script copies it into `window.log`.
- **VRAM budget:** "static weights" should drop from 4.79 GiB to about 4.61 GiB, and "planned units VRAM=598/615" should rise by about 10 a rank.
- **Load line:** "load: N weight(s) ... rearranged for their kernels" should be 160 higher (40 routers × 4 ranks).
- **Expected, harmless:** `gemm_nt_f32bf16` appears in the "host-site execution: … no host kernel" list.
- **Failure signature (not expected):** `[r4d] f32bf16: … nonzero low half … refused rather than rounded`, then `load: gemm_nt_f32bf16_bf16 could not produce '<weight>'`; the arm then fails at startup.
- **Shutdown:** "EP staging time: … waiting for the card at the routers" should fall, and replies must stay 8/8 identical.
- I added no shutdown counter: the switch has no runtime decision to count, and its effect shows in the lines above.
- **Selftest lines in `window.log`:**
  - every "gemm_nt_f32bf16 at M=… byte-identical" check passes;
  - the `--perff32` table prints f32 and bf16 times for each form;
  - gate: the bf16 rule at N=384, K=5120, M=1 is at or under 14 µs (f32 is 19.7 µs in the engine).

**Built artifacts**
- `/home/as/ds41-claude-build/aql-router/bin/radiance`
- `/home/as/ds41-claude-build/aql-router/bin/r4d_selftest`
- `/home/as/ds41-claude-build/aql-router/radiance_home/kernels/libr4d.so`
- `/home/as/ds41-claude-build/aql-router/radiance_home/architectures/deepseek41.so`
- Run home `/home/as/ds41-claude-build/home-router`: my `libr4d.so` and `deepseek41.so`, with `libref`, `libavx` and `libquant` copied from home2.

**Tests (all pass)**
- `f32bf16_relayout_test` (new): the conversion over all 65,536 bf16 patterns, the refusals, and the inverse.
- `r4d_emu_f32`: 132 cases on the host emulator, including both weight types at every tile form against the host arithmetic.
- The rest:
  - `glm5next_test` (new router case, and GLM's full graph unchanged)
  - `arch_test`
  - `host_expert_pool_test`
  - `place_test`
  - `mapped_slab_pregrow_test`
  - `relayout_parity_test`
  - `ops_doc`
  - `isa_tools`

**Unfinished or uncertain**
- `shipped_plugins_test` fails in this build: 6 cases fail because it needs every shipped plugin and I built only `deepseek41`, `libr4d`, `libref` and `spechub`. In it, `deepseek41.so` loads and claims its key.
- The bf16 tile rule copies the f32 rule. Bf16 loads carry half the bytes for the same number of load instructions, so the default may miss the 14 µs gate. The selftest sweep and `RbN` cover that; changing the rule is one line.
- A further exact step would store each row in an order that lets a lane load 8 steps in one 128-bit load. I estimate about 8 µs for the router GEMV; not implemented.
- The predictor path with the switch on is untested on the engine.
- Merging into the current main conflicts in 7 files, all adjacent insertions next to the FP8 and read-ahead changes:
  - `deepseek41.cpp`
  - `mover.h`
  - `OPS.md`
  - `r4d_plugin.h`
  - `tests/CMakeLists.txt`
  - `README.md`
  - `ab_window.sh`

# router: review

I found no way for the switch to change an output bit, and no race or lifetime bug. With the switch off, behaviour is identical to dc83cecc on this container. The `Rb` arm part really sets the switch. No code defect blocks a GPU A/B, but the window as proposed would not run the `Rb` arms at all (item 1 below).

I re-ran the host tests outside a window and they pass: `f32bf16_relayout_test` (61 checks), `glm5next_test` (29 cases), `r4d_emu_f32` (132 cases, 0 failures). The `shipped_plugins_test` failures are only the missing plugins (no libavx or other architectures in the build home), as the implementer said. Paths below are relative to `/home/as/ds41-claude-build/wt-router` unless absolute.

**Before a GPU A/B**

1. **Blocker – the window would skip the `Rb` arms.** `/home/as/ds41-claude-build/work/launch_window.sh:15` copies `ab_window.sh` from the main tree. Main's copy has no `Rb`/`RbN` parts, skips any arm with an unknown part (its line 361), and never reads `SELFTEST_F32BF16`. The proposed window would run three plain B arms and no card check.
   - **Fix:** put this branch's additions into the script the launcher copies: the log check (`ab_window.sh:160-172`), the selftest block (`:213-230`), the parts (`:284-285`) and the per-arm env (`:323-325`). Keep main's skip-on-unknown rule and F8 checks alongside them.
2. **Must read – the card byte-compare is the only real exactness check, and nothing gates on it.** `ab_window.sh:217-230` logs the FAIL count and exit code, then runs the `Rb` arms anyway. 8/8 identical greedy replies would not catch a 1-ulp logit difference. Accept an `Rb` result only with `f32bf16 check FAIL lines: 0`, `check exit 0`, and every `form … byte-identical` line ok. Better: skip the `Rb` arms automatically when either is nonzero.
3. **Must check for the merged A/B – the read-ahead's router predictor.** Main's best now includes `Pp4` (read-ahead in pieces), which needs `RouterPredictor::load` to succeed.
   - Main's `router_predictor.cpp` still compares `stored_bytes`. A router stored as bf16 halves fails that test, so read-ahead would be silently off and `Rb` would look like a ~3.6% regression.
   - The merge must keep this branch's `weight_file_bytes` check: `core/place/router_predictor.cpp:111-116`, `core/rad_core.h:263-267`, `core/place/mover.cpp:1246-1250`.
   - Every `Rb` arm's log needs `router predictor: 40 routers loaded` and no `…router or ffn_norm is not f32` warning.

**Findings**

- **Medium, missing test:** nothing tests the predictor's read of a relaid router (`rad_core.h:263-267`, `mover.cpp:1246-1250`). I checked it by reading (`file_offset` is the selected plane's, `core/engine_bringup.cpp:3083`), but with `Pp4` in the best arm it decides whether read-ahead works under `Rb`.
- **Low–medium, missing test:** nothing tests the deepseek41 side of the switch:
  - the compressor dtype from `weight_enc` on the kv|gate concat name (`arch/deepseek41/deepseek41.cpp:1270-1281`);
  - the refusal when no library declares the op (`:1939-1945`);
  - the counts.
  `glm5next_test` covers only `GlmRouter`.
- **Low:** the startup counts are unconditional (`deepseek41.cpp:1282-1285`, `:1480-1483`). A weight that ends up stored f32 (an f16 plane widened, or `weight_enc` failing) would still be counted, and the `Rb` log check trusts this line. They are correct for this container: I read its index and found 40 f32 routers [384, 5120] and 3 bf16 kv|gate [1024, 5120].
- **Low:** `RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16_FORM` (`libr4d/r4d_glm.hip:133-143`, `:162`) leaves no trace in the server log, so an `RbN` arm can't be checked against its name. It also applies to the compressor GEMVs, not only the routers.
- **Low:** on the card, the non-default tile forms are byte-compared only at M=1 (`libr4d/r4d_selftest.cpp:19318`). `RbN` applies its form up to 16 rows, which is every decode and prefill pass in the A/B; M=2–16 is checked only on the host emulator.
- **Info – behaviour change outside the switch:** `read_weight_file` now refuses a weight the loader widened or narrowed; before, it read past the plane. This container never hits it (`ffn_norm`, `correction` and the router are all plain f32), so switch-off is identical for the A/B.
- **Info – correct by reading, never exercised:** the split router (steps of 256+ rows) over the bf16 form. Its offset is counted in stored-dtype elements (`core/runtime/ctx.cpp:306-311`, `issue.cpp:730`, `arch/common/rad_glm_tp.h:141`), so it lands on the right row. No test covers it, and the A/B's 16-token passes never reach it.
- **Info – speed gate:** the default decode form (2x2x16) keeps the same 10 serialized memory round trips per wave as f32, so it may miss the 14 µs gate.
  - Forms 6, 7 and 8 issue 80–128 loads at once, above the card's 63 outstanding, so they partly serialize.
  - Form 5 (1x2x16, 48 loads) is the likeliest winner at one row.
  - Run the sweep in a short window first, so the A/B can include the right `RbN`.
- **Info:** this branch's `ab_window.sh` only logs unknown parts, so a typo like `Rb9` would run as a plain arm under an Rb name. Keep main's skip rule in the merge.
- **Info:** no shutdown counter is acceptable for a load-time switch; the startup, VRAM and load lines show its effect.

**Verified**

- **Bit-identical math:**
  - The only kernel change is how a weight is loaded. The bf16 half is widened by a 16-bit shift (`libr4d/r4d_common.h:69-71`), the same shift the loader uses (`abi/rad_plugin.h:176-181`).
  - Lanes, steps, fmaf order and butterfly are unchanged, and libr4d is built with `-ffp-contract=off`.
  - The relayout refuses any nonzero low half and never rounds.
- **No other reader of the router's card copy in ds41:** the fused router needs `!shared_ungated`, the expert prefetch needs `!sigmoid_router`, the drafter's routers are separate and stay f32, and the predictor reads the file.
- **Switch off:**
  - The default tile choice at `r4d_glm.hip:116` equals the old if-chain, and the dtype checks are unchanged.
  - No engine launch passes a `form` key.
  - GLM and all defaults declare exactly as before.
- **Arm parts:** `Rb` sets the switch to exactly `1`; `RbN` also sets the form to N. Every other arm sets both to `0`, which overrides anything in the window's environment. The startup-line check runs both ways.
- **Build:**
  - The `home-router` `libr4d.so` and `deepseek41.so` match the build's sha256.
  - Both were built after the last source edit and contain the new op and switches.
  - The cmake options match `build_hx.sh`.

**A/B to run (after item 1)**

On the branch build, with the implementer's env and `SELFTEST_F32BF16`, in alternating order:
```
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2
"$B $B-Rb $B-Rb-r2 $B-r2 $B-r3 $B-Rb-r3"
```
Add `$B-o $B-Rb-o` for per-call router times, and `$B-RbN` if the sweep beats the default.

Then on the merged tree, with `B2=$B-F8-Pp4` (confirm against main's README):
```
"$B2 $B2-Rb $B2-Rb-r2 $B2-r2 $B2-r3 $B2-Rb-r3"
```
A ~0.8% effect is only about 2σ over three pairs, so treat the selftest and `-o` router timings as the main evidence and the pairs as confirming the direction.

**Log checklist (additions)**
- `window.log`: FAIL lines 0, check exit 0, and the bf16 default form at or under 14 µs at N=384.
- `Rb` arms: the `ROUTER_BF16: 40 routers and 3 compressor kv|gate projections … 180.0 MiB` line.
- `Rb` arms: no `[r4d] f32bf16:` refusal.
- Merged `Rb` arms: `router predictor: 40 routers loaded`.
- 8/8 replies identical.

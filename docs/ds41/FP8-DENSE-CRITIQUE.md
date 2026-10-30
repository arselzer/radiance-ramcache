# FP8 design critique

## Verdict

The plan is sound and its exactness argument holds. The B registers are identical to fragU's. The split-K call is the same (cfg_for, BR_FLAT, NOTUNE falling back to the same defaults). Every wave runs one chain from zero in ascending k, followed by the same LDS reduction and store macros, proven ISA-identical. The multiplies are exact (no FMA), and the relayout check exhaustively accepts exactly the 60198 FP8 images per scale. Nearly every file:line anchor is right; the drift is noted under blocking_issues. The VRAM numbers (4520 to 2262 MiB on rank 0, 117-126 more experts per rank) and the "no planner code change" claim also check out.

Fix before any A/B:
- B1: add an ab_window.sh arm part for the knob. Today a missing part is ignored and the arm runs without it.
- B2: make the harness refuse an F8 arm when a tune cache or RADIANCE_TUNE is present.
- B3: land the prefetch byte-count fix, now required because Pa is committed.

Fold in as well:
- B4: commit 11.
- B5: take the scale offset from the weight's shape.

Close the test gaps:
- E2: the inline-zero accumulator in the peel (unreachable for ds41).
- E3: the one-hot test cannot see ±0 in B; dump the B registers directly.

Consider S2 (leave the shared expert for later) and S4 (r4d_emu) to cut risk.

Expect a smaller direct speedup than estimated. The shared expert overlaps the CPU-row wait, and Pa already serves four of the five attention weights from the on-die cache. The residency gain is the more reliable part: NIGHT-LOG:78 shows 182 more residents gave +9.5% decode.

## Blocking issues

Tree note: HEAD is now 16be66c5. PREFETCH_ATTN is committed and the plan's deepseek41.cpp anchors are unchanged. I changed no repository file. My only scratch files are in /home/as/.claude/jobs/635603c1/tmp/fp8design/review/ (sk_check.py).

MUST FIX before an A/B:

B1. The A/B wiring is missing, and when it is missing the arm still runs without the knob.
- ab_window.sh has no arm part for RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8. An unknown part is only logged and the arm runs anyway (tools/ds41-host-experts/ab_window.sh:278).
- launch_window.sh's VAR=val goes into every arm's environment, base included (work/launch_window.sh:20-24). So "launch_window.sh arms" (commit 9) cannot A/B within one window.
- Fix: add an `F8)` part (plus EXCLUDE) next to `Pa)` (ab_window.sh:245). Make the arm fail if its server log lacks the "dense weights read by gemm_nt_bf16f8" note. Run base and F8 on the same rebuilt libr4d.so and deepseek41 plugin from this branch. The homes currently hold copies of the loop's dev-build plugins.

B2. The tune precondition is not enforced.
- The bf16 rows m16 and m64 carry tGemmSkinny and r4d_tune_valid_gemm (libr4d/r4d_rows.cpp:7685, 7694). They take a cached point keyed by kernel name, or a RADIANCE_TUNE gemm_nt pin (core/plugin/select.cpp:324-369). The FP8 row is NOTUNE and always uses pick_sk.
- Today this holds: home/ and home2/ have no tune/, and ab_window.sh sets no RADIANCE_TUNE. Nothing keeps it that way.
- Fix: have the harness refuse an F8 arm whose log shows "tune cache: N row(s)" (core/build/startup.cpp:159-160), or when RADIANCE_TUNE is set.

B3. The prefetch over-read can now happen, because Pa is in HEAD.
- deepseek41.cpp:1392-1396 passes bf16 byte counts (n*k*2).
- The prefetches are issued from rad_block_moe_fp8.h:1837.
- r4d_rad_prefetch clamps only core dtypes (libr4d/r4d_prefetch.hip:63-70). rad_dtype_bits is 0 for a plugin dtype (abi/rad_types.h:184-188), and the kernel receives the stored dtype (core/runtime/ctx.cpp:306).
- Commits 5 and 8 must land before any arm with both Pa and F8. Otherwise every prefetch reads twice the blob: other weights get pulled into the 64 MB on-die cache, and the GPU faults if the blob is the last one in the pool.

STRONGLY RECOMMENDED:

B4. Make commit 11 part of the series, not optional.
- Problem: an unresolved gemm_nt_bf16f8 keeps its weight (core/build/rad_builder.cpp:825-835). resolve_sites then sets device_ok=0 (core/place/planner.cpp:145-164), which maps the whole dense layer unit (planner.cpp:296-345).
- Causes: drift between the lw8/LinearFP8 prechecks and the row, or a libr4d build without the row.
- The LoRA path already drops weights from declined ops (rad_builder.cpp:768). Commit 11 does the same for any op that resolves nowhere.

B5. Take the scale geometry from the weight, not from the issued tensors.
- The scratch kernel computes soff and ldws from g.N and g.K (scratch r4d_gemm_bf16f8_nt_m64.hip:151). Those come from y->shape[1] and a->shape[1] (r4d_gemm_args.h:371-373).
- The bf16 kernel tolerates a y narrower than the weight, because its row stride is K. The FP8 kernel would silently read the wrong scale rows.
- Fix: require g.N==b->shape[0] and g.K==b->shape[1] (the declared per-rank shape, ctx.cpp:303-305). No ds41 issue passes a narrower y today (deepseek41.cpp:2558-2561, rad_fp8.h:1572).

CORRECTIONS (plan text vs code, not blocking):
- §2.1: only RAD_E_DTYPE makes a row a non-candidate (select.cpp:236; rad_builder.cpp:524-544). RAD_E_SHAPE from the hook is a hard declare error (rad_builder.cpp:2182).
- §3.6: tiled_stage_check.py skips clamped rows (`if n0+rr>=N: continue`). The LDS words for a clamped tile do differ, which is harmless because those rows are never stored.
- Test B: the 570 entries hold 12.73 GiB (13.7 GB) of bf16 in model.rad, not ~8.4 GB. I summed the rad-info sizes. The per-rank VRAM figures (4520/4220 MiB) do check out.
- §4 omission: dense layer units count toward layer_unit_max (planner.cpp:259). That value sizes file_stage_reserve and therefore expert_floor (engine_bringup.cpp:1818, 1859). FP8 halves the engram layer's unit (~730 MB at bf16). There is no effect under --deterministic (vram_budget.cpp:170-175), but plans with lending will change.
- Host RAM at load: replicated dense weights now go through relayout staging. Each worker keeps b.out up to its largest output (engine_bringup.cpp:2353-2360, 2551-2558), up to 8 workers x 157 MB for the replicated engram. The box showed 9 GiB available, and ab_window.sh:145 stops a server below 3 GiB. Measure the load-time peak.

## Exactness holes

E1. The only realistic way an output bit changes is the split-K mismatch from a tune row or pin (B2).

E2. The peel uses a WMMA with an inline-zero accumulator, which the bf16 kernel never does.
- f8_m64.s has 20 instructions of the form `v_wmma_f32_16x16x16_bf16 V,V,V,0`, all in the SALIGN=false (Lb0) instantiations. This is the peel's first WMMA, where acc is a known zero. base_m64.s has none (0 of 180).
- C is +0.0 either way, so the bits are very likely identical, but no test covers this encoding with signed-zero sums.
- ds41 never reaches it. I ran pick_sk for every shape at TP1-TP4 (review/sk_check.py): every Ks is a multiple of 128 and every ksteps a multiple of 16, so neither the peel nor the tail loop executes.
- Fix: pass acc through an opaque asm before the peel, or GPU-test a peel case whose sums cancel to zero.

E3. The one-hot GPU test cannot see the sign of zero in B.
- acc starts at +0 and −0 products disappear in the sum, so a wrong −0/+0 in B would pass.
- Add a probe kernel that stores f8frag's 4 VGPRs for all 254 codes x 237 scales and compares them with r4d_bf16f8_expand. That checks v_cvt_pk_f32_fp8 directly on subnormals and 0x80 under clang 21.1.8 / HIP 7.1.

E4. M>64 depends on the tiled kernel matching m64. I read it and it does:
- the same split slices, each started from 0;
- tot+=acc in split order from 0.f (r4d_gemm_bf16_nt_tiled.hip:136-147);
- the same cfg_for call (:219-222).
The A/B never reaches M>64, since M≤16 (ab_window.sh:138).

CHECKED AND SOUND:
- **Split count.** The FP8 entry's cfg_for call is identical to r4d_gemm_bf16_nt_m64.hip:199. NOTUNE leaves r4d_cfg_gemm on the tGemmSkinny defaults (r4d_rows.cpp:5180-5186, 10194-10200), which equal the axis defaults the bf16 rows get appended (select.cpp:324-330). MB and WV cannot change bits under BR_FLAT (r4d_gemm_args.h:129-137). m16 forwards to m64 (r4d_gemm_bf16_nt_m16.hip:14-16).
- **Fragment layout.** The stored byte order matches fragU (m64.hip:41-46) and r4d_gdn_wmma.h:2-6. Perm 0x07060302 is the same pairing as f2bf2 (r4d_gdn_wmma.h:20-24).
- **Codegen.** Multiplies are only v_mul_f32 and v_dual_mul_f32: no legacy mul, no FMA, no ldexp. denorm_mode_32 is 3. Products are normal for s in [10,246].
- **Commit 1 refactor.** base_m64.s and refac5_m64.s differ only in the __hip_cuid symbol.
- **CPU reference.** I re-ran bf16f8_ref: 60198 patterns accepted, 0 disagreements, 0 emulation mismatches at every legal SK.
- **Other readers.** Only linear() (deepseek41.cpp:1113, 1120, 1210, 1212, 1217, 1311, 1386, 1391), LinearFP8::step (rad_fp8.h:1571-1573) and prefetch (RAD_LAYOUT_ANY) read these weights. r4d_fuse.cpp has no gemm_nt step. Only LoRA looks at op names (core/lora/lora.cpp:196; rad_builder.cpp:748). DSpark reads its stages' own wkv (deepseek41.cpp:1762), which stays bf16.
- **Block vs shard alignment.** Every span is a multiple of 32: q_b 8192, wo_b 2048, wo_a per-group entries of 1024 rows, shared expert in multiples of 256 (deepseek41.cpp:752-760), engram split 6400. Each 16x16 kernel step lies inside one 32x32 block. The relayout verifies every element, so misalignment can only refuse a weight, never change a value.
- **Runtime view.** The kernel receives the stored dtype and the declared shape (ctx.cpp:303-306).

## Simpler alternatives

S1. Store the weights FP8 in the container instead of deriving it at load.
- model.rad already keeps routed experts in their checkpoint encoding (`fp4_e2m1*e8m0[1x32] checkpoint`), next to the `cast dtype=bf16` dense entries.
- Re-converting the 570 entries as `fp8_e4m3*e8m0[32x32]` (the census shows rad-convert already reads them that way) removes:
  - the exponent derivation and the per-element verify at load;
  - the EXCLUDE list and the strict-load failure;
  - half of the dense bytes read at load.
- The new row could then serve plain gemm_nt. resolve_bands already filters out the hookless bf16 rows for a 2-plane weight (rad_builder.cpp:524-544), so no marker or new op is needed and LoRA keeps working.
- Costs:
  - a re-conversion;
  - lw() and LinearFP8 must accept that encoding (rad_fp8.h:1170-1178 refuses it today);
  - the A/B becomes old container vs new container.

S2. Leave the shared expert out of v1, which keeps arch/common untouched.
- Per 16be66c5, a layer's card work ends with the shared expert and the card then idles about 0.3 ms waiting for the CPU rows. A faster shared expert saves no time; it only frees VRAM: 450 MiB on rank 0 and 300 MiB on ranks 1-3, about 17-25 resident experts.
- Attention, indexer and engram through linear() keep 80-86% of the VRAM gain, with changes only in deepseek41.cpp and libr4d.

S3. Skip commit 1's refactor of the production m64 kernel. Copy the ~40 lines into the FP8 file instead: the ISA proof shows the refactor is safe, but it is not needed.

S4. Test the real kernel source on the CPU with tests/r4d_emu.
- The emulator already covers WMMA bf16, v_perm and readfirstlane (tests/r4d_emu/emu/hip/hip_runtime.h:82-139), and it already runs address_space(1) kernels such as r4d_moe_mxfp4.hip.
- It can compare the FP8 kernel against the bf16 m64 kernel bit for bit at every ds41 shape x SK x M x MB, in ctest, with no card. That is stronger than the hand-ported address emulation.
- To make it compile there, call r4d_fp8::e4m3_pk<HI> (r4d_fp8_wmma.h:116-127, which has a software fallback) instead of the raw builtin.

S5. Let Pa prefetch wo_b as well.
- With FP8, all five attention weights take about 36.75 MiB per rank, against 73.5 MiB at bf16, so they fit the on-die cache.
- The flip side: with Pa on, q_a, kv, q_b and wo_a already come from that cache, so the "~5% from bytes" estimate probably shrinks. Measure FP8 with and without Pa, as the plan intends.

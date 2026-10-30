Commits 4 through 8 and design commit 11 are in, and every CPU check I ran passed except two r4d_emu tests that already crash at the branch base. The device check is built but has not run (no GPU), and nothing with the knob on has run anywhere yet.

## Branch and commits
`claude/ds41-fp8-dense` in `/home/as/ds41-claude-build/wt-fp8`, HEAD `5ee48e07`, tree clean, not pushed. Author is Alexander Selzer <aselzer3@gmail.com> and every commit ends with the Co-Authored-By line.

- `32c2b291` design commit 1: shared m64 macros. Reviewed; kept as is.
- `efeb0ca3` design commit 2: bf16f8 stored form, hooks, CPU tests. Reviewed; kept as is.
- `1f749391` design commit 3: the scan tool. Reviewed; kept as is.
- `0e7eab0a` design commit 4: kernel, op `gemm_nt_bf16f8`, row `gemm_bf16f8_nt_m64` (NOTUNE, chained), OPS.md entry, selftest case, r4d_emu test. Built from the earlier attempt's uncommitted work, with fixes:
  - `fragU` moved into the shared header;
  - the B-register expansion moved to `r4d_bf16f8_frag.h` so the device check runs the same code;
  - the E2 and B5 fixes described below.
- `ba7dac1c` design commit 5: prefetch never reads past a bf16f8 weight's stored size.
- `e85aebef` design commit 6: `ds41_bf16f8_device_check` (written and built, not run), Makefile target, README.
- `f1c75102` design commit 7: optional bf16-op override in LinearFP8 and the MoE shared arm. Default is null, so GLM and Qwen declare exactly what they did.
- `f959ec41` design commit 11: an op that resolves nowhere owns no weight, plus a build_test case.
- `c1d189ee` design commit 8: deepseek41 knob `RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1`, the `_EXCLUDE` list (names or `*suffix`), prefetch byte counts, a per-rank startup line, README.
- `5ee48e07` ab_window.sh: `F8` and `F8n` arm parts with the checks from B1 and B2.

Knob off, nothing changes: the marking helper behaves exactly like the old one, prefetch sizes stay N*K*2, and the shared-arm override stays null. `ENGRAM_SPLIT` and `PREFETCH_ATTN` work with and without the knob (the split engram slices fall on 6400-row cuts).

## Scan result
The earlier run did cover the whole container: 570 of 570 entries re-encode exactly. That is 6,676,480 blocks with 0 refused and 0 inverse mismatches; the census shows 610 E4M3 pieces and all 57 tensor-parallel cuts fall on 32. The exclude list is empty. Output is in `/home/as/ds41-claude-build/aql-fp8/tools/dense_fp8_scan.{log,json}`. The hook code has not changed since that run, so I did not repeat it.

## Test results
- **Pass:**
  - bf16f8_layout_test, relayout_parity_test, build_test (including the new case);
  - place_test (no failures at all), lora_test, gen_lora_test, arch_test, glm5next_test;
  - mem_test, runtime_test, sample_test, sched_test, advance_test, p4b_kv_ring_test, host_experts_test, plugin_test (runtime_test and host_experts_test needed libavx built first);
  - r4d_emu_bf16f8 plus 14 other r4d_emu tests.
- **ops_doc_check, run by hand:** 282 rows, 0 disagree. The ctest entry skips because spechub.so isn't built.
- **New emulator test:** 223 launch pairs, and the A, B and C bits of all 3.49M lane-WMMAs match the bf16 kernel's. To prove it catches a lost sign of zero, I ran a mutant that decodes −0 as +0: 222 operand mismatches, 0 output differences.
- **build_test case:** fails without design commit 11, passes with it.
- **Kernel code:** the bf16 m64 kernel's gfx1201 assembly is byte-identical to 16be66c5's. The new kernel uses 102–121 VGPRs, no spills, no inline-zero WMMA.
- **Pre-existing failures:** r4d_emu_fp8f32 and r4d_emu_moemx segfault. The base `1f749391`, built the same way, crashes the same way. The emulator does a 16-byte aligned store into a padded `__shared__` row (P=520). Not caused by this branch.

## Built files for the run home
- `/home/as/ds41-claude-build/aql-fp8/radiance_home/kernels/libr4d.so` (md5 9a22f41d…)
- `/home/as/ds41-claude-build/aql-fp8/radiance_home/architectures/deepseek41.so` (md5 2b86f304…)
- Optional: `/home/as/ds41-claude-build/aql-fp8/bin/radiance`, which includes design commit 11.

`abi/`, `arch/` and `libr4d/` are unchanged on the main branch since 16be66c5, so these plugins also load in the main build's radiance. Base and F8 arms must both use these two files.

## Device check (commit 6)
Run inside a GPU window under both locks, never from a build shell:

`HIP_VISIBLE_DEVICES=0 /home/as/ds41-claude-build/aql-fp8/tools/ds41_bf16f8_device_check /home/as/ds41-claude-build/aql-fp8/radiance_home/kernels/libr4d.so all /home/as/radiance-setup/research/deepseek-v41-1017-20261024/native-models/ds41-native-mxfp4-v3-20261027T121600Z/model.rad`

On success it prints these lines, then a timing table and exits 0:
- `PASS frag: … all 60198 (code, exponent) pairs is r4d_bf16f8_expand's, bit for bit, every product exact (low half zero); 237 -0, 237 +0, 3318 subnormal codes …`
- `PASS onehot: 1310720 decoded elements … at splits 1 2 4 8 16 …`
- `PASS zeros: 625 launch pairs …`
- per-shape lines, then `PASS shapes: every output byte equal`
- `PASS real: 15 container slices, every output byte equal`
- `PASS: gemm_bf16f8_nt_m64 is byte-identical to gemm_bf16_nt_m64 in N launch pairs (T s)`

Any difference prints a FAIL line and exits 1.

## Expected VRAM change per rank (these weights only)

| Rank | Default | With `Es` (engram split) | With `F8n` (shared expert bf16) |
|---|---|---|---|
| 0 | 4520.0 → 2262.2 MiB (−2.205 GiB) | −1.985 GiB | −1.766 GiB |
| 1–3 | 4220.0 → 2112.1 MiB (−2.059 GiB) | −1.839 GiB | −1.766 GiB |

The design estimates about 117–126 more resident experts per rank. The startup log should show, for example, `rank 0: 330 dense weights read by gemm_nt_bf16f8 …: 2260.0 MiB of E4M3 codes where bf16 held 4520.0 MiB; 0 left on gemm_nt`.

## Critique items
- **B1:** `F8` sets the knob; `F8n` sets it and leaves the shared expert bf16. A window-level `F8_EXCLUDE` is passed through. An F8 arm isn't measured unless its server log has the "dense weights read by gemm_nt_bf16f8" line. Non-F8 arms run with the knob at 0 and fail if that line appears. An arm with an unknown part is now skipped instead of run without it.
- **B2:** an F8 arm is refused when `RADIANCE_TUNE` is set or any of its homes has a `tune/` file. It is also not measured if its server log shows a tune cache was loaded.
- **B3:** for weights read from codes, the prefetch asks for N*K bytes, and libr4d also clamps to the stored size. It prefetches the codes only, not the per-block exponents (about 0.1%).
- **B4:** done as design commit 11 (`f959ec41`). The startup check still refuses any op with an unresolved band, so the hazard was only reachable under `--debug-graph`. The change also stops such an op's resolved bands from claiming the weight's stored form.
- **B5:** the scale offset and pitch come from the weight's own shape. An issue whose input or output width differs from the weight's [N, K] is refused with RAD_E_SHAPE; the emulator test covers this.
- **E2:** the peel's accumulators are pinned with an empty asm, so the compiled kernel has 0 inline-zero WMMAs (it had 20). The emulator's peel cases and the device check's `zeros` part cover it.
- **E3:** the device check's `frag` probe dumps the B registers on the card, and the emulator compares every WMMA operand; the mutant run above shows the check works.
- **S2:** the shared expert stays in by default, since commit 7 is needed anyway and the exactness argument is the same. `F8n` or `EXCLUDE='*_shexp.weight'` leaves it out.
- **S4:** done; the emulator runs both kernels' real sources.

## Unfinished or uncertain
- **The A/B harness:** `launch_window.sh` copies the main repo's `ab_window.sh`, which has no `F8` part. An F8 arm launched today would silently run without the knob. That script has since gained `K`, `Sc` and `Iy` parts my branch's copy lacks, and my copy would skip arms that use them. My changes need merging into the main script before any F8 window.
- **Nothing has run with the knob on:** not the declare, not the relayout at load, not the startup line. Failures there would be loud: a declare error, a load refusal naming the weight, or the harness check.
- **Exactness preconditions:** the bf16 GEMMs must also use the default split, so the production home must have no tune rows for `gemm_bf16_nt_*` at these shapes and no pin; the HIP toolchain must stay at clang 21.1.8 / HIP 7.1.
- **Not measured:** decode speed, load-time CPU for the relayout, and the host-RAM peak while loading (the replicated engram is staged at about 157 MB per worker).
- **Not done, on purpose:** the design's extra sentence in `rad_abi.h`, to keep `abi/` byte-identical to the main branch; design commits 9 and 10, so prefill above 64 rows uses the slower skinny kernel (the A/B uses 16); and critique item S5 (prefetching wo_b).
- **Window detection:** `pgrep -f 'ab_[w]indow.sh'` also matches any shell whose command text mentions the script. One other agent's shell was waiting on itself this way. I used a stricter test that also respects `WINDOW_PENDING`: `/home/as/ds41-claude-build/aql-fp8/scratch/heavy_ok.sh`.

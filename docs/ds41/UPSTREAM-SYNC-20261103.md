# Upstream sync 2026-11-03: radiance 1.16.0 (769b4c64) into the ds41 engine

Branch `claude/main-v116` (worktree `/home/as/ds41-claude-build/wt-main-v116`, build
`/home/as/ds41-claude-build/aql-main-v116`) = `claude/ds41-decode-impl` 1c0a01b3 merged with upstream/main 769b4c64
(10 commits past b6acbd69: releases 1.15.1 and 1.16.0). PR 266 stays at bd65d4e5. Method as in
UPSTREAM-SYNC-20261031.md: upstream's text stays, ours follows it. Head 88de1129 = merge 3a019493 + RAMCACHE.md.
No fix commits were needed. Proposed as GitHub `main`; nothing pushed yet (PUSH-PLAN-20261103.md).

Divergence from upstream is unchanged by the merge: 359 files, +106,549 / -1,324 against upstream both before
(1c0a01b3 vs b6acbd69) and after (88de1129 vs 769b4c64, RAMCACHE.md aside).

## 1. Conflicts and resolutions (4 files, 6 hunks)

| File | Resolution |
|---|---|
| abi/rad_runtime.h | RadBatch: upstream's ABI 33 `sgen_b` first, then our private tail (consumer_full/rsv/end, lcp_admit_row/rsv, consumer_row0) unchanged. |
| abi/rad_abi_layout_rows.h | Upstream's `sgen_b` row, then our 6 RadBatch rows. |
| core/runtime/ctx.cpp | Upstream's comment (names sgen_b); `sizeof(RadBatch) == 448` = upstream 416 + our 32. |
| libr4d/r4d_qsa_tail_store_bf16.hip | PR 266's f32 form (templated store, `wide`) plus upstream's new `slots` bound, passed to both launches. Its diff against upstream is exactly PR 266's change. |

Auto-merged, checked by hand, no fix needed:
- RAD_ABI_VERSION is now upstream's 33. Our layout fingerprint still differs, so every plugin must come from
  this build.
- MAX_OPERANDS moved to core/rad_core.h.
- Stream ids are now generation × 1024 + slot. Our live-stamp and live-chain code uses the id only as an opaque
  key; nothing indexes `g_queue_by_id` by id.
- core/format, core/plugin/loader.cpp, core/engine*, the libr4d norms, and point_fourier in libref/libavx.

## 2. Upstream changes that touch ds41 numerics or kernels

| Change | ds41 effect |
|---|---|
| 55da38af, the norms' lane sum read eight at a time (`r4d_lanes_sumsq` in r4d_common.h; used by rmsnorm_body and add_rmsnorm_body in r4d_model_bf16.hip, and by lanes_sumsq in r4d_fused_quant_fp8.hip and r4d_rmsnorm_quant_i8.hip) | **On ds41's path.** It reaches the standalone `rmsnorm` at widths 5120, 1280 and 512 (every decode step and short chunk) and DSpark's `rmsnorm_quant_fp8`. Partition (256 lanes) and add order are unchanged; only the loads are batched. libr4d is built with `-ffp-contract=off`, so upstream states the bits are identical (shown on Qwen and TranslateGemma replies). **Not yet shown on ds41.** The ds41 fused norms (r4d_ds41_elem.hip) are untouched. |
| b264e9bd, head-256 prefill softmax rescaled per row (r4d_attn_prefill_h256_gqa6.hip and the h256 wrappers that include it) | Not on ds41's trunk, which attends with `attn_sparse_sink`. DSpark's drafter uses `attn_paged`. If its prefill ever runs a head-256 kernel, its rows round differently. That can move acceptance, but not reply bytes, because verification is exact. |
| qsa_tail_store bounds the slot index by the state plane | ds41 issues it (f32, the compressed-KV tail). Only rows with a slot past the plane change: before, they wrote out of bounds; libref already refused them. Valid runs keep their bytes. |
| Paged attention refuses kv_heads <= 0 (r4d_attn_args.h) | A shape check only. |
| GEMM | No libr4d GEMM change in this range. libr3d (gfx11) was rewritten, but it is not built for gfx1201. |
| New ops: point_fourier (libr4d, libref, libavx) and the Gemma 4 i8g junctions (rmsnorm_quant_i8g, gelu_mul_quant_i8g, ar_rmsnorm_quant_i8g) | Not used by ds41. libr4d.so is 94.1 MB. |
| ABI 33: RadBatch::sgen_b, RadShapeGenBatch/Decl, RadArchProbe::gen_shape, regenerated fingerprint | Our private RadBatch tail moves 8 bytes. aql-sync/home-sync plugins (ABI 32) are refused by this build, and vice versa. |

Other upstream behaviour that can show in a ds41 window:
- **Loader.** Two plugins declaring one op with different parameter roles are refused. Null tables, unknown roles,
  and more than 32 operands are also refused. Checked on the CPU: `rad-info --plugins` loads this build's libr4d
  (587 kernels, 291 schemas), libavx, libref, spechub, libquant and the 36 arch plugins with no refusal.
- **Container.** Tables must be element-aligned, each vocab count needs its array, and expert 1 must start at
  least one unit past expert 0. Checked on the CPU: `rad-info` opens
  ds41-native-mxfp4-v3-20261027T121600Z/model.rad (RadFile::open's validate + index_experts), rc 0, no warning.
- **Device ledger.** It now counts live bytes (rad_dev_free takes them back). The pin-cap room for the checkpoint
  store can only grow, so the `checkpoint slots:` startup line may rise. Also:
  - stream slots are recycled;
  - a block over 1024 work-items is refused;
  - an undrained stream keeps its queue (with a warning).
- **Asserts now refusals.** Two NDEBUG asserts became refusals: the KV runtime-window byte totals and the ring
  blocks per group.
- **Tune cache.** One measured on another machine is ignored. home-sync has none.
- **Server.** /v1/messages takes the sampling penalties. The harness uses /v1/completions and /v1/chat/completions.

## 3. Build and CPU tests (no card opened; ROCR_VISIBLE_DEVICES=99, HIP_VISIBLE_DEVICES=-1)

Configured as work/sync/build.sh (g++, clang-21 HIP, gfx1201, Release, FFMPEG off), with this branch's own paths,
RAD_DEFAULT_HOME pointing at an unused path, and no ccache (the disk was at 11-13 GB free). Scripts and logs are in
`/home/as/ds41-claude-build/work/main-v116/`.

Built with no errors (warnings are the existing ones in our files):
- radiance, libref, libavx, libquant, spechub, deepseek41;
- libr4d (94.1 MB) and r4d_selftest;
- all 36 arch plugins, the 20 test plugins, rad-info, rad-schemas, rad-kbench, rad-avx-check;
- 136 of the 143 `*_test` binaries, plus ds41_lcp_contract. Of the other 7, the two native mover tests are in the
  HIP-off build, and engine_e2e, clef_decision, laya_decision, tokenizer_perf and fixture_test were not built.

Host tests (`ctest -LE gpu`, exact names, under the build lock), 158 entries:
- **Chunk 1: 38/38 pass.** These are ABI, runtime, plugin, build, config, device, mem, kv, sched, place, format,
  checkpoint, server, chat, ref, arch, the new shape tests, kld, quant and others. The first run had 4 failures
  (build_test, mem_test, p4b_kv_ring_test, plugin_test), all "missing in-tree artifact" or "cannot open"
  radtest_*.so. They pass once the test plugins are built.
- **Chunk 2: 55/55 pass**, r4d_bf16_capacity_test skipping itself (it needs a card). These are every ds41_* test,
  p1_stager, p1_chunk, tbo_stager, the host-expert, readahead, route-trace and router-predictor tests, the dspark and
  draft bank tests, the spechub tests, the layout and relayout tests, and the host_expert_pool wake-off variant.
- **Chunk 3: 60/62 pass**, nemotron35_device_test skipping itself. These are the remaining unit tests, plus:
  - ds41_lcp_contract_{0,2048,3072,4096,4097,4112};
  - avx_check and avx_check_general (libavx against libref);
  - ops_doc (docs/OPS.md against the loaded schemas);
  - kbench_ref, vendor_sums, rad_export_test, kernel_report_check, isa_tools.

  The 2 failures are images_test and videos_test: "this build has no media decoder / no libavcodec"
  (RAD_WITH_FFMPEG=OFF). Pre-existing by construction, not a merge effect. tests/images_test.cpp, tests/videos_test.cpp and core/mm/ (the FFMPEG-off stubs whose messages the tests print) are identical in 1c0a01b3 and 88de1129. The cases fail at the encode step after generation ran, and the merge's only nearby changes (core/gen/video_driver.cpp, core/server/videos.cpp) are upstream refusals of zero-frame or oversized inputs that these cases do not hit. A run of the two tests built from 1c0a01b3 (worktree wt-main-v116-base) was queued, but a GPU window (winCF, 11:02) held the box, so I cancelled it rather than leave it running unattended. work/main-v116/baseline_iv.sh runs it.
- **HIP-off build** (`aql-main-v116-host`): native_file_cache_test, native_file_cache_mover_test and
  native_mapped_copy_mover_test pass. These are the tests that refuse a HIP binary; the last one covers the side
  store.

Not run, by instruction or design:
- the r4d_emu and halo_emu emulator tests;
- engine_e2e_test and the clef/laya decision tests (they start a radiance server on loopback);
- the ab_window_* shell tests (they run the window harness script, unchanged since 1c0a01b3);
- the gpu-labelled entries, and the perf entries.

## 4. What the GPU verification window must check

Use only this build's artifacts. The ABI 33 fingerprint refuses any mix with aql-sync or home-sync plugins.
- `AB_BIN=/home/as/ds41-claude-build/aql-main-v116/bin/radiance`.
- Home `/home/as/ds41-claude-build/aql-main-v116/radiance_home`, or a staged copy of its kernels/, architectures/
  and quantizers/.
- Tools from `wt-main-v116/tools/ds41-host-experts`, i.e. 1c0a01b3's harness, not wt-sync's research harness.
- The device-check binaries (ds41_mxfp4/vpack/bf16f8_device_check) still have to be built from this tree.

1. **Card checks first** (libr4d changed: the norm lane sum, qsa_tail_store, the new ops):
   1. `aql-main-v116/bin/r4d_selftest` on card 0 against this libr4d.so. It now includes upstream's
      attn_prefill_rows and i8g checks, plus our bf16f8 identity.
   2. `DEVICE_CHECKS="tall lcp p1 bal"`, as in chain24, with this libr4d. The tall sweep covers the standalone
      rmsnorm at ds41's widths at every height.
   3. The default swiglu/chain check (the CPU path's bit-exactness reference).
   4. VPACK_CHECK and SELFTEST_F32BF16 for any Rb/Vp arm.
2. **Base-engine arm** (host experts off, 16-token chunks) on the held-out set. Compare it with the 1.15-merge base
   (`tools/ds41_reply_check.py evidence/ds41/ab-20261031T104524Z/base.json <new>.json`). The expected result is
   byte-identical: 1.16.0 intends no ds41-path numerics change.
   - If identical, every existing reference stays valid.
   - If not, suspect 55da38af first: restore the one-at-a-time loop in r4d_lanes_sumsq's three users and rerun the
     arm.
3. **Recommended configuration** (Lc2048 + P1 + lent arena, the README block) at 1 stream on held-out. Check
   reply identity against the new base arm, and speed against 29.45 tok/s and the 3,086 / 6,817-token TTFT
   (7.2 / 17.2 s).
4. **One DSpark arm** on code (`Ds1` with DRAFT_BANK). Replies must match base. Acceptance should match the last
   window: this path reaches rmsnorm_quant_fp8 and attn_paged.
5. **One concurrency pair** (`-L1024-Pf-N4-C4-Pmr4`, `-N8-C8-Pmr8`), because stream slots and the device ledger
   changed. Compare with 37.2 / 41.5 tok/s.
6. **First arm's server log.** Look for:
   - no loader refusal ("has role", "operands than an issue carries", "missing table");
   - `checkpoint slots:` (may rise; ledger);
   - the `mover:` pools;
   - no "a stream was destroyed with work it could not drain";
   - no KV "runtime window" or "ring sizes" errors;
   - the `file stager:` and `host experts:` lines as before, and no `routing-aware reads skipped`.

About 6-8 arms in all.

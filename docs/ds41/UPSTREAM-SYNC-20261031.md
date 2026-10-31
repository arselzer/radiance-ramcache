# Upstream sync 2026-10-31: radiance main b6acbd69 into the ds41 engine

Branch `claude/ds41-sync` (worktree `/home/as/ds41-claude-build/wt-sync`, build `/home/as/ds41-claude-build/aql-sync`,
home `/home/as/ds41-claude-build/home-sync`) = `claude/ds41-decode-impl` 9b395fc8 merged with upstream/main b6acbd69
(60 commits since our base 4374bea1, releases 1.14.0 and 1.15.0; 680 files). The deepseek41 PR stays at bd65d4e5
(upstream/pr-266 not merged; looked at only to name things the way its author already resolved them).

Status, head, build and test results: section 6. Window plan: section 7.

## 1. Top findings

1. **ds41's numerics change: re-measure the base engine.** Upstream made the standalone RMS norm fold every row's
   sum of squares over a fixed 256-lane partition (`kR4dSumLanes`, libr4d/r4d_common.h; rmsnorm_body and
   add_rmsnorm_body in r4d_model_bf16.hip, in 1.15.0) so a row no longer depends on how many rows its launch carries.
   Before, the partition was the workgroup width, which `r4d_act_threads` makes 1024 up to 32 rows and 512 up to
   72/120. ds41 declares the standalone `rmsnorm` at widths 5120, 1280 and 512 (above 256), so **every decode step
   and every chunk of up to ~72 rows now rounds those norms differently**; prefill launches at 256 threads keep their
   bytes. The ds41 fused norms (mhc_collapse_rmsnorm, mhc_post_collapse_rmsnorm, rmsnorm_rope_roundtrip in
   r4d_ds41_elem.hip) still fold over the width, as before, so they keep their bytes. The base engine's replies can
   therefore differ from every reply set recorded so far. Our configurations stay consistent with the merged base
   (each op is still height-independent where it was: the standalone norm now always, the fused ones under
   EXACT_TALL), so the gate is unchanged in kind: byte-identical to the merged base, which needs new base arms.
   Kept upstream's kernel (a correctness fix that overlaps EXACT_TALL); the alternative, reverting rmsnorm_body and
   add_rmsnorm_body to the width partition (about 20 lines in an upstream kernel), would keep the old references.
2. **Op-name collision fixed: ds41's `rope_rows` is now `ds41_rope_rows`.** Upstream added `rope_rows` (and
   `rmsnorm_rows`) as row forms of `rope`/`rmsnorm`, with other parameters, and an `r4d_rad_rope_rows` symbol. With
   both, the loader refuses libr4d/libref/libavx ("op declared twice with different schemas") and libr4d does not
   link. Renamed the ds41 op, its libr4d entry `r4d_rad_ds41_rope_rows` and libref's kernel `ref_ds41_rope_rows`
   exactly as the PR266 author did in upstream/pr-266 (2dbe1b4b), so the line matches when the PR lands; also
   renamed the two fusion steps in r4d_fuse.cpp (the PR head still says `rope_rows` there, which no longer names the
   ds41 op). Kernel bytes unchanged; libr4d, libref, libavx and deepseek41.so must be the ones from this build.
3. **ABI 32 is now upstream's ABI 32, plus our private RadBatch tail.** Upstream released its own ABI 32
   (RadBuildCtx draft_tree, calib_*, gen_max_*; RadBatch score_only, plugin_words; RadDeviceProps::unified;
   RadChatFormat arg_json_value_prefix) and a layout fingerprint every plugin exports and the loader checks. Our
   private 32 had one extra RadBuildCtx field, `draft_head_off`. Upstream merged PR266's prompt lookup (33108e48) but
   replaced draft_head_off with a startup refusal of `--spec-ngram` beside a declared head, so I dropped it:
   RadBuildCtx is upstream's byte for byte, the 6 upstream arch plugins and the gemma4 dflash header that called
   `rad_ctx_draft_head` are back to upstream's text, and the engine keeps upstream's refusal. **Behaviour change (flagged):** prompt lookup alone on
   ds41 needs `RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1` now (the plugin then leaves its DSpark drafter out); the
   harness's `Sg`/`Sga` arms set it. Without it `--spec-ngram 3` is refused by the plugin (block 5 != 3), as
   upstream would. The private RadBatch fields (consumer skip, LCP admit, per-row cut) stay, appended after
   upstream's; RAD_ABI_VERSION stays 32, so the fingerprint differs from upstream's and upstream-built plugins are
   refused (wanted: every plugin must come from our tree).
4. **Upstream now has its own file-tier work that overlaps ours; kept ours, left theirs dormant.** Routing-aware
   file reads (9efdd00c, on by default upstream), a hot VRAM cache, pool-slot deposits and copyless relabels
   (8da043c2) duplicate our routed-only reads, RAM cache and packed residents. Resolution: FileStager::init takes
   both sources (upstream's RouteCounts, our CountsRow/EpCountsRow); the engine binds ours and passes no RouteCounts,
   so a pass our routed reads decline still reads the whole layer, as measured. The hot cache and deposits only run
   on a dynamic plan (we always run `--deterministic`) and are now refused for a unit whose weights are relaid or
   served from the RAM cache (they would have put raw container bytes in the slot). Mover stats follow upstream's
   rename file_* -> ssd_* (our extra counters appended to its new atomic MoverCounters). Two semantic conflicts
   outside the conflict hunks, both in Mover: upstream's new early return in stage_file_units when nothing goes
   through the read windows would have skipped every relaid or RAM-cached unit, i.e. ds41's whole file tier
   (caught by place_test; now it returns only when nothing is relaid either); and upstream's 8 MiB zeroed
   "dummy region" is reserved from the pinned host pool, which the planner's staging reserve does not grow for
   it, so with our small pools (`p256` parts: 572 MiB with the embedding) the reservation could fail at startup:
   it now falls back to an unbudgeted pinned allocation.
5. **Kept the routing counts in the core's row (flagged).** Upstream switched MoeFP8 to report the block's own
   `ecnt` buffer instead of writing the counts into the core's row (a raw-pointer operand does not survive tape
   replay). ds41's routed reads find a layer's counts by that row (Ctx::route_counts_live and the EP witness
   Ctx::note_ep_route_producer), so taking upstream's version would turn every routed-only read into a whole-layer
   read. Reverted those two hunks in arch/common/rad_block_moe_fp8.h; our passes are armed by the file stager and
   issued live, never replayed.
6. **Ideas worth taking, by value** (details in section 2):
   1. DSML reply format for ds41 (arch/common/rad_deepseek41_chat.h; the PR head already exports it): proper
      reasoning and tool-call parsing for agent harnesses. Small effort (two lines in deepseek41.cpp); changes the
      chat API's response fields, so gate it with the reply checker on the chat set first.
   2. Fused ds41 norms on the 256-lane partition (as upstream did for its own fused norms): makes them
      height-independent like the standalone norm, so EXACT_TALL no longer needs 1024-thread norms in tall passes
      and fused == unfused at every height. Medium effort (three kernels + the device-check sweep), a numerics change
      of its own (one more base re-measure), small prefill gain.
   3. `rmsnorm_rows` for ds41's 128-wide norms in prefill (the short-row form, byte-identical): a little prefill
      time; small effort, card check needed.
   4. `--kld-top N` (sparse KL reference) for the Kld gates: much smaller reference files and faster recording.
   5. The preempted-prefill gate and `--preempt-wait-ms` (merged as is): matter for concurrent decode once the KV
      pool is under pressure (N=8 with long contexts); nothing to do but watch `preempt` counters in Mb arms.
   Not worth it now: binding upstream's routing-aware reads for passes ours declines (with a 192 GiB cache prefill
   misses are rare), the hot VRAM cache (static packed residents already fill VRAM by profile), draft trees
   (DSpark is a chain), libhalo/libr3d (other cards).

## 2. What upstream changed since 4374bea1, by area

Classification: **(a)** conflicts with or duplicates our work, **(b)** idea worth taking, **(c)** irrelevant to
ds41 on gfx1201. Commit ids are upstream's; 1.14.0 (4a61d729) and 1.15.0 (9ed3ef7a) are squashed release commits
that carry most of the code (1.15.0 is a merge whose own diff brings libhalo, IQR, the codebooks and recipes).

### 2.1 Core scheduler and batching

| Change | Class | Notes |
|---|---|---|
| Preempted-prefill gate: wait for room for the whole recompute plus running headroom (e668d493, 187a015b), a bound on the wait and `--preempt-wait-ms` (33108e48), skip KV groups without blocks (8df9240d), reclaim cache and leaked state slots first (9560780b) | (b), merged | Only acts when a request is preempted (KV pool full). Our concurrent arms (Pf, Pmr, Mb at N=2-8) rarely preempt at 512-4K contexts; at long contexts it removes the preempt-reset loop. Auto-merged beside PREFILL_ALONE/TBO/DSpark gating; sched_test covers both. |
| Prompt lookup alone (`--spec-ngram`, verify only when it pays, adaptive window) landed in main inside 33108e48 | (a) | It is PR266's lookup (core/sched/lookup.*, ngram.* are now identical to upstream), without draft_head_off: see finding 3. |
| Scored steps (`rad_declare_scores`, RadBatch::score_only; ed1e5caf, 1c967b2f, d773992a) | (c) | /v1/systemone only; ABI field appended. |
| Prompt ids outside the vocabulary refused at admission (a28221eb) | (c), merged | Safety; the harness sends text. |
| Window snapshots inside a chunk for a drafter's window; `reserve_ckpt` offset `ckpt - n_computed - 1` (1.14.0/1.15.0) | (a), merged | Same value as our `want - 1` whenever the checkpoint ends the chunk, which every exact-mode checkpoint does (interval grid, csckpt band, tail); under `--prefix-resume nearest` with window checkpoints the tail snapshot is now taken inside its chunk (the Pcn arm only). Our EORS snapshot kept after it. |
| Draft trees (DFlash2 tree verification), `--draft-short-from` default auto (never for a chain) | (c) | DSpark is a chain; our README sets `--draft-short-from 1` explicitly. |

### 2.2 KV cache, prefix cache, snapshots

| Change | Class | Notes |
|---|---|---|
| Checkpoint store charged its floor (8 slots) before `--host-headroom-mib` (481fcd5d) | (a) small | Changes the slot count only where the reserve squeezed the store to zero; on the 256 GB box probably nothing. Check the `checkpoint slots` startup line of one Pc arm. Merged with our `cache_margin_hint()` in both messages. |
| A drafter's window rides in the snapshot, saved after the context pass (`drafter_snap_`, engine_draft.cpp) | (b) investigate | Would let DSpark keep drafting right after a prefix hit (harness turns with DSpark) without a refill hole. Merged into our deferred-context loop (`while` over block_context); inactive unless the KV groups declare a drafter window. |
| kvstream stage copies only the blocks its own tables name (#343); prefix/kv bookkeeping for groups with different block sizes | (c), merged | |

### 2.3 Expert placement and offload

| Change | Class | Notes |
|---|---|---|
| Routing-aware file-tier reads (9efdd00c): the stager reads only experts the router counted, publishes a dummy address for the rest | (a) | Duplicates our routed-only reads. Both kept; ours bound, theirs dormant (finding 4). Upstream's also clamps a read at the container's end, the same bug our `read_covering` fixed (0e01f955): took upstream's clamp in FileTier::read, so `DirectFile::read_covering` is now used only by tests. |
| Hot VRAM cache, deposits into free pool slots, copyless relabels, ssd_* counters (8da043c2) | (a) | Dynamic plans only (`p->dynamic`, i.e. without `--deterministic`), so inactive in every ds41 arm; guarded against relaid/RAM-cached units. The planner reserves its shadow slots only on a dynamic plan. |
| Draft-head (`mtp.`) units rank first in the expert fill | (c) | ds41's DSpark experts are preplaced (host_forced, draft_vram, or parked) in every bank mode, so the order never reaches them. |
| GLM-5 Next experts sharded by expert (9647b84e); `--placement layer_offload` spills dense layers for max-model-len | (c) | |
| `--expert-profile` refused when recorded on another model (trunk-layer check) | (c), ported | Our profile loading moved; the check is ported to our load site (native decode profiles are checked by their own identity). |
| Stall watchdog counts drive reads as progress; DMA lane's room check reads its own completion fence (6f6b5c3f) | (b), merged | Lists on the DMA lane, which carries P1's ring copies, now end by writing a finish word, and the room check trusts it instead of the written-back read pointer that can stall (upstream saw that wedge a large pool's load fill): a robustness fix for P1, no speed change expected. `/health` says `stuck` after a minute without progress (unused by the harness). |

### 2.4 libr4d (gfx1201)

| Change | Class | Notes |
|---|---|---|
| RMS norm folded over 256 lanes at every width (1.15.0, `kR4dSumLanes`) | (a) **numerics** | Finding 1. Upstream's own fused norms (rmsnorm_quant_fp8, the AR+norm kernels, qk/qkv norm-rope) moved too; ds41's did not. |
| Row forms `rmsnorm_rows`, `rope_rows` (b8cd1c95) | (a) collision, (b) | Finding 2. `rmsnorm_rows` (short-row form for n <= 256 at 1024+ rows) is idea 6.3. |
| BF16 GEMMs launched in row chunks when M*K passes 32-bit offsets (94c90b17); `r4d_gemm_bf16_nt_m64_max_m` now unbounded | (a), merged | Combined with PR266's pitched A/C exactly as the PR head did (both GEMM hosts now match upstream/pr-266 line for line, plus `m0 * lda` in the tiled loop). Same bytes: ds41 shapes fit one chunk. |
| Gated prefill fold, short-prompt split row (`gemm_nt_q_gated_wide`, `gemm_nt_q_short`), Qwen MXFP4 fold up to 16 rows | (c) | Qwen dense MXFP4; ds41's experts use their own moe_gemm_ggml_mxfp4 rows. |
| Split-KV row sum with fetch-inactive (#328), decode split law verify floor (#342, #? depth), paged prefill h256 gqa6 KLATE, prefill attention rescale by live rows | (c) | ds41 attends with its own `attn_sparse_sink`. |
| IQR GEMMs, nemotron/pixtral/embed-attn kernels, dflash tree kernels, gram_accum on matrix cores, fp8 prefill GEMM (r4d_gemm_8a8_prefill.h), fp8a8 narrow-form LDS scales | (c) | |
| A kernel's lazily made counter block zeroed outside the lock other ranks take (TP2 first-step hang) | (b), merged | |
| rowtopk: IQR head template parameter | (c), merged | Same code for ds41's row_topk. |

### 2.5 Collectives

| Change | Class | Notes |
|---|---|---|
| 2-rank host-mailbox all-reduce staged through a VRAM mirror (#323, `--p2p off`) | (c) | ds41 is 4 ranks over P2P. |
| AR + norm kernels fold over `kR4dSumLanes` | (c) | Flash-Next's fp8 path; ds41 runs a separate all_reduce and the bf16 norm. |

### 2.6 Speculative decoding

| Change | Class | Notes |
|---|---|---|
| Prompt lookup alone in main | (a) | See 2.1 and finding 3. |
| DFlash2 trees, coupled acceptance at T > 0 | (c) | |
| Drafter window in snapshots | (b) investigate | See 2.2. |

### 2.7 Quantization and conversion

| Change | Class | Notes |
|---|---|---|
| IQR V2/V3, GPTQ speedups, trellis GPTQ on the card, vector codebooks, `rad-convert --reuse` repeatable | (c) | |
| Calibration hooks `--calib-corpus/--calib-dir/--calib-layers` (RadBuildCtx calib_*) | (c) | |
| `--kld-top N` sparse KL reference | (b) | Idea 6.4. |
| GGUF Q8_0/Q4_K/Q6_K imported as planes (1.14.0) | (a) | PR266 imports GGUF blocks its own way (lq_ggml_planes, more types); kept PR266's branch in checkpoint.cpp. |
| DeepSeek-V4.1 converted from the release: Engram tables (data/engram_tables.py), fused imatrix, `rad-convert --chat-template` (1.14.0) | (b) later | Only for a container rebuild. |
| `X.shard_<i>.weight` under one `X.weight_scale` | (c), merged | Kept beside PR266's `X.scale` lookup. |

### 2.8 Server

| Change | Class | Notes |
|---|---|---|
| Hardening: bounded parses and JSON depth (b9c685a9, d54f3b9e), loopback no longer identity (ec53bce5: Host must name the loopback), bounded transport buffers (5f7244da), no internals on the wire, duplicated security headers refused (ad3f03db), proxy caller names (b6acbd69), `--api-key ""` refused | (c), check | The harness talks to 127.0.0.1 by that name (Host matches), uses /v1/completions, /v1/chat/completions, /stats, /get_tokenizer_info: none are the keyless destructive routes. The first arm of the window confirms. |
| cpp-httplib v0.46.1 -> v0.64.0, the drain patch reduced to one line | (c), check | Same as above. |
| `/health` 503 `stuck`; startup report lists only non-default options (`-v` all) | (c) | The harness parses none of the report. |
| /v1/systemone (clef, laya, mica), /v1/decisions, Nemotron embed/rerank | (c) | |

### 2.9 New libraries

libhalo (Strix Halo gfx1151, 1.15.0) and libr3d (RDNA3 gfx11, 1.14.0): (c). Not built in aql-sync.

### 2.10 deepseek41-adjacent

| Change | Class | Notes |
|---|---|---|
| `arch/common/rad_deepseek41_chat.h`: DSML reply format with typed argument tags (9cf8d3da) | (b) high | Idea 6.1. Upstream ships the format for the plugin it has not merged; the PR head exports it as `chat_format`. |
| `rope_rows` collision | (a) | Finding 2. |
| upstream/pr-266 head acc9ae58 merged 1.15.0 into the PR with "DS41 file relayout" in place/ | note | When the PR lands, its file relayout meets ours (weight_file_relaid, Mover::set_file_relayout): the next sync's largest conflict. |

## 3. The merge: conflicts and how each was resolved

`git merge upstream/main` from 9b395fc8 (merge base 4374bea1) stopped on 45 files (about 110 hunks). Rule applied:
upstream's text stays where it is, our additions go after it as separate lines, our variants of upstream lines
only where our behaviour needs them. "Both" below means upstream's block first, then ours.

| File | Resolution |
|---|---|
| abi/rad_abi.h | Upstream's comment (our "private 32 = upstream 31 + draft_head_off" note dropped). |
| abi/rad_abi_layout_rows.h (was tests/abi_layout_rows.h) | Upstream's rows, then our 6 RadBatch rows; the draft_head_off row dropped. |
| abi/rad_builder.h | Upstream's RadBuildCtx tail (draft_tree, calib_*, gen_max_*); draft_head_off and rad_ctx_draft_head() removed (finding 3). |
| abi/rad_runtime.h | RadBatch: upstream's score_only and plugin_words, then our private consumer-skip / LCP / per-row-cut fields. |
| arch/common/rad_block_mlp_fp8.h | Upstream's comment, PR266's `!gate_up.ggml` condition. |
| core/build/rad_build.h, rad_builder.cpp | Both: upstream's drafter_tree, then our consumer suffix and two-batch declares. |
| core/config.cpp | Upstream's `--weights-disk-tier` help, then our RAM-cache, mirror, row-prefetch, rewarm flags; `--debug-tape-dump` kept. |
| core/device/device.h | Both: upstream's map_file and stuck_ms, then our live_* entries (backends set them by name). |
| core/engine.cpp | Our pipelining decision (draft_experiment_) with upstream's new `!cfg_.calib()`; upstream's tree check, then our prefill hashes and post-stop. |
| core/engine.h | Both includes; copy_table keeps our owner argument under upstream's comment. |
| core/engine_bringup.cpp | copy_table: our owner file beside upstream's shared-table branch; expert profile: our load site and per-rank source mapping, upstream's trunk-layer check ported there; checkpoint slots: upstream's messages and floor-aware reserve, plus our cache-margin hint; FileStager::init: our counts sources, upstream's RouteCounts unbound (finding 4); draft_head_off and our refusal removed, upstream's refusal and lookup-only message kept. |
| core/engine_draft.cpp | This rank's own proposal handle (ours) plus upstream's tree parents; our deferred context loop (`while`) plus upstream's ADV_CTXTREE and drafter snapshot. |
| core/format/checkpoint.cpp | Both scale spellings (upstream's shard tables, PR266's `X.scale`); PR266's GGUF block import over upstream's comment-only change. |
| core/place/mover.cpp | FileTier::read: upstream's EOF clamp instead of our read_covering; stage_file_units: combined signature, our `file_off_card` test, then upstream's skip / hot-cache / deposit code guarded by `raw_unit` (not for relaid or RAM-cached units) and given our source bank in FileExt; stats renamed to ssd_*. |
| core/place/mover.h | FileTier: upstream's size_ and our epoch; MoverStats/MoverCounters: upstream's, our counters appended to both; stage_file_units: upstream's parameters, then our policy and trace. |
| core/place/planner.cpp | Our file-relayout condition plus upstream's draft-head marking; our pin-threshold comment (the code's fix of `--expert-pin-top`, unused by ds41, stays: upstream's threshold `shares[...] - 1.0f` sits below every share in [0, 1], so it pins every resident); our draft-bank placement, upstream's removal of a stale comment. |
| core/place/planner.h | Both fields; our draft-VRAM counters under upstream's comment. |
| core/place/stager.h, stager.cpp | Both APIs: init(prog, plan, mover, RouteCounts, ud, CountsRow, EpCountsRow); upstream's `start(li, s)` with our head-cache, cap, cache-policy and partial/pending state; every caller of ours now passes the stream; destructor frees both sets. |
| core/rad_core.h | Upstream's `--spec-ngram` comment; `debug_tape_dump` kept. |
| core/runtime/ctx.cpp | sizeof(RadBatch) 440 (upstream 408 + our 32). |
| core/sched/batch.h, metrics.h | Both fields; our DSpark gate/conf metrics. |
| core/sched/scheduler.cpp, scheduler.h | Ours (DSpark gate and confidence controller, solo hold, PREFILL_ALONE, compute-off, EORS, consumer checkpoints) with upstream's tree parents in set_draft, upstream's `reserve_ckpt` offset, upstream's preempt-gate declarations beside our trim_prefix_cache(spare). |
| docs/GUIDE.md, docs/OPS.md | Upstream's `--spec-ngram` text; `--debug-tape-dump` kept; OPS: upstream's new rows, then PR266's (with `ds41_rope_rows`). |
| libavx/avx_cases.cpp | PR266's ds41 cases (renamed op) then upstream's row-form copies. |
| libr4d/CMakeLists.txt | Upstream's declaration list (r4d_iqr_layout.cpp) plus PR266's ggml layout sources. |
| libr4d/r4d_gemm_bf16_nt_m64.hip, r4d_gemm_bf16_nt_tiled.hip | PR266's pitched A/C plus upstream's chunked launches, as upstream/pr-266 resolved them (the tiled host is identical to the PR head; the tiled chunk loop also offsets by lda/ldc). |
| libr4d/r4d_rows.cpp | Upstream's rad_abi_layout.h include, our exact-tall include; PR266's ds41 specs/schemas/rows first, then upstream's new ones (the PR head's order); rad_kernel_at with our tall twins after upstream's host twins. |
| libr4d/r4d_selftest.cpp | Both checks (our bf16f8 identity, upstream's band edge). |
| tests/CMakeLists.txt | Our NO_CORE list plus upstream's nemotron35_test; both appended blocks. |
| tests/abi_layout_table.h, abi_layout_test.cpp | Upstream's (see section 6 for the table). |
| tests/arch_fakes.h, config_test.cpp, engine_e2e_test.cpp, runtime_test.cpp, sched_test.cpp, r4d_emu/emu/emu.cpp, r4d_emu/emu/hip/hip_bf16.h | Both sides (hip_bf16.h: upstream's, which nothing of ours contradicts); engine_e2e's PR266 `--spec-ngram` test now runs on the head-less family and expects upstream's startup line. |
| tools/rad_convert.cpp | Both: our draft-bank options, upstream's repeatable `--reuse`. |

Semantic fixes outside the conflict hunks (in the merge commit unless marked):
- core/place/mover.cpp: stage_file_units returns early only when nothing is relaid either (upstream's new early
  return skipped ds41's whole file tier); the file-tier dummy region falls back to an unbudgeted pinned
  allocation when a small host pool has no room (follow-up commit).
- arch/common/rad_block_moe_fp8.h: routing counts written into and reported from the core's row again (finding 5).
- arch/{gemma3,gemma4_bf16,glm5next,llama_fp8,qwen35_fp8,qwen4exp_fp8}, arch/common/rad_gemma4_dflash.h: back to
  upstream's text (no rad_ctx_draft_head).
- arch/deepseek41/deepseek41.cpp: op `ds41_rope_rows`; `RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE`.
- libr4d/r4d_ds41_elem.{h,hip,inc}, r4d_fuse.cpp, libref/ref_registry.cpp, ref_shapes.cpp, libavx/avx_registry.cpp:
  the `ds41_rope_rows` rename.
- core/place/stager.cpp: the lent ring's stage_file_units call passes upstream's new arguments.
- tools/ds41-host-experts/ab_window.sh, README.md: `Sg`/`Sga` arms set LOOKUP_ALONE; the switch documented.
- Tests: the renamed Mover counters and the new stage_file_units / FileStager::init arguments in place_test and the
  two native mover tests; ref_test calls `ds41_rope_rows`; tests/CMakeLists.txt's P1 emulator block moved above
  upstream's new halo_emu section, which reuses its `dir`/`flags`/`objs` variables; the emulator's duplicate
  `emu_cvt_pk_f32_fp8` (upstream added the same helper) dropped and a `hipGetErrorString` stub added for
  upstream's chunked GEMM host code.

## 4. Divergence report: our footprint on upstream after the merge

Measured on the merge result against upstream/main b6acbd69 (`/tmp/ds41sync/footprint.py`: git diff, and git blame
on every line the diff adds to an upstream file, attributed to PR266 = bd65d4e5's own commits, the dev tree's
private commits up to 4b732f39, decode-impl = 4b732f39..9b395fc8, or the sync itself).

**Files we added** (absent upstream): 168 files, 67,112 lines: PR266 31 files / 18,719 lines (arch/deepseek41,
libr4d/r4d_ds41_*, lq_ggml_*), dev tree 20 / 4,630 (native/mapped file cache, read workers, decode profiles),
decode-impl 117 / 43,763 (tools/ds41-host-experts 15.4K, tests 16.8K + emulator 3.2K, core/place 9.1K, core/sched
1.2K, core/runtime 1.0K). These merge for free: upstream never touches them.

**Upstream files we modify**: 128 files, +26,106 / -1,313 lines. Of the added lines: PR266 6,155, dev tree 1,221,
decode-impl 18,486, sync resolutions 163 (81 more are upstream lines the diff realigns).

| Area | Files | + | - | Origin of + (P / D / C / S) |
|---|---:|---:|---:|---|
| tests | 24 | 8,328 | 49 | 2,242 / 360 / 5,668 / 55 |
| core/place | 6 | 8,266 | 119 | 335 / 460 / 7,349 / 61 |
| core/sched | 7 | 2,045 | 49 | 5 / 0 / 2,038 / 0 |
| core/engine* | 7 | 1,572 | 100 | 121 / 186 / 1,247 / 17 |
| libr4d (kernels, hosts, selftest) | 17 | 1,077 | 127 | 457 / 0 / 605 / 10 |
| libavx | 8 | 939 | 7 | 935 / 0 / 0 / 4 |
| libref | 8 | 709 | 9 | 706 / 0 / 0 / 3 |
| core/format | 5 | 545 | 89 | 422 / 101 / 18 / 0 |
| core/runtime | 5 | 535 | 11 | 211 / 73 / 250 / 1 |
| core/device | 5 | 337 | 3 | 16 / 0 / 321 / 0 |
| arch/common | 6 | 316 | 74 | 208 / 0 / 98 / 5 |
| core/config.cpp + rad_core.h | 2 | 224 | 3 | 76 / 21 / 127 / 0 |
| tools/rad_convert.cpp | 1 | 216 | 14 | 104 / 20 / 92 / 0 |
| libr4d/r4d_rows.cpp | 1 | 204 | 6 | 77 / 0 / 124 / 3 |
| core/mem | 2 | 146 | 17 | 0 / 0 / 146 / 0 |
| libquant | 4 | 136 | 626 | 136 / 0 / 0 / 0 |
| core/build | 2 | 135 | 2 | 6 / 0 / 129 / 0 |
| core/server | 5 | 123 | 0 | 0 / 0 / 123 / 0 |
| abi | 4 | 122 | 3 | 19 / 0 / 102 / 1 |
| docs, core/other, core/plugin, data | 9 | 131 | 5 | 79 / 0 / 49 / 3 |

**Hot spots** (where every sync will conflict):

| File | + / - | Mostly |
|---|---|---|
| core/place/stager.cpp | 4,646 / 57 | decode-impl 4,403: the CPU path (hx_*), EP routed reads, LCP, P1 and the lent ring, read-ahead, route trace, TBO, all inside FileStager |
| tests/sched_test.cpp | 2,957 / 2 | decode-impl: consumer skip, DSpark gate/conf, EORS, PREFILL_ALONE, P1 chunks |
| tests/place_test.cpp | 2,181 / 0 | PR266 335, dev 299, decode-impl 1,544 |
| core/place/mover.cpp | 1,690 / 23 | dev 322 (RAM cache), decode-impl 1,207 (host planes, packed cache, mirror, P1 ring) |
| core/sched/scheduler.cpp | 1,413 / 37 | decode-impl: DSpark gate/conf/solo, consumer skip, PREFILL_ALONE, EORS, P1 end-aligned chunks |
| core/place/stager.h | 936 / 1 | decode-impl: the stats structs and members of the above |
| tests/ref_test.cpp | 844 / 1 | PR266 (ds41 reference ops) |
| core/engine_bringup.cpp | 702 / 65 | decode-impl 430, dev 167, PR266 91 |
| core/engine.cpp | 666 / 12 | decode-impl: shutdown statistics |
| core/format/checkpoint.cpp, tests/checkpoint_test.cpp | 404 + 505 | PR266 (DeepSeek and GGUF import) |
| core/place/mover.h, planner.cpp | 450, 445 | decode-impl (host planes API, VRAM packed, draft VRAM placement) |

Lines naming a feature (a lower bound: function bodies that never name it are not counted;
`/tmp/ds41sync/features.py`): CPU path 1,455; route trace/profiles/traces 690; read-ahead/predictor/side store 482;
consumer skip 417; DSpark bank/draft VRAM 413; P1 card prefill 373; LCP/exact tall 282; RAM cache 267; plane mirror
257; VRAM packed 246; DSpark confidence controller 210; EORS/prefix 206; DSpark gate 162; live chain/tape dump
146; EP routed reads 130; TBO 83 (+23 two_batch); FP8 dense 74; lent ring 68.

### 4.1 How to shrink it

Ordered by lines saved per unit of effort and risk. "Safe" = no measured configuration uses it.

1. **Drop the miss-path machinery** (safe since the 256 GB box: the whole expert set fits the 192 GiB cache and
   misses vanished; HANDOFF: "don't invest further in miss-path tuning"): the read-ahead in pieces and its router
   predictor (`..._PREFETCH=pieces` and its `..._PREFETCH_*` knobs, `PpK`/`Pu`/`Pw`/`Pmr`/`Pi8` parts; own files
   readahead.{h,cpp}, router_predictor.{h,cpp}, router_score.h, ~2.1K lines plus ~1.9K test lines), the side store
   (`Ns7`, hx_side_store.h), the plane mirror (`R`/`Rd` parts, `--expert-plane-mirror*`; plane_mirror.{h,cpp} 310
   lines, 4 config flags), the pre-wake. Not the on-die reads (`Pa`/`Pm56`/`Pe`) or early issue (`Ei`): those act
   on every decode layer, misses or not.
   Saves roughly 700-900 lines in upstream files (FileStager functions ~440, their stats and members in stager.h
   ~150, Mover ~100, engine.cpp statistics ~60, config ~40), a sixth of the stager hot spot and its most intricate
   part. Effort: 1-2 days with the tests; the recommended configuration loses its `Pp4-Pu-Pi8-Ns7-R-Rd13` parts, so
   one window to confirm the speed is unchanged (with no misses they should cost nothing either way).
2. **Drop the instruments that served one study** (safe): the route trace (`..._ROUTE_TRACE`, route_trace.{h,cpp}
   563 lines, rtrace tools), the M1 prefetch trace, `..._LCP_PROFILE`'s PassProfile, the miss-read leg timings
   (MissReads), most of the shutdown statistics in engine.cpp. Saves ~450-600 lines in stager/engine/mover. Effort:
   half a day; no window needed (instruments change no bytes), but keep whatever the harness greps.
3. **Drop two-batch overlap** (parked: -3 to -5%): core/sched/tbo.h, ds41_tbo_arch_test, tbo_stager_test (~1,040
   lines of our files), `rad_declare_two_batch` in abi/rad_builder.h and core/build (~35 lines), Program::two_batch,
   the tbo_* functions and members in FileStager (~120 lines), Ctx::ep_rows_ok's split branch, deepseek41's
   micro-batch issue. Saves ~120-150 lines in upstream files and one ABI entry point. Safe.
4. **Drop the lent ring** (off by default, superseded by the lent arena): RingLender in card_expert_stager.*, Mover's
   `set_ring_resident`/`ring_resident`/`file_unit_in_slab` and the ring-copy publishing, the `..._P1_LEND` paths in
   FileStager. ~150-200 lines in mover/stager plus ~400 in card_expert_stager. Safe once the arena is the only
   lender (it is the recommended configuration).
5. **Drop the adaptive DSpark switches that measured no gain**: the confidence controller and load-aware drafting
   (draft_conf.h, `..._DSPARK_CONF*`), the acceptance gate (draft_gate.h), DSpark beside concurrent requests
   (opt-in, -4.5% at 4 streams), the deferred context pass if unused. Saves ~370 lines in scheduler.cpp/.h and
   sched_test (~200 more), plus engine_dconf.cpp. Check HANDOFF next step 3 first (cost-aware verification builds
   on the gate's statistics).
6. **Move ds41 code out of upstream files behind narrow hooks** (the structural fix; medium effort, no behaviour
   change, one exactness window): FileStager is the hot spot because the CPU path, EP routed reads, LCP and P1
   live in its methods: upstream's stager.cpp is 552 lines and stager.h 299, ours 5,141 and 1,234. Give FileStager one virtual hook
   interface (begin_pass / before_op / operands / end_pass already exist as OpStager overrides) and move
   everything ds41 into a `Ds41FileStager` (our file) that wraps or derives from it: stager.cpp would keep
   upstream's code plus ~30 lines of hook calls, and stager.h the same. The same pattern for Mover (a
   `HostPlanes`/RAM-cache side object owning the mapped cache, plane reads, packed cache and the P1 ring:
   mover.cpp's +1,690 (3,586 lines against upstream's 1,911) becomes ~100) and for Scheduler (DSpark gate/conf, consumer skip, PREFILL_ALONE behind a
   policy object: scheduler.cpp's +1,413 becomes ~150). Saves ~6,500 of the 8,300 core/place lines and ~1,200
   of core/sched; conflicts then land in a few hook call sites. 3-5 days with tests.
7. **Converge duplicate mechanisms with upstream's** (medium effort, each needs a window):
   - routed reads: once upstream's RouteCounts path is proven on ds41 (the block's ecnt read stream-ordered at the
     trigger), the CountsRow path and route_counts_live can go, and MoeFP8's counts revert to upstream's (removes
     finding 5's revert). The EP witness still needs a per-layer identity for the counts buffer.
   - `DirectFile::read_covering` (0e01f955): unused by the engine since FileTier::read takes upstream's clamp;
     remove it and its tests (~30 lines).
   - stats: our file_* counters next to upstream's ssd_* (fold file_relaid* and mapped_source_* into the new
     ssd_* family).
8. **Send small fixes upstream** (removes divergence for good): the `--expert-pin-top` threshold (planner.cpp:
   upstream's `shares[...] - 1.0f` sits below every share, so the pin takes the whole plane; the dev tree's fix makes
   ties stay movable as the comment says), the rank-local drafter proposal handle (engine_draft.cpp), the
   checkpoint/scale lookups already in PR266, `RAD_MAP_MUL`.
9. **When PR266 lands upstream**: take its arch/deepseek41 and libr4d/r4d_ds41_* wholesale and reapply our deltas
   (6,155 lines of PR266 origin in upstream files and 18.7K in its own files then disappear from our footprint);
   expect conflicts in its "DS41 file relayout" (acc9ae58) against our weight_file_relaid and RAM cache.

## 5. Behaviour changes and decisions, in one list

Where taking upstream's version changes what our configurations do, and what was decided:

| # | Change | Decision | Effect on measured configurations |
|---|---|---|---|
| 1 | RMS norm on the 256-lane partition (libr4d) | taken | **Numerics**: decode-height norms round differently; new base reference (finding 1). |
| 2 | `--spec-ngram` beside a declared head refused; draft_head_off gone | taken, ds41 switch added | `Sg`/`Sga` arms unchanged (harness sets `RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1`); by hand, `--spec-ngram K` on ds41 needs the switch. Other models: upstream's refusal. |
| 3 | MoeFP8 reports the block's `ecnt`, not the core's row | **not taken** (reverted) | Taking it would disable every routed-only and EP routed read (whole-layer reads, the CPU path refusing capped layers). |
| 4 | Routing-aware file reads (on by default upstream) | merged, **left unbound** | Binding it would change passes our routed reads decline (prefill chunks) from whole-layer reads to routed-only reads with dummy publications. |
| 5 | Hot VRAM cache, deposits, relabels | merged, dormant | Dynamic plans only; we always run `--deterministic`. Guarded against relaid/RAM-cached units in case someone runs without it. |
| 6 | FileTier::read clamps at end of file (vs our read_covering) | taken | Same bytes; the container's last extent takes the bounce path. |
| 7 | `--expert-pin-top` threshold | ours kept | Unused by ds41 (upstream's would pin every resident). |
| 8 | `reserve_ckpt` offset, window tail snapshot inside its chunk | taken | Same value at chunk ends (all exact-mode checkpoints); under `--prefix-resume nearest` with window checkpoints the tail snapshot moves inside the chunk (only the Pcn arm). |
| 9 | Checkpoint store floor before the host reserve | taken | Changes the slot count only when the reserve would leave the store empty. |
| 10 | Preempted-prefill gate, `--preempt-wait-ms` | taken | Only when a request is preempted (KV pool full). |
| 11 | `--draft-short-from` default auto | taken | Never for a chain (DSpark); our README passes `1` explicitly. |
| 12 | Startup lines: lookup-only message, report lists non-default options only | taken | No harness grep depends on them (checked). |
| 13 | Server: Host must name the loopback for keyless operator routes; duplicate security headers refused | taken | The harness uses 127.0.0.1 and plain routes. |
| 14 | `rope_rows` op renamed `ds41_rope_rows` (ours) | required | None in bytes; every plugin must be this build's. |
| 15 | stage_file_units returns early when nothing goes through the read windows | adjusted (ours) | Taken as is it would skip every relaid or RAM-cached unit: ds41's whole file tier. |
| 16 | File-tier dummy region reserved from the pinned host pool | taken, with a fallback (ours) | A full small pool would refuse to start; the region then comes from outside the budget. |

## 6. Status, build and tests

Head of `claude/ds41-sync`: 926d7e29 (merge 8f16f93b, then a8d44423 the dummy-region fallback, 5436a1fc the ref_test
geometry, 926d7e29 engine_e2e_test's duplicated seeded-sampling test). Build `/home/as/ds41-claude-build/aql-sync` (configured as aql: g++-15 + ccache, clang-21 HIP, gfx1201,
Release; RAD_DEFAULT_HOME = home-sync). Built: radiance, libr4d (92.5 MB, was 71), libref, libavx, libquant,
deepseek41, r4d_selftest, every arch plugin and test plugin, 75 host tests, the ds41 emulator tests.
Home staged: `/home/as/ds41-claude-build/home-sync` (architectures/deepseek41.so, kernels/libr4d.so libavx.so
libref.so, quantizers/libquant.so, all from this build; every plugin exports the ABI layout fingerprint).

Tests (ctest -LE gpu, exact names, under the build lock): 77 of 82 pass after the fixes, among them p1_stager_test,
ds41_tbo_arch_test, tbo_stager_test, ds41_dspark_defer_arch_test, ds41_dspark_fp8_arch_test,
ds41_work_rows_arch_test, ds41_shape_probe_arch_test, dspark_bank_guard_test, sched_test, config_test,
ab_window_p1_test, ab_window_vp_test, every other ds41_* test, place_test, mem_test, build_test, plugin_test,
shipped_plugins_test, p4b_kv_ring_test, abi_layout_test (upstream's table holds: our fields are growth),
server_test, transport_test, runtime_test, arch_test, kld_test, the host-expert and read-ahead tests, and the
emulator tests r4d_emu_ds41{bal,mask,mx,tall,tall_bal,tall_on,vp,vp_tail}, r4d_emu_bf16f8, r4d_emu_norm,
r4d_emu_mhc. Remaining at the time of writing:
- ref_test: the ds41 rope's sweep geometry still named `rope_rows`; fixed in 5436a1fc, now passes.
- chatparse_test (upstream's DSML format among its cases) and chat_test pass. engine_e2e_test did not compile (the
  merge kept upstream's and PR266's versions of one test, `oracle` declared twice); fixed in 926d7e29 by taking
  upstream's function, rerun started 09:57 (work/sync/tests-e2e2.out).
- native_file_cache_mover_test, native_mapped_copy_mover_test: refuse a HIP build by design; run in a HOST-only
  build (`/home/as/ds41-claude-build/aql-sync-host`, 09:53): native_file_cache_mover_test and native_file_cache_test
  pass; **native_mapped_copy_mover_test fails one subtest**, `mapped_packed_side_store_holds_refused_units_for_the_stager`
  (lines 1476-1478: after hx_choose the read-ahead entry of a unit served from the side store is not dropped as
  cached: `ra_cached` 0 not 1, the read-ahead not cancelled, `hx_ra_live_` not empty; an
  `E host experts: the container changed under layer 0's units ...` epoch line follows). Not yet known whether it
  is pre-existing or from the merge (FileTier::open now caches the file size; the epoch check is ours). The side
  store (`Ns7`) and the read-ahead in pieces (`Pp4`) are in the recommended configuration, so settle this before
  the window: run the same test in a HOST-only build of 9b395fc8, and if it passes there, bisect the merge's Mover
  and FileTier changes (log: aql-sync-host/ctest-host.log).
- r4d_emu_ds41tall_lcp, r4d_emu_ds41p1_quick: over ctest's 120 s under -j4; run alone they PASS (135 s and
  142 s: the tall twin and m64/tiled rows equal launches of 16 up to 256 rows; P1's twin held to the 16-token gemv
  on 86 cases), work/sync/emu-long.out.
- Device-check binaries built (09:52, work/sync/dc.out): aql-sync/dc/ds41_mxfp4_device_check,
  ds41_vpack_device_check, ds41_bf16f8_device_check; aql-sync/hx/ds41_host_experts_test; the mxfp4 check also in
  wt-sync/tools/ds41-host-experts/build/ for the default checks. engine_e2e_test, chatparse_test, chat_test
  queued after them (work/sync/chain2.log).

## 7. What the verification window must contain

Use only this build's artifacts: every plugin carries the ABI layout fingerprint and the loader refuses a mix
(home-main3's libavx/libref from Oct 28 would be refused), and libr4d's `ds41_rope_rows` must meet the same
deepseek41.so.

- Binary and home: `AB_BIN=/home/as/ds41-claude-build/aql-sync/bin/radiance`,
  `RADIANCE_HOME_DIR2=/home/as/ds41-claude-build/home-sync`, tools
  `AB_TOOLS=/home/as/ds41-claude-build/wt-sync/tools/ds41-host-experts` (its `build/` holds this build's
  ds41_mxfp4_device_check for the default checks).
- **Card checks first** (libr4d changed: the norm partition, the GEMM chunking, the rows forms, the rename):
  `DEVICE_CHECKS="tall lcp p1 bal"` with `TALL_CHECK=/home/as/ds41-claude-build/aql-sync/dc/ds41_mxfp4_device_check`,
  `TALL_LIBR4D=/home/as/ds41-claude-build/home-sync/kernels/libr4d.so`, `LCP_CHECK_C=2048`, `P1_CHECK_M=4096`,
  `P1_ROUTES`/`BAL_ROUTES=work/p0/routes-long3086.bin` (as chain24); `VPACK_CHECK=.../aql-sync/dc/ds41_vpack_device_check`
  and `SELFTEST_F32BF16=.../aql-sync/bin/r4d_selftest` for any Rb/Vp arm. Before the window (or as its first step,
  `work/gpu_cmd_window.sh`): `r4d_selftest` on card 0 against home-sync's libr4d.so (upstream's own kernel checks,
  including the new band-edge and our bf16f8 identity), and the default swiglu/chain check
  (`DEVICE_CHECKS=1` or the harness's `check 24 $SILU` step) since the CPU path's bit-exactness proof is against
  libr4d's decode kernels. The `sweep` inside `tall` now shows the standalone norm height-independent at every
  height (no "above the reach" rows for it).
- **A new base-engine arm first** (finding 1): `base` (host experts off, 16-token chunks, the decode-only settings
  the old base used) on the held-out set, compared with the reference base replies
  (`python3 tools/ds41_reply_check.py evidence/ds41/ab-20261028T223812Z/base.json <new base>.json` in the ds41
  worktree). Identical: the old references stay valid and the rest of this list stays small. Different
  (expected for at least some prompts, since decode rows round the 5120/1280/512-wide norms differently): this arm
  is the new reference for held-out, and the prompt sets the speed arms use (code, chat, Uharness turns, the
  concurrency set) each need a base arm from this build, which at ~5 tok/s is the window's main cost; consider one
  base window for all sets.
- **Byte gates, one per kind of change**: a `Kld` pair on the KLD corpus (`KLD_RECEIPTS=long chat`) between the
  merged base and the recommended prefill configuration (Lc2048 + P1 + lent arena), both recorded in this window
  (the reference must come from this build); `Kh` on the Uharness turns (cold vs Pc + Pce), as chain24.
- **Reply identity**: every speed arm's replies against this window's base arm.
- **Speed arms** (compare with the last numbers): decode-only best
  `all-t40-f-M16-H-p256-m196608-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Rb-Vp-Ss-Pu-Is-Pi8-Ns7-Fw` at 1 stream
  (31.24 tok/s, ab-20261030T205440Z); the recommended prefill configuration
  `all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Cs-Mb1`
  at 1 stream (29.45), with `-L1024-Pf-N4-C4-Pmr4` and `-L1024-Pf-N8-C8-Pmr8` (37.2 / 41.5 total, ab-20261031T030023Z)
  and with `-Qh100-Lm4096-Ulong` for the 3,086- and 6,817-token prefills (7.2 s / 17.2 s); one DSpark arm on code
  (`...-Ds1-...-Ucode256` with `DRAFT_BANK`, +12.3%). Optional: one `Sg3` arm to confirm the LOOKUP_ALONE path.
- Watch in the first arm's server log: `file stager:` and `host experts:` lines as before (routed reads engaged:
  the `routed`/EP counters non-zero), no `routing-aware reads skipped` line (upstream's path must stay unbound),
  `checkpoint slots:` (finding 481fcd5d), and no `refusing` from the plugin loader.
- About 8-11 arms in all (base, Kld pair, Kh pair, 4-5 speed arms), within the usual window size.

## 8. Risks

1. **The base reference** (finding 1). If the merged base's replies differ from the old ones, every reply set
   and Kld reference recorded before the sync is void for this build; the base engine runs at ~5 tok/s, so new
   references cost window time. Reverting the norm partition (about 20 lines in r4d_model_bf16.hip) is the
   fallback if that cost is not worth it.
2. **Dormant upstream paths in our build**: upstream's routing-aware reads (unbound), hot VRAM cache and
   deposits (dynamic plans only, guarded), the MoeFP8 ecnt reporting (reverted). A later change that binds
   RouteCounts or runs ds41 without `--deterministic` enters code no window has measured with our RAM cache.
3. **Startup on small host pools**: the dummy-region fallback is untested on the cards; the first arm's
   `mover:` line shows the pools. An arm that fails at startup with a host-pool shortfall names this.
4. **libr4d.so is 92.5 MB (71 MB before)**: IQR, Nemotron and other kernels; their code objects take some
   card memory before the budget is measured, so each rank may hold a VRAM resident expert or so fewer
   (18.8 MB each); compare the `holds N static residents` / `mover: ... static vram` lines with the last window.
5. **Scheduler under KV pressure**: upstream's preempted-prefill gate changes what happens after a
   preemption (waits for the whole recompute, then reclaims cache, then preempts a decoder after
   `--preempt-wait-ms`). Our concurrency arms rarely preempt; if one does, its timings are not comparable with
   older windows.
6. **Every plugin must be this build's** (layout fingerprint, `ds41_rope_rows`): a window that mixes home-main3's
   libavx/libref or an older deepseek41.so refuses to start (by design).
7. **Prompt lookup alone** by hand now needs `RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1` on ds41 (finding 3).
8. **The next sync**: PR266's head (acc9ae58) already merged 1.15.0 with its own "DS41 file relayout"; when the PR
   lands in main, its arch/deepseek41 and libr4d/r4d_ds41_* and that relayout meet ours (section 4.1, item 9).

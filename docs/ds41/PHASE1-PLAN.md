# Prefill Phase 1: first milestone (M1), card experts on large chunks

**Status.** This is a plan. No code exists yet beyond the shared header. I wrote it read-only against main `1aae0815` and used no GPU. The shared contract header is committed on branch `claude/ds41-p1-base` (`f20356ff`, worktree `/home/as/ds41-claude-build/wt-p1-base`) as `tools/ds41-host-experts/ds41_p1.h`.

**Inputs:**
- PREFILL-200-DESIGN.md, sections 0, 2.1–2.8, 3 and 4 (Phase 1), plus its critique;
- PREFILL-ENGINE-SURVEY.md, section 0 and Strata/DS41Engine sections B and C;
- PREFILL-P0.md and its raw logs (`work/p0/gpu1`; run 2 is still queued);
- the lcpfast diagnosis (uncommitted work in `wt-lcpfast`, README section "Balanced routed gemv");
- the code named below.

Line numbers refer to `1aae0815`.

## 0. Summary

**M1 in one paragraph.** `RADIANCE_EXPERIMENTAL_DS41_P1=1` runs beside `RADIANCE_EXPERIMENTAL_DS41_LCP=2048`, with chunks aligned to the end of the prompt. In every pure prefill pass taller than 16 rows, each routed expert row runs on the cards:
- **VRAM residents** run in place, in the plugin's own three expert ops. Every touched non-resident expert is published to those ops with a skip mark.
- **RAM-cache units** are borrowed read-only from the pinned packed slab. The card's DMA lane copies them, with copy lists, into a 12-slot VRAM ring.
- **Drive misses** are read by the kept readers and the plane mirror. A per-rank packer thread packs each one into a pinned ready slot (the same packer the RAM cache uses), and the DMA lane then copies it to the ring.
- **Waves.** The stager computes ring units in waves of 2 by launching the layer's gate_up, clamped SwiGLU and down again through a new "launched" hook. Its own expert table names the wave's ring slots and skip-marks every other entry. The card decodes the packed form in the twin's kernel (gemv_vp's vp2 kernel at any height), with no widening step.
- **Reading ahead.** The next layer's likely misses are read while the current layer finishes and the next dense half runs.
- **Cache policy.** The RAM cache stays read-only. Units used by the prompt's last 128 rows are admitted, using LCP's existing admission.
- **Unchanged.** The CPU expert path is untouched for decode and for passes of 16 rows or fewer.
- **The ring** is the file stager's two layer buffers, enlarged with `FILE_BUFFER_UNITS=6`, so no new VRAM mechanism is needed.

**Decisions made here:**

| Question | Decision | Why |
|---|---|---|
| How the MoE op gets per-expert ring pointers | The stager replays the plugin's own gate_up, SwiGLU and down launches, captured by a new `OpStager::launched` hook. Each replay carries a per-wave device table: ring slots for the wave's experts, `kMarkSkip` for everything else. The main launch skip-marks every non-resident expert. | The same resolved rows and arguments as the plugin's ops, so the bits match by construction. No plugin, ABI or graph change; decode's op graph is untouched. A ring smaller than a layer works without a device-side wait. |
| Packed decode or on-card widening | Packed decode inside the twin. Under P1 the twin runs the vp2 kernel, which already decodes packed entries bit for bit and is faster than the plain kernel for block entries too. | No landing slots, no extra kernel, 4.4% fewer bytes over PCIe. Widening is kept only as a fallback if vp2 fails at tall heights (see K-5). |
| Drive misses: CPU pack or card relayout | CPU pack (`ds41hx::pack_planes`, the RAM cache's packer, with its round-trip check) into pinned ready slots; block relayout (`Mover::relayout_`) for the 7 units the packer refuses, or with Pa off. | The ring then holds exactly the bytes the RAM cache would. Admission stays a plain insert. No new card kernel. It costs ~2.5 ms of one thread per unit, against the drive's 10 ms per unit per rank. |
| Ring and VRAM | The ring is the file stager's buffer region (`Mover::file_vram_`), sized by `FILE_BUFFER_UNITS=6`: 12 slots of 18,800,640 B, 215 MiB a card, taken from residents by the planner. | The planner already charges it (`planner.cpp:794`). Arm part `B6` instead of today's `B1`. |
| Chunk geometry | C = 2048 (LCP's maximum), aligned to the prompt's end on the 16-row grid (`ds41p1::first_chunk_end_aligned`). The kernel is bounded at 4096 so that C 3072/4096 can be tried once the dense sweep covers them. | The critique's replay: start alignment loses up to 10%. |
| Waves | 2 units a wave by default (`..._P1_WAVE`). RAM-cache units first, in descending row count; then drive units in the order they land. | The drive delivers one unit per rank about every 9–10 ms, and a unit computes in about 1–3 ms. Small waves keep the card on the drive's heels. |

**Expected exact prefill.** These come from the P0 model with measured constants (drive 2.31 ms a unit with copies draining, measured DMA rates, dense half 11.7 µs a row-layer); section 9 has the full table.

| Tokens | Expected tok/s (pessimistic) | Expected TTFT | Today |
|---|---|---|---|
| ~1K | **122 (114)** | | 37 tok/s at 933 tokens |
| ~2K | **216 (201)** | | |
| 3,086 | **220 (205)** | 14.0 s (15.1) | 62.3 s |
| 8K | **~295 (~270)**, extrapolated | 6,817 tokens in ~24 s | 114.5 s |

These hold for any twin cost up to 40 µs an expert-row. Today's served twin costs ~1,000 µs an expert-row because of the dropped-slot tail, which gives 18 tok/s. That is why T4 and T5 in the header are not optional.

## 1. What the measurements say (inputs used)

| Item | Value | Source |
|---|---|---|
| DMA lane, a card alone | 27.4 GB/s on every card (block and packed units) | P0 `m1-block-yield.log`, `m1-packed.log` |
| DMA lane, all four at once | 26.7 / 22.6 / 13.5 / 13.5 GB/s (54.3 total); cards 2+3 together 16.2 each | same |
| Copies beside a VRAM-bound kernel | the kernel loses 40% on cards 0/1 and 6% on cards 2/3; the copies lose 15% | same |
| Drive in layer batches, copies draining, one RAM-like copy per drive unit | 8.14 GB/s, 433 units/s, 18.8 MB a unit; slot-to-landed median 145 ms at 16 slots/8 readers per rank | `m2-drain-extra1.log` |
| Drive, engine order, no extra copies | 7.87 GB/s (drained), 7.40 GB/s (alone, 48 slots) | `m2-drain-engine.log`, `m2-alone-engine.log` |
| Dense half at 2,048 rows (measured ops) | 4.6–5.8 µs a row-layer + attention 3.1–3.3 µs a row + index_score 3.9 (ratio 2) or 8.0 (ratio 1) µs a row at 8K context | `m3-dense.log`, `m3-sparse.log` |
| Today's twin as served under LCP | 2.4–3.3 ms a row-layer at M 1024 (gate_up 1.9–2.6 s a call) | `m3-twin.log` |
| Why it is that slow (lcpfast) | `mx_range` gives the rank's last local expert every slot routed to other ranks (`[off[95], T)`, 4.6–5.5K of a 1,024-row chunk's 6,144 slots), one weight pass each at XLM 1, on that expert's few blocks. Fit: 5.1 µs a row group on a lone block. 388 s of a 463 s Lc1024 prefill was the issuing thread waiting for the card. | `wt-lcpfast` README section, window H |
| Real routing at M 2048 | 2,887–3,447 local rows a rank-layer (1.41–1.68 a token); 51–72 of 96 experts touched; the busiest expert 1,956–2,026 rows | `work/p0/routes-stats.txt` |
| Exactness today | LCP at C 256/512/1024: KLD records identical (3,397 positions × 20 documents); prefill KV hashes 5/5 for Cs, Lc1024 and Lc1024+Cs | window H |
| P0 run 2 (twin cut to local slots, live tail, gemv controls, M1 again, M2 over 20 layers) | **not in yet**: queued at 12:50 behind WINDOW_PENDING. K uses its `twin-cut` numbers as the baseline its kernel must beat | `work/p0/run_p0_gpu2.sh` |

**What the other engines show** (survey section B/C), and how M1 follows it:
- **All experts on the GPU in big batches.** DS41Engine and Strata compute every routed expert on the GPU, in 1K–12K-row batches. M1 does the same at 2,048 rows.
- **Small double-buffered staging.** Non-resident experts stream through small staging buffers: DS41Engine's pinned bounce halves, Strata's 16–1,200 MB ring. M1 uses 12 slots and waves of 2.
- **Pre-staging the next layer.** DS41Engine pre-stages the next layer during the dense half (+5.8%). M1 reads the next layer's likely misses ahead.
- **Routed-only reads.** DS41Engine reads only routed experts on demand, which beat whole-layer prefetch once its reader was deep. M1 reads touched units only.
- **Read-only RAM tier.** Both keep the RAM tier read-only during a prompt. M1 does too, with the last-128-row admission.
- **CPU experts for decode only.** Both use CPU experts only in decode and short turns. M1 keeps the CPU path for passes of 16 rows or fewer.
- **Where M1 is stricter.** Neither engine is exact. M1 keeps the decode-band arithmetic (the 16-row gemv's bits) at every height.

## 2. Scope of M1

**In:**
- the P1 switch and its refusals;
- the twin's P1 row (K);
- the launch-capture hook;
- the card expert stager: ring, RAM-cache DMA, drive staging with the packer, waves, pre-staging, lifetimes;
- end-aligned chunks at C ≤ 2048;
- read-only RAM cache with last-128-rows admission (LCP's `lcp_admit_row`);
- a host verify option;
- stats and log lines;
- `ab_window` parts and the p1 device-check gate;
- prompt receipts of ~1K/2K/5K/8K/16K tokens;
- the window arms in section 10.

**Out (later phases):**
- Phase 2 segments (dense pieces with the MoE once per segment);
- C > 2048, which needs the dense sweep at 3072/4096 first;
- speculative RAM-unit DMA ahead of the counts;
- peer-to-peer rebalancing of ranks 2/3;
- displacing residents instead of reserving (Phase 4);
- the Phase 3 row-batched kernel;
- Engram dedup per chunk;
- prefix-cache checkpoints inside P1 chunks: with `--prefix-resume exact`, M1 keeps start-aligned chunks and says so.

## 3. Execution, per chunk and per routed layer

The pass is an LCP chunk of R rows (17 ≤ R ≤ 2048), a pure prefill step (`RadBatch::lcp_admit_row` path) and armed. The consumer skip's row cut applies per layer as today (`FileStager::layer_r0`). Layers at or past `pass_full_` are skipped as today.

```
issuing thread (rank r)                             compute stream            mover stream (DMA lane)
---------------------------------------------------------------------------------------------------------
dense half of L, router L  ......................>  [dense L][router L][counts copy]
before_op(gate_up L)  -> start_ep_routed:
  wait counts (existing sync)  <...................
  classify the 96 local experts: rows==0 | resident | RAM hit | drive miss
  RAM hits: borrow_host_unit(touch = admitted)      
  publish: touched non-resident -> skip-marked alias; untouched -> alias (as today)
  post drive misses not already read ahead (descending rows; <= INFLIGHT in flight)
  queue DMA for the first ring-capacity RAM units .......................................>  [copy u1..u12 -> slots]
gate_up / SwiGLU / down launch  .................>  [gu L: residents only][sw][dn: residents]
  launched() hook: copy RadArgs + launch fn of each
before_op(first gather of L) -> run_waves(L):
  for each wave w (RAM units first, then drive units as they land and are packed):
     (drive) wait landed -> packer -> pinned ready slot; queue DMA ready->slot ..............>  [copy (wait slot free)]
     upload wave table w (1.5 KB) ...............>  [wait copy-ev w][tab w][gu w][sw][dn w][rec free(slots of w)]
     queue next RAM copies into slots w frees ..........................................>  [copy (wait free)]
  after the last demand read is posted: post read-ahead of layer L+1's likely misses (<= AHEAD)
  release leases and ready slots whose copy-done events have passed (poll); the rest at pass end
gather, all-reduce, mhc_post, dense half L+1 ...>
```

**Invariants (S must assert them in debug builds):**
- Every touched non-resident expert of the layer is in exactly one wave. Its rows are written only by that wave's gate_up and down.
- A skip-marked entry is never in a table the gemv row (≤ 16 rows) reads.
- A slot's copy waits on the free event of every earlier occupant (the ring laps, counted in slots as `PrefillStager` counts bytes).
- A lease and a pinned ready slot are released only after their copy's done event.
- Waves reuse the main launch's `RadArgs` unchanged, except for the weight table's `data` pointer.
- The SwiGLU replay runs over all of the layer's `rows*6` slots. This is idempotent: the residents' rows recompute from unchanged `egu` to the same bytes.

## 4. Data structures and mechanisms (package S unless marked K)

### 4.1 The ring

| Item | Value |
|---|---|
| Region | `Mover::file_vram_` (`mover.cpp:509`), today split into `buf_[0]` and `buf_[1]`. Under P1, `CardExpertStager` sees `[file_vram_, file_vram_ + 2*file_buf_bytes_)` as `n = 2*file_buf_bytes_ / kSlotBytes` slots. A new accessor, `Mover::file_region(int64_t* bytes)`, provides it. |
| Slots | `FILE_BUFFER_UNITS=6` gives 12 slots (215 MiB). P1 refuses fewer than `kMinRingSlots` (4), i.e. `B2`. `B8` gives 16 slots. |
| Slot layout | gate_up at +0, down at +`kDnOff` (12,533,760). A block unit fills the slot (18,800,640 B); a packed unit (11,999,296 + 6,001,984 B) leaves the tail unused. Table entries: `slot` and `slot + kDnOff`, each with `kMarkPacked` when packed. |
| Events | `free_ev[slot]`, recorded on the compute stream after the down of the wave that read the slot. `copy_ev[w]`, recorded on the mover stream after wave w's copies. `ready_ev[k]` behind each pinned ready slot's copy, and per lease. Created once at init, from small pools: slots, 2 × max waves, ready slots. |
| Hand-over with decode | Decode passes still use `buf_[0/1]` as alias targets and, under the greedy policy, as copy targets. A P1 pass's first copy waits on the FileStager's `free_[0]` and `free_[1]` (where recorded). A P1 pass's end records `p1_end_ev`, which the FileStager's next `stage_file_units`/copy into `buf_` waits on. |

### 4.2 Wave tables

- Pinned host array `[2 layer parities][kMaxWaves = 96][2 weights][96 entries]` of `const void*` (≈ 295 KB a rank), plus a device array of the same shape (`rad_dev_alloc`, outside the plan: tiny).
- Wave w of layer L fills `host[L&1][w][k][e]` for every local expert e:
  - the wave's experts: `slot (+ kDnOff for down) | kMarkPacked?`;
  - every other expert: `kMarkSkip | alias_base`.
- `rad_memcpy_async` sends the 1.5 KB to `dev[L&1][w]` on the compute stream, right before the wave's gate_up.
- **Why parity is enough.** Layer L+2 can reuse the host rows only after the host has synced on layer L+1's counts. That sync orders it after every wave of L.
- **Waves a layer** are at most `ceil(touched non-resident / WAVE)` ≤ 96. P1 refuses `..._P1_WAVE` < 1.

### 4.3 Classification and the units' sources

- **The touched set.** Per local expert e: rows = `counts_host_[e]`, from the existing counts copy, under the layer's row cut. "Touched" means rows > 0.
- **Unit ids and tiers** come from the plan: `layers_[li].units`, plus residency (`Mover::unit_base`, slab or file tier).
- **Refusal at init.** P1 refuses if any routed unit of an EP layer is neither VRAM-resident nor file-tier (a pinned-pool unit): "P1 reads only slab residents and file-tier units".
- **RAM hit:**
  - Source: `Mover::borrow_host_unit(u, &hu, touch = hx_admits(li, e))` (`mover.h:610`; touch only for an admitted unit, per `lcp_admit_row`). The slab is `RAD_MEM_HOST_MAPPED`, so the DMA reads it in place.
  - Copy list: two segments, `hu.host[0]` (`hu.slot[0]` bytes) → slot+0 and `hu.host[1]` → slot+`kDnOff`, using the same `RadCopySeg` list API as `Mover::stage_units` (`mover.cpp:792`). New: `Mover::copy_to_ring(const RingCopy*, int n, RadEvent after_each[], RadEvent done)`.
  - The lease is held until `copy_ev` passes.
- **Drive miss:**
  - Reads: `Mover::start_unit_planes`-style posting with the per-unit `landed` callback (`mover.h:680`), the plane mirror's share at the pass's row count, at most `INFLIGHT` units a rank in flight (default 16, as P0 M2 measured). HostPlanes stay malloc'd, as today.
  - On landing, the per-rank packer thread runs `ds41hx::pack_planes` (with `pack_plan` and the `packed_scales_equal` round-trip check, exactly as `insert_unit_packed`, `mover.cpp:1420`) into a pinned ready slot of `kSlotBytes`.
  - If the packer refuses (raw-round budget), or the packed cache is off, the unit is relaid to block form through `Mover::relayout_` (the file path's relayout) instead.
  - Ready slots: `READY` a rank (default 6, 113 MB pinned a rank), allocated at startup.
  - Afterwards the HostPlanes go to `hx_queue_inserts` if the unit is admitted (rows in the last 128); otherwise they are freed.
- **Order of reads:** misses in descending row count; matched read-ahead units first.

### 4.4 Read-ahead (pre-staging)

- **When.** Once layer L's demand reads are all posted, the stager posts layer L+1's predicted misses: the next routed layer whose MoE this pass runs.
- **Which.** Up to `AHEAD` units a rank (default 8). Candidates are the file-tier units of L+1 not in the RAM cache (probed with `borrow`/peek, no touch), ranked by the static expert profile prior (`expert_profile_map.h` counts; no profile means container order).
- **At L+1's classification:**
  - Matched units are not read again: landed ones go straight to the packer, in-flight ones are awaited.
  - Unselected ones are dropped, their HostPlanes freed, and counted in `ahead_dropped`.
- **Why it helps.** It covers the drive's idle time between L's last landing and L+1's counts: L's last waves, the gather, and L+1's dense half (~10–25 ms at 2K rows). The design credits 50–90% of the dense half.
- **Arm `Qa0`** turns it off, which measures what it is worth.

### 4.5 How a layer's MoE receives per-expert ring pointers: the launch capture and wave replay

**New in `core/runtime/stager.h`:**

```cpp
/* After op `h`'s kernel was launched on `s` (device domain, a pass the stager armed for): the
 * resolved arguments and the launch function, so the stager may launch the same row again later
 * in the pass with a changed weight table. `a` and what it points to live only for this call. */
virtual int launched(rad_op h, RadStream s, const RadArgs* a, int (*launch)(const RadArgs*, RadStream)) { return RAD_OK; }
```

**The call site.** `Ctx::issue_body` calls it right after `s = ip.launch(la, cur_stream())` (`issue.cpp:1110`), under the same condition as `operands()` (`staging_ && stager_ && dom == RAD_DOMAIN_DEVICE`). `StagerChain` forwards it.

**What the CardExpertStager keeps.** For each of the layer's `gu`, `sw` and `dn` ops it deep-copies:
- the tensors (n_t RadTensor structs);
- the params (n_p RadParam structs; their key and string pointers are program-lifetime);
- scratch, scratch_bytes, instance, rank and world_size;
- the launch pointer.

**Identifying the ops (op map at init).** By scanning `prog.ops` over each routed layer's span:
- gu: `hx_ops_[li].gu` (exists);
- dn: the `moe_gemm_ggml` with `a_order = "sorted"` after gu;
- sw: the op between them whose input is gu's output buffer (`swiglu_clamped` in ds41's bf16 path, `rad_fp8.h:583`);
- gathers: `hx_ops_[li].gather[]`.

P1 refuses at init a layer where any of these is missing or ambiguous, and at run time a layer whose gu is launched twice in one pass (a sliced block).

**A wave launch.** Copy the captured args, set `t[1].data` (the table operand of gu and dn) to the wave's device table, and call the captured launch for gu, then sw (args unchanged), then dn. All go on the compute stream, behind `rad_stream_wait_event(copy_ev[w])` and the table upload.

### 4.6 The twin under P1 (package K; contract T1–T6 in `ds41_p1.h`)

- **Registration.** `r4d_rows.cpp`'s `tall_twins()` (`r4d_rows.cpp:10118`) registers the twin as `moe_gemm_ggml_mxfp4_gemv_tall_p1` when `ds41p1::state()==1` (restated in `r4d_ds41_tall.h`), with M ≤ 4096 and priority gemv−1.
- **Launcher.** It routes table calls to the vp2 kernel (`mx_vp2_kernel`, `r4d_ds41_mxfp4.hip:1289`) at every M with:
  - the live-tail ranges (`ds41_live_tail::range`; T4), plus one zero-fill of `[off[96], T)` per call (as `mx_tail_zero`);
  - skip marks honoured in the live-list build and the work loop (T3): a skip entry contributes no work items and its rows are never stored;
  - work balanced by row passes (T5): lcpfast's chunking of an expert's passes into chunks that fill the grid once, applied to the vp2 body.
- **No skip handling below 17 rows.** The gemv and gemv_vp rows are unchanged; the stager never marks for them.
- **Vp.** Packed VRAM residents (Vp, bit 0 set by `Ctx::table_refresh`) are decoded by the same kernel, so Vp becomes legal beside LCP under P1.

### 4.7 VRAM and host memory

| Item | Per card / rank | Note |
|---|---|---|
| Ring (B6) | 215 MiB | vs today's B1 at 36 MiB: +179 MiB, ≈ 10 residents fewer (−0.5% decode) |
| LCP arena at C 2048 | ≈ 0.9–1.3 GiB (critique), +0.45 GiB over C 1024 | already how Lc arms run; costs ≈ 25–35 residents more than Lc1024 (−1.4 to −1.9% decode). M5's arena line is printed by the planner; record it. |
| Wave tables | 0.3 MB device + 0.3 MB pinned | |
| Pinned ready slots | 6 × 18.8 MB = 113 MB a rank (452 MB in all) | allocated at startup only under P1 |
| HostPlanes in flight + read ahead | (16 + 8) × 18.8 MB = 451 MB a rank (1.8 GB in all) | malloc, as LCP's 48 a rank (3.6 GB) did in window H; `.mem` files must show MemAvailable > 4 GB |
| Packer thread | 1 a rank | sleeps outside P1 passes |

### 4.8 Chunk geometry

- **End alignment** (`..._P1_ALIGN=end`, the default). In `Scheduler::plan_chunk` (`scheduler.cpp:1486`), when P1 is on, LCP is C and the step is a pure prefill of one prompt:
  - the first chunk of what remains is `ds41p1::first_chunk_end_aligned(remaining, C, 16)`, aligned on absolute positions (round the end, not the length, when `n_computed` is off the grid);
  - later chunks are C.
- **Examples:**
  - 3,086 tokens → 1,040 + 2,046;
  - 6,817 → 688 + 2,048 + 2,048 + 2,033;
  - ≤ 2,048 → one chunk.
- **Where start alignment is kept** (with a log line):
  - `--prefix-resume exact` (its checkpoint grid assumes start-aligned chunks);
  - `..._P1_ALIGN=start` (arm `Qs`).
- **Short head chunks.** A head chunk of ≤ 16 rows runs the CPU path, as today. A head chunk of 17–63 rows runs the card path: the design measured the card path as faster even at 82 tokens.
- **The checkpoint interval** must be at least C. `Lc2048` already passes `--max-num-batched-tokens 2048`, and the default interval is 2048. P1 refuses a smaller interval with the flag trap named.

### 4.9 Admission (read-only RAM cache)

- **Reused as is:** `RadBatch::lcp_admit_row` (`batch.cpp:1290`, `..._LCP_ADMIT`, default 128). `hx_admits(li, e)` says whether expert e's rows reach the admit row.
- **Hits:** borrowed with an LRU touch if admitted, read-only otherwise.
- **Misses:** if admitted, they are queued to the insert threads after packing (`hx_queue_inserts`, capped by `..._LCP_INSERTQ`); otherwise freed.
- **Effect.** Only the prompt's last chunk admits anything. The design's replay: −5.7% prefill reads across prompts, +113 decode reads over 5 prompts.

### 4.10 Verify and stats

- **Verify** (`..._P1_VERIFY=N`, default 0):
  - After a layer's last wave, N (expert, row) pairs are sampled among ring units: the HostExpertPool's `..._POLICY=verify` comparison, aimed at ring units.
  - Their `x` rows and `edn` rows are copied back. The host kernels (ds41hx, proven bit-identical to the card) recompute them from the host bytes the slot was copied from (the slab, or the ready slot).
  - Results are counted in the stats line ("verify rows R, differ D").
  - The pass waits for the readback, so verify is only for exactness arms.
- **Stats.** At exit each rank prints `kLogStats` with:
  - chunks, card-path layers;
  - units resident / RAM / drive / read-ahead used / read-ahead dropped;
  - waves, DMA GB, drive GB, drive wall-clock busy %;
  - issuing-thread waits: counts, landing, packer, ready slot;
  - packer ms per unit;
  - leases outstanding at pass end (must be 0);
  - verify.
- **Chunk line.** Rank 0 prints `kLogChunk` per chunk: rows, layers, wall s, drive units, GB/s, MemAvailable.

## 5. The interface between K and S

### 5.1 Shared header: `tools/ds41-host-experts/ds41_p1.h` (committed, `f20356ff`)

**What it holds:**
- the switch (`kEnv`, `state()`);
- the twin row's name and its promises T1–T6;
- the gemv row names the core admits;
- `kTwinMaxRows` 4096, `kMinArmRows` 17;
- the table marks (`kMarkPacked` = 1, `kMarkSkip` = 2) and their helpers;
- the slot layout (`kDnOff`, `kSlotBytes`);
- the stager's tunables (env names, defaults, `kMinRingSlots`);
- `first_chunk_end_aligned`;
- the three log-line prefixes;
- the device check's CLI and pass line.

**Rules:**
- Neither package edits it. A change goes through the integrator, as a commit on `claude/ds41-p1-base` that both merge.
- libr4d does not include it. K restates `kEnv`, `kTwinRow`, `kTwinMaxRows`, `kMarkPacked` and `kMarkSkip` in `libr4d/r4d_ds41_tall.h`, with a pointer to this header. The device check includes both and `static_assert`s that they agree.

### 5.2 What S may assume from K

- **The row.** With `RADIANCE_EXPERIMENTAL_DS41_P1=1` and `..._LCP=2048` in libr4d's environment, every `moe_gemm_ggml` band of ds41's trunk resolves as follows:
  - 1..16: `moe_gemm_ggml_mxfp4_gemv`, or `_gemv_vp` under Vp;
  - 17..2048: `moe_gemm_ggml_mxfp4_gemv_tall_p1`.
- **The launcher** honours T1–T6.
- **The handshake** is the band's row name. A libr4d without the switch gives `_gemv_tall`, which S's `lcp_init` refuses under P1 with the reason.

### 5.3 What K may assume from S

- S calls the P1 twin only with a weight table.
- Entries carry at most one of the two marks.
- Unmarked entries are 64-byte aligned.
- The offsets are `moe_scatter`'s (non-decreasing, `off[96]` = the local end).
- T and AR are as the plugin declares them.

## 6. Package K: the card kernel side

**Branch** `claude/ds41-p1-kernel` from `claude/ds41-p1-base`. **Worktree** `/home/as/ds41-claude-build/wt-p1k`. **Build dir** `/home/as/ds41-claude-build/aql-p1k`, targets libr4d and the r4d_emu tests; the device check via `tools/ds41-host-experts/Makefile mxfp4-check`. Builds go through `flock work/build.lock nice -n19 ... -j8`, and nothing heavy runs while `work/WINDOW_PENDING` exists.

**Files K owns (no one else edits them):**
- `libr4d/r4d_ds41_mxfp4.hip`, `libr4d/r4d_ds41_mxfp4.h`, `libr4d/r4d_ds41_tall.h`;
- `libr4d/r4d_rows.cpp` (only the `tall_twins()` section);
- `tools/ds41-host-experts/ds41_mxfp4_device_check.hip`, `tools/ds41-host-experts/Makefile`;
- `tests/r4d_emu/test_ds41p1.cpp` (new);
- the `# ---- P1 (K)` block appended at the end of `tests/CMakeLists.txt`.

**Tasks, in order:**
1. **K-1. lcpfast.**
   - If `claude/ds41-lcpfast` has commits past `1aae0815` when K starts, merge it first. Its MOE_BALANCE edits are in `mx_gemv_kernel`.
   - If not, read its diff in `wt-lcpfast` (read-only) and port the balance idea to the vp2 body in K-3. Do not touch `mx_gemv_kernel`'s plain path.
2. **K-2. The P1 row.** Restate the constants in `r4d_ds41_tall.h` (`r4d_ds41_p1()`, read once, on only beside a valid LCP). `tall_twins()` registers `kTwinRow` with M ≤ 4096 under P1, in place of `_gemv_tall`; `r4d_ds41_tall_rows()` returns 4096 under P1. Its launcher is a new `r4d_rad_moe_gemm_mxfp4_gemv_tall_p1` → `mx_moe(..., tall = true, p1 = true)`.
3. **K-3. The vp2 kernel at any M for P1 calls.** In `mx_moe`, for a table operand under p1, take the vp2 launch (`mx_launch_vp`) at any token count, with these additions:
   - (a) `A.live_end = 1` plus a zero-fill of `[off[ne], T)`, or the dropped passes stored as zeros without weight reads: one or the other, whichever K's timing prefers;
   - (b) a P1 flag in `MxGemvArgs`: the live-list build treats `wtab[e] & kMarkSkip` as no rows, and nothing else reads that entry;
   - (c) the work balance by row passes;
   - (d) a check that the vp2 body's passes of XLM rows and its `MC` template cover any M (today it serves ≤ 16).
   No change to any call without the flag.
4. **K-4. Host emulator test** `tests/r4d_emu/test_ds41p1.cpp`, on the emulator the existing `test_ds41tall` and `test_ds41vp` use:
   - the P1 twin against the gemv in passes of 16 tokens at M ∈ {17, 64, 256, 1024};
   - mixed block/packed/skip tables, six routings (uniform, skewed, one hot expert of ~M rows, tail-only, no local slot, real routing from `work/p0/routes-long3086.bin`);
   - the untouched sentinel rows for skipped experts and the zeroed tail;
   - the wave composition (main with half skipped + waves of 1–3 + SwiGLU relaunch = one launch).
   It runs with the cards hidden.
5. **K-5. Device check `p1`** (CLI in the header):
   - all of K-4 on a card at ds41's shapes, M up to 4096, plus NaN/Inf/subnormal rows;
   - outputs against the gemv/gemv_vp in passes of 16 and against the host kernels on sampled rows;
   - timing lines (µs an expert-row, real routing, M 256/1024/2048, block and packed; plus the served LCP twin and the P0 run-2 `twin-cut` form for comparison).
   - **Fallback.** If vp2 cannot be made exact at tall heights, K reports it and adds `r4d_ds41_pk_widen` (a packed → block widening kernel), with S then landing packed units in a scratch slot. Not expected: vp2's arithmetic is height-independent like the plain row's.
6. **K-6. Timing target.** P1 twin gate_up + down ≤ 40 µs an expert-row at M 1024–2048 with real routing (the model is flat up to there); stretch ≤ 20. Report the number. Also report the served twin vs P1 twin at M 64/256, so the window knows short chunks are not slower.

**Done means:**
- the emulator test passes;
- `make mxfp4-check` builds;
- a commit of libr4d that builds in `aql-p1k`;
- a note for the integrator: the libr4d sha256, the device check binary path, and the expected p1 PASS line.

The card run of `p1` itself happens in the integrator's window, under DEVICE_CHECKS.

## 7. Package S: the stager and engine side

**Branch** `claude/ds41-p1-stager` from `claude/ds41-p1-base`. **Worktree** `/home/as/ds41-claude-build/wt-p1s`. **Build dir** `/home/as/ds41-claude-build/aql-p1s`: radiance and tests; S builds no libr4d. Same lock and load rules as K.

**Files S owns:**
- new: `core/place/card_expert_stager.h`, `core/place/card_expert_stager.cpp`;
- `core/place/stager.h`, `core/place/stager.cpp` (FileStager glue, `lcp_init`/`hx_init` changes);
- `core/place/mover.h`, `core/place/mover.cpp` (`file_region`, `copy_to_ring`, a pinned ready pool, the packer entry);
- `core/runtime/stager.h`, `core/runtime/issue.cpp` (only the `launched` hook call);
- `core/sched/scheduler.cpp` (`plan_chunk` end alignment) and `core/sched/scheduler.h` if needed;
- `core/engine*.cpp` (the startup line, stats at exit);
- `core/CMakeLists.txt` (the new source);
- new: `tests/p1_stager_test.cpp`, plus the `# ---- P1 (S)` block at the end of `tests/CMakeLists.txt`;
- `tools/ds41-host-experts/ab_window.sh`, `tools/ds41-host-experts/ab_requests.py` (only if TTFT/hash parsing needs P1 lines);
- `tools/ds41-host-experts/README.md`, a new "Card experts in large chunks (P1)" section;
- new files under `/home/as/ds41-claude-build/work/extra-prompts/` (`p1len-receipt.json`, `p1reply-receipt.json`).

**Tasks:**
1. **S-1. Switch and refusals.** Read `ds41p1::state()`; invalid values refuse to start. P1 refuses:
   - without a valid LCP;
   - with C > 2048;
   - with fewer than `kMinRingSlots` ring slots (`FILE_BUFFER_UNITS` < 2);
   - with speculation, or with `FETCH_LIST`;
   - with a non-file-tier pooled routed unit;
   - with a layer whose gu/sw/dn/gather cannot be identified;
   - with a band of a routed MXFP4 GEMM above 16 rows not on `kTwinRow`.

   Under P1, `lcp_init` admits `kGemvVpRow` at ≤ 16 rows and so allows `VRAM_PACKED`. `hx_init` drops LCP's "CPU takes every routed unit" demands (`POLICY=all`, `MISSES ≥ C`), because chunks never go to the CPU; decode keeps whatever policy the arm sets. Print `kLogOn` with the ring slots, wave, in-flight, ready, ahead, verify and alignment.
2. **S-2. Launch hook** (section 4.5): `core/runtime/stager.h`, `StagerChain`, one call in `issue.cpp`, FileStager forwarding.
3. **S-3. CardExpertStager:**
   - classification, publishing with marks: a new `Mover::alias_file_units` variant taking a mark, `alias_file_units(units, base, staged, uintptr_t mark = 0)`;
   - copy lists, waves, wave tables;
   - the ring with its free/copy events and the hand-over events;
   - lease and ready-slot lifetimes;
   - the drive path: posting, landing callback → packer thread → ready slot;
   - read-ahead, admission, verify, stats.

   FileStager branches to it in `start_ep_routed` (after the counts sync; skipping `hx_choose`, `hx_submit` and the stage path) and in `before_op`/`operands` of the layer's first gather. It also calls it at `begin_pass` (arm: LCP chunk, `n_tok ≥ kMinArmRows`, pure prefill) and `end_pass` (drain: wait all copy events, release leases and ready slots, drop read-ahead, record `p1_end_ev`).
4. **S-4. End-aligned chunks** in `plan_chunk` (section 4.8), with a unit test over prompt lengths 17..20,000, the consumer-skip grid and the checkpoint interval.
5. **S-5. Unit tests** (`tests/p1_stager_test.cpp`, cards hidden):
   - the wave planner on a fake mover/device: every touched non-resident unit in exactly one wave; RAM before drive; descending rows;
   - ring lapping: each copy waits on the slot's last free event;
   - wave-table parity reuse; lease and ready-slot release only after the copy event (a fake event clock);
   - read-ahead match/drop; the admission predicate against `hx_admits`;
   - end alignment;
   - a 2-slot ring and wave-1 stress with randomised event completion (the critique's ring stress test, on the host).
6. **S-6. `ab_window` parts:**
   - `P1` (sets the switch; requires `LcN`; the arm is measured only if the server prints `kLogOn` and the `p1` device check passed on the arm's libr4d);
   - `Qw<n>` WAVE, `Qi<n>` INFLIGHT, `Qr<n>` READY, `Qa<n>` AHEAD, `Qv<n>` VERIFY, `Qs` ALIGN=start. Check these names against the existing part grammar first and rename on collision, without touching the header.
   - DEVICE_CHECKS gains `p1`, with `P1_CHECK_M` (default 4096) and the existing `BAL_ROUTES`-style `P1_ROUTES`.
   - The window summary extracts `kLogStats` and `kLogChunk`.
   - Lc2048 arms need `LCP_CHECK_C=2048`.
7. **S-7. Receipts.**
   - `work/extra-prompts/p1len-receipt.json`: five documents of ~1K, ~2K, ~5K, ~8K and ~16K tokens, `max_tokens` 16, chat_text as `long-receipt.json`.
   - `p1reply-receipt.json`: the 2K and 5K documents with `max_tokens` 256, for reply identity.
   - Sources: public-domain or GPL text already on disk, e.g. `/usr/share/common-licenses/GPL-3` and `Artistic`, perl's pod files under `/usr/share/perl*/`, Python's license and docs text. Use different text from `long8k` and `long16k` so the window covers new routing.
   - Measure token counts with the server's tokenizer in the window's first arm (the TTFT lines carry prompt tokens), and note them in the README.

**Done means:**
- the unit tests pass;
- radiance builds in `aql-p1s`;
- `ab_window.sh` passes `bash -n`, plus a dry parse of the new parts (the script's own arm-parse self-check, if present);
- a README section;
- a note for the integrator: the arm strings for section 10, and the expected log lines.

## 8. File ownership matrix

| Path | K | S | Integrator |
|---|---|---|---|
| `tools/ds41-host-experts/ds41_p1.h` | read | read | owns (base branch) |
| `libr4d/*` | **owns** | — | |
| `tools/ds41-host-experts/ds41_mxfp4_device_check.hip`, `Makefile` | **owns** | — | |
| `tests/r4d_emu/test_ds41p1.cpp` | **owns** | — | |
| `core/**` | — | **owns** | |
| `tools/ds41-host-experts/ab_window.sh`, `ab_requests.py`, `README.md` | — | **owns** (K sends its README text to the integrator) | merges K's README text |
| `tests/p1_stager_test.cpp` | — | **owns** | |
| `tests/CMakeLists.txt` | appends block `# ---- P1 (K)` at the end | appends block `# ---- P1 (S)` at the end | resolves the two appends |
| `work/extra-prompts/p1*-receipt.json` | — | **owns** | |
| `WINDOW-P1.md`, `claude/ds41-p1`, `aql-p1`, `home-p1` | — | — | **owns** |

## 9. Expected tok/s with the measured constants

**Model.** The critique's per-layer pipeline (`c2_pipeline.py`) and replay (`c1_floor.py`), imported unchanged through P0's `p0_model.py`. Measured parameters:
- u = 2.31 ms a drive unit (P0 M2 drained, with one RAM-like copy per unit);
- dense half 11.7 µs a row-layer (attention included);
- index 0.91 G scores/s;
- DMA 26.9 / 22.9 / 13.6 / 13.6 GB/s.

"Pessimistic" is u × 1.05, dense × 1.15, index × 0.8, twin × 1.1. I ran it at twin costs of 7.3, 20.7 and 40 µs an expert-row; the results below differ by at most 1 tok/s across those three. Phase 1 is C 2048, end-aligned, with the consumer skip.

| Prompt | Tokens | Phase 1, C 2048: central (pess.) | Same at C 1024 | Phase 2 (one segment), for reference | Today (Cs+Mr16) |
|---|---|---|---|---|---|
| 3,086-doc prefix | 1,024 | **122 (114)** | 122 | 120 | 933 tokens: 25.4 s = 37 |
| code doc | 1,890 | **184 (172)** | 121 | 182 | 47.7 s = 40 |
| 3,086-doc prefix | 2,048 | **216 (201)** | 139 | 213 | — |
| 3,086-token doc | 3,086 | **220 (205)** → TTFT 14.0 s (15.1) | 156 | 303–308 | 62.3 s = 50 |
| concatenation | 4,103 | **242 (226)** | 158 | 349–360 | — |
| concatenation | 6,263 | **268 (251)** | 175 | 417–529 | 6,817: 114.5 s = 60 |
| 8K (extrapolated) | 8,192 | **~295 (~270)** → 6,817 tokens in ~24 s | — | — | |

- **Twin rate.** With today's served twin (≈ 1,000 µs an expert-row, the dropped tail) the same model gives 18 tok/s at 1K. That is the floor M1 lifts by fixing T4/T5.
- **Drive-only floors** at 433 units/s: 1K 127, 2K 229, 3,086 at C 2048 229 tok/s. So M1 should sit within ~5% of the drive.
- **The 8K line** scales the critique's 6,263→8K growth by the measured drive rate. It becomes a measurement once the `p1len` 8K arm runs.

**Go/kill for M1** (from the critique, with these numbers):
- **Go:**
  - KLD record identical (skip off), KV hashes identical (skip on, 82 … 16K tokens), replies identical;
  - 3,086 tokens in ≤ 16.0 s and 6,817 in ≤ 29 s;
  - NVMe busy ≥ 85% inside chunks (stats line);
  - decode on the P1 server within the predicted cost of its residents lost (ring + C 2048 arena, ≈ −2% to −2.5%; residents in the planner line × 0.054%) ± 1%.
- **Kill:** any byte difference stops the work.
- **Profile:** 3,086 tokens taking over 18 s means profile with `Lf` (lcpfast's pass profile) before building further.

## 10. Test plan and window arms

### 10.1 Without cards (both packages, `ROCR_VISIBLE_DEVICES=99`)

- K: `r4d_emu` test_ds41p1 (K-4), plus the existing `test_ds41tall` and `test_ds41vp` unchanged.
- S: `p1_stager_test` (S-5), the existing place/sched tests, and a `bash -n` / parse check of `ab_window.sh`.
- Integrator: the whole `ctest` set at `-j8` under the build lock, with the known environment-only failures as before.

### 10.2 Device checks in the window (before any arm; gate the P1 arms)

DEVICE_CHECKS = `lcp sweep p1` (+ the Vp and Rb self-checks the best parts need). Settings:
- `LCP_CHECK_C=2048`
- `P1_CHECK_M=4096`
- `P1_ROUTES=work/p0/routes-long3086.bin`
- `TALL_CHECK`/`TALL_LIBR4D` pointing at the integrated build.

A P1 arm runs only on a libr4d whose `p1` and `lcp 2048` checks passed (sha256 recorded).

### 10.3 Window arms (WINDOW-P1)

`BESTP` is today's prefill-best part string (the README's recommended block plus `Cs-Mr16`) without `Rb`, since `gemm_nt_f32bf16` is not in the sweep. Every Lc arm drops `Vp` unless the arm says `+Vp`.

| # | Arm (parts after BESTP) | Prompts | Measures |
|---|---|---|---|
| 1 | — (reference) | Ulong, Up1len | today's TTFT per prompt |
| 2 | `Lc2048-P1-B6` | Ulong, Up1len | **M1 TTFT**, stats (drive busy %, waves, waits) |
| 3 | `Lc2048-P1-B6-Vp` | Ulong, Up1len | packed residents beside P1 (≈ 31 more residents) |
| 4 | `Lc2048-P1-B6-Qa0` | Ulong | what the read-ahead is worth |
| 5 | `Lc2048-P1-B6-Qw1` and `-Qw4` | Ulong | wave size |
| 6 | `Lc2048-P1-B8` | Ulong | 16 slots against 12 |
| 7 | `Lc1024-P1-B6` | Ulong | chunk size (model: C 1024 ≈ 35% slower from 2K up) |
| 8 | `Lc2048-P1-B6-Qs` | Ulong | start against end alignment |
| 9 | `Kld` and `Lc2048-P1-B6-Kld` (Cs dropped from both) | the KLD corpus + long documents | **record directories byte-identical** |
| 10 | `Kh` and `Lc2048-P1-B6-Kh` (with Cs) | Ulong, Ulong8k, Up1len (to 16K, with `Lm` ≥ 17,408) | **prefill KV and logits hashes identical** |
| 11 | `Lc2048-P1-B2-Qw1-Qv64-Kh` | Ulong | ring stress (4 slots, waves of 1) with host verify: hashes identical, verify 0 differ |
| 12 | reference and `Lc2048-P1-B6` | Up1reply (256 tokens), held-out 8 | **replies identical**; decode tok/s after long prompts |
| 13 | reference and `Lc2048-P1-B6` | held-out 8 (decode) | decode cost of the reservation (ring + arena) |

**Ordering.** Run the exactness arms (9–11) first. If any differs, stop the window and keep the logs.

**Time estimate.** About 2–2.5 h. Each prefill arm takes ~4 min of startup plus ~1–2 min of prompts with P1 (~10 min for the reference). Kld arms take ~15 min each.

## 11. Risks and open items

| Risk | Effect | Handling |
|---|---|---|
| P0 run 2 not in (twin cut to local slots) | K has no measured baseline yet | K's device check times it. The model is flat up to 40 µs an expert-row. |
| vp2 kernel not exact or slow at tall heights | K-3 slips | Widening fallback (K-5). The plain twin plus skip marks with Pa and Vp off (`Lc2048-P1-B6` minus `Pa`) can still be measured. |
| Swiglu replay over all `rows*6` slots each wave | ~0.3 ms a wave, ~3–5 ms a layer, ~2% at 2K | S may cut it to the wave's covering slot span (still idempotent). The card is not the binding resource. |
| Copies slow cards 0/1's kernels by 40% (M1) | The MoE on cards 0/1 runs slower while RAM units stream | Not binding: the card's work is ≈ 40% of the drive's time a layer. Watch the stats' card-wait. |
| Host RAM: 1.8 GB HostPlanes + 0.45 GB pinned | swap next to production QFN (outside windows QFN holds ~108 GB) | P1 allocations happen only under the switch and only in windows. `.mem` files; INFLIGHT and AHEAD are tunable down. |
| C 2048 arena VRAM | decode −2 to −2.5% on P1 servers | Accepted until Phase 4. Measured by arm 13. |
| `--prefix-resume exact` beside end alignment | checkpoints off-grid | M1 uses start alignment there and logs it. |
| lcpfast and K both edit `r4d_ds41_mxfp4.hip` | merge conflict | K merges lcpfast first if committed; otherwise K keeps to the vp2 path and the integrator resolves. |
| Rare races in the ring | wrong bytes that real prompts may not hit | S-5 host stress plus arm 11 (2-slot-class ring, waves of 1, verify). |

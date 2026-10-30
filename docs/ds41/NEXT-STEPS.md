# DS41 decode: next steps after the night of 28–29 Oct

Where things stand: the final configuration (branch `claude/ds41-decode-impl`, HEAD `bbf48615`, settings in
`tools/ds41-host-experts/README.md`) decodes the eight held-out prompts at about 12.0 tok/s (base 5.03). TTFT summed
over them is about 19.4 s (base 64.0). Every reply matches the base engine's. A decode token costs about 84 ms:
roughly 35 ms waiting in the all-reduces for the slowest rank's drive misses and CPU experts, about 13 ms waiting for
CPU rows at the gathers, and the card's own work (dense GEMVs about 11 ms, routed GEMVs, attention, norms).

Ordered by expected value for the effort.

## 1. The way to 15+ tok/s is hardware: a second NVMe drive, or more and faster RAM

Decode is drive-bound for about 45% of every token. About 8% of decode layers on a rank miss the RAM cache, which is
roughly 12 misses and 235 MB a token over the four ranks. At the 990 Pro's ~6.4 GB/s that is ~37 ms, and the
all-reduces wait ~35 ms a token for exactly that. 15 tok/s (66.7 ms) needs about half of it gone. The other costs are
near their floors:

- **CPU kernels.** They stream ~94 GB/s against a practical DRAM ceiling of ~100–103 GB/s (`membw_pf.cpp`: plain,
  prefetching and multi-stream reads at 40–48 threads). That leaves ~5%, worth ~1% of decode.
- **Power settings.** Measured, no effect (section 2).

Options:

- **A second NVMe drive (cheapest; ~15 tok/s by this arithmetic).** Board slot `M2_1` is free (the 990 Pro is in
  `M2_2`; PCIE3/4/7 and both OCuLink ports also report free). Two ways to use it:
  - Put a byte-identical copy of `model.rad` (388 GB) on it, and have the engine split each miss read between the
    two copies. This is a generalisation of the plane mirror: same identity checks and fallback, any share up to
    half. It is a few hours of engine work, worth doing once the drive is ordered.
  - Stripe the two drives under the container (LVM or md). That needs no code, but the 990 Pro's volume group has
    only 20 GB free (it also holds `models_hot`, 304 GB), so it means reorganising storage.
- **8 × 32 GB DDR4-3200 RDIMM** (the slots hold 8 × 16 GB Samsung M393A2G40EB1-CPB at 2133; check the ROMED8-2T's
  support list). That gives +50% bandwidth for the CPU path and a ~200 GiB cache, most of the 230 GiB file tier.
  Decode misses would fall several-fold. The plan's estimate of 19–27 tok/s for this case still looks plausible.

## 2. Power settings: measured, no effect

Window ab-20261029T080054Z alternated the final configuration under default settings with the same configuration
under the performance governor, NVMe latency tolerance 0 and the C2 idle state off. Each setting was applied for its
own arm only and restored afterwards (`ab_window.sh` Q parts).

- Decode: 11.87 / 11.72 tok/s default, 11.99 / 11.75 with all three (+0.6%).
- TTFT: 19.57 / 19.35 s default, 19.26 / 19.32 s with all three.
- Replies: all 32 identical.

PCIe ASPM was already disabled on every GPU and NVMe link. Leave the system defaults.

## 3. Speculative decoding with the MTP head: estimated, and not a clear win here

Earlier measurements (DSpark depth 5: about 5.9 tokens verified, 3.8 accepted, at the cost of about 4.3 single steps)
said no, because more tokens per pass meant more experts to supply. Last night I expected the CPU path to change
that. This morning's numbers say it mostly does not:

- **Distinct experts.** Tonight's access trace (481 decode tokens over 8 prompts, 40 layers, 4 ranks) gives the
  file-tier experts that a verify pass of k consecutive tokens needs, summed over the ranks. That is 0.87 of k
  separate steps at k = 2, 0.76 at k = 4 and 0.70 at k = 6. Drive misses scale with these (`mtp_windows.py`,
  `mtp_windows.out.txt`). The CPU pool and the drive are shared by all ranks, so totals matter, not the slowest rank.
- **Rows.** The CPU pool's work grows with rows as well. One row of an expert is memory-bound: 0.201 ms an expert at
  40 threads, about 94 GB/s. Each further row of the same expert cost 0.49–0.60 of that with last night's kernel
  (`mrbench.cpp`, `mrbench.out.txt`), and costs 0.37–0.46 since `bbf48615` (section 4). The pool's work for a
  k-token pass is therefore 0.94 (k = 2) to 0.86 (k = 6) of k separate steps with the old kernel, and about 0.92 to
  0.83 now.
- **Only the card's own work amortizes.** That is about 36 ms of the 84 ms a token.
- **Prefill is a measured upper bound.** In the final configuration, passes of up to 16 prompt tokens cost 0.72–0.76
  of a decode step a token. They read the drive far more than decode does: 69% of prefill layers read the drive
  against 8% of decode layers. A verify pass would cost less a row than prefill does.

**Estimate, per component (revised 29 Oct morning).** My first estimate bounded the cost with prefill passes. Prompt
tokens miss the cache far more than the model's own continuation, so that bound was too pessimistic. Splitting a
decode token's 84 ms instead:

- **Card fixed part (~31 ms).** Dense GEMVs, attention, norms, all-reduce latency and the host's per-layer round
  trips. It barely grows with rows: ×1.05–1.1 for 2–4 rows.
- **Card routed part plus drive stalls (~40 ms).** Scales with distinct experts: 1.74 / 2.43 / 3.05 steps' worth
  for k = 2 / 3 / 4.
- **CPU-row wait (~13 ms).** Scales with the pool's cost: about 1.85 / 2.6 / 3.4 for k = 2 / 3 / 4, with the
  multi-row kernel.

That puts a k-row verify pass at about 1.50 / 1.95 / 2.38 single steps for k = 2 / 3 / 4. DSpark's depth-5 numbers
imply about 0.8 acceptance a position, which gives 1.8 / 2.5 / 3.0 tokens a pass at depth 1 / 2 / 3. Charging ~2 ms a
draft step for the head, that is **about 1.2× at depth 1 and 1.24× at depth 2–3**, or 1.1× if passes cost what the old
engine's DSpark measurement implies (its model underestimated it by ~12%). Roughly 13–15 tok/s. It is the largest
software lever left, but it depends on the bank's acceptance and on the engine work below. The prefill-rate bound
(0.86×) remains the pessimistic floor.

Prompt-lookup speculation (`--spec-ngram`) needs no draft model and runs through the same CPU path, so it was
measured directly (section 4).

If you build it anyway:

- **The bank.** The draft bank your loop is converting has to be in place first; the current container has no drafter
  tensors.
- **The CPU path and the head.** The CPU path turns itself off when the DSpark head is configured
  (`FileStager::hx_init_impl`), because the head tail owns the routed layers' file buffers. It needs to cooperate
  with the head instead. The head's own experts (layers ≥ 40) are not mapped-cache eligible
  (`Mover::mapped_file_eligible` requires layer < 40) and can stay on the card.
- **Pass size.** The file buffer cap assumes passes of at most 16 tokens, which holds for draft depth up to 15.
- **Cheaper extra rows** (section 4, now done) lower the CPU part from 0.86 to about 0.83 of k steps at k = 6. That
  moves these numbers by only a few percent.

## 4. Small software items

- **Done (`bbf48615`): decode each MXFP4 round once for all of a job's rows.** `HostExpertPool::tile` made one
  `gemv_avx2` call a row, so every extra row decoded the codes again. It now calls `ds41hx::gemv_avx2_rows`, which
  decodes each 32-block round once and adds it to each row in the same order.
  - Bits: identical. The kernel test compares 2–5 rows with `gemv_avx2` in both layouts on every case, and window
    ab-20261029T073100Z gave 16/16 replies identical to the base engine's.
  - Cost: an extra row went from 0.52–0.62 to 0.37–0.46 of a row (2 / 3 / 4-row jobs 10 / 15 / 17% cheaper; pool-shaped
    bench at 40 threads). One-row jobs are unchanged at 0.200 ms an expert.
  - Effect today: decode unchanged (12.11 / 12.00 tok/s). TTFT sums were 19.22 / 19.38 s, inside the 19.1–20.0 s
    earlier windows measured, so the expected ~2% does not show.
  - Converting the fp16 weights once as well (`cvt1` in `mrbench.cpp`) added nothing at 3–4 rows.
  - It matters for any multi-row decode later (MTP, batched sequences).
- Dropped: issuing the shared expert before the routed gather. That is already how ops are issued.
  - `MoeFP8::pass` issues the shared arm's gate_up and down right behind the routed ones and the gather last. So the
    shared expert runs on the card while the issuing thread waits for CPU rows in `FileStager::hx_finish`.
  - The profile's op numbers are declaration order, which lists the shared arm after the gather; that is what
    misled the plan.
  - Nothing else that is independent of the gather is worth moving: the n-gram lookups take 0.6 ms a step.
- The first request after startup decodes about 8% slower when async prefill misses are on (background inserts
  overlap its decode); seen once, not reproduced, and worth watching rather than fixing.
- Move the experimental environment switches to command-line flags before anything goes upstream (upstream removed
  its environment switches; the plan's P0.1 note). Done for the plane mirror (see below); the host-expert
  switches remain.
- **Plane mirror as flags (done, uncommitted at the time of writing).** `--expert-plane-mirror DIR` turns it on;
  nothing in the engine or the scripts assumes where the checkpoint is.
  - Prefill share: `--expert-plane-mirror-share` (0.18), for passes of `--expert-plane-mirror-min-rows` (2) or more.
  - Decode share: `--expert-plane-mirror-decode-share` (0, off). 0.10 measured +3% once and needs repeats.
- **Host pool: the planner reserved 1.84 GiB a rank of staging that nothing used.** It sized four slots by the widest
  per-layer unit, a dense layer of 437.86 MiB, although in expert-tiered placement with the dense units in VRAM
  only routed experts reach the file tier (and the mover allocates for those, ~203 MiB). Fixed: the reserve is sized
  from routed units in that case, and the plan is refused if a dense unit still lands on the file tier.
  - `--host-pool-mib 256` now starts and frees ~6.4 GB: headroom against other jobs on the box (your hermes job,
    ~2.3 GB, stopped one window), or cache.
  - The replay says +7% capacity is −7% decode misses.
- **Cache policy: LRU stays.** Replayed on the night trace against LRU:
  - one cache shared by the four ranks: −0.1%
  - prefill experts inserted at the LRU tail: +17%
  - TinyLFU admission: +66%
  - (last night) SLRU, 2Q and aged LFU: all worse

  Recency dominates, so capacity is the lever (`cache_sim2.py`).
- **All-reduce: already the fast form.** Decode messages go through a one-shot P2P exact all-reduce, and all four cards
  have peer access. The ~100 µs per call under `--profile-ops` is profiler syncs and skew.
- **The host's per-layer decision (~44 µs) delays the CPU's start, not the card.** In most layers the gather waits
  for CPU rows (~0.5 ms of pool work against ~0.2 ms of card work), so moving the shared expert into that gap would
  not help. What helps is a sooner CPU start: a faster `hx_choose`/`hx_submit`, and pool wake-up (workers spin
  200 µs, then sleep; layers come ~2 ms apart).

- **Done (`ee90326a`): miss reads on kept readers into 2 MiB-page buffers: +4.6%.** A miss read started its threads
  per call and read into 4 KiB-page buffers, so the drive got 1 MiB requests and the mirror a cold bounce buffer.
  Now up to 8 kept workers a rank read into 2 MiB-aligned THP buffers (lone 18.8 MB read 3.03 → 2.74 ms).
  12.67 / 12.64 → 13.23 / 13.24 tok/s, TTFT −3.6%; with warm buffers the decode mirror share 0.13 beats 0.10
  (+0.6%), so the configuration now uses 0.13 (~13.3 tok/s).
- **Done (uncommitted at the time of writing): a missed unit is relaid straight into a reserved slab slot: +1.1%.**
  `insert_unit_planes` relaid into scratch and then copied 18.8 MB into the cache under its writer lock; it now
  reserves the slot (`reserve_fill`), relays into it and publishes. 13.33 / 13.33 → 13.42 / 13.54 tok/s.
  `RADIANCE_EXPERIMENTAL_DS41_INSERT_DIRECT=0` restores the old path.
- **Measured and dropped: keeping the pool awake between bursts.** MWAITX for 1.5 ms after the 200 µs spin cut the
  counts-to-submit time from 33 to 8.7 µs a layer (no futex wakes) but decode fell 7% and 30% in two runs; a 2.5 ms
  spin had cost 8.5%. C2-off alone was neutral, so the loss is CPU occupancy: rank, reader and insert threads need
  free CPUs. Any pre-wake has to keep its awake window to tens of microseconds before the counts land.
- **Measured and dropped: rank threads polling their decode counts** (300 µs or 2 ms before the driver wait):
  ±0.6%, router waits unchanged.

## 5. Larger prefill chunks (your loop's macro-batch work)

The CPU path and the file buffer cap assume passes of at most 16 tokens (`kEpRouteMaxRows`). For long prompts,
chunks of 64–2,048 rows touch nearly every expert, and the card's copy path or full-layer streaming is the better
supplier. Combining the two needs the buffers back for those passes. One option is a separate prefill buffer lent
from the decode residents for the duration of a long prefill, which is the plan's P2.6 idea in reverse.

## Not worth more time on this hardware

- Next-layer read-ahead of predicted misses: precision 78% for the top pick, but a one-layer lead (about 2.6 ms) is
  shorter than a drive read, and wider predictions cost more drive time than they save.
- Cache policies other than LRU (replayed on a captured trace): none better.
- Pinning the CPU pool to cores: 3.5% slower than letting it float.
- Fewer or lower-priority insert threads: no measurable effect.

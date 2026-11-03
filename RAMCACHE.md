# radiance-ramcache

A fork of [radiance](https://codeberg.org/StillDeadcode/radiance) for serving DeepSeek-V4.1 Flash
(ds41) on four RX 9070 XT cards (gfx1201, TP4), with most routed experts in host RAM instead of
VRAM. The tree keeps upstream's layout and upstream's README.md, which describes the engine itself.
This file covers only what the fork adds.

## What the fork adds

- **The deepseek41 plugin** from upstream PR 266 (taken at bd65d4e5; the PR has since moved on and is not yet in upstream main):
  arch/deepseek41, libr4d/r4d_ds41_*, and the matching libref/libavx reference ops.
- **Expert RAM cache**: routed experts held in host RAM, either as mapped slabs or in an optional
  CPU-only packed form. The cache is filled from a decode profile at startup. Missed experts are
  read from the container, routed experts only (`--expert-ram-cache-total-mib`,
  `RADIANCE_EXPERIMENTAL_NATIVE_CACHE_*`, `..._DS41_PACKED_CACHE`, `..._DS41_CACHE_WARM`).
- **ds41 CPU experts** (host experts): a pool of CPU workers computes every routed expert the
  cards do not hold. The CPU kernels are bit-exact with libr4d's MXFP4 GEMV, so replies stay
  byte-identical to the base engine's (core/place/host_expert_pool.*, `FileStager::hx_*`).
- **Prefill**:
  - large-chunk prefill (`..._DS41_LCP`);
  - Phase 1 (`..._DS41_P1`): layer passes of 320 rows or more compute every routed expert on
    the cards;
  - the lent activation arena;
  - the consumer-suffix skip.
- **Prefix reuse for agent harnesses**: end-of-request snapshots and a 16-token resume grid.
- **Decode**:
  - packed VRAM residents, early issue of routed card ops, next-layer on-die prefetch;
  - DSpark speculative decoding (draft bank, acceptance gate, FP8 drafter, draft experts in
    VRAM);
  - concurrent requests (`RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE`).
- **Tools** (tools/ds41-host-experts): the CPU kernels and their card checks, the A/B window
  harness, and the reply checker.

Each of these is opt-in, through a `RADIANCE_EXPERIMENTAL_*` switch or a command-line flag.

## Recommended configuration

[tools/ds41-host-experts/README.md](tools/ds41-host-experts/README.md), section "Configuration".
It has the measured best configuration, the harness and concurrency variants, and the A/B window
behind each number.

## Plugin ABI

RadBatch carries a private tail after upstream's fields: the consumer suffix, the large-chunk
admit row and the per-row cut. `RAD_ABI_VERSION` is upstream's (33), but the ABI layout
fingerprint differs from upstream's, so the loader refuses plugins built from upstream's tree.
Build every plugin (libr4d, libref, libavx, libquant, deepseek41, spechub) from this tree and
install them together.

## Branches

- `main`: this tree, which is upstream radiance 1.16.0 merged into `ds41-stable`.
- `ds41-stable`: the stable engine line. It is upstream main b6acbd69 (1.15.0 + 12 commits),
  PR 266, and the work above; the README's recommended configuration refers to it.
- `ds41-research`: `ds41-stable`'s parent plus about 70 commits of default-off prototypes that are not in
  `main`: packed and growing KV, DSpark bank lending, FP4/FP8 storage candidates, and Phase 2
  preparation.
- `ds41`: curated snapshots of the ds41 documentation and A/B evidence, with the engine under
  radiance/. These are snapshots, not engine history.
- `dev/ds41-controls-current-20261028`, `dev/ds41-native-ramcache-20261028`: earlier engine
  points, both ancestors of `ds41-stable`.

## Status of the 1.16.0 merge

The merge of upstream 1.16.0 is verified only by CPU builds and host tests. GPU verification is
pending. Until a GPU window has run the card checks, a base-engine reply comparison against the
1.15 merge, and the byte gates of the recommended configuration, take measured results from
`ds41-stable`.

Upstream changes that reach ds41's path:
- libr4d's RMS norm now reads its lane sum eight at a time. Upstream states the bits are
  unchanged, since the partition and add order are the same and libr4d is built with
  `-ffp-contract=off`.
- `qsa_tail_store` now skips slot indices outside the state plane.
- Container, plugin and device checks are stricter at load.

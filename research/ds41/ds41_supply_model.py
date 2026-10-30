#!/usr/bin/env python3
"""Decode expert-supply model for DeepSeek V4.1 Flash (native MXFP4, EP4, 4x RX 9070 XT).

Reads one run's logged EXPERT_PLACEMENT lines and its held-out decode routing histogram, then
estimates the per-token time spent supplying non-resident experts under candidate changes.
This is a model, not a measurement: it leaves out GPU work other than supply.

Model v2 (this version): the NVMe drive is shared. A layer's misses, from every rank, are read
together and finish together at total bytes / drive bandwidth. v1 let each rank read its own
misses at full drive speed in parallel, which flattered layers with misses on several ranks.
"""
import json
import os
import random
import re
import statistics as st
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/as/radiance-setup/research/deepseek-v41-1017-20261024/"
    "ds41-pack-tail-dev-window-tail1-heldout-20261028T085400Z")
EXPERT = 12_533_760 + 6_266_880   # gate/up + down bytes of one native expert
LINK = 27e9                       # host-to-card bytes/s, per card
NVME = 6.5e9                      # bytes/s, one drive shared by all ranks
CPU_BW = float(os.environ.get("DS41_CPU_GBPS", "80")) * 1e9
                                  # bytes/s for CPU expert GEMV; membw.out.txt measured 68 GB/s of
                                  # plain reads with 16 threads and 79-83 GB/s with 20-24
HANDOFF_MS = 0.1                  # moving activations and results between card and host, per layer
LAYERS, RANKS, PER_RANK, TOP_K, N_EXPERT = 40, 4, 96, 6, 384

decode = json.load(open(f"{RUN}/expert-routing.json"))["counts"]["decode"]  # [layer][expert]
tier = {}
pattern = (r"EXPERT_PLACEMENT v=1 rank=(\d) layer=(\d+) static=1 ids=source-global "
           r"vram=\[([^\]]*)\] pinned=\[([^\]]*)\] ssd=\[([^\]]*)\]")
for m in re.finditer(pattern, open(f"{RUN}/model-server.log").read()):
    layer = int(m.group(2))
    for name, group in (("vram", 3), ("pinned", 4), ("ssd", 5)):
        for e in filter(None, m.group(group).split(",")):
            tier[(layer, int(e))] = name
assert len(tier) == LAYERS * N_EXPERT, len(tier)


def sample(rng, layer):
    """Six distinct experts for one token, drawn by held-out decode frequency (+0.5 smoothing)."""
    weights = [decode[layer][e] + 0.5 for e in range(N_EXPERT)]
    picked = set()
    while len(picked) < TOP_K:
        picked.add(rng.choices(range(N_EXPERT), weights=weights)[0])
    return [(e, tier[(layer, e)]) for e in picked]


draw = random.Random(7)
LAYER_SAMPLES = [sample(draw, draw.randrange(LAYERS)) for _ in range(3000)]


def classify(rng, chosen, hit, prefetch):
    """Per expert: 'vram', 'host' (pinned pool or RAM hit) or 'miss' (read from NVMe)."""
    out = []
    for e, t in chosen:
        if t == "vram":
            out.append((e, "vram"))
        elif t == "pinned" or rng.random() < hit or rng.random() < prefetch:
            out.append((e, "host"))
        else:
            out.append((e, "miss"))
    return out


def layer_ms(rng, chosen, hit=0.78, cpu_ms=2.0, prefetch=0.0, mode="ep"):
    """One layer's supply time on the critical path, in ms.

    ep        today: each expert is supplied by its owner rank (copy to the card; a miss also
              waits for the shared drive, then cpu_ms of repack and cache copy)
    sliced    every rank holds a quarter of every expert
    balanced  host experts copied by the least-loaded rank instead of the owner
    cpu       host experts computed on the CPU; misses read straight into cache memory
    """
    kinds = classify(rng, chosen, hit, prefetch)
    misses = sum(1 for _, k in kinds if k == "miss")
    drive_ms = 1e3 * misses * EXPERT / NVME      # all of the layer's misses, shared drive
    copy_ms = 1e3 * EXPERT / LINK
    if mode == "cpu":
        hosts = sum(1 for _, k in kinds if k == "host")
        if not hosts and not misses:
            return 0.0
        cpu_hits = 1e3 * hosts * EXPERT / CPU_BW
        return max(cpu_hits, drive_ms) + 1e3 * misses * EXPERT / CPU_BW + HANDOFF_MS
    if mode == "hybrid":
        # CPU and card give the same bits, so each RAM-resident expert goes to whichever is free
        # sooner: the shared CPU pool or its owner rank's copy link. Misses go to the CPU, which
        # reads the release's planes directly and needs no repack.
        cpu_ms_used = 1e3 * misses * EXPERT / CPU_BW
        link = [0.0] * RANKS
        for e, k in kinds:
            if k != "host":
                continue
            r = e // PER_RANK
            cpu_cost = 1e3 * EXPERT / CPU_BW
            if link[r] + copy_ms <= cpu_ms_used + cpu_cost:
                link[r] += copy_ms
            else:
                cpu_ms_used += cpu_cost
        if not any(link) and not cpu_ms_used:
            return 0.0
        return max(max(link), max(cpu_ms_used - 1e3 * misses * EXPERT / CPU_BW, drive_ms)
                   + 1e3 * misses * EXPERT / CPU_BW) + HANDOFF_MS
    if mode == "sliced":
        off_card = sum(1 for _, k in kinds if k != "vram")
        if not off_card:
            return 0.0
        return (drive_ms + cpu_ms / 4 if misses else 0.0) + off_card * copy_ms / 4
    owner = {}
    if mode == "balanced":
        load = [0.0] * RANKS
        for _, k in sorted(kinds, key=lambda x: x[1] != "miss"):   # spread misses first
            if k == "vram":
                continue
            r = min(range(RANKS), key=load.__getitem__)
            load[r] += copy_ms + (cpu_ms if k == "miss" else 0.0)
            owner.setdefault(r, []).append(k)
    else:
        for e, k in kinds:
            if k != "vram":
                owner.setdefault(e // PER_RANK, []).append(k)
    worst = 0.0
    for ks in owner.values():
        has_miss = "miss" in ks
        worst = max(worst, (drive_ms + cpu_ms if has_miss else 0.0) + len(ks) * copy_ms)
    return worst


def per_token_ms(**kw):
    rng = random.Random(11)
    return LAYERS * st.mean(layer_ms(rng, s, **kw) for s in LAYER_SAMPLES)


total = sum(map(sum, decode))
share = {name: 0 for name in ("vram", "pinned", "ssd")}
for layer in range(LAYERS):
    for e in range(N_EXPERT):
        share[tier[(layer, e)]] += decode[layer][e]
print("held-out decode selections by tier:",
      ", ".join(f"{k} {v / total:.1%}" for k, v in share.items()))
busiest = st.mean(max(sum(1 for e, t in s if e // PER_RANK == r and t != "vram") for r in range(RANKS))
                  for s in LAYER_SAMPLES)
print(f"non-VRAM experts per layer: busiest rank {busiest:.2f}, "
      f"mean per rank {(1 - share['vram'] / total) * TOP_K / RANKS:.2f}")

# Calibration: the 64 -> 80 GiB slab runs moved the whole-run hit rate 0.706 -> 0.771 and
# decode 205.5 -> 191.9 ms/token (+7.1%). Non-supply time is taken from the 80 GiB run.
small, large = per_token_ms(hit=0.706), per_token_ms(hit=0.771)
other = 191.9 - large
predicted = (small + other) / (large + other) - 1
scale = 0.071 / predicted
print(f"calibration: model predicts {predicted:+.1%} for the 64 -> 80 GiB step, measured +7.1%; "
      f"scaled estimates multiply the model's gain by {scale:.2f}")

base = per_token_ms()
print(f"\n{'scenario':54s} {'supply ms/token':>15s} {'model':>7s} {'scaled':>7s}")
for label, kw in (
        ("current", {}),
        ("#1 CPU off the miss path (0.3 ms left)", dict(cpu_ms=0.3)),
        ("#3 prefetch hides 25% of misses", dict(prefetch=0.25)),
        ("#3 prefetch hides 50% of misses", dict(prefetch=0.5)),
        ("RAM hit rate 0.78 -> 0.85", dict(hit=0.85)),
        ("#5 TP-sliced experts", dict(mode="sliced")),
        ("host experts rebalanced across ranks", dict(mode="balanced")),
        ("host experts computed on the CPU", dict(mode="cpu")),
        ("host experts on CPU or copy link, whichever is free", dict(mode="hybrid")),
        ("#1 + #3 (50%)", dict(cpu_ms=0.3, prefetch=0.5)),
        ("CPU experts + #3 (50%)", dict(mode="cpu", prefetch=0.5)),
        ("hybrid + #3 (50%)", dict(mode="hybrid", prefetch=0.5))):
    value = per_token_ms(**kw)
    gain = base / value - 1
    print(f"{label:54s} {value:15.1f} {gain:+7.0%} {gain * scale:+7.0%}")

# Prefetch is bounded by NVMe bandwidth: every useful read also brings wasted ones.
MISS_GB = 0.5   # GB per decoded token read from NVMe today (estimate)
print("\nNVMe ceiling on decode, tokens/s = drive GB/s / (miss GB per token x read amplification):")
for drives, bw in (("one 990 Pro", 7.0), ("two drives striped", 14.0)):
    print(f"  {drives:20s}" + "".join(f"   x{a:.1f}: {bw / (MISS_GB * a):5.1f}" for a in (1.0, 1.5, 2.0, 3.0)))

print("\nidle decode VRAM lent to the next-hottest experts (held-out data, so an upper bound):")
for extra in (100, 170, 250):
    moved = 0
    for rank in range(RANKS):
        candidates = sorted((decode[l][e] for (l, e), t in tier.items()
                             if e // PER_RANK == rank and t != "vram"), reverse=True)
        moved += sum(candidates[:extra])
    print(f"  +{extra} experts per card: {moved / total:.1%} of decode selections move to VRAM")

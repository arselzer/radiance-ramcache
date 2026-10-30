#!/usr/bin/env python3
"""chain_report.py PREFIX [--layer N] [--top K]: the card's per-layer dispatch chain from a chain
trace (RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE, arm part Ct; core/runtime/live_chain.h).

PREFIX.ops.csv has one line per device launch of every traced pass (rank 0): its layer, op,
kernel, queue and packet indices, whether its first dispatch went without the barrier bit
(`unordered`) or would have with the overlap on (`would`), why it waited otherwise (`why`, `clash`
the launch it depended on), its dependences on earlier launches (`deps`: SEQ followed by r = it
reads what that one wrote, a = it writes what that one read, w = it writes what that one wrote)
and the buffers it reads and writes. PREFIX.stamps.csv has one line per kernel dispatched on card 0
in those passes, with its start and end on the card; a kernel no launch owns is the issue path's
own work (a copy, a fill, a weight-table patch, a CPU-row blit).

Per layer the report takes the stretch the routing-counts sync waits for: from the layer's first
launch to its router's scatter. It prints, per kind of layer (plain, indexer, source, Engram), the
median over traced passes of: launches, unordered (and would-be) first dispatches, why the others
waited, the stretch's wall time on the card, the busy time (union of kernels), the summed gaps
before ordered starts, and the critical chain. Then the fusion candidates: dependent neighbours on
one queue whose second launch is short and reads nothing but what the first wrote plus tables and
positions (`epilogue`), and GEMVs that read the same input (`shared input`, which needs each
segment's split count kept, since a wider N picks another split and changes the bits).

--layer N prints every launch of layer N in the first complete traced pass. --selftest runs on
synthetic files. Standard library only.
"""
import csv
import os
import statistics
import sys
import tempfile
from collections import defaultdict

SYNC_OPS = ("moe_scatter", "moe_topk_scatter", "router_gemm_topk_scatter", "sigmoid_scatter")
# Inputs that are not another launch's activation: an epilogue can read them as well.
SIDE_INPUTS = ("cs", "raw", "pos", "kv:", "wtab")


def load(prefix):
    ops = defaultdict(list)
    with open(prefix + ".ops.csv", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("seq", "layer", "lane", "queue", "idx", "packets", "unordered", "would",
                      "clash", "foreign"):
                r[k] = int(r[k])
            r["pass"] = int(r["pass"])
            r["deps_l"] = parse_deps(r["deps"])
            r["reads_l"] = [x for x in r["reads"].split("|") if x]
            r["writes_l"] = [x for x in r["writes"].split("|") if x]
            ops[r["pass"]].append(r)
    stamps = {}
    path = prefix + ".stamps.csv"
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r["start_ns"] or not r["end_ns"]:
                    continue
                stamps[(int(r["pass"]), int(r["queue"]), int(r["idx"]))] = (
                    r["kernel"], r["ordered"] == "1", int(r["start_ns"]), int(r["end_ns"]))
    return ops, stamps


def parse_deps(s):
    out = []
    if not s or s == "opaque":
        return out
    for tok in s.split():
        n = "".join(c for c in tok if c.isdigit())
        out.append((int(n), tok[len(n):]))
    return out


def attach_times(ops, stamps):
    """Each launch's start, end and own busy time from its packets' stamps; None when missing."""
    owned = set()
    for p, rows in ops.items():
        for r in rows:
            iv = []
            for i in range(r["packets"]):
                k = (p, r["queue"], r["idx"] + i)
                if k in stamps:
                    iv.append(stamps[k])
                    owned.add(k)
            r["t"] = (min(x[2] for x in iv), max(x[3] for x in iv), sum(x[3] - x[2] for x in iv),
                      iv) if iv and len(iv) == r["packets"] else None
    foreign = defaultdict(list)
    for k, v in stamps.items():
        if k not in owned:
            foreign[k[0]].append((k[1], k[2]) + v)
    return foreign


def union(iv):
    tot, a0, b0 = 0, None, None
    for a, b in sorted(iv):
        if b0 is None or a > b0:
            if b0 is not None:
                tot += b0 - a0
            a0, b0 = a, b
        elif b > b0:
            b0 = b
    if b0 is not None:
        tot += b0 - a0
    return tot


def layer_kind(rows):
    ops = {r["op"] for r in rows}
    k = "plain"
    if "index_score" in ops:
        k = "indexer"
    if "softmax_pool" in ops or "entry_store" in ops:
        k = "source"
    if "embed_lookup_planes" in ops or "embed_lookup_ggml" in ops:
        k += "+engram"
    return k


def stretches(rows):
    """Per layer: the launches from its first to its router's scatter (inclusive)."""
    by_layer = defaultdict(list)
    for r in rows:
        if r["layer"] >= 0:
            by_layer[r["layer"]].append(r)
    out = {}
    for l, rs in by_layer.items():
        rs.sort(key=lambda r: r["seq"])
        # From the layer's first launch on lane 0: the next layer's weights read ahead on lane 1
        # (DS41's pf_attn_tail) carry its number but run inside this layer's MoE.
        start = next((i for i, r in enumerate(rs) if r["lane"] == 0), None)
        end = next((i for i, r in enumerate(rs) if r["op"] in SYNC_OPS), None)
        if start is None or end is None or end < start:
            continue
        out[l] = (rs[start: end + 1], rs)
    return out


def prepare(rows):
    """Per launch: the latest-ending earlier launch on its queue (what an ordered start waits for),
    in one walk over the pass in issue order."""
    last = {}
    for r in rows:
        r["prev_q"] = last.get(r["queue"])
        if r["t"] is not None:
            cur = last.get(r["queue"])
            if cur is None or r["t"][1] > cur["t"][1]:
                last[r["queue"]] = r


def analyse(seg, all_rows):
    """Figures for one layer's stretch; times are None without complete stamps."""
    f = {"n": len(seg), "unordered": sum(r["unordered"] for r in seg),
         "would": sum(r["would"] for r in seg), "why": defaultdict(int)}
    for r in seg:
        if not r["would"]:
            f["why"][r["why"]] += 1
    if any(r["t"] is None for r in seg):
        return f
    iv = [(x[2], x[3]) for r in seg for x in r["t"][3]]
    t0 = min(a for a, _ in iv)
    f["wall"] = max(b for _, b in iv) - t0
    f["busy"] = union(iv)
    # Gap before each ordered start: its start minus the latest end of anything earlier on its
    # queue (any launch of the pass, the previous layer's included).
    gaps = [max(0, r["t"][0] - r["prev_q"]["t"][1]) for r in seg
            if not r["unordered"] and r["prev_q"] is not None]
    f["gaps"] = sum(gaps)
    # The critical chain, walked back from the scatter: each launch's latest-ending dependence
    # (or, for an ordered start, the latest-ending launch before it on its queue).
    by_seq = {r["seq"]: r for r in all_rows}
    chain, cur = [], seg[-1]
    while cur is not None and cur["t"] is not None:
        chain.append(cur)
        cands = [by_seq[s] for s, _ in cur["deps_l"] if s in by_seq and by_seq[s]["t"]]
        if not cur["unordered"] and cur["prev_q"] is not None:
            cands.append(cur["prev_q"])
        cands = [c for c in cands if c["seq"] >= seg[0]["seq"] and c["seq"] < cur["seq"]]
        cur = max(cands, key=lambda c: c["t"][1]) if cands else None
    f["chain"] = list(reversed(chain))
    return f


def name(r):
    lab = r["label"]
    if lab.startswith("blk."):
        lab = lab.split(".", 2)[2].replace(".weight", "")
    return "%s(%s)" % (r["op"], lab)


EPILOGUE_MAX_NS = 15000   # a row-wise follow-up (rope, norm, round trip, top-k, store), not a GEMM


def fusion_candidates(seg):
    """`epilogue`: a short row-wise launch B whose one producer in the stretch is A, on A's queue,
    reading nothing else but tables, positions and caches, and with no launch between them reading
    what B rewrites in place (that reader would see B's output once B moves into A).
    `shared input`: two GEMMs reading one buffer, neither reading what the other writes."""
    out = []
    by_seq = {r["seq"]: r for r in seg}
    for b in seg:
        if not b["t"] or b["op"].startswith(("gemm", "moe_gemm", "attn", "logits")) or \
                b["t"][2] > EPILOGUE_MAX_NS:
            continue
        prods = [by_seq[s] for s, k in b["deps_l"] if "r" in k and s in by_seq]
        if len(prods) != 1:
            continue
        a = prods[0]
        if a["queue"] != b["queue"] or not a["t"]:
            continue
        side = [x for x in b["reads_l"] if x not in a["writes_l"] and not x.startswith(SIDE_INPUTS)
                and x != "scratch"]
        if side:
            continue
        inplace = set(b["writes_l"]) & set(a["writes_l"]) - {"scratch"}
        between = [r for r in seg if a["seq"] < r["seq"] < b["seq"]]
        if inplace and any(inplace & set(r["reads_l"]) for r in between):
            continue
        dur = b["t"][2]
        gap = max(0, b["t"][0] - a["t"][1])
        # A follow-up that reduces over the whole row (a norm, a top-k) cannot run in a GEMV's
        # per-tile epilogue: it needs a last-block finisher that keeps the reduction's order.
        kind = "epilogue*" if any(w in b["op"] for w in ("norm", "topk", "mix")) else "epilogue"
        out.append((kind, name(a), name(b), dur, gap))
    gemv = [r for r in seg if r["op"].startswith("gemm") and r["t"]]
    for i, a in enumerate(gemv):
        for b in gemv[i + 1:]:
            shared = set(a["reads_l"]) & set(b["reads_l"]) - {"scratch"}
            if shared and not set(a["writes_l"]) & set(b["reads_l"]):
                out.append(("shared input", name(a), name(b), b["t"][2], 0))
    return out


def report(prefix, layer=None, top=12):
    ops, stamps = load(prefix)
    foreign = attach_times(ops, stamps)
    by_kind = defaultdict(list)
    cands = defaultdict(list)
    first = None
    for p in sorted(ops):
        rows = sorted(ops[p], key=lambda r: r["seq"])
        prepare(rows)
        for l, (seg, full) in sorted(stretches(rows).items()):
            k = layer_kind(full)
            f = analyse(seg, rows)
            by_kind[k].append(f)
            for c in fusion_candidates(seg):
                cands[(k,) + c[:3]].append(c[3:])
            if layer is not None and l == layer and first is None and "wall" in f:
                first = (p, seg, rows)
    print("%s: %d traced passes, %d launches, %d stamped kernels (%d not a launch's)" % (
        prefix, len(ops), sum(len(v) for v in ops.values()), len(stamps),
        sum(len(v) for v in foreign.values())))
    med = statistics.median
    for k, fs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        print("\n%s layers (%d stretches to the routing-counts sync)" % (k, len(fs)))
        print("  launches                         %.0f" % med(f["n"] for f in fs))
        print("  first dispatches unordered       %.1f (check frees %.1f)" % (
            med(f["unordered"] for f in fs), med(f["would"] for f in fs)))
        why = defaultdict(list)
        for f in fs:
            for w in set(list(f["why"]) + ["clash", "foreign", "opaque", "host", "after", "cap"]):
                why[w].append(f["why"].get(w, 0))
        print("  waited: " + ", ".join("%s %.1f" % (w, med(v)) for w, v in sorted(why.items())
                                       if med(v) > 0))
        timed = [f for f in fs if "wall" in f]
        if not timed:
            print("  (no complete stamps)")
            continue
        print("  wall on the card                 %.1f us" % (med(f["wall"] for f in timed) / 1e3))
        print("  busy (union of kernels)          %.1f us" % (med(f["busy"] for f in timed) / 1e3))
        print("  gaps before ordered starts       %.1f us" % (med(f["gaps"] for f in timed) / 1e3))
        ch = timed[len(timed) // 2]["chain"]
        print("  critical chain (%d launches): %s" % (len(ch), " > ".join(name(r) for r in ch)))
    if cands:
        print("\nfusion candidates (median us over occurrences; the saving is at most the gap plus a "
              "launch, the second kernel's own work stays unless its epilogue is cheaper; "
              "epilogue* reduces over a whole row: a last-block finisher keeping the order)")
        rows = []
        for key, v in cands.items():
            rows.append((len(v) * (med(x[1] for x in v) + 2.5), key, len(v), med(x[0] for x in v),
                         med(x[1] for x in v)))
        rows.sort(reverse=True)
        for score, (k, kind, a, b), n, dur, gap in rows[:top]:
            print("  %-12s %-9s %-40s -> %-40s x%-4d %6.1f us, gap %5.1f us" % (
                kind, k, a[:40], b[:40], n, dur / 1e3, gap / 1e3))
    if first:
        p, seg, rows = first
        t0 = seg[0]["t"][0]
        print("\nlayer %d of pass %d" % (layer, p))
        for r in seg:
            print("  %4d %-44s q%-3d %s%s start %8.1f end %8.1f  %-8s deps %s" % (
                r["seq"], name(r)[:44], r["queue"], "U" if r["unordered"] else "O",
                "w" if r["would"] else " ", (r["t"][0] - t0) / 1e3, (r["t"][1] - t0) / 1e3,
                r["why"], r["deps"]))


def selftest():
    d = tempfile.mkdtemp()
    pre = os.path.join(d, "t")
    with open(pre + ".ops.csv", "w") as f, open(pre + ".stamps.csv", "w") as s:
        f.write("pass,seq,layer,op,label,kernel,lane,queue,idx,packets,unordered,would,why,clash,"
                "foreign,deps,reads,writes\n")
        s.write("pass,queue,idx,kernel,ordered,chained,start_ns,end_ns\n")
        # (op, label, reads, writes, unordered, deps, dur_us)
        lay = [("mhc_post_collapse_rmsnorm", "blk.4.attn_norm.weight", "act|h1", "h0|xn", 0, "", 11),
               ("gemm_nt", "blk.4.attn_q_a.weight", "xn", "qr", 0, "0r", 30),
               ("gemm_nt", "blk.4.attn_kv.weight", "xn", "kv", 1, "", 15),
               ("rmsnorm", "blk.4.attn_q_a_norm.weight", "qr", "qrn", 0, "1r", 7),
               ("rope_rows", "-", "q|cs|raw", "q", 0, "3r", 7),
               ("moe_scatter", "-", "ids", "sorted", 0, "4r", 7)]
        t, idx = 1000000, 500
        for i, (op, lab, rd, wr, u, deps, dur) in enumerate(lay):
            start = t if not u else t - 25000
            f.write("1,%d,4,%s,%s,k%d,0,3,%d,1,%d,%d,%s,-1,0,%s,%s,%s\n" % (
                i, op, lab, i, idx, u, u, "free" if u else "clash", deps, rd, wr))
            s.write("1,3,%d,k%d,%d,0,%d,%d\n" % (idx, i, 1 - u, start, start + dur * 1000))
            t = start + dur * 1000 + 3000
            idx += 1
    report(pre, layer=4)
    print("\nselftest ran")


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--selftest":
        selftest()
        return
    layer, top = None, 12
    if "--layer" in a:
        i = a.index("--layer")
        layer = int(a[i + 1])
        del a[i:i + 2]
    if "--top" in a:
        i = a.index("--top")
        top = int(a[i + 1])
        del a[i:i + 2]
    report(a[0], layer, top)


if __name__ == "__main__":
    main()

"""DSpark draft-bank hot sets for RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM (core/place/draft_vram.h).

The capture is --expert-stats (core/place/xstats.h) in a DSpark run with the bank unmasked (ab_window.sh's
Xd part on a Ds arm): its `counts.draft` rows 40..42 are each stage's routed selections over the
stage's 128 experts by model-wide id, every query pass's rows (5 a pass, top-3 each), counted from the
router's own histograms. At EP4 rank r holds experts 32r .. 32r+31 of every stage (the deepseek41
plugin's split of 128 over 4 ranks; the engine checks each listed id against the container's
directory and refuses one the rank does not hold).

  draft_hotset.py write OUT.json XSTATS.json [XSTATS.json ...] [--top N]
      Sums the captures' draft counts and writes the hot-set file: for each stage and rank its experts
      by count, hottest first (ties to the lower id), the first N (default 32: every expert ranked, so
      one file serves every DvN; the engine keeps the first DvN of each list). Refuses a capture that
      is not a DSpark run's (format, 43 layers with 128 experts and top-3 at 40..42, ids from the
      container's directory, no draft passes) or a stage with no selections.
  draft_hotset.py show HOTSET.json [N]
      Each stage and rank's first N (default 8) ids and the share of the stage's sources' selections
      they hold.
  draft_hotset.py cover HOTSET.json XSTATS.json [N ...]
      The share of a capture's draft selections the first N of each list hold (N default 8 16 24), by
      stage and over all: held-out coverage when the capture is not one the hot set was written from.
  draft_hotset.py summary XSTATS.json
      A capture's draft passes, query-pass histograms and selections by stage, and how many experts
      a rank each stage's selections touched; exit 2 when it is not a DSpark run's or holds no draft
      pass (ab_window.sh's Xd check).
  draft_hotset.py check HOTSET.json N XSTATS.json
      A capture taken under the mask (a DvN arm with Xd): every draft selection is one of the first N
      of its rank's list. Exit 1, naming the experts, otherwise.
"""
import hashlib
import json
import sys

FORMAT = "radiance-ds41-draft-hotset/1"
XSTATS = "radiance-expert-stats/1"
LAYERS = (40, 41, 42)
RANKS = 4
EXPERTS = 128
PER_RANK = EXPERTS // RANKS
TOP_K = 3


class Refused(Exception):
    pass


def owner(g):
    """The rank holding model-wide expert g (the plugin's contiguous split)."""
    return g // PER_RANK


def load_capture(path):
    """A DSpark run's --expert-stats file: its draft counts at 40..42 and what they stand for."""
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict) or d.get("format") != XSTATS:
        raise Refused("%s: not a %s file" % (path, XSTATS))
    le, tk, how = d.get("layer_experts") or [], d.get("top_k") or [], d.get("map") or []
    if d.get("n_trunk_layers") != 40 or d.get("n_layers") != 43 or len(le) != 43:
        raise Refused("%s: not a DeepSeek-V4.1 run with the DSpark drafter (n_layers %s, trunk %s)" % (
            path, d.get("n_layers"), d.get("n_trunk_layers")))
    for L in LAYERS:
        if le[L] != EXPERTS or (len(tk) > L and tk[L] != TOP_K):
            raise Refused("%s: layer %d holds %s experts, top-%s; a DSpark stage holds 128, top-3" % (
                path, L, le[L], tk[L] if len(tk) > L else "?"))
        if len(how) > L and how[L] != "container":
            raise Refused("%s: layer %d's ids come from %r, not the container's directory" % (path, L, how[L]))
    rows = (d.get("counts") or {}).get("draft")
    if not isinstance(rows, list) or len(rows) != 43:
        raise Refused("%s: no draft counts" % path)
    counts = {}
    for L in LAYERS:
        r = rows[L]
        if not isinstance(r, list) or len(r) != EXPERTS or any(not isinstance(c, int) or c < 0 for c in r):
            raise Refused("%s: layer %d's draft counts are not 128 counts" % (path, L))
        counts[L] = r
    hist = (d.get("histograms") or {}).get("draft") or []
    passes = (d.get("steps") or {}).get("draft", 0)
    if not passes or not any(sum(counts[L]) for L in LAYERS):
        raise Refused("%s: no draft pass was counted (a run without DSpark drafting?)" % path)
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    dropped = sum((d.get("ranks") or {}).get("dropped") or [0])
    return dict(path=path, md5=md5, counts=counts, passes=passes,
                histograms=[hist[L] if len(hist) > L else None for L in LAYERS],
                selections=[sum(counts[L]) for L in LAYERS], dropped=dropped)


def rank_lists(counts, top):
    """[stage][rank]: the rank's experts by count, hottest first, ties to the lower id; the first top."""
    hot = []
    for L in LAYERS:
        c = counts[L]
        per = []
        for r in range(RANKS):
            ids = list(range(r * PER_RANK, (r + 1) * PER_RANK))
            ids.sort(key=lambda g: (-c[g], g))
            per.append(ids[:top])
        hot.append(per)
    return hot


def write(out, paths, top=PER_RANK):
    if not 1 <= top <= PER_RANK:
        raise Refused("--top takes 1..%d" % PER_RANK)
    caps = [load_capture(p) for p in paths]
    counts = {L: [sum(c["counts"][L][g] for c in caps) for g in range(EXPERTS)] for L in LAYERS}
    for L in LAYERS:
        if not sum(counts[L]):
            raise Refused("stage %d: no selections in the captures" % L)
    doc = {
        "format": FORMAT,
        "layers": list(LAYERS),
        "ranks": RANKS,
        "experts": EXPERTS,
        "top": top,
        "hot": rank_lists(counts, top),
        "counts": [counts[L] for L in LAYERS],
        "sources": [{k: c[k] for k in ("path", "md5", "passes", "histograms", "selections", "dropped")}
                    for c in caps],
    }
    text = json.dumps(doc, indent=1)
    with open(out, "w") as f:
        f.write(text + "\n")
    return doc


def load_hotset(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("format") != FORMAT or d.get("layers") != list(LAYERS) or d.get("ranks") != RANKS or \
            d.get("experts") != EXPERTS:
        raise Refused("%s: not a %s file at the bank's geometry" % (path, FORMAT))
    hot = d.get("hot")
    if not isinstance(hot, list) or len(hot) != len(LAYERS) or any(len(s) != RANKS for s in hot):
        raise Refused("%s: \"hot\" must hold one list a rank for each stage" % path)
    for s, L in enumerate(LAYERS):
        for r in range(RANKS):
            l = hot[s][r]
            if not l or len(set(l)) != len(l) or any(not isinstance(g, int) or owner(g) != r for g in l):
                raise Refused("%s: stage %d rank %d: a list of distinct experts the rank holds" % (path, L, r))
    return d


def kept(hot, n):
    """[stage]: the set of experts the first n of every rank's list keep."""
    if any(len(l) < n for s in hot for l in s):
        raise Refused("a list is shorter than %d" % n)
    return [set(g for l in s for g in l[:n]) for s in hot]


def show(path, n=8):
    d = load_hotset(path)
    counts = d.get("counts")
    for s, L in enumerate(LAYERS):
        tot = sum(counts[s]) if counts else 0
        for r in range(RANKS):
            l = d["hot"][s][r][:n]
            share = sum(counts[s][g] for g in l) / tot if tot else float("nan")
            print("stage %d rank %d: %s  (%.1f%% of the stage's selections)" % (
                L, r, ",".join(str(g) for g in l), 100.0 * share))
    return 0


def cover(path, cap, ns=(8, 16, 24)):
    d = load_hotset(path)
    c = load_capture(cap)
    print("%s against %s (%s query passes):" % (path, cap, c["histograms"][0]))
    for n in ns:
        k = kept(d["hot"], n)
        cells, a, t = [], 0, 0
        for s, L in enumerate(LAYERS):
            row = c["counts"][L]
            inn, tot = sum(row[g] for g in k[s]), sum(row)
            a, t = a + inn, t + tot
            cells.append("stage %d %.1f%%" % (L, 100.0 * inn / tot if tot else float("nan")))
        print("  N=%2d (%3d of %d a stage): %s; all %.1f%% of %d selections" % (
            n, n * RANKS, EXPERTS, ", ".join(cells), 100.0 * a / t if t else float("nan"), t))
    return 0


def summary(cap):
    c = load_capture(cap)
    cells = []
    for s, L in enumerate(LAYERS):
        row = c["counts"][L]
        touched = [sum(1 for g in range(r * PER_RANK, (r + 1) * PER_RANK) if row[g]) for r in range(RANKS)]
        cells.append("stage %d %d selections, experts touched a rank %s" % (L, sum(row), "/".join(map(str, touched))))
    print("capture %s: %s draft passes, %s query-pass histograms a stage; %s%s" % (
        cap, c["passes"], "/".join(str(h) for h in c["histograms"]), "; ".join(cells),
        "; %d histograms lost by the ring" % c["dropped"] if c["dropped"] else ""))
    return 0


def check(path, n, cap):
    d = load_hotset(path)
    c = load_capture(cap)
    k = kept(d["hot"], n)
    bad = 0
    for s, L in enumerate(LAYERS):
        out = {g: x for g, x in enumerate(c["counts"][L]) if x and g not in k[s]}
        if out:
            bad += 1
            print("stage %d: %d selections of experts outside the first %d: %s" % (
                L, sum(out.values()), n, ", ".join("%d (rank %d) x%d" % (g, owner(g), x) for g, x in sorted(out.items()))))
        else:
            print("stage %d: all %d selections among the %d kept experts" % (L, sum(c["counts"][L]), len(k[s])))
    print("mask check %s at N=%d over %s: %s" % (path, n, cap, "PASS" if not bad else "FAIL"))
    return 1 if bad else 0


def main(a):
    try:
        if len(a) >= 3 and a[0] == "write":
            top = PER_RANK
            if "--top" in a:
                i = a.index("--top")
                top = int(a[i + 1])
                a = a[:i] + a[i + 2:]
            d = write(a[1], a[2:], top)
            print("wrote %s: top %d of each stage and rank from %d capture(s) (%s query passes)" % (
                a[1], top, len(d["sources"]), "+".join(str(s["histograms"][0]) for s in d["sources"])))
            return 0
        if len(a) in (2, 3) and a[0] == "show":
            return show(a[1], int(a[2]) if len(a) == 3 else 8)
        if len(a) >= 3 and a[0] == "cover":
            return cover(a[1], a[2], tuple(int(x) for x in a[3:]) or (8, 16, 24))
        if len(a) == 2 and a[0] == "summary":
            return summary(a[1])
        if len(a) == 4 and a[0] == "check":
            return check(a[1], int(a[2]), a[3])
    except Refused as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

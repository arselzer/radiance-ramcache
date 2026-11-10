"""A placement profile for --expert-profile from more than one workload (core/place/native_decode_profile.h).

The planner keeps on the cards, and then in the RAM cache, the experts the profile's `decode` counts call
hottest. A profile recorded on one family of prompts places another family badly: prompts of different kinds
route to different experts. This tool sums decode counts of several --expert-stats captures, each times a
whole-number weight, into one profile. Placement only: which experts a row routes to and every unit's bytes
are untouched, so replies are the base engine's whatever profile is given.

  profile_blend.py write OUT.json W:BASE.json [W:STATS.json ...]
      BASE is a sealed decode profile (an expert-stats/1 file with `source_identity`, accepted by
      --expert-profile on this container); its identity, metadata and non-decode sections are kept. Every
      other file is a capture (--expert-stats FILE, written at shutdown; ab_window.sh's Xst part) or
      another profile of the same container. W is a whole number from 1 to 1000: the file's decode
      counts are taken W times, so a file of T decode tokens weighs W x T tokens in the result. Refuses a
      file that is not a complete pure-decode record of the same model geometry (format, final, model,
      arch, tp, container, layers, experts, top-k, map; no mixed steps; no unmapped, unclassified or
      dropped routing; per layer the counts sum to top-k x tokens) and an OUT that exists.
  profile_blend.py show FILE.json [FILE.json ...]
      Each file's decode tokens and steps, and for each pair of files the overlap of their hottest
      experts: the share of file A's decode selections that the top N experts of file B hold, N the
      number a rank keeps on a card in the current plan (default 715 of 3840, --top N to change),
      ranked within each rank's 96 experts a layer. 1.00 on the diagonal is the in-sample bound.

Weights: a capture of a few hundred tokens is noisy, so weigh families by what will be served, not by how
many tokens each capture happens to hold; `show` prints the tokens to work from.
"""
import hashlib
import json
import os
import sys

FORMAT = "radiance-expert-stats/1"
NL, NE, TOPK, RANKS = 40, 384, 6, 4
PER = NE // RANKS
GEOMETRY = ("format", "model", "arch", "tp", "container", "n_layers", "n_trunk_layers", "n_experts",
            "layer_experts", "top_k", "map")
MAX_WEIGHT = 1000


class Refused(Exception):
    pass


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise Refused("%s: not JSON (%s)" % (path, e))
    if not isinstance(doc, dict):
        raise Refused("%s: not an expert-stats object" % path)
    return doc, hashlib.sha256(raw).hexdigest()


def check(doc, path):
    """A complete pure-decode record of a 40-layer, 384-expert, top-6 trunk; returns its decode tokens."""
    def need(ok, what):
        if not ok:
            raise Refused("%s: %s" % (path, what))
    try:
        need(doc.get("format") == FORMAT, "format is not %s" % FORMAT)
        need(doc.get("final") is True, "not final (written before shutdown)")
        need(doc.get("arch") == "deepseek41" and doc.get("tp") == 4 and doc.get("n_trunk_layers") == NL
             and doc.get("n_experts") == NE, "not a DeepSeek-V4.1 EP4 record (arch, tp, trunk layers, experts)")
        need(doc.get("mixed_decode_tokens", 0) == 0 and doc["tokens"].get("mixed", 0) == 0,
             "mixed steps: their decode rows are not separable from prefill rows")
        for kind in ("credited", "unmapped", "unclassified", "dropped"):
            row = doc["ranks"][kind]
            need(len(row) == RANKS and all((n > 0) if kind == "credited" else (n == 0) for n in row),
                 "ranks.%s: lost, unmapped or unclassified routing" % kind)
        tokens = doc["tokens"]["decode"]
        need(isinstance(tokens, int) and tokens > 0 and doc["steps"]["decode"] > 0, "no decode tokens")
        counts = doc["counts"]["decode"]
        for key in ("layer_experts", "top_k", "map"):
            need(len(doc[key]) >= NL, "%s is short" % key)
        for l in range(NL):
            need(doc["layer_experts"][l] == NE and doc["top_k"][l] == TOPK and doc["map"][l] == "container",
                 "layer %d: not 384 experts, top-6, ids from the container's directory" % l)
            row = counts[l]
            need(len(row) == NE and all(isinstance(c, int) and 0 <= c <= tokens for c in row),
                 "layer %d: counts are not 384 whole numbers up to the decode tokens" % l)
            need(sum(row) == TOPK * tokens == doc["selections"]["decode"][l]
                 and doc["observed_tokens"]["decode"][l] == tokens,
                 "layer %d: counts do not sum to top-k x decode tokens" % l)
            need(0 < doc["histograms"]["decode"][l] <= doc["passes"]["decode"][l],
                 "layer %d: no histogram, or more histograms than passes" % l)
    except (KeyError, IndexError, TypeError) as e:
        raise Refused("%s: incomplete record (%r)" % (path, e))
    return tokens


def parse_weighted(arg):
    w, sep, path = arg.partition(":")
    if not sep or not w.isdigit() or not 1 <= int(w) <= MAX_WEIGHT or not path:
        raise Refused("%s: give W:FILE with W a whole number from 1 to %d" % (arg, MAX_WEIGHT))
    return int(w), path


def write(out, args):
    if len(args) < 1:
        raise Refused("write: give OUT.json W:BASE.json [W:STATS.json ...]")
    if os.path.exists(out):
        raise Refused("%s exists" % out)
    parts = []
    for a in args:
        w, path = parse_weighted(a)
        doc, sha = load(path)
        parts.append((w, path, doc, sha, check(doc, path)))
    base = parts[0][2]
    if not isinstance(base.get("source_identity"), dict):
        raise Refused("%s: the first file must be a sealed profile (it has no source_identity)" % parts[0][1])
    for w, path, doc, sha, tokens in parts[1:]:
        for key in GEOMETRY:
            if doc.get(key) != base.get(key):
                raise Refused("%s: %s differs from the base profile's" % (path, key))
        ident = doc.get("source_identity")
        if ident is not None and ident != base["source_identity"]:
            raise Refused("%s: sealed for another container generation than the base profile" % path)
    total = sum(w * t for w, _, _, _, t in parts)
    result = json.loads(json.dumps(base))
    for key in ("counts",):
        result[key]["decode"] = [[sum(w * d[key]["decode"][l][e] for w, _, d, _, _ in parts) for e in range(NE)]
                                 for l in range(NL)] + result[key]["decode"][NL:]
    for key in ("observed_tokens", "selections", "histograms", "passes"):
        result[key]["decode"] = [sum(w * d[key]["decode"][l] for w, _, d, _, _ in parts)
                                 for l in range(NL)] + result[key]["decode"][NL:]
    result["tokens"]["decode"] = total
    result["steps"]["decode"] = sum(w * d["steps"]["decode"] for w, _, d, _, _ in parts)
    result["blend"] = [{"weight": w, "file": os.path.basename(path), "sha256": sha, "decode_tokens": t}
                       for w, path, _, sha, t in parts]
    result["note"] = ("profile_blend.py: decode counts of %d files, each times its weight (blend); every other "
                      "section and the source identity are the first file's" % len(parts))
    check(result, out)
    with open(out, "x") as f:
        json.dump(result, f)
    print("wrote %s: %d decode tokens from %s" % (
        out, total, ", ".join("%d x %s (%d)" % (w, os.path.basename(p), t) for w, p, _, _, t in parts)))


def top_units(doc, top):
    """Per rank, its `top` hottest (layer, expert) units by decode count (ties: lower expert, lower layer)."""
    counts = doc["counts"]["decode"]
    kept = set()
    for r in range(RANKS):
        units = sorted(((l, PER * r + e) for l in range(NL) for e in range(PER)),
                       key=lambda u: (-counts[u[0]][u[1]], u[1], u[0]))
        kept.update(units[:top])
    return kept


def show(paths, top):
    docs = []
    for path in paths:
        doc, _ = load(path)
        tokens = check(doc, path)
        docs.append((os.path.basename(path), doc, tokens))
        print("%s: %d decode tokens in %d steps%s" % (
            docs[-1][0], tokens, doc["steps"]["decode"],
            ", sealed" if isinstance(doc.get("source_identity"), dict) else ""))
    if len(docs) < 2:
        return
    kept = [top_units(d, top) for _, d, _ in docs]
    print("share of the row file's decode selections on the column file's hottest %d units a rank:" % top)
    print("%-28s %s" % ("", " ".join("%8s" % ("[%d]" % j) for j in range(len(docs)))))
    for i, (name, doc, tokens) in enumerate(docs):
        counts = doc["counts"]["decode"]
        row = []
        for j in range(len(docs)):
            held = sum(counts[l][e] for l, e in kept[j])
            row.append("%8.3f" % (held / (TOPK * NL * tokens)))
        print("[%d] %-24s %s" % (i, name[:24], " ".join(row)))


def main(argv):
    try:
        if len(argv) >= 3 and argv[1] == "write":
            write(argv[2], argv[3:])
        elif len(argv) >= 3 and argv[1] == "show":
            args, top = argv[2:], 715
            if "--top" in args:
                i = args.index("--top")
                top = int(args[i + 1])
                args = args[:i] + args[i + 2:]
                if not 1 <= top <= NL * PER:
                    raise Refused("--top: 1 to %d units a rank" % (NL * PER))
            show(args, top)
        else:
            print(__doc__)
            return 2
    except Refused as e:
        print("profile_blend.py: refused: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

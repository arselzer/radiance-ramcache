#!/usr/bin/env python3
"""kl_docs.py REF CAND: like kl_across.py, but over the documents both --kld-record directories hold (matched by
source and tokens; their tokens compared), for corpora that differ in which documents they kept."""
import json
import sys

import numpy as np

ref, cand = sys.argv[1], sys.argv[2]
a, b = json.load(open(ref + "/kld.json")), json.load(open(cand + "/kld.json"))
V = a["n_vocab"]
assert V == b["n_vocab"]
ta = np.fromfile(ref + "/tokens.i32", dtype=np.int32)
tb = np.fromfile(cand + "/tokens.i32", dtype=np.int32)
fa = np.memmap(ref + "/logp.f16", dtype=np.float16, mode="r", shape=(a["rows"], V))
fb = np.memmap(cand + "/logp.f16", dtype=np.float16, mode="r", shape=(b["rows"], V))


def spans(m):   # each document's token offset in tokens.i32 and its scored rows (to the next row0), in record order
    out, off, ds = {}, 0, m["docs"]
    for i, d in enumerate(ds):
        end = ds[i + 1]["row0"] if i + 1 < len(ds) else m["rows"]
        out[(d.get("source", ""), d["tokens"])] = (d, off, end - d["row0"])
        off += d["tokens"]
    return out


sa, sb = spans(a), spans(b)
kls, same, top1, n, docs = [], 0, 0, 0, 0
for key, (da, oa, ra) in sa.items():
    if key not in sb:
        continue
    db, ob, rb = sb[key]
    if da["score_from"] != db["score_from"] or not np.array_equal(ta[oa:oa + da["tokens"]], tb[ob:ob + db["tokens"]]):
        print("skip %s: tokens differ" % key[0])
        continue
    rows = ra
    assert ra == rb, key
    x = np.asarray(fa[da["row0"]:da["row0"] + rows])
    y = np.asarray(fb[db["row0"]:db["row0"] + rows])
    same += int((x.view(np.uint16) == y.view(np.uint16)).all(1).sum())
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    p = np.exp(xf)
    with np.errstate(invalid="ignore"):
        kl = np.where(p > 0, p * (xf - yf), 0.0).sum(1)
    kls.append(kl)
    top1 += int((xf.argmax(1) == yf.argmax(1)).sum())
    n += rows
    docs += 1
k = np.concatenate(kls)
print("%d positions over %d shared documents: %d byte-identical rows, top-1 agreement %.4f" % (n, docs, same, top1 / n))
print("KL(ref || cand): mean %.3g, median %.3g, p99 %.3g, max %.3g" % (k.mean(), np.median(k), np.percentile(k, 99), k.max()))

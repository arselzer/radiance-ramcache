#!/usr/bin/env python3
"""dspark_conf_extract.py CHECKPOINT_DIR OUT.bin -- DSpark's confidence head for the engine's host-side
controller (RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF_HEAD).

The reference (inference/model.py DSparkConfidenceHead) scores block row i as

    conf[i] = proj . [x_i, markov_embed(output_ids[i])]      (fp32, no bias)

with x_i the last stage's hc collapse (before its norm), markov_embed the Markov head's embedding table
(mtp.{last}.markov_head.embed.weight, [vocab, rank]) at the row's input token, and proj
(mtp.{last}.confidence_head.proj.weight, [1, dim + rank], stored bf16). Neither tensor is in the draft
bank, and the second half only ever meets the table, so this writes the first half as is and the second
folded through the table, one score a token:

    header  8 bytes "DSCONF1\\0", then int32 dim, rank, vocab, 0
    w       f32[dim]     proj[0, :dim]
    mscore  f32[vocab]   sum_r proj[0, dim + r] * embed[t, r], accumulated in f64

so the engine computes conf = w . x + mscore[token] and never reads the table. Reads the two tensors'
bytes only (about 66 MB). Prints the tensors' sha256 and the output's.
"""
import hashlib, json, os, struct, sys
import numpy as np


def headers(d):
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        yield p, 8 + n, h


def find(d, names):
    got = {}
    for p, base, h in headers(d):
        for k in names:
            if k in h and k not in got:
                got[k] = (p, base, h[k])
        if len(got) == len(names):
            break
    return got


def read_bf16(p, base, info):
    if info["dtype"] != "BF16":
        raise SystemExit(f"{p}: expected BF16, got {info['dtype']}")
    a, b = info["data_offsets"]
    with open(p, "rb") as fh:
        fh.seek(base + a)
        raw = fh.read(b - a)
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(info["shape"]), hashlib.sha256(raw).hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    d, out = sys.argv[1], sys.argv[2]
    last = None
    for p, base, h in headers(d):
        for k in h:
            if k.startswith("mtp.") and k.endswith(".confidence_head.proj.weight"):
                last = int(k.split(".")[1])
        if last is not None:
            break
    if last is None:
        raise SystemExit(f"{d}: no mtp.*.confidence_head.proj.weight")
    kp = f"mtp.{last}.confidence_head.proj.weight"
    ke = f"mtp.{last}.markov_head.embed.weight"
    got = find(d, [kp, ke])
    if len(got) != 2:
        raise SystemExit(f"{d}: missing {[k for k in (kp, ke) if k not in got]}")
    proj, sp = read_bf16(*got[kp])
    emb, se = read_bf16(*got[ke])
    vocab, rank = emb.shape
    dim = proj.shape[1] - rank
    if proj.shape[0] != 1 or dim <= 0:
        raise SystemExit(f"{kp}: shape {proj.shape} does not fit {ke}'s {emb.shape}")
    w = proj[0, :dim].astype(np.float32)
    wm = proj[0, dim:].astype(np.float64)
    mscore = (emb.astype(np.float64) @ wm).astype(np.float32)
    blob = b"DSCONF1\0" + struct.pack("<4i", dim, rank, vocab, 0) + w.tobytes() + mscore.tobytes()
    with open(out, "wb") as fh:
        fh.write(blob)
    print(f"{kp} sha256 {sp}")
    print(f"{ke} sha256 {se}")
    print(f"{out}: dim {dim}, rank {rank}, vocab {vocab}; |w| {np.linalg.norm(w):.4f}, "
          f"mscore mean {mscore.mean():.4f} sd {mscore.std():.4f} [{mscore.min():.3f}, {mscore.max():.3f}]; "
          f"sha256 {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()

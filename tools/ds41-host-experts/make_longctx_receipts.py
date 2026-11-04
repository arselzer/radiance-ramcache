"""The long-context prompt receipts for ab_window.sh (U parts Ufill32k, Ulong48k, Ulong100k) and the
prefill-hash references (<name>-khref.json) the window holds Kh arms to.

  make_longctx_receipts.py OUTDIR      (run with a python that has the `tokenizers` package, e.g.
                                        /home/as/venvs/embed/bin/python -I; writes OUTDIR/*.json)

fill32k   the 32,768-token prompt of the only filled long-context run so far (M16 on the stable e184
          build, filled32k-common0118-base-window2: all-ranks prefill hash 71eaab9a74338840) and its
          82-token recovery request (112530d7d4d5b367), byte for byte the reference's request bodies:
          the 6,817 ids the server made of long8k-receipt.json's chat_text (BOS, <|User|>, the text,
          <|Assistant|>, </think>), five times over, cut at 32,768; 64 tokens out, then the first 82
          ids with 8 out. Its khref holds both prompts' prefill hashes (every rank's groups, the
          window layers) and replies, so a Kh arm is checked against the reference in the window.
long48k   "Summarize the following text in two sentences." and perldiag.pod's paragraphs (Perl's own
          documentation, GPL-1+ or Artistic: /usr/share/doc/perl-modules-5.40/copyright) up to 48,000
          prompt tokens; 128 tokens out.
long100k  the same head, all of perldiag.pod and then the licence texts of /usr/share/common-licenses
          in name order, up to 100,000 prompt tokens; 128 tokens out.

The texts have their tabs expanded (8-column stops) and form feeds dropped, so every character is
printable ASCII, a space or a newline. Token counts are the official DeepSeek-V4.1 tokenizer's
(tokenizer.json, sha256 c90dfa01...) through the `tokenizers` package, which first has to give,
token for token, the 6,817 ids the server made of long8k's chat_text (else nothing is written).
A receipt's `max_context_tokens` (its longest request's prompt plus max_tokens) lets ab_window.sh
refuse an arm whose --max-model-len is shorter and give its requests a longer timeout; `provenance`
records every source's path, size and sha256. long8k-khref.json holds the 6,817-token prompt's
reference (8c583a997ef6f1db in every Kh run of it since ab-20261101T062359Z) for an Ulong8k anchor.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_requests  # noqa: E402  (hash_log: the window's own parser of the prefill hash lines)

R = "/home/as/radiance-setup/research/deepseek-v41-1017-20261024"
W = "/home/as/ds41-claude-build/work"
AB = "/home/as/radiance-ds41-claude/tools/ds41-host-experts/build"
FILL = W + "/autonomous-20261101/filled32k-common0118-base-window2/inner"
SEED_ARM = ("all-t40-f-M16-H-p256-m196608-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-"
            "Rb-Vp-Cs-Kh-Lm8192-Ulong8k")
DEFAULTS = {
    "tokenizer": R + "/ds41-native-official-tokenizer-metadata-20261027T092900Z/tokenizer.json",
    "seed_json": AB + "/ab-20261101T062359Z/" + SEED_ARM + ".json",
    "seed_log": AB + "/ab-20261101T062359Z/" + SEED_ARM + ".server.log",
    "fixture": W + "/autonomous-20261101/filled32k-common0118-fixture-v2/fixture.json",
    "role": FILL + "/approved_e184.role.json",
    "reference_log": FILL + "/approved_e184.server.log",
    "long8k": W + "/extra-prompts/long8k-receipt.json",
    "perldiag": "/usr/share/perl/5.40.1/pod/perldiag.pod",
    "licences": "/usr/share/common-licenses",
}
# Pinned inputs: the official tokenizer and the 32K reference's files, as their receipts name them.
PINNED = {
    "tokenizer": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "seed_json": "4ce2089b4fd5d9d462a8a58256c15188aa599e968a4b8f810e3121fa48b7016e",
    "fixture": "15ca0db10b8134de0b2e123bf5a6935368f4627389971e35575a0c1a2d586ce5",
    "reference_log": "acbbbe6e8aa858e9378f756beaf7af0f744b83070d421777c4727f098df9e502",
}
REFERENCE_ALL = [(32768, "71eaab9a74338840"), (82, "112530d7d4d5b367")]
LONG8K_ALL = [(6817, "8c583a997ef6f1db")]
BOS, USER, ASSISTANT, END_THINK = 0, 128803, 128804, 128822
HEAD = "Summarize the following text in two sentences.\n\n"
MODEL = "ds41-native-smoke"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_ids(ids):
    """The sha256 of the ids as compact JSON ([0,128803,...]), the digest the fill receipts quote."""
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def source(p):
    return {"path": os.path.realpath(p), "bytes": os.path.getsize(p), "sha256": sha256_file(p)}


def die(msg):
    sys.exit("make_longctx_receipts: " + msg)


def clean(t):
    """Tabs expanded, form feeds dropped: printable ASCII, spaces and newlines only."""
    t = t.expandtabs(8).replace("\f", "")
    bad = sorted({c for c in t if not (c == "\n" or " " <= c <= "~")})
    if bad:
        die("characters outside printable ASCII after cleaning: %r" % bad[:10])
    return t


def frame(enc, text):
    return [BOS, USER] + enc(text) + [ASSISTANT, END_THINK]


def chat_doc(enc, paras, target):
    """The most leading paragraphs whose framed prompt (head + paragraphs joined by blank lines) holds at
    most `target` tokens, by the tokenizer's count of the whole text (a binary search)."""
    lo, hi = 1, len(paras)

    def n(k):
        return len(frame(enc, HEAD + "\n\n".join(paras[:k])))
    if n(lo) > target:
        die("the first paragraph alone exceeds %d tokens" % target)
    if n(hi) <= target:
        die("the source holds only %d tokens, short of %d" % (n(hi), target))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n(mid) <= target:
            lo = mid
        else:
            hi = mid
    text = HEAD + "\n\n".join(paras[:lo])
    return text, lo, len(frame(enc, text))


def hash_records(log, want, what):
    """The prefill hash records of a server log (ab_requests.hash_log), checked against (tokens, digest)."""
    recs = ab_requests.hash_log(log)
    got = [(r["tokens"], r["all"]) for r in recs]
    if got != want:
        die("%s: prefill hashes %s, expected %s" % (what, got, want))
    return recs


def khref(name, receipt_path, recs, texts, evidence, note):
    prompts = []
    for r, (rname, text) in zip(recs, texts):
        prompts.append({"name": rname, "tokens": r["tokens"], "all": r["all"], "ranks": r["ranks"],
                        "layers": r["layers"], "cut": r["cut"], "text": text})
    return {"schema": "ds41-prefill-hash-reference-v1", "receipt": os.path.basename(receipt_path),
            "receipt_sha256": sha256_file(receipt_path), "prompts": prompts, "evidence": evidence,
            "note": note}


def write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    a = ap.parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError:
        die("needs the `tokenizers` package (e.g. /home/as/venvs/embed/bin/python -I)")
    for k, want in PINNED.items():
        got = sha256_file(getattr(a, k))
        if got != want:
            die("%s %s: sha256 %s, pinned %s" % (k, getattr(a, k), got, want))
    tk = Tokenizer.from_file(a.tokenizer)
    for tok, i in (("<｜begin▁of▁sentence｜>", BOS), ("<｜User｜>", USER), ("<｜Assistant｜>", ASSISTANT),
                   ("</think>", END_THINK)):
        if tk.token_to_id(tok) != i:
            die("the tokenizer's %s is %s, not %d" % (tok, tk.token_to_id(tok), i))

    def enc(s):
        return tk.encode(s, add_special_tokens=False).ids
    out = a.out
    os.makedirs(out, exist_ok=True)
    tokenizer = dict(source(a.tokenizer), library="tokenizers %s" % __import__("tokenizers").__version__)

    # ---- the calibration: the server's ids for long8k's chat_text, captured by the seed arm
    long8k = a.long8k
    l8 = json.load(open(long8k))["benchmark_requests"]
    seed_rows = json.load(open(a.seed_json))["requests"]
    if len(l8) != 1 or len(seed_rows) != 1 or seed_rows[0]["name"] != l8[0]["name"]:
        die("the seed arm's requests are not long8k's")
    seed = seed_rows[0]["prompt_ids"]
    mine = frame(enc, l8[0]["request"]["chat_text"])
    if mine != seed:
        k = next((i for i, (x, y) in enumerate(zip(mine, seed)) if x != y), min(len(mine), len(seed)))
        die("the tokenizer's long8k prompt differs from the server's at id %d (%d against %d ids)" %
            (k, len(mine), len(seed)))
    print("calibration: the tokenizer gives the server's %d ids for long8k's chat_text, id for id" % len(seed))

    # ---- fill32k: the reference's two request bodies, checked against the fixture and the seed
    fx = json.load(open(a.fixture))
    role = json.load(open(a.role))
    if fx.get("seed_ids") != seed:
        die("the fixture's seed ids are not the seed arm's prompt ids")
    want_fill = (seed * 5)[:32768]
    reqs = []
    for i, (f_req, r_req) in enumerate(zip(fx["requests"], role["requests"])):
        if f_req["name"] != r_req["name"] or f_req["request"] != r_req["body"]:
            die("fixture request %d is not the request the reference sent" % i)
        reqs.append({"name": f_req["name"], "request": f_req["request"]})
    if len(reqs) != 2 or len(fx["requests"]) != 2 or len(role["requests"]) != 2:
        die("the reference sent %d requests, not 2" % len(role["requests"]))
    if reqs[0]["request"]["prompt"] != want_fill or reqs[0]["request"]["max_tokens"] != 64:
        die("the fill request is not long8k's ids x5 cut at 32,768 with 64 out")
    if reqs[1]["request"]["prompt"] != seed[:82] or reqs[1]["request"]["max_tokens"] != 8:
        die("the recovery request is not long8k's first 82 ids with 8 out")
    recs = hash_records(a.reference_log, REFERENCE_ALL, "the reference's server log")
    for r_req, (n, digest) in zip(role["requests"], REFERENCE_ALL):
        if r_req["kh"]["all"] != digest or int(r_req["prompt_tokens"]) != n:
            die("the reference's role record for %s does not name %s" % (r_req["name"], digest))
    fill_path = os.path.join(out, "fill32k-receipt.json")
    write(fill_path, {
        "benchmark_requests": reqs,
        "max_context_tokens": max(len(q["request"]["prompt"]) + q["request"]["max_tokens"] for q in reqs),
        "provenance": {
            "what": "the filled 32K reference's requests, byte for byte (synthetic: long8k's server ids "
                    "repeated; no natural-text or quality claim)",
            "derivation": fx["derivation"],
            "requests": [{"name": q["name"], "prompt_tokens": len(q["request"]["prompt"]),
                          "max_tokens": q["request"]["max_tokens"], "ids_sha256": sha256_ids(q["request"]["prompt"])}
                         for q in reqs],
            "seed": {"ids": len(seed), "ids_sha256": sha256_ids(seed), "arm_json": source(a.seed_json),
                     "receipt": source(long8k),
                     "tokenizer_check": "the official tokenizer gives these ids for the receipt's chat_text"},
            "reference": {"fixture": source(a.fixture), "role": source(a.role), "server_log": source(a.reference_log),
                          "all_ranks": {q["name"]: d for q, (_, d) in zip(reqs, REFERENCE_ALL)},
                          "config": "M16 (16-token passes), --max-model-len 32832, stable runtime e184c6b7"},
            "tokenizer": tokenizer,
        },
    })
    texts = [(r_req["name"], r_req["text"]) for r_req in role["requests"]]
    write(os.path.join(out, "fill32k-khref.json"), khref(
        "fill32k", fill_path, recs, texts, {"server_log": source(a.reference_log), "role": source(a.role)},
        "filled32k-common0118-base-window2: M16 on the stable e184 build at --max-model-len 32832 "
        "(fixed KV); the fixed-KV fa9 run (filled32k-fa9-window4) has the same two digests"))

    # ---- long8k's reference (no change to its receipt)
    seed_recs = hash_records(a.seed_log, LONG8K_ALL, "the seed arm's server log")
    write(os.path.join(out, "long8k-khref.json"), khref(
        "long8k", long8k, seed_recs, [(seed_rows[0]["name"], seed_rows[0]["text"])],
        {"server_log": source(a.seed_log), "arm_json": source(a.seed_json)},
        "ab-20261101T062359Z M16 at --max-model-len 8192; the same digest in every Kh run of it since "
        "(M16 and Lc2048 + P1, Lm8192 and Lm8448: ab-20261103T110235Z, T114720Z, T221104Z)"))

    # ---- long48k and long100k: chat_text, the server tokenizes them (as long8k and long16k)
    # Paragraphs (blank-line separated, empty ones dropped), each with the file it came from.
    srcs = [a.perldiag] + [os.path.join(a.licences, n) for n in sorted(os.listdir(a.licences))
                           if os.path.isfile(os.path.join(a.licences, n))
                           and not os.path.islink(os.path.join(a.licences, n))]
    paras = []
    for i, s in enumerate(srcs):
        t = clean(open(s, encoding="ascii").read())
        paras += [(p, i) for p in (t if i == 0 else t.strip()).split("\n\n") if p.strip()]
    n_perl = sum(1 for _, i in paras if i == 0)
    for name, use, target in (("long48k", n_perl, 48000), ("long100k", len(paras), 100000)):
        text, k, n = chat_doc(enc, [p for p, _ in paras[:use]], target)
        req = {"max_tokens": 128, "model": MODEL, "temperature": 0, "stream": False, "chat_text": text}
        rname = {"long48k": "long-perldiag-48k", "long100k": "long-perldiag-licences-100k"}[name]
        used = sorted({i for _, i in paras[:k]})
        write(os.path.join(out, name + "-receipt.json"), {
            "benchmark_requests": [{"name": rname, "request": req}],
            "max_context_tokens": n + req["max_tokens"],
            "provenance": {
                "what": "the head and the first %d paragraphs (blank-line separated, empty ones dropped; tabs "
                        "expanded, form feeds dropped) of %s" % (
                            k, "perldiag.pod" if len(used) == 1 else
                            "all of perldiag.pod (%d) and then the licence texts in name order" % n_perl),
                "requests": [{"name": rname, "prompt_tokens": n, "text_tokens": n - 4, "chars": len(text),
                              "text_sha256": sha256_text(text), "max_tokens": req["max_tokens"]}],
                "framing": "BOS <|User|> text <|Assistant|> </think> (ab_requests.py chat_text: 4 ids)",
                "sources": [source(srcs[i]) for i in used],
                "licence": "perldiag.pod: GPL-1+ or Artistic (/usr/share/doc/perl-modules-5.40/copyright); "
                           "the licence texts are the licences themselves (/usr/share/common-licenses)",
                "tokenizer": tokenizer,
            },
        })
        print("%s: %d paragraphs, %d characters, %d prompt tokens (+%d out), text sha256 %s" % (
            name, k, len(text), n, req["max_tokens"], sha256_text(text)[:16]))
    for name in ("fill32k-receipt", "fill32k-khref", "long8k-khref", "long48k-receipt", "long100k-receipt"):
        p = os.path.join(out, name + ".json")
        print("%s  %s  %d bytes" % (sha256_file(p), p, os.path.getsize(p)))


if __name__ == "__main__":
    main()

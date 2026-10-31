"""Builds the P1 prompt receipts (work/extra-prompts/p1len-receipt.json, p1reply-receipt.json) from
license texts on disk that the long, long8k and long16k receipts do not use. Characters a token for
this text, from long8k (34,331 characters, ~7K tokens): ~4.9; the window's first arm measures the real
counts (the TTFT lines carry prompt tokens)."""
import json, re, sys

L = "/usr/share/common-licenses/"
def text(*names):
    out = []
    for n in names:
        p = n if n.startswith("/") else L + n
        t = open(p, encoding="utf-8", errors="replace").read()
        out.append(re.sub(r"[ \t]+", " ", t).strip())
    return "\n\n".join(out)

CPT = 4.9
def cut(t, tokens):
    n = int(tokens * CPT)
    if len(t) < n:
        sys.exit(f"text of {len(t)} characters is short of {n}")
    t = t[:n]
    sp = t.rfind(" ")
    return t[:sp] if sp > n - 200 else t

docs = [
    ("p1len-1k", cut(text("CC0-1.0", "Artistic"), 1000)),
    ("p1len-2k", cut(text("MPL-2.0"), 2000)),
    ("p1len-5k", cut(text("LGPL-2.1"), 5000)),
    ("p1len-8k", cut(text("GFDL-1.3", "GPL-2"), 8000)),
    ("p1len-16k", cut(text("/usr/lib/python3.14/LICENSE.txt", "MPL-1.1", "LGPL-2", "GFDL-1.2", "GPL-1"), 16000)),
]
def req(name, body, max_tokens):
    return {"name": name, "request": {"max_tokens": max_tokens, "model": "ds41-native-smoke", "temperature": 0,
            "stream": False, "chat_text": "Summarize the following text in two sentences.\n\n" + body}}
out = sys.argv[1]
json.dump({"benchmark_requests": [req(n, b, 16) for n, b in docs]},
          open(out + "/p1len-receipt.json", "w"), indent=1)
json.dump({"benchmark_requests": [req(n.replace("p1len", "p1reply"), b, 256) for n, b in docs if n in ("p1len-2k", "p1len-5k")]},
          open(out + "/p1reply-receipt.json", "w"), indent=1)
for n, b in docs:
    print(n, len(b), "characters, ~%d tokens" % (len(b) / CPT))

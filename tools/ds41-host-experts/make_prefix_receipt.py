"""Writes the prefix-cache receipt: requests that share long prefixes, run in order on one server,
so a window can compare each reply with a cold server's byte for byte (the arm without Pc) and
read each request's TTFT and cached prompt tokens (ab_requests.py --ttft).

  python3 -I make_prefix_receipt.py OUT.json [MPL-2.0 text] [prefix.cpp]

Families, in the order sent (each request's reply is capped at MAX_TOKENS):
  F  a short chat prompt (about 25 tokens) and its exact repeat: under any snapshot interval past
     the prompt there is nothing to resume from (no tail snapshot inside the first interval).
  A  a license text (MPL-2.0 sections 2 and 3, about 1,100 tokens) with four different questions,
     then the first again: the RAG/agent shape. A2 resumes at a snapshot below the questions; once
     two prompts left the shared head differently, A3 takes a snapshot where they part (branch_at)
     and A4 resumes there; A5 repeats A1 whole (a full hit backs off to its last chunk).
  B  a conversation about a C++ function (about 650 tokens) with fixed replies, four user turns:
     each turn resumes at the previous prompt's tail snapshot and recomputes its last chunk, the
     reply and the new question. From B3 on a hit also adopts blocks the turn before last wrote
     in its short last chunk (what RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH, part Pcx, changes).
  D  B3 with its last question replaced (a regenerated or edited message), sent before B4.
  C  a story whose turns send the model's own previous reply back (assistant_from), as a chat
     client does: a hit there also adopts blocks decode steps wrote, under Pc without Pcx.
Prompts are built by ab_requests.py from chat_turns in DeepSeek-V4.1's chat framing, every text
tokenized by the server on its own, so each turn starts with the previous prompt token for token.
"""
import json
import sys

MAX_TOKENS = 48

CODE_INTRO = ("Here is a C++ function from the prefix cache of an LLM inference engine.\n\n"
              "```cpp\n%s```\n\n")
B_TURNS = [
    ("Explain in a few sentences which checkpoints GeometricBackoff::select keeps.",
     "It decides which checkpoints of one session to keep. It always keeps the tip, the deepest "
     "checkpoint. Then, for every scale S that doubles from the checkpoint interval, it looks at "
     "the bucket of width S that holds the tip and the bucket just before it, and keeps the "
     "shallowest checkpoint in each. The kept positions thin out geometrically with distance from "
     "the tip, so a session holds only about log2(tip / interval) + 2 snapshots."),
    ("Why does the loop stop when b is 0?",
     "When b is 0 the tip lies in the first bucket [0, S), and so does every other checkpoint. At "
     "that scale the only candidate is the shallowest checkpoint of all, and every larger scale "
     "would pick that same one again, so further iterations cannot keep anything new. Stopping "
     "there also bounds the loop to about log2(tip / interval) iterations."),
    ("What does CheckpointPolicy::held compute, and why does it also try a tip half an interval "
     "further on?",
     "held() replays one session that grows to ctx tokens with a checkpoint at every interval, "
     "applies the policy after each new checkpoint, and returns the most snapshots the session "
     "ever holds; that sizes the snapshot store per sequence. A session's tip is usually a prompt's "
     "tail between two interval boundaries, so each step also adds a tip half an interval past the "
     "last boundary and counts what the policy would keep then, which can be one more."),
    ("Write a short unit test for GeometricBackoff::select with an interval of 16 and checkpoints "
     "at 16, 32, 48, 64 and 80.", None),
]
D_QUESTION = ("Could GeometricBackoff::select ever drop the tip? Answer briefly and name the line "
              "that decides it.")

DOC_INTRO = "Read the following license text and answer the question after it.\n\n%s\nQuestion: %s"
A_QUESTIONS = [
    "Under which section may a recipient distribute the Covered Software in Executable Form, and "
    "on what condition? Answer in two sentences.",
    "Does this license grant any trademark rights? Quote the sentence that says so.",
    "Summarize section 2.3 in one sentence.",
    "What must a Larger Work comply with according to section 3.3?",
]
F_PROMPT = "List three practical uses of a hash table, with one sentence on each."
C_TURNS = [
    "Write the opening paragraph of a short story about a lighthouse keeper who finds a message "
    "in a bottle.",
    "Continue the story with a second paragraph in which the keeper decides to answer the message.",
    "Give the story a title and summarize it in one sentence.",
]


def req(name, turns):
    return {"name": name, "request": {"max_tokens": MAX_TOKENS, "model": "ds41-native-smoke",
                                      "temperature": 0, "stream": False, "chat_turns": turns}}


def main():
    out = sys.argv[1]
    mpl = sys.argv[2] if len(sys.argv) > 2 else "/usr/share/common-licenses/MPL-2.0"
    src = sys.argv[3] if len(sys.argv) > 3 else None
    lines = open(mpl).read().splitlines(keepends=True)
    doc = "".join(lines[85:218])             # sections 2 and 3: "2. License Grants" .. 3.5
    if src:
        code_lines = open(src).read().splitlines(keepends=True)
        code = "".join(code_lines[59:111])   # GeometricBackoff::select .. CheckpointPolicy::held
    else:
        raise SystemExit("give core/mem/prefix.cpp as the third argument")
    if not doc.startswith("2. License Grants") or "GeometricBackoff::select" not in code:
        raise SystemExit("unexpected source text (doc starts %r, code %r)" % (doc[:30], code[:60]))

    reqs = [req("pfx-F1", [{"user": F_PROMPT}]), req("pfx-F2", [{"user": F_PROMPT}])]
    for i, q in enumerate(A_QUESTIONS):
        reqs.append(req("pfx-A%d" % (i + 1), [{"user": DOC_INTRO % (doc, q)}]))
    reqs.append(req("pfx-A5", [{"user": DOC_INTRO % (doc, A_QUESTIONS[0])}]))
    b = []
    for i, (u, a) in enumerate(B_TURNS):
        user = (CODE_INTRO % code + u) if i == 0 else u
        turns = b + [{"user": user}]
        if i == 3:
            # D1: B3's history with another last question, before B4 continues B3.
            reqs.append(req("pfx-D1", b[:-1] + [{"user": D_QUESTION}]))
        reqs.append(req("pfx-B%d" % (i + 1), turns))
        b = b + [{"user": user, "assistant": a}] if a else b
    c = []
    for i, u in enumerate(C_TURNS):
        reqs.append(req("pfx-C%d" % (i + 1), c + [{"user": u}]))
        c = c + [{"user": u, "assistant_from": "pfx-C%d" % (i + 1)}]
    json.dump({"benchmark_requests": reqs}, open(out, "w"), indent=1, ensure_ascii=False)
    print("%s: %d requests; document %d chars, code %d chars" % (out, len(reqs), len(doc), len(code)))


if __name__ == "__main__":
    main()

"""Writes the prefix-cache receipts: requests that share long prefixes, run in order on one server,
so a window can compare each reply with a cold server's byte for byte (the same arm without Pc)
and read each request's TTFT and cached prompt tokens (ab_requests.py --ttft).

  python3 -I make_prefix_receipt.py OUT.json [MPL-2.0 text] [prefix.cpp]
      the chat receipt (work/extra-prompts/pfxchat-receipt.json, part Upfxchat)
  python3 -I make_prefix_receipt.py --long OUT.json LONG_RECEIPT.json
      the long-context receipt (work/extra-prompts/pfxlong-receipt.json, part Upfxlong), from
      long-receipt.json's 3,086-token document
  python3 -I make_prefix_receipt.py --harness OUT.json REPO [--one]
      the agent-harness receipt (work/extra-prompts/harness-receipt.json, part Uharness; with
      --one harness1-receipt.json, Uharness1, its first session alone), from the repository REPO's
      own public texts at the commit the window's run home was built from (see harness() below)

Every reply is capped at MAX_TOKENS. Prompts are built by ab_requests.py from chat_turns in
DeepSeek-V4.1's chat framing, every text tokenized by the server on its own, so each turn starts
with the previous prompt token for token; a one-turn chat_turns prompt is the chat_text prompt of
the same text.

What a hit is, under --prefix-resume exact at 16-token chunks: the tokens a request skips end at a
snapshot (a checkpoint slot: the compressor's f32 tail, the Engram history, the 128-token window),
taken at every --checkpoint-interval boundary and where a prompt's last chunk starts (its tail,
only past the first interval), and kept by geometric-backoff retention. A turn resumes at the
previous prompt's tail and recomputes from there, in the cold run's own 16-token chunks.

The chat receipt, families in the order sent:
  F  a short prompt (about 19 tokens) and its exact repeat: only an interval of 16 (Pci16) gives it
     a snapshot (at 16) to resume from.
  A  a license text (MPL-2.0 sections 2 and 3, about 1,250 tokens) with four different questions,
     then the first again: the RAG/agent shape. A2 resumes at a snapshot below its question; once
     two prompts left the shared head differently, A3 takes a snapshot where they part (branch_at)
     and A4 resumes there; A5 repeats A1 whole (a full hit backs off to its last chunk).
  B  a conversation about a C++ function (about 700 tokens) with fixed replies, four user turns:
     each turn resumes at the previous prompt's tail and recomputes its last chunk, the reply and
     the new question. From B3 on a hit also adopts blocks the turn before last wrote in its short
     last chunk (what RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH, part Pcx, changes).
  D  B3 with its last question replaced (a regenerated or edited message), sent before B4.
  C  a story (about 360 tokens) whose turns send the model's own previous reply back
     (assistant_from), as a chat client does: C3 adopts blocks C1 wrote in its short last chunk and
     in its decode steps (unless Pcx), at any interval up to 256.

The long-context receipt (contexts past 2,048 tokens, where the indexer's top-k selects among the
compressed keys rather than keeping them all), one 3,086-token GPL-3 text:
  L1 the long receipt's long-3500 request (cold everywhere; its TTFT is the 14:54 window's 100 s),
  L2 its exact repeat (a hit at L1's tail, 3,072),
  L3 the same text cut about 2,000 characters earlier (a hit below the cut, deep in the context),
  L4 L1, its reply and a second question; L5 L4, its reply and a third (a hit adopting L1's short
     last chunk and decode blocks beneath L4's tail).
"""
import json
import subprocess
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
# Long enough (about 360 tokens with its request) that C1 has a tail snapshot past an interval of
# 256, so C3 adopts C1's short last chunk and decode blocks at Pci256 too.
C_STORY = (
    "The lighthouse on Carrow Point had been automated for eleven years, but Ines Marrow still "
    "climbed its ninety-two steps every evening at dusk. The keepers' cottage at the foot of the "
    "tower was hers on a lease the harbour board renewed each spring without anyone reading it, "
    "and she paid for it by logging the weather, the shipping and anything else the sea brought "
    "in. Most days the sea brought kelp, a few plastic floats and once, memorably, a crate of "
    "oranges that fed the village for a week. On the morning after the October gale she walked "
    "the shingle below the cliff with her logbook and a canvas bag, as she always did after a "
    "storm. The tide was far out. Gulls worked the wrack line in pairs, and the air smelled of "
    "iodine and cold stone. Near the black rocks at the end of the beach, wedged between two "
    "boulders where the last wave had left it, lay a green glass bottle with a cork sealed in red "
    "wax. Ines had found bottles before, all of them empty or full of sand. This one was clean, "
    "and through the glass she could see a roll of paper tied with a strip of blue ribbon. She "
    "carried it back to the cottage unopened, set it on the kitchen table beside the kettle and "
    "looked at it for a long time while the water boiled. When she finally cut the wax and drew "
    "out the paper, the handwriting was small and careful, and the first line was addressed to "
    "the keeper of the Carrow Point light.")
C_TURNS = [
    "Here is the opening of a short story.\n\n%s\n\nContinue the story with one paragraph in which "
    "Ines reads the message." % C_STORY,
    "Continue the story with a second paragraph in which she decides to answer the message.",
    "Give the story a title and summarize it in one sentence.",
]

L_QUESTIONS = [
    "Which section of the text covers conveying the work in non-source forms, and what must "
    "accompany it? Answer in two sentences.",
    "What does the text say a \"patent license\" is? Quote its definition.",
]


def req(name, turns, max_tokens=MAX_TOKENS):
    return {"name": name, "request": {"max_tokens": max_tokens, "model": "ds41-native-smoke",
                                      "temperature": 0, "stream": False, "chat_turns": turns}}


def chat(out, mpl, src):
    lines = open(mpl).read().splitlines(keepends=True)
    doc = "".join(lines[85:218])             # sections 2 and 3: "2. License Grants" .. 3.5
    code_lines = open(src).read().splitlines(keepends=True)
    code = "".join(code_lines[59:111])       # GeometricBackoff::select .. KeepAll::held
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
    print("%s: %d requests; document %d chars, code %d chars, story %d chars" % (
        out, len(reqs), len(doc), len(code), len(C_STORY)))


def long_ctx(out, long_receipt):
    src = {r["name"]: r for r in json.load(open(long_receipt))["benchmark_requests"]}
    text = src["long-3500"]["request"]["chat_text"]
    head = "Summarize the following text in three sentences.\n\n"
    if not text.startswith(head) or len(text) < 12000:
        raise SystemExit("unexpected long-3500 text (%d chars, starts %r)" % (len(text), text[:60]))
    # L3: the text cut at the end of the sentence nearest 2,000 characters before its end.
    cut = text.rfind(". ", 0, len(text) - 2000) + 1
    if cut < len(text) // 2:
        raise SystemExit("no sentence end to cut at")
    short = text[:cut]
    l4 = [{"user": text, "assistant_from": "pfxL-1"}]
    reqs = [
        req("pfxL-1", [{"user": text}]),
        req("pfxL-2", [{"user": text}]),
        req("pfxL-3", [{"user": short}]),
        req("pfxL-4", l4 + [{"user": L_QUESTIONS[0]}]),
        req("pfxL-5", l4 + [{"user": L_QUESTIONS[0], "assistant_from": "pfxL-4"},
                            {"user": L_QUESTIONS[1]}]),
    ]
    json.dump({"benchmark_requests": reqs}, open(out, "w"), indent=1, ensure_ascii=False)
    print("%s: %d requests; text %d chars, cut at %d" % (out, len(reqs), len(text), cut))


# ------------------------------------------------------------------ the agent-harness receipt
#
# What an agent harness sends: every turn is the whole conversation again -- a system prompt, then
# the turns so far, each with the model's own reply sent back (assistant_from), then the new user
# message or tool output. Two sessions, interleaved turn by turn (A1 B1 A2 B2 ...), as two agents
# sharing a server would send them:
#   hxA  a code-review agent: a system prompt of about 2,900 tokens (its rules and the project's
#        NOTICE), a task, five tool outputs of source code from the repository (prefix.h in three
#        parts, the retention policies of prefix.cpp, geometry.h), and a closing request: 7 turns,
#        about 9.4K tokens at the last (the official tokenizer's count, replies at their cap).
#   hxB  a licence-compliance agent: a system prompt of about 2,300 tokens (its rules and the
#        Apache-2.0 LICENSE), a task, four tool outputs (the two vendored MIT licences,
#        consumer_skip.h, geometry.cpp) and a closing request: 6 turns, about 9.8K at the last.
# Each turn adds 0.4-3.4K tokens and asks for a reply of up to HX_MAX_TOKENS. Under the prefix
# cache with end-of-request snapshots a turn should prefill only what it added, plus at most 15
# reply tokens; cold, the whole conversation: about 79K prompt tokens over the 13 turns, against
# about 19K. The texts are sliced by line from HX_REV and their first line checked, so the receipt
# is the same wherever it is built.
HX_MAX_TOKENS = 128

HX_A_RULES = (
    "You are a code-review agent working inside a C++ inference engine's repository. You can call "
    "one tool, read_file(path, first_line, last_line), and the harness answers each call with the "
    "file's lines in a fenced block. Work in small steps: read what you need, then say what you "
    "found before you read more. Keep every answer under 120 words, name functions and fields "
    "exactly as the source spells them, and say when a conclusion rests on code you have not read. "
    "Never invent a line number. When you are asked to summarize, give five bullet points and "
    "nothing else. The project's NOTICE file follows; credit contributors by the names it uses.\n\n")
HX_B_RULES = (
    "You are a licence-compliance agent. You read the licence texts of a project and of the code "
    "it vendors and answer questions about what a redistributor must do. You can call one tool, "
    "read_file(path), and the harness answers each call with the file in a fenced block. Quote the "
    "clause you rely on, give its section number, and keep every answer under 120 words. When two "
    "licences impose different duties, list each one apart. The project's own LICENSE follows.\n\n")


# The texts are read from this commit of the repository (git show), not its working tree, so the
# receipt does not move when the files do.
HX_REV = "b0a84f7c"


def hx_slice(repo, path, first, last, starts):
    """Lines [first, last] (1-based) of `path` at HX_REV in the repository `repo`, which must begin
    with `starts`."""
    src = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (HX_REV, path)], check=True,
                         capture_output=True).stdout.decode("utf-8")
    lines = src.splitlines(keepends=True)
    text = "".join(lines[first - 1:last])
    if not text.lstrip().startswith(starts):
        raise SystemExit("%s:%d: expected %r, found %r" % (path, first, starts, text[:60]))
    return text


def hx_tool(path, first, last, text):
    return ("Tool output for read_file(\"%s\", %d, %d):\n```\n%s```\n" % (path, first, last, text))


def harness(out, repo, one=False):
    notice = hx_slice(repo, "NOTICE", 1, 130, "radiance")
    license_ = hx_slice(repo, "LICENSE", 1, 202, "Apache License")
    a_tools = [
        ("core/mem/prefix.h", 1, 60, "/* prefix.h -- prefix caching"),
        ("core/mem/prefix.h", 91, 162, "/* ------------------------------------------------------------------ the split lookup"),
        ("core/mem/prefix.h", 163, 235, "/* The policy --checkpoint-policy names"),
        ("core/mem/prefix.cpp", 58, 138, "/* ------------------------------------------------------------------ retention policies"),
        ("core/sched/geometry.h", 1, 101, "/* geometry.h"),
    ]
    a_asks = [
        "Read the top of core/mem/prefix.h first: what does the prefix cache key a block by, and why "
        "is a linear-state hit handled apart from an attention hit?",
        "Which fields of CacheHit tell the scheduler how far it may resume, and what does a HitCap do?",
        "What decides the size of the snapshot store, and what bounds it?",
        "Explain in your own words which snapshots GeometricBackoff::select keeps.",
        "How does the resume grid of --prefix-resume exact follow from the chunk geometry?",
    ]
    b_tools = [
        ("vendor/nlohmann/LICENSE.MIT", 1, 21, "MIT License"),
        ("vendor/cpp-httplib/LICENSE", 1, 22, "The MIT License"),
        ("core/sched/consumer_skip.h", 1, 151, "/* consumer_skip.h"),
        ("core/sched/geometry.cpp", 1, 239, "/* geometry.cpp"),
    ]
    b_asks = [
        "What must a redistributor of a binary that contains this code do under the MIT licence above?",
        "Does this second MIT licence add anything to the first? Answer by clause.",
        "This header is the project's own code: which licence covers it, and must a modified copy say "
        "so?",
        "Same question for this file, and name the section of the Apache licence that governs "
        "modified files.",
    ]
    for path, first, last, starts in a_tools + b_tools:
        hx_slice(repo, path, first, last, starts)   # every slice checked before anything is written

    def session(tag, system, task, tools, asks, closing):
        reqs, turns = [], []
        users = [task] + [hx_tool(p, f, l, hx_slice(repo, p, f, l, s)) + "\n" + q
                          for (p, f, l, s), q in zip(tools, asks)] + [closing]
        for i, u in enumerate(users):
            name = "%s-%d" % (tag, i + 1)
            r = req(name, turns + [{"user": u}], HX_MAX_TOKENS)
            r["request"]["chat_system"] = system
            r["family"] = tag
            reqs.append(r)
            turns = turns + [{"user": u, "assistant_from": name}]
        return reqs

    a = session("hxA", HX_A_RULES + notice,
                "We are reviewing how this engine reuses a conversation's prefix between turns. Start "
                "with core/mem/prefix.h and tell me what you will read first and why.",
                a_tools, a_asks, "Summarize your review.")
    b = session("hxB", HX_B_RULES + license_,
                "List the licences a binary distribution of this project must carry, and tell me which "
                "files you will read to be sure.",
                b_tools, b_asks, "Write the attribution section for a binary distribution, in at most "
                "eight lines.")
    reqs = a if one else [r for pair in zip(a, b) for r in pair] + a[len(b):] + b[len(a):]
    json.dump({"benchmark_requests": reqs}, open(out, "w"), indent=1, ensure_ascii=False)
    print("%s: %d requests (%s); system prompts %d and %d chars" % (
        out, len(reqs), "hxA alone" if one else "hxA and hxB interleaved",
        len(HX_A_RULES + notice), len(HX_B_RULES + license_)))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--long":
        long_ctx(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--harness":
        harness(sys.argv[2], sys.argv[3], one="--one" in sys.argv[4:])
        return
    out = sys.argv[1]
    mpl = sys.argv[2] if len(sys.argv) > 2 else "/usr/share/common-licenses/MPL-2.0"
    if len(sys.argv) <= 3:
        raise SystemExit("give core/mem/prefix.cpp as the third argument")
    chat(out, mpl, sys.argv[3])


if __name__ == "__main__":
    main()

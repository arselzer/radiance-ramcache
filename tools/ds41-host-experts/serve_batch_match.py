#!/usr/bin/env python3
"""serve-batch.sh against a harness launch: whether the launcher expresses an arm (ab_window.sh's Sb part,
serve_batch_test.sh).

  serve_batch_match.py params WORDS
      WORDS: an arm's launch, one word a line (ab_window.sh launch_words: the environment assignments, the binary,
      its flags). Prints the serve-batch.sh arguments for it, one a line: --model, --silu-table, --streams,
      --ctx-per-stream (--max-model-len / --max-num-seqs, which must divide), --port, --served-name,
      --radiance-home, --shared-head, --hybrid (..._P1_HYBRID: QyP's P, 0 or unset off; a held share, Qyf, is
      refused), and --profile, --mirror, --reserve-mib, --max-seq-len (SlN) where the launch has them. Exit 1
      with the reason when a value is missing or does not fit.
  serve_batch_match.py compare WORDS DRYRUN [FLAG ...]
      DRYRUN: serve-batch.sh --dry-run's output. Each side resolved as the server sees it (every variable at its
      last assignment, every flag at its last value); prints each difference and exits 1 when there is one. The
      FLAGs named are left out of the comparison.
"""
import shlex
import sys

NOVAL = {"--weights-disk-tier", "--deterministic", "--no-prefix-cache"}
SILU = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU"
SHEAD = "RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD"
RESERVE = "RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB"
HYBRID = "RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID"
ADAPT = "RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT"


def resolve(words):
    env, order, i = {}, [], 0
    while i < len(words) and "=" in words[i] and not words[i].startswith("-"):
        k, v = words[i].split("=", 1)
        if k not in env:
            order.append(k)
        env[k] = v
        i += 1
    if i >= len(words):
        raise ValueError("no binary after the environment")
    binary, flags, i = words[i], {}, i + 1
    while i < len(words):
        k = words[i]
        if k in NOVAL:
            flags[k] = None
            i += 1
        elif i + 1 < len(words):
            flags[k] = words[i + 1]
            i += 2
        else:
            raise ValueError("flag %s without its value" % k)
    return [(k, env[k]) for k in order], binary, flags


def words_file(path):
    with open(path) as f:
        return [l for l in f.read().split("\n") if l != ""]


def dryrun_file(path):
    with open(path) as f:
        lines = f.read().splitlines()
    cmd = [l for l in lines if l.startswith("cmd ")]
    if len(cmd) != 1:
        raise ValueError("%s: not serve-batch.sh --dry-run output" % path)
    return [l[4:] for l in lines if l.startswith("env ")] + shlex.split(cmd[0][4:])


def params(words):
    env, _, flags = resolve(words)
    env = dict(env)
    out = []

    def need(what, v):
        if v is None or v == "":
            raise ValueError("the launch has no %s" % what)
        return v
    out += ["--model", need("--model", flags.get("--model"))]
    out += ["--silu-table", need(SILU, env.get(SILU))]
    seqs, ctx = need("--max-num-seqs", flags.get("--max-num-seqs")), need("--max-model-len", flags.get("--max-model-len"))
    if not (seqs.isdigit() and ctx.isdigit() and int(seqs) > 0 and int(ctx) % int(seqs) == 0):
        raise ValueError("--max-model-len %s is not a whole number of tokens for each of --max-num-seqs %s" % (ctx, seqs))
    out += ["--streams", seqs, "--ctx-per-stream", str(int(ctx) // int(seqs))]
    out += ["--port", need("--port", flags.get("--port"))]
    out += ["--served-name", need("--served-model-name", flags.get("--served-model-name"))]
    if flags.get("--radiance-home"):
        out += ["--radiance-home", flags["--radiance-home"]]
    out += ["--shared-head", need(SHEAD, env.get(SHEAD))]
    if flags.get("--expert-profile"):
        out += ["--profile", flags["--expert-profile"]]
    if flags.get("--expert-plane-mirror"):
        out += ["--mirror", flags["--expert-plane-mirror"]]
    if env.get(RESERVE):
        out += ["--reserve-mib", env[RESERVE]]
    if env.get(ADAPT):
        raise ValueError("%s=%s (Qyf: a held hybrid share), which serve-batch.sh does not set" % (ADAPT, env[ADAPT]))
    out += ["--hybrid", env.get(HYBRID) or "0"]
    if flags.get("--max-seq-len"):
        out += ["--max-seq-len", flags["--max-seq-len"]]
    return out


def compare(h, s, allowed):
    bad = []
    hd, sd = dict(h[0]), dict(s[0])
    for k in sorted(set(hd) | set(sd)):
        if hd.get(k) != sd.get(k):
            bad.append("env %s: the arm %r, serve-batch.sh %r" % (k, hd.get(k), sd.get(k)))
    if h[1] != s[1]:
        bad.append("binary: the arm %r, serve-batch.sh %r" % (h[1], s[1]))
    for k in sorted(set(h[2]) | set(s[2])):
        if k in allowed:
            continue
        if h[2].get(k, "<absent>") != s[2].get(k, "<absent>"):
            bad.append("flag %s: the arm %r, serve-batch.sh %r" % (k, h[2].get(k, "<absent>"), s[2].get(k, "<absent>")))
    return bad


def main(argv):
    try:
        if len(argv) == 3 and argv[1] == "params":
            print("\n".join(params(words_file(argv[2]))))
            return 0
        if len(argv) >= 4 and argv[1] == "compare":
            h, s = resolve(words_file(argv[2])), resolve(dryrun_file(argv[3]))
            bad = compare(h, s, set(argv[4:]))
            for b in bad:
                print(b)
            if not bad:
                print("serve-batch.sh gives the arm's launch: %d variables, %d flags" % (len(h[0]), len(h[2])))
            return 1 if bad else 0
    except (OSError, ValueError) as e:
        print(str(e))
        return 1
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

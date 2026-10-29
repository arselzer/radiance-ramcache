"""The routing trace read back, and the lossy track's gates computed from it.

RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=<prefix> (ab_window.sh's Rt part; core/place/route_trace.h
has the format) writes, per EP-routed layer of every pass, the six picks with their weights and
biased keys, rank 0's 384 router logits on one-token passes, what the rank owning each pick did
with it (on the card, a RAM hit, read ahead, or a miss read while the layer waited), each rank's
RAM-cache residency at the choice, and which request and position every pass carried. From it,
the gates of RESEARCH-3-LOSSY.md ("Recommended order" 2) for decode tokens:

  - the routing weight a miss carries (its share of the six weights): the mass a drop removes;
  - weight-threshold drop: the share of misses below tau (0.1) -- go at 50% or more;
  - cache-aware routing: the share of misses a held expert (on the card, or in the RAM cache at
    that moment) could replace with a key within delta (5%) of the 6th key -- go at 40% or more,
    kill below 20% (`assignable`: each held near-tie replaces one miss of its layer);
  - the misses' slots (the researcher's protect-top-2 gate: 60% or more at slot 2 or below);
  - all of it per prompt family and per layer, plus a replay estimate of decode tok/s under
    drop-all, drop below tau, a drop cap of 6 a token and the researcher's protect-top-k arms
    (drop only misses at slot 2 or below / 4 or below), from the assessment's time model
    T = a + b x waited misses + c x CPU hits a token (both summed over the ranks): (24.7, 2.54,
    0.133) and the constrained fit (31, 2.28, 0.108), each also with `a` refitted so the model
    meets the arm's measured decode time; a dropped miss is still read and inserted in the
    background (--insert-ms a drop, 0.15).

A miss here is a pick whose layer waited for its read: status M (read from the drive) or P (read
ahead in part, the rest read while the layer waited). A (read ahead whole) waits for nothing and
counts as a CPU hit, as H does. Post-stop passes (a request's pass issued ahead after it had
stopped, its output discarded) are left out of the gates and the replay and counted on their own.

  rtrace.py report PREFIX [ARM.json] [--tau T] [--delta D] [--cap N] [--insert-ms MS] [--json OUT]
        PREFIX: the arm's rtrace-<arm>/rt; ARM.json: its ab_requests.py output (the requests'
        names, families and measured decode times, in the order sent); without it requests are
        numbered and nothing calibrates the replay's fixed part.
  rtrace.py dump PREFIX [N]          each file's header and first N records, decoded
  rtrace.py synth PREFIX ARM.json    a small synthetic trace with known answers (rtrace_test.py)

Standard library only; run with python3 -I.
"""
import array
import json
import math
import os
import struct
import sys

MAGIC = b"RADRTRC1"
HEADER = struct.Struct("<8s8IQ")
REC = struct.Struct("<HHI")
PASS = struct.Struct("<iIhhiiiiiIQ")
LAYER = struct.Struct("<IiHHHBBI3I")
PICK = struct.Struct("<HBB")
STEP = struct.Struct("<ihhiHBBQ")
ENTRY = struct.Struct("<QiiBBH")
END = struct.Struct("<iiQQ")
T_PASS, T_LAYER, T_BIAS, T_TIERS, T_STEP, T_END = 1, 2, 3, 4, 5, 6
F_DECIDED, F_ROUTE, F_LOGITS, F_RAM = 1, 2, 4, 8
E_PREFILL, E_PRODUCES, E_DEV = 1, 2, 4
STATUS = "VHMAPCUT"   # on the card, RAM hit, miss, read ahead whole / in part, copy path, undecided, host-pinned
WAITED = ("M", "P")
CPU_HIT = ("H", "A")
TIER_VRAM = 0
STEPS = 4
MODELS = {"fit": (24.7, 2.54, 0.133), "constrained": (31.0, 2.28, 0.108)}


# ---- the files ---------------------------------------------------------------------------------

def path_of(prefix, s):
    return prefix + (".steps.bin" if s == STEPS else ".rank%d.bin" % s)


def read_records(path):
    """(header dict, [(type, flags, bytes, offset of the fields)]) and the file's bytes."""
    with open(path, "rb") as f:
        b = f.read()
    if len(b) < 64:
        raise ValueError("%s: no header" % path)
    m, ver, stream, world, ne, k, local, nl, hb, t0 = HEADER.unpack_from(b, 0)
    if m != MAGIC or ver != 1:
        raise ValueError("%s: not a version-1 route trace" % path)
    h = dict(stream=stream, world=world, n_expert=ne, top_k=k, local=local, n_layers=nl, t0=t0)
    out, o = [], hb
    while o + 8 <= len(b):
        t, fl, n = REC.unpack_from(b, o)
        if n < 8 or n % 4 or o + n > len(b):
            raise ValueError("%s: a torn record at byte %d" % (path, o))
        out.append((t, fl, n, o + 8))
        o += n
    return h, out, b


class Rank:
    """One rank's file: passes by their counter, layer records by (pass, layer), the tiers, and
    (rank 0) the biases."""

    def __init__(self, prefix, r, keep_logits=True):
        self.h, recs, b = read_records(path_of(prefix, r))
        self.passes, self.layers, self.bias, self.tiers = {}, {}, {}, None
        ne, k = self.h["n_expert"], self.h["top_k"]
        for t, _fl, _n, o in recs:
            if t == T_PASS:
                step, p, phase, draft, n_tok, n_seq, n_spec, n_tok_dec, ctx, _z, t_ns = PASS.unpack_from(b, o)
                self.passes[p] = dict(step=step, phase=phase, draft=draft, n_tok=n_tok, n_seq=n_seq,
                                      n_spec=n_spec, ctx=ctx, t_ns=t_ns)
            elif t == T_LAYER:
                p, step, layer, rows, n_sel, fl, _z, us, r0, r1, r2 = LAYER.unpack_from(b, o)
                q = o + LAYER.size
                picks = {}
                for i in range(n_sel):
                    e, nr, st = PICK.unpack_from(b, q + 4 * i)
                    picks[e] = (STATUS[st] if st < len(STATUS) else "?", nr)
                q += 4 * n_sel
                route = []
                if fl & F_ROUTE:
                    for _ in range(rows):
                        ids = struct.unpack_from("<%dH" % k, b, q)
                        w = struct.unpack_from("<%df" % k, b, q + 2 * k)
                        keys = struct.unpack_from("<%df" % k, b, q + 6 * k)
                        route.append((ids, w, keys))
                        q += 10 * k
                logits = None
                if fl & F_LOGITS and keep_logits:
                    logits = array.array("f")
                    logits.frombytes(b[q:q + 4 * ne * rows])
                self.layers[(p, layer)] = dict(step=step, rows=rows, flags=fl, us=us,
                                               ram=r0 | (r1 << 32) | (r2 << 64), picks=picks,
                                               route=route, logits=logits)
            elif t == T_BIAS:
                layer, n = struct.unpack_from("<HH", b, o)
                self.bias[layer] = struct.unpack_from("<%df" % n, b, o + 4)
            elif t == T_TIERS:
                nl, local = struct.unpack_from("<HH", b, o)
                q, tiers = o + 4, {}
                for _ in range(nl):
                    layer, _z = struct.unpack_from("<HH", b, q)
                    tiers[layer] = bytes(b[q + 4:q + 4 + local])
                    q += 4 + local
                self.tiers = tiers


def read_steps(prefix):
    """Step records by step number, and the end records."""
    _h, recs, b = read_records(path_of(prefix, STEPS))
    steps, ends = {}, []
    for t, _fl, _n, o in recs:
        if t == T_STEP:
            step, phase, draft, n_tok, n, ahead, _z, t_ns = STEP.unpack_from(b, o)
            es = [ENTRY.unpack_from(b, o + STEP.size + ENTRY.size * i)[:5] for i in range(n)]
            steps[step] = dict(phase=phase, draft=draft, n_tok=n_tok, ahead=ahead, t_ns=t_ns,
                               entries=[dict(seq=e[0], ctx=e[1], n=e[2], flags=e[3], n_spec=e[4]) for e in es])
        elif t == T_END:
            step, ahead_step, seq, n_out = END.unpack_from(b, o)
            ends.append(dict(step=step, ahead_step=ahead_step, seq=seq, n_out=n_out))
    return steps, ends


# ---- the decode tokens -------------------------------------------------------------------------

def score(x):
    """The card's score of a logit (sqrt(softplus(x)); x > 20: sqrt(x)), in double."""
    return math.sqrt(x if x > 20.0 else math.log1p(math.exp(x)))


def load(prefix, arm_json=None):
    """Every one-token pass that carried one request's decode row, with its 40 layers joined across
    the ranks: [dict(req, name, family, token, post_stop, step, pass, t_ns, layers)], plus the
    requests' table and the joins that did not hold (format or engine bugs: want none)."""
    ranks = [Rank(prefix, r) for r in range(4)]
    steps, ends = read_steps(prefix)
    arm = []
    if arm_json:
        with open(arm_json) as f:
            arm = json.load(f)["requests"]
    # Requests in the order their first step carried them; the arm's json names them in order sent.
    order, prompt = [], {}
    for s in sorted(steps):
        for e in steps[s]["entries"]:
            if e["seq"] not in prompt:
                order.append(e["seq"])
                prompt[e["seq"]] = 0
            if e["flags"] & E_PREFILL:
                prompt[e["seq"]] += e["n"]
    reqs = {}
    for i, seq in enumerate(order):
        a = arm[i] if i < len(arm) else None
        reqs[seq] = dict(index=i, seq=seq, prompt=prompt[seq],
                         name=a["name"] if a else "request-%02d" % (i + 1),
                         family=(a.get("family") or a["name"].rsplit("-", 1)[0]) if a else "?",
                         timings=(a or {}).get("timings") or {},
                         extra=bool(arm) and a is None)   # past ARM.json's list: an Lq pass's
    post = {(e["ahead_step"], e["seq"]) for e in ends if e["ahead_step"] >= 0}
    problems = []
    tokens = []
    r0 = ranks[0]
    for p in sorted(r0.passes):
        ps = r0.passes[p]
        st = steps.get(ps["step"])
        if ps["n_tok"] != 1 or ps["draft"] != 0 or not st or len(st["entries"]) != 1:
            continue
        e = st["entries"][0]
        if e["flags"] & E_PREFILL:
            continue
        rq = reqs[e["seq"]]
        layers = {}
        for layer in range(r0.h["n_layers"]):
            rec0 = r0.layers.get((p, layer))
            if rec0 is None:
                continue
            if not rec0["route"]:
                problems.append("pass %d layer %d: rank 0 has no route" % (p, layer))
                continue
            ids, w, _keys = rec0["route"][0]
            picks = []
            for slot, eid in enumerate(ids):
                owner = ranks[eid // 96]
                rec = owner.layers.get((p, layer))
                if rec:
                    st_ = rec["picks"].get(eid, ("X", 0))[0]
                else:   # a rank traces only its layers with file-tier experts: else all on the card
                    t = owner.tiers
                    st_ = "V" if t and layer in t and t[layer][eid % 96] == TIER_VRAM else "X"
                if st_ == "X":
                    problems.append("pass %d layer %d: rank %d has no status for its pick %d" % (
                        p, layer, eid // 96, eid))
                if rec and rec["step"] != ps["step"]:
                    problems.append("pass %d layer %d: rank %d's step %d, rank 0's %d" % (
                        p, layer, eid // 96, rec["step"], ps["step"]))
                picks.append(dict(slot=slot, expert=eid, weight=w[slot], status=st_))
            sel = set(ids)
            for r in range(4):   # an owner's selected expert rank 0 did not route: a join failure
                rec = ranks[r].layers.get((p, layer))
                for eid in (rec["picks"] if rec else {}):
                    if eid not in sel:
                        problems.append("pass %d layer %d: rank %d selected %d, rank 0 did not" % (p, layer, r, eid))
            layers[layer] = dict(picks=picks, logits=rec0["logits"], rec=[ranks[r].layers.get((p, layer)) for r in range(4)])
        tokens.append(dict(req=rq["index"], seq=e["seq"], name=rq["name"], family=rq["family"],
                           token=e["ctx"] - rq["prompt"], post_stop=(ps["step"], e["seq"]) in post,
                           extra=rq["extra"], step=ps["step"], pass_=p, t_ns=ps["t_ns"],
                           layers=layers, ms=None))
    # A token's time: from its pass's start to its request's next decode pass's start on rank 0
    # (the rank threads issue a pass as the one before drains, so the interval is the token's).
    for t0, t1 in zip(tokens, tokens[1:]):
        if t1["seq"] == t0["seq"] and t1["token"] == t0["token"] + 1:
            t0["ms"] = (t1["t_ns"] - t0["t_ns"]) / 1e6
    return dict(ranks=ranks, steps=steps, ends=ends, reqs=reqs, tokens=tokens, problems=problems)


def held(ranks, rec, layer, e):
    """Whether expert e was held by its owner at that layer's choice: on the card (the plan's
    tier), or in the RAM cache by the owner's snapshot; None when the snapshot is missing."""
    r, local = e // 96, e % 96
    t = ranks[r].tiers
    if t and layer in t and t[layer][local] == TIER_VRAM:
        return "V"
    lr = rec[r]
    if not lr or not (lr["flags"] & F_RAM):
        return None
    return "R" if (lr["ram"] >> local) & 1 else ""


def substitutes(d, layer, L, delta):
    """The layer's near-ties a held expert offers: unselected experts with a key within delta of
    the 6th key (|K6| relative), held on the card or in the RAM cache; None without logits or bias."""
    bias = d["ranks"][0].bias.get(layer)
    lg = L["logits"]
    if lg is None or bias is None:
        return None
    sel = {p["expert"] for p in L["picks"]}
    keys = [score(lg[e]) + bias[e] for e in range(len(lg))]
    k6 = min(keys[e] for e in sel)
    floor = k6 - delta * abs(k6)
    out = []
    for e in range(len(keys)):
        if e in sel or keys[e] < floor:
            continue
        h = held(d["ranks"], L["rec"], layer, e)
        if h:
            out.append((e, h, (k6 - keys[e]) / abs(k6) if k6 else 0.0))
    return out


# ---- the gates ---------------------------------------------------------------------------------

def share(p, picks):
    tot = sum(x["weight"] for x in picks)
    return p["weight"] / tot if tot else 0.0


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


class Tally:
    """Misses of one group (all, a family, a layer)."""

    def __init__(self):
        self.tokens = 0
        self.misses = []        # (share, slot, assignable, any)
        self.picks = {}

    def add(self, sh, slot, assignable, anysub):
        self.misses.append((sh, slot, assignable, anysub))

    def row(self, tau):
        m = self.misses
        n = len(m)
        known = [x for x in m if x[2] is not None]
        return dict(misses=n, per_token=n / self.tokens if self.tokens else float("nan"),
                    share_mean=sum(x[0] for x in m) / n if n else float("nan"),
                    share_p50=pct([x[0] for x in m], .5),
                    below_tau=sum(1 for x in m if x[0] < tau) / n if n else float("nan"),
                    slot2plus=sum(1 for x in m if x[1] >= 2) / n if n else float("nan"),
                    assignable=sum(1 for x in known if x[2]) / len(known) if known else float("nan"),
                    any_sub=sum(1 for x in known if x[3]) / len(known) if known else float("nan"),
                    known=len(known))


def gates(d, tau=0.1, delta=0.05):
    """Per decode token (post-stop passes apart): the misses' shares, slots and substitutes, by
    family and layer; the picks by status; the per-token counts the replay uses."""
    allt, fams, lays, post = Tally(), {}, {}, Tally()
    status = {}
    per_token = []   # (token, [(layer, share) of each waited miss], cpu hits)
    extra = 0
    for t in d["tokens"]:
        if t["extra"]:
            extra += 1
            continue
        tal = post if t["post_stop"] else allt
        tal.tokens += 1
        fam = fams.setdefault(t["family"], Tally())
        if not t["post_stop"]:
            fam.tokens += 1
        misses, hits = [], 0
        for layer in sorted(t["layers"]):
            L = t["layers"][layer]
            if not t["post_stop"]:
                lays.setdefault(layer, Tally()).tokens += 1
            lm = [p for p in L["picks"] if p["status"] in WAITED]
            for p in L["picks"]:
                status[p["status"]] = status.get(p["status"], 0) + (0 if t["post_stop"] else 1)
                hits += p["status"] in CPU_HIT
            if not lm:
                continue
            subs = substitutes(d, layer, L, delta)
            n_sub = None if subs is None else len(subs)
            # The layer's misses take its held near-ties best share first (a protect rule would
            # keep the heaviest); each near-tie replaces one miss.
            for rank_, p in enumerate(sorted(lm, key=lambda x: -x["weight"])):
                sh = share(p, L["picks"])
                assignable = None if n_sub is None else rank_ < n_sub
                anysub = None if n_sub is None else n_sub > 0
                tal.add(sh, p["slot"], assignable, anysub)
                if not t["post_stop"]:
                    fam.add(sh, p["slot"], assignable, anysub)
                    lays.setdefault(layer, Tally()).add(sh, p["slot"], assignable, anysub)
                    misses.append((layer, sh, p["slot"]))
        if not t["post_stop"]:
            per_token.append(dict(req=t["req"], name=t["name"], misses=misses, hits=hits, ms=t["ms"]))
    return dict(all=allt, families=fams, layers=lays, post=post, status=status, per_token=per_token,
                extra=extra)


def fit_tokens(toks, least=30):
    """T = a + b x waited misses + c x CPU hits, least squares over the tokens whose time the trace
    measured (rtrace's own: rank 0's pass starts), with R^2; None with fewer than `least` tokens or
    no spread in the misses. An extrapolation past the misses the tokens had, as the assessment's."""
    rows = [(len(t["misses"]), t["hits"], t["ms"]) for t in toks if t.get("ms") is not None]
    if len(rows) < least or len({w for w, _h, _y in rows}) < 2:
        return None
    # normal equations, X = [1, w, h]
    s = [[0.0] * 3 for _ in range(3)]
    v = [0.0] * 3
    for w, h, y in rows:
        x = (1.0, float(w), float(h))
        for i in range(3):
            v[i] += x[i] * y
            for j in range(3):
                s[i][j] += x[i] * x[j]
    m = [s[i] + [v[i]] for i in range(3)]
    for i in range(3):   # Gauss-Jordan with partial pivoting
        piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[piv][i]) < 1e-12:
            return None
        m[i], m[piv] = m[piv], m[i]
        for r in range(3):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [m[r][k] - f * m[i][k] for k in range(4)]
    a, b, c = (m[i][3] / m[i][i] for i in range(3))
    mean = sum(y for _w, _h, y in rows) / len(rows)
    tot = sum((y - mean) ** 2 for _w, _h, y in rows)
    res = sum((y - (a + b * w + c * h)) ** 2 for w, h, y in rows)
    return dict(a=a, b=b, c=c, r2=1.0 - res / tot if tot else float("nan"), n=len(rows),
                mean_ms=mean, max_misses=max(w for w, _h, _y in rows))


def replay(g, d, cap=6, tau=0.1, insert_ms=0.15):
    """Decode ms a token under each policy and model; `a` refitted so the exact policy meets the
    arm's measured decode time (its requests' predicted_ms / predicted_n) when known."""
    toks = g["per_token"]
    if not toks:
        return {}
    def counts(policy):
        out = []
        for t in toks:
            w = len(t["misses"])
            if policy == "exact":
                drop = 0
            elif policy == "drop-all":
                drop = w
            elif policy == "drop<tau":
                drop = sum(1 for _l, sh, _s in t["misses"] if sh < tau)
            elif policy == "cap":   # online: every miss in layer order until `cap` were dropped
                drop = min(w, cap)
            elif policy.startswith("slot>="):   # the researcher's protect-top-k arms
                drop = sum(1 for _l, _sh, sl in t["misses"] if sl >= int(policy[6:]))
            out.append((w - drop, t["hits"], drop))
        return out
    measured = None
    ms = sum((r["timings"].get("predicted_ms") or 0.0) for r in d["reqs"].values() if not r["extra"])
    n = sum((r["timings"].get("predicted_n") or 0) for r in d["reqs"].values() if not r["extra"])
    if ms and n:
        measured = ms / n
    res = {"measured_ms": measured, "tokens": len(toks)}
    policies = ["exact", "drop-all", "drop<tau", "cap", "slot>=2", "slot>=4"]
    models = dict(MODELS)
    fit = fit_tokens(toks)
    if fit:
        res["trace_fit"] = fit
        models["trace"] = (fit["a"], fit["b"], fit["c"])
    for name, (a, b, c) in models.items():
        if name == "trace":   # fitted on these very tokens: no refit of its fixed part
            ex = counts("exact")
            res[name] = {}
            for pol in policies:
                cs = counts(pol)
                t = sum(a + b * w + c * h + insert_ms * dr for w, h, dr in cs) / len(cs)
                res[name][pol] = dict(ms=t, tok_s=1000.0 / t, drops=sum(x[2] for x in cs) / len(cs))
            continue
        for fit in (False, True):
            if fit and measured is None:
                continue
            ex = counts("exact")
            a2 = a
            if fit:
                a2 = measured - sum(b * w + c * h for w, h, _ in ex) / len(ex)
            key = name + (" a=%.1f (measured)" % a2 if fit else "")
            res[key] = {}
            for pol in policies:
                cs = counts(pol)
                t = sum(a2 + b * w + c * h + insert_ms * dr for w, h, dr in cs) / len(cs)
                res[key][pol] = dict(ms=t, tok_s=1000.0 / t, drops=sum(x[2] for x in cs) / len(cs))
    return res


def report(prefix, arm_json=None, tau=0.1, delta=0.05, cap=6, insert_ms=0.15, out_json=None):
    d = load(prefix, arm_json)
    g = gates(d, tau, delta)
    rp = replay(g, d, cap, tau, insert_ms)
    toks = [t for t in d["tokens"] if not t["post_stop"]]
    print("route trace %s: %d decode tokens (%d post-stop passes apart), %d requests" % (
        prefix, len(toks), sum(1 for t in d["tokens"] if t["post_stop"]), len(d["reqs"])))
    if g["extra"]:
        print("  %d decode tokens of requests past ARM.json's list (an Lq pass's) left out" % g["extra"])
    if d["problems"]:
        print("  JOIN PROBLEMS (%d), first: %s" % (len(d["problems"]), "; ".join(d["problems"][:5])))
    tot = sum(g["status"].values())
    print("  picks by status: " + ", ".join("%s %d (%.1f%%)" % (s, n, 100.0 * n / tot)
                                            for s, n in sorted(g["status"].items())) if tot else "  no picks")
    a = g["all"].row(tau)
    print("  misses (waited: M, P): %d, %.2f a token" % (a["misses"], a["per_token"]))
    shares = [m[0] for m in g["all"].misses]
    print("  weight share a miss carries (its weight over the six): mean %.3f; p10 %.3f p25 %.3f "
          "p50 %.3f p75 %.3f p90 %.3f" % (a["share_mean"], pct(shares, .1), pct(shares, .25),
                                         pct(shares, .5), pct(shares, .75), pct(shares, .9)))
    pt = g["per_token"]
    if pt:
        mass = [sum(sh for _l, sh, _s in t["misses"]) for t in pt]
        nm = [len(t["misses"]) for t in pt]
        print("  a token: %.2f misses (max %d; %.1f%% of tokens with 20 or more); drop-all removes %.3f "
              "of routing weight a token (its misses' shares summed over its layers)" % (
                  sum(nm) / len(nm), max(nm), 100.0 * sum(1 for x in nm if x >= 20) / len(nm),
                  sum(mass) / len(mass)))
    print("  gate weight-threshold drop: %.1f%% of misses below tau %.2f -- %s (go at 50%%)" % (
        100 * a["below_tau"], tau, "GO" if a["below_tau"] >= 0.5 else "no"))
    verdict = "GO" if a["assignable"] >= 0.4 else ("KILL" if a["assignable"] < 0.2 else "between")
    print("  gate cache-aware routing: %.1f%% of misses have a held substitute within delta %.0f%% of "
          "the 6th key (assignable; any: %.1f%%; of %d with logits) -- %s (go at 40%%, kill below 20%%)" % (
              100 * a["assignable"], 100 * delta, 100 * a["any_sub"], a["known"], verdict))
    print("  misses at slot 2 or below: %.1f%% (protect-top-2 gate: 60%%)" % (100 * a["slot2plus"]))
    slots = [0] * 6
    for m in g["all"].misses:
        slots[m[1]] += 1
    print("  misses by slot (key order): " + " ".join("%d:%d" % (i, n) for i, n in enumerate(slots)))
    if g["post"].misses:
        print("  post-stop passes: %d, %d misses (left out above)" % (g["post"].tokens, len(g["post"].misses)))
    print("  by family:")
    print("    %-14s %6s %7s %7s %8s %8s %8s %8s" % ("family", "tokens", "misses", "a tok", "share", "<tau", "subst", "slot>=2"))
    for f in sorted(g["families"]):
        r = g["families"][f].row(tau)
        print("    %-14s %6d %7d %7.2f %8.3f %7.1f%% %7.1f%% %7.1f%%" % (
            f, g["families"][f].tokens, r["misses"], r["per_token"], r["share_mean"],
            100 * r["below_tau"], 100 * r["assignable"], 100 * r["slot2plus"]))
    print("  by layer with misses (misses a token / mean share / <tau / substitutes):")
    line = []
    for layer in sorted(g["layers"]):
        r = g["layers"][layer].row(tau)
        if not r["misses"]:
            continue
        line.append("L%d %.2f/%.2f/%.0f%%/%.0f%%" % (layer, r["per_token"], r["share_mean"],
                                                   100 * r["below_tau"], 100 * r["assignable"]))
    for i in range(0, len(line), 5):
        print("    " + "  ".join(line[i:i + 5]))
    if rp:
        print("  replay: decode tok/s (ms a token) under each drop policy, %d tokens%s, %.2f ms of "
              "background insert a drop:" % (rp["tokens"], ", measured %.1f ms (%.2f tok/s)" % (
                  rp["measured_ms"], 1000 / rp["measured_ms"]) if rp["measured_ms"] else "", insert_ms))
        if rp.get("trace_fit"):
            f = rp["trace_fit"]
            print("    the trace's own token times (%d tokens, mean %.1f ms): T = %.1f + %.3f x misses + "
                  "%.4f x CPU hits ms, R^2 %.3f (misses up to %d a token: drop-all extrapolates)" % (
                      f["n"], f["mean_ms"], f["a"], f["b"], f["c"], f["r2"], f["max_misses"]))
        rows = [(k, v) for k, v in rp.items() if isinstance(v, dict) and k != "trace_fit"]
        if rows:
            pols = list(rows[0][1])
            print("    %-30s" % "model" + "".join("%15s" % (p_ if p_ != "cap" else "cap %d" % cap) for p_ in pols))
            for k, v in rows:
                print("    %-30s" % k + "".join("%15s" % ("%.2f (%.1f)" % (v[p_]["tok_s"], v[p_]["ms"])) for p_ in pols))
            print("    %-30s" % "drops a token" + "".join("%15.2f" % rows[0][1][p_]["drops"] for p_ in pols))
    if out_json:
        js = dict(prefix=prefix, tokens=len(toks), problems=d["problems"][:100],
                  status=g["status"], all=a,
                  families={f: g["families"][f].row(tau) for f in g["families"]},
                  layers={str(l): g["layers"][l].row(tau) for l in g["layers"]},
                  slots=slots, replay=rp, tau=tau, delta=delta, cap=cap)
        with open(out_json, "w") as f:
            json.dump(js, f, indent=1)
    return d, g, rp


def dump(prefix, n=12):
    for s in range(5):
        p = path_of(prefix, s)
        if not os.path.exists(p):
            print("%s: none" % p)
            continue
        h, recs, b = read_records(p)
        print("%s: %s, %d records" % (p, h, len(recs)))
        for t, fl, sz, o in recs[:n]:
            if t == T_PASS:
                print("  pass", PASS.unpack_from(b, o))
            elif t == T_LAYER:
                print("  layer", LAYER.unpack_from(b, o), "bytes", sz)
            elif t == T_STEP:
                print("  step", STEP.unpack_from(b, o))
            elif t == T_END:
                print("  end", END.unpack_from(b, o))
            else:
                print("  type %d flags %d bytes %d" % (t, fl, sz))


# ---- a synthetic trace ---------------------------------------------------------------------------

class Writer:
    """The engine's layout from Python, for synth."""

    def __init__(self, prefix, s, t0=1000):
        self.f = open(path_of(prefix, s), "wb")
        self.f.write(HEADER.pack(MAGIC, 1, s, 4, 384, 6, 96, 40, 64, t0) + b"\0" * 16)

    def rec(self, t, payload, flags=0):
        n = 8 + len(payload)
        pad = (-n) % 4
        self.f.write(REC.pack(t, flags, n + pad) + payload + b"\0" * pad)

    def close(self):
        self.f.close()


def synth(prefix, arm_json):
    """Two requests (families alpha, beta), a 5-token prompt each and 5 reply tokens: 4 decode
    passes each, and request alpha has a post-stop pass behind its last. Every layer of every pass
    picks the same six file-tier experts (all RAM hits, H), keys 2.0 .. 1.5 in slot order, weights
    summing to 1; each rank's local experts 0..9 are on the card, the rest on the file tier, and no
    RAM-cache snapshot holds anything. Layer 3: the slot-5 pick (global 340, rank 3's local 52) is
    a miss (M) with weight share 0.05, and an unselected expert on the card (global 200, rank 2's
    local 8) has a key 2% below the 6th (and global 211, 1% below, is on the file tier, not
    cached: no substitute). Layer 17: the slot-1 pick (global 330) is a miss with share 0.25, and
    the nearest held expert (global 290, on the card) is 10% below. So per decode token: 2 misses,
    1 below tau 0.1, 1 with a substitute at delta 5%, 1 at slot 2 or below, 238 CPU hits; the
    post-stop pass adds 2 misses the gates leave out. Measured: 60 ms a token."""
    ranks = [Writer(prefix, r) for r in range(4)]
    stw = Writer(prefix, STEPS)
    tiers = bytes([0] * 10 + [3] * 86)
    for r in range(4):
        ranks[r].rec(T_TIERS, struct.pack("<HH", 40, 96) + b"".join(
            struct.pack("<HH", l, 0) + tiers for l in range(40)))
    for l in range(40):
        ranks[0].rec(T_BIAS, struct.pack("<HH", l, 384) + struct.pack("<384f", *([0.0] * 384)))

    def inv_score(s):   # the logit whose score is s
        return math.log(math.expm1(s * s)) if s * s < 20 else s * s

    step, pass_, t = 0, 0, 10 ** 9
    arm = []
    for q, (name, fam) in enumerate((("alpha-01", "alpha"), ("beta-01", "beta"))):
        seq = 100 + q
        step += 1; pass_ += 1; t += 10 ** 7   # the prompt: one 5-token pass
        stw.rec(T_STEP, STEP.pack(step, 1, 0, 5, 1, 0, 0, t) + ENTRY.pack(seq, 0, 5, E_PREFILL | E_PRODUCES, 0, 0))
        for r in range(4):
            ranks[r].rec(T_PASS, PASS.pack(step, pass_, 1, 0, 5, 1, 0, 0, 0, 0, t))
        for k in range(5 if q == 0 else 4):
            step += 1; pass_ += 1; t += 50 * 10 ** 6
            stw.rec(T_STEP, STEP.pack(step, 2, 0, 1, 1, 1, 0, t) + ENTRY.pack(seq, 5 + k, 1, E_PRODUCES | E_DEV, 0, 0))
            for r in range(4):
                ranks[r].rec(T_PASS, PASS.pack(step, pass_, 2, 0, 1, 1, 0, 1, 5 + k, 0, t))
            for layer in range(40):
                ids = [10, 116, 150, 250, 30, 340]
                w = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
                miss = None
                if layer == 3:
                    w, miss = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05], 340
                if layer == 17:
                    ids[1] = 330
                    w, miss = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10], 330
                lg = [-30.0] * 384
                for e, key in zip(ids, (2.0, 1.9, 1.8, 1.7, 1.6, 1.5)):
                    lg[e] = inv_score(key)
                if layer == 3:
                    lg[200] = inv_score(1.5 * 0.98)
                    lg[211] = inv_score(1.5 * 0.99)
                if layer == 17:
                    lg[290] = inv_score(1.5 * 0.90)
                lg = list(struct.unpack("<384f", struct.pack("<384f", *lg)))   # as the card's f32
                for r in range(4):
                    mine = sorted(e for e in ids if e // 96 == r)
                    fl = F_DECIDED | F_RAM | ((F_ROUTE | F_LOGITS) if r == 0 else 0)
                    payload = LAYER.pack(pass_, step, layer, 1, len(mine), fl, 0, 100, 0, 0, 0)
                    payload += b"".join(PICK.pack(e, 1, 2 if e == miss else 1) for e in mine)
                    if r == 0:
                        payload += (struct.pack("<6H", *ids) + struct.pack("<6f", *w) +
                                    struct.pack("<6f", *[score(lg[e]) for e in ids]) +
                                    struct.pack("<384f", *lg))
                    ranks[r].rec(T_LAYER, payload)
        # alpha ended at its 4th decode step's commit with its 5th already issued: post-stop
        stw.rec(T_END, END.pack(step - 1, step, seq, 5) if q == 0 else END.pack(step, -1, seq, 5))
        arm.append(dict(name=name, family=fam, text="x", timings=dict(predicted_ms=4 * 60.0, predicted_n=4)))
    for x in ranks + [stw]:
        x.close()
    with open(arm_json, "w") as f:
        json.dump({"requests": arm}, f)

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(2)
    if a[0] == "report":
        opts = {"--tau": 0.1, "--delta": 0.05, "--cap": 6, "--insert-ms": 0.15, "--json": None}
        pos = []
        i = 1
        while i < len(a):
            if a[i] in opts:
                opts[a[i]] = a[i + 1] if a[i] == "--json" else float(a[i + 1])
                i += 2
            else:
                pos.append(a[i])
                i += 1
        report(pos[0], pos[1] if len(pos) > 1 else None, tau=opts["--tau"], delta=opts["--delta"],
               cap=int(opts["--cap"]), insert_ms=opts["--insert-ms"], out_json=opts["--json"])
    elif a[0] == "dump":
        dump(a[1], int(a[2]) if len(a) > 2 else 12)
    elif a[0] == "synth":
        synth(a[1], a[2])
    else:
        print(__doc__)
        sys.exit(2)

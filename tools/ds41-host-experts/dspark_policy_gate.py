#!/usr/bin/env python3
"""Configuration/startup/action guard only. No fitting, model calls or cost inference."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

MARKER = "DSpark prefix policy"
START = MARKER + ": on config="
ACTION = MARKER + " action: "
CYCLE = MARKER + " cycle: "
TOTALS = MARKER + " totals: "
CONCURRENT = MARKER + " concurrent: "
PAUSE = MARKER + " pause: "
RESUME = MARKER + " resume: "
ENV = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_PREFIX_POLICY"
SUFFIX = (f" ({ENV}); query_rows=2 unchanged at every action, observer=1, "
          "wait=step_end before issue_ahead; candidate_p=candidate-set-share, "
          "invalid_numeric=full-prefix2; action/cycle logging=on, fixed_measurement=off")
NUMBER = r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?"
SPEC = re.compile(rf"v1:({NUMBER}):({NUMBER})")


class Invalid(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Invalid(why)


def parse_config(text):
    if text in (None, "0", "off"):
        return None
    require(isinstance(text, str) and len(text) < 160, "policy spec absent/too long")
    match = SPEC.fullmatch(text)
    require(match is not None, "policy requires explicit v1:T1:T2 decimal thresholds")
    values = tuple(float(x) for x in match.groups())
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in values), "threshold outside [0,1]")
    return values


def check_arm(config, ds, db, nseq, conflicts=False, defer=False, native_bank=False, tbo=False):
    require(parse_config(config) is not None, "Dpp requires DSPARK_POLICY_CONFIG=v1:T1:T2")
    concurrent = isinstance(nseq, str) and nseq.isdecimal() and int(nseq) > 1
    require(ds == "Ds" and db == "2" and not conflicts and
            (nseq == "1" or concurrent and defer and native_bank and not tbo),
            "Dpp requires bare Ds/Db2, no Dme/Dc/Dl/Dg; concurrency additionally needs explicit Dx, native bank and no TBO")


def integer(x, low=0):
    return type(x) is int and x >= low


def check_log(text, config, complete=False, nseq=1):
    thresholds = parse_config(config)
    lines = text.splitlines()
    policy = [line for line in lines if MARKER in line]
    if thresholds is None:
        require(not policy, "policy marker in an arm without Dpp")
        return
    require(not any("DSpark matched early " in line for line in lines),
            "policy log contains fixed measurement markers")
    startup = [line.split(START, 1)[1] for line in lines if START in line]
    require(startup == [config + SUFFIX], "policy startup missing, duplicate or mismatched config/profile")
    require(type(nseq) is int and nseq >= 1, "invalid configured sequence bound")
    concurrent = nseq > 1
    extra = [line.split(CONCURRENT, 1)[1] for line in lines if CONCURRENT in line]
    if concurrent:
        require(len(extra) == 1, "concurrent policy requires one actual Scheduler solo/Dx startup proof")
        try: scope = json.loads(extra[0])
        except (ValueError, TypeError) as exc: raise Invalid("malformed concurrent scope") from exc
        require(isinstance(scope, dict) and scope.get("max_seqs") == nseq and scope.get("tp") == 4 and
                all(scope.get(k) is True for k in ("dsolo", "deferred", "native_bank")) and
                scope.get("tbo") is False and scope.get("query_rows") == 2 and scope.get("full_rows") == 5 and
                scope.get("depth") == 5 and integer(scope.get("from"), 1) and scope["from"] == 1 and integer(scope.get("bank_hot_n")) and
                scope["bank_hot_n"] <= 32, "wrong concurrent native-bank/query/solo scope")
    else:
        require(not extra and not any(PAUSE in x or RESUME in x for x in lines), "concurrent markers in the original S1 profile")
    allowed = (START, ACTION, CYCLE, TOTALS) + ((CONCURRENT, PAUSE, RESUME) if concurrent else ())
    require(all(any(marker in line for marker in allowed) for line in policy), "unknown policy marker")
    if not complete:
        return
    actions, cycles = {}, {}
    for marker, dest in ((ACTION, actions), (CYCLE, cycles)):
        for line in lines:
            if marker not in line:
                continue
            try:
                row = json.loads(line.split(marker, 1)[1])
            except (ValueError, TypeError) as exc:
                raise Invalid("malformed policy JSON") from exc
            require(isinstance(row, dict), "policy event is not an object")
            epoch = row.get("epoch" if marker == ACTION else "next_epoch")
            require(integer(epoch, 1) and epoch not in dest, "missing/duplicate policy query epoch")
            dest[epoch] = row
    require(actions and actions.keys() == cycles.keys(), "every policy query needs action and outcome")
    valid_costs = [0, 0, 0]
    for epoch, action in actions.items():
        require(integer(action.get("seq"), 1) and integer(action.get("ctx")) and
                integer(action.get("card")) and integer(action.get("step")) and
                integer(action.get("anchor")) and action.get("query_rows") == 2,
                "policy action lacks query binding")
        require(isinstance(action.get("ids"), list) and len(action["ids"]) == 2 and
                all(integer(x) for x in action["ids"]), "policy action lacks full Db2 IDs")
        signal, mask = action.get("candidate_p"), action.get("invalid_depths")
        require(isinstance(signal, list) and len(signal) == 2 and integer(mask) and mask <= 3,
                "missing policy signal/fallback mask")
        bad = any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in signal)
        fallback = bool(mask or bad)
        require(not bad or mask != 0, "invalid candidate share without numeric-invalid depth")
        k = 2 if fallback else 0 if signal[0] < thresholds[0] else 1 if signal[1] < thresholds[1] else 2
        require(type(action.get("k")) is int and action["k"] == k and
                type(action.get("fallback")) is bool and action["fallback"] == fallback,
                "action differs from supplied hierarchical thresholds/full-prefix fallback")
        require(integer(action.get("wait_ns")) and type(action.get("pure_decode")) is bool,
                "action lacks actual wait/profile")
        if concurrent:
            require(integer(action.get("frame"), 1) and integer(action.get("solo_span"), 1) and
                    integer(action.get("owner_slot")) and action["card"] < nseq,
                    "action lacks actual sole-owner frame/span/slot")
        cycle = cycles[epoch]
        require(cycle.get("seq") == action["seq"] and cycle.get("next_k") == k and
                cycle.get("verify_step") == action["step"] and cycle.get("wait_ns") == action["wait_ns"] and
                cycle.get("next_invalid_depths") == mask and integer(cycle.get("output")),
                "outcome/query binding differs")
        if concurrent:
            require(cycle.get("frame") == action["frame"] and cycle.get("solo_span") == action["solo_span"] and
                    integer(cycle.get("previous_span")), "outcome lost sole-owner frame/span")
        require(all(type(cycle.get(key)) is bool for key in ("cost_valid", "live", "published")),
                "outcome missing lifetime flags")
        require(integer(cycle.get("epoch")) and integer(cycle.get("cycle_ns")), "missing measured cycle")
        if cycle["cost_valid"]:
            previous = actions.get(cycle["epoch"])
            require(previous is not None and cycle["epoch"] < epoch and cycle["cycle_ns"] > 0 and
                    cycle["live"] and cycle["published"] and previous["pure_decode"] and action["pure_decode"] and
                    previous["seq"] == action["seq"] and previous["card"] == action["card"] and
                    cycle.get("k") == previous["k"] and
                    integer(cycle.get("accepted")) and cycle["accepted"] <= previous["k"] and
                    action["ctx"] - previous["ctx"] == 1 + cycle["accepted"] and
                    cycle.get("proposal_invalid_depths") == previous["invalid_depths"],
                    "cost lacks exact previous-action/context/commit binding")
            if concurrent:
                require(action["step"] == previous["step"] + 1 and action["frame"] == previous["frame"] + 1 and
                        action["owner_slot"] == previous["owner_slot"] and
                        action["solo_span"] == previous["solo_span"] == cycle["previous_span"],
                        "busy/idle/reused-frame gap priced as a pure solo cycle")
            valid_costs[previous["k"]] += 1
    if concurrent:
        check_concurrent_events(lines, actions, cycles)
    require(sum(valid_costs) > 0, "policy has no bound pure-decode cycle costs")
    totals = [line.split(TOTALS, 1)[1] for line in lines if TOTALS in line]
    require(len(totals) == 1, "policy totals missing/duplicate")
    fields = dict(re.findall(r"([a-z_]+)=([0-9]+)", totals[0]))
    n = len(actions)
    require(all(int(fields.get(key, -1)) == expected for key, expected in
                (("queries", n), ("query_rows", 2*n), ("observers", n), ("waits", n))),
            "policy query/observer/wait totals differ")
    require(f"costs=[{','.join(map(str, valid_costs))}]" in totals[0], "policy cost counts differ")


def check_concurrent_events(lines, actions, cycles):
    ordered = list(actions.values())
    require(all(b["epoch"] > a["epoch"] and b["frame"] > a["frame"] and b["step"] > a["step"] and
                b["solo_span"] >= a["solo_span"] for a, b in zip(ordered, ordered[1:])),
            "nonmonotone/duplicate query owner frame")
    require(ordered[0]["solo_span"] == 1, "missing initial solo span")
    pauses, resumes = {}, {}
    for marker, dest, key in ((PAUSE, pauses, "next_span"), (RESUME, resumes, "solo_span")):
        for line in lines:
            if marker not in line: continue
            try: row = json.loads(line.split(marker, 1)[1])
            except (ValueError, TypeError) as exc: raise Invalid("malformed pause/resume") from exc
            require(isinstance(row, dict) and integer(row.get(key), 2) and row[key] not in dest,
                    "duplicate/invalid pause/resume span")
            dest[row[key]] = row
    for span, pause in pauses.items():
        require(all(integer(pause.get(k), 1) for k in ('seq', 'epoch', 'previous_span')),
                'pause lacks numeric identity/span')
        old = actions.get(pause.get("epoch"))
        require(old is not None and pause.get("seq") == old["seq"] and
                pause.get("previous_span") == old["solo_span"] == span-1 and pause.get("query_rows") == 0 and
                pause.get("reason") in ("idle", "no_query") and cycles[old["epoch"]]["published"] and
                cycles[old["epoch"]]["live"], "pause not bound to the previous published query")
        require(old is [a for a in ordered if a['solo_span'] == span-1][-1],
                "pause reused a stale query epoch")
        require((pause["reason"] == "idle" and pause.get("step") == -1 and pause.get("frame") == 0) or
                (pause["reason"] == "no_query" and integer(pause.get("step")) and
                 pause["step"] > old["step"] and integer(pause.get("frame"), 1) and pause["frame"] > old["frame"]),
                "pause not bound to an actual later no-query/idle frame")
    for span, resume in resumes.items():
        require(all(integer(resume.get(k), 1) for k in ('seq', 'epoch', 'previous_seq', 'previous_epoch')) and
                all(integer(resume.get(k)) for k in ('step', 'frame', 'card', 'owner_slot', 'ctx')),
                'resume lacks numeric identity/frame')
        pause = pauses.get(span)
        current = actions.get(resume.get("epoch"))
        require(pause is not None and current is not None and resume.get("ready") is True and
                resume.get("query_rows") == 2 and resume.get("previous_seq") == pause["seq"] and
                resume.get("previous_epoch") == pause["epoch"], "resume lacks its exact pause/query")
        require(all(resume.get(k) == current[k] for k in ("seq", "step", "frame", "solo_span", "card", "owner_slot", "ctx")),
                "resume rebound another slot/anchor/frame")
        first = next((a for a in ordered if a["solo_span"] == span), None)
        require(first is current and not cycles[current["epoch"]]["cost_valid"] and
                cycles[current["epoch"]]["epoch"] == 0 and cycles[current["epoch"]]["cycle_ns"] == 0,
                "first resumed query priced the paused interval")
    for action in ordered:
        if action["solo_span"] > 1:
            require(action["solo_span"] in pauses and action["solo_span"] in resumes,
                    "query after pause lacks actual resume proof")
    require(all(span in pauses for span in resumes), "resume without pause")


def check_mixed_resume(text, anchor, minimum_ctx=None):
    """Live-prefix proof for the mixed client, including a legitimate resumed k0 query.
    Complete totals, all query/outcome checks and cross-run capture comparisons remain offline."""
    require(integer(anchor, 1), "missing actual anchor sequence")
    rows = {}
    for marker in (ACTION, CYCLE, PAUSE, RESUME):
        rows[marker] = []
        for line in text.splitlines():
            if marker not in line: continue
            try: rows[marker].append(json.loads(line.split(marker, 1)[1]))
            except (ValueError, TypeError) as exc: raise Invalid("malformed mixed policy proof") from exc
    matches = []
    for resume in rows[RESUME]:
        if resume.get('seq') != anchor or resume.get('previous_seq') != anchor: continue
        if minimum_ctx is not None and (not integer(resume.get('ctx')) or resume['ctx'] <= minimum_ctx): continue
        actions = [a for a in rows[ACTION] if a.get('epoch') == resume.get('epoch') and a.get('seq') == anchor]
        pauses = [p for p in rows[PAUSE] if p.get('epoch') == resume.get('previous_epoch') and p.get('seq') == anchor]
        cycles = [c for c in rows[CYCLE] if c.get('next_epoch') == resume.get('epoch') and c.get('seq') == anchor]
        require(len(actions) == len(pauses) == len(cycles) == 1, "mixed resume lacks unique actual action/pause/outcome")
        action, pause, cycle = actions[0], pauses[0], cycles[0]
        require(resume.get('ready') is True and resume.get('query_rows') == 2 and
                all(resume.get(k) == action.get(k) for k in ('seq', 'step', 'frame', 'solo_span', 'card', 'owner_slot', 'ctx')) and
                pause.get('next_span') == action.get('solo_span') and pause.get('reason') == 'no_query' and
                action.get('frame', -1) > pause.get('frame', -1) and action.get('step', -1) > pause.get('step', -1),
                "mixed resume rebound an idle/stale/foreign owner")
        require(cycle.get('published') is True and cycle.get('live') is True and cycle.get('cost_valid') is False and
                cycle.get('epoch') == 0 and cycle.get('cycle_ns') == 0,
                "resumed query was unpublished or charged its concurrent gap")
        matches.append(dict(seq=anchor, epoch=action['epoch'], step=action['step'], frame=action['frame'],
                            ctx=action['ctx'], solo_span=action['solo_span'], k=action['k']))
    require(matches, "no actual surviving-anchor policy query after a no-query pause")
    return matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--nseq", type=int, default=1)
    parser.add_argument("--ctx-defer", choices=("0", "1"), default="0")
    parser.add_argument("--native-bank", choices=("0", "1"), default="0")
    parser.add_argument("--tbo", choices=("0", "1"), default="0")
    parser.add_argument("--arm", nargs=4, metavar=("DS", "DB", "NSEQ", "CONFLICTS"))
    args = parser.parse_args(argv)
    try:
        parse_config(args.config)
        if args.arm:
            check_arm(args.config, *args.arm[:3], conflicts=args.arm[3] != "0", defer=args.ctx_defer == "1",
                      native_bank=args.native_bank == "1", tbo=args.tbo == "1")
        if args.log:
            check_log(args.log.read_text(), args.config, args.complete, args.nseq)
    except (Invalid, OSError) as exc:
        print(f"DSpark policy gate: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

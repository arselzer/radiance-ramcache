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


def check_arm(config, ds, db, nseq, conflicts=False):
    require(parse_config(config) is not None, "Dpp requires DSPARK_POLICY_CONFIG=v1:T1:T2")
    require(ds == "Ds" and db == "2" and nseq == "1" and not conflicts,
            "Dpp requires bare Ds, Db2, one sequence and no Dme/Dc/Dl/Dg")


def integer(x, low=0):
    return type(x) is int and x >= low


def check_log(text, config, complete=False):
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
    allowed = (START, ACTION, CYCLE, TOTALS)
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
        cycle = cycles[epoch]
        require(cycle.get("seq") == action["seq"] and cycle.get("next_k") == k and
                cycle.get("verify_step") == action["step"] and cycle.get("wait_ns") == action["wait_ns"] and
                cycle.get("next_invalid_depths") == mask and integer(cycle.get("output")),
                "outcome/query binding differs")
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
            valid_costs[previous["k"]] += 1
    require(sum(valid_costs) > 0, "policy has no bound pure-decode cycle costs")
    totals = [line.split(TOTALS, 1)[1] for line in lines if TOTALS in line]
    require(len(totals) == 1, "policy totals missing/duplicate")
    fields = dict(re.findall(r"([a-z_]+)=([0-9]+)", totals[0]))
    n = len(actions)
    require(all(int(fields.get(key, -1)) == expected for key, expected in
                (("queries", n), ("query_rows", 2*n), ("observers", n), ("waits", n))),
            "policy query/observer/wait totals differ")
    require(f"costs=[{','.join(map(str, valid_costs))}]" in totals[0], "policy cost counts differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--arm", nargs=4, metavar=("DS", "DB", "NSEQ", "CONFLICTS"))
    args = parser.parse_args(argv)
    try:
        parse_config(args.config)
        if args.arm:
            check_arm(args.config, *args.arm[:3], conflicts=args.arm[3] != "0")
        if args.log:
            check_log(args.log.read_text(), args.config, args.complete)
    except (Invalid, OSError) as exc:
        print(f"DSpark policy gate: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

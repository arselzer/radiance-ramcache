#!/usr/bin/env python3
"""Validate read-only DSpark traces; calibrate and fit one-decision threshold candidates.

No engine policy, network, model loading, or GPU use. Candidate-set softmax is not an acceptance
probability. Measured C0/C1/C2 must all query the SAME drafter/observer with the SAME readback wait;
Ds0/plain C0 is unsuitable. Held-out data never selects thresholds or the fixed-depth reference.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

MARKER = "DSpark proposal capture: "
IMMUTABLE = ("v", "seq", "epoch", "anchor_pos", "pos", "depth", "token", "predecessor",
             "candidate_count", "candidate_p", "score", "margin", "margin_unbounded", "signal_valid")
DISCARDED = {"verify_cap", "request_closed", "not_published", "superseded", "proposal_mismatch",
             "not_verified", "concurrency", "preempted", "reaped", "shutdown", "scheduler_fini",
             "slot_reused", "cancelled"}


class Invalid(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Invalid(why)


def integer(x, minimum=0):
    return type(x) is int and x >= minimum


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def reject_constant(value):
    raise Invalid("nonstandard JSON constant " + value)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def named_paths(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        require(sep and name and path and name not in result, "expected unique RUN=PATH: " + value)
        result[name] = path
    return result


def validate_event(e):
    require(isinstance(e, dict) and all(k in e for k in IMMUTABLE), "missing trace identity/signal fields")
    require(e["v"] == 1 and type(e["v"]) is int, "unsupported trace version")
    for k in ("seq", "epoch", "candidate_count"):
        require(integer(e[k], 1), "invalid " + k)
    for k in ("anchor_pos", "pos", "token", "predecessor"):
        require(integer(e[k]), "invalid " + k)
    require(e["depth"] in (1, 2) and type(e["depth"]) is int, "depth is not 1/2")
    require(e["pos"] == e["anchor_pos"] + e["depth"], "position is not anchor+depth")
    require(e["candidate_count"] <= 32, "candidate count exceeds selector scope")
    for k in ("signal_valid", "margin_unbounded", "cancelled"):
        require(type(e.get(k)) is bool, "invalid boolean " + k)
    for k in ("candidate_p", "score", "margin"):
        require(e[k] is None or finite(e[k]), "nonfinite/non-numeric JSON signal " + k)
    if e["signal_valid"]:
        require(finite(e["candidate_p"]) and 0 <= e["candidate_p"] <= 1, "invalid candidate_p")
        require(e["candidate_p"] + 1e-6 >= 1 / e["candidate_count"], "winner share is below 1/K")
        require(finite(e["score"]), "valid signal has no score")
        require((finite(e["margin"]) and e["margin"] >= 0 and not e["margin_unbounded"]) or
                (e["margin"] is None and e["margin_unbounded"]), "invalid winning margin")
    if e.get("event") == "proposal":
        require(e.get("status") == "pending" and all(e.get(k) is None for k in
                ("verified", "reached", "accepted")), "proposal already carries an outcome")
        return
    require(e.get("event") == "outcome", "unknown event")
    require(type(e.get("verified")) is bool and type(e.get("reached")) is bool,
            "outcome needs verified/reached booleans")
    expected = {"accepted": (True, True, True), "rejected": (True, True, False),
                "unreached": (True, False, None)}
    status = e.get("status")
    if status in expected:
        require(tuple(e.get(k) for k in ("verified", "reached", "accepted")) == expected[status],
                "inconsistent " + status + " outcome")
        if status != "unreached":
            require(type(e["accepted"]) is bool, "accepted must be boolean when reached")
        else:
            require(e["depth"] == 2, "depth1 cannot be unreached")
    else:
        require(status in DISCARDED and not e["verified"] and not e["reached"] and e.get("accepted") is None,
                "discarded outcome is mislabeled as a trial")


def pair_events(run, lines, allow_incomplete=False):
    records, epochs, outcomes = {}, {}, {}
    last_epoch = 0
    candidates = None
    for number, line in enumerate(lines, 1):
        if MARKER in line:
            raw = line.split(MARKER, 1)[1].strip()
        elif line.lstrip().startswith("{"):
            raw = line.strip()
        else:
            continue
        try:
            e = json.loads(raw, parse_constant=reject_constant)
            if MARKER not in line and (not isinstance(e, dict) or "event" not in e):
                continue
            validate_event(e)
            if candidates is None:
                candidates = e["candidate_count"]
            require(e["candidate_count"] == candidates, "candidate count changed within run")
            key = (e["seq"], e["epoch"], e["depth"])
            identity = (e["seq"], e["anchor_pos"])
            require(epochs.setdefault(e["epoch"], identity) == identity, "epoch reused for another sequence/anchor")
            if e["event"] == "proposal":
                require(key not in records and e["epoch"] >= last_epoch, "duplicate/nonmonotone proposal epoch")
                last_epoch = e["epoch"]
                records[key] = e
            else:
                require(key in records and key not in outcomes, "outcome has no unique preceding proposal")
                require(all(e[k] == records[key][k] for k in IMMUTABLE), "outcome changed proposal ID/position/signal")
                require(not records[key]["cancelled"] or e["cancelled"], "cancellation was reversed")
                require(e["status"] != "proposal_mismatch", "engine reported a proposal identity mismatch")
                outcomes[key] = e
        except (Invalid, ValueError, TypeError) as error:
            raise Invalid(f"{run}:{number}: {error}") from error
    require(records, run + ": no proposal records")
    grouped = defaultdict(dict)
    for key, p in records.items():
        grouped[key[:2]][key[2]] = (p, outcomes.get(key))
    for key, depths in grouped.items():
        require(1 in depths, f"{run}:{key}: depth2 without depth1")
        if 2 not in depths:
            continue
        a, b = depths[1][0], depths[2][0]
        require(b["predecessor"] == a["token"] and b["anchor_pos"] == a["anchor_pos"],
                f"{run}:{key}: depth2 is not conditioned on depth1's ID")
        oa, ob = depths[1][1], depths[2][1]
        if oa and ob and ob["verified"]:
            require(oa["verified"] and oa["reached"], f"{run}:{key}: depth2 verified without depth1")
            require(ob["reached"] == oa["accepted"], f"{run}:{key}: reached prefix contradicts depth1 acceptance")
    missing = len(records) - len(outcomes)
    require(allow_incomplete or not missing, f"{run}: {missing} proposal(s) lack terminal outcomes")
    return [{"run": run, "proposal": p, "outcome": outcomes.get(k)} for k, p in records.items()], missing


def load_metadata(path):
    data = read_json(path)
    require(isinstance(data, dict) and data.get("schema") == "ds41-proposal-metadata-v1", "wrong metadata schema")
    require(isinstance(data.get("requests"), list), "metadata requests must be a list")
    result, explicit_splits = {}, {}
    for row in data.get("requests", []):
        require(isinstance(row, dict) and integer(row.get("seq"), 1), "invalid metadata seq")
        for k in ("run", "family", "prompt_key"):
            require(isinstance(row.get(k), str) and row[k], "missing metadata " + k)
        require(row["family"] != "all", "family 'all' is reserved for the aggregate")
        row = dict(row)
        row["split_group"] = row.get("split_group", row["prompt_key"])
        require(isinstance(row["split_group"], str) and row["split_group"], "empty split group")
        key = (row["run"], row["seq"])
        require(key not in result, "duplicate metadata request " + str(key))
        if "split" in row:
            require(row["split"] in ("train", "heldout"), "invalid explicit split")
            group = row["split_group"]
            require(explicit_splits.setdefault(group, row["split"]) == row["split"], "split-group leakage")
        result[key] = row
    for row in result.values():
        if row["split_group"] in explicit_splits:
            row["split"] = explicit_splits[row["split_group"]]
    return result


def split_of(row, seed, heldout_fraction):
    if "split" in row:
        return row["split"]
    digest = hashlib.sha256((seed + "\0" + row["split_group"]).encode()).digest()
    return "heldout" if int.from_bytes(digest[:8], "big") / 2**64 < heldout_fraction else "train"


def wilson(accepted, n):
    if not n:
        return None
    z = 1.959963984540054
    p, den = accepted / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0, center-half), min(1, center+half)]


def calibration(records, bins):
    grouped = defaultdict(list)
    for row in records:
        p, o = row["proposal"], row["outcome"]
        if not o or o["cancelled"] or not o["reached"] or not p["signal_valid"]:
            continue
        b = min(bins-1, int(p["candidate_p"] * bins))
        for family in ("all", row["meta"]["family"]):
            grouped[(family, p["depth"], row["split"], b)].append(row)
    result = []
    for (family, depth, split, b), rows in sorted(grouped.items()):
        accepted = sum(r["outcome"]["accepted"] for r in rows)
        result.append({"family": family, "depth": depth, "split": split, "lo": b/bins,
                       "hi": (b+1)/bins, "n": len(rows), "accepted": accepted,
                       "acceptance_rate": accepted/len(rows), "binomial_wilson95_descriptive": wilson(accepted, len(rows)),
                       "mean_candidate_p": sum(r["proposal"]["candidate_p"] for r in rows)/len(rows),
                       "split_groups": len({r["meta"]["split_group"] for r in rows})})
    return result


def dispositions(records):
    grouped = defaultdict(Counter)
    for row in records:
        p, o = row["proposal"], row["outcome"]
        for family in ("all", row["meta"]["family"]):
            counts = grouped[(family, p["depth"], row["split"])]
            counts["proposals"] += 1
            counts["invalid_signals"] += not p["signal_valid"]
            counts["status:" + (o["status"] if o else "missing_outcome")] += 1
            counts["cancelled_outcomes"] += bool(o and o["cancelled"])
    return [{"family": f, "depth": d, "split": s, "counts": dict(c)}
            for (f, d, s), c in sorted(grouped.items())]


def replay_rows(records, trim):
    epochs, by_request = defaultdict(dict), defaultdict(list)
    for row in records:
        p = row["proposal"]
        epochs[(row["run"], p["seq"], p["epoch"])][p["depth"]] = row
    excluded = Counter()
    for key, depths in epochs.items():
        if set(depths) != {1, 2} or any(not r["outcome"] or not r["outcome"]["verified"] for r in depths.values()):
            excluded["not_fixed_depth2_verified"] += 1
            continue
        if any(r["outcome"]["cancelled"] or not r["proposal"]["signal_valid"] for r in depths.values()):
            excluded["cancelled_or_invalid_signal"] += 1
            continue
        a, b = depths[1], depths[2]
        row = {"family": a["meta"]["family"], "group": a["meta"]["split_group"], "split": a["split"],
               "run": key[0], "seq": key[1], "epoch": key[2], "p1": a["proposal"]["candidate_p"],
               "p2": b["proposal"]["candidate_p"], "a1": int(a["outcome"]["accepted"]),
               # None after a rejected predecessor is NOT a conditional negative trial. It
               # contributes zero to the accepted PREFIX length, which is nevertheless known.
               "a2": int(b["outcome"]["accepted"] is True)}
        by_request[key[:2]].append(row)
    kept = []
    for rows in by_request.values():
        rows.sort(key=lambda r: r["epoch"])
        kept.extend(rows[trim:len(rows)-trim] if trim else rows)
        excluded["request_boundary_epochs"] += min(len(rows), 2*trim)
    return kept, dict(excluded)


def load_costs(path):
    costs = read_json(path)
    require(isinstance(costs, dict) and costs.get("schema") == "ds41-proposal-costs-v1", "wrong cost schema")
    require(costs.get("same_drafter_observer_wait") is True,
            "C0/C1/C2 must all query the same drafter/observer and take the same wait; Ds0 is not C0")
    require(isinstance(costs.get("measurement_profile"), str) and costs["measurement_profile"], "cost profile needs provenance")
    require(isinstance(costs.get("families"), dict), "cost families must be an object")
    for family, row in costs.get("families", {}).items():
        require(isinstance(row, dict), "invalid cost family " + family)
        require(isinstance(row.get("source"), str) and row["source"], "cost family needs measured source: " + family)
        values = [row.get("c%d_ms" % k) for k in range(3)]
        require(all(finite(x) and x > 0 for x in values) and values[0] <= values[1] <= values[2],
                "costs must be measured positive, nondecreasing C0/C1/C2: " + family)
    require(costs.get("families"), "no measured step costs")
    return costs


def cost_for(costs, family, action):
    row = costs["families"].get(family, costs["families"].get("all"))
    require(row is not None, "no matched measured costs for family " + family)
    return row["c%d_ms" % action]


def evaluate(rows, costs, thresholds=None, fixed=None):
    tokens, milliseconds, actions = 0, 0.0, Counter()
    for row in rows:
        action = fixed if fixed is not None else (0 if row["p1"] < thresholds[0] else
                                                   1 if row["p2"] < thresholds[1] else 2)
        tokens += 1 + (row["a1"] if action >= 1 else 0) + (row["a2"] if action >= 2 else 0)
        milliseconds += cost_for(costs, row["family"], action)
        actions[action] += 1
    return {"epochs": len(rows), "tokens": tokens, "modeled_ms": milliseconds,
            "tokens_per_ms": tokens/milliseconds if milliseconds else None,
            "actions": dict(sorted(actions.items()))}


def bootstrap_gain(rows, costs, thresholds, fixed, samples, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    names = sorted(groups)
    if len(names) < 2 or not samples:
        return None
    rng, gains = random.Random(seed), []
    for _ in range(samples):
        draw = [r for _ in names for r in groups[rng.choice(names)]]
        p = evaluate(draw, costs, thresholds=thresholds)["tokens_per_ms"]
        b = evaluate(draw, costs, fixed=fixed)["tokens_per_ms"]
        gains.append(p/b - 1)
    gains.sort()
    return [gains[int((len(gains)-1)*q)] for q in (.025, .975)]


def threshold_grid(rows, signal, points):
    values = sorted({r[signal] for r in rows})
    if len(values) > points:
        values = [values[round(i*(len(values)-1)/(points-1))] for i in range(points)]
    # Training quantiles retain resolution when most shares are near1.0. Zero means always
    # eligible;2 means never. Held-out signal values must not determine these cut points.
    return sorted({0.0, 2.0, *values})


def fit_candidates(rows, costs, grid_points, min_train_groups, min_heldout_groups, min_trials, bootstrap, seed):
    result = []
    # "all" is a single global policy. Family-specific fits are diagnostics: the engine does
    # not have a code/chat semantic classifier, and the harness sends both as completions.
    for family in ["all"] + sorted({r["family"] for r in rows}):
        selected = [r for r in rows if family == "all" or r["family"] == family]
        train = [r for r in selected if r["split"] == "train"]
        heldout = [r for r in selected if r["split"] == "heldout"]
        ng = lambda rs: len({r["group"] for r in rs})
        row = {"family": family, "train_epochs": len(train), "heldout_epochs": len(heldout),
               "train_groups": ng(train), "heldout_groups": ng(heldout), "family_fit_is_diagnostic": family != "all",
               "train_conditional_depth2_trials": sum(r["a1"] for r in train),
               "heldout_conditional_depth2_trials": sum(r["a1"] for r in heldout)}
        if (ng(train) < min_train_groups or ng(heldout) < min_heldout_groups or
                len(train) < min_trials or len(heldout) < min_trials):
            row["status"] = "insufficient_request_group_support"
            result.append(row)
            continue
        train_fixed = [evaluate(train, costs, fixed=k) for k in range(3)]
        reference = max(range(3), key=lambda k: (train_fixed[k]["tokens_per_ms"], -k))
        grid1, grid2 = threshold_grid(train, "p1", grid_points), threshold_grid(train, "p2", grid_points)
        best, best_eval, best_key = None, None, None
        for t1 in grid1:
            for t2 in grid2:
                fit = evaluate(train, costs, thresholds=(t1, t2))
                key = (fit["tokens_per_ms"], -sum(int(k)*n for k, n in fit["actions"].items()), t1, t2)
                if best_key is None or key > best_key:
                    best, best_eval, best_key = (t1, t2), fit, key
        validation = evaluate(heldout, costs, thresholds=best)
        heldout_fixed = [evaluate(heldout, costs, fixed=k) for k in range(3)]
        gain = validation["tokens_per_ms"] / heldout_fixed[reference]["tokens_per_ms"] - 1
        interval = bootstrap_gain(heldout, costs, best, reference, bootstrap, seed)
        status = ("no_heldout_replay_gain" if gain <= 0 else "positive_heldout_replay_without_interval" if interval is None
                  else "inconclusive_heldout_replay" if interval[0] <= 0 else "candidate_for_on_policy_window")
        row.update(status=status, thresholds=list(best), train=best_eval,
                   heldout=validation, fixed_reference_chosen_on_train=reference,
                   train_fixed=train_fixed, heldout_fixed=heldout_fixed,
                   heldout_one_decision_gain_vs_train_selected_fixed=gain,
                   heldout_group_bootstrap95=interval, train_threshold_grid=[grid1, grid2])
        result.append(row)
    return result


def analyze(args):
    require(not args.costs or args.metadata, "threshold fitting needs explicit request metadata")
    records, incomplete = [], 0
    for run, path in named_paths(args.trace).items():
        with open(path, encoding="utf-8") as stream:
            rows, missing = pair_events(run, stream, args.allow_incomplete)
        records.extend(rows)
        incomplete += missing
    summary = Counter()
    for row in records:
        p, o = row["proposal"], row["outcome"]
        summary["proposals"] += 1
        summary["invalid_signals"] += not p["signal_valid"]
        if o:
            summary["outcome:" + o["status"]] += 1
            summary["cancelled_outcomes"] += o["cancelled"]
    result = {"schema": "ds41-proposal-analysis-v1", "validation": "incomplete" if incomplete else "passed", "counts": dict(summary),
              "incomplete_proposals": incomplete, "one_decision_replay_only": True,
              "limitations": ["Smaller verify depths change future query anchors; this is not a full on-policy simulation.",
                              "Conditional depth2 calibration excludes unreached/discarded/cancelled trials.",
                              "Thresholds and the fixed-depth reference are chosen only on train groups.",
                              "Historical0.49/0.70 acceptance break-evens are not candidate_p thresholds."]}
    if not args.metadata:
        result["calibration"] = "not_requested_without_explicit_request_metadata"
        return result
    metadata = load_metadata(args.metadata)
    for row in records:
        key = (row["run"], row["proposal"]["seq"])
        require(key in metadata, "no explicit request metadata for " + str(key))
        row["meta"] = metadata[key]
        row["split"] = split_of(row["meta"], args.seed, args.heldout_fraction)
    result["split"] = {"seed": args.seed, "heldout_fraction": args.heldout_fraction,
                       "group_field": "split_group (defaults to prompt_key)",
                       "groups": {g: sorted({r["meta"]["split_group"] for r in records if r["split"] == g})
                                  for g in ("train", "heldout")}}
    result["calibration"] = calibration(records, args.bins)
    result["dispositions"] = dispositions(records)
    mismatch = summary["outcome:proposal_mismatch"]
    if incomplete or mismatch:
        result["candidates"] = {"status": "disabled_for_incomplete_or_mismatched_capture"}
    elif args.costs:
        costs = load_costs(args.costs)
        rows, excluded = replay_rows(records, args.trim_epochs)
        result["cost_profile"] = costs
        result["replay_exclusions"] = excluded
        result["candidates"] = fit_candidates(rows, costs, args.grid_points, args.min_train_groups,
                                              args.min_heldout_groups, args.min_trials, args.bootstrap, args.seed)
    else:
        result["candidates"] = {"status": "awaiting_matched_measured_C0_C1_C2"}
    return result


def make_metadata(args):
    require(args.assert_sequential, "explicit --assert-sequential is required; request IDs are not saved in legacy results")
    rows = []
    for run, path in named_paths(args.results).items():
        data = read_json(path)
        require(isinstance(data, dict) and data.get("concurrency") == 1, "sequential mapping requires concurrency exactly1")
        requests = data.get("requests", [])
        require(isinstance(requests, list) and requests and all(isinstance(r, dict) and r.get("name") and r.get("family") and not r.get("error") and
                                r.get("text") is not None for r in requests), "cannot map missing/failed requests by order")
        end = args.first_seq + len(requests) - 1
        stats = data.get("stats") or {}
        require(isinstance(stats, dict), "invalid stats object")
        require(isinstance(stats.get("requests", []), list), "invalid stats requests list")
        known = [r["id"] for r in stats.get("requests", []) if isinstance(r, dict) and integer(r.get("id"), 1)]
        require(not known or (min(known) >= args.first_seq and max(known) == end),
                "stats IDs contradict sequential mapping; supply a manual manifest")
        for i, r in enumerate(requests):
            rows.append({"run": run, "seq": args.first_seq+i, "family": r["family"],
                         "prompt_key": r["name"], "split_group": r["name"], "results_source": str(path)})
    return {"schema": "ds41-proposal-metadata-v1", "mapping_basis":
            "caller_asserted_sequential_order_no_hidden_generation_requests", "requests": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--trace", action="append", required=True, metavar="RUN=SERVER_LOG")
    a.add_argument("--metadata")
    a.add_argument("--costs")
    a.add_argument("--allow-incomplete", action="store_true")
    a.add_argument("--seed", default="ds41-prob-v1")
    a.add_argument("--heldout-fraction", type=float, default=.25)
    a.add_argument("--bins", type=int, default=10)
    a.add_argument("--grid-points", type=int, default=21)
    a.add_argument("--min-train-groups", type=int, default=4)
    a.add_argument("--min-heldout-groups", type=int, default=2)
    a.add_argument("--min-trials", type=int, default=40)
    a.add_argument("--trim-epochs", type=int, default=1,
                   help="drop first/last eligible epoch per request from cost replay to avoid prefill/end-budget edges")
    a.add_argument("--bootstrap", type=int, default=200)
    m = sub.add_parser("make-metadata")
    m.add_argument("--results", action="append", required=True, metavar="RUN=ARM_JSON")
    m.add_argument("--assert-sequential", action="store_true")
    m.add_argument("--first-seq", type=int, default=1)
    for command in (a, m):
        command.add_argument("--out", help="JSON report/manifest; stdout if omitted")
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            require(0 < args.heldout_fraction < 1 and 1 <= args.bins <= 100 and
                    2 <= args.grid_points <= 101 and args.min_train_groups >= 1 and
                    args.min_heldout_groups >= 1 and args.min_trials >= 1 and
                    args.trim_epochs >= 0 and 0 <= args.bootstrap <= 10000, "invalid analysis limits")
            report = analyze(args)
        else:
            require(args.first_seq >= 1, "first sequence ID must be positive")
            report = make_metadata(args)
    except (Invalid, OSError, ValueError) as error:
        print("dspark probability analysis: " + str(error), file=sys.stderr)
        return 2
    try:
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except (OSError, ValueError) as error:
        print("dspark probability analysis output: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

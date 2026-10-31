#!/usr/bin/env python3
"""Validate explicit matched Db2 measurements and extract quiet C0/C1/C2. No fitting or engine work.

Run under the box-wide build lock/nice rules. Logs establish instrument evidence; deployment and
request/config authority MUST be supplied by the operator's manifest, not inferred from log text.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

_spec = importlib.util.spec_from_file_location("dspark_prob_analyze", Path(__file__).with_name("dspark_prob_analyze.py"))
prob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prob)
Invalid, require = prob.Invalid, prob.require

EARLY = "DSpark matched early verify: "
TOTALS = "DSpark matched early totals: "
COST = "DSpark matched early costs: "
CYCLE = "DSpark matched early cycle: "
QUIET = "quiet (debug logs follow RADIANCE_LOG/-v)"
START_RE = re.compile(r"on k=([012]) \(RADIANCE_EXPERIMENTAL_DS41_DSPARK_EARLY_VERIFY\); query_rows=([0-9]+) "
                      r"unchanged at every action, observer=1, wait=step_end before issue_ahead, "
                      r"logging=(quiet|capture); fixed measurement, no probability policy")
TOTAL_KEYS = ("queries", "query_rows", "observers", "waits", "wait_ns", "transition", "no_query",
              "end_cancel", "unpublished", "invalid_queries", "invalid_cycles", "unclosed_epoch")
COST_RE = re.compile(r"k=([012]) n=([0-9]+) cycle_ns=([0-9]+) wait_ns=([0-9]+) accepted=([0-9]+) "
                     r"emitted=([0-9]+) mean_ms=([0-9]+\.[0-9]{6}); "
                     r"consecutive same-sequence/epoch-bound pure-decode cycles only; "
                     r"includes query/observer/wait and logging=(.+)")
HEX64 = re.compile(r"[0-9a-f]{64}")
FORBIDDEN = ("experimental draft compute OFF:", "experimental draft verify cap:",
             "DSpark acceptance gate: on", "DSpark confidence head:",
             "DSpark load-aware drafting: on", "DSpark draft bank with up to ")


def integer(value, low=0):
    return type(value) is int and value >= low


def string(value):
    return isinstance(value, str) and bool(value.strip())


def hashed(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(base, value):
    require(string(value), "missing input path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def payload(line, marker):
    return line.split(marker, 1)[1].strip()


def parse_log(lines, intended_k, logging):
    starts, totals, costs, cycles = [], [], {}, []
    capture, proposal_lines, debug = 0, 0, 0
    for number, line in enumerate(lines, 1):
        require(not any(x in line for x in FORBIDDEN), f"line{number}: stray/conflicting measurement scope")
        capture += "DSpark read-only proposal capture: on" in line
        proposal_lines += prob.MARKER in line
        debug += bool(re.search(r"drafter 'dspark': sequence [0-9]+ proposes(?: |$)", line))
        if EARLY in line:
            match = START_RE.fullmatch(payload(line, EARLY))
            require(match is not None, f"line{number}: unknown/failed early scope")
            starts.append((int(match[1]), int(match[2]), match[3]))
        elif TOTALS in line:
            fields = payload(line, TOTALS).split()
            require(len(fields) == len(TOTAL_KEYS), "unknown totals fields (need invalid-signal-aware build)")
            row = {}
            for key, field in zip(TOTAL_KEYS, fields):
                require(re.fullmatch(key + r"=[0-9]+", field) is not None, "invalid totals field " + key)
                row[key] = int(field.split("=", 1)[1])
            totals.append(row)
        elif COST in line:
            match = COST_RE.fullmatch(payload(line, COST))
            require(match is not None, f"line{number}: unknown cost summary")
            k, n, ns, wait, accepted, emitted = map(int, match.groups()[:6])
            require(k not in costs, "duplicate cost summary")
            require(match[8] == (QUIET if logging == "quiet" else "capture"), "cost logging profile differs")
            mean = float(match[7])
            require(abs(mean - (ns/n*1e-6 if n else 0)) <= 0.00000050001, "rounded mean contradicts integer costs")
            costs[k] = dict(n=n, cycle_ns=ns, wait_ns=wait, accepted=accepted, emitted=emitted)
        elif CYCLE in line:
            cycles.append(json.loads(payload(line, CYCLE), parse_constant=prob.reject_constant))
        elif "DSpark matched early " in line:
            raise Invalid(f"line{number}: unknown early measurement marker")
    require(starts == [(intended_k, 2, logging)], "expected one intended Db2 early scope per server")
    require(len(totals) == 1 and set(costs) == {0, 1, 2}, "need one totals and exactly three costs summaries")
    require(capture == (1 if logging == "capture" else 0), "capture startup disagrees with declared logging")
    if logging == "quiet":
        require(not cycles and not proposal_lines and not debug, "quiet cost run contains JSON/debug proposal instrumentation")
    else:
        require(proposal_lines and cycles, "Dpc audit lacks proposal/cycle events")
    t, c = totals[0], costs[intended_k]
    q = t["queries"]
    require(q > 1 and t["observers"] == q and t["waits"] == q and t["query_rows"] == 2*q,
            "Db2 must have positive equal query/observer/wait counts and two rows per query")
    require(t["wait_ns"] > 0 and 0 < c["n"] < q and c["cycle_ns"] > 0 and c["wait_ns"] > 0,
            "intended prefix lacks positive steady cycles/waits")
    require(c["wait_ns"] <= c["cycle_ns"] and c["wait_ns"] <= t["wait_ns"], "wait cost exceeds containing interval")
    require(c["accepted"] <= intended_k*c["n"] and c["emitted"] == c["n"]+c["accepted"], "steady yield contradicts prefix")
    require(t["invalid_queries"] <= q and t["invalid_cycles"] <= min(q-1, 2*t["invalid_queries"]),
            "invalid-signal exclusions cannot come from the observed queries")
    excluded = sum(t[k] for k in ("transition", "no_query", "end_cancel", "unpublished", "invalid_cycles"))
    require(c["n"] + excluded + bool(t["unclosed_epoch"]) <= q, "steady/excluded counts exceed query boundaries")
    for k, row in costs.items():
        require(k == intended_k or not any(row.values()), "another fixed prefix accumulated costs")
    return dict(totals=t, costs=costs, cycles=cycles, debug_proposal_lines=debug, intended_k=intended_k)


def audit_cycles(run, lines, parsed, allowed_seqs):
    records, missing = prob.pair_events(run, lines, False)
    require(missing == 0, "unclosed capture audit")
    epochs = defaultdict(dict)
    for row in records:
        p = row["proposal"]
        require(p["seq"] in allowed_seqs, "audit query belongs to an undeclared request")
        epochs[(p["seq"], p["epoch"])][p["depth"]] = row
    require(all(set(d) == {1, 2} for d in epochs.values()), "Db2 audit needs both proposal depths per epoch")
    require(len(epochs) == parsed["totals"]["queries"], "capture epochs differ from query counter")
    masks = {key: sum((not depths[d]["proposal"]["signal_valid"]) << (d-1) for d in (1, 2))
             for key, depths in epochs.items()}
    require(sum(bool(m) for m in masks.values()) == parsed["totals"]["invalid_queries"], "invalid query mask count differs")
    unclosed = parsed["totals"]["unclosed_epoch"]
    require(not unclosed or any(epoch == unclosed for _, epoch in epochs), "unclosed epoch has no proposal")
    sums = {k: dict(n=0, cycle_ns=0, wait_ns=0, accepted=0, emitted=0) for k in range(3)}
    seen, next_seen, previous_step, invalid_cycles = set(), set(), -1, 0
    for c in parsed["cycles"]:
        require(isinstance(c, dict), "invalid cycle event")
        for key in ("seq", "epoch", "next_epoch", "cycle_ns"):
            require(integer(c.get(key), 1), "invalid cycle " + key)
        for key in ("verify_step", "wait_ns", "k", "proposal_invalid_depths", "next_invalid_depths"):
            require(integer(c.get(key)), "invalid cycle " + key)
        require(type(c.get("accepted")) is int, "invalid cycle acceptance")
        for key in ("steady", "live", "published"):
            require(type(c.get(key)) is bool, "invalid cycle " + key)
        key, nxt = (c["seq"], c["epoch"]), (c["seq"], c["next_epoch"])
        require(key in epochs and nxt in epochs and key not in seen and nxt not in next_seen,
                "cycle has no unique previous/next proposal")
        require(c["next_epoch"] > c["epoch"] and c["verify_step"] > previous_step, "nonmonotone cycle epoch/step")
        require(c["proposal_invalid_depths"] == masks[key] and c["next_invalid_depths"] == masks[nxt], "cycle signal masks differ")
        require(c["k"] == parsed["intended_k"] and c["wait_ns"] <= c["cycle_ns"], "invalid cycle prefix/wait")
        seen.add(key); next_seen.add(nxt); previous_step = c["verify_step"]
        if not c["steady"]:
            invalid_cycles += c["live"] and c["published"] and bool(masks[key] or masks[nxt])
            continue
        require(c["live"] and c["published"] and not masks[key] and not masks[nxt] and 0 <= c["accepted"] <= c["k"],
                "excluded/invalid/cancelled cycle labelled steady")
        previous, current = epochs[key], epochs[nxt]
        require(current[1]["proposal"]["anchor_pos"] == previous[1]["proposal"]["anchor_pos"] + 1 + c["accepted"],
                "steady actual anchor/acceptance advance differs")
        accepted = 0
        for d in (1, 2):
            o = previous[d]["outcome"]
            require(not o["cancelled"] and o["verified"] == (d <= c["k"]), "cycle verified prefix differs from proposal outcome")
            require(d <= c["k"] or o["status"] == "verify_cap", "unverified steady proposal was not capped")
            accepted += o["accepted"] is True
        require(accepted == c["accepted"], "cycle acceptance differs from paired proposal outcomes")
        row = sums[c["k"]]
        row["n"] += 1; row["cycle_ns"] += c["cycle_ns"]; row["wait_ns"] += c["wait_ns"]
        row["accepted"] += accepted; row["emitted"] += 1+accepted
    require(sums == parsed["costs"], "steady cycle sums differ from cost summaries")
    require(invalid_cycles == parsed["totals"]["invalid_cycles"], "cycle invalid exclusions differ")
    return dict(proposals=len(records), epochs=len(epochs), cycles=len(seen), steady_cycles=sum(r["n"] for r in sums.values()))


def checked_hash(value, label):
    require(isinstance(value, str) and HEX64.fullmatch(value), "invalid SHA256 " + label)
    return value


def extract(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    base, manifest = manifest_path.parent, prob.read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get("schema") == "ds41-matched-cost-manifest-v1", "wrong extraction manifest schema")
    require(manifest.get("same_drafter_observer_wait") is True and string(manifest.get("authority_assertion")), "explicit matching/authority assertion required")
    require(string(manifest.get("measurement_profile")), "measurement profile required")
    profile = manifest.get("profile")
    require(isinstance(profile, dict) and string(profile.get("id")) and string(profile.get("hardware")), "profile id/hardware required")
    require(re.fullmatch(r"[0-9a-f]{40}", profile.get("engine_commit", "")), "full engine commit required")
    checked_hash(profile.get("common_config_sha256"), "common settings excluding explicit family hotset")
    libs = profile.get("libraries")
    require(isinstance(libs, dict) and "libr4d.so" in libs and "deepseek41.so" in libs, "coherent kernel+architecture library hashes required")
    for name, digest in libs.items():
        require(string(name), "empty library name"); checked_hash(digest, name)
    family_profiles = manifest.get("family_profiles")
    require(isinstance(family_profiles, dict) and family_profiles, "explicit family configuration/hotsets required")
    for family, fp in family_profiles.items():
        require(string(family) and family != "all" and isinstance(fp, dict) and string(fp.get("id")), "invalid family profile")
        checked_hash(fp.get("config_sha256"), family+" full normalized config INCLUDING its hotset")
        hotset = fp.get("hotset")
        require(isinstance(hotset, dict), family+": explicit original hotset required")
        hotset_path = resolve(base, hotset.get("path"))
        require(hashed(hotset_path) == checked_hash(hotset.get("sha256"), family+" hotset"), family+": original hotset bytes differ")
    frozen = manifest.get("frozen_split")
    require(isinstance(frozen, dict) and string(frozen.get("seed")) and
            prob.finite(frozen.get("heldout_fraction")) and 0 < frozen["heldout_fraction"] < 1, "explicit frozen seed/fraction required")
    frozen_path = resolve(base, frozen.get("metadata"))
    require(hashed(frozen_path) == checked_hash(frozen.get("sha256"), "frozen metadata"), "frozen metadata hash differs")
    metadata = prob.load_metadata(frozen_path)
    runs = manifest.get("runs")
    require(isinstance(runs, list) and runs, "manifest has no runs")
    names, log_hashes, provenance, aggregate, inventories = set(), set(), [], defaultdict(dict), {}
    for run in runs:
        require(isinstance(run, dict) and string(run.get("run")) and run["run"] not in names, "unique run names required")
        name = run["run"]; names.add(name)
        require(run.get("scope") in ("cost_train", "audit_only"), name+": explicit scope required")
        quiet = run["scope"] == "cost_train"
        require(run.get("logging") == ("quiet" if quiet else "capture"), name+": cost/audit logging differs")
        require(run.get("log_level") in (("info",) if quiet else ("info", "debug")), name+": log-level assertion required")
        require(integer(run.get("k")) and run["k"] <= 2, name+": intended k required")
        require(string(run.get("family")) and run["family"] in family_profiles, name+": intended family profile required")
        fp = family_profiles[run["family"]]
        require(run.get("profile_id") == profile["id"] and run.get("engine_commit") == profile["engine_commit"] and
                run.get("common_config_sha256") == profile["common_config_sha256"] and
                run.get("family_profile_id") == fp["id"] and run.get("config_sha256") == fp["config_sha256"] and
                run.get("hotset") == fp["hotset"] and run.get("libraries") == libs,
                name+": declared engine/config/library profile differs")
        require(string(run.get("provenance_assertion")) and string(run.get("request_mapping_assertion")), name+": explicit source/mapping authority required")
        require(string(run.get("arm")), name+": explicit arm required")
        parts = run["arm"].split("-")
        require("Ds" in parts and "Db2" in parts and [p for p in parts if p.startswith("Dme")] == ["Dme"+str(run["k"])],
                name+": arm must declare bare Ds, Db2 and one intended Dme")
        require(not any(p in ("Dsa", "Dc", "Dl", "Dg") or re.fullmatch(r"Ds[0-9]+", p) or
                        re.fullmatch(r"(?:Dc|Dl|Dg)[0-9].*", p) for p in parts), name+": conflicting arm scope")
        require(("Dpc" in parts) == (not quiet), name+": Dpc/cost logging differs")
        requests = run.get("requests")
        require(isinstance(requests, list) and requests, name+": explicit request list required")
        inventory = []
        for i, row in enumerate(requests, 1):
            require(isinstance(row, dict) and row.get("seq") == i and type(row.get("seq")) is int, name+": fresh serial seq1..N mapping required")
            require(string(row.get("frozen_run")) and integer(row.get("frozen_seq"), 1), name+": invalid frozen request reference")
            key = (row.get("frozen_run"), row.get("frozen_seq"))
            require(key in metadata, name+": request absent from frozen metadata")
            meta = metadata[key]
            require(meta["family"] == run["family"], name+": mixed request families")
            split = prob.split_of(meta, frozen["seed"], frozen["heldout_fraction"])
            require(not quiet or split == "train", name+": held-out request cannot price training costs")
            inventory.append((meta["prompt_key"], meta["split_group"], split))
        require(len({r[0] for r in inventory}) == len(inventory), name+": duplicate prompts in run")
        results_path = resolve(base, run.get("results"))
        results = prob.read_json(results_path)
        require(isinstance(results, dict) and type(results.get("concurrency")) is int and results["concurrency"] == 1,
                name+": results must assert concurrency1")
        receipt = results.get("requests")
        require(isinstance(receipt, list) and len(receipt) == len(inventory), name+": result/request inventory differs")
        for row, expected in zip(receipt, inventory):
            require(isinstance(row, dict) and row.get("name") == expected[0] and row.get("family") == run["family"] and
                    not row.get("error") and isinstance(row.get("text"), str), name+": failed/reordered/foreign result request")
        stats = results.get("stats") or {}
        require(isinstance(stats, dict) and isinstance(stats.get("requests", []), list), name+": invalid stats requests")
        known = [r["id"] for r in stats.get("requests", []) if isinstance(r, dict) and integer(r.get("id"), 1)]
        require(not known or min(known) >= 1 and max(known) == len(inventory), name+": stats contradict serial mapping")
        log_path = resolve(base, run.get("log")); digest = hashed(log_path)
        require(digest not in log_hashes, name+": log reused by another run"); log_hashes.add(digest)
        lines = log_path.read_text().splitlines()
        parsed = parse_log(lines, run["k"], run["logging"])
        audit = audit_cycles(name, lines, parsed, set(range(1, len(inventory)+1))) if not quiet else None
        source = dict(run=name, family=run["family"], k=run["k"], scope=run["scope"], arm=run["arm"],
                      log=str(log_path), log_sha256=digest, results=str(results_path), results_sha256=hashed(results_path),
                      profile_id=run["profile_id"], logging=run["logging"], log_level=run["log_level"],
                      family_profile_id=fp["id"], config_sha256=fp["config_sha256"], hotset=fp["hotset"],
                      requests=len(inventory), split_groups=sorted({r[1] for r in inventory}),
                      request_inventory=[dict(row, prompt_key=expected[0], split_group=expected[1], split=expected[2])
                                         for row, expected in zip(requests, inventory)],
                      totals=parsed["totals"], costs=parsed["costs"], cycle_audit=audit,
                      provenance_assertion=run["provenance_assertion"], request_mapping_assertion=run["request_mapping_assertion"])
        provenance.append(source)
        if quiet:
            family = run["family"]
            require(inventories.setdefault(family, inventory) == inventory, family+": fixed costs use different prompt/group order")
            row = aggregate[family].setdefault(run["k"], dict(n=0, cycle_ns=0, runs=[]))
            row["n"] += parsed["costs"][run["k"]]["n"]
            row["cycle_ns"] += parsed["costs"][run["k"]]["cycle_ns"]
            row["runs"].append(name)
    require(aggregate, "no quiet training cost runs")
    families = {}
    for family, actions in aggregate.items():
        require(set(actions) == {0, 1, 2}, family+": missing quiet fixed0/1/2")
        values = [actions[k]["cycle_ns"] / actions[k]["n"] * 1e-6 for k in range(3)]
        require(values[0] <= values[1] <= values[2], family+": measured means are not nondecreasing; repeat/investigate, never sort or clamp")
        families[family] = dict(zip(("c0_ms", "c1_ms", "c2_ms"), values))
        families[family]["source"] = "; ".join("C%d=" % k + ",".join(actions[k]["runs"]) for k in range(3))
        families[family]["measurements"] = actions
        families[family]["profile"] = family_profiles[family]
    return dict(schema="ds41-proposal-costs-v1", measurement_profile=manifest["measurement_profile"],
                same_drafter_observer_wait=True, families=families,
                extraction=dict(schema="ds41-matched-cost-extraction-v1", manifest=str(manifest_path), manifest_sha256=hashed(manifest_path),
                                authority_boundary="Log checks establish instrumentation evidence, not operator-supplied deployment/config/request authority.",
                                authority_assertion=manifest["authority_assertion"], profile=profile, family_profiles=family_profiles,
                                frozen_split=dict(frozen, metadata=str(frozen_path)),
                                fit_performed=False, cost_scope="quiet_train_groups_only", runs=provenance))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        result = extract(args.manifest)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
        return 0
    except (Invalid, OSError, ValueError, TypeError, KeyError) as error:
        print("cost extraction refused: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

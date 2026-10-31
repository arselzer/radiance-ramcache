#!/usr/bin/env python3
"""Strict receipt for the fixed0/Dx1 query-free measurement; no model calls or fitting."""
import argparse
from pathlib import Path
import re
import sys

ENV = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_QUERY_FREE"
PREFIX = "DSpark query-free "
START = (f"DSpark query-free control: on ({ENV}); matched_fixed=0 declared_query_rows=2 "
         "actual_query_rows=0 observer=0 query_readback=0; context/reservations/Db2 bank retained, "
         "Dx=1, wait=step_end before issue_ahead; greedy request guard, no predictor or policy")
FIRST = re.compile(r"DSpark query-free first: seq=(\d+) step=(\d+) boundary=1 slot=(\d+) card=(\d+) "
                   r"query_rows=0 observer_rows=0 query_epochs=0 context_rows=(\d+) wait=step_end$")
TOTALS = re.compile(r"DSpark query-free totals: opportunities=(\d+) waits=(\d+) wait_ns=(\d+) context_rows=(\d+) "
                    r"query_issues=0 query_rows=0 observers=0 query_epochs=0 publications=0 "
                    r"transition=(\d+) no_opportunity=(\d+) end_cancel=(\d+) unclosed_boundary=(\d+)$")
COST = re.compile(r"DSpark query-free costs: label=C0NQ n=(\d+) cycle_ns=(\d+) wait_ns=(\d+) emitted=(\d+) "
                  r"mean_ms=([0-9]+\.[0-9]{6}); consecutive target-commit-bound pure-decode opportunities only; "
                  r"context retained, early wait retained, no query/observer/query readback$")


class Invalid(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Invalid(why)


def check_log(text, enabled=False, complete=False):
    rows = [line[line.index(PREFIX):] for line in text.splitlines() if PREFIX in line]
    if not enabled:
        require(not rows, "query-free markers in an ordinary arm")
        return
    require(rows.count(START) == 1, "missing/duplicate/wrong fixed0 Dx1 query-free startup")
    require(all(row == START or FIRST.fullmatch(row) or TOTALS.fullmatch(row) or COST.fullmatch(row)
                for row in rows), "unknown/malformed query-free receipt")
    require("DSpark context pass deferred (" in text, "actual Scheduler Dx startup absent")
    require(not any(marker in text for marker in (
        "DSpark matched early verify:", "DSpark matched early totals:", "DSpark matched early costs:",
        "DSpark prefix policy", "DSpark read-only proposal capture: on", "DSpark confidence head:",
        "DSpark acceptance gate: on", "DSpark load-aware drafting: on", "DSBANKLEND")),
        "query-free arm also reports query measurement/policy/capture/gating/lender")
    if not complete:
        return
    first = [FIRST.fullmatch(row) for row in rows if FIRST.fullmatch(row)]
    total = [TOTALS.fullmatch(row) for row in rows if TOTALS.fullmatch(row)]
    cost = [COST.fullmatch(row) for row in rows if COST.fullmatch(row)]
    require(len(first) == len(total) == len(cost) == 1, "need one actual first use, totals and C0NQ cost")
    seq, step, slot, card, first_context = map(int, first[0].groups())
    require(seq > 0 and card == 0 and first_context > 0, "first actual opportunity lacks owner/context")
    opportunities, waits, wait_all, contexts, transitions, no_opportunity, ends, unclosed = map(int, total[0].groups())
    n, cycle_ns, wait_ns, emitted = map(int, cost[0].groups()[:4])
    require(opportunities > 1 and waits == opportunities and contexts >= first_context,
            "no real opportunity/wait/context coverage")
    require(n > 0 and emitted == n and n + transitions + ends == opportunities,
            "unclosed or inconsistent target-bound opportunity accounting")
    require(cycle_ns > 0 and wait_ns <= wait_all and cycle_ns >= wait_ns,
            "invalid all-cost cycle/wait accounting")
    require(unclosed <= opportunities, "invalid last boundary")
    mean = float(cost[0].group(5))
    require(abs(mean - cycle_ns / n * 1e-6) <= 0.00000051, "reported C0NQ mean differs from exact sums")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enabled", choices=("0", "1"), required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--complete", action="store_true")
    args = ap.parse_args()
    try:
        require(args.log.stat().st_size <= 64 * 1024 * 1024, "server receipt exceeds 64MiB read bound")
        check_log(args.log.read_text(), args.enabled == "1", args.complete)
    except (OSError, UnicodeError, Invalid) as exc:
        print(f"QUERY_FREE_GATE INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"QUERY_FREE_GATE PASS enabled={args.enabled} complete={int(args.complete)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

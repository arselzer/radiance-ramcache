#!/usr/bin/env python3
"""Compare A/B arms' replies with a reference arm's, request by request; no model, device or network use.

    python3 tools/ds41_reply_check.py REFERENCE.json ARM.json [ARM.json ...]

Each file is an ab_requests.py result: `requests`, each with a `name` and the reply `text`. Every
request of an arm whose name the reference also has must carry the same text, byte for byte.
Requests the reference lacks are counted apart, and an arm with none in common (another prompt set:
code, chat, long documents) is skipped. Exits 0 only when at least one request was compared and none
differed.

Example (the README's identity claim):

    python3 tools/ds41_reply_check.py evidence/ds41/ab-20261028T223812Z/base.json \\
        evidence/ds41/ab-20261030T081259Z/*.json
"""

import json
from pathlib import Path
import sys


def replies(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    requests = data.get("requests") if isinstance(data, dict) else None
    if not isinstance(requests, list):
        return None
    out = {}
    for r in requests:
        if isinstance(r, dict) and isinstance(r.get("name"), str) and isinstance(r.get("text"), str):
            out[r["name"]] = r["text"]
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    reference = replies(argv[1])
    if not reference:
        print("REFUSED: no requests in the reference " + argv[1], file=sys.stderr)
        return 2
    failed = False
    total_same = total = 0
    for path in argv[2:]:
        arm = replies(path)
        if arm is None:
            print("skip  %s: not an arm result" % path)
            continue
        common = sorted(set(arm) & set(reference))
        if not common:
            print("skip  %s: no request in common with the reference (%d requests)" % (path, len(arm)))
            continue
        same = sum(1 for name in common if arm[name] == reference[name])
        differ = [name for name in common if arm[name] != reference[name]]
        print("%s  %s: %d of %d replies identical%s%s" % (
            "ok  " if not differ else "FAIL", path, same, len(common),
            "; differ: " + ", ".join(differ) if differ else "",
            "; %d not in the reference" % (len(arm) - len(common)) if len(arm) > len(common) else ""))
        failed |= bool(differ)
        total_same += same
        total += len(common)
    failed |= total == 0
    print("%s: %d of %d compared replies identical to %s" % (
        "FAIL" if failed else "PASS", total_same, total, argv[1]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Hash/source-body checks only. No target imports, builds, models or device calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    receipt = json.loads((HERE / "SOURCE_RECEIPT.json").read_bytes())
    for name, expected in receipt["files_sha256"].items():
        assert sha(HERE / name) == expected, ("package source changed", name)
    core = ROOT / receipt["core_directory"]
    core_pins = json.loads((HERE / "CORE_SOURCE_PINS.json").read_bytes())
    for name, expected in core_pins.items():
        assert sha(core / name) == expected, ("whole Core changed", name)
    helpers = json.loads((HERE / "HELPER_SOURCE_PINS.json").read_bytes())
    for name, expected in helpers.items():
        assert sha(ROOT / name) == expected, ("reused helper changed", name)
    # New additive methods must not rewrite the old Tiny quote implementations.
    for row in receipt["preserved_tiny_quote_bodies"]:
        source = (core / row["file"]).read_text()
        body = source[source.index(row["begin"]):source.index(row["end"])].strip()
        assert hashlib.sha256(body.encode()).hexdigest() == row["sha256"], row["file"]
    print("PASS source hashes", len(receipt["files_sha256"]), "package;", len(core_pins),
          "whole Core;", len(helpers), "helper; two original Tiny quote bodies")

if __name__ == "__main__":
    verify()

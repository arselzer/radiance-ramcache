#!/usr/bin/env python3
"""NEW runtime source comparison. Does not modify/rebaseline the sealed d89 checker.

Root-only host gate: compare unchanged original functions to literal53 and pin BOTH complete
files by hash. This is a bounded textual comparison, not a compiler or numeric/card proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA = "d7e6a8693f1a162089e872b628f0283cf9ee16ba6343d97cb14b3e1b3d5a4946"
NAMES = ("ds_bf16", "ds_bf", "ds_e4m3", "ds_to_e4m3", "ds_max", "ds_clamp",
         "ds_ceil_log2", "ds_pow2", "block_scale", "ds_e2m1_rt", "roundtrip",
         "quant_group_kernel", "quant_serial_kernel", "quant_rt_body", "rope_rt_kernel",
         "rope_rt_body", "fz_rope_args", "fz_quant_fmt", "r4d_rad_quant_roundtrip",
         "r4d_rad_rope_rows_roundtrip")

def definition(text, name):
    # Frozen definitions have no braces in their string/comment bodies. Signature is included;
    # template/attribute prefixes remain visible in the full candidate diff/codegen review.
    needle = " " + name + "("
    at = text.index(needle)
    start = text.rfind("\n", 0, at) + 1
    brace = text.index("{", at)
    end, depth = brace + 1, 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError("unterminated frozen definition: " + name)
    return text[start:end]

def compare(original, candidate):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA:
        raise ValueError("original is not the pinned literal53 source")
    old, new = original.decode("utf-8"), candidate.decode("utf-8")
    rows = {}
    for name in NAMES:
        a, b = definition(old, name), definition(new, name)
        if a != b:
            raise ValueError("changed original function: " + name)
        rows[name] = hashlib.sha256(a.encode()).hexdigest()
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    paths = [x.resolve(strict=True) for x in (args.original, args.candidate)]
    if paths[0] == paths[1] or any(x.stat().st_size > 2 * 1024 * 1024 for x in paths):
        raise ValueError("distinct bounded source files required")
    data = [x.read_bytes() for x in paths]
    rows = compare(*data)
    result = {"schema": "ds41-fp8-runtime-literal-source-v1", "functions": rows,
              "original": {"path": str(paths[0]), "sha256": hashlib.sha256(data[0]).hexdigest()},
              "candidate": {"path": str(paths[1]), "sha256": hashlib.sha256(data[1]).hexdigest()},
              "scope": "textual_source_only", "serving_proof": False, "gpu_executed": False,
              "integrated": False, "performance_eligible": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("PASS ds41_fp8_runtime_literal functions=20 exact_text=1 serving_proof=0")

if __name__ == "__main__":
    main()

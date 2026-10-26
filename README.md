# Radiance expert RAM cache

Experimental work on a GLM 5.3 Flash expert cache for models whose experts exceed system RAM. The cache keeps bounded expert records in RAM and reads cold records from disk, then verifies and uploads the selected records for GPU consumers.

Current source is exactly Core12 (1,536 files) with Fixture17. A fresh public Release build and all sixteen portable HOST checks passed. Matching private Release, ASan/UBSan and TSan suites also passed twenty regressions and five focused checks per mode. Actual four-rank declaration, metadata and complete-data prewalk passed their separate scopes.

The formatted-prompt GPU trial **failed with exit 88**, after four VRAM mover setup receipts. It produced no final result JSON. Those logs do not distinguish late startup from the first model step; normal text, quality, throughput and long-context runtime remain unvalidated. Diagnosis continues separately. This is a research checkpoint and is not ready for serving GLM chats.

The same original QFN container became healthy during restoration and passed an independent four-token completion in 0.063 seconds. Its original maintenance receipt retains a failed settling deadline: only 23.6 of the required 40 stable seconds remained before the 900-second limit. No extra restart, driver reset or hardware change was needed for the independent recovery check.

On Linux, use Python 3.10+, CMake 3.24+ and C++20 tools:

```sh
python3 tools/verify_snapshot.py
python3 tools/build_host.py --mode release --run-host-checks
```

The default builds without executing. `--run-host-checks` invokes sixteen CPU programs and needs no model download, GPU or API server. It does not invoke the archived model or maintenance modes. See [the portable guide](research/portable/README.txt).

- `radiance/`: the exact 1,536-file frozen Core12 source.
- `research/`: immutable source fixtures and the portable HOST entrypoint.
- `evidence/`: original receipts and logs, including failed runs.
- `docs/provenance/`: source copy maps and reconstructible earlier checkpoints.

The tested public manifest is `b9bb3cbe` (3,715 files). It and its overwritten documentation are archived at `docs/provenance/portable-tested-core12-b9bb3cbe/`; the final delta adds evidence and metadata only. The preceding published Core8 checkpoint and its raw two-output `[gMASK]` test remain preserved with their original scope. No useful-text result follows from that earlier test.

Source is based on Radiance commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766`; upstream licensing and notices remain. Model/native-store/tokenizer payloads, research binaries, private serving configuration and credentials are excluded. Cache-retention Core13 and Fixture19 diagnostics are separate work and are not included here.

See [STATUS.md](STATUS.md) and [the exact validation summary](evidence/glm53-main-verified-v12/VALIDATION_SUMMARY.json). No GLM serving deployment is part of this export.

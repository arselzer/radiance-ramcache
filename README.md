# Radiance expert RAM cache

Work in progress on an expert RAM-cache tier for Radiance, focused on GLM 5.3 Flash and models whose experts exceed system RAM.

The small synthetic cache pipeline works, including a correctness check on one RX 9070 XT. Full-model Engine integration is still unfinished. This checkpoint does not enable cached GLM serving.

The cache keeps a bounded set of expert records in RAM and reads cold records from disk. A RAM hit avoids the disk read, but still uploads the expert to its GPU consumer. Persistent GPU expert caching, prefill overlap and admission policies remain future work.

- `radiance/`: the current Core source, including the experimental native expert machinery.
- `research/`: synthetic fixtures, cache/provider implementations and portable HOST checks.
- `evidence/`: selected test receipts and logs, with each run's scope preserved.
- `docs/`: design notes, independent reviews and original source hashes.

The source is based on Radiance commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766`. Upstream licensing and notices are retained. The export omits two generated Python bytecode files; the provenance record identifies them. Model downloads, local serving configuration, API keys and compiled research libraries are excluded.

On Linux, use Python 3.10+, CMake 3.24+ and a C++20 compiler. Start with:

```sh
python3 tools/verify_snapshot.py
python3 tools/build_host.py --mode release --run-host-checks
```

The HOST checks build CPU fixtures. They do not start an API server or use a GPU. Sanitizer modes and further details are described by `python3 tools/build_host.py --help`.

The original package READMEs and receipts retain their historical paths and source pins. Their legacy verification scripts require the original workspace. Use the portable entrypoint above for this repository. Optional physical-source and composite-exporter packages are included as research source; their original CMake files target older pinned Core revisions.

See [STATUS.md](STATUS.md) for current validation and remaining work.

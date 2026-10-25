# Radiance expert RAM cache

Work in progress on an expert RAM-cache tier for Radiance, focused on GLM 5.3 Flash and models whose experts exceed system RAM.

The current source is Core V6, Engine fixture V8 and Components V12. Matching HOST regression suites passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer. The complete actual-model prewalk also passed. The four-GPU startup attempt failed before source admission at a vocabulary-size guard; no GLM token was generated. The same original QFN container was restored and independently checked. Cached GLM serving is not enabled.

The cache keeps a bounded set of expert records in RAM and reads cold records from disk. A RAM hit avoids the disk read and still uploads the expert to its GPU consumer. Persistent GPU expert caching, prefill overlap and admission policies remain future work.

- `radiance/`: exactly 1,535 pinned Core source files.
- `research/`: immutable source fixtures, cache/provider implementations and portable HOST checks.
- `evidence/`: selected original receipts, logs and results with their run scopes preserved.
- `docs/`: designs, independent reviews, historical source maps and export provenance.

The source is based on Radiance commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766`. Upstream licensing and notices are retained. Model weights, the complete native expert store, tokenizer payloads, serving configuration, API keys and compiled research libraries are excluded.

On Linux, use Python 3.10+, CMake 3.24+ and a C++20 compiler:

```sh
python3 tools/verify_snapshot.py
python3 tools/build_host.py --mode release --run-host-checks
```

The portable entrypoint builds twelve HOST checkers. Their current build and all twelve runtime checks passed using a source-identical existing Core. The runtime receipt binds the preserved 3,021-file `f0eaee18` snapshot; this final export adds evidence and normalized documentation without changing source, tools or CMake. These checks use no downloaded model or GPU and start no API server. `--run-host-checks` explicitly requests execution; without it the tools only build. See [the portable guide](research/portable/README.txt) for sanitizer modes, archive reuse and compiler guards.

The separate real-data prewalk verified all four rank authorities, omitted 193,536 native-owned expert views covering 161,715,585,024 bytes, and prepared 5,088 ordinary load jobs. It took 1,046.67 seconds and executed no loader, Engine or model kernel. [The exact result](evidence/glm53-main-verified-v6/complete-prewalk/result.json) is a source, layout and budget check rather than an inference result.

The private manual Engine path targets one prompt token and one output token, maxTok1/seq1/spec0, exact wire, context512 and no fixed resident experts. The current Core V6 cannot complete startup: its exact-equality guard rejects the tokenizer’s 154,856 entries against the model’s 154,880 padded rows. Normal prompts, 262K context, quality, throughput and deployment are not validated. [STATUS.md](STATUS.md) separates each passed stage from the failed startup and restoration result.

The current archived maintenance source is Runner V4 for the reviewed Core V6/fixture V8 pair. It has no default build, import or execution action and requires its original private gates and reviewed environment. No maintenance, download or GPU action is invoked by the portable tools.

Historical Core V5, component and runner revisions retain their exact sources and failed/pass receipts. Two archived V5 loader files reconstruct the earlier Core inventory over the unchanged 1,533 files. The plain-source scale fix in V6 and the V7/V8 fixture repairs are described in [the current publication scope](docs/GLM53_VERIFIED_MAIN_V6_PUBLICATION_SCOPE.txt); archived package READMEs retain their original paths and historical statements.

[The completed maintenance window](evidence/glm53-main-verified-v6/GPU_WINDOW_SUMMARY.json) reached four gfx1201 devices and tokenizer loading, then returned `RAD_E_FORMAT` in `ColdAdmitted`, with no source admission, expert reads/uploads or model computation. Restoration passed more than 40 seconds of stable health; an independent four-token completion checked the original QFN server. An earlier attempt aborted on a health timeout before stopping anything. No additional GPU window is queued. A possible padded-vocabulary fix is separate future work, not included in this Core V6 snapshot.

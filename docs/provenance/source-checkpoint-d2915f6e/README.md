# Radiance expert RAM cache

Work in progress on an expert RAM-cache tier for Radiance, focused on GLM 5.3 Flash and models whose experts exceed system RAM.

The retained evidence covers the complete GLM expert store and small synthetic GPU fixtures, including a correctness check on one RX 9070 XT. The selected Core V6 contains a manual real-source Engine path and a narrow correction for plain tensors’ null scale placeholders. Matching V6 builds, HOST/sanitizer checks, full prewalk and actual GLM startup/token results remain pending. Earlier Core V5 and portable passes retain their original scope. Cached GLM serving is not enabled.

The cache keeps a bounded set of expert records in RAM and reads cold records from disk. A RAM hit avoids the disk read, but still uploads the expert to its GPU consumer. Persistent GPU expert caching, prefill overlap and admission policies remain future work.

- `radiance/`: the current Core source, including the experimental native expert machinery.
- `research/`: synthetic fixtures, cache/provider implementations and portable HOST checks.
- `evidence/`: selected test receipts and logs, with each run's scope preserved.
- `docs/`: design notes, independent reviews and original source hashes.

The source is based on Radiance commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766`. Upstream licensing and notices are retained. The selected V6 Core contains exactly 1,535 pinned files; its frozen inventory already omits the two historical generated Python bytecode files. Original pin manifests are retained as provenance. Model downloads, local serving configuration, API keys and compiled research libraries are excluded.

On Linux, use Python 3.10+, CMake 3.24+ and a C++20 compiler. Start with:

```sh
python3 tools/verify_snapshot.py
python3 tools/build_host.py --mode release --run-host-checks
```

The twelve HOST checks build CPU fixtures. The ten retained checks cover the tiny Engine and four-rank protocol/upload machinery; two new checks cover the real-source Engine's cold refusals and pure component quotes. None starts an API server, runs a model step or uses a GPU. The preceding V5 export passed all twelve. Current Core V6 with Engine fixture V8 adds same-production source validation cases to the Engine gate and retains Components V12’s unchanged HOST quote body; matching snapshot verification and portable rerun remain pending. Sanitizer modes and further details are described by `python3 tools/build_host.py --help`.

The original package READMEs and receipts retain their historical paths and source pins. Their legacy verification scripts require the original workspace. Use the portable entrypoint above for this repository. Optional physical-source and composite-exporter packages are included as research source; their original CMake files target older pinned Core revisions.

See [STATUS.md](STATUS.md) for current validation and remaining work.

The archived maintenance runner is server-specific research source. It has no default build or execution action and requires its original private runtime gates and reviewed operating environment. No maintenance, model download or GPU action is invoked by the portable tools. See [the current publication scope](docs/GLM53_VERIFIED_MAIN_V6_PUBLICATION_SCOPE.txt).

Component V10 is retained as a historical metadata/prewalk fixture. Historical V7 remains exact: its complete prewalk failed before Engine and exposed a declaration-metadata lifetime defect. V9 retained that metadata correctly but rejected the actual flattened KV count with an incorrect 4096-entry ceiling. V10 checks exact owned counts without that invented ceiling. V10 later reached ordinary loading as described below; current Components V12 and four-GPU model execution remain separate pending gates.

The V10 complete prewalk passed all four source-rank bindings, NativeInventory bindings and final Native plans, then refused ordinary loading. A fast V11 diagnostic confirmed that valid plain tensors carried null scale placeholders: 1,067 per rank, with no other reported materializer, layout, destination or share refusal. V6 requires those nulls for canonical plain encoding and physical nonnull scales for affine encoding; it does not skip missing affine data. Source-reviewed implementation is separate from the pending matching runtime gates.

Engine fixture V7 uses valid boolean output sentinels for the new source tests; Core V6 is unchanged from fixture V6’s source. Both fixture versions remain archived. The current archived runner V3 requires the explicit reviewed Core V6/fixture V7 build pair and preserves V2’s maintenance behavior; it has no default action.

Engine fixture V8 additionally synchronizes the nested strictP3 Core pin map; its C++ cases are identical to V7. The initial Core V6 HOST configuration failure is retained. Runner V4 selects the reviewed Core V6/fixture V8 pair, with its maintenance bodies unchanged. These source corrections do not establish the pending matching runtime/prewalk or GPU results.

# ds41 measurement programs

These are small programs, and some of their outputs, that the reports in `docs/ds41/` cite. They are copied byte for byte from the working folder. They were run by hand on the box outside GPU windows, and none is part of the engine build. Their inputs (access traces, the model container, a scratch file on the drive) are not included.

| File | What it measured | Cited in |
|---|---|---|
| `ds41_supply_model.py`, `ds41_supply_model*.out.txt` | A model, not a measurement: per-token expert-supply time under candidate changes, from one run's placement lines and routing histogram | `REVIEW.md` |
| `packbench.cpp`, `packbench.out.txt` | SSE2 MXFP4 pack against `memcpy`, and one O_DIRECT read | `REVIEW.md` |
| `membw.cpp`, `membw.out.txt` | Aggregate DRAM read bandwidth at 1-24 threads | `REVIEW.md`, `NIGHT-LOG.md` |
| `membw_pf.cpp` | The same at 40 and 48 threads, with software prefetch and interleaved streams; its output is in `NIGHT-LOG.md` (07:58) | `NIGHT-LOG.md`, `NEXT-STEPS.md` |
| `mrbench.cpp` | The cost of an extra token row in the CPU pool's tiles. It includes `radiance/tools/ds41-host-experts/ds41_host_experts.cpp`. Its output is left out because it names the machine's hostname; the numbers are in `NIGHT-LOG.md` (07:15) | `NIGHT-LOG.md`, `NEXT-STEPS.md` |
| `mtp_windows.py`, `mtp_windows.out.txt` | MTP re-priced offline: the experts a k-token verify pass needs, against k single steps, over an access trace | `NIGHT-LOG.md` (07:08), `NEXT-STEPS.md` |
| `cache_sim2.py` | RAM-cache strategies replayed on the per-rank access traces: a shared LRU, prefill at the LRU tail, TinyLFU, +7% capacity | `NIGHT-LOG.md` (08:25) |

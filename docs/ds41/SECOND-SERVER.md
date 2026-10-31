# A hypothetical second ds41 server

Source/document audit,2026-10-31. No implementation, build, test, replay, benchmark, model/GPU/service
action or large data scan. This is a design comparison, not a measured two-server result.

Assume a second independent256GB/equivalent-CPU machine, four RX9070 **non-XT** cards, full-rate100GbE
between hosts, local model storage, and comparable PCIe/root-port topology. AMD specifies56 CUs,
16GB and640GB/s for the9070; the current9070XT has64 CUs and the same advertised memory capacity/
bandwidth. Same memory bandwidth does not prove equal kernel speed or byte identity.
([AMD9070](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html),
[AMD series specifications](https://newsroom.amd.com/news/amd-unveils-next-generation-amd-rdna-4-architectu/))

**Whole-request, sticky TP4 replicas are the practical first architecture.** Each host keeps its own
complete TP4 graph, weights/cache and request state. Route a conversation/session back to its owner
to retain prefix/KV reuse; transmit requests and generated text, not experts or per-layer collectives.
With enough independent traffic, aggregate decode capacity is R_primary+R_secondary. The present
host measures roughly37–42tok/s total atN4/N8; **70–85tok/s combined is conditional arithmetic if
the second machine sustains a comparable rate**, not a9070 measurement or single-stream gain.
Short-request/prefill/end-to-end rates are lower than decode counters. Exact replies require the
same checkpoint, format/template/options, deterministic numerical paths and per-request sampling
state, plus actual9070 byte gates; XT results alone do not prove that.

**Remote expert execution is plausible research, with a small network payload and a difficult
latency/lifetime contract.** Retain the primary's logical TP4 expert ownership and local gather/
collectives. Send complete expert jobs to complementary resident experts on the second host and
return each expert's unchanged BF16 down row into its original sorted slot. Do not turn this into
TP8 or pre-sum remote subsets.

The actual native shapes are hidden5120, FF2304, gate/up[4608,5120], down[5120,2304], top6 across
40 layers/384 experts. One expert is18,800,640B in native block_mxfp4:12,533,760B GU plus6,266,880B
DN. All15,360 pairs are268.945GiB. Exact packed-scale storage normally costs18,001,920B/pair
(exceptions may retain native form).

| Payload | Raw bytes, excluding identifiers/framing |
|---|---:|
| Input activation per token/layer |10,240B BF16; once per remote host if cross-rank equality is proved, otherwise up to four copies40,960B |
| Output per remotely executed expert |10,240B BF16 |
| Worst case all240 expert calls remote, per generated token |2,457,600B output +409,600–1,638,400B input |
| Same synthetic worst case,32768-token prefill |80.53GB output +13.42–53.69GB input |
| Sending weights instead of computing remotely |18,800,640B per expert;4.512GB/token if all240 calls fetched |

100Gb/s is at most12.5GB/s line rate. Decode's all-remote serialization lower bound is about
0.23–0.33ms/token before protocol, PCIe/host staging, queueing and compute; at42tok/s its response
traffic is only103MB/s. Weight streaming instead takes at least1.504ms/expert and saturates the link
at664.9 expert pairs/s, or2.77tok/s in the artificial all240-fetched case. A remote weight cache could
serve rare cold misses, but cannot feed the entire decode weight stream at100GbE.

Bandwidth therefore is not the obvious decode obstacle for *remote computation*. There are still
up to40 causally dependent layer rendezvous/token. An exposed100µs per-layer round trip consumes
4ms/token before remote computation; it may replace existing CPU waits, so it is not automatically
an added4ms. Host staging is assumed; GPU-direct/RDMA support has not been established. Prefill can
expose materially larger activation traffic and needs its own measurement. A **40–50tok/s** example
from a30tok/s baseline requires **25–40% net critical-step time removed after every remote overhead**:
30/(1−0.25)=40;30/(1−0.40)=50. This is a condition, not a forecast.

**The new VRAM can hold more expert identities than another full replica's expert allotment.**
An expert-only worker need not duplicate dense trunk weights or KV. With R GiB/card reserved for
driver/code/queues/scratch, native capacity is4×floor((16−R)GiB/18,800,640):3652 pairs at the impossible
zero-overhead ceiling;3424 at an illustrative1GiB/card reserve;3196 at2GiB. Packed counterparts are
3816/3576/3340 if the actual weights pass existing exact packing rules. These reserve choices are
examples, not allocations. Keep explicit host/GPU budgets and price staging buffers. Complementary
remote RAM can also distribute CPU bandwidth/cold storage; it does not make primary RAM faster.

**Concentration does not imply more than twice the hit rate.** Expert count coverage, GPU selection
coverage and RAM backing-cache hit rate have different denominators. For equally sized experts
ranked optimally on one fixed request distribution, H(2C)≤2H(C): the nextC experts cannot carry more
probability mass than the hottestC. Concentration normally causes diminishing returns. Two identical
sticky replicas do not double per-request hits at all. Specialized routing can change conditional
distributions; an expert-only worker can add more than C slots because it avoids the replica's trunk
overhead. Those are different mechanisms and require held-out routing counts.

A current-plan example is already far above17% **selection** coverage: chain34'sN4/P1/Vp startup
reports665/674/674/674 GPU-resident expert pairs,2687/15360=17.49% of identities, but summed VRAM
calibration mass26.039759/40=**65.10%** on the sealed721-token placement profile. Even perfect100%
coverage would be only1.536× that hit fraction. The log explicitly warns calibration coverage is
not held-out performance. Historical40.9% chat/54.6% held-out GPU selection shares are from an older
128GB/profile setting and are not current measurements. RAM backing hits are already high: one
retained-slab counter in the same currentN4 log is30,552 hits/343 misses (98.9% of that counter's
lookups, mixed prefill/decode), not98.9% GPU execution. More RAM may eliminate residual SSD reads;
GPU-resident execution instead avoids CPU weight streaming on RAM hits.

**Exact source contracts and remaining gates.** In current source, deepseek41.cpp:1274–1288 declares
the routed FFN BF16 throughout, including expert_down[T×6,5120]. rad_block_moe_fp8.h:1119–1129
writes the full expert down row. stager.h:889–890 and stager.cpp:4298–4347 already inject exact CPU
uint16 rows before the unchanged gather. r4d_moe.hip:2103–2137 sums in ascending original slot order;
the exact collective folds ranks in its existing ascending order (r4d_ar_entry.hip:662ff). Four FP32
rank vectors would be80KiB/layer at a later seam; they are not required by this BF16 job interface.
Combining a remote-only subset can change rounding when local and remote slots interleave.

The worker must preserve original expert shape, BF16 intermediate roundings, SiLU table, reduction
partition and logical rank/slot placement. Bind every response to model/library, sequence/epoch,
position, layer, expert and original slot; drain or reject stale/cancelled work. Failures must wait
for an exact fallback before gather or fail the request. No partial/approximate contribution.

History covered: full HANDOFF/current NIGHT tail, the prior all31-report inventory, fresh text
search across the complete archive, current README and relevant source; only bounded headers/tails
of one chain34 server log were read. No remote execution/network tier has been implemented or
measured: NEXT-STEPS:375 explicitly leaves network expert tiers future work. MORE-OPTIMIZATIONS:1024
dropped twoTP2 replicas **on one shared box**; it is not evidence against two independentTP4 hosts.
RESEARCH-3-LOSSY:60 rejected hot-expert duplication across existing ranks sharing one CPU/drive.
L3 staging, ordinary extra overlap and streaming RAM weights onto cards were already tried/limited;
their shared-memory bottleneck is not a measured result for a second independent memory system.

Evidence: [chain34 currentN4/P1/Vp log](/home/as/radiance-ds41-claude/tools/ds41-host-experts/build/ab-20261031T144623Z/all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs-L1024-Pf-N4-C4-Pmr4.server.log:75)
(also startup84/93/102); profile normalization is core/place/xstats.cpp:520–536, ranking is
core/place/planner.cpp:424–435/483–487. Actual held-out GPU execution requires the existing
route_trace.h per-pick statuses (lines1–21/38–56), not startup calibration mass. No such new trace
or complementary-placement replay was run for this hypothetical.

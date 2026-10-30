# vpack: implementation report

Branch `claude/ds41-vpack` (worktree `/home/as/ds41-claude-build/wt-vpack`, 3 commits on dc83cecc, not pushed) holds the card's static expert residents in the packed-scale form, bit-exact and opt-in. It is built and every required test passes on the CPU. Nothing has run on a card: neither the device check nor the A/B.

**Commits**
- `172394ed` libr4d: a new row, `moe_gemm_ggml_mxfp4_gemv_vp`, a SIMT-emulator test, and the CMake registration.
- `481162bb` core: `core/place/vram_packed.{h,cpp}`, planner, mover, loader, table marking, `place_test`.
- `3792abf6` tools: `ds41_vpack_device_check.hip`, the `ab_window.sh` arm part `Vp`, README section.

**Switch and arm part**
- `RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1`; optional `..._VRAM_PACKED_RAW=<gate_up>,<down>` (default 512,320, the RAM cache's budget).
- Arm part `Vp`. It needs this branch's `libr4d.so`, `--deterministic` and passes of at most 16 tokens; the best arm has the last two.

**How it works**
- **Selection:** the engine sets the machine fact `device.ds41_vram_packed` before kernel selection. That lets the new row (priority 59) take the routed GEMMs' decode band. It declares the old block_mxfp4 stored form, so the pinned pool, file tier, RAM cache and CPU path are unchanged; only the VRAM copies change.
- **Planning:** before charging a resident unit, the planner reads that unit's e8m0 scales from the container (about 1.1 MB a unit). It charges 18,001,920 bytes instead of 18,800,640 (×1.0444), so about 26 more residents a rank (598/615 → ~624/642). A weight over its budget stays block_mxfp4 in VRAM and is charged at full size. Only a static plan packs.
- **Loading:** each packed weight is checked before upload: scales round-trip byte for byte and the codes plane is unchanged. Any mismatch refuses the load.
- **Kernel:** a packed weight's table entry carries bit 0. The row decodes a marked entry from the packed form and any other entry as block_mxfp4, with the same arithmetic in the same order. A marked entry with a bad header gives NaN rows rather than wrong numbers.

**Tests (host backend, cards hidden)**
| Test | Result |
|---|---|
| `r4d_emu_ds41vp` (new) | 20/20 cases, every output value bit-identical to the plain row, including 3 experts with the release's real scales at the served shapes |
| `place_test` | 118/118 (6 new cases) |
| `host_expert_pool_test` | 6/6 |
| `mapped_slab_pregrow_test` | 3/3 |

**A/B setup**
1. The main repo's `ab_window.sh` (the copy `launch_window.sh` uses) has no `Vp` part. Bring in the `ab_window.sh` hunk of `3792abf6` first; otherwise `Vp` arms log "unknown part" and run without the switch.
2. Make a run home that uses this `libr4d.so` for both arms:
   ```
   cp -a /home/as/ds41-claude-build/home2 /home/as/ds41-claude-build/home-vpack
   cp /home/as/ds41-claude-build/aql-vpack/radiance_home/kernels/libr4d.so /home/as/ds41-claude-build/home-vpack/kernels/
   ```
3. Run the window:
   ```
   B=all-t40-f-M16-H3-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2
   launch_window.sh vpack "$B $B-Vp $B-r2 $B-Vp-r2 $B-r3 $B-Vp-r3" \
     RADIANCE_BIN=/home/as/ds41-claude-build/aql-vpack/bin/radiance \
     RADIANCE_HOME_DIR3=/home/as/ds41-claude-build/home-vpack \
     MIRROR=/srv/bulk/source-copies/ds41-native-mxfp4/d8b2e3c20f5f4b6d5044d37f20b8d53f260f83e466f72259572be76d7e756bf3-main \
     VPACK_CHECK=/home/as/ds41-claude-build/aql-vpack/vpack-tool/ds41_vpack_device_check \
     VPACK_LIBR4D=/home/as/ds41-claude-build/home-vpack/kernels/libr4d.so
   ```
   With `VPACK_CHECK` set, the device check runs first on card 0 (`check 48`: byte compare, must print PASS; then `time`). By hand on a free card: `ds41_vpack_device_check LIBR4D.so check 48`, then `... time`. `RADIANCE_BIN` is used in place (not snapshotted), so don't rebuild `aql-vpack` during the window.

**What to check in the server logs (`Vp` arms)**
- Startup: "VRAM packed experts: rank 0: 7680 routed weights can be held packed", with no refusal reason after it.
- Startup: "rank R holds N static residents in the packed-scale form …", with the scan time.
- Startup: "expert profile: rank 0 planned units VRAM=~624" (others ~642, against 598/615 in the base arm).
- Shutdown (mover report): "VRAM packed experts: … weights packed and checked at load …".
- Fewer "experts handed to the CPU" and "routed experts were not cached", less gather wait.
- 8/8 replies identical to the base arm.
- The base arm's log should have no "VRAM packed" lines.

**Artifacts** (sha256 prefixes)
- `/home/as/ds41-claude-build/aql-vpack/bin/radiance` (`ce7ba934…`)
- `/home/as/ds41-claude-build/aql-vpack/radiance_home/kernels/libr4d.so` (`aa9f961a…`)
- `/home/as/ds41-claude-build/aql-vpack/vpack-tool/ds41_vpack_device_check` (`9e6a5362…`)
- `deepseek41.so` is unchanged; use home2's.

**Open points**
- **Card behaviour unproven:** bit-exactness on the card and the new kernel's speed are unproven until the device check runs. The emulator runs the same kernel source, but not the real compiler or hardware.
- **Expected gain is my estimate:** about +1–2% decode, scaled from the plan's FP8 VRAM-half figures. Unmeasured.
- **Startup cost:** the scale scan reads about 0.7 GB a rank; the startup log prints the actual time.
- **Not packed in some configurations:** dynamic plans and routed ops with bands over 16 tokens pack nothing (the log says why). `RADIANCE_OP_ORACLE` is refused while the switch is on.
- **GPU-rule breach:** my first `place_test` run (14:23) did not hide the cards. The new mover case opened device 0 for about 0.1 s (created streams, allocated 1.3 MiB, freed) while the production container was serving. Every later run hid the cards.

# vpack: review

(pending)

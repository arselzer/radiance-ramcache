# Host-only build: no ROCm needed. -ffp-contract=off and no -ffast-math or -mfma keep every
# product and sum a separate rounded operation, as libr4d's kernels are compiled.
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -mavx2 -ffp-contract=off -fno-fast-math -Wall -Wextra -pthread
# Where `make scan` puts its binary (`make scan O=<build dir>` keeps it out of the tree).
O ?= .
TREE := ../..

ds41_host_experts_test: ds41_host_experts_test.cpp ds41_host_experts.cpp ds41_host_experts.h
	$(CXX) $(CXXFLAGS) -o $@ ds41_host_experts_test.cpp ds41_host_experts.cpp

# The exactness scan of the dense weights gemm_nt_bf16f8 would store
# (RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8), through libr4d's own host-only layout hooks, compiled in
# as the tests compile them.
SCAN_SRC := ds41_dense_fp8_scan.cpp $(TREE)/libr4d/r4d_layout.cpp $(TREE)/libr4d/r4d_fp8_layout.cpp
$(O)/ds41_dense_fp8_scan: $(SCAN_SRC) $(TREE)/libr4d/r4d_bf16f8_layout.h
	$(CXX) $(CXXFLAGS) -Wno-unused-parameter -Wno-missing-field-initializers -I$(TREE)/abi \
	  -I$(TREE)/libr4d -I$(TREE)/vendor -o $@ $(SCAN_SRC)
scan: $(O)/ds41_dense_fp8_scan

# The read-ahead predictor's quantized scoring against its bf16 scoring, offline, on a container's
# routers (see its header): the predictor's own arithmetic (core/place/router_score.h).
$(O)/router_score_eval: router_score_eval.cpp $(TREE)/core/place/router_score.h
	$(CXX) -O2 -std=c++17 -mavx2 -mfma -Wall -Wextra -pthread -I$(TREE)/core/place -o $@ router_score_eval.cpp
router-eval: $(O)/router_score_eval

# The card check of gemm_nt_bf16f8 against gemm_nt (run it in a GPU window; see its header). The
# fragment probe includes the kernel's own expansion (libr4d/r4d_bf16f8_frag.h), compiled as
# libr4d's plugin flags compile it: no FMA contraction, no fast math.
HIPCC ?= hipcc
HIPFLAGS ?= -O2 -std=c++17 --offload-arch=gfx1201 -ffp-contract=off -fno-fast-math
$(O)/ds41_bf16f8_device_check: ds41_bf16f8_device_check.hip $(TREE)/libr4d/r4d_bf16f8_layout.h \
                               $(TREE)/libr4d/r4d_bf16f8_frag.h
	$(HIPCC) $(HIPFLAGS) -I$(TREE)/abi -I$(TREE)/libr4d -o $@ ds41_bf16f8_device_check.hip -ldl
device-check: $(O)/ds41_bf16f8_device_check

# The card check of the host experts against libr4d's decode kernels (check, swiglu, time, ptime)
# and of RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL (`tall`: the routed GEMM's twin row, the sparse
# attention and the indexer score at passes taller than 16 tokens against the same rows in smaller
# launches; `sweep` / `sweep0`: the other height-dependent ops, the norms among them, at heights
# 1..48 with and without the switch). Run it in a GPU window:
# HIP_VISIBLE_DEVICES=0 ds41_mxfp4_device_check <libr4d.so> tall (sweep, sweep0). `p1 [MMAX] [ROUTES]` holds
# prefill Phase 1's twin (ds41_p1.h) to the gemv in passes of 16 and times it.
# Built as build_hx.sh builds it: the host kernels by $(CXX), the harness by hipcc, then linked.
MXFLAGS ?= -O2 -std=c++17 --offload-arch=gfx1201
$(O)/ds41_host_experts_dc.o: ds41_host_experts.cpp ds41_host_experts.h
	$(CXX) -O2 -std=c++17 -mavx2 -ffp-contract=off -fno-fast-math -fPIC -c ds41_host_experts.cpp -o $@
$(O)/ds41_mxfp4_device_check.o: ds41_mxfp4_device_check.hip ds41_host_experts.h ds41_p1.h $(TREE)/libr4d/r4d_bf16f8_layout.h \
                                 $(TREE)/libr4d/r4d_ds41_tall.h
	$(HIPCC) $(MXFLAGS) -I. -I$(TREE)/abi -I$(TREE)/libr4d -c ds41_mxfp4_device_check.hip -o $@
$(O)/ds41_mxfp4_device_check: $(O)/ds41_mxfp4_device_check.o $(O)/ds41_host_experts_dc.o
	$(HIPCC) $(O)/ds41_mxfp4_device_check.o $(O)/ds41_host_experts_dc.o -o $@ -ldl
mxfp4-check: $(O)/ds41_mxfp4_device_check

# Same n128 oracle cases as the SIMT emulator, over a built libr4d. Run only in a GPU window.
$(O)/ds41_n128_device_check: $(TREE)/tests/r4d_emu/test_ds41n128.cpp $(TREE)/libr4d/r4d_ds41_elem.h
	$(HIPCC) $(HIPFLAGS) -DDS41_N128_DEVICE -I$(TREE)/abi -I$(TREE)/libr4d -o $@ $< -ldl
n128-check: $(O)/ds41_n128_device_check

clean:
	rm -f ds41_host_experts_test ds41_dense_fp8_scan ds41_bf16f8_device_check

.PHONY: clean scan device-check mxfp4-check n128-check router-eval

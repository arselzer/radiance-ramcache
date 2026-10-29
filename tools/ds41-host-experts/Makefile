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

# The card check of gemm_nt_bf16f8 against gemm_nt (run it in a GPU window; see its header). The
# fragment probe includes the kernel's own expansion (libr4d/r4d_bf16f8_frag.h), compiled as
# libr4d's plugin flags compile it: no FMA contraction, no fast math.
HIPCC ?= hipcc
HIPFLAGS ?= -O2 -std=c++17 --offload-arch=gfx1201 -ffp-contract=off -fno-fast-math
$(O)/ds41_bf16f8_device_check: ds41_bf16f8_device_check.hip $(TREE)/libr4d/r4d_bf16f8_layout.h \
                               $(TREE)/libr4d/r4d_bf16f8_frag.h
	$(HIPCC) $(HIPFLAGS) -I$(TREE)/abi -I$(TREE)/libr4d -o $@ ds41_bf16f8_device_check.hip -ldl
device-check: $(O)/ds41_bf16f8_device_check

clean:
	rm -f ds41_host_experts_test ds41_dense_fp8_scan ds41_bf16f8_device_check

.PHONY: clean scan device-check

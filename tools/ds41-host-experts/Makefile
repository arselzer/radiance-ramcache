# Host-only build: no ROCm needed. -ffp-contract=off and no -ffast-math or -mfma keep every
# product and sum a separate rounded operation, as libr4d's kernels are compiled.
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -mavx2 -ffp-contract=off -fno-fast-math -Wall -Wextra -pthread

ds41_host_experts_test: ds41_host_experts_test.cpp ds41_host_experts.cpp ds41_host_experts.h
	$(CXX) $(CXXFLAGS) -o $@ ds41_host_experts_test.cpp ds41_host_experts.cpp

clean:
	rm -f ds41_host_experts_test

.PHONY: clean

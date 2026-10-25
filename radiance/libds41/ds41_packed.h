/* Lightweight ABI-facing contract, also included by device translation units. */
#pragma once
#include "rad_abi.h"
#include "ds41_packed_decode.h"
namespace rad::ds41::packed {
struct Geometry {
    Type type{};int64_t m=0,n=0,k=0,r=0,experts=0,row_bytes=0;
    uint32_t a_dtype=0,y_dtype=0;
};
struct Launch {
    Geometry g;
    int64_t input_rows=0,jobs=0,a_pitch=0,y_pitch=0;
};
int prepare(const RadArgs*,bool routed,Launch*);
}
extern "C" {
int ds41_packed_gemm(const RadArgs*,RadStream);
int ds41_packed_routed(const RadArgs*,RadStream);
int ds41_ref_packed_gemm(const RadArgs*,RadStream);
int ds41_ref_packed_routed(const RadArgs*,RadStream);
}

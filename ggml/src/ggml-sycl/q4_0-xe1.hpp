#pragma once

#include "common.hpp"
#include "fattn-xe1dec.hpp"

// [a770-opt] single-column mat-vec for reordered Q4_0 on Arc Alchemist (see q4_0-xe1.cpp)
bool ggml_sycl_q4_0_xe1_enabled(int device);
void ggml_sycl_q4_0_xe1_mul_mat_vec(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                    dpct::queue_ptr stream);
// same-input matrices (<= 3) in one launch; vy is the q8_1 SoA quantization of the shared activation
bool ggml_sycl_q4_0_xe1_mul_mat_vec_multi(const void * const * vx, float * const * dst, const int * nrows, int nm,
                                          const void * vy, int ncols, dpct::queue_ptr stream);
// fused gate / up mat-vecs + SWIGLU: dst[r] = silu(gate . y) * (up . y)
void ggml_sycl_q4_0_xe1_swiglu(const void * wg, const void * wu, const void * vy, float * dst, int ncols, int nrows,
                               dpct::queue_ptr stream);

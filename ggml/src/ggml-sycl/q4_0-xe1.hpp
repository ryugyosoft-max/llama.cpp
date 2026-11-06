#pragma once

#include "common.hpp"
#include "fattn-xe1dec.hpp"

// [a770-opt] single-column mat-vec for reordered Q4_0 on Arc Alchemist (see q4_0-xe1.cpp)
bool ggml_sycl_q4_0_xe1_enabled(int device);
void ggml_sycl_q4_0_xe1_mul_mat_vec(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                    dpct::queue_ptr stream);

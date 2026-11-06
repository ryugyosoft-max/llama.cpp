#ifndef GGML_SYCL_FWHT_HPP
#define GGML_SYCL_FWHT_HPP

#include "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

bool ggml_sycl_fwht_supported(const ggml_tensor * src, const ggml_tensor * dst);

// Same, with the Hadamard sign flip (dst = H * (src * signs)) done during the load.
// signs: F32, ne[0] a multiple of the block width, repeating per block row.
bool ggml_sycl_op_fwht_signed(ggml_backend_sycl_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                              ggml_tensor * dst);

#endif  // GGML_SYCL_FWHT_HPP

// [a770-opt] one row of K = 4096: [residual add +] RMS norm * w + sign flip + 4096-point rotation in one kernel;
// add_out / normed may be null (not needed after). False = not taken.
bool ggml_sycl_fwht4096_add_norm_rot(ggml_backend_sycl_context & ctx, const float * a, const float * b, float * add_out,
                                     const float * w, float eps, float * normed, const float * signs, ggml_tensor * mm);

// [a770-opt] a * sigmoid(gate view) + sign flip + 4096-point rotation in one kernel (attention output gate); false = not taken
bool ggml_sycl_fwht4096_gated(ggml_backend_sycl_context & ctx, const float * a, const float * gbase, int gstride, int D,
                              const float * signs, ggml_tensor * mm);

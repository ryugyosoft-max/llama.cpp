#pragma once

#include "common.hpp"

// [a770-opt] Llama-style single-token decode fusions for Arc Alchemist (see llama-xe1.cpp).
//
// q8 stash: an op that produces a mat-vec activation can also write its q8_1 SoA quantization (the layout of
// quantize_and_reorder_q8_1_soa) into a fixed device slot and register it under the f32 tensor's data pointer; the
// Q4_0 mat-vecs then skip their own quantization kernel. Entries live for one graph build (cleared at its start), slots
// are a ring, so a consumer must run within the next few producers (true inside a transformer layer).

void * ggml_sycl_xe1_stash_slot(dpct::queue_ptr stream, size_t bytes);  // nullptr if too large / no memory
void   ggml_sycl_xe1_stash_put(const void * f32, const void * q8);
const void * ggml_sycl_xe1_stash_get(const void * f32);
void   ggml_sycl_xe1_stash_clear();
// forget entries whose f32 tensor starts inside [p, p + n): that memory is being rewritten by another op
void   ggml_sycl_xe1_stash_drop_range(const void * p, size_t n);

// [add +] rms_norm * w for one row of K floats (K % 32 == 0, K <= 16384): add_out (nullable) = a + b, normed = norm * w,
// q8 = SoA quantization of normed. b may be null (no add).
void ggml_sycl_xe1_add_rmsnorm_q8(dpct::queue_ptr stream, const float * a, const float * b, float * add_out,
                                  const float * w, float eps, float * normed, void * q8, int K);

// silu(g) * u for K floats (K % 32 == 0), out f32 and its q8 SoA quantization
void ggml_sycl_xe1_swiglu_q8(dpct::queue_ptr stream, const float * g, const float * u, float * out, void * q8, int K);

// one 32-value q8_1 block per sub-group of 16 lanes, 2 values per lane (same math as quantize_q8_1_impl)
static inline void ggml_sycl_xe1_quant_block_store(const float y0, const float y1, int8_t * q8, const int K, const int blk,
                                     const sycl::sub_group & sg) {
    const int   lane = sg.get_local_linear_id();
    const float sum  = sycl::reduce_over_group(sg, y0 + y1, sycl::plus<float>());
    const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(y0), sycl::fabs(y1)), sycl::maximum<float>());
    const float d    = amax == 0 ? 1.0f : amax / 127;
    const int8_t q0  = (int8_t) sycl::round(y0 / d);
    const int8_t q1  = (int8_t) sycl::round(y1 / d);
    sycl::vec<int8_t, 2> qv(q0, q1);
    *reinterpret_cast<sycl::vec<int8_t, 2> *>(q8 + blk * 32 + 2 * lane) = qv;
    if (lane == 0) {
        *((sycl::half2 *) (q8 + K) + blk) = sycl::half2(sycl::half(amax == 0 ? 0.0f : d), sycl::half(sum));
    }
}


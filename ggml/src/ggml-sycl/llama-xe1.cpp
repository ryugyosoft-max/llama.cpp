// [a770-opt] Llama-style decode fusions for Arc Alchemist. Each dependent kernel boundary costs ~4-6 us of GPU time on
// the A770 (graph replay included), and a Llama 2 7B token had ~17 kernels per layer: 2 of them were q8 quantizations
// of a norm output and 1 the quantization of the SWIGLU output. Here the norm and the SWIGLU write the q8_1 SoA copy
// themselves (bit-identical to quantize_and_reorder_q8_1_soa) and register it in the stash for the Q4_0 mat-vecs.

#include "llama-xe1.hpp"

#include <map>

namespace {

constexpr int    STASH_SLOTS      = 8;
constexpr size_t STASH_SLOT_BYTES = 64 * 1024;
constexpr int    STASH_ENTRIES    = 16;

struct stash_entry {
    const void * f32 = nullptr;
    const void * q8  = nullptr;
};
stash_entry g_stash[STASH_ENTRIES];
int         g_stash_next = 0;

}  // namespace

void * ggml_sycl_xe1_stash_slot(dpct::queue_ptr stream, size_t bytes) {
    static std::map<int, char *> bufs;
    static int                   next = 0;
    if (bytes > STASH_SLOT_BYTES) {
        return nullptr;
    }
    char *& b = bufs[ggml_sycl_get_device()];
    if (!b) {
        b = (char *) sycl::malloc_device(STASH_SLOTS * STASH_SLOT_BYTES, *stream);
        if (!b) {
            return nullptr;
        }
    }
    next = (next + 1) % STASH_SLOTS;
    return b + (size_t) next * STASH_SLOT_BYTES;
}

void ggml_sycl_xe1_stash_put(const void * f32, const void * q8) {
    for (auto & e : g_stash) {  // a re-used f32 buffer (allocator) replaces its old entry
        if (e.f32 == f32) {
            e.q8 = q8;
            return;
        }
    }
    g_stash[g_stash_next] = { f32, q8 };
    g_stash_next          = (g_stash_next + 1) % STASH_ENTRIES;
}

const void * ggml_sycl_xe1_stash_get(const void * f32) {
    for (const auto & e : g_stash) {
        if (e.f32 == f32) {
            return e.q8;
        }
    }
    return nullptr;
}

void ggml_sycl_xe1_stash_drop_range(const void * p, size_t n) {
    const char * a = (const char *) p;
    for (auto & e : g_stash) {
        const char * f = (const char *) e.f32;
        if (f && f >= a && f < a + n) {
            e = {};
        }
    }
}

void ggml_sycl_xe1_stash_clear() {
    for (auto & e : g_stash) {
        e = {};
    }
    g_stash_next = 0;
}

void ggml_sycl_xe1_add_rmsnorm_q8(dpct::queue_ptr stream, const float * a, const float * b, float * add_out,
                                  const float * w, float eps, float * normed, void * q8, int K) {
    GGML_ASSERT(K % 32 == 0 && K <= 16384);
    constexpr int WG = 1024;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sx(sycl::range<1>(K), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(WG), sycl::range<1>(WG)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                             const int tid = it.get_local_id(0);
                             float     ss  = 0.0f;
                             for (int e = tid; e < K; e += WG) {
                                 float x = a[e];
                                 if (b) {
                                     x += b[e];
                                 }
                                 if (add_out) {
                                     add_out[e] = x;
                                 }
                                 sx[e] = x;
                                 ss += x * x;
                             }
                             ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                             const float scale = sycl::rsqrt(ss / K + eps);
                             // reduce_over_group is a work-group barrier: sx is visible to every work-item here
                             const auto sg   = it.get_sub_group();
                             const int  lane = sg.get_local_linear_id();
                             const int  nsg  = sg.get_group_linear_range();
                             for (int blk = sg.get_group_linear_id(); blk < K / 32; blk += nsg) {
                                 const int   e0 = blk * 32 + 2 * lane;
                                 const float y0 = sx[e0] * scale * w[e0];
                                 const float y1 = sx[e0 + 1] * scale * w[e0 + 1];
                                 normed[e0]     = y0;
                                 normed[e0 + 1] = y1;
                                 ggml_sycl_xe1_quant_block_store(y0, y1, (int8_t *) q8, K, blk, sg);
                             }
                         });
    });
}

void ggml_sycl_xe1_swiglu_q8(dpct::queue_ptr stream, const float * g, const float * u, float * out, void * q8, int K) {
    GGML_ASSERT(K % 32 == 0);
    constexpr int SGS  = 8;  // sub-groups (blocks) per work-group
    const int     nblk = K / 32;
    const int     nwg  = (nblk + SGS - 1) / SGS;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGS * 16), sycl::range<1>(SGS * 16)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                             const auto sg  = it.get_sub_group();
                             const int  blk = it.get_group(0) * SGS + sg.get_group_linear_id();
                             if (blk >= nblk) {
                                 return;
                             }
                             const int   e0 = blk * 32 + 2 * sg.get_local_linear_id();
                             const float g0 = g[e0], g1 = g[e0 + 1];
                             // same expression as the SWIGLU kernel (op_silu(x) = x / (1 + exp(-x)))
                             const float y0 = g0 / (1.0f + sycl::native::exp(-g0)) * u[e0];
                             const float y1 = g1 / (1.0f + sycl::native::exp(-g1)) * u[e0 + 1];
                             out[e0]        = y0;
                             out[e0 + 1]    = y1;
                             ggml_sycl_xe1_quant_block_store(y0, y1, (int8_t *) q8, K, blk, sg);
                         });
    });
}

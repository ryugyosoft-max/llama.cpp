#include "fwht.hpp"
#include "llama-xe1.hpp"
#include "fattn-xe1dec.hpp"

static void launch_fwht4096_wg1024(const float * src, float * dst, const int64_t n_rows, const float scale,
                                   const float * signs, const int n_blk, dpct::queue_ptr stream, int8_t * q8 = nullptr);

#include <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
static constexpr float H12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N }
};

static constexpr float H20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N }
};

#undef P
#undef N

template <int N, typename T>
static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = static_cast<float>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N, typename T>
static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <int N, int NT, typename T>
static void fwht_kernel_wide(const T * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item,
                             float *                  smem) {
    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    const int tid = item.get_local_id(1);

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = static_cast<float>(src[i * NT + tid]) * scale;
    }

    const sycl::sub_group sg   = item.get_sub_group();
    const int             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    for (int h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
        for (int j = 0; j < el_w; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <int N, int NT, typename T>
static void launch_fwht_wide(const T *       src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    const sycl::range<2> global(n_rows, NT);
    const sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <int N, int m, typename T>
static void kronecker_kernel(const T * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int blocks_per_group = N / m;
    constexpr int el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    float     reg[el_w * m];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;

#pragma unroll
        for (int j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<float>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    for (int b = 0; b < el_w; ++b) {
        float z[m] = { 0.0f };

#pragma unroll
        for (int i = 0; i < m; ++i) {
#pragma unroll
            for (int j = 0; j < m; ++j) {
                const float h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        for (int i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
#pragma unroll
            for (int k = 0; k < m; ++k) {
                const float val  = reg[j * m + k];
                const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    for (int h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int s = 0; s < step; ++s) {
#pragma unroll
                for (int k = 0; k < m; ++k) {
                    const float x = reg[(j + s) * m + k];
                    const float y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;
#pragma unroll
        for (int k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <int N, int m, typename T>
static void launch_kronecker(const T *       src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
static bool ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *       src_d  = (const T *) src->data;
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}

// [a770-opt] same acceptance test as ggml_sycl_op_fwht(), without launching (used by the SYCL graph compatibility scan)
bool ggml_sycl_fwht_supported(const ggml_tensor * src, const ggml_tensor * dst) {
    if (dst->type != GGML_TYPE_F32 || !ggml_are_same_shape(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
        return false;
    }
    switch (src->ne[0]) {
        case 64: case 128: case 256: case 512: case 384: case 768: case 640: case 1280:
        case 1024: case 2048: case 4096: case 8192:
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    ggml_sycl_xe1_stash_drop_range(dst->data, ggml_nbytes(dst));  // [a770-opt] no q8 copy: an older one here is stale
    {  // [a770-opt] decode-sized 4096-point rows: the 1024-wide kernel (as the signed rotations)
        static const bool on = ggml_sycl_knob("GGML_SYCL_FWHT4096", ggml_sycl_device_is_alchemist(ctx.device) ? 1 : 0) &&
                               ggml_sycl_knob("GGML_SYCL_FWHT1024", 1);
        if (on && src->type == GGML_TYPE_F32 && src->ne[0] == 4096 && ggml_nrows(src) <= 8) {
            launch_fwht4096_wg1024((const float *) src->data, (float *) dst->data, ggml_nrows(src), 1.0f / 64.0f, nullptr,
                                   1, ctx.stream(), nullptr);
            return true;
        }
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_sycl_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        default:
            return false;
    }
}

// ARC-LAB (Torchit1/llama.cpp arc-b580): FWHT with the Hadamard sign flip done during the load
// (dst = H * (src * signs)), F32 or F16 input. Kept beside the general FWHT above, which carries the
// wide and Kronecker sizes from upstream; only the fused sign-flip entry point below uses it.
namespace fwht_signed_impl {

template <int N, typename T, bool has_signs>
static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const float * __restrict__ signs, const int n_blk,
                        const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = (float) src[i * WARP_SIZE + lane] * scale;
        if constexpr (has_signs) {
            reg[i] *= signs_row[i * WARP_SIZE + lane];
        }
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

// Wide rows: one work-group per row, butterflies through local memory.
template <int N, typename T, bool has_signs>
static void fwht_kernel_slm(const T * __restrict__ src, float * __restrict__ dst, const float scale,
                            const float * __restrict__ signs, const int n_blk, float * s, const sycl::nd_item<1> & item) {
    const int64_t r   = item.get_group(0);
    const int     tid = item.get_local_id(0);
    const int     nt  = item.get_local_range(0);

    src += r * N;
    dst += r * N;
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    for (int i = tid; i < N; i += nt) {
        float v = (float) src[i] * scale;
        if constexpr (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int h = 1; h < N; h *= 2) {
        for (int idx = tid; idx < N / 2; idx += nt) {
            const int   j = (idx / h) * 2 * h + idx % h;
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int i = tid; i < N; i += nt) {
        dst[i] = s[i];
    }
}

template <int N, typename T, bool has_signs>
static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale, const float * signs,
                        const int n_blk, dpct::queue_ptr stream) {
    if constexpr (N <= 1024) {
        constexpr int rows_per_block = 4;

        const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

        // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
        const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
        const sycl::range<2> local(rows_per_block, WARP_SIZE);

        stream->parallel_for(sycl::nd_range<2>(global, local),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 fwht_kernel<N, T, has_signs>(src, dst, n_rows, scale, signs, n_blk, item);
                             });
    } else {
        constexpr int nt = 256;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s(sycl::range<1>(N), cgh);
            cgh.parallel_for(sycl::nd_range<1>(n_rows * nt, nt), [=](sycl::nd_item<1> item) {
                fwht_kernel_slm<N, T, has_signs>(src, dst, scale, signs, n_blk,
                                                 s.get_multi_ptr<sycl::access::decorated::no>().get(), item);
            });
        });
    }
}

// [a770-opt] 4096-point rows (every Bonsai / Qwen3.5-prism rotation; the FFN-down one is 3 such rows) for Arc
// Alchemist decode. fwht_kernel_slm runs 12 barrier-separated SLM passes in one work-group per row - one row is a
// single work-group on the whole GPU, so each rotation was ~10 us of serial latency, ~130 times per token. Here 256
// work-items hold 16 values each (element k * 256 + t): the 4 butterfly stages over the register index run in
// registers, the 4 over the lane index through sub-group shuffles, and one SLM transpose turns the sub-group index
// into the register index for the last 4. With q8 set the kernel also writes the q8_1 SoA quantization of the
// result (quantize_and_reorder_q8_1_soa layout and math over the rows taken as one column) for the PTQ1_0 mat-vecs.
template <bool has_signs>
static void fwht4096_kernel(const float * __restrict__ src, float * __restrict__ dst, const float scale,
                            const float * __restrict__ signs, const int n_blk, int8_t * __restrict__ q8,
                            const int ncols_q8, float * s, const sycl::nd_item<1> & item) {
    constexpr int N = 4096, NT = 256, R = 16;
    const auto    sg   = item.get_sub_group();
    const int     r    = (int) item.get_group(0);
    const int     t    = (int) item.get_local_id(0);
    const int     lane = sg.get_local_linear_id();
    src += (int64_t) r * N;
    dst += (int64_t) r * N;
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float v[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
        const int i = k * NT + t;
        v[k]        = src[i] * scale;
        if constexpr (has_signs) {
            v[k] *= signs_row[i];
        }
    }
    auto reg_stages = [&]() {
#pragma unroll
        for (int h = 1; h < R; h *= 2) {
#pragma unroll
            for (int k = 0; k < R; ++k) {
                if ((k & h) == 0) {
                    const float x = v[k], y = v[k + h];
                    v[k]     = x + y;
                    v[k + h] = x - y;
                }
            }
        }
    };
    reg_stages();  // index bits 8..11
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {  // bits 0..3 (lane)
#pragma unroll
        for (int k = 0; k < R; ++k) {
            const float o = dpct::permute_sub_group_by_xor(sg, v[k], h, WARP_SIZE);
            v[k]          = (lane & h) == 0 ? v[k] + o : o - v[k];
        }
    }
#pragma unroll
    for (int k = 0; k < R; ++k) {
        s[k * NT + t] = v[k];
    }
    item.barrier(sycl::access::fence_space::local_space);
    // work-item t = hi * 16 + lo now holds elements hi * 256 + k * 16 + lo: bits 4..7 are the register index
    const int hi = t >> 4, lo = t & 15;
#pragma unroll
    for (int k = 0; k < R; ++k) {
        v[k] = s[(hi << 8) | (k << 4) | lo];
    }
    reg_stages();
#pragma unroll
    for (int k = 0; k < R; ++k) {
        dst[(hi << 8) | (k << 4) | lo] = v[k];
    }
    if (q8) {
        // q8_1 block b = hi * 8 + m holds elements hi * 256 + 32 m + {lo, 16 + lo}: registers 2m, 2m + 1 of a sub-group
        int8_t *      qr = q8 + (int64_t) r * N;
        sycl::half2 * ds = (sycl::half2 *) (q8 + ncols_q8) + r * (N / QK8_1);
#pragma unroll
        for (int m = 0; m < R / 2; ++m) {
            const float y0   = v[2 * m], y1 = v[2 * m + 1];
            const float sum  = sycl::reduce_over_group(sg, y0 + y1, sycl::plus<float>());
            const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(y0), sycl::fabs(y1)), sycl::maximum<float>());
            const float d    = amax == 0 ? 1.0f : amax / 127;
            const int   i0   = (hi << 8) | (m << 5) | lo;
            qr[i0]           = (int8_t) sycl::round(y0 / d);
            qr[i0 + 16]      = (int8_t) sycl::round(y1 / d);
            if (lane == 0) {
                ds[hi * 8 + m] = sycl::half2(sycl::half(amax == 0 ? 0.0f : d), sycl::half(sum));
            }
        }
    }
}

static void launch_fwht4096(const float * src, float * dst, const int64_t n_rows, const float scale,
                            const float * signs, const int n_blk, int8_t * q8, dpct::queue_ptr stream) {
    const int ncols_q8 = (int) n_rows * 4096;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> s(sycl::range<1>(4096), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_rows * 256, 256), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            float * sp = s.get_multi_ptr<sycl::access::decorated::no>().get();
            if (signs) {
                fwht4096_kernel<true>(src, dst, scale, signs, n_blk, q8, ncols_q8, sp, item);
            } else {
                fwht4096_kernel<false>(src, dst, scale, nullptr, 1, q8, ncols_q8, sp, item);
            }
        });
    });
}

template <typename T>
static bool fwht_dispatch(const T * src_d, float * dst_d, const int n, const int64_t rows, const float scale,
                          const float * signs, const int n_blk, dpct::queue_ptr stream) {
    switch (n) {
#define FWHT_CASE(NN)                                                                      \
        case NN:                                                                           \
            if (signs) {                                                                   \
                launch_fwht<NN, T, true>(src_d, dst_d, rows, scale, signs, n_blk, stream); \
            } else {                                                                       \
                launch_fwht<NN, T, false>(src_d, dst_d, rows, scale, nullptr, 1, stream);  \
            }                                                                              \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        FWHT_CASE(512)
        FWHT_CASE(1024)
        FWHT_CASE(2048)
        FWHT_CASE(4096)
        FWHT_CASE(8192)
#undef FWHT_CASE
        default:
            return false;
    }
}

static bool fwht_run(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                     const ggml_tensor * signs_t) {
    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_nelements(src) != ggml_nelements(dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    const float * signs = nullptr;
    int           n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = (int) (signs_t->ne[0] / n);
    }

    const float scale = 1.0f / std::sqrt((float) n);
    ggml_sycl_xe1_stash_drop_range(dst->data, ggml_nbytes(dst));  // a q8 copy of an older tensor here is stale now
    static const bool skip_rot = getenv("GGML_SYCL_SKIP_ROT") != nullptr;  // timing ablation only: output is garbage
    if (skip_rot && rows <= 8) {
        return true;
    }
    // [a770-opt] decode-sized 4096-point rotations: register/sub-group kernel, plus the q8_1 copy for the mat-vecs
    static const bool r4096 = ggml_sycl_knob("GGML_SYCL_FWHT4096", ggml_sycl_device_is_alchemist(ctx.device) ? 1 : 0) != 0;
    if (r4096 && n == 4096 && src->type == GGML_TYPE_F32 && rows <= 8) {
        static const bool q8_on = ggml_sycl_knob("GGML_SYCL_FWHT_Q8", 0) != 0;  // costs more than the quantize launch it saves
        int8_t * q8 = nullptr;
        if (q8_on && rows <= 3) {  // one column of up to 12288 (FFN down: 3 rows) -> one mat-vec input
            q8 = (int8_t *) ggml_sycl_xe1_stash_slot(ctx.stream(), (size_t) rows * 4096 + rows * 128 * 4);
        }
        static const bool wg1024 = ggml_sycl_knob("GGML_SYCL_FWHT1024", 1) != 0;
        static const bool rot_q8 = ggml_sycl_knob("GGML_SYCL_ROT_Q8", 0) != 0;  // slower: lengthens the 1-work-group kernel
        if (wg1024) {
            int8_t * q8w = q8;
            if (!q8w && rot_q8 && rows <= 3) {
                q8w = (int8_t *) ggml_sycl_xe1_stash_slot(ctx.stream(), (size_t) rows * 4096 + rows * 128 * 4);
            }
            launch_fwht4096_wg1024((const float *) src->data, (float *) dst->data, rows, scale, signs, n_blk, ctx.stream(), q8w);
            if (q8w) {
                ggml_sycl_xe1_stash_put(dst->data, q8w);
            }
            return true;
        } else {
            launch_fwht4096((const float *) src->data, (float *) dst->data, rows, scale, signs, n_blk, q8, ctx.stream());
        }
        if (q8) {
            ggml_sycl_xe1_stash_put(dst->data, q8);
        }
        return true;
    }
    if (src->type == GGML_TYPE_F32) {
        return fwht_dispatch((const float *) src->data, (float *) dst->data, n, rows, scale, signs, n_blk, ctx.stream());
    }
    return fwht_dispatch((const sycl::half *) src->data, (float *) dst->data, n, rows, scale, signs, n_blk, ctx.stream());
}

} // namespace fwht_signed_impl

// [a770-opt] the 4096-point butterflies of one row held by a 1024-wide work-group, 4 values per work-item: in v[k] the
// element k * 1024 + t; out: v[k] holds element lo4 | lane << 4 | k << 8 | hi2 << 10 (lo4 = (t >> 4) & 15, hi2 = t >> 8)
static inline void fwht4096_wg1024(float (&v)[4], float * s, const sycl::nd_item<1> & it) {
    const auto sg   = it.get_sub_group();
    const int  t    = (int) it.get_local_id(0);
    const int  lane = sg.get_local_linear_id();
    auto reg2 = [&]() {
#pragma unroll
        for (int h = 1; h < 4; h *= 2) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((k & h) == 0) {
                    const float x = v[k], y = v[k + h];
                    v[k] = x + y;
                    v[k + h] = x - y;
                }
            }
        }
    };
    auto lanes4 = [&]() {
#pragma unroll
        for (int h = 1; h < 16; h *= 2) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float o = sycl::permute_group_by_xor(sg, v[k], h);
                v[k] = (lane & h) == 0 ? v[k] + o : o - v[k];
            }
        }
    };
    auto pad = [](int i) { return i + (i >> 4); };
    reg2();
    lanes4();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s[pad(k * 1024 + t)] = v[k];
    }
    it.barrier(sycl::access::fence_space::local_space);
    const int lo4 = (t >> 4) & 15, hi2 = t >> 8;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = s[pad(lo4 | (lane << 4) | (k << 8) | (hi2 << 10))];
    }
    lanes4();
    reg2();
}

// q8_1 SoA copy of a rotated row (quantize_and_reorder_q8_1_soa math) from the final register layout of
// fwht4096_wg1024: the values go through SLM so each sub-group quantizes 2 contiguous 32-blocks (2 reductions each)
static inline void fwht4096_wg1024_q8(const float (&v)[4], float * s, const sycl::nd_item<1> & it, int8_t * qrow,
                                      sycl::half2 * ds_row) {
    const auto sg   = it.get_sub_group();
    const int  t    = (int) it.get_local_id(0);
    const int  lane = sg.get_local_linear_id();
    const int  lo4 = (t >> 4) & 15, hi2 = t >> 8;
    it.barrier(sycl::access::fence_space::local_space);  // the transpose reads are done
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s[lo4 | (lane << 4) | (k << 8) | (hi2 << 10)] = v[k];
    }
    it.barrier(sycl::access::fence_space::local_space);
    const int sgid = (int) sg.get_group_linear_id();
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
        const int   b    = sgid * 2 + bb;
        const float y0   = s[32 * b + lane], y1 = s[32 * b + 16 + lane];
        const float sum  = sycl::reduce_over_group(sg, y0 + y1, sycl::plus<float>());
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(y0), sycl::fabs(y1)), sycl::maximum<float>());
        const float d    = amax == 0 ? 1.0f : amax / 127;
        qrow[32 * b + lane]      = (int8_t) sycl::round(y0 / d);
        qrow[32 * b + 16 + lane] = (int8_t) sycl::round(y1 / d);
        if (lane == 0) {
            ds_row[b] = sycl::half2(sycl::half(amax == 0 ? 0.0f : d), sycl::half(sum));
        }
    }
}

// standalone 4096-point rotation of up to a few rows (one 1024-wide work-group per row)
static void launch_fwht4096_wg1024(const float * src, float * dst, const int64_t n_rows, const float scale,
                                   const float * signs, const int n_blk, dpct::queue_ptr stream, int8_t * q8) {
    const int ncols_q8 = (int) n_rows * 4096;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sl(sycl::range<1>(4096 + 256), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_rows * 1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int     r  = (int) it.get_group(0);
            const int     t  = (int) it.get_local_id(0);
            const float * sr = src + r * 4096;
            const float * gr = signs ? signs + (r % n_blk) * 4096 : nullptr;
            float         v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = k * 1024 + t;
                v[k]        = sr[i] * scale * (gr ? gr[i] : 1.0f);
            }
            fwht4096_wg1024(v, sl.get_multi_ptr<sycl::access::decorated::no>().get(), it);
            const int lane = it.get_sub_group().get_local_linear_id(), lo4 = (t >> 4) & 15, hi2 = t >> 8;
            float *   dr   = dst + r * 4096;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dr[lo4 | (lane << 4) | (k << 8) | (hi2 << 10)] = v[k];
            }
            if (q8) {
                fwht4096_wg1024_q8(v, sl.get_multi_ptr<sycl::access::decorated::no>().get(), it, q8 + r * 4096,
                                   (sycl::half2 *) (q8 + ncols_q8) + r * 128);
            }
        });
    });
}

// [a770-opt] one row of 4096: [residual add +] RMS norm * w + sign flip + 4096-point rotation in one work-group of
// 1024 (4 values per work-item: few registers - a 256-wide version with 16 values plus four input streams spilled and
// lost to the separate kernels). Element i = k * 1024 + t: bits 10..11 are the register index, bits 0..3 the lane;
// one SLM transpose then gives work-item t' = hi2 * 64 + lo4 * 4 ... the elements lo4 | lane' << 4 | k' << 8 | hi2 << 10
// (lane' = bits 4..7 through shuffles, k' = bits 8..9 in registers).
bool ggml_sycl_fwht4096_add_norm_rot(ggml_backend_sycl_context & ctx, const float * a, const float * b, float * add_out,
                                     const float * w, float eps, float * normed, const float * signs, ggml_tensor * mm) {
    if (!ggml_sycl_knob("GGML_SYCL_FWHT4096", ggml_sycl_device_is_alchemist(ctx.device) ? 1 : 0) ||
        !ggml_sycl_knob("GGML_SYCL_FWHT_NORM", 1) || mm->type != GGML_TYPE_F32 || ggml_nelements(mm) != 4096) {
        return false;
    }
    ggml_sycl_xe1_stash_drop_range(mm->data, ggml_nbytes(mm));
    float * dst = (float *) mm->data;
    static const bool rot_q8 = ggml_sycl_knob("GGML_SYCL_ROT_Q8", 0) != 0;  // slower: lengthens the 1-work-group kernel
    int8_t * q8 = rot_q8 ? (int8_t *) ggml_sycl_xe1_stash_slot(ctx.stream(), 4096 + 128 * 4) : nullptr;
    ctx.stream()->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sl(sycl::range<1>(4096 + 256 + 64), cgh);
        cgh.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg   = it.get_sub_group();
            const int  t    = (int) it.get_local_id(0);
            const int  lane = sg.get_local_linear_id();
            float *    s    = sl.get_multi_ptr<sycl::access::decorated::no>().get();
            float      v[4], ss = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = k * 1024 + t;
                float     x = a[i];
                if (b) {
                    x += b[i];
                }
                v[k] = x;
                ss += x * x;
            }
            {  // sub-group sums -> SLM -> one sub-group (the generic 1024-wide reduce_over_group is slower)
                static_assert(1024 / 16 == 64, "64 sub-groups");
                const auto sg0 = it.get_sub_group();
                ss = sycl::reduce_over_group(sg0, ss, sycl::plus<float>());
                if (sg0.get_local_linear_id() == 0) {
                    s[4096 + 256 + sg0.get_group_linear_id()] = ss;  // past the padded transpose area (max index 4350)
                }
                it.barrier(sycl::access::fence_space::local_space);
                const int l = sg0.get_local_linear_id();
                float     p = s[4352 + l] + s[4352 + 16 + l] + s[4352 + 32 + l] + s[4352 + 48 + l];
                ss          = sycl::reduce_over_group(sg0, p, sycl::plus<float>());
            }
            const float rnorm = sycl::rsqrt(ss / 4096 + eps);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = k * 1024 + t;
                if (add_out) {
                    add_out[i] = v[k];
                }
                const float y = v[k] * rnorm * w[i];
                if (normed) {
                    normed[i] = y;
                }
                v[k] = y * signs[i] * (1.0f / 64.0f);
            }
            fwht4096_wg1024(v, s, it);
            const int lo4 = (t >> 4) & 15, hi2 = t >> 8;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst[lo4 | (lane << 4) | (k << 8) | (hi2 << 10)] = v[k];
            }
            if (q8) {
                fwht4096_wg1024_q8(v, s, it, q8, (sycl::half2 *) (q8 + 4096));
            }
        });
    });
    if (q8) {
        ggml_sycl_xe1_stash_put(mm->data, q8);
    }
    return true;
}

// [a770-opt] attention output gate in the rotation's load: x = a * sigmoid(gate) with the gate read straight from its
// strided view (row h of D values at gbase + h * gstride), then signs + 4096-point rotation (1024-wide kernel)
bool ggml_sycl_fwht4096_gated(ggml_backend_sycl_context & ctx, const float * a, const float * gbase, int gstride, int D,
                              const float * signs, ggml_tensor * mm) {
    if (!ggml_sycl_knob("GGML_SYCL_FWHT4096", ggml_sycl_device_is_alchemist(ctx.device) ? 1 : 0) ||
        !ggml_sycl_knob("GGML_SYCL_FWHT1024", 1) || !ggml_sycl_knob("GGML_SYCL_FWHT_GATE", 1) ||
        mm->type != GGML_TYPE_F32 || ggml_nelements(mm) != 4096 || D <= 0 || (D & (D - 1)) != 0 || 4096 % D != 0) {
        return false;
    }
    ggml_sycl_xe1_stash_drop_range(mm->data, ggml_nbytes(mm));
    int dl = 0;
    while ((1 << dl) < D) ++dl;
    const int dmask = D - 1;
    float *   dst   = (float *) mm->data;
    ctx.stream()->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sl(sycl::range<1>(4096 + 256), cgh);
        cgh.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int t = (int) it.get_local_id(0);
            float     v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int   i = k * 1024 + t;
                const float g = gbase[(i >> dl) * gstride + (i & dmask)];
                v[k]          = a[i] / (1.0f + sycl::exp(-g)) * signs[i] * (1.0f / 64.0f);
            }
            fwht4096_wg1024(v, sl.get_multi_ptr<sycl::access::decorated::no>().get(), it);
            const int lane = it.get_sub_group().get_local_linear_id(), lo4 = (t >> 4) & 15, hi2 = t >> 8;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst[lo4 | (lane << 4) | (k << 8) | (hi2 << 10)] = v[k];
            }
        });
    });
    return true;
}

bool ggml_sycl_op_fwht_signed(ggml_backend_sycl_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                              ggml_tensor * dst) {
    return fwht_signed_impl::fwht_run(ctx, src, dst, signs);
}

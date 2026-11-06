// [a770-opt] Single-column Q4_0 mat-vec for Arc Alchemist (reordered Q4_0: all 16-byte qs blocks row-major, then the
// fp16 scales; activations: q8_1 SoA = int8 quants, then half2 (d, d * sum) per 32 values).
//
// The generic reorder kernel gives each lane 8 bytes of a block through 2-byte loads (the qs helper only assumes 2-byte
// alignment), ~100 instructions per 4 dp4a and few loads in flight (Llama 2 7B: 60 t/s, ~210 GB/s). Here a lane takes
// whole blocks: one 16-byte weight load + 32 activation bytes per block, NR rows per subgroup share the activation
// loads, UNROLL blocks per lane per iteration keep several 16-byte loads in flight, offsets are 32-bit.

#include "q4_0-xe1.hpp"

namespace {

template <int NR, int UNROLL>
static void q4_0_xe1_kernel(const uint8_t * __restrict__ vx, const int8_t * __restrict__ vy, float * __restrict__ dst,
                            const int ncols, const int nrows, const sycl::nd_item<1> & it) {
    const auto sg   = it.get_sub_group();
    const int  lane = sg.get_local_linear_id();
    const int  nsg  = sg.get_group_linear_range();
    const int  row0 = (it.get_group(0) * nsg + sg.get_group_linear_id()) * NR;
    if (row0 >= nrows) {
        return;
    }

    const int nblk = ncols / QK4_0;  // blocks per row
    const sycl::half * d_base = (const sycl::half *) (vx + (uint32_t) (ncols / 2) * (uint32_t) nrows);
    const sycl::half2 * ys = (const sycl::half2 *) (vy + ncols);

    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        acc[r] = 0.0f;
    }

    constexpr int STEP = WARP_SIZE * UNROLL;
    for (int b0 = lane; b0 < nblk; b0 += STEP) {
        // issue every load of the iteration first
        sycl::int4 a_lo[UNROLL], a_hi[UNROLL];
        sycl::half2 ds[UNROLL];
        sycl::uint4 q[UNROLL][NR];
        sycl::half dw[UNROLL][NR];
        bool ok[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int b = b0 + u * WARP_SIZE;
            ok[u] = b < nblk;
            const int bb = ok[u] ? b : b0;
            const sycl::int4 * ya = (const sycl::int4 *) (vy + (uint32_t) bb * QK8_1);
            a_lo[u] = ya[0];
            a_hi[u] = ya[1];
            ds[u]   = ys[bb];
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                const int      row = sycl::min(row0 + r, nrows - 1);
                const uint32_t ib  = (uint32_t) row * (uint32_t) nblk + (uint32_t) bb;
                q[u][r]  = *(const sycl::uint4 *) (vx + ib * 16u);
                dw[u][r] = d_base[ib];
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (!ok[u]) {
                continue;
            }
            const sycl::float2 dsf = ds[u].template convert<float, sycl::rounding_mode::automatic>();
            const int al[4] = { a_lo[u].x(), a_lo[u].y(), a_lo[u].z(), a_lo[u].w() };
            const int ah[4] = { a_hi[u].x(), a_hi[u].y(), a_hi[u].z(), a_hi[u].w() };
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                const uint32_t qq[4] = { q[u][r].x(), q[u][r].y(), q[u][r].z(), q[u][r].w() };
                int sumi = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    // byte j of qs: value j in the low nibble, value j + 16 in the high nibble
                    sumi = dpct::dp4a((int) (qq[k] & 0x0F0F0F0Fu), al[k], sumi);
                    sumi = dpct::dp4a((int) ((qq[k] >> 4) & 0x0F0F0F0Fu), ah[k], sumi);
                }
                acc[r] += (float) dw[u][r] * (sumi * dsf.x() - 8.0f * dsf.y());
            }
        }
    }

#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float s = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
        if (lane == 0 && row0 + r < nrows) {
            dst[row0 + r] = s;
        }
    }
}

template <int NR, int UNROLL>
static void launch(const void * vx, const void * vy, float * dst, int ncols, int nrows, dpct::queue_ptr stream) {
    constexpr int SGS  = 8;  // subgroups per workgroup
    const int     nwg  = (nrows + NR * SGS - 1) / (NR * SGS);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGS * WARP_SIZE), sycl::range<1>(SGS * WARP_SIZE)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             q4_0_xe1_kernel<NR, UNROLL>((const uint8_t *) vx, (const int8_t *) vy, dst, ncols, nrows, it);
                         });
    });
}

}  // namespace

bool ggml_sycl_q4_0_xe1_enabled(int device) {
    static const int env = getenv("GGML_SYCL_Q4_0_XE1") ? atoi(getenv("GGML_SYCL_Q4_0_XE1")) : -1;
    if (env >= 0) {
        return env != 0;
    }
    return ggml_sycl_device_is_alchemist(device);
}

void ggml_sycl_q4_0_xe1_mul_mat_vec(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                    dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    static const int nr = getenv("GGML_SYCL_Q4_0_XE1_NR") ? atoi(getenv("GGML_SYCL_Q4_0_XE1_NR")) : 1;  // A770: 1 row 79.8, 2 rows 79.5, 4 rows 74.9 t/s (Llama 2 7B)
    static const int un = getenv("GGML_SYCL_Q4_0_XE1_UNROLL") ? atoi(getenv("GGML_SYCL_Q4_0_XE1_UNROLL")) : 1;
    if (nr == 1) {
        un == 1 ? launch<1, 1>(vx, vy, dst, ncols, nrows, stream) : launch<1, 2>(vx, vy, dst, ncols, nrows, stream);
    } else if (nr == 4) {
        un == 1 ? launch<4, 1>(vx, vy, dst, ncols, nrows, stream) : launch<4, 2>(vx, vy, dst, ncols, nrows, stream);
    } else {
        un == 1 ? launch<2, 1>(vx, vy, dst, ncols, nrows, stream) : launch<2, 2>(vx, vy, dst, ncols, nrows, stream);
    }
}

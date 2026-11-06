// [a770-opt] Single-column Q4_0 mat-vec for Arc Alchemist (reordered Q4_0: all 16-byte qs blocks row-major, then the
// fp16 scales; activations: q8_1 SoA = int8 quants, then half2 (d, d * sum) per 32 values).
//
// The generic reorder kernel gives each lane 8 bytes of a block through 2-byte loads (the qs helper only assumes 2-byte
// alignment), ~100 instructions per 4 dp4a and few loads in flight (Llama 2 7B: 60 t/s, ~210 GB/s). Here a lane takes
// whole blocks: one 16-byte weight load + 32 activation bytes per block, NR rows per subgroup share the activation
// loads, UNROLL blocks per lane per iteration keep several 16-byte loads in flight, offsets are 32-bit.

#include "q4_0-xe1.hpp"

namespace {
static int q4_0_xe1_ks();
template <int KS> static void launch_ks(const void * vx, const void * vy, float * dst, int ncols, int nrows, dpct::queue_ptr stream);
}
static bool q4_0_xe1_try_coop(const void * const * vx, float * const * dst, const int * nrows, int nm, const void * vy,
                              int ncols, dpct::queue_ptr stream);

namespace {

template <int NR, int UNROLL>
static inline void q4_0_xe1_rows(const uint8_t * __restrict__ vx, const int8_t * __restrict__ vy, float * __restrict__ dst,
                                 const int ncols, const int nrows, const int row0, const sycl::sub_group & sg) {
    const int lane = sg.get_local_linear_id();

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
static void q4_0_xe1_kernel(const uint8_t * __restrict__ vx, const int8_t * __restrict__ vy, float * __restrict__ dst,
                            const int ncols, const int nrows, const sycl::nd_item<1> & it) {
    const auto sg   = it.get_sub_group();
    const int  row0 = (it.get_group(0) * sg.get_group_linear_range() + sg.get_group_linear_id()) * NR;
    if (row0 >= nrows) {
        return;
    }
    q4_0_xe1_rows<NR, UNROLL>(vx, vy, dst, ncols, nrows, row0, sg);
}

// gate + up + SWIGLU: one sub-group per output row computes the gate row and the up row from the same activation loads
// and writes silu(g) * u (no gate / up intermediates, no separate GLU kernel)
static void q4_0_xe1_swiglu_kernel(const uint8_t * __restrict__ wg, const uint8_t * __restrict__ wu,
                                   const int8_t * __restrict__ vy, float * __restrict__ dst, const int ncols,
                                   const int nrows, const sycl::nd_item<1> & it) {
    const auto sg   = it.get_sub_group();
    const int  lane = sg.get_local_linear_id();
    const int  row  = it.get_group(0) * sg.get_group_linear_range() + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }
    const int           nblk   = ncols / QK4_0;
    const uint32_t      dofs   = (uint32_t) (ncols / 2) * (uint32_t) nrows;
    const sycl::half *  dg     = (const sycl::half *) (wg + dofs);
    const sycl::half *  du     = (const sycl::half *) (wu + dofs);
    const sycl::half2 * ys     = (const sycl::half2 *) (vy + ncols);
    float               acc[2] = { 0.0f, 0.0f };
    for (int b = lane; b < nblk; b += WARP_SIZE) {
        const sycl::int4 *  ya  = (const sycl::int4 *) (vy + (uint32_t) b * QK8_1);
        const sycl::int4    alo = ya[0];
        const sycl::int4    ahi = ya[1];
        const sycl::float2  dsf = ys[b].template convert<float, sycl::rounding_mode::automatic>();
        const uint32_t      ib  = (uint32_t) row * (uint32_t) nblk + (uint32_t) b;
        const sycl::uint4   q[2] = { *(const sycl::uint4 *) (wg + ib * 16u), *(const sycl::uint4 *) (wu + ib * 16u) };
        const float         d[2] = { (float) dg[ib], (float) du[ib] };
        const int al[4] = { alo.x(), alo.y(), alo.z(), alo.w() };
        const int ah[4] = { ahi.x(), ahi.y(), ahi.z(), ahi.w() };
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t qq[4] = { q[r].x(), q[r].y(), q[r].z(), q[r].w() };
            int            sumi  = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                sumi = dpct::dp4a((int) (qq[k] & 0x0F0F0F0Fu), al[k], sumi);
                sumi = dpct::dp4a((int) ((qq[k] >> 4) & 0x0F0F0F0Fu), ah[k], sumi);
            }
            acc[r] += d[r] * (sumi * dsf.x() - 8.0f * dsf.y());
        }
    }
    const float g = sycl::reduce_over_group(sg, acc[0], sycl::plus<float>());
    const float u = sycl::reduce_over_group(sg, acc[1], sycl::plus<float>());
    if (lane == 0) {
        dst[row] = g / (1.0f + sycl::native::exp(-g)) * u;
    }
}

// K-split: KS sub-groups of one work-group share a row, each taking every KS-th group of 16 blocks; partial sums meet in
// local memory. More sub-groups in flight for the short (4096-row) matrices, where one row per sub-group is one wave.
static inline float q4_0_xe1_partial(const uint8_t * __restrict__ vx, const int8_t * __restrict__ vy, const int ncols,
                                     const int nrows, const int row, const int b_first, const int b_step) {
    const int           nblk   = ncols / QK4_0;
    const sycl::half *  d_base = (const sycl::half *) (vx + (uint32_t) (ncols / 2) * (uint32_t) nrows);
    const sycl::half2 * ys     = (const sycl::half2 *) (vy + ncols);
    float               acc    = 0.0f;
    for (int b = b_first; b < nblk; b += b_step) {
        const sycl::int4 *  ya  = (const sycl::int4 *) (vy + (uint32_t) b * QK8_1);
        const sycl::int4    alo = ya[0];
        const sycl::int4    ahi = ya[1];
        const sycl::float2  dsf = ys[b].template convert<float, sycl::rounding_mode::automatic>();
        const uint32_t      ib  = (uint32_t) row * (uint32_t) nblk + (uint32_t) b;
        const sycl::uint4   q   = *(const sycl::uint4 *) (vx + ib * 16u);
        const float         dw  = (float) d_base[ib];
        const uint32_t qq[4] = { q.x(), q.y(), q.z(), q.w() };
        const int      al[4] = { alo.x(), alo.y(), alo.z(), alo.w() };
        const int      ah[4] = { ahi.x(), ahi.y(), ahi.z(), ahi.w() };
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sumi = dpct::dp4a((int) (qq[k] & 0x0F0F0F0Fu), al[k], sumi);
            sumi = dpct::dp4a((int) ((qq[k] >> 4) & 0x0F0F0F0Fu), ah[k], sumi);
        }
        acc += dw * (sumi * dsf.x() - 8.0f * dsf.y());
    }
    return acc;
}

template <int KS>
static inline void q4_0_xe1_ks_row(const uint8_t * vx, const int8_t * vy, float * dst, const int ncols, const int nrows,
                                   const int row, const sycl::nd_item<1> & it, float * sh) {
    const auto sg   = it.get_sub_group();
    const int  sgid = sg.get_group_linear_id();
    const int  ks   = sgid % KS;
    const int  lane = sg.get_local_linear_id();
    float      acc  = 0.0f;
    if (row < nrows) {
        acc = q4_0_xe1_partial(vx, vy, ncols, nrows, row, lane + ks * WARP_SIZE, WARP_SIZE * KS);
    }
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) {
        sh[sgid] = acc;
    }
    sycl::group_barrier(it.get_group());
    if (ks == 0 && lane == 0 && row < nrows) {
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < KS; ++j) {
            t += sh[sgid + j];
        }
        dst[row] = t;
    }
}

// Cooperative rows: a work-group of 8 sub-groups owns RPW consecutive rows; sub-group g takes the 256-byte chunks
// (16 blocks) g, g + 8, ... of every one of those rows, row by row. At any time the 8 sub-groups read 8 adjacent chunks
// of the same row (2 KB contiguous) instead of 8 rows 2.3 KB apart, which is what the A770's DRAM wants (read probe,
// 1 GiB of k=4096 rows: 218-288 GB/s one-row-per-sub-group vs 421 GB/s this way), and each sub-group loads its
// activation chunk once for all RPW rows. Partial sums meet in local memory.
template <int RPW>
static inline void q4_0_xe1_coop(const uint8_t * __restrict__ vx, const int8_t * __restrict__ vy, float * __restrict__ dst,
                                 const int ncols, const int nrows, const int row0, const sycl::nd_item<1> & it,
                                 float * sh) {
    constexpr int SGS  = 8;
    const auto    sg   = it.get_sub_group();
    const int     lane = sg.get_local_linear_id();
    const int     sgid = sg.get_group_linear_id();
    const int     nblk = ncols / QK4_0;
    const sycl::half *  d_base = (const sycl::half *) (vx + (uint32_t) (ncols / 2) * (uint32_t) nrows);
    const sycl::half2 * ys     = (const sycl::half2 *) (vy + ncols);
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        acc[r] = 0.0f;
    }
    for (int b = sgid * WARP_SIZE + lane; b - lane < nblk; b += SGS * WARP_SIZE) {
        if (b < nblk) {
            const sycl::int4 * ya  = (const sycl::int4 *) (vy + (uint32_t) b * QK8_1);
            const sycl::int4   alo = ya[0];
            const sycl::int4   ahi = ya[1];
            const sycl::float2 dsf = ys[b].template convert<float, sycl::rounding_mode::automatic>();
            const int al[4] = { alo.x(), alo.y(), alo.z(), alo.w() };
            const int ah[4] = { ahi.x(), ahi.y(), ahi.z(), ahi.w() };
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const uint32_t    ib = (uint32_t) (row0 + r) * (uint32_t) nblk + (uint32_t) b;
                const sycl::uint4 q  = *(const sycl::uint4 *) (vx + ib * 16u);
                const float       dw = (float) d_base[ib];
                const uint32_t qq[4] = { q.x(), q.y(), q.z(), q.w() };
                int sumi = 0;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    sumi = dpct::dp4a((int) (qq[kk] & 0x0F0F0F0Fu), al[kk], sumi);
                    sumi = dpct::dp4a((int) ((qq[kk] >> 4) & 0x0F0F0F0Fu), ah[kk], sumi);
                }
                acc[r] += dw * (sumi * dsf.x() - 8.0f * dsf.y());
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const float t = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
        if (lane == 0) {
            sh[r * SGS + sgid] = t;
        }
    }
    sycl::group_barrier(it.get_group());
    const int tid = it.get_local_id(0);
    if (tid < RPW) {
        float t = 0.0f;
#pragma unroll
        for (int g = 0; g < SGS; ++g) {
            t += sh[tid * SGS + g];
        }
        dst[row0 + tid] = t;
    }
}

// up to 3 matrices with the same input (q + k + v, gate + up) in one launch: one sub-group per row of the concatenation
struct q4_0_multi_args {
    const uint8_t * vx[3];
    float *         dst[3];
    int             nrows[3];
};

static void q4_0_xe1_multi_kernel(const q4_0_multi_args a, const int8_t * __restrict__ vy, const int ncols,
                                  const sycl::nd_item<1> & it) {
    const auto sg = it.get_sub_group();
    int        g  = it.get_group(0) * sg.get_group_linear_range() + sg.get_group_linear_id();
    int        m  = 0;
    if (g >= a.nrows[0]) {
        g -= a.nrows[0];
        m = 1;
        if (g >= a.nrows[1]) {
            g -= a.nrows[1];
            m = 2;
            if (g >= a.nrows[2]) {
                return;
            }
        }
    }
    const uint8_t * vx    = m == 0 ? a.vx[0] : (m == 1 ? a.vx[1] : a.vx[2]);
    float *         dst   = m == 0 ? a.dst[0] : (m == 1 ? a.dst[1] : a.dst[2]);
    const int       nrows = m == 0 ? a.nrows[0] : (m == 1 ? a.nrows[1] : a.nrows[2]);
    q4_0_xe1_rows<1, 1>(vx, vy, dst, ncols, nrows, g, sg);
}

static int q4_0_xe1_sgs() {
    static const int v = getenv("GGML_SYCL_Q4_0_XE1_SGS") ? atoi(getenv("GGML_SYCL_Q4_0_XE1_SGS")) : 8;
    return v;
}

template <int NR, int UNROLL>
static void launch(const void * vx, const void * vy, float * dst, int ncols, int nrows, dpct::queue_ptr stream) {
    const int     SGS  = q4_0_xe1_sgs();  // subgroups per workgroup
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
    {
        const void * vxa[1] = { vx };
        float *      da[1]  = { dst };
        const int    na[1]  = { nrows };
        if (q4_0_xe1_try_coop(vxa, da, na, 1, vy, ncols, stream)) {
            return;
        }
    }
    const int ks = q4_0_xe1_ks();
    if (ks == 2 || ks == 4 || ks == 8) {
        ks == 2 ? launch_ks<2>(vx, vy, dst, ncols, nrows, stream) :
                  (ks == 4 ? launch_ks<4>(vx, vy, dst, ncols, nrows, stream) : launch_ks<8>(vx, vy, dst, ncols, nrows, stream));
        return;
    }
    if (nr == 1) {
        un == 1 ? launch<1, 1>(vx, vy, dst, ncols, nrows, stream) : launch<1, 2>(vx, vy, dst, ncols, nrows, stream);
    } else if (nr == 4) {
        un == 1 ? launch<4, 1>(vx, vy, dst, ncols, nrows, stream) : launch<4, 2>(vx, vy, dst, ncols, nrows, stream);
    } else {
        un == 1 ? launch<2, 1>(vx, vy, dst, ncols, nrows, stream) : launch<2, 2>(vx, vy, dst, ncols, nrows, stream);
    }
}

bool ggml_sycl_q4_0_xe1_mul_mat_vec_multi(const void * const * vx, float * const * dst, const int * nrows, int nm,
                                          const void * vy, int ncols, dpct::queue_ptr stream) {
    if (nm < 1 || nm > 3 || ncols % QK4_0 != 0) {
        return false;
    }
    if (q4_0_xe1_try_coop(vx, dst, nrows, nm, vy, ncols, stream)) {
        return true;
    }
    q4_0_multi_args a{};
    int             total = 0;
    for (int m = 0; m < 3; ++m) {
        a.vx[m]    = (const uint8_t *) (m < nm ? vx[m] : vx[0]);
        a.dst[m]   = m < nm ? dst[m] : dst[0];
        a.nrows[m] = m < nm ? nrows[m] : 0;
        total += a.nrows[m];
    }
    const int     SGS = q4_0_xe1_sgs();
    const int     nwg = (total + SGS - 1) / SGS;
    const int8_t * y  = (const int8_t *) vy;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGS * WARP_SIZE), sycl::range<1>(SGS * WARP_SIZE)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             q4_0_xe1_multi_kernel(a, y, ncols, it);
                         });
    });
    return true;
}

void ggml_sycl_q4_0_xe1_swiglu(const void * wg, const void * wu, const void * vy, float * dst, int ncols, int nrows,
                               dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr int SGS = 8;
    const int     nwg = (nrows + SGS - 1) / SGS;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGS * WARP_SIZE), sycl::range<1>(SGS * WARP_SIZE)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             q4_0_xe1_swiglu_kernel((const uint8_t *) wg, (const uint8_t *) wu, (const int8_t *) vy, dst,
                                                    ncols, nrows, it);
                         });
    });
}

namespace {
static int q4_0_xe1_ks() {
    static const int v = getenv("GGML_SYCL_Q4_0_XE1_KS") ? atoi(getenv("GGML_SYCL_Q4_0_XE1_KS")) : 1;
    return v;
}

template <int KS>
static void launch_ks(const void * vx, const void * vy, float * dst, int ncols, int nrows, dpct::queue_ptr stream) {
    constexpr int SGS = 8;               // sub-groups per work-group, SGS / KS rows
    constexpr int RPW = SGS / KS;
    const int     nwg = (nrows + RPW - 1) / RPW;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(SGS), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGS * WARP_SIZE), sycl::range<1>(SGS * WARP_SIZE)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             const int row = it.get_group(0) * RPW + it.get_sub_group().get_group_linear_id() / KS;
                             q4_0_xe1_ks_row<KS>((const uint8_t *) vx, (const int8_t *) vy, dst, ncols, nrows, row, it,
                                                 sh.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}
}  // namespace

namespace {
constexpr int COOP_RPW = 8;

static bool q4_0_xe1_coop_on() {
    // off by default: in-model 87.6 vs 91.2 t/s (isolated k=14336: 88 vs 86 us), despite the read probe
    static const bool v = getenv("GGML_SYCL_Q4_0_XE1_COOP") && atoi(getenv("GGML_SYCL_Q4_0_XE1_COOP")) != 0;
    return v;
}

static void launch_coop(const q4_0_multi_args a, int nm, const void * vy, int ncols, dpct::queue_ptr stream) {
    int wg0[4] = { 0, 0, 0, 0 };  // first work-group of each matrix
    for (int m = 0; m < nm; ++m) {
        wg0[m + 1] = wg0[m] + a.nrows[m] / COOP_RPW;
    }
    const int8_t * y = (const int8_t *) vy;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(COOP_RPW * 8), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) wg0[nm] * 8 * WARP_SIZE), sycl::range<1>(8 * WARP_SIZE)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             const int g = it.get_group(0);
                             const int m = g >= wg0[1] ? (g >= wg0[2] ? 2 : 1) : 0;
                             const uint8_t * vx    = m == 0 ? a.vx[0] : (m == 1 ? a.vx[1] : a.vx[2]);
                             float *         dst   = m == 0 ? a.dst[0] : (m == 1 ? a.dst[1] : a.dst[2]);
                             const int       nrows = m == 0 ? a.nrows[0] : (m == 1 ? a.nrows[1] : a.nrows[2]);
                             const int       row0  = (g - (m == 0 ? 0 : (m == 1 ? wg0[1] : wg0[2]))) * COOP_RPW;
                             q4_0_xe1_coop<COOP_RPW>(vx, y, dst, ncols, nrows, row0, it,
                                                     sh.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}
}  // namespace

// cooperative path when every matrix's rows are a multiple of the rows per work-group
static bool q4_0_xe1_try_coop(const void * const * vx, float * const * dst, const int * nrows, int nm, const void * vy,
                              int ncols, dpct::queue_ptr stream) {
    if (!q4_0_xe1_coop_on() || nm < 1 || nm > 3) {
        return false;
    }
    q4_0_multi_args a{};
    for (int m = 0; m < 3; ++m) {
        if (m < nm && nrows[m] % COOP_RPW != 0) {
            return false;
        }
        a.vx[m]    = (const uint8_t *) (m < nm ? vx[m] : vx[0]);
        a.dst[m]   = m < nm ? dst[m] : dst[0];
        a.nrows[m] = m < nm ? nrows[m] : 0;
    }
    launch_coop(a, nm, vy, ncols, stream);
    return true;
}

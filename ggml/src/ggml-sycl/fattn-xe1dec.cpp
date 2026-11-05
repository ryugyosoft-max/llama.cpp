// [a770-opt] Flash-decoding for one query token with grouped-query attention and an f16 KV cache,
// tuned for Arc Alchemist (Xe-HPG, 8-lane subgroups). Port of the Vulkan fa_decode_xe1 kernel.
//
// One workgroup = one KV head x one contiguous chunk of the KV sequence ("split").
// Subgroups are indexed (dsub, hsub, rsub):
//   hsub: query head kv_head * GQA + hsub; all heads of a workgroup read the same K/V rows, so only
//         the first read of a row reaches memory
//   dsub: which 1/DS of the head dims the subgroup owns; partial scores are summed through local
//         memory once per step (small per-lane state leaves registers for loads in flight)
//   rsub: steps of the split are interleaved over R subgroup rows; each row writes its own partial
// A lane owns DPL consecutive dims. Per step T = 8 tokens: partial dots for 8 tokens, a butterfly
// reduce-scatter leaves lane t with token t's score, one subgroup max for the online softmax, and
// the V accumulation broadcasts each token's probability from its lane.
// A second kernel merges the partials (O, L, M) of all splits.

#include "fattn-xe1dec.hpp"

#include <cmath>
#include <cstring>

namespace {

constexpr int SG = 8;  // subgroup size (DPAS-native SIMD8 on Alchemist)
constexpr int T  = SG; // tokens per step

template <int D> constexpr int xe1_ds() { return D == 256 ? 4 : 2; }
template <int D, int GQA> constexpr int xe1_r() { return 16 / (GQA * xe1_ds<D>()) > 0 ? 16 / (GQA * xe1_ds<D>()) : 1; }

struct xe1dec_params {
    const char * q; const char * k; const char * v; const sycl::half * m;
    float * part;           // [k_num * ne3][ne1][D] then L/M: [k_num * ne3][2][ne1]
    int64_t nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3;
    int64_t m_batch;        // mask batch stride in halfs
    int     nem3;
    int     KV, split_kv, k_num, ne1;
    float   scale;
};

template <int D, int GQA>
static void xe1dec_kernel(const xe1dec_params p, const sycl::nd_item<3> & it, float * sh_s) {
    constexpr int DS  = xe1_ds<D>();
    constexpr int R   = xe1_r<D, GQA>();
    constexpr int DPL = D / (SG * DS);  // dims per lane
    static_assert(DPL % 8 == 0, "a lane loads whole 16-byte groups");
    constexpr int NV  = DPL / 8;        // 16-byte loads per lane per row

    const auto sg    = it.get_sub_group();
    const int  lane  = sg.get_local_linear_id();
    const int  sgid  = sg.get_group_linear_id();
    const int  dsub  = sgid % DS;
    const int  hsub  = (sgid / DS) % GQA;
    const int  rsub  = sgid / (DS * GQA);
    const int  split = it.get_group(2);
    const int  kvh   = it.get_group(1);
    const int  iq3   = it.get_group(0);
    const int  head  = kvh * GQA + hsub;
    const int  dim0  = dsub * (D / DS) + lane * DPL;

    sycl::half2 q[DPL / 2];
    {
        const float * qp = (const float *) (p.q + iq3 * p.nbq3 + head * p.nbq2) + dim0;
#pragma unroll
        for (int i = 0; i < DPL / 2; ++i) {
            q[i] = sycl::half2((sycl::half) (qp[2 * i] * p.scale), (sycl::half) (qp[2 * i + 1] * p.scale));
        }
    }
    float o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
        o[i] = 0.0f;
    }

    const float NEG_HALF_FLT_MAX = -3.402823466e+38f * 0.5f;
    float m_run  = NEG_HALF_FLT_MAX;
    float l_lane = 0.0f;

    const int start   = split * p.split_kv;
    const int end     = sycl::min(p.KV, start + p.split_kv);
    const int n_steps = (end - start + T * R - 1) / (T * R);  // uniform over the workgroup

    const char * kb = p.k + iq3 * p.nbk3 + kvh * p.nbk2 + dim0 * 2;
    const char * vb = p.v + iq3 * p.nbv3 + kvh * p.nbv2 + dim0 * 2;
    const sycl::half * mb = p.m ? p.m + (iq3 % p.nem3) * p.m_batch : nullptr;
    const int sh_row = (rsub * GQA + hsub) * DS;

    for (int st = 0; st < n_steps; ++st) {
        const int j = start + (st * R + rsub) * T;

        float s[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int tok = sycl::min(j + t, end - 1);
            const sycl::uint4 * kr = (const sycl::uint4 *) (kb + (int64_t) tok * p.nbk1);
            sycl::half2 a = sycl::half2(0.0f, 0.0f);
#pragma unroll
            for (int i = 0; i < NV; ++i) {
                sycl::half2 kh[4];
                *(sycl::uint4 *) kh = kr[i];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    a = sycl::fma(kh[c], q[4 * i + c], a);
                }
            }
            s[t] = (float) a.x() + (float) a.y();
        }

        // butterfly reduce-scatter: lane t ends with the (partial) score of token t
#pragma unroll
        for (int off = T / 2; off > 0; off >>= 1) {
            const bool upper = (lane & off) != 0;
#pragma unroll
            for (int i = 0; i < off; ++i) {
                const float send = upper ? s[i] : s[i + off];
                const float keep = upper ? s[i + off] : s[i];
                s[i] = keep + sycl::permute_group_by_xor(sg, send, off);
            }
        }

        float score = s[0];
        if constexpr (DS > 1) {
            const int buf = (st & 1) * (R * GQA * DS * T);
            sh_s[buf + (sh_row + dsub) * T + lane] = score;
            sycl::group_barrier(it.get_group());
            score = 0.0f;
#pragma unroll
            for (int d2 = 0; d2 < DS; ++d2) {
                score += sh_s[buf + (sh_row + d2) * T + lane];
            }
        }

        const int my_tok = j + lane;
        if (my_tok >= end) {
            score = -INFINITY;
        } else if (mb) {
            score += (float) mb[my_tok];
        }

        const float m_new = sycl::max(m_run, sycl::reduce_over_group(sg, score, sycl::maximum<float>()));
        const float alpha = sycl::exp(m_run - m_new);
        const float pr    = sycl::exp(score - m_new);
        m_run  = m_new;
        l_lane = l_lane * alpha + pr;

#pragma unroll
        for (int i = 0; i < DPL; ++i) {
            o[i] *= alpha;
        }

#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float pt  = sycl::select_from_group(sg, pr, t);
            const int   tok = sycl::min(j + t, end - 1);
            const sycl::uint4 * vr = (const sycl::uint4 *) (vb + (int64_t) tok * p.nbv1);
#pragma unroll
            for (int i = 0; i < NV; ++i) {
                sycl::half2 h[4];
                *(sycl::uint4 *) h = vr[i];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    o[8 * i + 2 * c + 0] += pt * (float) h[c].x();
                    o[8 * i + 2 * c + 1] += pt * (float) h[c].y();
                }
            }
        }
    }

    const float l_sum = sycl::reduce_over_group(sg, l_lane, sycl::plus<float>());

    const int osplit = split * R + rsub;
    float * op = p.part + ((int64_t) (osplit + p.k_num * iq3) * p.ne1 + head) * D + dim0;
#pragma unroll
    for (int i = 0; i < DPL; i += 4) {
        *(sycl::float4 *) (op + i) = sycl::float4(o[i], o[i + 1], o[i + 2], o[i + 3]);
    }
    if (lane == 0 && dsub == 0) {
        const int64_t n_o  = (int64_t) D * p.ne1 * p.k_num * it.get_group_range(0);
        float *       lm   = p.part + n_o + (int64_t) (osplit + p.k_num * iq3) * 2 * p.ne1;
        lm[head]           = l_sum;
        lm[p.ne1 + head]   = m_run;
    }
}

// merge: workgroup = one head x 64 dims (x batch); split weights computed once into local memory
static void xe1dec_reduce(const float * part, const float * sinks, float * dst, int D, int ne1, int ne2, int ne3, int k_num,
                          const sycl::nd_item<3> & it, float * sh_w) {
    const int n   = it.get_group(2);
    const int tid = it.get_local_id(2);
    const int i3  = it.get_group(0);
    const auto grp = it.get_group();

    const int64_t n_o       = (int64_t) D * ne1 * k_num * ne3;
    const float * lm        = part + n_o + (int64_t) k_num * i3 * 2 * ne1 + n;
    const int     lm_stride = 2 * ne1;

    float m_max = -INFINITY;
    for (int k = tid; k < k_num; k += 64) {
        m_max = sycl::max(m_max, lm[ne1 + k * lm_stride]);
    }
    m_max = sycl::reduce_over_group(grp, m_max, sycl::maximum<float>());

    float L = 0.0f;
    for (int k = tid; k < k_num; k += 64) {
        const float w = sycl::exp(lm[ne1 + k * lm_stride] - m_max);
        sh_w[k] = w;
        L += w * lm[k * lm_stride];
    }
    L = sycl::reduce_over_group(grp, L, sycl::plus<float>());

    float ms = 1.0f;
    if (sinks) {
        const float sink = sinks[n];
        float vs = 1.0f;
        if (sink > m_max) {
            ms = sycl::exp(m_max - sink);
        } else {
            vs = sycl::exp(sink - m_max);
        }
        L = L * ms + vs;
    }
    L = L == 0.0f ? 0.0f : 1.0f / L;

    const int d = it.get_group(1) * 64 + tid;
    if (d >= D) {
        return;
    }
    const int64_t o_stride = (int64_t) D * ne1;
    const float * ob       = part + o_stride * k_num * i3 + (int64_t) D * n + d;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int k = 0;
    for (; k + 8 <= k_num; k += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[u] += sh_w[k + u] * ob[(k + u) * o_stride];
        }
    }
    for (; k < k_num; ++k) {
        acc[0] += sh_w[k] * ob[k * o_stride];
    }
    float O = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    O *= ms * L;
    dst[(int64_t) i3 * ne2 * ne1 * D + (int64_t) D * n + d] = sycl::clamp(O, -3.402823466e+38f, 3.402823466e+38f);
}

template <int D, int GQA>
static void launch(const xe1dec_params & p, int nkvh, int ne3, int wgs, dpct::queue_ptr stream) {
    constexpr int DS = xe1_ds<D>();
    constexpr int R  = xe1_r<D, GQA>();
    constexpr int WG = SG * GQA * DS * R;
    constexpr int SH = DS > 1 ? 2 * R * GQA * DS * T : 1;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(SH), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, nkvh, (size_t) wgs * WG), sycl::range<3>(1, 1, WG)),
                         [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
                             xe1dec_kernel<D, GQA>(p, it, sh.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

template <int D>
static void launch_gqa(int gqa, const xe1dec_params & p, int nkvh, int ne3, int wgs, dpct::queue_ptr stream) {
    switch (gqa) {
        case 2: launch<D, 2>(p, nkvh, ne3, wgs, stream); break;
        case 3: launch<D, 3>(p, nkvh, ne3, wgs, stream); break;
        case 4: launch<D, 4>(p, nkvh, ne3, wgs, stream); break;
        case 5: launch<D, 5>(p, nkvh, ne3, wgs, stream); break;
        case 6: launch<D, 6>(p, nkvh, ne3, wgs, stream); break;
        case 7: launch<D, 7>(p, nkvh, ne3, wgs, stream); break;
        case 8: launch<D, 8>(p, nkvh, ne3, wgs, stream); break;
        default: GGML_ABORT("xe1dec: unsupported GQA %d", gqa);
    }
}

static int xe1dec_r(int D, int gqa) {
    const int ds = D == 256 ? 4 : 2;
    return std::max(1, 16 / (gqa * ds));
}

}  // namespace

bool ggml_sycl_device_is_alchemist(int device) {
    const auto arch = ggml_sycl_info().devices[device].hw_info.arch;
    return arch == gpu_arch::intel_gpu_acm_g10 || arch == gpu_arch::intel_gpu_acm_g11 || arch == gpu_arch::intel_gpu_acm_g12 ||
           arch == gpu_arch::intel_gpu_dg2_g10 || arch == gpu_arch::intel_gpu_dg2_g11 || arch == gpu_arch::intel_gpu_dg2_g12;
}

bool ggml_sycl_flash_attn_ext_xe1dec_supported(int device, const ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_SYCL_XE1DEC_OFF") != nullptr;
    // the generic kernels lose even at a 256-token cache (180 vs ~90 us), so no minimum by default
    static const int  min_kv   = getenv("GGML_SYCL_XE1DEC_MIN_KV") ? atoi(getenv("GGML_SYCL_XE1DEC_MIN_KV")) : 0;
    if (disabled) {
        return false;
    }
    if (!ggml_sycl_device_is_alchemist(device)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int64_t D = K->ne[0];
    if (Q->ne[1] != 1 || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32 || V->ne[0] != D || (D != 128 && D != 256) || K->ne[1] < min_kv ||
        max_bias != 0.0f || softcap != 0.0f || (sinks && sinks->type != GGML_TYPE_F32)) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t gqa = Q->ne[2] / K->ne[2];
    if (gqa < 2 || gqa > 8 || K->ne[2] != V->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[2] != 1)) {
        return false;
    }
    auto al16 = [](size_t x) { return x % 16 == 0; };
    return al16(Q->nb[2]) && al16(Q->nb[3]) && al16(K->nb[1]) && al16(K->nb[2]) && al16(K->nb[3]) && al16(V->nb[1]) &&
           al16(V->nb[2]) && al16(V->nb[3]) && ggml_is_contiguous(dst);
}

void ggml_sycl_flash_attn_ext_xe1dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    dpct::queue_ptr stream = ctx.stream();

    const int D    = (int) K->ne[0];
    const int KV   = (int) K->ne[1];
    const int nkvh = (int) K->ne[2];
    const int ne1  = (int) Q->ne[2];  // query heads
    const int ne3  = (int) Q->ne[3];
    const int gqa  = ne1 / nkvh;
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    static const int target_wg = getenv("GGML_SYCL_XE1DEC_WG") ? atoi(getenv("GGML_SYCL_XE1DEC_WG")) : 256;
    const int R       = xe1dec_r(D, gqa);
    const int want    = std::max(2, (target_wg + nkvh * ne3 - 1) / (nkvh * ne3));
    const int nsplit  = std::min(512 / R, want);
    int split_kv      = (KV + nsplit - 1) / nsplit;
    split_kv          = (split_kv + 15) / 16 * 16;
    const int wgs     = (KV + split_kv - 1) / split_kv;
    const int k_num   = wgs * R;

    ggml_sycl_pool_alloc<float> part(ctx.pool(), ((size_t) D * ne1 + 2 * ne1) * k_num * ne3);

    xe1dec_params p;
    p.q = (const char *) Q->data; p.k = (const char *) K->data; p.v = (const char *) V->data;
    p.m = mask ? (const sycl::half *) mask->data : nullptr;
    p.part = part.get();
    p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.m_batch = mask ? (int64_t) (mask->nb[3] / sizeof(sycl::half)) : 0;
    p.nem3 = mask ? (int) std::max<int64_t>(1, mask->ne[3]) : 1;
    p.KV = KV; p.split_kv = split_kv; p.k_num = k_num; p.ne1 = ne1; p.scale = scale;

    if (D == 256) {
        launch_gqa<256>(gqa, p, nkvh, ne3, wgs, stream);
    } else {
        launch_gqa<128>(gqa, p, nkvh, ne3, wgs, stream);
    }

    const float * pp = part.get();
    const float * sk = sinks ? (const float *) sinks->data : nullptr;
    float *       dd = (float *) dst->data;
    const int     ne2 = (int) Q->ne[1];  // = 1
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_w(sycl::range<1>(k_num), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, (D + 63) / 64, (size_t) ne1 * 64), sycl::range<3>(1, 1, 64)),
                         [=](sycl::nd_item<3> it) {
                             xe1dec_reduce(pp, sk, dd, D, ne1, ne2, ne3, k_num, it,
                                           sh_w.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

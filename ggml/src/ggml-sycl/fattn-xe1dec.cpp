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

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <cmath>
#include <cstring>

namespace {

constexpr int SG = 8;  // subgroup size (DPAS-native SIMD8 on Alchemist)
constexpr int T  = SG; // tokens per step

template <int DS, int GQA> constexpr int xe1_r() { return 16 / (GQA * DS) > 0 ? 16 / (GQA * DS) : 1; }

struct xe1dec_params {
    const char * q; const char * k; const char * v; const sycl::half * m;
    float * part;           // [k_num * ne3][ne1][D] then L/M: [k_num * ne3][2][ne1]
    int64_t nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3;
    int64_t m_batch;        // mask batch stride in halfs
    int     nem3;
    int     nq;             // query tokens (1 for decode, 2..4 to verify MTP drafts): group 0 = batch * nq + query
    int64_t nbq1;           // Q stride between query tokens (bytes)
    int64_t m_row;          // mask stride between query tokens (halfs)
    int     KV, split_kv, k_num, ne1;
    float   scale;
};

// One subgroup block read: lane l receives dwords l, l + 8, l + 16, ... of the N * 8 dwords at p
// (IGC turns plain 16-byte loads here into per-dword / per-halfword gathers).
template <int N>
static inline void xe1_block_load(const sycl::sub_group & sg, const char * p, uint32_t (&out)[N]) {
    namespace exp = sycl::ext::oneapi::experimental;
    auto gp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
        (const uint32_t *) p);
    exp::group_load(sg, gp, sycl::span<uint32_t, N>(out), exp::properties{ exp::data_placement_striped, exp::full_group });
}

template <int D, int GQA, int DS>
static void xe1dec_kernel(const xe1dec_params p, const sycl::nd_item<3> & it, float * sh_s) {
    constexpr int R   = xe1_r<DS, GQA>();
    constexpr int DPL = D / (SG * DS);  // dims per lane
    static_assert(DPL % 2 == 0, "a lane owns whole dwords");
    constexpr int NW  = DPL / 2;        // dwords (half2) per lane per row
    // striped layout: within a subgroup's D/DS-dim chunk, lane l owns dwords l + SG * c, c < NW

    const auto sg    = it.get_sub_group();
    const int  lane  = sg.get_local_linear_id();
    const int  sgid  = sg.get_group_linear_id();
    const int  dsub  = sgid % DS;
    const int  hsub  = (sgid / DS) % GQA;
    const int  rsub  = sgid / (DS * GQA);
    const int  split = it.get_group(2);
    const int  kvh   = it.get_group(1);
    const int  ibq   = it.get_group(0);     // batch * nq + query token
    const int  iq3   = ibq / p.nq;
    const int  iqt   = ibq % p.nq;
    const int  head  = kvh * GQA + hsub;
    const int  cdim0 = dsub * (D / DS);  // first dim of the subgroup's chunk

    sycl::half2 q[NW];
    {
        const float * qp = (const float *) (p.q + iq3 * p.nbq3 + iqt * p.nbq1 + head * p.nbq2) + cdim0;
#pragma unroll
        for (int c = 0; c < NW; ++c) {
            const int d = 2 * (lane + SG * c);
            q[c] = sycl::half2((sycl::half) (qp[d] * p.scale), (sycl::half) (qp[d + 1] * p.scale));
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

    const char * kb = p.k + iq3 * p.nbk3 + kvh * p.nbk2 + cdim0 * 2;
    const char * vb = p.v + iq3 * p.nbv3 + kvh * p.nbv2 + cdim0 * 2;
    const sycl::half * mb = p.m ? p.m + (iq3 % p.nem3) * p.m_batch + iqt * p.m_row : nullptr;
    const int sh_row = (rsub * GQA + hsub) * DS;
    // 32-bit row offsets: Alchemist has no native 64-bit integer multiply
    const uint32_t nbk1 = (uint32_t) p.nbk1, nbv1 = (uint32_t) p.nbv1;

    for (int st = 0; st < n_steps; ++st) {
        const int j = start + (st * R + rsub) * T;

        float s[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int tok = sycl::min(j + t, end - 1);
            uint32_t kw[NW];
            xe1_block_load<NW>(sg, kb + (uint32_t) tok * nbk1, kw);
            sycl::half2 a = sycl::half2(0.0f, 0.0f);
#pragma unroll
            for (int c = 0; c < NW; ++c) {
                a = sycl::fma(sycl::bit_cast<sycl::half2>(kw[c]), q[c], a);
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
        const float alpha = sycl::native::exp(m_run - m_new);
        const float pr    = sycl::native::exp(score - m_new);
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
            uint32_t vw[NW];
            xe1_block_load<NW>(sg, vb + (uint32_t) tok * nbv1, vw);
#pragma unroll
            for (int c = 0; c < NW; ++c) {
                const sycl::half2 h = sycl::bit_cast<sycl::half2>(vw[c]);
                o[2 * c + 0] += pt * (float) h.x();
                o[2 * c + 1] += pt * (float) h.y();
            }
        }
    }

    const float l_sum = sycl::reduce_over_group(sg, l_lane, sycl::plus<float>());

    const int osplit = split * R + rsub;
    float * op = p.part + ((int64_t) (osplit + p.k_num * ibq) * p.ne1 + head) * D + cdim0;
#pragma unroll
    for (int c = 0; c < NW; ++c) {
        *(sycl::float2 *) (op + 2 * (lane + SG * c)) = sycl::float2(o[2 * c], o[2 * c + 1]);
    }
    if (lane == 0 && dsub == 0) {
        const int64_t n_o  = (int64_t) D * p.ne1 * p.k_num * it.get_group_range(0);
        float *       lm   = p.part + n_o + (int64_t) (osplit + p.k_num * ibq) * 2 * p.ne1;
        lm[head]           = l_sum;
        lm[p.ne1 + head]   = m_run;
    }
}

// Quantized KV (q8_0: 34-byte blocks, q4_0: 18-byte blocks of 32 values, fp16 scale first).
// A subgroup's D/DS = 64 dims are 2 blocks = 68 (q8_0) / 36 (q4_0) bytes, starting 4-byte aligned. They are read with
// subgroup block reads (lane l gets dwords l, l + 8) plus one uniform dword for the tail, and each lane owns the values
// that land in its own dwords ("slots"), so no data moves between lanes except the two block scales.
// q8_0: slot = 1 byte = 1 value; q4_0: slot = 1 byte = 2 values (low nibble j, high nibble j + 16).
template <int KVT> struct xe1q {
    static constexpr int BB    = KVT == GGML_TYPE_Q8_0 ? 34 : 18;  // block bytes
    static constexpr int NBLK  = KVT == GGML_TYPE_Q8_0 ? 2 : 1;    // block-read dwords per lane (8 * NBLK dwords)
    static constexpr int NSLOT = 4 * NBLK + 4;                      // bytes a lane may own (tail dword: lane 0 only)
    static constexpr int VPS   = KVT == GGML_TYPE_Q8_0 ? 1 : 2;    // values per slot
    static constexpr int TAIL  = 32 * NBLK;                         // byte offset of the tail dword
};

// byte offset (within the 2-block chunk) of slot s of lane l
template <int KVT> static inline int xe1q_byte(int lane, int s) {
    constexpr int NB = xe1q<KVT>::NBLK;
    return s < 4 * NB ? 32 * (s / 4) + 4 * lane + (s % 4) : xe1q<KVT>::TAIL + (s - 4 * NB);
}

// chunk dim of value v (0..VPS-1) of slot s, or -1 when the byte is a scale / not this lane's
template <int KVT> static inline int xe1q_dim(int lane, int s, int v) {
    if (s >= 4 * xe1q<KVT>::NBLK && lane != 0) {
        return -1;
    }
    const int k   = xe1q_byte<KVT>(lane, s);
    const int blk = k >= xe1q<KVT>::BB ? 1 : 0;
    const int kk  = k - blk * xe1q<KVT>::BB;
    if (kk < 2) {
        return -1;
    }
    return blk * 32 + (kk - 2) + v * 16;
}

template <int KVT> static inline int xe1q_blk(int lane, int s) {
    return xe1q_byte<KVT>(lane, s) >= xe1q<KVT>::BB ? 1 : 0;
}

// read one 2-block chunk: lane dwords w[0..NBLK-1] by block read, tail dword uniform
template <int KVT>
static inline void xe1q_read(const sycl::sub_group & sg, const char * chunk, uint32_t (&w)[xe1q<KVT>::NBLK + 1]) {
    constexpr int NB = xe1q<KVT>::NBLK;
    uint32_t b[NB];
    xe1_block_load<NB>(sg, chunk, b);
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        w[i] = b[i];
    }
    w[NB] = *(const uint32_t *) (chunk + xe1q<KVT>::TAIL);
}

template <int KVT> static inline uint32_t xe1q_slot_byte(const uint32_t (&w)[xe1q<KVT>::NBLK + 1], int s) {
    constexpr int NB = xe1q<KVT>::NBLK;
    return (w[s < 4 * NB ? s / 4 : NB] >> (8 * (s % 4))) & 0xFF;
}

// the two block scales: block 0 at bytes 0..1 (lane 0, dword 0), block 1 at bytes BB..BB+1
template <int KVT>
static inline void xe1q_scales(const sycl::sub_group & sg, const uint32_t (&w)[xe1q<KVT>::NBLK + 1], float & d0, float & d1) {
    const uint32_t a = sycl::select_from_group(sg, w[0], 0);
    d0 = (float) sycl::bit_cast<sycl::half>((uint16_t) (a & 0xFFFF));
    if constexpr (KVT == GGML_TYPE_Q8_0) {  // bytes 34..35 = dword 8 high half = lane 0, w[1]
        const uint32_t b = sycl::select_from_group(sg, w[1], 0);
        d1 = (float) sycl::bit_cast<sycl::half>((uint16_t) (b >> 16));
    } else {                                // bytes 18..19 = dword 4 high half = lane 4, w[0]
        const uint32_t b = sycl::select_from_group(sg, w[0], 4);
        d1 = (float) sycl::bit_cast<sycl::half>((uint16_t) (b >> 16));
    }
}

template <int KVT> static inline float xe1q_val(uint32_t byte, int v) {
    if constexpr (KVT == GGML_TYPE_Q8_0) {
        return (float) (int8_t) byte;
    } else {
        return (float) ((int) (v ? byte >> 4 : byte & 0xF) - 8);
    }
}

template <int D, int GQA, int DS, int KVT>
static void xe1dec_kernel_q(const xe1dec_params p, const sycl::nd_item<3> & it, float * sh_s) {
    using QI = xe1q<KVT>;
    constexpr int R  = xe1_r<DS, GQA>();
    constexpr int NS = QI::NSLOT, VP = QI::VPS;
    static_assert(D / DS == 64, "a subgroup owns two quant blocks");

    const auto sg    = it.get_sub_group();
    const int  lane  = sg.get_local_linear_id();
    const int  sgid  = sg.get_group_linear_id();
    const int  dsub  = sgid % DS;
    const int  hsub  = (sgid / DS) % GQA;
    const int  rsub  = sgid / (DS * GQA);
    const int  split = it.get_group(2);
    const int  kvh   = it.get_group(1);
    const int  ibq   = it.get_group(0);     // batch * nq + query token
    const int  iq3   = ibq / p.nq;
    const int  iqt   = ibq % p.nq;
    const int  head  = kvh * GQA + hsub;
    const int  cdim0 = dsub * (D / DS);

    // per-slot query values (0 for scale bytes / foreign tail) and block of the slot
    float q[NS][VP];
    bool  b1[NS];
    {
        const float * qp = (const float *) (p.q + iq3 * p.nbq3 + iqt * p.nbq1 + head * p.nbq2) + cdim0;
#pragma unroll
        for (int sl = 0; sl < NS; ++sl) {
            b1[sl] = xe1q_blk<KVT>(lane, sl) != 0;
#pragma unroll
            for (int v = 0; v < VP; ++v) {
                const int d = xe1q_dim<KVT>(lane, sl, v);
                q[sl][v] = d >= 0 ? qp[d] * p.scale : 0.0f;
            }
        }
    }
    float o[NS][VP];
#pragma unroll
    for (int sl = 0; sl < NS; ++sl) {
#pragma unroll
        for (int v = 0; v < VP; ++v) {
            o[sl][v] = 0.0f;
        }
    }

    const float NEG_HALF_FLT_MAX = -3.402823466e+38f * 0.5f;
    float m_run  = NEG_HALF_FLT_MAX;
    float l_lane = 0.0f;

    const int start   = split * p.split_kv;
    const int end     = sycl::min(p.KV, start + p.split_kv);
    const int n_steps = (end - start + T * R - 1) / (T * R);

    const int    chunk0 = (cdim0 / 32) * QI::BB;
    const char * kb     = p.k + iq3 * p.nbk3 + kvh * p.nbk2 + chunk0;
    const char * vb     = p.v + iq3 * p.nbv3 + kvh * p.nbv2 + chunk0;
    const sycl::half * mb = p.m ? p.m + (iq3 % p.nem3) * p.m_batch + iqt * p.m_row : nullptr;
    const int sh_row = (rsub * GQA + hsub) * DS;
    const uint32_t nbk1 = (uint32_t) p.nbk1, nbv1 = (uint32_t) p.nbv1;

    for (int st = 0; st < n_steps; ++st) {
        const int j0 = start + (st * R + rsub) * T;

        float s[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int tok = sycl::min(j0 + t, end - 1);
            uint32_t w[QI::NBLK + 1];
            xe1q_read<KVT>(sg, kb + (uint32_t) tok * nbk1, w);
            float d0, d1;
            xe1q_scales<KVT>(sg, w, d0, d1);
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int sl = 0; sl < NS; ++sl) {
                const uint32_t by = xe1q_slot_byte<KVT>(w, sl);
                float          a  = 0.0f;
#pragma unroll
                for (int v = 0; v < VP; ++v) {
                    a = sycl::fma(q[sl][v], xe1q_val<KVT>(by, v), a);
                }
                if (b1[sl]) { a1 += a; } else { a0 += a; }
            }
            s[t] = a0 * d0 + a1 * d1;
        }

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

        const int my_tok = j0 + lane;
        if (my_tok >= end) {
            score = -INFINITY;
        } else if (mb) {
            score += (float) mb[my_tok];
        }

        const float m_new = sycl::max(m_run, sycl::reduce_over_group(sg, score, sycl::maximum<float>()));
        const float alpha = sycl::native::exp(m_run - m_new);
        const float pr    = sycl::native::exp(score - m_new);
        m_run  = m_new;
        l_lane = l_lane * alpha + pr;

#pragma unroll
        for (int sl = 0; sl < NS; ++sl) {
#pragma unroll
            for (int v = 0; v < VP; ++v) {
                o[sl][v] *= alpha;
            }
        }

#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float pt  = sycl::select_from_group(sg, pr, t);
            const int   tok = sycl::min(j0 + t, end - 1);
            uint32_t w[QI::NBLK + 1];
            xe1q_read<KVT>(sg, vb + (uint32_t) tok * nbv1, w);
            float d0, d1;
            xe1q_scales<KVT>(sg, w, d0, d1);
            const float p0 = pt * d0, p1 = pt * d1;
#pragma unroll
            for (int sl = 0; sl < NS; ++sl) {
                const uint32_t by = xe1q_slot_byte<KVT>(w, sl);
                const float    pb = b1[sl] ? p1 : p0;
#pragma unroll
                for (int v = 0; v < VP; ++v) {
                    o[sl][v] = sycl::fma(pb, xe1q_val<KVT>(by, v), o[sl][v]);
                }
            }
        }
    }

    const float l_sum = sycl::reduce_over_group(sg, l_lane, sycl::plus<float>());

    const int osplit = split * R + rsub;
    float * op = p.part + ((int64_t) (osplit + p.k_num * ibq) * p.ne1 + head) * D + cdim0;
#pragma unroll
    for (int sl = 0; sl < NS; ++sl) {
#pragma unroll
        for (int v = 0; v < VP; ++v) {
            const int d = xe1q_dim<KVT>(lane, sl, v);
            if (d >= 0) {
                op[d] = o[sl][v];
            }
        }
    }
    if (lane == 0 && dsub == 0) {
        const int64_t n_o  = (int64_t) D * p.ne1 * p.k_num * it.get_group_range(0);
        float *       lm   = p.part + n_o + (int64_t) (osplit + p.k_num * ibq) * 2 * p.ne1;
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

template <int D, int GQA, int DS>
static void launch(const xe1dec_params & p, int nkvh, int ne3, int wgs, dpct::queue_ptr stream) {
    constexpr int R  = xe1_r<DS, GQA>();
    constexpr int WG = SG * GQA * DS * R;
    constexpr int SH = DS > 1 ? 2 * R * GQA * DS * T : 1;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(SH), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, nkvh, (size_t) wgs * WG), sycl::range<3>(1, 1, WG)),
                         [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
                             xe1dec_kernel<D, GQA, DS>(p, it, sh.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

template <int D, int GQA, int DS, int KVT>
static void launch_q(const xe1dec_params & p, int nkvh, int ne3, int wgs, dpct::queue_ptr stream) {
    constexpr int R  = xe1_r<DS, GQA>();
    constexpr int WG = SG * GQA * DS * R;
    constexpr int SH = DS > 1 ? 2 * R * GQA * DS * T : 1;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(SH), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, nkvh, (size_t) wgs * WG), sycl::range<3>(1, 1, WG)),
                         [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
                             xe1dec_kernel_q<D, GQA, DS, KVT>(p, it, sh.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

template <int D, int DS, int KVT>
static void launch_gqa_q(int gqa, const xe1dec_params & p, int nkvh, int ne3, int wgs, dpct::queue_ptr stream) {
    switch (gqa) {
        case 2: launch_q<D, 2, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        case 3: launch_q<D, 3, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        case 4: launch_q<D, 4, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        case 5: launch_q<D, 5, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        case 6: launch_q<D, 6, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        case 7: launch_q<D, 7, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        case 8: launch_q<D, 8, DS, KVT>(p, nkvh, ne3, wgs, stream); break;
        default: GGML_ABORT("xe1dec: unsupported GQA %d", gqa);
    }
}

template <int D, int DS>
static void launch_gqa(int gqa, const xe1dec_params & p, int nkvh, int ne3, int wgs, dpct::queue_ptr stream) {
    switch (gqa) {
        case 2: launch<D, 2, DS>(p, nkvh, ne3, wgs, stream); break;
        case 3: launch<D, 3, DS>(p, nkvh, ne3, wgs, stream); break;
        case 4: launch<D, 4, DS>(p, nkvh, ne3, wgs, stream); break;
        case 5: launch<D, 5, DS>(p, nkvh, ne3, wgs, stream); break;
        case 6: launch<D, 6, DS>(p, nkvh, ne3, wgs, stream); break;
        case 7: launch<D, 7, DS>(p, nkvh, ne3, wgs, stream); break;
        case 8: launch<D, 8, DS>(p, nkvh, ne3, wgs, stream); break;
        default: GGML_ABORT("xe1dec: unsupported GQA %d", gqa);
    }
}

static int xe1dec_ds(int D) {
    static const int ds256 = getenv("GGML_SYCL_XE1DEC_DS256") ? atoi(getenv("GGML_SYCL_XE1DEC_DS256")) : 4;
    return D == 256 ? (ds256 == 2 ? 2 : 4) : 2;
}

static int xe1dec_r(int D, int gqa) {
    return std::max(1, 16 / (gqa * xe1dec_ds(D)));
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
    static const bool quant_off = getenv("GGML_SYCL_XE1DEC_QUANT_OFF") != nullptr;
    const bool kv_f16   = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    const bool kv_quant = !quant_off && K->type == V->type && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0);
    if (Q->ne[1] < 1 || Q->ne[1] > 4 || Q->type != GGML_TYPE_F32 || !(kv_f16 || kv_quant) ||
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
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[2] != 1 || mask->ne[1] < Q->ne[1] ||
                 mask->nb[1] % 2 != 0)) {
        return false;
    }
    auto al16 = [](size_t x) { return x % 16 == 0; };
    if (kv_quant) {  // 2-block chunks read with 4-byte block reads
        auto al4 = [](size_t x) { return x % 4 == 0; };
        return al16(Q->nb[2]) && al16(Q->nb[3]) && al4(K->nb[1]) && al4(K->nb[2]) && al4(K->nb[3]) && al4(V->nb[1]) &&
               al4(V->nb[2]) && al4(V->nb[3]) && al4((uintptr_t) K->data) && al4((uintptr_t) V->data) &&
               K->nb[0] == ggml_type_size(K->type) && ggml_is_contiguous(dst);
    }
    if (Q->ne[1] > 1 && !al16(Q->nb[1])) {
        return false;
    }
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
    const int nq   = (int) Q->ne[1];          // query tokens: each is its own row of the grid
    const int ne3  = (int) Q->ne[3] * nq;     // batch x query rows; dst [D, heads, nq, batch] matches this order
    const int gqa  = ne1 / nkvh;
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    static const int target_wg = getenv("GGML_SYCL_XE1DEC_WG") ? atoi(getenv("GGML_SYCL_XE1DEC_WG")) : 256;
    const bool kv_q   = K->type != GGML_TYPE_F16;  // quantized kernels always use DS 4 (D 256) / 2 (D 128)
    const int R       = kv_q ? std::max(1, 16 / (gqa * (D == 256 ? 4 : 2))) : xe1dec_r(D, gqa);
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
    p.nq   = nq;
    p.nbq1 = Q->nb[1];
    p.m_row = mask ? (int64_t) (mask->nb[1] / sizeof(sycl::half)) : 0;
    p.KV = KV; p.split_kv = split_kv; p.k_num = k_num; p.ne1 = ne1; p.scale = scale;

    if (getenv("GGML_SYCL_XE1DEC_DBG") && atoi(getenv("GGML_SYCL_XE1DEC_DBG")) == 2) {
        // timing only: merge without the partial kernel
    } else if (K->type == GGML_TYPE_Q8_0) {
        if (D == 256) { launch_gqa_q<256, 4, GGML_TYPE_Q8_0>(gqa, p, nkvh, ne3, wgs, stream); }
        else          { launch_gqa_q<128, 2, GGML_TYPE_Q8_0>(gqa, p, nkvh, ne3, wgs, stream); }
    } else if (K->type == GGML_TYPE_Q4_0) {
        if (D == 256) { launch_gqa_q<256, 4, GGML_TYPE_Q4_0>(gqa, p, nkvh, ne3, wgs, stream); }
        else          { launch_gqa_q<128, 2, GGML_TYPE_Q4_0>(gqa, p, nkvh, ne3, wgs, stream); }
    } else if (D == 256) {
        if (xe1dec_ds(256) == 2) {
            launch_gqa<256, 2>(gqa, p, nkvh, ne3, wgs, stream);
        } else {
            launch_gqa<256, 4>(gqa, p, nkvh, ne3, wgs, stream);
        }
    } else {
        launch_gqa<128, 2>(gqa, p, nkvh, ne3, wgs, stream);
    }

    static const int dbg = getenv("GGML_SYCL_XE1DEC_DBG") ? atoi(getenv("GGML_SYCL_XE1DEC_DBG")) : 0;
    if (dbg == 1) {
        return;  // timing only: skip the merge
    }
    const float * pp = part.get();
    const float * sk = sinks ? (const float *) sinks->data : nullptr;
    float *       dd = (float *) dst->data;
    const int     ne2 = 1;  // query rows are folded into ne3
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_w(sycl::range<1>(k_num), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, (D + 63) / 64, (size_t) ne1 * 64), sycl::range<3>(1, 1, 64)),
                         [=](sycl::nd_item<3> it) {
                             xe1dec_reduce(pp, sk, dd, D, ne1, ne2, ne3, k_num, it,
                                           sh_w.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

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
#include "llama-xe1.hpp"

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/intel/esimd.hpp>

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
                          const sycl::nd_item<3> & it, float * sh_w, int8_t * q8 = nullptr, float * sh_o = nullptr) {
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
    O = sycl::clamp(O, -3.402823466e+38f, 3.402823466e+38f);
    dst[(int64_t) i3 * ne2 * ne1 * D + (int64_t) D * n + d] = O;
    if (q8) {  // [a770-opt] single token, D % 64 == 0: also the q8_1 SoA copy for the output projection (llama-xe1)
        sh_o[tid] = O;
        sycl::group_barrier(grp);
        const auto sg = it.get_sub_group();
        const int  s  = sg.get_group_linear_id();
        if (s < 2) {
            const int lane = sg.get_local_linear_id();
            const int blk  = (D * n + it.get_group(1) * 64) / 32 + s;
            ggml_sycl_xe1_quant_block_store(sh_o[32 * s + 2 * lane], sh_o[32 * s + 2 * lane + 1], q8, D * ne1, blk, sg);
        }
    }
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
        case 1: launch_q<D, 1, DS, KVT>(p, nkvh, ne3, wgs, stream); break;  // [a770-opt] MHA (Llama 2)
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
        case 1: launch<D, 1, DS>(p, nkvh, ne3, wgs, stream); break;  // [a770-opt] MHA (Llama 2)
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
    return D == 256 ? (ds256 == 2 ? 2 : 4) : D == 64 ? 1 : 2;  // [a770-opt] D 64: LFM2 / LFM2.5
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

static bool xe1dec_supported_impl(int device, const ggml_tensor * dst, int & why) {
    static const bool disabled = getenv("GGML_SYCL_XE1DEC_OFF") != nullptr;
    // the generic kernels lose even at a 256-token cache (180 vs ~90 us), so no minimum by default
    static const int  min_kv   = getenv("GGML_SYCL_XE1DEC_MIN_KV") ? atoi(getenv("GGML_SYCL_XE1DEC_MIN_KV")) : 0;
    if (disabled) {
        return (why = 1, false);
    }
    if (!ggml_sycl_device_is_alchemist(device)) {
        return (why = 2, false);
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int64_t D = K->ne[0];
    static const bool quant_off = getenv("GGML_SYCL_XE1DEC_QUANT_OFF") != nullptr;
    const bool kv_f16   = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    const bool kv_quant = !quant_off && K->type == V->type && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0);
    // [a770-opt] the XMX path (quantized cache, long enough) also takes 5..16 tokens (a prompt's tail, which the
    // generic kernels serve by dequantizing the whole cache: 13 tokens at 32K took 650 ms)
    static const int xmx_min = getenv("GGML_SYCL_XE1X_MIN_KV") ? atoi(getenv("GGML_SYCL_XE1X_MIN_KV")) : 2048;
    const int max_q = kv_quant && xmx_min > 0 && K->ne[1] >= xmx_min && Q->ne[2] / K->ne[2] <= 8 && Q->ne[3] == 1 ? 16 : 4;
    if (Q->ne[1] < 1 || Q->ne[1] > max_q || Q->type != GGML_TYPE_F32 || !(kv_f16 || kv_quant) ||
        dst->type != GGML_TYPE_F32 || V->ne[0] != D || (D != 64 && D != 128 && D != 256) || K->ne[1] < min_kv ||
        max_bias != 0.0f || softcap != 0.0f || (sinks && sinks->type != GGML_TYPE_F32)) {
        return (why = 3, false);
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return (why = 4, false);
    }
    const int64_t gqa = Q->ne[2] / K->ne[2];
    if (gqa < 1 || gqa > 8 || K->ne[2] != V->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return (why = 5, false);
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[2] != 1 || mask->ne[1] < Q->ne[1] ||
                 mask->nb[1] % 2 != 0)) {
        return (why = 6, false);
    }
    auto al16 = [](size_t x) { return x % 16 == 0; };
    if (kv_quant && Q->ne[1] > 1 && !al16(Q->nb[1])) {
        return (why = 9, false);
    }
    if (kv_quant) {  // 2-block chunks read with 4-byte block reads
        auto al4 = [](size_t x) { return x % 4 == 0; };
        return al16(Q->nb[2]) && al16(Q->nb[3]) && al4(K->nb[1]) && al4(K->nb[2]) && al4(K->nb[3]) && al4(V->nb[1]) &&
               al4(V->nb[2]) && al4(V->nb[3]) && al4((uintptr_t) K->data) && al4((uintptr_t) V->data) &&
               K->nb[0] == ggml_type_size(K->type) && ggml_is_contiguous(dst);
    }
    if (Q->ne[1] > 1 && !al16(Q->nb[1])) {
        return (why = 7, false);
    }
    return al16(Q->nb[2]) && al16(Q->nb[3]) && al16(K->nb[1]) && al16(K->nb[2]) && al16(K->nb[3]) && al16(V->nb[1]) &&
           al16(V->nb[2]) && al16(V->nb[3]) && ggml_is_contiguous(dst);
}

// [a770-opt] GGML_SYCL_XE1DEC_WHY=1: print (once per reason) why a FLASH_ATTN_EXT is not taken, with its shapes
bool ggml_sycl_flash_attn_ext_xe1dec_supported(int device, const ggml_tensor * dst) {
    int why = 0;
    const bool ok = xe1dec_supported_impl(device, dst, why);
    static const bool dbg = getenv("GGML_SYCL_XE1DEC_WHY") && atoi(getenv("GGML_SYCL_XE1DEC_WHY"));
    static uint64_t seen = 0;
    const int key = why * 2 + (dst->src[0]->ne[1] <= 4 ? 1 : 0);
    if (!ok && dbg && key < 64 && !(seen & (1ull << key))) {
        seen |= 1ull << key;
        const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * m = dst->src[3];
        fprintf(stderr, "xe1dec: declined (check %d) Q[%lld %lld %lld %lld] nb[%zu %zu %zu] K %s [%lld %lld %lld] nb[%zu %zu %zu] V %s nb[%zu %zu %zu] mask %s [%lld %lld] nb1 %zu\n",
                why, (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                ggml_type_name(K->type), (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], K->nb[1], K->nb[2], K->nb[3],
                ggml_type_name(V->type), V->nb[1], V->nb[2], V->nb[3], m ? ggml_type_name(m->type) : "-", m ? (long long) m->ne[0] : 0LL,
                m ? (long long) m->ne[1] : 0LL, m ? m->nb[1] : (size_t) 0);
    }
    return ok;
}

// [a770-opt] Long-context decode on a quantized cache with the XMX units (ESIMD, DG2's 8-wide DPAS).
// The vector kernels above are ALU-bound on q4_0 / q8_0 caches (Bonsai 27B at 32K: ~700 us per call, ~50 GB/s). Here
// the GQA query heads sharing a KV head are the 8 columns of a DPAS: Q.K^T is an int8 DPAS (q8_0 K as s8, q4_0 K as
// u4 with Q's 32 values permuted into the nibble order and the -8 offset folded in through sum(q)), P.V an f16 DPAS on
// V dequantized once per KV head for all of them. A work-group = D/32 ESIMD threads (one 32-dim block each) over one KV
// head and a slice of positions; the per-block Q.K^T partials meet in SLM every 16 positions. Output = xe1dec partials.
namespace xe1x {
namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;
using xmx::dpas_argument_type;

constexpr int TP = 16;  // positions per step (two 8-row DPAS for Q.K^T, one 16-deep DPAS per 8 dims for P.V)

template <int KVT> struct kv;
template <> struct kv<GGML_TYPE_Q4_0> { static constexpr int BB = 18, QD = 4, RD = 8;  };  // qs dwords, dwords read
template <> struct kv<GGML_TYPE_Q8_0> { static constexpr int BB = 34, QD = 8, RD = 16; };

// R rows of one block each: RD dwords from the dword-aligned start of the block (one LSC gather per 8 dwords x R
// rows); returns the QD qs dwords per row (funnel-shifted when the block starts 2 bytes into a dword) and the scales
// R rows of one block each: RD dwords from the dword-aligned start of the block, one LSC gather per 8 dwords x R rows.
// A non-transposed LSC gather returns component-major data: dword k of address r at k * R + r. issue() only starts
// the loads (the next step's rows are fetched while this one computes), finish() unpacks them.
template <int KVT, int R>
ESIMD_INLINE esimd::simd<uint32_t, R * kv<KVT>::RD> issue_rows(const char * base, esimd::simd<uint32_t, R> ro) {
    using namespace esimd;
    constexpr int RD = kv<KVT>::RD;
    simd<uint32_t, R * RD> wc;
#pragma unroll
    for (int c = 0; c < RD / 8; ++c) {
        wc.template select<R * 8, 1>(c * 8 * R) = gather<uint32_t, R * 8, 8>((const uint32_t *) base, ro + 32u * c);
    }
    return wc;
}

// odd block (starts 2 bytes into a dword, base = 2 bytes before it): scale = high half of dword 0, qs = dwords 1..
// even block (dword aligned): scale = low half of dword 0, qs = (dword i >> 16) | (dword i + 1 << 16)
template <int KVT, int R>
ESIMD_INLINE void finish_rows(esimd::simd<uint32_t, R * kv<KVT>::RD> wc, const bool odd,
                              esimd::simd<uint32_t, R * kv<KVT>::QD> & qs, esimd::simd<float, R> & d) {
    using namespace esimd;
    constexpr int QD = kv<KVT>::QD;
    simd<uint32_t, R> w0 = wc.template select<R, 1>(0);
    simd<uint16_t, R> sh = convert<uint16_t>(odd ? simd<uint32_t, R>(w0 >> 16) : simd<uint32_t, R>(w0 & 0xFFFFu));
#pragma unroll
    for (int i = 0; i < QD; ++i) {
        simd<uint32_t, R> lo = wc.template select<R, 1>(i * R);
        simd<uint32_t, R> hi = wc.template select<R, 1>((i + 1) * R);
        qs.template select<R, QD>(i) = odd ? hi : simd<uint32_t, R>((lo >> 16) | (hi << 16));
    }
    d = convert<float>(sh.template bit_cast_view<sycl::half>().read());
}

template <int D, int KVT>
ESIMD_INLINE void thread(const xe1dec_params p, const int gqa, const int kvh, const int split, const int iqt, const int tid) {
    using namespace esimd;
    constexpr int NT = D / 32;
    constexpr int BB = kv<KVT>::BB, QD = kv<KVT>::QD, RD = kv<KVT>::RD;
    slm_init<NT * TP * 8 * sizeof(float)>();

    const int head0 = kvh * gqa;
    simd<uint32_t, 64> bq = 0;
    simd<float, 8>     dq = 0.0f, qs8 = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < gqa) {
            const float *    qp = (const float *) (p.q + (int64_t) iqt * p.nbq1 + (int64_t) (head0 + c) * p.nbq2) + tid * 32;
            simd<float, 32>  qf = block_load<float, 32>(qp) * p.scale;
            const float      am = hmax<float>(abs(qf));
            const float      id = am > 0.0f ? 127.0f / am : 0.0f;
            simd<float, 32>  qr = rnde<float, 32>(qf * id);
            simd<int8_t, 32> qe;
            if constexpr (KVT == GGML_TYPE_Q4_0) {  // u4 element 2j = byte j low = dim j, 2j+1 = high = dim j + 16
                qe.template select<16, 2>(0) = convert<int8_t>(simd<float, 16>(qr.template select<16, 1>(0)));
                qe.template select<16, 2>(1) = convert<int8_t>(simd<float, 16>(qr.template select<16, 1>(16)));
            } else {
                qe = convert<int8_t>(qr);
            }
            bq.template select<8, 8>(c) = qe.template bit_cast_view<uint32_t>().read();
            dq[c]  = am / 127.0f;
            qs8[c] = KVT == GGML_TYPE_Q4_0 ? 8.0f * reduce<float>(qr, std::plus<>()) : 0.0f;
        }
    }
    const simd<float, 64> qs8r = qs8.template replicate<8>();  // [r][c]
    const simd<float, 64> dqr  = dq.template replicate<8>();

    const int    start = split * p.split_kv;
    const int    end   = p.KV < start + p.split_kv ? p.KV : start + p.split_kv;
    const bool   odd   = (tid * BB) % 4 != 0;  // the block starts 2 bytes into a dword (rows are 4-byte multiples)
    const char * kb    = p.k + (int64_t) kvh * p.nbk2 + tid * BB + (odd ? -2 : 0);
    const char * vb    = p.v + (int64_t) kvh * p.nbv2 + tid * BB + (odd ? -2 : 0);
    const sycl::half * mb = p.m ? p.m + (int64_t) iqt * p.m_row : nullptr;

    simd<float, 64> O[4];  // dims 8g + d, [d][c]
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        O[g] = 0.0f;
    }
    simd<float, 8> m_run = -3.402823466e+38f * 0.5f;
    simd<float, 8> l_run = 0.0f;
    const simd<uint32_t, TP> lane(0, 1);

    for (int p0 = start; p0 < end; p0 += TP) {
        simd<uint32_t, TP> pos = lane + p0;
        simd_mask<TP>      ok  = pos < (uint32_t) end;
        simd<uint32_t, TP> pc  = pos;
        pc.merge(simd<uint32_t, TP>(end - 1), !ok);
        // all loads of the step first: K, V (16 rows each), mask
        simd<uint32_t, TP * RD> kw = issue_rows<KVT, TP>(kb, pc * (uint32_t) p.nbk1);
        simd<uint32_t, TP * RD> vw = issue_rows<KVT, TP>(vb, pc * (uint32_t) p.nbv1);
        simd<float, TP> mk = 0.0f;
        if (mb) {
            simd<uint16_t, TP> mh = gather<uint16_t, TP>((const uint16_t *) mb, pc * 2u);
            mk = convert<float>(mh.template bit_cast_view<sycl::half>().read());
        }
        mk.merge(simd<float, TP>(-INFINITY), !ok);
        simd<uint32_t, TP * QD> kq;
        simd<float, TP>         dk;
        finish_rows<KVT, TP>(kw, odd, kq, dk);

        // Q.K^T for this block: [16 pos][8 c]
        simd<float, TP * 8> sp;
#pragma unroll
        for (int h = 0; h < TP / 8; ++h) {
            simd<uint32_t, 8 * QD> a = kq.template select<8 * QD, 1>(h * 8 * QD);
            simd<int, 64> ci = 0;
            if constexpr (KVT == GGML_TYPE_Q4_0) {
                ci = xmx::dpas<8, 8, int, int, uint32_t, uint32_t, dpas_argument_type::s8, dpas_argument_type::u4>(ci, bq, a);
            } else {
                ci = xmx::dpas<8, 8, int, int, uint32_t, uint32_t, dpas_argument_type::s8, dpas_argument_type::s8>(ci, bq, a);
            }
            simd<float, 8> dkh = dk.template select<8, 1>(8 * h);
            sp.template select<64, 1>(64 * h) = (convert<float>(ci) - qs8r) * dqr * dkh.template replicate_vs_w_hs<8, 1, 8, 0>(0);
        }
        slm_block_store<float, TP * 8>(tid * TP * 8 * sizeof(float), sp);
        barrier();
        simd<float, TP * 8> s = slm_block_load<float, TP * 8>(0);
#pragma unroll
        for (int t = 1; t < NT; ++t) {
            s += slm_block_load<float, TP * 8>(t * TP * 8 * sizeof(float));
        }
        barrier();

        // masked positions (-inf) may hold garbage K / V (unwritten cache slots): their scores are set, not added to,
        // and their V scales zeroed, so a NaN / inf there cannot reach the sums (0 * inf in the DPAS)
        const simd<float, TP * 8> mkr = mk.template replicate_vs_w_hs<TP, 1, 8, 0>(0);
        s += mkr;
        s.merge(simd<float, TP * 8>(-INFINITY), mkr == -INFINITY);
        simd<float, 64> m64 = max(simd<float, 64>(s.template select<64, 1>(0)), simd<float, 64>(s.template select<64, 1>(64)));
        simd<float, 32> m32 = max(simd<float, 32>(m64.template select<32, 1>(0)), simd<float, 32>(m64.template select<32, 1>(32)));
        simd<float, 16> m16 = max(simd<float, 16>(m32.template select<16, 1>(0)), simd<float, 16>(m32.template select<16, 1>(16)));
        simd<float, 8>  mx  = max(simd<float, 8>(m16.template select<8, 1>(0)), simd<float, 8>(m16.template select<8, 1>(8)));
        const simd<float, 8> m_new = max(m_run, mx);
        const simd<float, 8> so    = exp(m_run - m_new);
        s = exp(s - m_new.template replicate<TP>());
        simd<float, 64> s64 = simd<float, 64>(s.template select<64, 1>(0)) + simd<float, 64>(s.template select<64, 1>(64));
        simd<float, 32> s32 = simd<float, 32>(s64.template select<32, 1>(0)) + simd<float, 32>(s64.template select<32, 1>(32));
        simd<float, 16> s16 = simd<float, 16>(s32.template select<16, 1>(0)) + simd<float, 16>(s32.template select<16, 1>(16));
        l_run = l_run * so + (simd<float, 8>(s16.template select<8, 1>(0)) + simd<float, 8>(s16.template select<8, 1>(8)));
        m_run = m_new;
        const simd<float, 64> sor = so.template replicate<8>();  // O is [d][c]
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            O[g] *= sor;
        }

        // B = P [16 pos][8 c] in VNNI pairs of positions: halfs (kp, c, j) = P[2kp + j][c]
        simd<sycl::half, TP * 8> ph = convert<sycl::half>(s);
        simd<sycl::half, TP * 8> pb;
#pragma unroll
        for (int kp = 0; kp < TP / 2; ++kp) {
            pb.template select<8, 2>(kp * 16)     = ph.template select<8, 1>((2 * kp) * 8);
            pb.template select<8, 2>(kp * 16 + 1) = ph.template select<8, 1>((2 * kp + 1) * 8);
        }
        const simd<uint32_t, 64> pb32 = pb.template bit_cast_view<uint32_t>().read();

        // V block dequantized to f16 [pos][32 dims], A_g = V^T rows of dims 8g..8g+7 x 16 positions
        simd<uint32_t, TP * QD> vq;
        simd<float, TP>         dv;
        finish_rows<KVT, TP>(vw, odd, vq, dv);
        simd<sycl::half, TP * 32> vh;
        if constexpr (KVT == GGML_TYPE_Q4_0) {
            simd<uint8_t, TP * 16> vbytes = vq.template bit_cast_view<uint8_t>().read();
            simd<sycl::half, TP * 16> lo = convert<sycl::half>(simd<int16_t, TP * 16>(vbytes & 0xF) - 8);
            simd<sycl::half, TP * 16> hi = convert<sycl::half>(simd<int16_t, TP * 16>(vbytes >> 4) - 8);
#pragma unroll
            for (int r = 0; r < TP; ++r) {
                vh.template select<16, 1>(r * 32)      = lo.template select<16, 1>(r * 16);
                vh.template select<16, 1>(r * 32 + 16) = hi.template select<16, 1>(r * 16);
            }
        } else {
            simd<int8_t, TP * 32> vbytes = vq.template bit_cast_view<int8_t>().read();
            vh = convert<sycl::half>(vbytes);
        }
        simd<float, TP> dvz = dv;
        dvz.merge(simd<float, TP>(0.0f), mk == -INFINITY);
        vh *= convert<sycl::half>(dvz).template replicate_vs_w_hs<TP, 1, 32, 0>(0);
        vh.merge(simd<sycl::half, TP * 32>(0.0f), convert<sycl::half>(dvz).template replicate_vs_w_hs<TP, 1, 32, 0>(0) == 0.0f);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            simd<sycl::half, 8 * TP> av;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                av.template select<TP, 1>(d * TP) = vh.template select<TP, 32>(8 * g + d);
            }
            O[g] = xmx::dpas<8, 8, float, float, uint32_t, uint32_t, dpas_argument_type::fp16, dpas_argument_type::fp16>(
                O[g], pb32, av.template bit_cast_view<uint32_t>().read());
        }
    }

    // partials in the xe1dec layout (split = osplit, one query row): O unnormalised, then l and m per head
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < gqa) {
            float * op = p.part + ((int64_t) (split + p.k_num * iqt) * p.ne1 + head0 + c) * D + tid * 32;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                block_store<float, 8>(op + 8 * g, simd<float, 8>(O[g].template select<8, 8>(c)));
            }
        }
    }
    if (tid == 0) {
        const int64_t n_o = (int64_t) D * p.ne1 * p.k_num;
        const int64_t n_o_all = n_o * p.nq;  // partials of all query rows come first
        float *       lm  = p.part + n_o_all + (int64_t) (split + p.k_num * iqt) * 2 * p.ne1;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < gqa) {
                lm[head0 + c]         = l_run[c];
                lm[p.ne1 + head0 + c] = m_run[c];
            }
        }
    }
}

template <int D, int KVT>
static void launch(const xe1dec_params & p, int gqa, int nkvh, int nsplit, dpct::queue_ptr stream) {
    constexpr int NT = D / 32;
    const int     nq = p.nq;  // one work-group per query token (MTP verification re-reads the cache per token)
    stream->parallel_for(sycl::nd_range<1>((size_t) nkvh * nsplit * nq * NT, NT), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        int g = (int) it.get_group(0);
        const int kvh = g % nkvh;
        g /= nkvh;
        thread<D, KVT>(p, gqa, kvh, g % nsplit, g / nsplit, (int) it.get_local_id(0));
    });
}
}  // namespace xe1x

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

    // [a770-opt] split target: 256 work-groups suits few KV heads (GQA); MHA (Llama 2, 32 KV heads) is faster with 64
    // (fewer partials to merge: 7B decode 90.1 -> 91.4 t/s at depth 0, 72.0 -> 73.8 at 2048)
    static const int env_wg    = getenv("GGML_SYCL_XE1DEC_WG") ? atoi(getenv("GGML_SYCL_XE1DEC_WG")) : 0;
    const int        target_wg = env_wg > 0 ? env_wg : (K->ne[2] >= 16 ? 64 : 256);
    const bool kv_q   = K->type != GGML_TYPE_F16;  // quantized kernels always use DS 4 (D 256) / 2 (D 128)
    const int R       = kv_q ? std::max(1, 16 / (gqa * (D == 256 ? 4 : D == 64 ? 1 : 2))) : xe1dec_r(D, gqa);
    const int want    = std::max(2, (target_wg + nkvh * ne3 - 1) / (nkvh * ne3));
    const int nsplit  = std::min(512 / R, want);
    int split_kv      = (KV + nsplit - 1) / nsplit;
    split_kv          = (split_kv + 15) / 16 * 16;
    const int wgs     = (KV + split_kv - 1) / split_kv;
    int       k_num   = wgs * R;

    // [a770-opt] XMX path: one decode token, quantized cache, <= 8 query heads per KV head, long enough cache
    static const int xmx_min = getenv("GGML_SYCL_XE1X_MIN_KV") ? atoi(getenv("GGML_SYCL_XE1X_MIN_KV")) : 2048;
    const bool use_xmx = kv_q && xmx_min > 0 && ne3 == nq && KV >= xmx_min && gqa <= 8;  // batch 1, 1-16 tokens
    GGML_ASSERT(use_xmx || nq <= 4);
    if (use_xmx) {
        // positions per work-group (GGML_SYCL_XE1X_SPLIT); partials as xe1dec with R = 1
        static const int per = getenv("GGML_SYCL_XE1X_SPLIT") ? atoi(getenv("GGML_SYCL_XE1X_SPLIT")) : 256;
        int ns   = std::min(2048, std::max(2, (KV + per - 1) / per));
        split_kv = ((KV + ns - 1) / ns + xe1x::TP - 1) / xe1x::TP * xe1x::TP;
        k_num    = (KV + split_kv - 1) / split_kv;
    }

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
    } else if (use_xmx) {
        if (K->type == GGML_TYPE_Q4_0) {
            D == 256 ? xe1x::launch<256, GGML_TYPE_Q4_0>(p, gqa, nkvh, k_num, stream)
            : D == 64 ? xe1x::launch<64, GGML_TYPE_Q4_0>(p, gqa, nkvh, k_num, stream)
                     : xe1x::launch<128, GGML_TYPE_Q4_0>(p, gqa, nkvh, k_num, stream);
        } else {
            D == 256 ? xe1x::launch<256, GGML_TYPE_Q8_0>(p, gqa, nkvh, k_num, stream)
            : D == 64 ? xe1x::launch<64, GGML_TYPE_Q8_0>(p, gqa, nkvh, k_num, stream)
                     : xe1x::launch<128, GGML_TYPE_Q8_0>(p, gqa, nkvh, k_num, stream);
        }
    } else if (K->type == GGML_TYPE_Q8_0) {
        if (D == 256) { launch_gqa_q<256, 4, GGML_TYPE_Q8_0>(gqa, p, nkvh, ne3, wgs, stream); }
        else if (D == 64) { launch_gqa_q<64, 1, GGML_TYPE_Q8_0>(gqa, p, nkvh, ne3, wgs, stream); }
        else          { launch_gqa_q<128, 2, GGML_TYPE_Q8_0>(gqa, p, nkvh, ne3, wgs, stream); }
    } else if (K->type == GGML_TYPE_Q4_0) {
        if (D == 256) { launch_gqa_q<256, 4, GGML_TYPE_Q4_0>(gqa, p, nkvh, ne3, wgs, stream); }
        else if (D == 64) { launch_gqa_q<64, 1, GGML_TYPE_Q4_0>(gqa, p, nkvh, ne3, wgs, stream); }
        else          { launch_gqa_q<128, 2, GGML_TYPE_Q4_0>(gqa, p, nkvh, ne3, wgs, stream); }
    } else if (D == 256) {
        if (xe1dec_ds(256) == 2) {
            launch_gqa<256, 2>(gqa, p, nkvh, ne3, wgs, stream);
        } else {
            launch_gqa<256, 4>(gqa, p, nkvh, ne3, wgs, stream);
        }
    } else if (D == 64) {
        launch_gqa<64, 1>(gqa, p, nkvh, ne3, wgs, stream);
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
    // [a770-opt] one decode token: the merge also writes the q8 copy the Q4_0 output projection reads (llama-xe1 stash)
    int8_t * q8 = nullptr;
    static const bool q8_on = ggml_sycl_knob("GGML_SYCL_XE1_FA_Q8", 1) && ggml_sycl_knob("GGML_SYCL_XE1_FUSE", 1);
    if (q8_on && ne3 == 1 && D % 64 == 0 && ggml_is_contiguous(dst)) {
        q8 = (int8_t *) ggml_sycl_xe1_stash_slot(stream, (size_t) D * ne1 + (size_t) D * ne1 / 32 * 4);
    }
    if (q8) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> sh_w(sycl::range<1>(k_num), cgh);
            sycl::local_accessor<float, 1> sh_o(sycl::range<1>(64), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, D / 64, (size_t) ne1 * 64), sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                                 xe1dec_reduce(pp, sk, dd, D, ne1, ne2, ne3, k_num, it,
                                               sh_w.get_multi_ptr<sycl::access::decorated::no>().get(), q8,
                                               sh_o.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        ggml_sycl_xe1_stash_put(dst, q8);
        return;
    }
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_w(sycl::range<1>(k_num), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(ne3, (D + 63) / 64, (size_t) ne1 * 64), sycl::range<3>(1, 1, 64)),
                         [=](sycl::nd_item<3> it) {
                             xe1dec_reduce(pp, sk, dd, D, ne1, ne2, ne3, k_num, it,
                                           sh_w.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

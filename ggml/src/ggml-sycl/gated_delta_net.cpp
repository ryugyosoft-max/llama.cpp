#include <unordered_map>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>


template <int S_v, bool KDA, bool keep_rs_t>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K,
                          const int32_t *   s_ids,
                          int64_t           s_row_stride) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // fused gather: read this sequence's live state straight from its cache row s_ids[sequence]
    const int64_t state_in_offset      = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : (int64_t) sequence * H * S_v * S_v) +
                                         h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// ARC-LAB replay (Feature A; independent implementation of the technique described by llamADALL): where a
// GATED_DELTA_NET_REPLAY pass puts the state after token t_base (the base) and the update records of the tokens after
// it. Record layout (R = H*(2*S + 1) floats): [delta S*H | k S*H | g H], see ggml_gated_delta_net_replay.
struct gdn_emit {
    float *         base           = nullptr;  // [S, S, H] per seq; nullptr = not written
    float *         rec            = nullptr;  // record 0 of seq s at rec + (rec_rows ? rec_rows[s] : s) * rec_row_stride
    const int64_t * rec_rows       = nullptr;  // fused cache write: the SET_ROWS row (bank) per seq
    int64_t         rec_row_stride = 0;        // floats
    int64_t         t_base         = 0;        // -1: base = the replayed input state (replay-only launch)
    int64_t         R              = 0;
};

// ARC-LAB token-blocked sequential kernel (non-KDA, S_v = 128). Same per-token recurrence and per-column
// arithmetic/reduction order as gated_delta_net_sycl, but the sequential kernel stalls on global-memory latency
// every token (~0.4 us/step on the B580 even with the GPU mostly idle) and needs ~5 rounds of threads at 48 heads.
// Here each subgroup owns NC columns and shares its k/q registers across them (CUDA's cols_per_warp), and the
// work-group stages T tokens of k, q, v, exp(g), beta in SLM; the next block's global loads are issued into
// registers before the current block is computed, so their latency overlaps T steps of compute.
// emit_t (ARC-LAB replay): also write the base state and the update records described by `em`; the per-token
// arithmetic is untouched, the emission only stores values the step already computed.
template <int NC, int NW, int T, bool keep_rs_t, bool emit_t = false>
static void gdn_blocked_sycl(const float * q, const float * k, const float * v, const float * g, const float * beta,
                             const float * curr_state, float * dst, float * state, int64_t H, int64_t n_tokens,
                             int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2,
                             int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3,
                             float scale, int64_t state_slot_stride, int K, const int32_t * s_ids,
                             int64_t s_row_stride, dpct::queue_ptr stream, const gdn_emit em = gdn_emit()) {
    constexpr int S = 128, L = 16, RPL = S / L;
    constexpr int NCW = NC * NW, WGS = NW * L, ncb = S / NCW;
    constexpr int PK = T * S / WGS, PV = T * NCW / WGS;  // per-thread prefetch registers
    static_assert((T * S) % WGS == 0 && (T * NCW) % WGS == 0 && T <= WGS && S % NCW == 0, "gdn_blocked shape");
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(T * S), cgh);
        sycl::local_accessor<float, 1> sq(sycl::range<1>(T * S), cgh);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(T * NCW), cgh);
        sycl::local_accessor<float, 1> sgb(sycl::range<1>(2 * T), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n_seqs * H * ncb * WGS)), sycl::range<1>(WGS)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int     tid  = it.get_local_id(0);
                const int     lane = tid % L;
                const int     w    = tid / L;
                const int64_t grp  = it.get_group(0);
                const int64_t seq  = grp / (H * ncb);
                const int64_t h    = (grp / ncb) % H;
                const int     colb = (int) (grp % ncb) * NCW;  // first column of the work-group
                const int     col0 = colb + w * NC;            // first column of this subgroup
                const int64_t iq1  = h % neqk1;
                const int64_t iq3  = seq / rq3;

                const float * qb   = q + iq3 * sq3 + iq1 * sq1;
                const float * kb   = k + iq3 * sq3 + iq1 * sq1;
                const float * vb   = v + seq * sv3 + h * sv1 + colb;
                const float * gp   = g + seq * sb3 + h * sb1;
                const float * bp   = beta + seq * sb3 + h * sb1;
                float *       attn = dst + (seq * n_tokens * H + h) * S;
                float *       st   = state + (seq * H + h) * S * S;
                const float * cs   = curr_state + (s_ids ? (int64_t) s_ids[seq] * s_row_stride : seq * H * S * S) + h * S * S;

                float s[NC][RPL];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
#pragma unroll
                    for (int r = 0; r < RPL; ++r) {
                        s[c][r] = cs[(col0 + c) * S + r * L + lane];
                    }
                }

                float pk[PK], pq[PK], pv[PV], pg = 0.0f, pb = 0.0f;
                auto fetch = [&](int64_t t0) {
#pragma unroll
                    for (int j = 0; j < PK; ++j) {
                        const int  e  = tid + j * WGS, t = e / S, i = e % S;
                        const bool ok = t0 + t < n_tokens;
                        pk[j] = ok ? kb[(t0 + t) * sq2 + i] : 0.0f;
                        pq[j] = ok ? qb[(t0 + t) * sq2 + i] : 0.0f;
                    }
#pragma unroll
                    for (int j = 0; j < PV; ++j) {
                        const int e = tid + j * WGS, t = e / NCW, c = e % NCW;
                        pv[j] = t0 + t < n_tokens ? vb[(t0 + t) * sv2 + c] : 0.0f;
                    }
                    if (tid < T && t0 + tid < n_tokens) {
                        pg = gp[(t0 + tid) * sb2];
                        pb = bp[(t0 + tid) * sb2];
                    }
                };
                fetch(0);

                for (int64_t t0 = 0; t0 < n_tokens; t0 += T) {
                    it.barrier(sycl::access::fence_space::local_space);  // previous block's SLM reads are done
#pragma unroll
                    for (int j = 0; j < PK; ++j) {
                        sk[tid + j * WGS] = pk[j];
                        sq[tid + j * WGS] = pq[j];
                    }
#pragma unroll
                    for (int j = 0; j < PV; ++j) {
                        sv[tid + j * WGS] = pv[j];
                    }
                    if (tid < T) {
                        sgb[tid]     = sycl::native::exp(pg);
                        sgb[T + tid] = pb;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (t0 + T < n_tokens) {
                        fetch(t0 + T);  // in flight while this block is computed
                    }

                    const int nt = (int) sycl::min((int64_t) T, n_tokens - t0);
                    for (int t = 0; t < nt; ++t) {
                        float kr[RPL], qr[RPL];
#pragma unroll
                        for (int r = 0; r < RPL; ++r) {
                            kr[r] = sk[t * S + r * L + lane];
                            qr[r] = sq[t * S + r * L + lane];
                        }
                        const float g_val    = sgb[t];
                        const float beta_val = sgb[T + t];
                        float       out      = 0.0f;
                        [[maybe_unused]] float dl[NC];  // emit_t: this token's delta per column
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            float kv_shard = 0.0f;
#pragma unroll
                            for (int r = 0; r < RPL; ++r) {
                                kv_shard += s[c][r] * kr[r];
                            }
                            const float kv_col    = warp_reduce_sum<L>(kv_shard);
                            const float delta_col = (sv[t * NCW + w * NC + c] - g_val * kv_col) * beta_val;
                            if constexpr (emit_t) {
                                dl[c] = delta_col;
                            }
                            float       attn_partial = 0.0f;
#pragma unroll
                            for (int r = 0; r < RPL; ++r) {
                                s[c][r] = g_val * s[c][r] + kr[r] * delta_col;
                                attn_partial += s[c][r] * qr[r];
                            }
                            const float attn_col = warp_reduce_sum<L>(attn_partial);  // same sum on every lane
                            if (lane == c) {
                                out = attn_col;
                            }
                        }
                        if (lane < NC) {
                            attn[(t0 + t) * S * H + col0 + lane] = out * scale;
                        }
                        if constexpr (emit_t) {
                            const int64_t tt = t0 + t;
                            if (tt == em.t_base) {
                                float * bs = em.base + (seq * H + h) * S * S;
#pragma unroll
                                for (int c = 0; c < NC; ++c) {
#pragma unroll
                                    for (int r = 0; r < RPL; ++r) {
                                        bs[(col0 + c) * S + r * L + lane] = s[c][r];
                                    }
                                }
                            } else if (tt > em.t_base) {
                                float * rr = em.rec + (em.rec_rows ? em.rec_rows[seq] : seq) * em.rec_row_stride +
                                             (tt - em.t_base - 1) * em.R;
#pragma unroll
                                for (int c = 0; c < NC; ++c) {
                                    if (lane == c) {
                                        rr[h * S + col0 + c] = dl[c];
                                    }
                                }
                                if (colb == 0 && w == 0) {
#pragma unroll
                                    for (int r = 0; r < RPL; ++r) {
                                        rr[H * S + h * S + r * L + lane] = kr[r];
                                    }
                                    if (lane == 0) {
                                        rr[2 * H * S + h] = g_val;
                                    }
                                }
                            }
                        }
                        if constexpr (keep_rs_t) {
                            const int target_slot = (int) (n_tokens - 1 - (t0 + t));
                            if (target_slot >= 0 && target_slot < K) {
                                float * cst = st + target_slot * state_slot_stride;
#pragma unroll
                                for (int c = 0; c < NC; ++c) {
#pragma unroll
                                    for (int r = 0; r < RPL; ++r) {
                                        cst[(col0 + c) * S + r * L + lane] = s[c][r];
                                    }
                                }
                            }
                        }
                    }
                }

                if constexpr (!keep_rs_t) {
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
#pragma unroll
                        for (int r = 0; r < RPL; ++r) {
                            st[(col0 + c) * S + r * L + lane] = s[c][r];
                        }
                    }
                }
            });
    });
}

template <bool keep_rs_t, bool emit_t = false>
static void launch_gdn_blocked(int variant, const float * q, const float * k, const float * v, const float * g,
                               const float * beta, const float * s0, float * dst, float * state, int64_t H,
                               int64_t n_tokens, int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1,
                               int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1,
                               int64_t rq3, float scale, int64_t state_slot_stride, int K, const int32_t * s_ids,
                               int64_t s_row_stride, dpct::queue_ptr stream, const gdn_emit & em = gdn_emit()) {
#define GDN_BLK(NC, NW, T) gdn_blocked_sycl<NC, NW, T, keep_rs_t, emit_t>(q, k, v, g, beta, s0, dst, state, H, n_tokens, \
        n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, \
        stream, em)
    if (variant < 0) {
        // auto: widest NC that still leaves ~768 subgroups (~60% of the B580's 1280 thread slots) to hide latency.
        // B580, 1024 tokens (old kernel -> this): 48 heads 3383 -> 1688 us (NC 8, T 8); 32 heads 2762 -> 1024 (NC 4);
        // 4 heads 403 -> 312 (NC 1). NC 8 at 32 heads = 1767, NC 4 at 48 heads = 1934 (more than one round of threads).
        const int64_t cols = H * n_seqs * 128;
        variant = cols >= 768 * 8 ? 9 : cols >= 768 * 4 ? 3 : cols >= 768 * 2 ? 4 : 5;
    }
    switch (variant) {  // lab sweep (GGML_SYCL_GDN_BLK_VARIANT); NC columns/subgroup, NW subgroups, T tokens/block
        case 1: GDN_BLK(8, 16, 16); break;
        case 2: GDN_BLK(8, 8, 16); break;
        case 3: GDN_BLK(4, 16, 16); break;
        case 4: GDN_BLK(2, 16, 16); break;
        case 5: GDN_BLK(1, 16, 16); break;
        case 6: GDN_BLK(4, 32, 16); break;
        case 7: GDN_BLK(4, 8, 32); break;
        case 8: GDN_BLK(8, 8, 8); break;
        case 9: GDN_BLK(8, 16, 8); break;
        case 10: GDN_BLK(4, 16, 8); break;
        default: GDN_BLK(4, 8, 16); break;
    }
#undef GDN_BLK
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   const int32_t * s_ids,
                                   int64_t         s_row_stride,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    if constexpr (!KDA) {
        static const bool blk_off     = getenv("GGML_SYCL_GDN_BLOCKED_OFF") != nullptr;
        static const int  blk_variant = getenv("GGML_SYCL_GDN_BLK_VARIANT") ? atoi(getenv("GGML_SYCL_GDN_BLK_VARIANT")) : -1;
        static const int  blk_min     = getenv("GGML_SYCL_GDN_BLK_MIN") ? atoi(getenv("GGML_SYCL_GDN_BLK_MIN")) : 16;
        if (!blk_off && S_v == 128 && n_tokens >= blk_min) {
            launch_gdn_blocked<keep_rs_t>(blk_variant, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                          n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                                          state_slot_stride, K, s_ids, s_row_stride, stream);
            return;
        }
    }
    if constexpr (!KDA && !keep_rs_t) {
        // [a770-opt] one decode token, S_v = 128: a sub-group per value column, each lane 8 contiguous state rows as two
        // float4 (the general kernel makes 8 strided 4-byte loads per lane and carries 64-bit index math); same
        // per-column arithmetic and reduction structure. GGML_SYCL_GDN_DEC1=0 restores the general kernel.
        static const bool dec1 = !getenv("GGML_SYCL_GDN_DEC1") || atoi(getenv("GGML_SYCL_GDN_DEC1")) != 0;
        if (dec1 && S_v == 128 && n_tokens == 1 && n_seqs == 1 && sq1 % 4 == 0 && ((uintptr_t) q_d % 16) == 0 &&
            ((uintptr_t) k_d % 16) == 0 && ((uintptr_t) s_d % 16) == 0 && ((uintptr_t) state_d % 16) == 0 &&
            s_row_stride % 4 == 0 && H * 128 * 128 < INT32_MAX) {
            const int Hi = (int) H, sq1i = (int) sq1, sv1i = (int) sv1, sb1i = (int) sb1, nk = (int) neqk1;
            constexpr int NSG = 8;  // columns per work-group
            stream->parallel_for(sycl::nd_range<1>((size_t) Hi * (128 / NSG) * NSG * 16, NSG * 16),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    const auto sg   = it.get_sub_group();
                    const int  lane = sg.get_local_linear_id();
                    const int  grp  = (int) it.get_group(0);
                    const int  h    = grp / (128 / NSG);
                    const int  col  = (grp % (128 / NSG)) * NSG + (int) sg.get_group_linear_id();
                    const int  hk   = h % nk;
                    const float * sin = s_d + (s_ids ? (int64_t) s_ids[0] * s_row_stride : (int64_t) 0) + h * 16384 + col * 128 + lane * 8;
                    const sycl::float4 s0 = *(const sycl::float4 *) sin, s1 = *(const sycl::float4 *) (sin + 4);
                    const float * kh = k_d + hk * sq1i + lane * 8;
                    const float * qh = q_d + hk * sq1i + lane * 8;
                    const sycl::float4 k0 = *(const sycl::float4 *) kh, k1 = *(const sycl::float4 *) (kh + 4);
                    const sycl::float4 q0 = *(const sycl::float4 *) qh, q1 = *(const sycl::float4 *) (qh + 4);
                    const float g_val    = sycl::native::exp(g_d[h * sb1i]);
                    const float beta_val = b_d[h * sb1i];
                    const float v_col    = v_d[h * sv1i + col];
                    float st[8] = { s0.x(), s0.y(), s0.z(), s0.w(), s1.x(), s1.y(), s1.z(), s1.w() };
                    const float kk[8] = { k0.x(), k0.y(), k0.z(), k0.w(), k1.x(), k1.y(), k1.z(), k1.w() };
                    const float qq[8] = { q0.x(), q0.y(), q0.z(), q0.w(), q1.x(), q1.y(), q1.z(), q1.w() };
                    float kv = 0.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        kv += st[r] * kk[r];
                    }
                    kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
                    const float delta = (v_col - g_val * kv) * beta_val;
                    float       at    = 0.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        st[r] = g_val * st[r] + kk[r] * delta;
                        at += st[r] * qq[r];
                    }
                    at = sycl::reduce_over_group(sg, at, sycl::plus<float>());
                    if (lane == 0) {
                        dst_d[h * 128 + col] = at * scale;
                    }
                    float * so = state_d + h * 16384 + col * 128 + lane * 8;
                    *(sycl::float4 *) so       = sycl::float4(st[0], st[1], st[2], st[3]);
                    *(sycl::float4 *) (so + 4) = sycl::float4(st[4], st[5], st[6], st[7]);
                });
            return;
        }
    }
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const int num_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
                                     });
            }
            break;
        case 64: {
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
                                        });
            }
            break;
        }
        case 128: {
            {
                constexpr int sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
                                        });
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// ARC-LAB: chunked prefill for the scalar-gate gated delta rule (S_k = S_v = 128, one sequence).
// Per token:  delta = beta * (v - gamma * S^T k),  S <- gamma * S + k delta^T,  o = scale * S^T q,  gamma = exp(g).
// Per chunk of C tokens (UT / WY form, as build_delta_net_chunking), with G = cumsum(g) inside the chunk:
//   A[t][s]  = beta_t exp(G_t - G_s) k_t.k_s        (s < t)
//   R        = beta * (V - exp(G) * (K S0))         -> Delta = (I + A)^-1 R  (forward substitution)
//   O        = scale * (exp(G) * (Q S0) + (exp(G_t - G_s) q_t.k_s)_{s<=t} Delta)
//   S_C      = exp(G_last) S0 + K^T (exp(G_last - G) * Delta)
// One work-group per (head, block of W value columns): the value columns of S evolve independently, the C x C
// terms are recomputed per block. q/k of a chunk are shared in SLM by all W columns; every product is fused, so
// the serial chain is one step per chunk instead of one per token. fp32 throughout.
namespace {
constexpr int GDN_CH_C  = 16;   // tokens per chunk
constexpr int GDN_CH_W  = 32;   // value columns per work-group
constexpr int GDN_CH_S  = 128;  // head size
constexpr int GDN_CH_KP = GDN_CH_S + 1;  // padded SLM row stride of the K / Q tiles
constexpr int GDN_CH_WG = 256;
}

template <int C, int W>
static void gdn_chunked_sycl_t(const float * q, const float * k, const float * v, const float * g, const float * beta,
                             const float * s_in, float * dst, float * s_out, int64_t H, int64_t n_proc,
                             int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2, int64_t sb1, int64_t sb2,
                             int64_t neqk1, float scale, const int32_t * s_ids, int64_t s_row_stride,
                             dpct::queue_ptr stream) {
    constexpr int S = GDN_CH_S, KP = GDN_CH_KP, WG = GDN_CH_WG;
    constexpr int ncb = S / W;
    static_assert((C * W) % WG == 0 || C * W < WG, "C*W vs WG");
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sS(sycl::range<1>(S * W), cgh);
        sycl::local_accessor<float, 1> sK(sycl::range<1>(C * KP), cgh);
        sycl::local_accessor<float, 1> sQ(sycl::range<1>(C * KP), cgh);
        sycl::local_accessor<float, 1> sV(sycl::range<1>(C * W), cgh);
        sycl::local_accessor<float, 1> sD(sycl::range<1>(C * W), cgh);
        sycl::local_accessor<float, 1> sA(sycl::range<1>(C * C), cgh);
        sycl::local_accessor<float, 1> sQK(sycl::range<1>(C * C), cgh);
        sycl::local_accessor<float, 1> sG(sycl::range<1>(C), cgh);
        sycl::local_accessor<float, 1> sB(sycl::range<1>(C), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) H * ncb * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int     tid  = it.get_local_id(0);
                const int     grp  = it.get_group(0);
                const int     h    = grp / ncb;
                const int     col0 = (grp % ncb) * W;
                const int64_t hk   = h % neqk1;

                const float * sin = s_in + (s_ids ? (int64_t) s_ids[0] * s_row_stride : (int64_t) 0) + (int64_t) h * S * S;
                for (int e = tid; e < S * W; e += WG) {
                    const int c = e / S, i = e % S;
                    sS[i * W + c] = sin[(int64_t) (col0 + c) * S + i];
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int64_t t0 = 0; t0 < n_proc; t0 += C) {
                    const int nc = (int) sycl::min((int64_t) C, n_proc - t0);
                    for (int e = tid; e < C * S; e += WG) {
                        const int t = e / S, i = e % S;
                        float kv = 0.0f, qv = 0.0f;
                        if (t < nc) {
                            const int64_t off = (t0 + t) * sq2 + hk * sq1 + i;
                            kv = k[off];
                            qv = q[off];
                        }
                        sK[t * KP + i] = kv;
                        sQ[t * KP + i] = qv;
                    }
                    for (int e = tid; e < C * W; e += WG) {
                        const int t = e / W, c = e % W;
                        sV[e] = t < nc ? v[(t0 + t) * sv2 + h * sv1 + col0 + c] : 0.0f;
                    }
                    if (tid < C) {
                        // padded tokens: g = 0 keeps G at the last real value, beta = 0 zeroes their delta
                        sG[tid] = tid < nc ? g[(t0 + tid) * sb2 + h * sb1] : 0.0f;
                        sB[tid] = tid < nc ? beta[(t0 + tid) * sb2 + h * sb1] : 0.0f;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (tid == 0) {
                        float acc = 0.0f;
                        for (int t = 0; t < C; ++t) {
                            acc += sG[t];
                            sG[t] = acc;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // C x C terms: A (strictly lower, beta and decay folded in) and the decayed q.k (lower incl. diag)
                    for (int e = tid; e < 2 * C * C; e += WG) {
                        const int which = e / (C * C), ts = e % (C * C), t = ts / C, s2 = ts % C;
                        const int ra = which ? t * KP : t * KP;
                        float d = 0.0f;
                        if (which == 0) {
                            for (int i = 0; i < S; ++i) d += sK[ra + i] * sK[s2 * KP + i];
                            sA[ts] = s2 < t ? sB[t] * sycl::exp(sG[t] - sG[s2]) * d : 0.0f;
                        } else {
                            for (int i = 0; i < S; ++i) d += sQ[ra + i] * sK[s2 * KP + i];
                            sQK[ts] = s2 <= t ? sycl::exp(sG[t] - sG[s2]) * d : 0.0f;
                        }
                    }
                    // K S0 and Q S0 for this column block (C x W each), sharing the S0 reads
                    float qs_reg[(C * W + WG - 1) / WG];
#pragma unroll
                    for (int r = 0; r < (C * W + WG - 1) / WG; ++r) {
                        const int e = tid + r * WG, t = e / W, c = e % W;
                        if (e >= C * W) break;
                        float ks = 0.0f, qs = 0.0f;
                        for (int i = 0; i < S; ++i) {
                            const float sv = sS[i * W + c];
                            ks += sK[t * KP + i] * sv;
                            qs += sQ[t * KP + i] * sv;
                        }
                        const float eg = sycl::exp(sG[t]);
                        sD[e]     = sB[t] * (sV[e] - eg * ks);  // R
                        qs_reg[r] = eg * qs;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Delta = (I + A)^-1 R: forward substitution, one column per work-item
                    if (tid < W) {
                        const int c = tid;
                        for (int t = 1; t < C; ++t) {
                            float r = sD[t * W + c];
                            for (int s2 = 0; s2 < t; ++s2) r -= sA[t * C + s2] * sD[s2 * W + c];
                            sD[t * W + c] = r;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                    for (int r = 0; r < (C * W + WG - 1) / WG; ++r) {
                        const int e = tid + r * WG, t = e / W, c = e % W;
                        if (e < C * W && t < nc) {
                            float o = qs_reg[r];
                            for (int s2 = 0; s2 <= t; ++s2) o += sQK[t * C + s2] * sD[s2 * W + c];
                            dst[((t0 + t) * H + h) * S + col0 + c] = o * scale;
                        }
                    }

                    const float glast = sG[C - 1];
                    const float egl   = sycl::exp(glast);
                    float dec[C];
#pragma unroll
                    for (int t = 0; t < C; ++t) dec[t] = sycl::exp(glast - sG[t]);
                    for (int e = tid; e < S * W; e += WG) {
                        const int i = e / W, c = e % W;
                        float acc = egl * sS[e];
#pragma unroll
                        for (int t = 0; t < C; ++t) acc += sK[t * KP + i] * dec[t] * sD[t * W + c];
                        sS[e] = acc;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                float * so = s_out + (int64_t) h * S * S;
                for (int e = tid; e < S * W; e += WG) {
                    const int c = e / S, i = e % S;
                    so[(int64_t) (col0 + c) * S + i] = sS[i * W + c];
                }
            });
    });
}

static void gdn_chunked_sycl(const float * q, const float * k, const float * v, const float * g, const float * beta,
                             const float * s_in, float * dst, float * s_out, int64_t H, int64_t n_proc,
                             int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2, int64_t sb1, int64_t sb2,
                             int64_t neqk1, float scale, const int32_t * s_ids, int64_t s_row_stride,
                             dpct::queue_ptr stream) {
    static const int variant = getenv("GGML_SYCL_GDN_CH_VARIANT") ? atoi(getenv("GGML_SYCL_GDN_CH_VARIANT")) : 0;  // lab sweep
#define GDN_CH_CALL(CC, WW) gdn_chunked_sycl_t<CC, WW>(q, k, v, g, beta, s_in, dst, s_out, H, n_proc, sq1, sq2, sv1, sv2, sb1, sb2, neqk1, scale, s_ids, s_row_stride, stream)
    switch (variant) {
        case 1: GDN_CH_CALL(16, 64); break;
        case 2: GDN_CH_CALL(32, 32); break;
        case 3: GDN_CH_CALL(32, 16); break;
        case 4: GDN_CH_CALL(8, 32); break;
        case 5: GDN_CH_CALL(16, 16); break;
        case 6: GDN_CH_CALL(8, 64); break;
        default: GDN_CH_CALL(16, 32); break;
    }
#undef GDN_CH_CALL
}


// [a770-opt] Chunked prefill in three kernels (S_k = S_v = 128, one sequence). Same per-chunk algebra as
// gdn_chunked_sycl, split so that everything independent of the state runs for all chunks in parallel:
//   dots (per (head, chunk)):  G = cumsum(g),  A = (beta_t e^{G_t-G_s} k_t.k_s)_{s<t},  M = (e^{G_t-G_s} q_t.k_s)_{s<=t},
//                              e^G and e^{G_last-G}
//   wu   (per (head, chunk)):  T = (I + A)^-1,  W = T diag(beta e^G) K,  U = T diag(beta) V
//   scan (per (head, 16 value columns), chunks in order):
//        Delta = U - W S,   O = scale (e^G (Q S) + M Delta),   S <- e^{G_last} S + (e^{G_last-G} K)^T Delta
// Alchemist notes. dots / wu are SIMT: one operand of every product is one value per lane, broadcast with a
// constant-index select_from_group (IGC emits a scalar register region in the mad, no shuffle), and loop trip counts
// are compile-time so the loads of a step are in flight together. The scan is ESIMD: in SIMT form IGC hoisted every
// load of the unrolled chunk body and spilled 2-9 KB per thread (DG2 has 128 GRF; a SIMD16 float costs 2), which
// made it slower than the token-serial kernel. Measured limits: L2-resident row loads reach ~0.8 TB/s, so the scan's
// floor is its W / Q / K / M traffic, and a work-group barrier costs ~0.5 us. A770, Qwen3.5 9B (32 heads), 512
// tokens per layer: token-blocked kernel ~730 us, this ~550 us (dots 130, wu 110, scan 315); pp512 2381 -> 2472.
// The serial chain is n_tokens / 32 steps. fp32 throughout.
namespace {
constexpr int GDN_CK_C = 32;   // tokens per chunk
constexpr int GDN_CK_S = 128;  // head size
}

static void gdn_ck_dots(const float * q, const float * k, const float * g, const float * beta, float * Ad, float * Md,
                        float * Xd, int64_t H, int64_t n_proc, int64_t nch, int64_t sq1, int64_t sq2, int64_t sb1,
                        int64_t sb2, int64_t neqk1, dpct::queue_ptr stream) {
    constexpr int C = GDN_CK_C, S = GDN_CK_S, L = 16, NSG = 8, WG = NSG * L, TG = 4;
    // 16 tasks per (head, chunk): 8 groups of 4 query/key rows t x 2 blocks of 16 key rows s (lane = s)
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (H * nch * WG)), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto    sg   = it.get_sub_group();
            const int     lane = sg.get_local_linear_id();
            const int     w    = sg.get_group_linear_id();
            const int64_t grp  = it.get_group(0);
            const int64_t h    = grp / nch, ch = grp % nch;
            const int64_t t0   = ch * C;
            const int     nc   = (int) sycl::min((int64_t) C, n_proc - t0);
            const int64_t hk   = h % neqk1;

            // padded tokens: g = 0 keeps G at the last real value, beta = 0 zeroes their rows
            const float g_lo = lane < nc ? g[(t0 + lane) * sb2 + h * sb1] : 0.0f;
            const float g_hi = L + lane < nc ? g[(t0 + L + lane) * sb2 + h * sb1] : 0.0f;
            const float b_lo = lane < nc ? beta[(t0 + lane) * sb2 + h * sb1] : 0.0f;
            const float b_hi = L + lane < nc ? beta[(t0 + L + lane) * sb2 + h * sb1] : 0.0f;
            const float G_lo = sycl::inclusive_scan_over_group(sg, g_lo, sycl::plus<float>());
            const float G_hi = sycl::inclusive_scan_over_group(sg, g_hi, sycl::plus<float>()) +
                               sycl::select_from_group(sg, G_lo, L - 1);
            if (w == 0) {
                const float Gl = sycl::select_from_group(sg, G_hi, L - 1);
                float *     X  = Xd + grp * 2 * C;
                X[lane]         = sycl::exp(G_lo);
                X[L + lane]     = sycl::exp(G_hi);
                X[C + lane]     = sycl::exp(Gl - G_lo);
                X[C + L + lane] = sycl::exp(Gl - G_hi);
            }

#pragma unroll 1
            for (int task = w; task < 2 * NSG; task += NSG) {
                const int tg = task / 2, bb = task % 2;
                const int s  = bb * L + lane;
                float *   Ao = Ad + grp * C * C;
                float *   Mo = Md + grp * C * C;
                if (bb * L > tg * TG + TG - 1) {  // every s > t: zero block
#pragma unroll
                    for (int jj = 0; jj < TG; ++jj) {
                        Ao[(tg * TG + jj) * C + s] = 0.0f;
                        Mo[(tg * TG + jj) * C + s] = 0.0f;
                    }
                    continue;
                }
                const float * ks_row = k + (t0 + s) * sq2 + hk * sq1;
                const bool    s_ok   = s < nc;
                float         ak[TG], aq[TG];
#pragma unroll
                for (int jj = 0; jj < TG; ++jj) {
                    ak[jj] = 0.0f;
                    aq[jj] = 0.0f;
                }
#pragma unroll 2
                for (int m = 0; m < S / L; ++m) {
                    float kt[TG], qt[TG];
#pragma unroll
                    for (int jj = 0; jj < TG; ++jj) {
                        const int     t   = tg * TG + jj;
                        const int64_t off = (t0 + t) * sq2 + hk * sq1 + m * L + lane;
                        kt[jj] = t < nc ? k[off] : 0.0f;
                        qt[jj] = t < nc ? q[off] : 0.0f;
                    }
                    sycl::vec<float, 4> kv4[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        kv4[u] = s_ok ? *(const sycl::vec<float, 4> *) (ks_row + m * L + 4 * u) : sycl::vec<float, 4>(0.0f);
                    }
#pragma unroll
                    for (int r = 0; r < L; ++r) {
                        const float kv = kv4[r / 4][r % 4];
#pragma unroll
                        for (int jj = 0; jj < TG; ++jj) {
                            ak[jj] += sycl::select_from_group(sg, kt[jj], r) * kv;
                            aq[jj] += sycl::select_from_group(sg, qt[jj], r) * kv;
                        }
                    }
                }
                const float Gs = bb ? G_hi : G_lo;
#pragma unroll
                for (int jj = 0; jj < TG; ++jj) {
                    const int   t   = tg * TG + jj;  // uniform
                    const float Gt  = sycl::group_broadcast(sg, t < L ? G_lo : G_hi, t % L);
                    const float bt  = sycl::group_broadcast(sg, t < L ? b_lo : b_hi, t % L);
                    const float dec = s <= t ? sycl::exp(Gt - Gs) : 0.0f;
                    Ao[t * C + s]   = s < t ? bt * dec * ak[jj] : 0.0f;
                    Mo[t * C + s]   = dec * aq[jj];
                }
            }
        });
}

static void gdn_ck_wu(const float * k, const float * v, const float * beta, const float * Ad, const float * Xd,
                      float * Wd, float * Ud, int64_t H, int64_t n_proc, int64_t nch, int64_t sq1, int64_t sq2,
                      int64_t sv1, int64_t sv2, int64_t sb1, int64_t sb2, int64_t neqk1, dpct::queue_ptr stream) {
    constexpr int C = GDN_CK_C, S = GDN_CK_S, L = 16, WG = 2 * S, CP = C + 1;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sA(sycl::range<1>(C * CP), cgh);
        sycl::local_accessor<float, 1> sT(sycl::range<1>(C * CP), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (H * nch * WG)), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const auto    sg   = it.get_sub_group();
                const int     tid  = it.get_local_id(0);
                const int     lane = tid % L;
                const int64_t grp  = it.get_group(0);
                const int64_t h    = grp / nch, ch = grp % nch;
                const int64_t t0   = ch * C;
                const int     nc   = (int) sycl::min((int64_t) C, n_proc - t0);
                const int64_t hk   = h % neqk1;
#pragma unroll
                for (int jj = 0; jj < C * C / WG; ++jj) {
                    const int e = tid + jj * WG;
                    sA[(e / C) * CP + e % C] = Ad[grp * C * C + e];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // T = (I + A)^-1 (unit lower triangular): work-items 0..31 own one column each, right-looking
                // substitution; column s of A is held two rows per lane and broadcast
                if (tid < C) {
                    float x[C];
#pragma unroll
                    for (int t = 0; t < C; ++t) {
                        x[t] = t == tid ? 1.0f : 0.0f;
                    }
#pragma unroll
                    for (int s = 0; s < C - 1; ++s) {
                        const float a_lo = sA[lane * CP + s], a_hi = sA[(L + lane) * CP + s];
#pragma unroll
                        for (int t = s + 1; t < C; ++t) {
                            x[t] -= sycl::select_from_group(sg, t < L ? a_lo : a_hi, t % L) * x[s];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < C; ++t) {
                        sT[t * CP + tid] = x[t];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // W = T diag(beta e^G) K (work-items 0..127, column j) and U = T diag(beta) V (128..255)
                const int     j   = tid % S;
                const bool    isw = tid < S;
                const float * eG  = Xd + grp * 2 * C;
                float         b[C];
#pragma unroll
                for (int t = 0; t < C; ++t) {
                    const float bt = t < nc ? beta[(t0 + t) * sb2 + h * sb1] : 0.0f;
                    const float x  = t < nc ? (isw ? eG[t] * k[(t0 + t) * sq2 + hk * sq1 + j]
                                                   : v[(t0 + t) * sv2 + h * sv1 + j]) : 0.0f;
                    b[t] = bt * x;
                }
                float * out = (isw ? Wd : Ud) + grp * C * S;
#pragma unroll 2
                for (int t = 0; t < C; ++t) {
                    const float tr_lo = sT[t * CP + lane], tr_hi = sT[t * CP + L + lane];
                    float       a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                    for (int s = 0; s < L; ++s) {
                        a0 += sycl::select_from_group(sg, tr_lo, s) * b[s];
                        a1 += sycl::select_from_group(sg, tr_hi, s) * b[L + s];
                    }
                    out[t * S + j] = a0 + a1;
                }
            });
    });
}

// scan (ESIMD): one work-group = 16 threads = one head x CW value columns; thread w owns state rows 8w..8w+7 (all CW
// columns) as explicit registers and computes tokens w and w + 16 of every chunk. Every product is a scalar x row
// mad: the row-indexed operand (W, Q, M, K) is a register element, the other operand a CW-wide row from SLM. In SIMT
// form IGC hoisted every load of the unrolled chunk body and spilled 2-9 KB per thread; here registers and load
// placement are explicit. SLM (CW * 512 B) holds the state copy for the products and then Delta.
namespace gdn_ck_esimd {
namespace esimd = sycl::ext::intel::esimd;

template <int CW>
ESIMD_INLINE void scan_thread(const float * q, const float * k, const float * Wd, const float * Ud, const float * Md,
                              const float * Xd, const float * s_in, float * dst, float * s_out, int H, int n_proc,
                              int nch, int sq1, int sq2, int neqk1, float scale, const int32_t * s_ids,
                              int64_t s_row_stride, int grp, int w) {
    using namespace esimd;
    constexpr int C = GDN_CK_C, S = GDN_CK_S, NT = 16, RPT = S / NT, HS = S / 2;
    slm_init<S * CW * sizeof(float)>();
    const int h    = grp / (S / CW);
    const int col0 = (grp % (S / CW)) * CW;
    const int row0 = w * RPT;
    const int hk   = h % neqk1;

    const float * sin = s_in + (s_ids ? (int64_t) s_ids[0] * s_row_stride : (int64_t) 0) + (int64_t) h * S * S;
    simd<float, RPT * CW> s;  // [r][c]
    {
        // state is column-major (element (i, c) at c * S + i): gather the 8 rows of every column
        simd<uint32_t, CW> off(0, S * sizeof(float));
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            s.template select<CW, 1>(r * CW) = gather<float, CW>(sin + (int64_t) col0 * S + row0 + r, off);
        }
    }

    for (int ch = 0; ch < nch; ++ch) {
        const int     t0  = ch * C;
        const int     nc  = n_proc - t0 < C ? n_proc - t0 : C;
        const int64_t cb  = (int64_t) h * nch + ch;
        const float * Wc  = Wd + cb * C * S;
        const float * Uc  = Ud + cb * C * S;
        const float * Mc  = Md + cb * C * C;
        const float * eG  = Xd + cb * 2 * C;
        const float * qb  = q + (int64_t) t0 * sq2 + (int64_t) hk * sq1;
        const float * kb  = k + (int64_t) t0 * sq2 + (int64_t) hk * sq1;
        const int     ta  = w, tb = w + NT;  // this thread's tokens

#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            slm_block_store<float, CW>((row0 + r) * CW * sizeof(float), s.template select<CW, 1>(r * CW));
        }
        // first half rows of W / Q for both tokens (pad tokens: W rows are 0, Q not loaded)
        simd<float, HS> wa = block_load<float, HS>(Wc + ta * S), wb = block_load<float, HS>(Wc + tb * S);
        simd<float, HS> qa = 0.0f, qbv = 0.0f;
        if (ta < nc) qa = block_load<float, HS>(qb + (int64_t) ta * sq2);
        if (tb < nc) qbv = block_load<float, HS>(qb + (int64_t) tb * sq2);
        barrier();

        simd<float, CW> aw0 = 0.0f, aw1 = 0.0f, aq0 = 0.0f, aq1 = 0.0f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if (half == 1) {
                wa = block_load<float, HS>(Wc + ta * S + HS);
                wb = block_load<float, HS>(Wc + tb * S + HS);
                if (ta < nc) qa = block_load<float, HS>(qb + (int64_t) ta * sq2 + HS);
                if (tb < nc) qbv = block_load<float, HS>(qb + (int64_t) tb * sq2 + HS);
            }
#pragma unroll 8
            for (int i = 0; i < HS; ++i) {
                const simd<float, CW> sr = slm_block_load<float, CW>((half * HS + i) * CW * sizeof(float));
                aw0 += sr * (float) wa[i];
                aw1 += sr * (float) wb[i];
                aq0 += sr * (float) qa[i];
                aq1 += sr * (float) qbv[i];
            }
        }
        const simd<float, CW> ua = block_load<float, CW>(Uc + ta * S + col0);
        const simd<float, CW> ub = block_load<float, CW>(Uc + tb * S + col0);
        const simd<float, C>  egv = block_load<float, C>(eG);
        const simd<float, C>  dec = block_load<float, C>(eG + C);
        const simd<float, C>  ma  = block_load<float, C>(Mc + ta * C);
        const simd<float, C>  mb  = block_load<float, C>(Mc + tb * C);
        barrier();  // the state copy is no longer read: Delta goes over it

        slm_block_store<float, CW>(ta * CW * sizeof(float), ua - aw0);
        slm_block_store<float, CW>(tb * CW * sizeof(float), ub - aw1);
        // K rows row0..row0+7 of every token of the chunk, for the update (issued here: the W / Q rows are dead)
        simd<float, C * RPT> kr = 0.0f;
#pragma unroll
        for (int t = 0; t < C; ++t) {
            if (t < nc) kr.template select<RPT, 1>(t * RPT) = block_load<float, RPT>(kb + (int64_t) t * sq2 + row0);
        }
        barrier();

        simd<float, CW> oa = aq0 * (float) egv[ta], ob = aq1 * (float) egv[tb];
#pragma unroll 8
        for (int s2 = 0; s2 < C; ++s2) {
            const simd<float, CW> d = slm_block_load<float, CW>(s2 * CW * sizeof(float));
            oa += d * (float) ma[s2];  // M is zero above the diagonal
            ob += d * (float) mb[s2];
        }
        if (ta < nc) block_store<float, CW>(dst + ((int64_t) (t0 + ta) * H + h) * S + col0, oa * scale);
        if (tb < nc) block_store<float, CW>(dst + ((int64_t) (t0 + tb) * H + h) * S + col0, ob * scale);

        // S <- e^{G_last} S + K^T (e^{G_last - G} Delta) on this thread's rows
        s *= (float) egv[C - 1];
#pragma unroll 4
        for (int t = 0; t < C; ++t) {
            const simd<float, CW> d = slm_block_load<float, CW>(t * CW * sizeof(float)) * (float) dec[t];
#pragma unroll
            for (int r = 0; r < RPT; ++r) {
                s.template select<CW, 1>(r * CW) += d * (float) kr[t * RPT + r];
            }
        }
        barrier();  // Delta read by everyone; the next chunk rewrites the buffer
    }

    float * so = s_out + (int64_t) h * S * S;
    simd<uint32_t, CW> off(0, S * sizeof(float));
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        scatter<float, CW>(so + (int64_t) col0 * S + row0 + r, off, s.template select<CW, 1>(r * CW));
    }
}
}  // namespace gdn_ck_esimd

template <int CW>
static void gdn_ck_scan_esimd(const float * q, const float * k, const float * Wd, const float * Ud, const float * Md,
                              const float * Xd, const float * s_in, float * dst, float * s_out, int64_t H,
                              int64_t n_proc, int64_t nch, int64_t sq1, int64_t sq2, int64_t neqk1, float scale,
                              const int32_t * s_ids, int64_t s_row_stride, dpct::queue_ptr stream) {
    constexpr int NT = 16;
    const int     nwg = (int) H * (GDN_CK_S / CW);
    stream->parallel_for(sycl::nd_range<1>((size_t) nwg * NT, NT), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        gdn_ck_esimd::scan_thread<CW>(q, k, Wd, Ud, Md, Xd, s_in, dst, s_out, (int) H, (int) n_proc, (int) nch,
                                      (int) sq1, (int) sq2, (int) neqk1, scale, s_ids, s_row_stride,
                                      (int) it.get_group(0), (int) it.get_local_id(0));
    });
}

static void gdn_ck_sycl(ggml_backend_sycl_context & ctx, const float * q, const float * k, const float * v,
                        const float * g, const float * beta, const float * s_in, float * dst, float * s_out,
                        int64_t H, int64_t n_proc, int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2, int64_t sb1,
                        int64_t sb2, int64_t neqk1, float scale, const int32_t * s_ids, int64_t s_row_stride,
                        dpct::queue_ptr stream) {
    constexpr int C = GDN_CK_C, S = GDN_CK_S;
    const int64_t nch = (n_proc + C - 1) / C;
    ggml_sycl_pool_alloc<float> W(ctx.pool(), (size_t) (H * nch * C * S));
    ggml_sycl_pool_alloc<float> U(ctx.pool(), (size_t) (H * nch * C * S));
    ggml_sycl_pool_alloc<float> A(ctx.pool(), (size_t) (H * nch * C * C));
    ggml_sycl_pool_alloc<float> M(ctx.pool(), (size_t) (H * nch * C * C));
    ggml_sycl_pool_alloc<float> X(ctx.pool(), (size_t) (H * nch * 2 * C));
    gdn_ck_dots(q, k, g, beta, A.get(), M.get(), X.get(), H, n_proc, nch, sq1, sq2, sb1, sb2, neqk1, stream);
    gdn_ck_wu(k, v, beta, A.get(), X.get(), W.get(), U.get(), H, n_proc, nch, sq1, sq2, sv1, sv2, sb1, sb2, neqk1,
              stream);
    // 16 value columns per work-group: 4096 threads, single wave; 32 measured the same (half the threads, half the
    // L2 traffic) - GGML_SYCL_GDN_CK_SCAN=32 selects it
    static const int scan_cw = getenv("GGML_SYCL_GDN_CK_SCAN") ? atoi(getenv("GGML_SYCL_GDN_CK_SCAN")) : 16;
    if (scan_cw == 32) {
        gdn_ck_scan_esimd<32>(q, k, W.get(), U.get(), M.get(), X.get(), s_in, dst, s_out, H, n_proc, nch, sq1, sq2,
                              neqk1, scale, s_ids, s_row_stride, stream);
    } else {
        gdn_ck_scan_esimd<16>(q, k, W.get(), U.get(), M.get(), X.get(), s_in, dst, s_out, H, n_proc, nch, sq1, sq2,
                              neqk1, scale, s_ids, s_row_stride, stream);
    }
}

// GET_ROWS (recurrent-state gather) -> GATED_DELTA_NET src[5] registrations, per context, reset per graph:
// the gather is skipped and the kernel indexes the cache rows directly (port of the CUDA #220 fusion).
struct ggml_sycl_gdn_gather {
    const float *   base;
    const int32_t * ids;
    int64_t         row_stride;
};
static std::unordered_map<const ggml_backend_sycl_context *, std::unordered_map<const ggml_tensor *, ggml_sycl_gdn_gather>> g_gdn_gathers;

// [a770-opt] g / beta inputs read from where the fused DeltaNet gate kernel left them (instead of a copy back), keyed
// by the GDN node; reset with the gathers
struct ggml_sycl_gdn_gate_ovr {
    const float * g;
    const float * b;
};
static std::unordered_map<const ggml_backend_sycl_context *, std::unordered_map<const ggml_tensor *, ggml_sycl_gdn_gate_ovr>> g_gdn_gate_ovr;

void ggml_sycl_gdn_gate_override(const ggml_backend_sycl_context & ctx, const ggml_tensor * gdn, const float * g,
                                 const float * b) {
    g_gdn_gate_ovr[&ctx][gdn] = { g, b };
}

void ggml_sycl_gdn_gather_reset(const ggml_backend_sycl_context & ctx) {
    g_gdn_gathers[&ctx].clear();
    g_gdn_gate_ovr[&ctx].clear();
}

void ggml_sycl_gdn_gather_set(const ggml_backend_sycl_context & ctx, const ggml_tensor * gdn, const float * base,
                              const int32_t * ids, int64_t row_stride) {
    g_gdn_gathers[&ctx][gdn] = { base, ids, row_stride };
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float *   s_d          = (const float *) src_state->data;
    float *         dst_d        = (float *) dst->data;
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    {
        auto & mo = g_gdn_gate_ovr[&ctx];
        auto   io = mo.find(dst);
        if (io != mo.end()) {
            g_d = io->second.g ? io->second.g : g_d;
            b_d = io->second.b ? io->second.b : b_d;
        }
    }
    {
        auto & m  = g_gdn_gathers[&ctx];
        auto   it = m.find(dst);
        if (it != m.end()) {
            s_d          = it->second.base;
            s_ids        = it->second.ids;
            s_row_stride = it->second.row_stride;
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        }
    } else {
        // ARC-LAB chunked prefill (OPT-IN, GGML_SYCL_GDN_CHUNKED=1): all but the last K tokens (the snapshot window)
        // go through gdn_chunked_sycl; the sequential kernel then finishes the tail from that state. Correct, but on
        // the B580 every shape measured slower than the sequential kernel (scalar SLM math, no XMX) - see README.
        static const bool chunk_off = getenv("GGML_SYCL_GDN_CHUNKED") == nullptr;
        const int64_t n_tail = keep_rs ? (int64_t) K : 0;
        const int64_t n_proc = n_tokens - n_tail;
        // [a770-opt] two-pass chunked prefill (gdn_ck_sycl), default on; GGML_SYCL_GDN_CK=0 restores the
        // token-blocked kernel, GGML_SYCL_GDN_CK_MIN sets the minimum chunked token count
        static const bool    ck_on  = !getenv("GGML_SYCL_GDN_CK") || atoi(getenv("GGML_SYCL_GDN_CK")) != 0;
        static const int64_t ck_min = getenv("GGML_SYCL_GDN_CK_MIN") ? atoll(getenv("GGML_SYCL_GDN_CK_MIN")) : 64;
        const bool ck_vec4 = sq1 % 4 == 0 && sq2 % 4 == 0 && ((uintptr_t) k_d % 16) == 0;  // vec4 key-row loads
        if (ck_on && chunk_off && ck_vec4 && S_v == GDN_CK_S && n_seqs == 1 && neq3 == 1 && n_proc >= ck_min) {
            float * so = state_d;
            ggml_sycl_pool_alloc<float> mid(ctx.pool());
            if (n_tail != 0) {
                so = mid.alloc((size_t) H * S_v * S_v);
            }
            gdn_ck_sycl(ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, so, H, n_proc, sq1, sq2, sv1, sv2, sb1, sb2, neqk1,
                        scale, s_ids, s_row_stride, stream);
            if (n_tail != 0) {
                launch_gated_delta_net<false, true>(q_d + n_proc * sq2, k_d + n_proc * sq2, v_d + n_proc * sv2,
                    g_d + n_proc * sb2, b_d + n_proc * sb2, mid.get(), dst_d + n_proc * S_v * H, state_d,
                    S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, stream);
            }
            return;
        }
        if (!chunk_off && S_v == GDN_CH_S && n_seqs == 1 && neq3 == 1 && n_proc >= 64) {
            if (n_tail == 0) {
                float * so = state_d;  // final state straight to its destination (dst tail or cache slot 0)
                gdn_chunked_sycl(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, so, H, n_proc, sq1, sq2, sv1, sv2, sb1, sb2,
                                 neqk1, scale, s_ids, s_row_stride, stream);
                return;
            }
            ggml_sycl_pool_alloc<float> mid(ctx.pool(), (size_t) H * S_v * S_v);
            gdn_chunked_sycl(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, mid.get(), H, n_proc, sq1, sq2, sv1, sv2, sb1, sb2,
                             neqk1, scale, s_ids, s_row_stride, stream);
            // tail: the sequential kernel on tokens [n_proc, n_tokens) from the mid state; n_seqs == 1 so shifting
            // the base pointers by n_proc tokens keeps every layout (dst row = t * H + h)
            launch_gated_delta_net<false, true>(q_d + n_proc * sq2, k_d + n_proc * sq2, v_d + n_proc * sv2,
                g_d + n_proc * sb2, b_d + n_proc * sb2, mid.get(), dst_d + n_proc * S_v * H, state_d,
                S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, stream);
            return;
        }
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}

// ARC-LAB: GATED_DELTA_NET_REPLAY (Feature A: DeltaNet rollback by replay instead of snapshots; independent
// implementation of the technique described by llamADALL). See ggml_gated_delta_net_replay for the data layout.
//
// Sequential kernel: gated_delta_net_sycl's scalar-gate path (same grid, same per-column arithmetic and reduction
// order, lines kept verbatim) with
//   - a replay prologue that re-applies n_rec stored records to the loaded state, using the step's own update
//     expression (g * S + k * delta) on the stored g, k and delta, so the state matches what the step produced;
//   - an epilogue that stores the base state (after token t_base) and the records of the tokens after it.
// With n_tokens == 0 and t_base == -1 it only replays and writes the result to em.base (used before the blocked
// kernel, which has no replay prologue).
template <int S_v>
static void gdn_replay_seq_sycl(const float * q, const float * k, const float * v, const float * g, const float * beta,
                                const float * curr_state, float * dst, int64_t H, int64_t n_tokens, int64_t sq1,
                                int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1,
                                int64_t sb2, int64_t sb3, const sycl::uint3 neqk1_magic, const sycl::uint3 rq3_magic,
                                float scale, const int32_t * s_ids, int64_t s_row_stride, const float * rec_in,
                                int64_t rec_in_seq_stride, int64_t rec_in_bank_stride, const int32_t * ctrl,
                                const gdn_emit em) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : (int64_t) sequence * H * S_v * S_v) +
                                    h_idx * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // replay prologue
    if (rec_in != nullptr) {
        const int32_t n_rec = ctrl[2 * sequence + 0];
        const int32_t bank  = ctrl[2 * sequence + 1];
        const float * rb    = rec_in + sequence * rec_in_seq_stride + bank * rec_in_bank_stride;
        for (int32_t j = 0; j < n_rec; j++) {
            const float * rj        = rb + j * em.R;
            const float   g_val     = rj[2 * H * S_v + h_idx];
            const float   delta_col = rj[h_idx * S_v + col];
            const float * k_t       = rj + H * S_v + h_idx * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
            }
        }
    }

    float * base_out = em.base ? em.base + (sequence * H + h_idx) * S_v * S_v + col * S_v : nullptr;
    float * rec_seq  = em.rec ? em.rec + (em.rec_rows ? em.rec_rows[sequence] : (int64_t) sequence) * em.rec_row_stride
                              : nullptr;

    if (em.t_base < 0 && base_out) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            base_out[r * warp_size + lane] = s_shard[r];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset;

        const float beta_val = *beta_t;

        const float g_val = sycl::native::exp(*g_t);

        // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            kv_shard += s_shard[r] * k_t[i];
        }
        float kv_col = warp_reduce_sum<warp_size>(kv_shard);

        // delta[col] = (v[col] - g * kv[col]) * beta
        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
        // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
            attn_partial += s_shard[r] * q_t[i];
        }

        float attn_col = warp_reduce_sum<warp_size>(attn_partial);

        if (lane == 0) {
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;

        // epilogue: base state, then one record per later token
        if (t == em.t_base) {
            if (base_out) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    base_out[r * warp_size + lane] = s_shard[r];
                }
            }
        } else if (t > em.t_base) {
            float * rt = rec_seq + (t - em.t_base - 1) * em.R;
            if (lane == 0) {
                rt[h_idx * S_v + col] = delta_col;
            }
            if (col == 0) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    rt[H * S_v + h_idx * S_v + i] = k_t[i];
                }
                if (lane == 0) {
                    rt[2 * H * S_v + h_idx] = g_val;
                }
            }
        }
    }
}

static void launch_gdn_replay_seq(const float * q_d, const float * k_d, const float * v_d, const float * g_d,
                                  const float * b_d, const float * s_d, float * dst_d, int64_t S_v, int64_t H,
                                  int64_t n_tokens, int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1,
                                  int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1,
                                  int64_t rq3, float scale, const int32_t * s_ids, int64_t s_row_stride,
                                  const float * rec_in, int64_t rec_in_seq_stride, int64_t rec_in_bank_stride,
                                  const int32_t * ctrl, const gdn_emit & em, dpct::queue_ptr stream) {
    GGML_ASSERT(S_v == 128);  // supports_op
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const int num_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    constexpr int sv = 128;
    const gdn_emit e = em;
    stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             gdn_replay_seq_sycl<sv>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, sq1, sq2, sq3,
                                                     sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, s_ids,
                                                     s_row_stride, rec_in, rec_in_seq_stride, rec_in_bank_stride, ctrl,
                                                     e);
                         });
}

void ggml_sycl_op_gated_delta_net_replay(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                         const ggml_sycl_gdn_replay_fused * fused) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/8);

    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_k    = dst->src[1];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_g    = dst->src[3];
    const ggml_tensor * src_beta = dst->src[4];
    const ggml_tensor * src_s    = dst->src[5];
    const ggml_tensor * src_rec  = dst->src[6];
    const ggml_tensor * src_ctrl = dst->src[7];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;
    const int64_t K        = ggml_get_op_params_i32(dst, 0);
    const int64_t R        = H * (2 * S_v + 1);
    const int64_t D        = S_v * S_v * H;
    const int64_t m        = std::min<int64_t>(n_tokens - 1, K - 1);

    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta) && ggml_is_contiguous(src_s));

    const int64_t neqk1 = neq1;
    const int64_t rq3   = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    float *       dst_d = (float *) dst->data;

    // fused recurrent-state gather (GET_ROWS skipped, see ggml_sycl_try_gdn_gather_skip)
    const float *   s_d          = (const float *) src_s->data;
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    {
        auto & mg = g_gdn_gathers[&ctx];
        auto   it = mg.find(dst);
        if (it != mg.end()) {
            s_d          = it->second.base;
            s_ids        = it->second.ids;
            s_row_stride = it->second.row_stride;
        }
    }

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    const float *   rec_in             = (const float *) src_rec->data;
    const int64_t   rec_in_bank_stride = src_rec->nb[1] / sizeof(float);
    const int64_t   rec_in_seq_stride  = src_rec->nb[2] / sizeof(float);
    const int32_t * ctrl               = (const int32_t *) src_ctrl->data;

    gdn_emit em;
    em.R      = R;
    em.t_base = n_tokens - 1 - m;
    if (fused) {
        em.base           = fused->base;
        em.rec            = fused->rec;
        em.rec_rows       = fused->rec_rows;
        em.rec_row_stride = fused->rec_row_stride;
    } else {
        em.base           = dst_d + S_v * H * n_tokens * n_seqs;
        em.rec            = em.base + D * n_seqs;
        em.rec_row_stride = (K - 1) * R;
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: ARC-LAB GDN rollback by replay active (cache write %s)\n", __func__,
                      fused ? "fused" : "via copy");
    }

    dpct::queue_ptr stream = ctx.stream();

    // same kernel choice as the snapshot path (launch_gated_delta_net, scalar gate, S_v == 128)
    static const bool blk_off     = getenv("GGML_SYCL_GDN_BLOCKED_OFF") != nullptr;
    static const int  blk_variant = getenv("GGML_SYCL_GDN_BLK_VARIANT") ? atoi(getenv("GGML_SYCL_GDN_BLK_VARIANT")) : -1;
    static const int  blk_min     = getenv("GGML_SYCL_GDN_BLK_MIN") ? atoi(getenv("GGML_SYCL_GDN_BLK_MIN")) : 16;
    if (!blk_off && n_tokens >= blk_min) {
        // the blocked kernel has no replay prologue: replay into a scratch state first, then run it from there
        ggml_sycl_pool_alloc<float> s_rpl(ctx.pool(), (size_t) D * n_seqs);
        gdn_emit em0;
        em0.base   = s_rpl.get();
        em0.t_base = -1;
        em0.R      = R;
        launch_gdn_replay_seq(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, S_v, H, 0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                              sb1, sb2, sb3, neqk1, rq3, scale, s_ids, s_row_stride, rec_in, rec_in_seq_stride,
                              rec_in_bank_stride, ctrl, em0, stream);
        // K = 0 with keep_rs_t: no snapshot slots and no final-state store, only the emission
        launch_gdn_blocked<true, true>(blk_variant, q_d, k_d, v_d, g_d, b_d, s_rpl.get(), dst_d, em.base, H, n_tokens,
                                       n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, 0, 0,
                                       nullptr, 0, stream, em);
        return;
    }
    launch_gdn_replay_seq(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                          sb1, sb2, sb3, neqk1, rq3, scale, s_ids, s_row_stride, rec_in, rec_in_seq_stride,
                          rec_in_bank_stride, ctrl, em, stream);
}

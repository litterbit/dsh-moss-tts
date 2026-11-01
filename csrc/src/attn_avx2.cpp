// AVX2 因果注意力：内积与加权求和都走 256 位 FMA。
#include <immintrin.h>

#include <algorithm>
#include <cmath>

#include "moss/attn.h"

namespace moss {
namespace {

inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

}  // namespace

void attention_causal_avx2(const float *qkv, int t, int q0, int q1, int dim, int heads, int window, float *out,
                           float *scores) {
    (void)t;
    const int stride = 3 * dim;
    const int head_dim = dim / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int vec = head_dim / 8;

    for (int h = 0; h < heads; ++h) {
        const int hoff = h * head_dim;
        for (int i = q0; i < q1; ++i) {
            const float *q = qkv + static_cast<size_t>(i) * stride + hoff;
            const int u_begin = window > 0 ? std::max(0, i - window + 1) : 0;
            float maxs = -1e30f;
            int u = u_begin;
            for (; u + 4 <= i + 1; u += 4) {
                __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
                for (int k = 0; k < vec; ++k) {
                    const __m256 qv = _mm256_loadu_ps(q + k * 8);
                    const float *k0 = qkv + static_cast<size_t>(u + 0) * stride + dim + hoff + k * 8;
                    const float *k1 = qkv + static_cast<size_t>(u + 1) * stride + dim + hoff + k * 8;
                    const float *k2 = qkv + static_cast<size_t>(u + 2) * stride + dim + hoff + k * 8;
                    const float *k3 = qkv + static_cast<size_t>(u + 3) * stride + dim + hoff + k * 8;
                    acc0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0), acc0);
                    acc1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k1), acc1);
                    acc2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k2), acc2);
                    acc3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k3), acc3);
                }
                float s0 = hsum_ps(acc0) * scale;
                float s1 = hsum_ps(acc1) * scale;
                float s2 = hsum_ps(acc2) * scale;
                float s3 = hsum_ps(acc3) * scale;
                scores[u + 0] = s0;
                scores[u + 1] = s1;
                scores[u + 2] = s2;
                scores[u + 3] = s3;
                maxs = std::max({maxs, s0, s1, s2, s3});
            }
            for (; u <= i; ++u) {
                const float *k = qkv + static_cast<size_t>(u) * stride + dim + hoff;
                __m256 acc = _mm256_setzero_ps();
                for (int kk = 0; kk < vec; ++kk) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + kk * 8), _mm256_loadu_ps(k + kk * 8), acc);
                }
                const float s = hsum_ps(acc) * scale;
                scores[u] = s;
                maxs = std::max(maxs, s);
            }

            float sum = 0.0f;
            for (int u2 = u_begin; u2 <= i; ++u2) {
                scores[u2] = std::exp(scores[u2] - maxs);
                sum += scores[u2];
            }
            const float inv = 1.0f / sum;
            for (int u2 = u_begin; u2 <= i; ++u2) scores[u2] *= inv;

            float *o = out + static_cast<size_t>(i) * dim + hoff;
            for (int kk = 0; kk < vec; ++kk) {
                __m256 acc = _mm256_setzero_ps();
                for (int u2 = u_begin; u2 <= i; ++u2) {
                    const __m256 v = _mm256_loadu_ps(qkv + static_cast<size_t>(u2) * stride + 2 * dim + hoff + kk * 8);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(scores[u2]), v, acc);
                }
                _mm256_storeu_ps(o + kk * 8, acc);
            }
        }
    }
}

void attention_window_avx2(const float *qkv, int t, int q0, int q1, int dim, int heads, const float *cache_k,
                           const float *cache_v, int cache_n, int window, float *out, float *scores) {
    (void)t;
    const int stride = 3 * dim;
    const int head_dim = dim / heads;
    const int vec = head_dim / 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int i = q0; i < q1; ++i) {
        const int u_first = window > 0 ? std::max(0, i + cache_n - window + 1) : 0;
        const int u_last = i + cache_n;  // 含
        // 键分两段：缓存段（连续 dim 跨步）与当前块（连续 3*dim 跨步）
        const int cache_end = std::min(u_last, cache_n - 1);
        const int blk_first = std::max(u_first, cache_n);
        for (int h = 0; h < heads; ++h) {
            const int hoff = h * head_dim;
            const float *q = qkv + static_cast<size_t>(i) * stride + hoff;
            float maxs = -1e30f;

            // 1) 缓存段：一次算 4 个键
            int u = u_first;
            for (; u + 3 <= cache_end; u += 4) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int k = 0; k < vec; ++k) {
                    const __m256 qv = _mm256_loadu_ps(q + k * 8);
                    a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(cache_k + static_cast<size_t>(u + 0) * dim + hoff + k * 8), a0);
                    a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(cache_k + static_cast<size_t>(u + 1) * dim + hoff + k * 8), a1);
                    a2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(cache_k + static_cast<size_t>(u + 2) * dim + hoff + k * 8), a2);
                    a3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(cache_k + static_cast<size_t>(u + 3) * dim + hoff + k * 8), a3);
                }
                const float s0 = hsum_ps(a0) * scale, s1 = hsum_ps(a1) * scale;
                const float s2 = hsum_ps(a2) * scale, s3 = hsum_ps(a3) * scale;
                scores[u + 0] = s0;
                scores[u + 1] = s1;
                scores[u + 2] = s2;
                scores[u + 3] = s3;
                maxs = std::max({maxs, s0, s1, s2, s3});
            }
            for (; u <= cache_end; ++u) {
                __m256 acc = _mm256_setzero_ps();
                for (int k = 0; k < vec; ++k) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + k * 8),
                                          _mm256_loadu_ps(cache_k + static_cast<size_t>(u) * dim + hoff + k * 8), acc);
                }
                const float s = hsum_ps(acc) * scale;
                scores[u] = s;
                maxs = std::max(maxs, s);
            }

            // 2) 当前块：同样 4 路
            u = blk_first;
            for (; u + 3 <= u_last; u += 4) {
                const int b = u - cache_n;
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int k = 0; k < vec; ++k) {
                    const __m256 qv = _mm256_loadu_ps(q + k * 8);
                    a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(qkv + static_cast<size_t>(b + 0) * stride + dim + hoff + k * 8), a0);
                    a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(qkv + static_cast<size_t>(b + 1) * stride + dim + hoff + k * 8), a1);
                    a2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(qkv + static_cast<size_t>(b + 2) * stride + dim + hoff + k * 8), a2);
                    a3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(qkv + static_cast<size_t>(b + 3) * stride + dim + hoff + k * 8), a3);
                }
                const float s0 = hsum_ps(a0) * scale, s1 = hsum_ps(a1) * scale;
                const float s2 = hsum_ps(a2) * scale, s3 = hsum_ps(a3) * scale;
                scores[u + 0] = s0;
                scores[u + 1] = s1;
                scores[u + 2] = s2;
                scores[u + 3] = s3;
                maxs = std::max({maxs, s0, s1, s2, s3});
            }
            for (; u <= u_last; ++u) {
                const int b = u - cache_n;
                __m256 acc = _mm256_setzero_ps();
                for (int k = 0; k < vec; ++k) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + k * 8),
                                          _mm256_loadu_ps(qkv + static_cast<size_t>(b) * stride + dim + hoff + k * 8), acc);
                }
                const float s = hsum_ps(acc) * scale;
                scores[u] = s;
                maxs = std::max(maxs, s);
            }

            // 3) softmax（在 [u_first, u_last] 连续区间上）
            float sum = 0.0f;
            for (int u2 = u_first; u2 <= u_last; ++u2) {
                scores[u2] = std::exp(scores[u2] - maxs);
                sum += scores[u2];
            }
            const float inv = 1.0f / sum;
            float *o = out + static_cast<size_t>(i) * dim + hoff;
            for (int k = 0; k < vec; ++k) _mm256_storeu_ps(o + k * 8, _mm256_setzero_ps());

            // 4) 加权求和：缓存段 + 当前块段
            for (int u2 = u_first; u2 <= cache_end; ++u2) {
                const __m256 wgt = _mm256_set1_ps(scores[u2] * inv);
                for (int k = 0; k < vec; ++k) {
                    const __m256 v = _mm256_loadu_ps(cache_v + static_cast<size_t>(u2) * dim + hoff + k * 8);
                    __m256 acc = _mm256_loadu_ps(o + k * 8);
                    _mm256_storeu_ps(o + k * 8, _mm256_fmadd_ps(wgt, v, acc));
                }
            }
            for (int u2 = blk_first; u2 <= u_last; ++u2) {
                const int b = u2 - cache_n;
                const __m256 wgt = _mm256_set1_ps(scores[u2] * inv);
                for (int k = 0; k < vec; ++k) {
                    const __m256 v = _mm256_loadu_ps(qkv + static_cast<size_t>(b) * stride + 2 * dim + hoff + k * 8);
                    __m256 acc = _mm256_loadu_ps(o + k * 8);
                    _mm256_storeu_ps(o + k * 8, _mm256_fmadd_ps(wgt, v, acc));
                }
            }
        }
    }
}

}  // namespace moss

#include <algorithm>
#include <cmath>

#include "moss/attn.h"
#include "moss/cpu_features.h"

namespace moss {

void attention_causal_scalar(const float *qkv, int t, int q0, int q1, int dim, int heads, int window, float *out,
                             float *scores) {
    (void)t;
    const int stride = 3 * dim;
    const int head_dim = dim / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int h = 0; h < heads; ++h) {
        for (int i = q0; i < q1; ++i) {
            const float *q = qkv + static_cast<size_t>(i) * stride + h * head_dim;
            const int u_begin = window > 0 ? std::max(0, i - window + 1) : 0;
            float maxs = -1e30f;
            for (int u = u_begin; u <= i; ++u) {
                const float *k = qkv + static_cast<size_t>(u) * stride + dim + h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
                dot *= scale;
                scores[u] = dot;
                maxs = std::max(maxs, dot);
            }
            float sum = 0.0f;
            for (int u = u_begin; u <= i; ++u) {
                scores[u] = std::exp(scores[u] - maxs);
                sum += scores[u];
            }
            const float inv = 1.0f / sum;
            float *o = out + static_cast<size_t>(i) * dim + h * head_dim;
            for (int d = 0; d < head_dim; ++d) o[d] = 0.0f;
            for (int u = u_begin; u <= i; ++u) {
                const float w = scores[u] * inv;
                const float *v = qkv + static_cast<size_t>(u) * stride + 2 * dim + h * head_dim;
                for (int d = 0; d < head_dim; ++d) o[d] += w * v[d];
            }
        }
    }
}

void attention_causal(const float *qkv, int t, int q0, int q1, int dim, int heads, int window, float *out,
                       float *scores) {
    if (cpu_features().avx2) {
        attention_causal_avx2(qkv, t, q0, q1, dim, heads, window, out, scores);
    } else {
        attention_causal_scalar(qkv, t, q0, q1, dim, heads, window, out, scores);
    }
}

void attention_window_scalar(const float *qkv, int t, int q0, int q1, int dim, int heads, const float *cache_k,
                             const float *cache_v, int cache_n, int window, float *out, float *scores) {
    (void)t;
    const int stride = 3 * dim;
    const int head_dim = dim / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int i = q0; i < q1; ++i) {
        const int u_first = window > 0 ? std::max(0, i + cache_n - window + 1) : 0;
        const int u_last = i + cache_n;  // 含
        for (int h = 0; h < heads; ++h) {
            const int hoff = h * head_dim;
            const float *q = qkv + static_cast<size_t>(i) * stride + hoff;
            float maxs = -1e30f;
            for (int u = u_first; u <= u_last; ++u) {
                const float *k = u < cache_n ? cache_k + static_cast<size_t>(u) * dim + hoff
                                             : qkv + static_cast<size_t>(u - cache_n) * stride + dim + hoff;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
                dot *= scale;
                scores[u] = dot;
                if (dot > maxs) maxs = dot;
            }
            float sum = 0.0f;
            for (int u = u_first; u <= u_last; ++u) {
                scores[u] = std::exp(scores[u] - maxs);
                sum += scores[u];
            }
            const float inv = 1.0f / sum;
            float *o = out + static_cast<size_t>(i) * dim + hoff;
            for (int d = 0; d < head_dim; ++d) o[d] = 0.0f;
            for (int u = u_first; u <= u_last; ++u) {
                const float w = scores[u] * inv;
                const float *v = u < cache_n ? cache_v + static_cast<size_t>(u) * dim + hoff
                                             : qkv + static_cast<size_t>(u - cache_n) * stride + 2 * dim + hoff;
                for (int d = 0; d < head_dim; ++d) o[d] += w * v[d];
            }
        }
    }
}

void attention_window(const float *qkv, int t, int q0, int q1, int dim, int heads, const float *cache_k,
                      const float *cache_v, int cache_n, int window, float *out, float *scores) {
    if (cpu_features().avx2) {
        attention_window_avx2(qkv, t, q0, q1, dim, heads, cache_k, cache_v, cache_n, window, out, scores);
    } else {
        attention_window_scalar(qkv, t, q0, q1, dim, heads, cache_k, cache_v, cache_n, window, out, scores);
    }
}

}  // namespace moss
#pragma once

namespace moss {

// 因果多头注意力（单序列），只计算查询区间 [q0, q1)。
// qkv: [t][3*dim]，行内按 [q(dim) | k(dim) | v(dim)]；out: [t][dim]。
// scores: 调用方缓冲，长度 >= q1（每个工作线程一份，避免逐行分配）。
void attention_causal(const float *qkv, int t, int q0, int q1, int dim, int heads, int window, float *out,
                      float *scores);
void attention_causal_scalar(const float *qkv, int t, int q0, int q1, int dim, int heads, int window, float *out,
                             float *scores);
void attention_causal_avx2(const float *qkv, int t, int q0, int q1, int dim, int heads, int window, float *out,
                           float *scores);

// 流式窗口注意力：keys = 缓存 cache_k/cache_v（cache_n 个，绝对位置 offset-cache_n .. offset-1）
// + 当前块 qkv[0..t)（绝对位置 offset .. offset+t-1）。
// 只保留 0 <= (offset+i) - pos < window 的键；scores 需要调用方缓冲，长度 >= cache_n+t。
void attention_window(const float *qkv, int t, int q0, int q1, int dim, int heads, const float *cache_k,
                      const float *cache_v, int cache_n, int window, float *out, float *scores);
void attention_window_scalar(const float *qkv, int t, int q0, int q1, int dim, int heads, const float *cache_k,
                             const float *cache_v, int cache_n, int window, float *out, float *scores);
void attention_window_avx2(const float *qkv, int t, int q0, int q1, int dim, int heads, const float *cache_k,
                           const float *cache_v, int cache_n, int window, float *out, float *scores);

}  // namespace moss
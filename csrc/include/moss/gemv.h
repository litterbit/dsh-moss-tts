#pragma once

#include <vector>

#include "moss/common.h"
#include "moss/parallel.h"

namespace moss {

// GEMV/GEMM 的复用缓冲，避免每步分配
struct GemvScratch {
    QuantizedActivation act;

    void ensure(int k) {
        if (act.k != k || act.blocks.size() != static_cast<size_t>(k / 32)) act.resize(k);
    }
};

// 线程池 + 每线程一份 scratch + 共享的量化激活。
// 引擎在构造时 bind 到自己的 Parallel；之后 gemv_par/gemm_par 全靠它。
struct GemvPoolScratch {
    Parallel *par = nullptr;
    std::vector<GemvScratch> per_thread;
    QuantizedActivation shared;
    std::vector<BlockQ8_0> batch;     // prefill 时整批激活的量化结果 [M][nb]
    std::vector<float> batch_scales;  // 对应的块 scale [M][nb]

    // 绑定线程池（引用必须比本对象活得久），并预分配每线程 scratch
    void bind(Parallel &pool, int k) {
        par = &pool;
        per_thread.assign(static_cast<size_t>(pool.threads()), GemvScratch{});
        for (auto &s : per_thread) s.ensure(k);
    }

    // 未绑定时视为单线程，便于测试里直接用
    int threads() const { return par ? par->threads() : 1; }

    GemvScratch &scratch(int t) {
        if (per_thread.size() <= static_cast<size_t>(t)) per_thread.resize(static_cast<size_t>(t) + 1);
        return per_thread[static_cast<size_t>(t)];
    }
};

// y[rows] = W[rows, cols] · x[cols]（单线程）
void gemv(const Matrix &w, const float *x, float *y, GemvScratch &scratch);

// 多线程版本：按输出行切分
void gemv_par(const Matrix &w, const float *x, float *y, GemvPoolScratch &pool);

// Y[M, rows] = X[M, cols] · W[rows, cols]^T
void gemm(const Matrix &w, const float *x, int m, float *y, GemvScratch &scratch);

// 批量矩阵乘：输出行在外层、M 在内层，权重只读一遍（prefill 用）
void gemm_par(const Matrix &w, const float *x, int m, float *y, GemvPoolScratch &pool);

// 具体实现（行区间版本）：act 非空时直接用它（已量化）
void gemv_scalar_range(const Matrix &w, const float *x, const QuantizedActivation *act, float *y, int r0,
                       int r1, GemvScratch &scratch);
void gemv_avx2_range(const Matrix &w, const float *x, const QuantizedActivation *act, float *y, int r0,
                     int r1, GemvScratch &scratch);

// 4x4 repack 单行点积（group 指向该 4 行组的起始字节，k 是组内行号）
float row_dot_q8_4x4(const uint8_t *group, int nb, int k, const QuantizedActivation &act);
float row_dot_q4_4x4(const uint8_t *group, int nb, int k, const QuantizedActivation &act);
float row_dot_q8_4x4_scalar(const uint8_t *group, int nb, int k, const QuantizedActivation &act);
float row_dot_q4_4x4_scalar(const uint8_t *group, int nb, int k, const QuantizedActivation &act);

// 单行点积（gemm_par 用）
float row_dot_q8(const BlockQ8_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                 const float *w_scales);
float row_dot_q4(const BlockQ4_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                 const float *w_scales);
float row_dot_q8_scalar(const BlockQ8_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                        const float *w_scales);
float row_dot_q4_scalar(const BlockQ4_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                        const float *w_scales);
float row_dot_q8_avx2(const BlockQ8_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                      const float *w_scales);
float row_dot_q4_avx2(const BlockQ4_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                      const float *w_scales);

}  // namespace moss

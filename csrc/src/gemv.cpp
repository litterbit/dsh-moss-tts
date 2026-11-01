#include "moss/gemv.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/fp16.h"
#include "moss/parallel.h"
#include "moss/quant.h"

namespace moss {
namespace {

inline float scale_or(const float *table, int index, uint16_t half_bits) {
    return table ? table[index] : half_to_float(half_bits);
}

}  // namespace

float row_dot_q8_scalar(const BlockQ8_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                        const float *w_scales) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t acc = 0;
        for (int i = 0; i < 32; ++i) acc += static_cast<int32_t>(row[b].qs[i]) * static_cast<int32_t>(act[b].qs[i]);
        sum += static_cast<float>(acc) * scale_or(w_scales, b, row[b].d) * scale_or(act_scales, b, act[b].d);
    }
    return sum;
}

float row_dot_q4_scalar(const BlockQ4_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                        const float *w_scales) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t acc = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte = row[b].qs[j];
            acc += (static_cast<int>(byte & 0x0F) - 8) * static_cast<int>(act[b].qs[2 * j]) +
                   (static_cast<int>(byte >> 4) - 8) * static_cast<int>(act[b].qs[2 * j + 1]);
        }
        sum += static_cast<float>(acc) * scale_or(w_scales, b, row[b].d) * scale_or(act_scales, b, act[b].d);
    }
    return sum;
}

float row_dot_q8(const BlockQ8_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                 const float *w_scales) {
    return cpu_features().avx2 ? row_dot_q8_avx2(row, nb, act, act_scales, w_scales)
                               : row_dot_q8_scalar(row, nb, act, act_scales, w_scales);
}

float row_dot_q4(const BlockQ4_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                 const float *w_scales) {
    return cpu_features().avx2 ? row_dot_q4_avx2(row, nb, act, act_scales, w_scales)
                               : row_dot_q4_scalar(row, nb, act, act_scales, w_scales);
}


// 4x4 repack 的标量路径（非 AVX2 机器兜底）
float row_dot_q8_4x4_scalar(const uint8_t *group, int nb, int k, const QuantizedActivation &act) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 136;
        uint16_t dh;
        std::memcpy(&dh, blk + k * 2, 2);
        const float dw = half_to_float(dh);
        int32_t acc = 0;
        const int8_t *qs = reinterpret_cast<const int8_t *>(blk + 8 + static_cast<size_t>(k) * 32);
        for (int i = 0; i < 32; ++i) acc += static_cast<int32_t>(qs[i]) * static_cast<int32_t>(act.blocks[b].qs[i]);
        sum += static_cast<float>(acc) * dw * act.scales[b];
    }
    return sum;
}

float row_dot_q4_4x4_scalar(const uint8_t *group, int nb, int k, const QuantizedActivation &act) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 72;
        uint16_t dh;
        std::memcpy(&dh, blk + k * 2, 2);
        const float dw = half_to_float(dh);
        const uint8_t *qs = blk + 8 + static_cast<size_t>(k) * 16;
        int32_t acc = 0;
        for (int j = 0; j < 16; ++j) {
            acc += (static_cast<int>(qs[j] & 0x0F) - 8) * static_cast<int>(act.blocks[b].qs[2 * j]) +
                   (static_cast<int>(qs[j] >> 4) - 8) * static_cast<int>(act.blocks[b].qs[2 * j + 1]);
        }
        sum += static_cast<float>(acc) * dw * act.scales[b];
    }
    return sum;
}
void gemv_scalar_range(const Matrix &w, const float *x, const QuantizedActivation *act, float *y, int r0,
                       int r1, GemvScratch &scratch) {
    const int cols = w.cols;
    switch (w.dtype) {
        case DType::F32: {
            const float *base = static_cast<const float *>(w.data);
            for (int r = r0; r < r1; ++r) {
                const float *row = base + static_cast<size_t>(r) * cols;
                float acc = 0.0f;
                for (int c = 0; c < cols; ++c) acc += row[c] * x[c];
                y[r] = acc;
            }
            break;
        }
        case DType::F16: {
            const uint16_t *base = static_cast<const uint16_t *>(w.data);
            for (int r = r0; r < r1; ++r) {
                const uint16_t *row = base + static_cast<size_t>(r) * cols;
                float acc = 0.0f;
                for (int c = 0; c < cols; ++c) acc += half_to_float(row[c]) * x[c];
                y[r] = acc;
            }
            break;
        }
        case DType::Q8_0: {
            const BlockQ8_0 *a = act ? act->blocks.data() : scratch.act.blocks.data();
            const float *as = act ? act->scales.data() : scratch.act.scales.data();
            const BlockQ8_0 *base = static_cast<const BlockQ8_0 *>(w.data);
            const int nb = cols / 32;
            for (int r = r0; r < r1; ++r) {
                y[r] = row_dot_q8_scalar(base + static_cast<size_t>(r) * nb, nb, a, as,
                                         w.scales ? w.scales + static_cast<size_t>(r) * nb : nullptr);
            }
            break;
        }
        case DType::Q4_0: {
            const BlockQ8_0 *a = act ? act->blocks.data() : scratch.act.blocks.data();
            const float *as = act ? act->scales.data() : scratch.act.scales.data();
            const BlockQ4_0 *base = static_cast<const BlockQ4_0 *>(w.data);
            const int nb = cols / 32;
            for (int r = r0; r < r1; ++r) {
                y[r] = row_dot_q4_scalar(base + static_cast<size_t>(r) * nb, nb, a, as,
                                         w.scales ? w.scales + static_cast<size_t>(r) * nb : nullptr);
            }
            break;
        }
        case DType::Q8_0_4X4:
        case DType::Q4_0_4X4: {
            const QuantizedActivation &A = act ? *act : scratch.act;
            const bool q8 = w.dtype == DType::Q8_0_4X4;
            const int nb = cols / 32;
            const size_t bpb = q8 ? 136 : 72;
            const uint8_t *base = static_cast<const uint8_t *>(w.data);
            for (int r = r0; r < r1; ++r) {
                const uint8_t *grp = base + static_cast<size_t>(r / 4) * nb * bpb;
                y[r] = q8 ? row_dot_q8_4x4_scalar(grp, nb, r % 4, A) : row_dot_q4_4x4_scalar(grp, nb, r % 4, A);
            }
            break;
        }
    }
}

namespace {

inline void prepare_activation(const Matrix &w, const float *x, GemvScratch &scratch) {
    quantize_activation(x, scratch.act, w.cols);
}

// 池未绑定时（测试/单线程场景）退化为串行，语义一致
inline void run_tasks(GemvPoolScratch &pool, int items, const std::function<void(int)> &body) {
    if (pool.par) {
        pool.par->run(items, body);
        return;
    }
    for (int i = 0; i < items; ++i) body(i);
}

}  // namespace

void gemv(const Matrix &w, const float *x, float *y, GemvScratch &scratch) {
    QuantizedActivation *act = nullptr;
    const bool quantized = w.dtype == DType::Q8_0 || w.dtype == DType::Q4_0 ||
                           w.dtype == DType::Q8_0_4X4 || w.dtype == DType::Q4_0_4X4;
    if (quantized) {
        prepare_activation(w, x, scratch);
        act = &scratch.act;
    }
    if (cpu_features().avx2) {
        gemv_avx2_range(w, x, act, y, 0, w.rows, scratch);
    } else {
        gemv_scalar_range(w, x, act, y, 0, w.rows, scratch);
    }
}

void gemv_par(const Matrix &w, const float *x, float *y, GemvPoolScratch &pool) {
    const int n = pool.threads();
    if (n <= 1 || w.rows < 256) {
        gemv(w, x, y, pool.scratch(0));
        return;
    }

    const bool quantized = w.dtype == DType::Q8_0 || w.dtype == DType::Q4_0 ||
                           w.dtype == DType::Q8_0_4X4 || w.dtype == DType::Q4_0_4X4;
    const QuantizedActivation *act = nullptr;
    if (quantized) {
        quantize_activation(x, pool.shared, w.cols);
        act = &pool.shared;
    }

    const bool avx2 = cpu_features().avx2;
    run_tasks(pool, n, [&](int t) {
        const int per = (w.rows + n - 1) / n;
        const int r0 = t * per;
        const int r1 = std::min(w.rows, r0 + per);
        if (r0 >= r1) return;
        GemvScratch &s = pool.scratch(t);
        if (avx2) {
            gemv_avx2_range(w, x, act, y, r0, r1, s);
        } else {
            gemv_scalar_range(w, x, act, y, r0, r1, s);
        }
    });
}

void gemm_par(const Matrix &w, const float *x, int m, float *y, GemvPoolScratch &pool) {
    const int cols = w.cols;
    const int nb = cols / 32;
    const int n = std::max(1, pool.threads());

    if (w.dtype == DType::Q8_0_4X4 || w.dtype == DType::Q4_0_4X4) {
        // prefill：每行激活单独准备（含 even/odd 拆分），按 4 行组切分
        const bool q8 = w.dtype == DType::Q8_0_4X4;
        const size_t bpb = q8 ? 136 : 72;
        std::vector<QuantizedActivation> acts(m);
        for (int i = 0; i < m; ++i) quantize_activation(x + static_cast<size_t>(i) * cols, acts[i], cols);
        const uint8_t *base = static_cast<const uint8_t *>(w.data);
        run_tasks(pool, n, [&](int t) {
            const int groups = w.rows / 4;
            const int per = (groups + n - 1) / n;
            const int g0 = t * per;
            const int g1 = std::min(groups, g0 + per);
            for (int g = g0; g < g1; ++g) {
                const uint8_t *grp = base + static_cast<size_t>(g) * nb * bpb;
                for (int i = 0; i < m; ++i) {
                    for (int k = 0; k < 4; ++k) {
                        const float v = q8 ? row_dot_q8_4x4(grp, nb, k, acts[i])
                                           : row_dot_q4_4x4(grp, nb, k, acts[i]);
                        y[static_cast<size_t>(i) * w.rows + g * 4 + k] = v;
                    }
                }
            }
        });
        return;
    }

    if (w.dtype == DType::Q8_0 || w.dtype == DType::Q4_0) {
        pool.batch.resize(static_cast<size_t>(m) * nb);
        pool.batch_scales.resize(static_cast<size_t>(m) * nb);
        for (int i = 0; i < m; ++i) {
            quantize_row_q8_0_scaled(
                x + static_cast<size_t>(i) * cols, pool.batch.data() + static_cast<size_t>(i) * nb,
                pool.batch_scales.data() + static_cast<size_t>(i) * nb, cols);
        }
        const bool q8 = w.dtype == DType::Q8_0;
        run_tasks(pool, n, [&](int t) {
            const int per = (w.rows + n - 1) / n;
            const int r0 = t * per;
            const int r1 = std::min(w.rows, r0 + per);
            for (int r = r0; r < r1; ++r) {
                const float *ws = w.scales ? w.scales + static_cast<size_t>(r) * nb : nullptr;
                for (int i = 0; i < m; ++i) {
                    const BlockQ8_0 *act = pool.batch.data() + static_cast<size_t>(i) * nb;
                    const float *as = pool.batch_scales.data() + static_cast<size_t>(i) * nb;
                    const float v =
                        q8 ? row_dot_q8(static_cast<const BlockQ8_0 *>(w.data) + static_cast<size_t>(r) * nb, nb, act,
                                        as, ws)
                           : row_dot_q4(static_cast<const BlockQ4_0 *>(w.data) + static_cast<size_t>(r) * nb, nb, act,
                                        as, ws);
                    y[static_cast<size_t>(i) * w.rows + r] = v;
                }
            }
        });
        return;
    }

    std::vector<std::vector<float>> wbuf(n, std::vector<float>(cols));
    run_tasks(pool, n, [&](int t) {
        const int per = (w.rows + n - 1) / n;
        const int r0 = t * per;
        const int r1 = std::min(w.rows, r0 + per);
        std::vector<float> &row = wbuf[t];
        for (int r = r0; r < r1; ++r) {
            if (w.dtype == DType::F16) {
                const uint16_t *src = static_cast<const uint16_t *>(w.data) + static_cast<size_t>(r) * cols;
                for (int c = 0; c < cols; ++c) row[c] = half_to_float(src[c]);
            } else {
                std::memcpy(row.data(), static_cast<const float *>(w.data) + static_cast<size_t>(r) * cols,
                            sizeof(float) * cols);
            }
            for (int i = 0; i < m; ++i) {
                const float *xi = x + static_cast<size_t>(i) * cols;
                float acc = 0.0f;
                for (int c = 0; c < cols; ++c) acc += row[c] * xi[c];
                y[static_cast<size_t>(i) * w.rows + r] = acc;
            }
        }
    });
}

void gemm(const Matrix &w, const float *x, int m, float *y, GemvScratch &scratch) {
    for (int i = 0; i < m; ++i) {
        gemv(w, x + static_cast<size_t>(i) * w.cols, y + static_cast<size_t>(i) * w.rows, scratch);
    }
}

}  // namespace moss

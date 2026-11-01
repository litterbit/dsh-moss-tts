#pragma once

#include "moss/common.h"

namespace moss {

// 把一行 fp32 激活量化成 Q8_0 块（k 必须是 32 的倍数）
void quantize_row_q8_0(const float *x, BlockQ8_0 *blocks, int k);

// 同上，额外把每块的 scale 展开成 fp32（内核热路径用）
void quantize_row_q8_0_scaled(const float *x, BlockQ8_0 *blocks, float *scales, int k);

// 4x4 repack 布局的单行反量化（group 指向该 4 行组起始，k 是组内行号）
void dequantize_row_q8_0_4x4(const uint8_t *group, int nb, int k, float *out);
void dequantize_row_q4_0_4x4(const uint8_t *group, int nb, int k, float *out);

// 完整量化：blocks + scales + even/odd 拆分（Q4 内核用）
class QuantizedActivation;
void quantize_activation(const float *x, QuantizedActivation &act, int k);

void dequantize_row_q8_0(const BlockQ8_0 *blocks, float *x, int k);
void dequantize_row_q4_0(const BlockQ4_0 *blocks, float *x, int k);

// 标量参考点积（用于对拍与不支持 AVX2 的机器）
// 返回 sum_i w_i * a_i，其中 w/a 都是 Q8_0 块
float vec_dot_q8_0_q8_0(int nb, const BlockQ8_0 *w, const BlockQ8_0 *a);

}  // namespace moss

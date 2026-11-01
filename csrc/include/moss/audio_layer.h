#pragma once

#include <vector>

#include "moss/common.h"
#include "moss/gemv.h"
#include "moss/parallel.h"

namespace moss {

// MOSS-Audio-Tokenizer 的一层：
//   x += ls1 * attn(LN1(x));  x += ls2 * ffn(LN2(x))
// 解码器与编码器完全同构，差别只有窗口大小与张量尺寸，所以抽出来共用。
struct AudioLayer {
    Matrix attn_w, proj_w, f1_w, f2_w;
    Vector n1w, n1b, n2w, n2b, ls1, ls2;
};

// 逐层复用的缓冲（按层调用时避免反复分配）
struct AudioLayerBuffers {
    std::vector<float> ln, qkv, attn, res, ffn;
};

void audio_layernorm(const float *x, const Vector &w, const Vector &b, int n, float *out);
// interleaved RoPE，绝对位置
void audio_rope(float *vec, int pos, int heads, int head_dim, float max_period = 10000.0f);
float audio_gelu(float v);

// 一层前向（因果 + 窗口注意力），x 布局 [t][c]，原地更新
void audio_layer_forward(const AudioLayer &layer, std::vector<float> &x, int t, int c, int window,
                         Parallel &par, GemvPoolScratch &pool, AudioLayerBuffers &buf);

}  // namespace moss

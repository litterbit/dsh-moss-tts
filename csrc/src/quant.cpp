#include "moss/quant.h"

#include <algorithm>
#include <cmath>

#include "moss/fp16.h"

namespace moss {

void quantize_row_q8_0(const float *x, BlockQ8_0 *blocks, int k) {
    const int nb = k / 32;
    for (int b = 0; b < nb; ++b) {
        const float *src = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[i]));

        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        blocks[b].d = float_to_half(d);
        for (int i = 0; i < 32; ++i) {
            const int q = static_cast<int>(std::lrintf(src[i] * id));
            blocks[b].qs[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
        }
    }
}


void quantize_row_q8_0_scaled(const float *x, BlockQ8_0 *blocks, float *scales, int k) {
    const int nb = k / 32;
    for (int b = 0; b < nb; ++b) {
        const float *src = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[i]));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        blocks[b].d = float_to_half(d);
        scales[b] = d;
        for (int i = 0; i < 32; ++i) {
            const int q = static_cast<int>(std::lrintf(src[i] * id));
            blocks[b].qs[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
        }
    }
}

void quantize_activation(const float *x, QuantizedActivation &act, int k) {
    if (act.k != k) act.resize(k);
    quantize_row_q8_0_scaled(x, act.blocks.data(), act.scales.data(), k);
    for (int b = 0; b < k / 32; ++b) {
        for (int j = 0; j < 16; ++j) {
            act.even[static_cast<size_t>(b) * 16 + j] = act.blocks[b].qs[2 * j];
            act.odd[static_cast<size_t>(b) * 16 + j] = act.blocks[b].qs[2 * j + 1];
        }
    }
}

void dequantize_row_q8_0_4x4(const uint8_t *group, int nb, int k, float *out) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 136;
        uint16_t dh;
        std::memcpy(&dh, blk + k * 2, 2);
        const float d = half_to_float(dh);
        const int8_t *qs = reinterpret_cast<const int8_t *>(blk + 8 + static_cast<size_t>(k) * 32);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = d * static_cast<float>(qs[i]);
    }
}

void dequantize_row_q4_0_4x4(const uint8_t *group, int nb, int k, float *out) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 72;
        uint16_t dh;
        std::memcpy(&dh, blk + k * 2, 2);
        const float d = half_to_float(dh);
        const uint8_t *qs = blk + 8 + static_cast<size_t>(k) * 16;
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + 2 * j] = d * static_cast<float>(static_cast<int>(qs[j] & 0x0F) - 8);
            out[b * 32 + 2 * j + 1] = d * static_cast<float>(static_cast<int>(qs[j] >> 4) - 8);
        }
    }
}
void dequantize_row_q8_0(const BlockQ8_0 *blocks, float *x, int k) {
    const int nb = k / 32;
    for (int b = 0; b < nb; ++b) {
        const float d = half_to_float(blocks[b].d);
        for (int i = 0; i < 32; ++i) x[b * 32 + i] = d * static_cast<float>(blocks[b].qs[i]);
    }
}

void dequantize_row_q4_0(const BlockQ4_0 *blocks, float *x, int k) {
    const int nb = k / 32;
    for (int b = 0; b < nb; ++b) {
        const float d = half_to_float(blocks[b].d);
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte = blocks[b].qs[j];
            x[b * 32 + 2 * j] = d * static_cast<float>(static_cast<int>(byte & 0x0F) - 8);
            x[b * 32 + 2 * j + 1] = d * static_cast<float>(static_cast<int>(byte >> 4) - 8);
        }
    }
}

float vec_dot_q8_0_q8_0(int nb, const BlockQ8_0 *w, const BlockQ8_0 *a) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t acc = 0;
        for (int i = 0; i < 32; ++i) acc += static_cast<int32_t>(w[b].qs[i]) * static_cast<int32_t>(a[b].qs[i]);
        sum += static_cast<float>(acc) * half_to_float(w[b].d) * half_to_float(a[b].d);
    }
    return sum;
}

}  // namespace moss

// AVX2 内核：Q8_0 / Q4_0 的 W8A8 点积、fp16 权重路径。
// 只在本文件启用 /arch:AVX2；调用前必须确认 cpu_features().avx2。
#include "moss/gemv.h"

#include <immintrin.h>

#include <algorithm>

#include "moss/fp16.h"
#include "moss/quant.h"

namespace moss {
namespace {

inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// 一个 32 元素块：Q8_0 权重 × Q8 激活（u8×i8 用符号技巧）
inline __m256i dot_q8_block_i32(const BlockQ8_0 *w, const BlockQ8_0 *a) {
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w->qs));
    const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a->qs));
    const __m256i abs_w = _mm256_sign_epi8(wv, wv);
    const __m256i sgn_a = _mm256_sign_epi8(av, wv);
    const __m256i p16 = _mm256_maddubs_epi16(abs_w, sgn_a);
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

inline int32_t dot_q4_block(const BlockQ4_0 *w, const BlockQ8_0 *a) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w->qs));
    const __m256i w16 = _mm256_cvtepu8_epi16(packed);
    const __m256i lo = _mm256_sub_epi16(_mm256_and_si256(w16, _mm256_set1_epi16(0x0F)), _mm256_set1_epi16(8));
    const __m256i hi = _mm256_sub_epi16(_mm256_srli_epi16(w16, 4), _mm256_set1_epi16(8));

    const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a->qs));
    const __m256i shuf_even = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
                                               0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i shuf_odd = _mm256_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                              1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i ev = _mm256_shuffle_epi8(av, shuf_even);
    const __m256i od = _mm256_shuffle_epi8(av, shuf_odd);
    const __m128i ev16 = _mm_unpacklo_epi64(_mm256_castsi256_si128(ev), _mm256_extracti128_si256(ev, 1));
    const __m128i od16 = _mm_unpacklo_epi64(_mm256_castsi256_si128(od), _mm256_extracti128_si256(od, 1));
    const __m256i a_even = _mm256_cvtepi8_epi16(ev16);
    const __m256i a_odd = _mm256_cvtepi8_epi16(od16);

    const __m256i p32 = _mm256_add_epi32(_mm256_madd_epi16(lo, a_even), _mm256_madd_epi16(hi, a_odd));
    return hsum_epi32(p32);
}


inline int32_t dot_q4_block_split(const BlockQ4_0 *w, const int16_t *even, const int16_t *odd) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w->qs));
    const __m256i w16 = _mm256_cvtepu8_epi16(packed);
    const __m256i lo = _mm256_sub_epi16(_mm256_and_si256(w16, _mm256_set1_epi16(0x0F)), _mm256_set1_epi16(8));
    const __m256i hi = _mm256_sub_epi16(_mm256_srli_epi16(w16, 4), _mm256_set1_epi16(8));
    const __m256i ae = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(even));
    const __m256i ao = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(odd));
    const __m256i p32 = _mm256_add_epi32(_mm256_madd_epi16(lo, ae), _mm256_madd_epi16(hi, ao));
    return hsum_epi32(p32);
}
inline float scale_or(const float *table, int index, uint16_t half_bits) {
    return table ? table[index] : half_to_float(half_bits);
}

}  // namespace

float row_dot_q8_avx2(const BlockQ8_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                      const float *w_scales) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        sum += static_cast<float>(hsum_epi32(dot_q8_block_i32(row + b, act + b))) * scale_or(w_scales, b, row[b].d) *
               scale_or(act_scales, b, act[b].d);
    }
    return sum;
}

float row_dot_q4_avx2(const BlockQ4_0 *row, int nb, const BlockQ8_0 *act, const float *act_scales,
                      const float *w_scales) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        sum += static_cast<float>(dot_q4_block(row + b, act + b)) * scale_or(w_scales, b, row[b].d) *
               scale_or(act_scales, b, act[b].d);
    }
    return sum;
}


// ---------------- 4x4 repack 内核：4 行共享激活与 scale 载入 ----------------

float row_dot_q8_4x4(const uint8_t *group, int nb, int k, const QuantizedActivation &act) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 136;
        const __m128i dh = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blk));
        float dw[4];
        _mm_storeu_ps(dw, _mm_cvtph_ps(dh));
        const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 8 + static_cast<size_t>(k) * 32));
        const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(act.blocks[b].qs));
        const __m256i abs_w = _mm256_sign_epi8(qv, qv);
        const __m256i sgn_a = _mm256_sign_epi8(av, qv);
        const __m256i p32 =
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_w, sgn_a), _mm256_set1_epi16(1));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(dw[k] * act.scales[b]), acc);
    }
    return hsum_ps(acc);
}

float row_dot_q8_4x4_actf32(const uint8_t *group, int nb, int k, const float *act) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 136;
        const __m128i dh = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blk));
        float dw[4];
        _mm_storeu_ps(dw, _mm_cvtph_ps(dh));
        const int8_t *qs = reinterpret_cast<const int8_t *>(blk + 8 + static_cast<size_t>(k) * 32);
        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs + 16));
        const __m256i w0 = _mm256_cvtepi8_epi16(q0);
        const __m256i w1 = _mm256_cvtepi8_epi16(q1);
        const float *a = act + static_cast<size_t>(b) * 32;
        __m256 acc = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w0))), _mm256_loadu_ps(a));
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w0, 1))),
            _mm256_loadu_ps(a + 8), acc);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w1))),
            _mm256_loadu_ps(a + 16), acc);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w1, 1))),
            _mm256_loadu_ps(a + 24), acc);
        sum += hsum_ps(acc) * dw[k];
    }
    return sum;
}

float row_dot_q4_4x4(const uint8_t *group, int nb, int k, const QuantizedActivation &act) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = group + static_cast<size_t>(b) * 72;
        const __m128i dh = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blk));
        float dw[4];
        _mm_storeu_ps(dw, _mm_cvtph_ps(dh));
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 8 + static_cast<size_t>(k) * 16));
        const __m256i w16 = _mm256_cvtepu8_epi16(packed);
        const __m256i lo = _mm256_sub_epi16(_mm256_and_si256(w16, _mm256_set1_epi16(0x0F)), _mm256_set1_epi16(8));
        const __m256i hi = _mm256_sub_epi16(_mm256_srli_epi16(w16, 4), _mm256_set1_epi16(8));
        const __m256i ae = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(act.even.data() + static_cast<size_t>(b) * 16));
        const __m256i ao = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(act.odd.data() + static_cast<size_t>(b) * 16));
        const __m256i p32 = _mm256_add_epi32(_mm256_madd_epi16(lo, ae), _mm256_madd_epi16(hi, ao));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(dw[k] * act.scales[b]), acc);
    }
    return hsum_ps(acc);
}
void gemv_avx2_range(const Matrix &w, const float *x, const QuantizedActivation *act, float *y, int r0, int r1,
                     GemvScratch &scratch) {
    // 4x4 repack：4 行一组，组内共享 scale/激活载入
    if (w.dtype == DType::Q8_0_4X4 || w.dtype == DType::Q4_0_4X4) {
        const QuantizedActivation &A = act ? *act : scratch.act;
        const int nb = w.cols / 32;
        const bool q8 = w.dtype == DType::Q8_0_4X4;
        const size_t bpb = q8 ? 136 : 72;
        const uint8_t *base = static_cast<const uint8_t *>(w.data);
        for (int r = r0; r < r1; ++r) {
            const uint8_t *grp = base + static_cast<size_t>(r / 4) * nb * bpb;
            if (q8 && act16_enabled()) {
                y[r] = row_dot_q8_4x4_actf32(grp, nb, r % 4, x);
            } else {
                y[r] = q8 ? row_dot_q8_4x4(grp, nb, r % 4, A) : row_dot_q4_4x4(grp, nb, r % 4, A);
            }
        }
        return;
    }
    const int cols = w.cols;

    if (w.dtype == DType::F32) {
        const float *base = static_cast<const float *>(w.data);
        for (int r = r0; r < r1; ++r) {
            const float *row = base + static_cast<size_t>(r) * cols;
            __m256 acc = _mm256_setzero_ps();
            int c = 0;
            for (; c + 8 <= cols; c += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(x + c), acc);
            float tail = 0.0f;
            for (; c < cols; ++c) tail += row[c] * x[c];
            y[r] = hsum_ps(acc) + tail;
        }
        return;
    }

    if (w.dtype == DType::F16) {
        const uint16_t *base = static_cast<const uint16_t *>(w.data);
        for (int r = r0; r < r1; ++r) {
            const uint16_t *row = base + static_cast<size_t>(r) * cols;
            __m256 acc = _mm256_setzero_ps();
            int c = 0;
            for (; c + 8 <= cols; c += 8) {
                const __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(row + c)));
                acc = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c), acc);
            }
            float tail = 0.0f;
            for (; c < cols; ++c) tail += half_to_float(row[c]) * x[c];
            y[r] = hsum_ps(acc) + tail;
        }
        return;
    }

    const BlockQ8_0 *a = act ? act->blocks.data() : scratch.act.blocks.data();
    const float *as = act ? act->scales.data() : scratch.act.scales.data();
    const int nb = cols / 32;

    if (w.dtype == DType::Q8_0) {
        const BlockQ8_0 *base = static_cast<const BlockQ8_0 *>(w.data);
        int r = r0;
        for (; r + 4 <= r1; r += 4) {
            const float *ws = w.scales ? w.scales + static_cast<size_t>(r) * nb : nullptr;
            float acc[4] = {0, 0, 0, 0};
            for (int b = 0; b < nb; ++b) {
                const float da = as[b];
                for (int k = 0; k < 4; ++k) {
                    const BlockQ8_0 *wk = base + (static_cast<size_t>(r + k)) * nb + b;
                    acc[k] += static_cast<float>(hsum_epi32(dot_q8_block_i32(wk, a + b))) *
                              scale_or(ws ? ws + static_cast<size_t>(k) * nb : nullptr, b, wk->d) * da;
                }
            }
            y[r + 0] = acc[0];
            y[r + 1] = acc[1];
            y[r + 2] = acc[2];
            y[r + 3] = acc[3];
        }
        for (; r < r1; ++r) {
            const BlockQ8_0 *row = base + static_cast<size_t>(r) * nb;
            y[r] = row_dot_q8_avx2(row, nb, a, as, w.scales ? w.scales + static_cast<size_t>(r) * nb : nullptr);
        }
        return;
    }

    // Q4_0 权重 × Q8 激活（W4A8）：激活已按 even/odd 拆好，免每行 shuffle
    const bool has_split = act != nullptr && act->even.size() == static_cast<size_t>(cols);
    const BlockQ4_0 *base = static_cast<const BlockQ4_0 *>(w.data);
    for (int r = r0; r < r1; ++r) {
        const BlockQ4_0 *row = base + static_cast<size_t>(r) * nb;
        const float *ws = w.scales ? w.scales + static_cast<size_t>(r) * nb : nullptr;
        if (has_split) {
            float sum = 0.0f;
            for (int b = 0; b < nb; ++b) {
                sum += static_cast<float>(dot_q4_block_split(row + b, act->even.data() + static_cast<size_t>(b) * 16,
                                                             act->odd.data() + static_cast<size_t>(b) * 16)) *
                       scale_or(ws, b, row[b].d) * scale_or(as, b, (act->blocks[b]).d);
            }
            y[r] = sum;
        } else {
            y[r] = row_dot_q4_avx2(row, nb, a, as, ws);
        }
    }
}

}  // namespace moss

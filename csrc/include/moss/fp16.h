#pragma once

#include <cstdint>

namespace moss {

// IEEE half <-> float。软实现（不依赖 F16C），保证任何 x64 都能用；
// 热路径（AVX2 GEMV）里改用 _mm256_cvtph_ps。
float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

}  // namespace moss

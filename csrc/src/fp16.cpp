#include "moss/fp16.h"

#include <cstring>

namespace moss {

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // ±0
        } else {
            // 次正规数：规整化
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (((bits >> 23) & 0xFF) == 0xFF) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));  // inf / nan
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);  // 下溢为 0
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) ++half_mant;  // 就近舍入
        return static_cast<uint16_t>(sign | half_mant);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // 上溢为 inf

    uint32_t half_mant = mant >> 13;
    if (mant & 0x1000u) ++half_mant;
    if (half_mant == 0x400u) {
        half_mant = 0;
        ++exp;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_mant);
}

}  // namespace moss

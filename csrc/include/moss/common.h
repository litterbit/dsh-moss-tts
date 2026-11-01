#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace moss {

[[noreturn]] inline void fail(const std::string &msg) { throw std::runtime_error(msg); }

// GGML 对齐的块格式：32 个权重一组
struct BlockQ8_0 {
    uint16_t d;      // fp16 scale
    int8_t qs[32];   // 量化值 -127..127
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

struct BlockQ4_0 {
    uint16_t d;      // fp16 scale
    uint8_t qs[16];  // 两个 4bit nibble 一个字节
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

enum class DType : uint8_t { F16 = 0, Q8_0 = 1, Q4_0 = 2, F32 = 3, Q8_0_4X4 = 4, Q4_0_4X4 = 5 };

inline const char *dtype_name(DType t) {
    switch (t) {
        case DType::F16: return "f16";
        case DType::Q8_0: return "q8_0";
        case DType::Q4_0: return "q4_0";
        case DType::F32: return "f32";
        case DType::Q8_0_4X4: return "q8_0_4x4";
        case DType::Q4_0_4X4: return "q4_0_4x4";
    }
    return "?";
}

// 量化后的激活：每 32 元素一块，块数 = K/32
struct QuantizedActivation {
    std::vector<BlockQ8_0> blocks;
    std::vector<float> scales;  // 每块的 scale 展开成 fp32，内核里不必再转
    std::vector<int16_t> even, odd;  // 按 nibble 通道拆好的激活（Q4 内核直用，免每行 shuffle）
    int k = 0;

    void resize(int k_elements) {
        k = k_elements;
        blocks.assign(static_cast<size_t>(k) / 32, BlockQ8_0{});
        scales.assign(static_cast<size_t>(k) / 32, 0.0f);
        even.assign(static_cast<size_t>(k), 0);
        odd.assign(static_cast<size_t>(k), 0);
    }
};

// 权重矩阵的只读视图：指向 .moss 文件映射的内存
struct Matrix {
    const void *data = nullptr;
    int rows = 0;
    int cols = 0;
    DType dtype = DType::F32;
    // 量化矩阵的块 scale（已展开为 fp32，长度 rows*cols/32），避免内核里反复做 fp16→fp32
    const float *scales = nullptr;

    size_t bytes() const {
        switch (dtype) {
            case DType::F16: return static_cast<size_t>(rows) * cols * 2;
            case DType::F32: return static_cast<size_t>(rows) * cols * 4;
            case DType::Q8_0: return static_cast<size_t>(rows) * (cols / 32) * sizeof(BlockQ8_0);
            case DType::Q4_0: return static_cast<size_t>(rows) * (cols / 32) * sizeof(BlockQ4_0);
            case DType::Q8_0_4X4: return static_cast<size_t>(rows / 4) * (cols / 32) * 136;
            case DType::Q4_0_4X4: return static_cast<size_t>(rows / 4) * (cols / 32) * 72;
        }
        return 0;
    }
};

// 向量（LN 权重、bias 等）：永远是 fp32
struct Vector {
    const float *data = nullptr;
    int size = 0;
};

}  // namespace moss

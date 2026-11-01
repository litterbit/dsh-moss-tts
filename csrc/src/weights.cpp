#include "moss/weights.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "moss/fp16.h"

namespace moss {
namespace {

constexpr size_t kHeaderSize = 16;
constexpr size_t kEntrySize = 80;
constexpr size_t kNameSize = 48;

struct RawEntry {
    std::string name;
    uint8_t dtype = 0;
    uint8_t ndim = 0;
    uint32_t dims[3] = {0, 0, 0};
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

}  // namespace

bool WeightFile::load(const std::string &path, std::string *err) {
    std::FILE *fh = std::fopen(path.c_str(), "rb");
    if (!fh) {
        if (err) *err = "无法打开权重文件: " + path;
        return false;
    }
    std::fseek(fh, 0, SEEK_END);
    const long size = std::ftell(fh);
    std::fseek(fh, 0, SEEK_SET);
    if (size <= static_cast<long>(kHeaderSize)) {
        std::fclose(fh);
        if (err) *err = "权重文件过小";
        return false;
    }
    data_.resize(static_cast<size_t>(size));
    const size_t read = std::fread(data_.data(), 1, data_.size(), fh);
    std::fclose(fh);
    if (read != data_.size()) {
        if (err) *err = "权重文件读取不完整";
        return false;
    }

    if (std::memcmp(data_.data(), "MOSW", 4) != 0) {
        if (err) *err = "magic 不匹配（不是 .moss 文件）";
        return false;
    }
    uint32_t version = 0, n_tensors = 0;
    std::memcpy(&version, data_.data() + 4, 4);
    std::memcpy(&n_tensors, data_.data() + 8, 4);
    if (version != 1) {
        if (err) *err = "不支持的权重格式版本";
        return false;
    }

    for (uint32_t i = 0; i < n_tensors; ++i) {
        const size_t base = kHeaderSize + static_cast<size_t>(i) * kEntrySize;
        RawEntry raw;
        raw.name.assign(reinterpret_cast<const char *>(data_.data() + base), kNameSize);
        raw.name = raw.name.c_str();  // 截到第一个 \0
        raw.dtype = data_[base + kNameSize];
        raw.ndim = data_[base + kNameSize + 1];
        std::memcpy(raw.dims, data_.data() + base + 52, sizeof(raw.dims));
        std::memcpy(&raw.offset, data_.data() + base + 64, 8);
        std::memcpy(&raw.nbytes, data_.data() + base + 72, 8);

        if (raw.offset + raw.nbytes > data_.size()) {
            if (err) *err = "张量越界: " + raw.name;
            return false;
        }

        const auto dtype = static_cast<DType>(raw.dtype);
        const void *ptr = data_.data() + raw.offset;
        if (raw.ndim == 1 || (raw.dims[0] == 1 && raw.dims[2] == 1)) {
            vectors_[raw.name] = Vector{static_cast<const float *>(ptr), static_cast<int>(raw.dims[0] * raw.dims[1])};
        } else {
            Matrix m;
            m.data = ptr;
            m.rows = static_cast<int>(raw.dims[0]);
            m.cols = static_cast<int>(raw.dims[1]);
            m.dtype = dtype;
            const int nb = m.cols / 32;
            if (dtype == DType::Q8_0 || dtype == DType::Q4_0) {
                std::vector<float> scales(static_cast<size_t>(m.rows) * nb);
                const int block_bytes = (dtype == DType::Q8_0) ? 34 : 18;
                for (int r = 0; r < m.rows; ++r) {
                    for (int b = 0; b < nb; ++b) {
                        uint16_t half_bits;
                        std::memcpy(&half_bits,
                                    static_cast<const uint8_t *>(ptr) +
                                        (static_cast<size_t>(r) * nb + b) * block_bytes,
                                    2);
                        scales[static_cast<size_t>(r) * nb + b] = half_to_float(half_bits);
                    }
                }
                scale_tables_.push_back(std::move(scales));
                m.scales = scale_tables_.back().data();
            } else if (dtype == DType::Q8_0_4X4 || dtype == DType::Q4_0_4X4) {
                // repack 布局：[group][block][4 个 scale 相邻]
                const int groups = m.rows / 4;
                const int block_bytes = (dtype == DType::Q8_0_4X4) ? 136 : 72;
                std::vector<float> scales(static_cast<size_t>(groups) * nb * 4);
                for (int g = 0; g < groups; ++g) {
                    for (int b = 0; b < nb; ++b) {
                        const uint8_t *blk = static_cast<const uint8_t *>(ptr) +
                                             (static_cast<size_t>(g) * nb + b) * block_bytes;
                        for (int k = 0; k < 4; ++k) {
                            uint16_t half_bits;
                            std::memcpy(&half_bits, blk + k * 2, 2);
                            scales[(static_cast<size_t>(g) * nb + b) * 4 + k] = half_to_float(half_bits);
                        }
                    }
                }
                scale_tables_.push_back(std::move(scales));
                m.scales = scale_tables_.back().data();
            }
            matrices_[raw.name] = m;
        }
    }
    return true;
}

const Matrix &WeightFile::matrix(const std::string &name) const {
    const auto it = matrices_.find(name);
    if (it == matrices_.end()) fail("缺少权重矩阵: " + name);
    return it->second;
}

const Vector &WeightFile::vector(const std::string &name) const {
    const auto it = vectors_.find(name);
    if (it == vectors_.end()) fail("缺少权重向量: " + name);
    return it->second;
}

bool WeightFile::has(const std::string &name) const {
    return matrices_.count(name) != 0 || vectors_.count(name) != 0;
}

}  // namespace moss

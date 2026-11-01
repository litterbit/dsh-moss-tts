#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "moss/common.h"

namespace moss {

// 读 .moss 文件：整份读进内存，Matrix/Vector 指向其中（地址稳定）
class WeightFile {
public:
    bool load(const std::string &path, std::string *err);

    const Matrix &matrix(const std::string &name) const;
    const Vector &vector(const std::string &name) const;
    bool has(const std::string &name) const;

    size_t bytes() const { return data_.size(); }

private:
    std::vector<uint8_t> data_;
    std::unordered_map<std::string, Matrix> matrices_;
    std::vector<std::vector<float>> scale_tables_;  // 量化矩阵的 fp32 块 scale
    std::unordered_map<std::string, Vector> vectors_;
};

}  // namespace moss

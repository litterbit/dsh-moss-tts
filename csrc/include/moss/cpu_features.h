#pragma once

#include <string>

namespace moss {

// 启动时探测一次，之后只读
struct CpuFeatures {
    bool sse2 = true;
    bool ssse3 = false;
    bool avx2 = false;
    bool f16c = false;
    bool avx_vnni = false;  // Alder Lake+ P-core
    int physical_cores = 1;
    int logical_cores = 1;
    std::string brand;

    // 选中的实现："avx2" / "scalar"
    const char *gemv_backend() const { return avx2 ? "avx2" : "scalar"; }
};

const CpuFeatures &cpu_features();

}  // namespace moss

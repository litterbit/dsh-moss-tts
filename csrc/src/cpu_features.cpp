#include "moss/cpu_features.h"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace moss {
namespace {

struct Regs {
    int eax, ebx, ecx, edx;
};

Regs cpuid(int leaf, int subleaf) {
    Regs r{0, 0, 0, 0};
#if defined(_MSC_VER)
    int out[4];
    __cpuidex(out, leaf, subleaf);
    r.eax = out[0];
    r.ebx = out[1];
    r.ecx = out[2];
    r.edx = out[3];
#else
    unsigned a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    r.eax = static_cast<int>(a);
    r.ebx = static_cast<int>(b);
    r.ecx = static_cast<int>(c);
    r.edx = static_cast<int>(d);
#endif
    return r;
}

CpuFeatures detect() {
    CpuFeatures f;
    const Regs leaf1 = cpuid(1, 0);
    f.ssse3 = (leaf1.ecx & (1 << 9)) != 0;
    f.avx2 = false;  // 需要再查 leaf 7
    f.f16c = (leaf1.ecx & (1 << 29)) != 0;

    const Regs leaf7 = cpuid(7, 0);
    f.avx2 = (leaf7.ebx & (1 << 5)) != 0;
    f.avx_vnni = (leaf7.eax & (1 << 4)) != 0;  // AVX-VNNI (256-bit)

    // 允许强制走标量路径（用于验证降级分支；生产环境用不到）
    if (std::getenv("MOSS_FORCE_SCALAR") != nullptr) {
        f.avx2 = false;
        f.f16c = false;
        f.avx_vnni = false;
    }

    // 品牌名
    char brand[49] = {0};
    const Regs b0 = cpuid(0x80000000u, 0);
    if (static_cast<unsigned>(b0.eax) >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) {
            const Regs r = cpuid(0x80000002 + i, 0);
            std::memcpy(brand + i * 16, &r.eax, 4);
            std::memcpy(brand + i * 16 + 4, &r.ebx, 4);
            std::memcpy(brand + i * 16 + 8, &r.ecx, 4);
            std::memcpy(brand + i * 16 + 12, &r.edx, 4);
        }
        // 去掉前导空格
        const char *p = brand;
        while (*p == ' ') ++p;
        f.brand = p;
    }
    return f;
}

}  // namespace

const CpuFeatures &cpu_features() {
    static const CpuFeatures features = [] {
        CpuFeatures f = detect();
        // 逻辑核数交给上层（std::thread::hardware_concurrency），这里只做保守估计
        f.logical_cores = 1;
        f.physical_cores = 1;
        return f;
    }();
    return features;
}

}  // namespace moss

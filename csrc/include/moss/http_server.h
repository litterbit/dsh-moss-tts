// 本地 HTTP/1.1 服务（Winsock，零第三方依赖）。只监听 127.0.0.1。
//
// 设计要点见 doc/产品化与分发方案.md：
//   - 传输层只做编解码，模型与任务状态都在 Runtime
//   - 流式 PCM 走 chunked；统计走 /jobs/{id}
//   - Host 白名单挡 DNS rebinding；Origin 精确匹配挡跨源读取
//   - 静态文件路径规范化后再校验，拒绝目录穿越
#pragma once

#include <string>
#include <vector>

namespace moss {

class Runtime;

struct ServerOptions {
    std::string web_dir;                 // 静态文件根目录（web/app）
    std::string host = "127.0.0.1";
    int port = 8759;
    int port_attempts = 5;               // 端口占用时顺延次数
    bool open_browser = true;
    int max_connections = 16;
    std::vector<std::string> allow_origins;  // 额外允许的精确 Origin
};

// 阻塞运行。启动成功后通过 out_url 返回实际访问地址。
bool serve_forever(Runtime &runtime, const ServerOptions &options, std::string *out_url, std::string *err);

}  // namespace moss

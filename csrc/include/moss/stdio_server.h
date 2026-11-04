// 进程内通道：stdin/stdout 上的 JSON 行协议，给 DSH 插件这类"自己管理子进程"的宿主用。
//
// 为什么要它：插件不需要开端口、不需要处理端口冲突、不需要猜服务是否已启动；
// 父进程 spawn `moss_tts.exe --stdio` 之后，一行请求、一行响应，子进程退出即资源释放。
// 与 HTTP 服务共享同一个 Runtime，所以请求语义、统计、取消完全一致。
//
// 协议（每行一个 JSON，UTF-8，\n 结尾）：
//   请求  {"id":"r1","cmd":"speak","text":"...","voice":"Junhao","seed":0,...}
//         {"id":"r2","cmd":"cancel","job":"<job id>"}
//         {"id":"r3","cmd":"voices"} / {"cmd":"info"} / {"cmd":"ping"}
//         {"id":"r4","cmd":"clone","name":"me","wav":"<base64>"}
//   响应  {"id":"r1","type":"begin","job":"<job id>","format":{...}}
//         {"id":"r1","type":"audio","pcm":"<base64 s16le>"}   // 每个解码块一行
//         {"id":"r1","type":"done","job":{...}}               // 状态与统计
//         {"id":"r1","type":"error","code":"...","error":"..."}
#pragma once

namespace moss {

class Runtime;

// 阻塞直到 stdin 关闭。返回 0 表示正常结束。
int run_stdio_server(Runtime &runtime);

}  // namespace moss

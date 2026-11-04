# 本地 HTTP 接口（API v1）

服务只监听 `127.0.0.1`，默认端口 8759（占用时顺延到 8760~8763，实际地址会打印在控制台）。
所有接口以 `/api/v1` 开头；v1 内向后兼容，破坏性变更走 v2。

## 1. 通用约定

| 项 | 约定 |
|---|---|
| 音频格式 | `s16le`，48000 Hz，2 声道交织（4 字节 / 帧） |
| 编码 | 请求 / 响应 JSON 一律 UTF-8 |
| 连接 | 每个请求一条连接（`Connection: close`），流式用 `Transfer-Encoding: chunked` |
| 认证 | 无 token。只靠"只绑 loopback + Host 白名单 + Origin 校验" |
| 版本 | 响应头 `X-Moss-Version`（引擎版本）、`X-Moss-Api-Version: v1` |
| 错误 | 非 2xx 返回 `{"ok":false,"code":"...","error":"中文可读信息"}` |

**安全模型**：`Host` 必须是 `127.0.0.1:<port>`、`localhost:<port>` 或 `[::1]:<port>`；
带 `Origin` 的请求必须精确匹配本服务自己的 origin（或启动时 `--allow-origin` 追加的项），
否则 403。客户端（Node 插件、curl、Python）不带 `Origin` 即可直接调用。

## 2. 端点

### `GET /api/v1/health`

```json
{"ok":true,"version":"0.1.0","api":"v1","queue":0,"max":4}
```

### `GET /api/v1/info`

```json
{"ok":true,"version":"0.1.0","api":"v1",
 "model":{"id":"moss-tts-nano-100m-q8x4","revision":"q8x4-f16-r1"},
 "engine":{"llm_threads":4,"decode_threads":6,"backend":"avx2","physical_cores":14},
 "queue":{"depth":0,"max":4}}
```

`backend` 为 `avx2` 或 `scalar`（老 CPU / `MOSS_FORCE_SCALAR=1`）。

### `GET /api/v1/voices`

```json
{"ok":true,
 "builtin":[{"id":"Junhao","frames":98,"seconds":7.84}],
 "user":[{"id":"me","frames":120,"seconds":9.6}]}
```

`user` 来自用户数据目录（`<data>/voices/user/`），与内置音色物理隔离，升级覆盖不了。

### `POST /api/v1/voices/clone?name=<id>`

请求体是 WAV 原始字节（`Content-Type: audio/wav`）。
`name` 只能包含字母、数字、`_`、`-`，长度 ≤ 32。
返回 `{"ok":true,"name":"me","builtin":[...],"user":[...]}`。
参考音频要求：PCM 16/24/32-bit 或 float32，任意采样率（内部重采样到 48 kHz），建议 5~15 秒。

### `POST /api/v1/tts/stream`

请求：

```json
{"text":"你好，世界。",
 "voice":"Junhao",        // 内置音色 id
 "scope":"builtin",       // "builtin" | "user"
 "seed":0,
 "greedy":false,
 "max_frames":375,        // 1~2400，1 帧 = 80 ms 音频
 "chunk_frames":8,        // 1~64，解码块；首块固定 4 帧
 "min_frames":0,
 "temperature":0.8,       // 0 = 默认
 "top_k":25,
 "top_p":0.95,
 "repetition_penalty":1.2}
```

响应：

```
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Transfer-Encoding: chunked
X-Moss-Request-Id: 5f43c1a200000003
X-Moss-Audio-Format: s16le;rate=48000;channels=2
```

body 是纯 PCM 分块流。**不要**把它当成 `audio/L16`（RFC 2586 的 L16 是 big-endian，这里是 little-endian）。
HTTP 的 chunk 边界不等于音频帧边界：客户端必须按 4 字节对齐累积再解释。

统计不混在音频里：流结束后用 `X-Moss-Request-Id` 查 `GET /api/v1/jobs/{id}`。

### `POST /api/v1/tts/wav`

同样的请求体，返回完整 WAV（带 `Content-Disposition`）。可选 `"trim":true` 裁掉尾部低于
峰值 28 dB 的低语（默认关闭）。

### `GET /api/v1/jobs/{id}`

```json
{"ok":true,"id":"5f43c1a200000003","state":"completed","frames":64,
 "audio_ms":5120,"compute_ms":764,"first_pcm_ms":98,"rtf":0.149,
 "text_tokens":12,"cache_hit":true,"error":""}
```

`state`：`queued | running | completed | cancelled | failed`。
`rtf = compute_ms / audio_ms`。任务记录保留 10 分钟 / 最近 32 条。

### `POST /api/v1/jobs/{id}/cancel`

立即置取消标志。引擎在下一个生成帧边界（约 7 ms）或下一个解码块边界停下；
prefill 期间（约 170 ms）不能中断。返回 `{"ok":true,"id":"..."}`。

### 静态资源

`GET /` → `web/index.html`；`GET /app.js`、`/pcm-worklet.js` 等。
路径规范化后必须在 `web/` 目录内，否则 400/403。

## 3. 并发与限额

- 单引擎实例、单工作线程，任务串行执行；队列深度默认 4，满了返回 `429 {"code":"busy"}`。
- 文本 ≤ 4000 字符；JSON body ≤ 256 KB；克隆上传 ≤ 20 MB。
- 单个请求最长 10 分钟（`max_frames` 2400 ≈ 3.2 分钟音频）。
- 客户端断开连接会自动取消该任务。

## 4. C ABI

见 `include/moss/tts.h`：`moss_create / moss_set_voice / moss_speak / moss_set_sampling /
moss_clone / moss_text_tokens / moss_last_error / moss_version`。链接 `lib/moss_tts.lib`
（静态 CRT，无第三方依赖）。`moss_speak` 的回调按块返回交织 int16 PCM。

## 5. stdio 通道（给插件用）

宿主自己管理子进程时，不需要开端口：

```bat
moss_tts.exe --models models --stdio
```

stdin/stdout 上每行一个 JSON（UTF-8，`\n` 结尾）。请求：

```json
{"id":"r1","cmd":"speak","text":"你好","voice":"Junhao","seed":0,"max_frames":375}
{"id":"r2","cmd":"cancel","job":"r1"}
{"id":"r3","cmd":"voices"}          // 也有 "info" / "ping"
{"id":"r4","cmd":"clone","name":"me","wav":"<base64 的 WAV>"}
```

响应（同一个 `id`）：

```json
{"id":"r1","type":"begin","job":"r1","format":{"encoding":"s16le","rate":48000,"channels":2}}
{"id":"r1","type":"audio","pcm":"<base64，一个解码块一行>"}
{"id":"r1","type":"done","job":{"state":"completed","frames":64,"audio_ms":5120,...}}
{"id":"r1","type":"error","code":"busy","error":"队列已满"}
```

- `begin` 一定在第一个 `audio` 之前发出；`done` 一定在所有 `audio` 之后。
- `speak` 是异步的：主线程继续读 stdin，所以合成过程中就能 `cancel`。
- 关闭 stdin 服务就退出（退出码 0）。
- 与 HTTP 完全同一套 Runtime：同一队列、同一取消语义、同一统计字段。

Python 侧参考实现与回归脚本：`tools/stdio_smoke.py`。

## 6. 命令行等价物

```bat
:: 起服务（默认打开浏览器）
moss_tts.exe
moss_tts.exe --serve --port 9000 --no-open --web web --models models

:: 单次合成
moss_tts.exe --text "你好" --voice Junhao --out a.wav
moss_tts.exe --text "你好" --pcm-raw - > a.pcm
moss_tts.exe --selftest
moss_tts.exe --version
```

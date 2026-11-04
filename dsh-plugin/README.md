# dsh-plugin — DeepSeek Harness 的 MOSS-TTS 插件

把本机 MOSS-TTS 引擎接成 DSH 里的语音工具。和网页试听台、命令行**共用同一个 `moss_tts.exe`**
（走 `--stdio` JSON 行协议），所以不存在"插件一套推理、网页另一套"的问题。

## 前置

装好发布包（`MossTTS-win-x64-full.zip`）即可，**不需要先启动服务**：插件自己 spawn 一个常驻子进程，
权重只加载一次，退出时回收。

## 安装

在 profile 的 `cordis.patch.yml` 里加一段（路径换成你自己的）：

```yaml
- insert:
    - id: moss-tts
      name: D:\workplace\audio\dsh-moss-tts\dsh-plugin\index.mjs
      config:
        exe: D:\MossTTS\moss_tts.exe
        models: D:\MossTTS\models
        voice: Junhao
        # data: D:\MossTTS\userdata        # 可选：缓存/用户音色的位置
        # llmThreads: 4                    # 可选：0 = 按物理核自动
        # decodeThreads: 6
```

`exe` / `models` 省略时会依次找：`MOSS_TTS_EXE` 环境变量 → `%LOCALAPPDATA%\MossTTS\` →
当前目录的 `moss_tts.exe` / `dist/moss-tts-cpu-win64/moss_tts.exe`。

## 工具

| 工具 | 作用 |
|---|---|
| `moss_tts_say` | 朗读一段文字；`wait: false` 立刻返回，`save_to: D:\a.wav` 只出文件不播放 |
| `moss_tts_voices` | 列出内置 + 自己克隆的音色 |
| `moss_tts_stop` | 立刻停掉当前合成/播放 |
| `moss_tts_clone` | 用 5~15 秒 WAV 创建克隆音色 |

## 实测（本机）

| 项 | 数值 |
|---|---|
| 启动 + 加载权重（一次性） | ~90 ms |
| 2.4 s 语音的合成 | 687 ms（RTF 0.287；短句固定开销占比高） |
| 5 s 语音 | ~0.76 s（RTF 0.15） |
| 流式分块 | 24 帧 = 4 块，字节数与帧数严格对应（帧 × 15360 B） |
| 取消 | 600 ms 处请求，返回 `cancelled`，不再继续吐音频 |

## 独立测试（不需要 DSH）

```bash
node dsh-plugin/test-backend.mjs     # 协议层：启动/voices/speak/取消/克隆/落 WAV/dispose
node dsh-plugin/test-plugin.mjs      # 插件入口：假 ctx 里注册工具并真的合成
```

## 已知限制

- 播放用 Windows 自带的 `System.Media.SoundPlayer`：**先合成完再播**，不是边收边播；
  合成 5 秒语音约 0.76 秒，听感上仍是"立等可听"。要毫秒级首音就得做流式播放（v2）。
- `moss_tts_say` 默认阻塞到播放结束（这样对话不会在朗读中途继续往下说）；
  批量场景传 `wait: false`。
- 还没做"助手回复自动朗读"：那需要 DSH 的消息事件钩子，属下一步；现在由模型/用户显式调用工具。
- 引擎升级后如果 `--stdio` 协议变了，`backend.mjs` 要跟着改（协议见 `doc/api.md` §5）。

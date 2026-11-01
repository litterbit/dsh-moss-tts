# vendor/moss_tts_nano_onnx — 只读参考实现

这里的文件是从上游 **原样复制** 的，**任何人不得修改**。

- 上游仓库：`https://github.com/OpenMOSS/MOSS-TTS-Nano`
- 本地副本：`D:\workplace\audio\github\MOSS-TTS-Nano`
- commit：`8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb`（2026-09-06）

## 复制了哪些文件

| 文件 | 作用 |
|---|---|
| `onnx_tts_runtime.py` | 面向应用的封装：内置音色、文本分块、写 WAV |
| `ort_cpu_runtime.py` | 真正的 ONNX 推理：8 个 session、帧循环、codec 流式解码（纯 numpy + onnxruntime，**不依赖 torch**） |
| `text_normalization_pipeline.py` | 文本清洗编排（WeTextProcessing 在 Windows 不可用，走 robust 分支） |
| `tts_robust_normalizer_single_script.py` | 纯标准库的鲁棒清洗 |
| `moss_tts_nano/defaults.py` | 默认输出目录等常量 |
| `moss_tts_nano/__init__.py` | 版本号 |

## 使用约定

- 只允许 `import`，不允许就地改。
- 需要改行为时：在 `src/dsh_moss_tts/` 里**重写一份自己的实现**，不要碰这里。
- 上游升级时整体覆盖本目录，并更新上面的 commit 号。

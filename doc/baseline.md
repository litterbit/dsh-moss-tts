# ONNX CPU baseline（2026-11-01）

## 结论

- 5.36 s 中文（67 帧 × 80 ms），48 kHz 双声道，音频数据正常（peak 0.85 / RMS 0.117）。
- 4 线程 CPU：模型加载 6.0 s，合成 **2.3 s**，**RTF 0.42~0.43**，35 ms/帧。
- 固定 seed 跑两次，67×16 codes 逐位一致（`deterministic: true`）。
- 首次运行合成 6.8 s；670 MB 的 `.data` 页缓存热了之后稳定 2.3 s。**对比耗时前必须先热身**。

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.12.16（conda-forge，env `audio`） |
| onnxruntime | 1.31.1（CPU 构建 `py312h27e101f_1_cpu`） |
| pytorch / torchaudio | 2.13.0 cuda134 / 2.11.1 cuda134（ONNX 路径不参与计算） |
| CPU | i5-14600KF，20 逻辑线程（6P+8E） |
| GPU | RTX 5070 Ti（sm_120），驱动 616.56 / CUDA 13.4 |

## 命令

```bash
"C:\Users\PC\miniforge3\envs\audio\python.exe" scripts/run_baseline.py
```

`scripts/run_baseline.py` → `dsh_moss_tts.onnx_backend.OnnxBackend` → `vendor/moss_tts_nano_onnx/onnx_tts_runtime.OnnxTtsRuntime`。

## 数据

| 项 | 值 |
|---|---|
| 模型加载（8 个 ORT session） | 6.0 s |
| 合成 | 2.26 ~ 2.32 s |
| 每帧 | 34 ~ 35 ms |
| RTF | 0.42 ~ 0.43 |
| 音频 | 5.36 s / 48 kHz / 2 ch / 1,029,164 B |
| 帧数 | 67（× 16 码，无重复行） |

## 产物

`dsh-moss-tts\runs\baseline\`：`zh_seed0.wav`、`zh_seed0.frames.npy`、`zh_seed0.metrics.json`
（`zh_seed0_run2.wav` 是确定性验证的第二次输出）

## 注意

1. **不热身就比耗时没有意义**：首次冷启动 6.8 s，热态 2.3 s。
2. `enable_wetext=False`：`WeTextProcessing` 依赖 `pynini`，Windows 无轮子；robust 清洗仍生效。
3. 采样超参被烘进 `moss_tts_local_fixed_sampled_frame.onnx`，`sample_mode=fixed` 时改 CLI 的 temp/top_k/top_p 无效；要扫参得用 `full` 模式。
4. 线程数固定 4，未调优（后续再做 6/8/12 的扫描与绑核）。
5. 仓库自带 `infer_onnx.py` 会写自己的 `generated_audio/`，在 DSH 沙箱里必然失败，故走 `scripts/run_baseline.py`。

# 原始 PyTorch 模型推理（本地直载，零 HF 缓存）

> 环境：`C:\Users\PC\miniforge3\envs\audio\python.exe`（Python 3.12 / torch 2.13.0 / transformers 5.22.1）
> 权重：`D:\workplace\audio\github\MOSS-TTS-Nano-100M` + `MOSS-Audio-Tokenizer-Nano`
> 时间：2026-11-01。ONNX/C++ 只作为对拍基准，不参与本路径计算。

## 0. 一句话结论

**不是模型、权重、prompt、分词或音色克隆的问题。**
是 transformers 5.x 的 `from_pretrained` 把 checkpoint 的**非持久 buffer**（RoPE 的 `inv_freq`）当成
未初始化内存重建，导致 **RoPE 静默失效**，音频全错但每个权重都加载得逐位正确。

修法：**不降级 transformers**，改成「直接 import 模型目录里的 .py + 从 config.json 构造 + `load_state_dict`」。
实现见 `src/dsh_moss_tts/torch_ref.py`，入口见 `scripts/run_torch_reference.py`。

## 1. 根因（已实测）

transformers 5.x `from_pretrained` 在 meta device 上建模型，然后：

```python
# transformers/modeling_utils.py:4793-4797
for key, buffer in self.named_non_persistent_buffers():
    buffer_device = get_device(device_map, key, valid_torch_device=True)
    value = torch.empty_like(buffer, device=buffer_device)   # 未初始化！
    _load_parameter_into_model(self, key, value)
```

checkpoint 的 `MossTTSNanoGPT2RotaryEmbedding` 里：

```python
self.register_buffer("inv_freq", inv_freq, persistent=False)   # gpt2_decoder.py:42
```

它不在 `pytorch_model.bin` 里，本该由 `__init__` 算出来；被 `torch.empty_like` 覆盖后就是垃圾值。
实测加载后：

```
inv_freq[:8] = [-3.16e-11, 1.93e-42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
max |inv_freq| = 3.16e-11        # 正确值应是 1.0, 0.75, 0.56, ...
```

于是 `cos≈1, sin≈0`，**RoPE 变成恒等变换**。全局 12 层 + local 1 层，共 13 个 `rotary_emb.inv_freq`
全部中招。

## 2. 证据链

| # | 实验 | 结果 |
|---|---|---|
| 1 | 权重完整性 | 194/194，**0 missing / 0 unexpected / 0 shape mismatch**，逐张量 maxdiff=0 |
| 2 | 第 0 层 `present_value_0` vs ONNX | **cos=1.000000**（embedding + c_attn 的线性通路完全正确） |
| 3 | 第 0 层 `present_key_0` vs ONNX | cos=0.7516，**恰好等于「完全不做 RoPE」** |
| 4 | 手工把正确 RoPE 公式套到 pre-RoPE key | 与 ONNX **相对误差 0.000000** |
| 5 | 直载后 prefill 末位 hidden vs ONNX | **cos=1.000000，relL2=1.26e-6** |
| 6 | 直载后 assistant logits vs ONNX | slot `+13.984` / end `-1.035`，**完全相同** |
| 7 | 同一份 C++ 已验 prompt 跑贪心 | **帧 0 codes 16/16 逐位相同**；帧 1 只差 1 个通道（= Q8 量化本底） |

结论：模型数学没有问题，唯一坏掉的是 RoPE 的 buffer；换加载方式即可。

## 3. 正确用法

```powershell
$py = "C:\Users\PC\miniforge3\envs\audio\python.exe"
cd D:\workplace\audio\dsh-moss-tts

# 贪心（短句，会自然收尾）
& $py scripts/run_torch_reference.py --greedy --text "今天天气不错。" --frames 40

# 采样（默认 audio 0.8 / top_p 0.95 / top_k 25 / rep 1.2，与官方 infer.py 一致）
& $py scripts/run_torch_reference.py --seed 0 --text "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。" --frames 200

# 自定义参考音频 / 输出目录
& $py scripts/run_torch_reference.py --prompt-audio D:\path\ref.wav --out-dir runs\official
```

产物：`runs/official/torch_<tag>.wav`（48 kHz 立体声）+ `torch_<tag>_frames.txt`（每行 16 个 code）。

路径全部来自 `src/dsh_moss_tts/paths.py`，可用环境变量覆盖：

| 变量 | 默认 |
|---|---|
| `MOSS_MODELS_DIR` | `D:\workplace\audio\github` |
| `MOSS_TTS_CKPT_DIR` | `$MOSS_MODELS_DIR\MOSS-TTS-Nano-100M` |
| `MOSS_CODEC_CKPT_DIR` | `$MOSS_MODELS_DIR\MOSS-Audio-Tokenizer-Nano` |
| `MOSS_TTS_REPO_DIR` | `$MOSS_MODELS_DIR\MOSS-TTS-Nano`（只取 `assets/audio/*.wav`） |

## 4. 代码结构

| 文件 | 作用 |
|---|---|
| `src/dsh_moss_tts/torch_ref.py` | 合成包 import 模型目录的 .py；config.json 构造；`load_state_dict`；RoPE 断言 |
| `scripts/run_torch_reference.py` | 官方 `model.inference()` 的 CLI（本地权重、固定 seed） |
| `tools/torch_reference.py` | 旧命令兼容入口（内部转调上面的 CLI） |
| `tools/check_torch_frame0.py` | 硬回归：喂 `runs/probe/p_short.txt`，帧 0 必须与 C++/ONNX 16/16 |
| `tools/ab_prefill_logits.py` | prefill hidden / assistant logits 与 ONNX 对拍 |
| `tools/check_weights_load.py` / `check_rope.py` | 权重与 RoPE 体检 |

加载器做了三件关键事：

1. `types.ModuleType` + `__path__` 把模型目录注册成合成包，`importlib` 直接 import，**不写 HF 动态模块缓存、不需要 `trust_remote_code`**。
2. 先构造再 `load_state_dict`，非持久 buffer 保留 `__init__` 算出的值。
3. `assert_rope_ok()` 显式校验每个 `inv_freq == 1/base^(arange(0,head_dim,2)/head_dim)`（实测最大相对误差 **6.97e-08**）；不合法直接抛异常，不让它再静默出噪声。

## 5. 验收口径与实测

```powershell
& $py tools/check_weights_load.py          # 0 missing / 0 unexpected
& $py tools/check_rope.py                  # max rel error 6.97e-08
& $py tools/check_torch_frame0.py          # PASS: 帧 0 16/16
& $py tools/ab_prefill_logits.py --text "今天天气不错。"   # hidden relL2 ~1e-6，logits 相同
& $py scripts/run_torch_reference.py --greedy --text "今天天气不错。" --frames 40
& $py scripts/run_torch_reference.py --seed 0 --text "..." --frames 200   # 跑两次，codes/wav SHA256 相同
```

实测（CPU fp32，sdpa）：

| 项 | 值 |
|---|---|
| 加载 | 4.5 s |
| 贪心「今天天气不错。」 | 18 帧 / 1.44 s / peak 0.891 / rms 0.150 / 合成 2.28 s |
| 采样长句 seed 0 | 76 帧 / 6.08 s / peak 1.095 / rms 0.143 |
| 采样同 seed 两次 | codes 与 WAV **逐位相同** |
| 与 ONNX 同句（采样） | ONNX 67 帧 / 5.36 s / peak ≈1.0；帧数不同是 RNG 与采样实现差异，不是判据 |

> `peak` 可能略大于 1.0：这是模型输出本身（ONNX 同句也到 0.99997），`torchaudio` 写的是 float32 WAV，
> 不会裁切；播放端若按 16-bit 输出需要自己留 headroom。

## 6. 已知差异：prompt 模板差一个 `<|im_start|>`

- PyTorch `build_inference_input_ids()`：`[im_start(4)] + "user\n" + ...`，同一句是 **181 行**。
- ONNX/C++ 的 manifest 模板：从 `600` 开始，**没有前导 im_start**，同一句是 **180 行**。

两种都能出声，但 codes 不同。**本仓库的 `model.inference()` 走官方 PyTorch builder（带 im_start）**，
所以 `scripts/run_torch_reference.py` 的输出按官方路径才对。帧 0 硬回归之所以用 C++ 的 180 行 prompt，
是为了只验模型数学、不混入 prompt 构造差异。

## 7. 如果你仍想降级 transformers（可选，不是本仓库默认路线）

**A. 隔离覆盖（不动 `audio` 环境，推荐给必须原样跑官方 `infer.py` 的场景）**

```powershell
$py  = "C:\Users\PC\miniforge3\envs\audio\python.exe"
$tf4 = "D:\workplace\audio\envs\tf4"
& $py -m pip install --target $tf4 --ignore-installed "transformers==4.57.1"
# pip 会自动把 tokenizers / huggingface_hub 装进 $tf4

$env:PYTHONPATH = "$tf4;D:\workplace\audio\github\MOSS-TTS-Nano"
$env:HF_MODULES_CACHE = "D:\workplace\audio\.hfmods\modules"   # 仍要显式指定，避免写用户缓存
& $py -c "import transformers; print(transformers.__version__, transformers.__file__)"

& $py D:\workplace\audio\github\MOSS-TTS-Nano\infer.py `
  --checkpoint D:\workplace\audio\github\MOSS-TTS-Nano-100M `
  --audio-tokenizer-pretrained-name-or-path D:\workplace\audio\github\MOSS-Audio-Tokenizer-Nano `
  --prompt-audio-path D:\workplace\audio\github\MOSS-TTS-Nano\assets\audio\zh_1.wav `
  --text "今天天气不错。" --output-audio-path D:\workplace\audio\dsh-moss-tts\runs\official\official_tf4.wav `
  --device cpu --dtype float32 --do-sample 0 --max-new-frames 60
```

注意：torch 仍是 2.13（官方钉 2.7），属于「4.57.1 + 新 torch」组合；撤销只需删 `$tf4` 目录。

**B. 完整官方组合（独立 conda 环境，需联网）**

```powershell
mamba create -n moss-torch4 python=3.12 "pytorch=2.7.*=*cpu*" "torchaudio=2.7.*=*cpu*" -c conda-forge -c pytorch
```

**C. 直接降共享 `audio` 环境（不推荐）**

```powershell
& $py -m pip install "transformers==4.57.1"
# 回滚：& $py -m pip install "transformers==5.22.1" "tokenizers==0.23.3" "huggingface_hub==2.4.0"
```

会连带降 `tokenizers` / `huggingface_hub`，影响同环境其他项目；mamba 在沙箱里还要每条命令一次完全权限。

## 8. 排错

| 现象 | 处理 |
|---|---|
| `assert_rope_ok` 报 `inv_freq 不是训练时的 RoPE 频率` | 说明有人又走了 `from_pretrained`；改回 `torch_ref.load_all` |
| `FileNotFoundError: ... 缺少文件` | 检查 `MOSS_TTS_CKPT_DIR` / `MOSS_CODEC_CKPT_DIR` 是否指向完整权重目录 |
| 帧 0 与 C++ 不一致 | 先确认 `check_weights_load.py` 与 `check_rope.py` 都过；再确认 `--attn sdpa`（eager 也应一致到 1e-6） |
| 音频有开头多余元音 / 停不下来 | 这是采样与结束判定的旧问题，见 `doc/音频问题排查.md`，不是本次加载问题 |

# AGENTS.md

MOSS-TTS-Nano 的 CPU 推理后端，最终做成 DeepSeek TTS 插件。推理部分必须能独立运行，前后端可自由替换。

## 目录约定

| 目录 | 规矩 |
|---|---|
| `vendor/` | 上游代码**原样复制**，只读。要改行为就在 `src/` 重写一份，绝不就地改。 |
| `src/dsh_moss_tts/` | 自有代码。`paths.py` 是路径唯一来源；`onnx_backend.py` 是推理后端（封装 vendor）。 |
| `scripts/` | 可执行入口，`python scripts/xxx.py` 直接跑。 |
| `doc/` | 设计与实测记录。 |

- 权重不进仓库：`D:\workplace\audio\github\MOSS-TTS-Nano-100M-ONNX` + `MOSS-Audio-Tokenizer-Nano-ONNX`（`MOSS_MODELS_DIR` 可覆盖）。
- 产物落在项目内 `dsh-moss-tts/runs/<name>/`（`MOSS_RUNS_DIR` 可覆盖）。

## 环境

- Python：`C:\Users\PC\miniforge3\envs\audio\python.exe`（3.12）
- 装包只用 mamba，且**显式钉 CPU 构建**（不钉会拉 CUDA 那套）：
  `mamba install -n audio -c conda-forge 'onnxruntime=1.31.1=*cpu*'`
- 当前：`onnxruntime 1.31.1 cpu` / `pytorch 2.13.0 cuda134` / `torchaudio 2.11.1 cuda134`
- `WeTextProcessing`（依赖 pynini）在 Windows 无轮子 → 一律 `enable_wetext=False`，robust 清洗仍生效。

## 运行

```bash
"C:\Users\PC\miniforge3\envs\audio\python.exe" scripts/run_baseline.py
# 原始 PyTorch 参考：本地直载，不写 HF 缓存、不联网、不用 trust_remote_code
"C:\Users\PC\miniforge3\envs\audio\python.exe" scripts/run_torch_reference.py --greedy --text "今天天气不错。"
```

- PyTorch 权重加载一律走 `src/dsh_moss_tts/torch_ref.py`（见 `doc/torch_reference.md`）；
  **不要**再用 `from_pretrained(trust_remote_code=True)`，原因见下面「已知坑 6」。
- 硬回归：`python tools/check_torch_frame0.py`（帧 0 必须与 C++/ONNX 16/16）。

## 已知坑

1. DSH 沙箱进程是**低完整性级别**，只能写带 Low 完整性标签的对象。本工作区已一次性打标（管理员 PowerShell）：
   `icacls "D:\workplace\audio" /setintegritylevel (OI)(CI)Low /T /C`
   新建目录自动继承；换机器或新建工作区时要重做，否则沙箱写不进既有目录。
2. mamba 在沙箱里 TLS 握手失败（schannel 取不到凭据），每条 mamba 命令要一次「完全权限」。
3. conda-forge 的 `onnxruntime` 默认优先 CUDA 构建，会连带装 cudnn/cuda。
4. Windows 下别用 PowerShell 的 `Get-Content`/`Set-Content` 改中文源码（GBK 往返会乱码），用编辑器。
5. 仓库里 `infer_onnx.py` 会把输出写到自己目录下的 `generated_audio/`，沙箱里必然失败 → 用 `scripts/run_baseline.py` 走同样的运行时、指定输出目录。
6. **transformers 5.x 的 `from_pretrained` 会把非持久 buffer 变成未初始化内存**（`modeling_utils.py:4793-4797` 用 `torch.empty_like` 重建）。checkpoint 的 `RotaryEmbedding.inv_freq` 正好是 `persistent=False`，于是 RoPE 静默失效、音频全错，而 194 个权重逐位加载正确。PyTorch 参考一律走 `src/dsh_moss_tts/torch_ref.py`（构造 + `load_state_dict`），并用 `assert_rope_ok` 兜底。详见 `doc/torch_reference.md`。
7. **prompt 模板以 `prompting.py` 为准，别抄 manifest 的 token 数组**：
   `browser_poc_manifest.json` 的 `user_prompt_prefix_token_ids` 少了开头的 `<|im_start|>`(4)，
   而手抄 `after_reference` 时又多抄过一个 `505, 587`（`'▁N'+'one'`），prompt 变成
   `- Text: None<正文>`，模型先把 "None" 念出来 —— 这是"每段开头多一声"的历史根因。
   自检：行数 = `1 + 12 + 1 + N + 1 + 56 + text + 6 + 1`，且与 torch
   `build_inference_input_ids` 逐字节相同（`python tools/make_prompt.py` + 对照 `runs/ab/prompt_rows.txt`）。

## 当前基线

5.36 s 中文 / 48 kHz 立体声 / 4 线程 CPU / **RTF 0.43**（热态；冷启动首跑 6.8 s）/ 两次运行逐位一致。见 `doc/baseline.md`。

## C++ 引擎（csrc/）

```bash
cd csrc
build.bat                 # 调 cl 直接编译，产物 build/moss_llm.exe、build/test_kernels.exe
build\moss_llm.exe --weights ..\weights\llm_q8.moss --prompt <prompt.txt> --frames 24 --greedy --threads 4
```

- 权重导出：`python tools/export_weights.py --ckpt <pytorch_model.bin> --out weights/llm_q8.moss --plan q8`
- prompt 生成：`python tools/make_prompt.py --text "..." --voice Junhao --out runs/cpp/prompt.txt`
- ONNX 对照：`python tools/onnx_reference.py --text "..." --frames 24 --out runs/onnx_greedy`

### 编译相关的三个坑

1. **`build.bat` 必须是纯 ASCII**：cmd.exe 按 936 代码页读 .bat，UTF-8 中文注释会把命令拆碎。
2. **C++ 源必须 `/utf-8` 编译**：否则中文注释让 MSVC 按 GBK 解码，`__m256i` 之类会报"未声明"。
3. **不要用 CMake**：沙箱里 cmake 的编译器 ABI 探测会挂死（管道受限）。CMakeLists.txt 保留给沙箱外使用。

### 精度指标口径

自由生成的 codes 一致率**不可用**（argmax 对量化误差极敏感，翻转会级联）。
一律用 teacher forcing：`--force-codes <参考 codes>`，比较 `teacher_flip_rate`；后续补 logits KL。
**坑**：teacher forcing 时也必须加 `--greedy`，否则 `teacher_flip_rate` 统计的是采样命中率
（≈1/top_k），会得到 83% 的假翻转。KL 不受影响，只有翻转率要用贪心口径。

### 当前实测

| 配置 | prefill | ms/帧（4 线程） |
|---|---|---|
| **Q8_0_4x4（生产）** | **176 ms** | **6.39** |
| Q8_0（行内块） | 253 ms | 7.06 |
| f16 | 346 ms | 10.05 |

纯量化成本（teacher forcing，喂自家 f16 codes）：Q8 翻转 **6.0%**，f16 自洽 0%。
### 端到端（67 帧，Q8，4 线程）

prefill 170 ms + 生成 446 ms = **616 ms**（LLM 段，5.36 s 音频，RTF 0.115；Q8_0_4x4）。
ONNX 基线同句 2260 ms（含 codec 解码）。

精度（teacher forcing）：Q8 vs f16 → KL 0.0009、top-25 重叠 97.4%、code 翻转 6.0%。
Q4 全量化翻转 70.6%，且只快 5%（内核算力受限），暂不采用。
### 4×4 repack 与阶段二定稿（2026-11-01）

- 权重格式新增 `q8_0_4x4` / `q4_0_4x4`（4 行一组，组内 4 个 scale 相邻）。
- **生产配置：`Q8_0_4x4`，6.39 ms/帧（36 GB/s），prefill 176 ms，KL 0.001 / top-25 97%。**
- 逐张量 Q4 消融：单组只省 4%，三组一起省 9% 但 KL 0.09、翻转 38% → **Q4 不进入发布配置**。
  若将来要用，global `attn.w` 是唯一性价比可接受的（单位收益 17.9）。
- 导出：`--plan q8x4` / `--plan q4x4`；消融扫描：`python tools/sweep_q4.py ...`。

### prefill vs decode 的精度拆分（2026-11-03）

`moss_llm` 新增 `--save-state/--load-state [--ignore-state-fp]`：把 prefill 完的「KV + 最后一行 hidden」
整份存盘，可跨权重精度移植（f16 prefill → int8 decode），用来把两段误差拆开。
同句 60 帧、同一段 in-domain codes 的结果：

| 组合 | KL | top-25 | code 翻转 |
|---|---|---|---|
| f16 prefill + int8 decode | 0.00318 | 96.3% | 8.2% |
| int8 prefill + f16 decode | 0.00053 | 98.6% | 2.7% |
| int8 + int8（生产） | 0.00403 | 96.0% | 10.0% |

**结论：精度损失主体在 decode 的逐 token 激活量化，prefill 不到 1/3。** 别再把"效果偏差"归到 prefill 上。
**坑**：teacher forcing 的 codes 必须 in-domain（同一 prompt 自己生成的），拿别的句子的 codes 会得到 88% 的假翻转。
全景（各段精度/加速/耗时/体积）：`doc/推理栈现状.md`。

### 权重精度下限：Q4 不可用（2026-11-03）

39 个张量组逐个降到 Q4 的扫描（`tools/lab_sweep.py`，4 组语料含克隆）结论：

| 组合 | 体积 | flip | KL | top-1 |
|---|---|---|---|---|
| Q8（生产） | 119.3 MB | 8.7% | 0.0042 | 91.7% |
| Q4 仅全局 fc | 92.3 MB | 64.1% | 0.444 | 37.0% |
| Q4 全局 | 78.8 MB | 74.5% | 0.777 | 26.1% |
| Q4 全局+local | 75.4 MB | 84.5% | 1.146 | 16.4% |
| Q4 全量化 | 63.4 MB | 85.9% | 1.161 | 14.7% |

- **没有"便宜"的区域**：decode 侧任一组 Q4 都 ≥0.0045 ΔKL（≈ 当前全部量化损失），最多换 2.4 MB/帧（0.9%）。
- **local 层最贵**（ΔKL 0.567、top-1 27%），它每帧要跑 17 次，误差放大 17 倍。
- Q4 的音频响度暴涨 5 倍并削顶（RMS 0.074 → 0.402），人耳不用听也能判死。
- 速度上 Q4 确实快 20%（6.43 → 5.17 ms/帧），但**端到端无收益**：解码器 5.9 ms/帧已成为瓶颈。
- 结论：**8 bit 是这台模型的底线**；再压只能换量化方案（Q6/Q4_K 类，需要新内核）或优化解码器。
  完整数据：`doc/LLM精度实验记录.md`。

### W8A16（质量档，2026-11-03）

权重仍 Q8、激活不量化（fp32），`--a16 off|decode|all`（或环境变量 `MOSS_A16`）。
AVX2 没有 f16 算术，但 int8→fp32 展宽比预想便宜：

| 模式 | flip | KL | LLM ms/帧 | prefill | 首音 |
|---|---|---|---|---|---|
| off（生产） | 8.7% | 0.0042 | 6.43 | 175 ms | 48 ms |
| **decode** | 5.2% | 0.00145 | 7.39 | 167 ms | 52 ms |
| all | 4.4% | 0.00102 | 7.50 | 303 ms | 65 ms |

**prefill 开 A16 会慢 73%，只开 decode 最划算**（精度 2.9 倍、首音只 +4 ms）。
**发布默认：`--a16 decode`**（C API 里也默认设成 decode，`--a16 off` 可退回全 A8 做实验）。
实现：`gemv.cpp` 的 `act16_enabled()` + `row_dot_q8_4x4_actf32`。

### 线程数怎么给（2026-11-03）

两个独立线程池：`--threads`（LLM 的 GEMV/GEMM）+ `--decode-threads`（音频解码器），流水线并行，
所以同一时刻最多 threads+decode_threads 个线程在忙。线程不绑核，每步有 barrier，超订反而更慢。

实测同一台（6P+8E）：**4+4 = 6.88 ms/帧，3+3 = 7.65，2+2 = 11.35**。
按物理核数给：2C→2+2，4C→3+3，6C+→4+4（解码器是热点，可 4+6）。超线程不算独立核。
默认 4+4；**线程数不影响输出**（只切行），可以按机器随便调。

### 前缀缓存（.mosskv）

发布包不带 `models/cache`（每音色 9~19 MB），首次用到某音色会自动建目录并重建：
首次首音 224 ms，之后 80 ms。删掉无损。

### 两项已落地的零损失优化（2026-11-03）

- `LlmModel::assistant_logits`：判句末只算 logits[7]/[9] 两行，不再整表 16384×768。
  每帧少读 13.4 MB，local 段 2.51 → 2.37 ms，与整表路径 **codes 逐位相同**（`--dump-gate` 对照）。
- 流水线首块 4 帧（之后回 8 帧）：首音 112 → **77 ms**。
  2 帧能到 50 ms，但首块只有 160 ms 缓冲，弱机上生成下一块要 150~250 ms，会欠载卡顿，所以取 4。

## 阶段三（解码器）

- 权重导出：`python tools/export_decoder.py --src <safetensors> --out weights/decoder_f32.moss --dtype f32`
- 语义参考：`python tools/decoder_reference.py --codes runs/cpp/codes_q8x4_67.txt`
  → 与 ONNX `decode_full` 对拍 **SNR 52.6 dB**（fp16 vs fp32 的差），形状与峰值一致。
- 解码链路：codes[nq,T] → 16 路 LFQ 查表求和(512) → output_proj(768) → patch4 → 4 层 → patch2 → 2 层
  → patch2 → 2 层 → patch2 → 4 层(→240) → patch240 → 去交织成立体声。
  每层：`x += ls1*attn(LN1(x)); x += ls2*ffn(LN2(x))`，ls 是每通道向量，FFN 无 bias、GELU 用精确 erf。

### 环境坑（重要）

**conda numpy 2.5.4 在本机做 matmul 会静默崩溃**（`0xc06d007f`，BLAS DLL 初始化失败）；逐元素运算正常。
所有需要矩阵乘的 Python 工具一律用 torch。
### 解码器 CLI（阶段三）

```bash
build\moss_decoder.exe --weights ..\weights\decoder_f32.moss --codes <codes.txt> --out audio.f32 --threads 4
```
- `codes.txt` 每行 16 个整数（一帧）；输出是交织立体声 float32。
- 对拍：`python tools/compare_audio.py --cpp <audio.f32> --codes <codes.txt> --save-wav out.wav`
- 实测：67 帧解码 576 ms（RTF 0.107），与 ONNX fp16 52.59 dB、与 torch fp32 参考 112 dB。
### 解码器精度选型（实测）

| 权重 | 体积 | 耗时 | SNR vs ONNX fp16 |
|---|---|---|---|
| fp32 | 43.7 MB | 537 ms | 52.59 dB |
| **fp16（生产）** | **22.3 MB** | 534 ms | **52.03 dB** |
| Q8_0 | 12.2 MB | 515 ms | 25.88 dB（不可用） |

解码器对权重量化远比 LLM 敏感，**用 fp16**；且换精度几乎不提速 → 瓶颈在注意力而非权重带宽。
stage 耗时分布：21 / 26 / 64 / **418 ms**（stage4 的 T=2144 因果注意力是热点）。
### 流式解码（D3/D4 完成）

- `--stream S`：每段环形 KV（窗口 500/800/1200/1600 帧），块间靠 KV 衔接。
- **流式 vs 整段 maxdiff 0.0（逐位相同）**；整段/流式对 ONNX 均 **61.1 dB**（必须带 context 窗口，否则只有 52 dB）。
  2026-11-01 修过一个真 bug：AVX2 窗口注意力最初把「worker 切片的 qkv 指针」当整块起点，键地址偏移 i0，
  分块解码与整段差 0.85；改成传整块基址 + 查询区间 [q0,q1) 后逐位一致。
- S=8：首块 35 ms、总 414 ms（6.2 ms/帧），整段 531 ms（7.9 ms/帧）。
- 解码器实测 6.2 ms/帧（流式 chunk 8），与 LLM 的 6.4 ms/帧接近；剩下的瓶颈是层内小批次 barrier。
### 全链路 CLI（moss_tts，产品形态）

```bash
# 从权重目录跑（models/ 里有 llm/decoder/encoder/tokenizer/voices）
build\moss_tts.exe --models ..\models --text "你好，世界。" --voice Junhao --out hello.wav
build\moss_tts.exe --models ..\models --text "..." --profile voices\me.mossclone --pcm-raw -
build\moss_tts.exe --models ..\models --selftest          # 加载+合成+峰值检查
build\moss_tts.exe --models ..\models --list-voices

# 参考音频 -> 音色档案（PCM WAV；压缩格式先转）
build\moss_clone.exe --encoder ..\models\encoder_f16.moss --in ref.wav --out voices\me.mossclone
```

实测（打包后，4+6 线程）：5.12 s 中文 64 帧、首音 **128 ms**、总 **764 ms**、RTF **0.149**；
13.0 s 中文（Xiaoyu，冷缓存）163 帧、首音 400 ms、RTF 0.178。依赖：`dumpbin /dependents` **只有 KERNEL32.dll**。

调试用的旧 CLI 仍在：`moss_llm.exe`（prompt 文件、teacher forcing、`--dump-gate`）与
`moss_decoder.exe`（codes → f32）。

**坑**：解码器的 `decode_chunk` 要 **nq 主序**（`codes[i*T+t]`），不是帧主序；传错不会崩，只会音频全错。

### 文本前端（自带分词，零依赖）

- `tools/spm_dump.py` → `weights/tokenizer.mossvocab`（427 KB，含 charsmap）；`spm.cpp` 实现
  Darts trie + NFKC + BPE + 字节回退。**61/61 句与官方 sentencepiece 逐 id 一致**
  （`tests/data/spm_corpus.txt`；`build\moss_spmtest.exe --vocab .. --corpus .. --golden ..` 复跑）。
- 不做语义归一化（数字/单位/日期原样喂模型）；清洗留给调用侧。
- 长文本按 75 token 切句（句末标点 → 从句标点 → 二分），段间 0.40/0.24 s 停顿。

### 音色克隆

- `moss_clone.exe`：WAV（PCM16/24/32/float32，任意采样率）→ 48 kHz 立体声 → 编码器 → `.mossclone`（几 KB）。
- 需要 `weights/encoder_f16.moss`（`tools/export_encoder.py` 导出，21.8 MB，210 张量）。
- 与 ONNX 编码器 codes 一致率 91.6~97.7%（帧数完全一致；剩余差异来自导出图的混合精度）。
  **回放 SNR 6.03 dB vs ONNX 6.01 dB**，克隆 prompt 的 teacher-forced 翻转 5.99%（= 量化本底）。
- **坑**：`data/clone_wav/*.wav` 是 MP3-in-WAV（fmt tag 0x55），必须先转 PCM；
  编码器 LFQ 的最近邻在归一化空间，但**重建残差用原始码本向量**（写错会让 codes 与 ONNX 只有 7% 相同）。

### 发版

```bash
cd csrc && build.bat                 # 全部 exe + moss_tts.lib（静态 CRT /MT）
cmd /c scripts\pack_release.bat zip  # -> dist/moss-tts-cpu-win64{,.zip}
```

包结构：`moss_tts.exe / moss_clone.exe / models/{llm_q8x4,decoder_f16,encoder_f16}.moss +
tokenizer.mossvocab + voices/*.mossclone + include/moss/tts.h + lib/moss_tts.lib + README.md`。
170.8 MB（zip 159.3 MB）。C API 冒烟：`tests/test_capi.c` 纯 C 编译链接 `moss_tts.lib` 后实跑。

### 线程模型（阶段四 A，已重做并实测）


- `Parallel` 是**普通对象**：`LlmModel` / `AudioDecoder` 各自持有，`GemvPoolScratch::bind(par, k)` 绑定；
  全局 `init_parallel()` / `parallel()` / thread_local 全部删除。
- `Parallel::run` 带重入/并发降级：同一个池被重入或两个线程同时用时退化为串行并置 `degraded`，不进 barrier。
- 流水线：LLM 在生产者线程、主线程消费，中间是有界帧队列（4×chunk，满则阻塞 = 背压）。
- 实测（67 帧 / 5.36 s，贪心，chunk 8）：串行 1353 ms → 并行 4+4 **1037 ms** → 4+6 **956 ms**（RTF 0.178）。
  解码器 6 线程饱和（526 ms）；LLM 不要低于 4 线程（2 线程首音掉到 418 ms）。
- 稳定性：10 连跑 64 帧一致、896~951 ms；1+1 / 2+2 / 2+4 / 8+8 全部正常；**线程数不影响输出**（只切行）。
- 数值回归：重构前后 WAV/codes **SHA256 相同**；pipeline 0 vs 1 逐位相同。

### 结束判定（早停的根因与修法）

上游 `_sample_assistant_text_token` 只在 `{audio_assistant_slot=9, audio_end=7}` 两个候选上**按概率抽**，
「说完了」= 每帧一次伯努利试验 → 长句必然有概率提前结束。

实测 `logit[9]-logit[7]`：
- 贪心出音频：400 帧全程 **+10~+14**，永不收敛 → 只能靠帧数上限；
- 采样出音频（0.8/25/0.95/1.2）：末 6 帧 **12.2→9.3→6.9→3.8→2.5→0.59→-0.50**，第 64 帧干净收尾。

所以：**默认采样出音频 + 结束判定贪心**（`end_greedy`，`--end-mode sample` 可还原上游），
外加 `--min-frames` 安全阀与 `--dump-gate gate.csv`（`frame,argmax_id,argmax_minus_slot,slot_minus_end`）。
纯贪心模式（`--greedy`）不会自然结束，必须显式 `--frames`。

### 采样参数（2026-11-03）

- **16 个音频 token**：默认 temp 0.8 / top-k 25 / top-p 0.95 / rep 1.2，全部可调：
  `--temp 0.8 --top-k 25 --top-p 0.95 --rep 1.2`；C API 用 `moss_set_sampling()`。
- **结束判定（logits[7] vs [9]）始终贪心**，不受这些参数影响。
- `--greedy-frames` 默认已从 1 改为 **0**：起音问题的根因是 prompt 模板（见坑 7），不是首帧采样。
  5 个种子实测起音包络都是「静音 → 渐强」，没有"多一声"。
- 采样是逐通道串行的：16 路各自维护历史（重复惩罚按通道算），第 ch 路以第 ch-1 路的 code 作输入。

### 运行时降级（已实测）

`MOSS_FORCE_SCALAR=1` 强制走标量路径：
- 解码器：773 ms（AVX2 501 ms），输出与 AVX2 **SNR 120.5 dB**（数值等价）。
- LLM：16.75 ms/帧（AVX2 6.39），codes 只有 12.9% 与 AVX2 相同 —— 这是 int8 累加顺序 + argmax
  级联的必然结果，听感等价但波形不同。若要求两路径波形一致，降级路径需改用 fp16 权重。
### 泛化验收（已做）

三个新用例（中文长句含数字 / 英文 Ava / 中文女声 Xiaoyu）teacher-forced 翻转率 12.5% / 19.9% / 14.8%，
与基准句 6.5% 同量级 → tokenizer、prompt 模板、音色 codes、长句分块都正常。

### 阶段四的坑（已填）

旧实现用 thread_local 池 + 全局 `parallel()`，`--pipeline 1` 在 threads≥2 时死锁；现在按上文「线程模型」重做。
结果：10 连跑与 1+1 / 2+2 / 8+8 都正常，输出与串行逐位一致（见「线程模型」）。

## 产品层（v0.1 发布形态，2026-11-04）

### 三个外壳，一套推理

```
moss_tts.exe
├── 无参数 / --serve  → Runtime（队列/取消/统计）← JsonHttpServer（127.0.0.1）← web/app（浏览器）
├── --stdio           → 同一个 Runtime，JSON 行协议走 stdin/stdout（DSH 插件走这条）
└── --text ...        → TtsEngine（CLI，单次进程）
```

- `csrc/src/stdio_server.cpp`：`{"id","cmd"}` 一行进、`begin` / `audio`(base64) / `done` 一行出。
  `speak` 异步提交，主线程继续读 stdin 才能收到 `cancel`；`begin` 用客户端 id 当 job id，
  保证它一定排在第一个 `audio` 之前。测试：`tools/stdio_smoke.py`。

- `csrc/src/runtime.cpp`：单工作线程 + 有界队列（默认 4）+ job 表（TTL 10 min / 32 条）+ 取消 + 统计 + 音色枚举/克隆 + 路径解析。传输层只做编解码，**模型状态只在这里**。
- `csrc/src/server.cpp`：Winsock HTTP/1.1，只绑 `127.0.0.1`；`SO_EXCLUSIVEADDRUSE`（Windows 上 `SO_REUSEADDR` 允许多个进程绑同一端口，别用）；端口 8759→8763 顺延。
- `csrc/src/json.cpp`：最小 JSON（深度 8、单个字符串 8 MB——stdio 会把整段 WAV 用 base64 塞进一行、
  拒绝 NaN/Inf、支持 `\uXXXX` 代理对）。零依赖是硬约束，不引 cpp-httplib / nlohmann。

### API v1（细节见 `doc/api.md`）

`/api/v1/{health,info,voices}`、`POST /tts/stream`（chunked s16le 48k 立体声）、`POST /tts/wav`、
`POST /voices/clone?name=`、`GET /jobs/{id}`、`POST /jobs/{id}/cancel`。
音频头是 `application/octet-stream` + `X-Moss-Audio-Format: s16le;rate=48000;channels=2`，
**不要**用 `audio/L16`（RFC 2586 是 big-endian）。

### 安全模型

只绑 loopback + `Host` 白名单（`127.0.0.1:<port>` / `localhost:<port>` / `[::1]:<port>`，挡 DNS rebinding）
+ `Origin` 精确匹配（默认只允许本服务自己的 origin，`--allow-origin` 可追加）
+ POST 强制 JSON/WAV Content-Type + 文本 4000 字符 / JSON 256 KB / 上传 20 MB 上限。
**不做 token**：同源页面、插件和 curl 都要能直连，token 在这个威胁模型下不增加实质防护，
却会破坏"双击即用"和第三方调用。

### 数据目录与便携

查找顺序 `--models` → exe 同级 `models/` → `%LOCALAPPDATA%\MossTTS\models`。
便携模式（models 在旁边）时用户数据写 `exe/userdata/`；否则写 `%LOCALAPPDATA%\MossTTS\`。
`--serve` 的缓存固定进 `<data>/cache/`（`--cache` 显式覆盖），CLI 仍用 `models/cache/`。
用户克隆音色写 `<data>/voices/user/`，与内置 `models/voices/` 物理隔离，升级覆盖不到。

### 打包与发布

```bat
cd csrc && build.bat                                    :: 新增 json/runtime/server + ws2_32/shell32
pwsh scripts/make_model_lock.ps1 -Weights weights       :: 生成 scripts/model_lock.json（信任根）
pwsh scripts/pack_release.ps1 -Mode all -Tag v0.1.0     :: dist/MossTTS-win-x64-{full,lite}.zip + sdk
```

- 资产名**固定不带版本**，所以 `releases/latest/download/MossTTS-win-x64-full.zip` 永久有效。
- `scripts/model_lock.json` 是**已提交的信任根**：打包前逐个校验大小 + sha256。
  CI 里绝不能"下载完再生成清单再自校验"，那是循环论证。
- 发版流水线：`ci.yml`（编译+内核/分词回归，不下载权重）、`release.yml`（tag → 拉固定 revision 权重 →
  校验 → 真模型 selftest + `tools/http_smoke.py` → 打包 → `gh release create`）、`pages.yml`（只发 `web/landing`）。
- `build.bat` 已改成 vswhere 回退（本地走写死路径的快路径），GitHub Runner 上直接可跑。

### 这一层踩到的坑（都已填）

1. **解码器跨请求状态泄漏**：`AudioDecoder` 是流式的，`synthesize` 不 `reset_stream()` 的话，
   常驻进程里第二句会带着上一句的 KV，导致同一文本同一 seed 每次结果都不同。
   CLI 每进程只跑一次，所以以前没暴露。现在 `synthesize` 入口统一 reset。
2. **POST 没有 Content-Length 要按 0 处理**：`/jobs/{id}/cancel` 就是无体 POST，早期版本直接 400，
   浏览器点"停止"实际没取消。
3. **`raw_json` 不做包裹**：`voices_json` / `job_json` 返回的是片段，忘补花括号会得到非法 JSON
   （picker 报 "Extra data: line 1 column 10"）。
4. **PowerShell 5.1 读无 BOM 的 UTF-8 `.ps1` 会按 ANSI 解码**，中文字节吃掉引号 → here-string 报
   "missing terminator"。`.ps1` 一律存 UTF-8 **with BOM**；`.bat` 仍保持纯 ASCII。
5. **本机 `%TEMP%` / `%LOCALAPPDATA%` 在 DSH 沙箱里写不进去**：测试一律用项目内 `runs/`；
   前缀缓存写不进去时引擎会打印一行并继续重算，不影响合成。
6. **`GetLogicalProcessorInformationEx` 第一次调用必然返回 FALSE**（`ERROR_INSUFFICIENT_BUFFER`），
   只看 `len` 不看返回值；之前 `physical_cores` 直接写死 1，自动线程数会退化成 1+1。
7. **浏览器到 loopback 的通道**：https 页面 fetch `http://127.0.0.1` 是允许的（loopback 被视为可信来源），
   但 Chrome 的 Private Network Access 预检要求服务端回 `Access-Control-Allow-Private-Network: true`；
   `ws://` 没有这个豁免，所以协议只用 HTTP chunked。v0.1 默认不放开跨域（门户页只做下载/跳转）。

### 验收（已实测）

| 项 | 结果 |
|---|---|
| CLI 与 HTTP 的原始 PCM | SHA256 **相同**（同参数、同 seed、greedy） |
| 常驻进程内连续请求 | 逐字节一致（先修好了上面的解码器泄漏） |
| 取消 | 15~20 ms 内连接结束，状态 `cancelled`，不吐完整段 |
| 安全 | 错误 Host / 跨源 Origin / 目录穿越 / 超大 JSON / 坏 JSON / 错误方法 全部 4xx |
| 包内运行 | `runs/pkg` 解压后 `--version`/`--selftest`/`--serve` 全部正常，模型信息来自 `models/manifest.json` |
| 连续合成 | 20 连跑 0 失败，RSS 222 → 252 MB（首音色前缀 KV + job 表），队列归零 |
| stdio 通道 | `tools/stdio_smoke.py`：ping/info/voices/speak/cancel/clone/关 stdin 退出全过；`begin` 恒在首个 `audio` 之前 |
| 冒烟脚本 | `python tools/http_smoke.py --base http://127.0.0.1:8759` / `python tools/stdio_smoke.py`（须用 conda 的 python；控制台先 `chcp 65001`） |

## 插件层（dsh-plugin/，2026-11-04）

DSH 宿主插件接入的是 `--stdio`，不是 HTTP：不需要端口、不需要先起服务，插件自己 spawn 常驻子进程。

```
index.mjs    DSH 入口：export { name, inject:['tools'], apply(ctx, config) }
             ctx.tools.register(defineTool(...)) 注册 4 个工具；ctx.effect 里 dispose 子进程
backend.mjs  TtsBackend：spawn --stdio、按行解析、把 audio 行 base64 解成 PCM 回调、取消、克隆
player.mjs   Windows 播放：写临时 WAV → System.Media.SoundPlayer.PlaySync()（阻塞到播完）
```

- `defineTool` 从宿主 asar 里 import：`<resources>/app.asar/dsh/node_modules/@deepseek-ai/dsh-tools/lib/index.js`，
  先用 `process.resourcesPath` 推，再退回写死路径；两边都失败就退化成恒等函数，保证插件至少能加载。
- 两个回归脚本都在仓库里、都不需要 DSH：
  `node dsh-plugin/test-backend.mjs`（协议层 10 项）、`node dsh-plugin/test-plugin.mjs`（假 ctx 里注册并真合成）。
- 播放刻意选"整段合成完再播"：引 ffplay/sox 会加依赖，而 5 s 语音只要 0.76 s 合成；流式播放留到 v2。
- 还没做"助手回复自动朗读"（需要宿主消息事件），现在只有显式工具调用。


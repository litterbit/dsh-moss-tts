# 权重仓库（HuggingFace）

<https://huggingface.co/litterbit/moss-tts-nano-moss>（public，Apache-2.0）

## 为什么要有它

发布包 159 MB，其中 164 MB 是权重。把权重复用到一个公开仓库有三个好处：

1. CI 不用上传 160 MB 到 Release，`release.yml` 拉一次、按 `model_lock.json` 校验，然后打包。
2. 只想跑程序的人下 lite 包（1 MB），权重单独 `git clone` 到 `models/`。
3. 模型卡能把"这是什么、精度损失多少、许可"一次说清（正文源文件是 `doc/hf_readme.md`）。

## 布局是硬约束

**仓库根目录 = 发布包里的 `models/`**，路径和 `scripts/model_lock.json` 的 `path` 一一对应：

```
README.md  manifest.json  .gitattributes
llm_q8x4.moss  decoder_f16.moss  encoder_f16.moss  tokenizer.mossvocab
voices/*.mossclone        <- 必须是子目录
```

`scripts/fetch_weights.ps1` 拼的是 `$Base/<path>`，所以 `voices` 一旦散在根目录就会 404；
反过来，多出来的文件（例如 `cache/`）不会有人校验，别放进去。

## 更新流程

```bash
# 1) 本地权重就位后，刷新信任根（会保留已有的 hf_commit 字段）
pwsh scripts/make_model_lock.ps1 -Weights weights

# 2) 组装到 runs/hf：按 lock 逐个核对 bytes/sha256，渲染 README，生成 manifest.json
"C:\Users\PC\miniforge3\envs\audio\python.exe" tools/hf_upload.py --models weights

# 3) 真上传（用本机缓存的 token：huggingface-cli login 一次；脚本不会打印它）
"C:\Users\PC\miniforge3\envs\audio\python.exe" tools/hf_upload.py --upload --verify

# 4) 把打印出来的 sha 填进 scripts/model_lock.json 的 "hf_commit"，再提交
```

`--verify` 会：大文件比 LFS 的 sha256（就是内容 sha256），小文件下载回来自己算。
2026-11-05 首次整理后的结果：25 个文件、22 个权重条目与 lock 完全一致，
head 见 `scripts/model_lock.json` 的 `hf_commit`。

## 发布流水线怎么用它

`release.yml` 里：

- `HF_MODEL_REPO`：默认 `litterbit/moss-tts-nano-moss`，仓库变量可覆盖（fork 时改这个就行）。
- revision：优先 `vars.HF_MODEL_REVISION`；没有就用 `scripts/model_lock.json` 的 `hf_commit`。
  **不要**让流水线跟着 `main` 漂——校验虽然是逐文件 sha256，但 revision 固定能让"同一个 tag 重跑"
  拿到同一批字节。
- `scripts/fetch_weights.ps1 -Base https://huggingface.co/<repo>/resolve/<rev>` 逐文件校验；
  校验失败直接失败，绝不"下完再生成清单自校验"（那是循环论证）。

本地想试别的来源：`scripts/fetch_weights.ps1 -Base <url> -Weights <dir>`，
或直接 `MOSS_WEIGHTS_BASE=<url>`。

## 踩过的坑

1. **HF REST 的 LFS 字段叫 `lfs.sha256`**（huggingface_hub 早期版本叫 `oid`）。LFS 的 sha256 就是
   文件内容的 sha256，可以直接和 `model_lock.json` 比，不用下载。
2. **小于 10 MB 的文件 HF 不走 LFS**，没有哈希可查，只能下载回来算（`voices/` 全在这一档，合计 60 KB）。
3. 上传用 `upload_folder(delete_patterns=[...])` 清理旧布局；它只删"不在本次提交里"的文件，
   所以同一次提交里同时传 `voices/` 和删根目录副本是安全的。
4. 这台机器上 `pwsh` 不在 PATH（`powershell` 才是），脚本里别互相调用 `pwsh`。

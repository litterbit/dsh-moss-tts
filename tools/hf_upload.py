#!/usr/bin/env python
"""整理并上传权重到 HuggingFace（模型卡 + manifest + 权重 + 音色），或只组装到本地。

    python tools/hf_upload.py --models weights                 # 组装到 runs/hf，不联网
    python tools/hf_upload.py --models weights --upload        # 真上传（用本机缓存的 token）
    python tools/hf_upload.py --upload --verify                # 传完再从网上抽检一遍

做的事：
  1. 以 scripts/model_lock.json 为信任根，逐个核对本地文件的字节数与 sha256
  2. 组装 runs/hf/：lock 里的相对路径（含 voices/ 子目录）+ 生成的 manifest.json
     + README.md（doc/hf_readme.md 渲染，文件表由 lock 生成，不手抄）
  3. 一次性 commit 上传；delete_patterns 清掉旧布局（根目录下散放的 *.mossclone）
  4. 打印新的 commit sha —— 填回 scripts/model_lock.json 的 hf_commit，
     CI/发布就按固定 revision 拉权重，而不是跟着 main 漂

为什么要这个脚本：HF 上的目录结构必须和发布包里的 models/ 一致（voices/ 是子目录），
否则 `fetch_weights.ps1 -Base .../resolve/<rev>` 会 404，或者校验对不上。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "scripts" / "model_lock.json"
CARD = ROOT / "doc" / "hf_readme.md"
STAGE = ROOT / "runs" / "hf"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# HF 用 LFS 存大文件；这几类必须走 LFS，否则网页上无法预览/克隆很慢
GITATTRIBUTES = """*.moss filter=lfs diff=lfs merge=lfs -text
*.mossclone filter=lfs diff=lfs merge=lfs -text
*.mossvocab filter=lfs diff=lfs merge=lfs -text
"""


def human(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / 1048576:.1f} MB"
    if n >= 1 << 10:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_table(files: list[dict]) -> str:
    rows = ["| 文件 | 大小 | sha256 |", "|---|---|---|"]
    total = 0
    for entry in files:
        total += entry["bytes"]
        rows.append(f"| `{entry['path']}` | {human(entry['bytes'])} | `{entry['sha256']}` |")
    rows.append(f"| **合计** | **{human(total)}** | |")
    return "\n".join(rows)


def stage(models: Path, lock: dict) -> Path:
    files = lock["files"]
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    bad = []
    for entry in files:
        src = models / entry["path"]
        if not src.is_file():
            bad.append(f"缺文件 {entry['path']}")
            continue
        size = src.stat().st_size
        if size != entry["bytes"]:
            bad.append(f"{entry['path']}: {size} 字节，lock 里是 {entry['bytes']}")
            continue
        digest = sha256(src)
        if digest != entry["sha256"]:
            bad.append(f"{entry['path']}: sha256 {digest[:12]}… != {entry['sha256'][:12]}…")
            continue
        dst = STAGE / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if bad:
        raise SystemExit("[x] 本地权重和 model_lock.json 对不上：\n    " + "\n    ".join(bad))
    print(f"[ok] {len(files)} 个文件校验通过，共 {human(sum(f['bytes'] for f in files))}")

    # manifest.json 就是 lock 本身去掉 hf_commit 之类的仓库私有字段
    manifest = {k: v for k, v in lock.items() if k != "hf_commit"}
    (STAGE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    card = CARD.read_text(encoding="utf-8")
    if "<!--FILES-->" not in card:
        raise SystemExit("[x] doc/hf_readme.md 里没有 <!--FILES--> 标记")
    (STAGE / "README.md").write_text(card.replace("<!--FILES-->", file_table(files)), encoding="utf-8")
    (STAGE / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    print(f"[ok] 组装完成：{STAGE}（{sum(1 for _ in STAGE.rglob('*') if _.is_file())} 个文件）")
    return STAGE


def upload(repo: str, commit_message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(STAGE),
        commit_message=commit_message,
        # 旧布局是散在根目录的 *.mossclone；只删「不在本次提交里」的匹配项
        delete_patterns=["*.mossclone", "*.moss", "*.mossvocab"],
    )
    sha = str(info.oid)
    print(f"[ok] 已上传 {repo} @ {sha}")
    print(f"     把下面这行填进 scripts/model_lock.json：\"hf_commit\": \"{sha}\"")
    return sha


def verify(repo: str, lock: dict) -> int:
    """抽检：大文件比 LFS oid，小文件直接下载算 sha256。"""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    problems = []
    for entry in lock["files"]:
        node = remote.get(entry["path"])
        if node is None:
            problems.append(f"缺 {entry['path']}")
            continue
        if node.size is not None and node.size != entry["bytes"]:
            problems.append(f"{entry['path']}: 远端 {node.size} != {entry['bytes']}")
            continue
        oid = None
        if node.lfs is not None:
            # huggingface_hub 的 LFSFileInfo 字段名是 sha256（REST 里也是 lfs.sha256），旧版叫 oid
            oid = getattr(node.lfs, "sha256", None) or getattr(node.lfs, "oid", None)
        if oid:
            if oid != entry["sha256"]:
                problems.append(f"{entry['path']}: LFS oid 不匹配")
            continue
        if entry["bytes"] > 8 << 20:  # 大文件却没有 LFS：说明传坏了
            problems.append(f"{entry['path']}: 大文件没走 LFS")
            continue
        local = hf_hub_download(repo_id=repo, filename=entry["path"], cache_dir=str(ROOT / "runs" / "hf_cache"))
        if sha256(Path(local)) != entry["sha256"]:
            problems.append(f"{entry['path']}: 下载后 sha256 不匹配")
    if problems:
        print("[x] 远端校验失败：\n    " + "\n    ".join(problems))
        return 1
    print(f"[ok] 远端 {len(lock['files'])} 个文件与 model_lock.json 一致")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="weights", help="本地权重目录（布局同 model_lock.json）")
    parser.add_argument("--repo", default="litterbit/moss-tts-nano-moss")
    parser.add_argument("--upload", action="store_true", help="真上传（默认只组装到 runs/hf）")
    parser.add_argument("--verify", action="store_true", help="上传后再从远端校验一遍")
    parser.add_argument("--skip-stage", action="store_true", help="复用已有的 runs/hf")
    parser.add_argument("--message", default="weights: 整理布局 + 模型卡（voices/ 子目录）")
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not args.skip_stage:
        stage((ROOT / args.models).resolve(), lock)
    if not args.upload:
        print("[i] 只组装没上传；加 --upload 才会写远端")
        return 0
    upload(args.repo, args.message)
    if args.verify:
        return verify(args.repo, lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

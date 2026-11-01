"""兼容入口：真正的实现在 ``scripts/run_torch_reference.py``。

保留这个文件是为了让文档里的旧命令继续可用：

    python tools/torch_reference.py --greedy --text "今天天气不错。"

注意：旧版本用 ``from_pretrained(trust_remote_code=True)`` 加载，在 transformers 5.x
上会因为非持久 buffer 被 ``torch.empty_like`` 重建而丢掉 RoPE（音频变噪声）。
现在统一走 ``src/dsh_moss_tts/torch_ref.py`` 的本地直载路径。
"""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_torch_reference.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")

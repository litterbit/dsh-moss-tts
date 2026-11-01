"""ONNX 推理后端。

薄封装 `vendor/moss_tts_nano_onnx` 里的官方运行时：只负责路径、计时和结果落盘，
不改它的任何代码。将来替换成自研实现时，保持 `OnnxBackend.synthesize` 的签名不变。

依赖：numpy + onnxruntime（CPU）；torch 只被 vendor 代码用在参考音频重采样上，
内置音色路径不会触发。
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .paths import MODELS_DIR, RUNS_DIR, VENDOR_DIR

FRAME_SECONDS = 0.08  # 12.5 fps


def _vendor_runtime_class() -> Any:
    vendor = str(VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from onnx_tts_runtime import OnnxTtsRuntime  # noqa: PLC0415

    return OnnxTtsRuntime


@dataclass(frozen=True)
class Synthesis:
    """一次合成的结果与实测指标。"""

    tag: str
    wav_path: Path
    frames: np.ndarray
    sample_rate: int
    channels: int
    samples: int
    load_seconds: float
    synth_seconds: float

    @property
    def audio_seconds(self) -> float:
        return self.samples / self.sample_rate

    @property
    def rtf(self) -> float:
        return self.synth_seconds / self.audio_seconds

    @property
    def ms_per_frame(self) -> float:
        return self.synth_seconds * 1000.0 / max(1, self.frames.shape[0])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["wav_path"] = str(self.wav_path)
        data["frames"] = int(self.frames.shape[0])
        data["audio_seconds"] = round(self.audio_seconds, 3)
        data["rtf"] = round(self.rtf, 3)
        data["ms_per_frame"] = round(self.ms_per_frame, 1)
        return data


class OnnxBackend:
    """官方 ONNX 运行时的进程内后端。"""

    def __init__(
        self,
        *,
        threads: int = 4,
        provider: str = "cpu",
        max_new_frames: int = 375,
        sample_mode: str = "fixed",
        model_dir: Path = MODELS_DIR,
        output_dir: Path = RUNS_DIR,
    ) -> None:
        runtime_class = _vendor_runtime_class()
        started = time.perf_counter()
        self._runtime = runtime_class(
            model_dir=str(model_dir),
            output_dir=Path(output_dir),
            thread_count=threads,
            max_new_frames=max_new_frames,
            sample_mode=sample_mode,
            execution_provider=provider,
        )
        self.load_seconds = time.perf_counter() - started
        self.max_new_frames = max_new_frames
        self.sample_mode = sample_mode
        self.provider = provider
        self.threads = threads

    def synthesize(
        self,
        text: str,
        *,
        tag: str,
        voice: str = "Junhao",
        seed: int = 0,
        streaming: bool = True,
        enable_wetext: bool = False,
    ) -> Synthesis:
        """合成一段文本，WAV 与帧序列落在 output_dir/<tag>.*。"""
        output_dir = Path(self._runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / f"{tag}.wav"

        started = time.perf_counter()
        result = self._runtime.synthesize(
            text=text,
            voice=voice,
            output_audio_path=wav_path,
            sample_mode=self.sample_mode,
            do_sample=True,
            streaming=streaming,
            max_new_frames=self.max_new_frames,
            enable_wetext=enable_wetext,
            seed=seed,
        )
        elapsed = time.perf_counter() - started

        waveform = np.asarray(result["waveform"], dtype=np.float32)
        frames = np.asarray(result["audio_token_ids"], dtype=np.int32)
        np.save(output_dir / f"{tag}.frames.npy", frames)

        return Synthesis(
            tag=tag,
            wav_path=Path(result["audio_path"]),
            frames=frames,
            sample_rate=int(result["sample_rate"]),
            channels=int(waveform.shape[1]),
            samples=int(waveform.shape[0]),
            load_seconds=round(self.load_seconds, 2),
            synth_seconds=round(elapsed, 3),
        )

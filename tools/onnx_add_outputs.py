"""把 ONNX 模型里的中间张量加成输出，方便和 C++/torch 参考对拍。

用法：
    python tools/onnx_add_outputs.py --src m.onnx --dst m_dbg.onnx --tensor /encoder.1/output_proj/MatMul_output_0 [...]

做法：在 GraphProto 末尾追加 output 记录（protobuf 直接拼接，不依赖 onnx 包）。
注意 dst 要放在 src 同目录，外部权重（.data）按相对路径解析。
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

FLOAT = 1


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, i


def write_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def iter_fields(buf: bytes):
    """逐个产出 (field_no, wire_type, value, value_start, value_end)。"""
    i = 0
    while i < len(buf):
        key, i = read_varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            v, j = read_varint(buf, i)
            yield field, wt, v, i, j
            i = j
        elif wt == 2:
            n, j = read_varint(buf, i)
            yield field, wt, buf[j : j + n], i, j + n
            i = j + n
        elif wt == 5:
            yield field, wt, buf[i : i + 4], i, i + 4
            i += 4
        elif wt == 1:
            yield field, wt, buf[i : i + 8], i, i + 8
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wt}")


def length_delimited(field: int, payload: bytes) -> bytes:
    return write_varint((field << 3) | 2) + write_varint(len(payload)) + payload


def value_info(name: str, elem_type: int = FLOAT) -> bytes:
    # TypeProto.tensor_type.elem_type
    tensor = write_varint((1 << 3) | 0) + write_varint(elem_type)
    type_proto = length_delimited(1, tensor)
    return length_delimited(1, name.encode()) + length_delimited(2, type_proto)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--tensor", action="append", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    data = src.read_bytes()

    graph_field = None
    for field, wt, value, start, end in iter_fields(data):
        if field == 7 and wt == 2:
            graph_field = (value, start, end)
            break
    if graph_field is None:
        raise SystemExit("找不到 graph 字段")
    graph, g_start, g_end = graph_field

    extra = b"".join(length_delimited(12, value_info(name)) for name in args.tensor)
    new_graph = graph + extra
    new_model = data[:g_start] + write_varint(len(new_graph)) + new_graph + data[g_end:]

    dst = Path(args.dst)
    dst.write_bytes(new_model)
    print(f"wrote {dst} ({dst.stat().st_size} bytes), +{len(args.tensor)} outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

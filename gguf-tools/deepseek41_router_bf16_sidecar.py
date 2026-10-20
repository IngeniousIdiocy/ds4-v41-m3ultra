#!/usr/bin/env python3
"""Write the DeepSeek V4.1 router gates as a small BF16 GGUF sidecar.

The engine maps it when DS4_V41_ROUTER_BF16_SIDECAR points at it; the
resident Metal decode router then reads it. The 40 `layers.N.ffn.gate.weight`
tensors are copied byte-for-byte from the released BF16 safetensors, so the
values are exactly the ones the main GGUF stores widened to F32.
Optionally --check-gguf compares them with that GGUF's F32 upper halves.
"""

import argparse
import json
import os
import struct
import sys

ALIGN = 16384
QTYPE_F32, QTYPE_BF16 = 0, 30


def pstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return pstr(k) + struct.pack("<I", 8) + pstr(v)


def kv_u32(k, v):
    return pstr(k) + struct.pack("<II", 4, v)


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def source_tensor(hf, index, name):
    shard = os.path.join(hf, index[name])
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(n))[name]
        start, end = info["data_offsets"]
        f.seek(8 + n + start)
        return info, f.read(end - start)


def gguf_tensor_table(path):
    """name -> (type, dims, absolute offset), header only."""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rs():
            return f.read(rd("<Q")).decode()

        def skip(t):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if t == 8:
                rs()
            elif t == 9:
                et, cnt = rd("<I"), rd("<Q")
                for _ in range(cnt):
                    skip(et)
            else:
                f.read(sizes[t])
        assert f.read(4) == b"GGUF"
        _, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
        alignment = 32
        for _ in range(nkv):
            k, t = rs(), rd("<I")
            if k == "general.alignment":
                alignment = rd("<I")
            else:
                skip(t)
        table = {}
        for _ in range(nt):
            name = rs()
            dims = [rd("<Q") for _ in range(rd("<I"))]
            t, off = rd("<I"), rd("<Q")
            table[name] = (t, dims, off)
        start = (f.tell() + alignment - 1) // alignment * alignment
    return {k: (t, d, start + o) for k, (t, d, o) in table.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-gguf", help="compare with this main GGUF's F32 router (reads 315 MB of payload)")
    a = ap.parse_args()

    index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    tensors = []
    for il in range(a.layers):
        info, data = source_tensor(a.hf, index, f"layers.{il}.ffn.gate.weight")
        if info["dtype"] != "BF16" or len(info["shape"]) != 2:
            sys.exit(f"layers.{il}.ffn.gate.weight: expected 2-D BF16, got {info['dtype']} {info['shape']}")
        rows, cols = info["shape"]
        assert len(data) == rows * cols * 2
        tensors.append((f"blk.{il}.ffn_gate_inp.weight", [cols, rows], data))

    records = [kv_str("general.architecture", "deepseek41-router-bf16"),
               kv_str("general.name", "DeepSeek V4.1 Flash router gates (BF16 sidecar)"),
               kv_str("general.source.revision", a.source_revision),
               kv_u32("general.alignment", ALIGN)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(records)) + b"".join(records)
    offset = 0
    for name, dims, data in tensors:
        header += pstr(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<IQ", QTYPE_BF16, offset)
        offset += align(len(data))
    with open(a.out + ".tmp", "wb") as f:
        f.write(header + bytes(align(len(header)) - len(header)))
        for _, _, data in tensors:
            f.write(data + bytes(align(len(data)) - len(data)))
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {len(tensors)} BF16 tensors, {os.path.getsize(a.out)} bytes")

    if a.check_gguf:
        table = gguf_tensor_table(a.check_gguf)
        with open(a.check_gguf, "rb") as f:
            for name, dims, data in tensors:
                t, gdims, off = table[name]
                if t != QTYPE_F32 or gdims != dims:
                    sys.exit(f"{name}: main GGUF has type {t} dims {gdims}")
                f.seek(off)
                f32 = f.read(len(data) * 2)
                hi = b"".join(f32[i + 2:i + 4] for i in range(0, len(f32), 4))
                lo_zero = all(f32[i] == 0 and f32[i + 1] == 0 for i in range(0, len(f32), 4))
                if hi != data or not lo_zero:
                    sys.exit(f"{name}: F32 router is not the widened BF16 source")
        print(f"check: all {len(tensors)} F32 routers in {a.check_gguf} equal the BF16 values << 16")


if __name__ == "__main__":
    main()

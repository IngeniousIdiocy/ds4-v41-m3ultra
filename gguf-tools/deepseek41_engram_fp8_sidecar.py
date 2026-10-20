#!/usr/bin/env python3
"""Write the DeepSeek V4.1 Engram KV projections as an FP8 GGUF sidecar.

The engine maps it when DS4_V41_ENGRAM_FP8_SIDECAR points at it; the resident
Metal decode Engram projection then reads it. For each Engram layer the original `layers.N.engram.wkv`
E4M3 codes [25600, 6144] and E8M0 32x32 block scales [800, 192] are copied
byte-for-byte from the released safetensors, as two raw byte tensors:
  blk.N.engram_kv.fp8   dims [6144, 25600]  (row-major: 25600 output rows of 6144 codes)
  blk.N.engram_kv.e8m0  dims [192, 800]     (one byte per 32 rows x 32 columns)
The main GGUF's F16 engram_kv is the same codes decoded with FP32 scales, then
rounded to F16, so these bytes are the source values (F16 loses some of them).
"""
import argparse, json, os, struct, sys

ALIGN = 16384
QTYPE_I8 = 24           # raw bytes; the sidecar's architecture string defines their meaning


def pstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v): return pstr(k) + struct.pack("<I", 8) + pstr(v)
def kv_u32(k, v): return pstr(k) + struct.pack("<II", 4, v)
def align(n): return (n + ALIGN - 1) // ALIGN * ALIGN


def source_tensor(hf, index, name):
    with open(os.path.join(hf, index[name]), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(n))[name]
        start, end = info["data_offsets"]
        f.seek(8 + n + start)
        return info, f.read(end - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--layers", default="1,14")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    tensors = []
    for il in (int(x) for x in a.layers.split(",")):
        wi, w = source_tensor(a.hf, index, f"layers.{il}.engram.wkv.weight")
        si, sc = source_tensor(a.hf, index, f"layers.{il}.engram.wkv.scale")
        rows, cols = wi["shape"]
        if wi["dtype"] != "F8_E4M3" or si["dtype"] != "F8_E8M0" or si["shape"] != [(rows + 31) // 32, (cols + 31) // 32]:
            sys.exit(f"layer {il}: unexpected wkv {wi['dtype']} {wi['shape']} / scale {si['dtype']} {si['shape']}")
        tensors.append((f"blk.{il}.engram_kv.fp8", [cols, rows], w))
        tensors.append((f"blk.{il}.engram_kv.e8m0", si["shape"][::-1], sc))
    records = [kv_str("general.architecture", "deepseek41-engram-fp8"),
               kv_str("general.name", "DeepSeek V4.1 Flash Engram KV (E4M3 + E8M0 32x32 sidecar)"),
               kv_str("general.source.revision", a.source_revision),
               kv_u32("general.alignment", ALIGN)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(records)) + b"".join(records)
    off = 0
    for name, dims, data in tensors:
        header += pstr(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<IQ", QTYPE_I8, off)
        off += align(len(data))
    with open(a.out + ".tmp", "wb") as f:
        f.write(header + bytes(align(len(header)) - len(header)))
        for _, _, data in tensors:
            f.write(data + bytes(align(len(data)) - len(data)))
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {len(tensors)} tensors, {os.path.getsize(a.out)} bytes")


if __name__ == "__main__":
    main()

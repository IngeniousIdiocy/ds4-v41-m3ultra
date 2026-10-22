#!/usr/bin/env python3
"""Write the DeepSeek V4.1 indexer query projection (indexer.wq_b) as an FP8 GGUF sidecar.

The engine loads it when --v41-sidecar names it; the resident Metal decode
index-source q projection then reads it. For each index-source layer
(2, 8, 14, 20, 24, 28, 32, 36) the released `layers.N.attn.indexer.wq_b` E4M3 codes and E8M0 32x32 block
scales are copied byte-for-byte from the safetensors as raw byte tensors (GGUF dims are [cols, rows]):
  blk.N.indexer.attn_q_b.fp8   [1280, 4096]   4096 output rows (32 heads x 128) of 1280 codes
  blk.N.indexer.attn_q_b.e8m0  [40, 128]      one byte per 32 rows x 32 columns
The main GGUF's F16 indexer.attn_q_b is these values decoded with FP32 scales, then rounded to F16
(gguf-tools/deepseek41_quantize.py: ("indexer.attn_q_b.weight", "indexer.wq_b.weight"), no permutation),
so these bytes are the source values. Tensors are streamed one at a time (RSS ~ one tensor).
The engine checks the written file's size and SHA-256 against the screened artifact.
"""
import argparse, hashlib, json, os, struct, sys

ALIGN = 16384
QTYPE_I8 = 24           # raw bytes; the sidecar's architecture string defines their meaning
SHAPES = {"indexer.wq_b": ([4096, 1280], [128, 40], "indexer.attn_q_b")}


def pstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v): return pstr(k) + struct.pack("<I", 8) + pstr(v)
def kv_u32(k, v): return pstr(k) + struct.pack("<II", 4, v)
def align(n): return (n + ALIGN - 1) // ALIGN * ALIGN


def source_info(hf, index, name):
    path = os.path.join(hf, index[name])
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(n))[name]
    return path, 8 + n + info["data_offsets"][0], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    layers = (sorted({int(k.split(".")[1]) for k in index if k.startswith("layers.") and k.endswith(".attn.indexer.wq_b.weight")})
              if a.layers == "all" else [int(x) for x in a.layers.split(",")])
    plan = []   # (gguf name, gguf dims, source path, byte offset, byte count)
    for il in layers:
        for t, (wshape, sshape, gname) in SHAPES.items():
            wp, wo, wi = source_info(a.hf, index, f"layers.{il}.attn.{t}.weight")
            sp, so, si = source_info(a.hf, index, f"layers.{il}.attn.{t}.scale")
            if wi["dtype"] != "F8_E4M3" or wi["shape"] != wshape or si["dtype"] != "F8_E8M0" or si["shape"] != sshape:
                sys.exit(f"layer {il} {t}: unexpected {wi['dtype']} {wi['shape']} / scale {si['dtype']} {si['shape']}")
            plan.append((f"blk.{il}.{gname}.fp8", wshape[::-1], wp, wo, wshape[0] * wshape[1]))
            plan.append((f"blk.{il}.{gname}.e8m0", sshape[::-1], sp, so, sshape[0] * sshape[1]))
    records = [kv_str("general.architecture", "deepseek41-indexer-fp8"),
               kv_str("general.name", "DeepSeek V4.1 Flash indexer query projection indexer.wq_b (E4M3 + E8M0 32x32 sidecar)"),
               kv_str("general.source.revision", a.source_revision),
               kv_u32("general.alignment", ALIGN)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records)) + b"".join(records)
    off = 0
    for name, dims, _, _, nbytes in plan:
        header += pstr(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<IQ", QTYPE_I8, off)
        off += align(nbytes)
    manifest, hist, nan_free = [], [0] * 256, 0
    with open(a.out + ".tmp", "wb") as f:
        f.write(header + bytes(align(len(header)) - len(header)))
        for name, _, path, pos, nbytes in plan:
            with open(path, "rb") as src:
                src.seek(pos); data = src.read(nbytes)
            if len(data) != nbytes: sys.exit(f"{name}: short read")
            if name.endswith(".fp8") and (data.count(0x7f) or data.count(0xff)): sys.exit(f"{name}: E4M3 NaN code")
            if name.endswith(".e8m0") and (data.count(0) or max(data) > 246): sys.exit(f"{name}: E8M0 byte 0 or > 246 (outside the kernel fold)")
            if name.endswith(".e8m0"):
                for b in range(256): hist[b] += data.count(b)
            else: nan_free += nbytes
            manifest.append((name, nbytes, hashlib.sha256(data).hexdigest()))
            f.write(data + bytes(align(nbytes) - nbytes))
            del data
    os.replace(a.out + ".tmp", a.out)
    with open(a.out + ".manifest", "w") as m:
        for name, nbytes, h in manifest: m.write(f"{h}  {nbytes}  {name}\n")
    print("codes checked (no 0x7f/0xff):", nan_free, "; E8M0 histogram:", {b: n for b, n in enumerate(hist) if n})
    print(f"wrote {a.out}: {len(plan)} tensors from layers {layers[0]}..{layers[-1]} ({len(layers)}), {os.path.getsize(a.out)} bytes")


if __name__ == "__main__":
    main()

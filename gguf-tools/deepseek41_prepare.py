#!/usr/bin/env python3
"""Prepare DeepSeek V4.1 Flash 1017 for this engine from the public release.

  prepare  deepseek41_prepare.py --shards DIR --dspark-source DIR --out DIR
  plan     deepseek41_prepare.py --dry-run --shards DIR --dspark-source DIR --out DIR
  verify   deepseek41_prepare.py --verify --out DIR [--shards DIR] [--dspark-source DIR]
  digest   deepseek41_prepare.py --digest FILE.gguf [FILE.gguf ...]

--shards holds the nine llama.cpp-split MXFP4 GGUFs of antirez/deepseek-v4.1-flash-gguf.
They are checked against the pinned sizes and SHA-256 sums while they are merged into one
GGUF with 16 KiB alignment and no split keys: the tensor bytes, tensor order and metadata
are the release's own; only general.alignment changes and the split.* keys go.

--dspark-source holds config.json, tokenizer.json and model-0004{4,5,6}-of-00048.safetensors
of deepseek-ai/DeepSeek-V4.1-Flash-1017 (the 2,401 mtp.* tensors). The drafter is converted by
deepseek41_quantize.py --quant mxfp4 --dspark-out, the converter that made the production
drafter, and carries dspark.target_architecture=deepseek41.

Every expected value lives in deepseek41_prepare.json beside this script (format
deepseek41-prepare/1). A run writes deepseek41-prepare-run.json into --out.
Reads use F_NOCACHE (macOS) or POSIX_FADV_DONTNEED; nothing is memory-mapped.
Exit: 0 ok; 1 a mismatch; 2 usage, input or I/O error; 3 no mismatch but a value is unpinned.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED = os.path.join(HERE, "deepseek41_prepare.json")
FORMAT = "deepseek41-prepare/1"
ALIGN = 16384
CHUNK = 64 << 20
RESERVE = 32 << 30
MAX_HEADER = 256 << 20
# ggml type id -> (elements per block, bytes per block) for the types these files hold
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 24: (1, 1), 30: (1, 2), 39: (32, 17)}
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Mismatch(Exception):
    pass


def align(n, a=ALIGN):
    return (n + a - 1) // a * a


def nocache(fd):
    if hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)


def drop(fd, off, n):
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)


def pread_all(fd, n, off):
    data = os.pread(fd, n, off)
    if len(data) != n:
        raise OSError(f"short read at {off}")
    drop(fd, off, n)
    return data


def pwrite_all(fd, data, off):
    view = memoryview(data)
    while view:
        done = os.pwrite(fd, view, off)
        view, off = view[done:], off + done


# ---- GGUF headers (bounded reads) ----

class Cursor:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise EOFError
        self.o += n
        return self.b[self.o - n:self.o]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return bytes(self.take(self.u64())).decode("utf-8")

    def skip(self, t):
        if t in FIXED:
            self.take(FIXED[t])
        elif t == 8:
            self.take(self.u64())
        elif t == 9:
            at, n = self.u32(), self.u64()
            if at in FIXED:
                self.take(FIXED[at] * n)
            else:
                for _ in range(n):
                    self.skip(at)
        else:
            raise ValueError(f"unknown GGUF value type {t}")


def tensor_bytes(name, ggml_type, dims):
    if ggml_type not in TYPES:
        raise ValueError(f"{name}: unsupported ggml type {ggml_type}")
    per, size = TYPES[ggml_type]
    ne = 1
    for d in dims:
        ne *= d
    if ne % per:
        raise ValueError(f"{name}: {ne} elements are not whole blocks")
    return ne // per * size


def parse(buf):
    c = Cursor(buf)
    if c.take(4) != b"GGUF" or c.u32() != 3:
        raise ValueError("not a GGUF v3 file")
    nt, nkv = c.u64(), c.u64()
    kvs = []
    for _ in range(nkv):
        start = c.o
        key, t = c.string(), c.u32()
        c.skip(t)
        kvs.append((key, t, bytes(buf[start:c.o])))
    tensors = []
    for _ in range(nt):
        start = c.o
        name = c.string()
        dims = [c.u64() for _ in range(c.u32())]
        ggml_type = c.u32()
        prefix = bytes(buf[start:c.o])
        tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": c.u64(), "prefix": prefix,
                        "nbytes": tensor_bytes(name, ggml_type, dims)})
    alignment = 32
    for key, t, raw in kvs:
        if key == "general.alignment":
            if t != 4:
                raise ValueError("general.alignment is not a u32")
            alignment = struct.unpack("<I", raw[-4:])[0]
    return {"kvs": kvs, "tensors": tensors, "alignment": alignment, "header_end": c.o,
            "data_start": align(c.o, alignment)}


def read_header(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        cap = 1 << 20
        while True:
            buf = os.pread(fd, min(cap, size), 0)
            try:
                h = parse(memoryview(buf))
                break
            except EOFError:
                if cap >= MAX_HEADER or cap >= size:
                    raise ValueError(f"{path}: truncated or oversized GGUF header")
                cap *= 4
    finally:
        os.close(fd)
    h["size"], h["path"] = size, path
    end = 0
    for t in sorted(h["tensors"], key=lambda t: t["offset"]):
        if t["offset"] < end or t["offset"] % h["alignment"]:
            raise ValueError(f"{path}: {t['name']} overlaps or is misaligned")
        end = t["offset"] + t["nbytes"]
    if h["data_start"] + end > size:
        raise ValueError(f"{path}: tensor data runs past the end of the file")
    return h


def kv_value(h, key):
    for k, t, raw in h["kvs"]:
        if k == key:
            if t in (2, 4, 5):
                fmt = {2: "<H", 4: "<I", 5: "<i"}[t]
                return struct.unpack(fmt, raw[-struct.calcsize(fmt):])[0]
            if t == 8:
                at = 8 + len(key.encode()) + 4
                n = struct.unpack("<Q", raw[at:at + 8])[0]
                return raw[at + 8:at + 8 + n].decode("utf-8")
            raise ValueError(f"{key}: unexpected type {t}")
    return None


def digest_line(t, sha):
    return f"{t['name']}\t{t['type']}\t{','.join(map(str, t['dims']))}\t{t['nbytes']}\t{sha}\n"


def tensor_digest(tensors, shas):
    h = hashlib.sha256()
    for t in tensors:
        h.update(digest_line(t, shas[t["name"]]).encode())
    return h.hexdigest()


# ---- digest of any GGUF: header hash, per-tensor hashes, zero padding ----

def digest_gguf(path, jobs):
    h = read_header(path)
    order = sorted(h["tensors"], key=lambda t: t["offset"])
    fd = os.open(path, os.O_RDONLY)
    nocache(fd)
    try:
        header_sha = hashlib.sha256(pread_all(fd, h["data_start"], 0)).hexdigest()

        def one(i):
            t = order[i]
            start = h["data_start"] + t["offset"]
            stop = h["data_start"] + order[i + 1]["offset"] if i + 1 < len(order) else h["size"]
            sha, pos, end = hashlib.sha256(), start, start + t["nbytes"]
            while pos < end:
                n = min(CHUNK, end - pos)
                sha.update(pread_all(fd, n, pos))
                pos += n
            pad = pread_all(fd, stop - end, end) if stop > end else b""
            return t["name"], sha.hexdigest(), pad.count(0) == len(pad)

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            # largest first keeps the long tensors from finishing last
            results = list(pool.map(one, sorted(range(len(order)), key=lambda i: -order[i]["nbytes"])))
    finally:
        os.close(fd)
    shas = {name: sha for name, sha, _ in results}
    bad_pad = sorted(name for name, _, ok in results if not ok)
    return {"size": h["size"], "data_start": h["data_start"], "alignment": h["alignment"],
            "tensors": len(h["tensors"]), "header_sha256": header_sha,
            "tensor_digest": tensor_digest(h["tensors"], shas), "nonzero_padding_after": bad_pad,
            "tensor_sha256": {t["name"]: shas[t["name"]] for t in h["tensors"]}}


def sha256_file(path):
    fd = os.open(path, os.O_RDONLY)
    nocache(fd)
    try:
        sha, pos, size = hashlib.sha256(), 0, os.fstat(fd).st_size
        while pos < size:
            n = min(CHUNK, size - pos)
            sha.update(pread_all(fd, n, pos))
            pos += n
        return sha.hexdigest()
    finally:
        os.close(fd)


def git_oid(path):
    with open(path, "rb") as fp:
        data = fp.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---- inputs ----

def check_present(directory, files, what):
    for f in files:
        path = os.path.join(directory, f["name"])
        if not os.path.isfile(path):
            raise ValueError(f"{what}: missing {path}")
        if os.path.getsize(path) != f["size"]:
            raise ValueError(f"{what}: {path} is {os.path.getsize(path)} bytes, expected {f['size']}")


def check_dspark_small_files(directory, pin):
    for f in pin["files"]:
        if "git_oid" in f and git_oid(os.path.join(directory, f["name"])) != f["git_oid"]:
            raise Mismatch(f"dspark source: {f['name']} is not git blob {f['git_oid']}")


def hash_inputs(directory, files, jobs, log):
    names = [f["name"] for f in files if "sha256" in f]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        got = dict(zip(names, pool.map(lambda n: sha256_file(os.path.join(directory, n)), names)))
    bad = [f["name"] for f in files if "sha256" in f and got[f["name"]] != f["sha256"]]
    for n in names:
        log(f"  {n} {got[n]}{'  MISMATCH' if n in bad else ''}")
    return got, bad


# ---- main GGUF: merge the split release ----

def merged_layout(shard_dir, pin):
    shards = [read_header(os.path.join(shard_dir, f["name"])) for f in pin["files"]]
    for i, h in enumerate(shards):
        if kv_value(h, "split.no") != i or kv_value(h, "split.count") != len(shards):
            raise ValueError(f"{h['path']}: split.no/split.count do not match its place in the set")
        if kv_value(h, "split.tensors.count") != pin["tensors"]:
            raise ValueError(f"{h['path']}: split.tensors.count is not {pin['tensors']}")
    kvs = []
    for key, t, raw in shards[0]["kvs"]:
        if key.startswith("split."):
            continue
        if key == "general.alignment":
            raw = raw[:-4] + struct.pack("<I", ALIGN)
        kvs.append(raw)
    tensors, seen, offset = [], set(), 0
    for h in shards:
        for t in h["tensors"]:
            if t["name"] in seen:
                raise ValueError(f"duplicate tensor {t['name']}")
            seen.add(t["name"])
            tensors.append(dict(t, shard=h, out_offset=offset))
            offset += align(t["nbytes"])
    if len(tensors) != pin["tensors"]:
        raise ValueError(f"the shards hold {len(tensors)} tensors, expected {pin['tensors']}")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    header += b"".join(t["prefix"] + struct.pack("<Q", t["out_offset"]) for t in tensors)
    data_start = align(len(header))
    return shards, tensors, header + bytes(data_start - len(header)), data_start + offset


def copy_shard(h, tensors, out_fd, data_start):
    fd = os.open(h["path"], os.O_RDONLY)
    nocache(fd)
    whole, pos, shas = hashlib.sha256(), 0, {}
    try:
        def feed(upto, sha=None, delta=None):
            nonlocal pos
            while pos < upto:
                n = min(CHUNK, upto - pos)
                data = pread_all(fd, n, pos)
                whole.update(data)
                if sha is not None:
                    sha.update(data)
                    pwrite_all(out_fd, data, pos + delta)
                pos += n

        for t in sorted(tensors, key=lambda t: t["offset"]):
            start = h["data_start"] + t["offset"]
            feed(start)
            sha = hashlib.sha256()
            feed(start + t["nbytes"], sha, data_start + t["out_offset"] - start)
            shas[t["name"]] = sha.hexdigest()
        feed(h["size"])
    finally:
        os.close(fd)
    return whole.hexdigest(), shas


def prepare_main(args, pin, out_path, log):
    spin = pin["shards"]
    shards, tensors, header, total = merged_layout(args.shards, spin)
    want = pin["outputs"]["main"]
    header_sha = hashlib.sha256(header).hexdigest()
    if total != want["size"] or len(header) != want["data_start"]:
        raise Mismatch(f"merged layout is {total} B / data start {len(header)}, "
                       f"expected {want['size']} / {want['data_start']}")
    if want["header_sha256"] and header_sha != want["header_sha256"]:
        raise Mismatch(f"merged header {header_sha} is not the pinned {want['header_sha256']}")
    partial = out_path + ".partial"
    if os.path.exists(partial):
        raise ValueError(f"{partial} exists from an interrupted run; remove it and run again")
    free = shutil.disk_usage(os.path.dirname(out_path)).free
    if free < total + RESERVE:
        raise ValueError(f"{free} B free beside {out_path}; the merge needs {total} B plus 32 GiB")
    started = time.monotonic()
    out_fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        nocache(out_fd)
        os.ftruncate(out_fd, total)
        pwrite_all(out_fd, header, 0)
        by_shard = {id(h): [t for t in tensors if t["shard"] is h] for h in shards}
        order = sorted(shards, key=lambda h: -h["size"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(copy_shard, h, by_shard[id(h)], out_fd, len(header)): h for h in order}
            results = {}
            for f in concurrent.futures.as_completed(futures):
                h = futures[f]
                results[h["path"]] = f.result()
                log(f"  merged {os.path.basename(h['path'])} {results[h['path']][0]} "
                    f"({time.monotonic() - started:.0f} s)")
        os.fsync(out_fd)
    finally:
        os.close(out_fd)
    shard_sha = {os.path.basename(p): r[0] for p, r in results.items()}
    bad = [f["name"] for f in spin["files"] if shard_sha[f["name"]] != f["sha256"]]
    if bad:
        raise Mismatch(f"shard SHA-256 differs from the pin: {bad}; {partial} is left for inspection")
    shas = {}
    for _, s in results.values():
        shas.update(s)
    digest = tensor_digest(tensors, shas)
    if want["tensor_digest"] and digest != want["tensor_digest"]:
        raise Mismatch(f"merged tensor digest {digest} is not the pinned {want['tensor_digest']}")
    os.rename(partial, out_path)
    # The copy pass hashed every tensor; --verify re-reads the written file.
    return {"size": total, "data_start": len(header), "alignment": ALIGN, "tensors": len(tensors),
            "header_sha256": header_sha, "tensor_digest": digest, "nonzero_padding_after": [],
            "tensor_sha256": {t["name"]: shas[t["name"]] for t in tensors},
            "shards_sha256": shard_sha, "seconds": round(time.monotonic() - started, 1)}


# ---- DSpark drafter: stage the mtp.* source, run the converter ----

def safetensors_names(path):
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        if n > MAX_HEADER:
            raise ValueError(f"{path}: oversized safetensors header")
        return [k for k in json.loads(fp.read(n)) if k != "__metadata__"]


def stage_dspark(src, stage, pin):
    os.makedirs(stage, exist_ok=True)
    weight_map = {}
    for f in pin["files"]:
        target, link = os.path.abspath(os.path.join(src, f["name"])), os.path.join(stage, f["name"])
        if os.path.lexists(link):
            if os.path.realpath(link) != os.path.realpath(target):
                raise ValueError(f"{link} points elsewhere; remove the stage directory {stage}")
        else:
            os.symlink(target, link)
        if f["name"].endswith(".safetensors"):
            for name in safetensors_names(target):
                weight_map[name] = f["name"]
    if len(weight_map) != pin["mtp_tensors"] or any(not n.startswith("mtp.") for n in weight_map):
        raise Mismatch(f"the DSpark source shards hold {len(weight_map)} tensors, "
                       f"expected {pin['mtp_tensors']} mtp.* tensors")
    index = os.path.join(stage, "model.safetensors.index.json")
    with open(index + ".tmp", "w", encoding="utf-8") as fp:
        json.dump({"metadata": {"note": "mtp.* subset staged by deepseek41_prepare.py"},
                   "weight_map": dict(sorted(weight_map.items()))}, fp, indent=1)
    os.replace(index + ".tmp", index)


def quants_library():
    name = "libds4quants." + ("dylib" if sys.platform == "darwin" else "so")
    return os.path.join(HERE, name), name


def prepare_dspark(args, pin, out_path, log):
    dpin = pin["dspark_source"]
    stage = os.path.join(args.out, ".dspark-source-" + dpin["revision"][:8])
    stage_dspark(args.dspark_source, stage, dpin)
    lib, name = quants_library()
    if not os.path.isfile(lib):
        log(f"  building {name}")
        subprocess.run(["make", "-C", HERE, name], check=True)
    cmd = [sys.executable, os.path.join(HERE, "deepseek41_quantize.py"), "--hf", stage,
           "--dspark-out", out_path, "--source-revision", dpin["revision"], "--quant", "mxfp4",
           "--threads", str(args.threads)]
    if os.path.exists(out_path + ".partial"):
        cmd.append("--resume")
    log("  " + " ".join(cmd))
    started = time.monotonic()
    subprocess.run(cmd, check=True)
    return {"seconds": round(time.monotonic() - started, 1), "command": cmd[1:]}


# ---- verification against the pin ----

def compare(label, got, want, log):
    status = 0
    for key in ("size", "data_start", "tensors", "header_sha256", "tensor_digest"):
        if want.get(key) is None:
            log(f"  {label} {key}: {got[key]} (UNPINNED)")
            status = max(status, 3)
        elif got[key] != want[key]:
            log(f"  {label} {key}: {got[key]} MISMATCH, pinned {want[key]}")
            status = 1
    if got["alignment"] != ALIGN:
        log(f"  {label} general.alignment {got['alignment']} MISMATCH, expected {ALIGN}")
        status = 1
    if got["nonzero_padding_after"]:
        log(f"  {label} nonzero padding after {got['nonzero_padding_after'][:5]} MISMATCH")
        status = 1
    if status == 0:
        log(f"  {label}: OK")
    return 1 if status == 1 else status


def verify(args, pin, log):
    status = 0
    for role in ("main", "dspark"):
        path = os.path.join(args.out, pin["outputs"][role]["name"])
        if not os.path.isfile(path):
            log(f"  {role}: missing {path}")
            status = 1
            continue
        status = max(status, compare(role, digest_gguf(path, args.jobs), pin["outputs"][role], log))
    if args.shards:
        check_present(args.shards, pin["shards"]["files"], "shards")
        _, bad = hash_inputs(args.shards, pin["shards"]["files"], args.jobs, log)
        status = 1 if bad else status
    if args.dspark_source:
        check_present(args.dspark_source, pin["dspark_source"]["files"], "dspark source")
        check_dspark_small_files(args.dspark_source, pin["dspark_source"])
        _, bad = hash_inputs(args.dspark_source, pin["dspark_source"]["files"], args.jobs, log)
        status = 1 if bad else status
    return status


# ---- run record, plan ----

def file_sha(path):
    return sha256_file(path) if os.path.isfile(path) else None


def tool_identity():
    ident = {"python": sys.version.split()[0], "platform": platform.platform(), "files": {}}
    for f in ("deepseek41_prepare.py", "deepseek41_prepare.json", "deepseek41_quantize.py",
              "deepseek41_metadata.py", "glm53_quantize.py", "glm53_manifest.py", "quants.c", "quants.h",
              "Makefile", quants_library()[1]):
        ident["files"][f] = file_sha(os.path.join(HERE, f))
    for mod in ("numpy", "tokenizers", "sympy"):
        try:
            ident[mod] = __import__(mod).__version__
        except ImportError:
            ident[mod] = None
    cc = shutil.which("cc")
    out = subprocess.run([cc, "--version"], capture_output=True, text=True) if cc else None
    ident["cc"] = out.stdout.splitlines()[0] if out and out.returncode == 0 and out.stdout else None
    return ident


def plan(args, pin, log):
    need = 0
    for role in ("main", "dspark"):
        o = pin["outputs"][role]
        exists = os.path.isfile(os.path.join(args.out, o["name"]))
        need += 0 if exists else o["size"]
        log(f"  {role}: {o['name']}, {o['size'] / 2**30:.1f} GiB{' (exists; kept)' if exists else ''}")
    check_present(args.shards, pin["shards"]["files"], "shards")
    check_present(args.dspark_source, pin["dspark_source"]["files"], "dspark source")
    reads = sum(f["size"] for f in pin["shards"]["files"]) + \
        sum(f["size"] for f in pin["dspark_source"]["files"] if f["name"].endswith(".safetensors"))
    where = args.out
    while not os.path.isdir(where):
        where = os.path.dirname(os.path.abspath(where))
    free = shutil.disk_usage(where).free
    log(f"  inputs present with pinned sizes; reads {reads / 1e9:.1f} GB, writes {need / 1e9:.1f} GB")
    log(f"  free beside --out {free / 1e9:.1f} GB, needed {(need + RESERVE) / 1e9:.1f} GB with a 32 GiB reserve")
    missing = [m for m in ("numpy", "tokenizers", "sympy") if importlib.util.find_spec(m) is None]
    log(f"  python modules for the drafter: {'missing ' + ', '.join(missing) if missing else 'present'}")
    log(f"  quantizer library: {'present' if os.path.isfile(quants_library()[0]) else 'built by make on first run'}")
    return 2 if free < need + RESERVE or missing else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", help="directory with the nine public split GGUFs")
    p.add_argument("--dspark-source", help="directory with the DSpark source files of the 1017 checkpoint")
    p.add_argument("--out", help="output directory")
    p.add_argument("--verify", action="store_true", help="re-read the outputs (and given inputs) against the pin")
    p.add_argument("--dry-run", action="store_true", help="check inputs, space and tools; write nothing")
    p.add_argument("--digest", nargs="+", metavar="GGUF", help="print header and tensor digests of GGUF files")
    p.add_argument("--jobs", type=int, default=3, help="files or tensors read in parallel (default 3)")
    p.add_argument("--threads", type=int, default=8, help="converter threads for the drafter (default 8)")
    args = p.parse_args()
    log = lambda s: print(s, flush=True)
    with open(PINNED, encoding="utf-8") as fp:
        pin = json.load(fp)
    if pin.get("format") != FORMAT:
        p.error(f"{PINNED} is not {FORMAT}")
    if not 1 <= args.jobs <= 16:
        p.error("--jobs must be between 1 and 16")

    if args.digest:
        for path in args.digest:
            d = digest_gguf(path, args.jobs)
            log(json.dumps(dict(d, path=path), indent=1))
        return 0
    if not args.out:
        p.error("--out is required")
    if args.verify:
        log(f"verify {args.out} against {FORMAT} ({pin['release']})")
        return verify(args, pin, log)
    if not args.shards or not args.dspark_source:
        p.error("--shards and --dspark-source are required")
    if args.dry_run:
        log(f"plan for {args.out}")
        return plan(args, pin, log)
    os.makedirs(args.out, exist_ok=True)

    check_present(args.shards, pin["shards"]["files"], "shards")
    check_present(args.dspark_source, pin["dspark_source"]["files"], "dspark source")
    check_dspark_small_files(args.dspark_source, pin["dspark_source"])
    run = {"format": FORMAT, "release": pin["release"], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tools": tool_identity(), "inputs": {"shards": os.path.abspath(args.shards),
                                                 "dspark_source": os.path.abspath(args.dspark_source)}}
    main_path = os.path.join(args.out, pin["outputs"]["main"]["name"])
    dspark_path = os.path.join(args.out, pin["outputs"]["dspark"]["name"])
    if os.path.exists(main_path):
        log(f"main: {main_path} exists; kept (check it with --verify)")
    else:
        log(f"main: merging {len(pin['shards']['files'])} shards into {main_path}")
        run["main"] = prepare_main(args, pin, main_path, log)
    if os.path.exists(dspark_path):
        log(f"dspark: {dspark_path} exists; kept (check it with --verify)")
    else:
        log("dspark: hashing the source shards")
        got, bad = hash_inputs(args.dspark_source, pin["dspark_source"]["files"], args.jobs, log)
        if bad:
            raise Mismatch(f"DSpark source SHA-256 differs from the pin: {bad}")
        log(f"dspark: converting into {dspark_path}")
        run["dspark"] = prepare_dspark(args, pin, dspark_path, log)
    log("checking the outputs")
    status = 0
    if "dspark" in run:
        d = digest_gguf(dspark_path, args.jobs)
        run["dspark"].update(d)
    for role in ("main", "dspark"):
        if role in run:
            status = max(status, compare(role, run[role], pin["outputs"][role], log))
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = os.path.join(args.out, "deepseek41-prepare-run.json")
    with open(record + ".tmp", "w", encoding="utf-8") as fp:
        json.dump(run, fp, indent=1, sort_keys=True)
    os.replace(record + ".tmp", record)
    log(f"run record: {record}")
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Mismatch as error:
        sys.exit(print(f"deepseek41-prepare: MISMATCH: {error}", file=sys.stderr) or 1)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(print(f"deepseek41-prepare: {error}", file=sys.stderr) or 2)

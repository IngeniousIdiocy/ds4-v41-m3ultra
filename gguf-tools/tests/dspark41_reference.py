#!/usr/bin/env python3
"""CPU fp32 reference oracle for the DeepSeek V4.1 Flash DSpark drafter.

The released `inference/` stack cannot run here: `sparse_attn`, `hc_split_sinkhorn`
and `act_quant` are tilelang kernels that only compile for CUDA. This file is the
extracted equivalent of `Transformer.forward_spec` -- `DSparkBlock.forward_embed`,
`Block.forward` over the three `mtp.*` stages, and `DSparkBlock.forward_head` --
re-expressed in plain CPU PyTorch, with the three kernels reimplemented from their
tilelang source. Every weight is dequantized with `deepseek41_quantize.NativeQuantizer`,
the same routine the converter uses, so the oracle and the GGUF share one source of
truth for FP8-E4M3 / packed-FP4 / E8M0 decoding.

The drafter forward is fully defined for any `[b, s, 3*5120]` main_hidden, so the
550B backbone is never loaded: main_hidden is synthesized from a fixed seed. The
token embedding and the tied output head are *not* in the drafter's own shards
(`embed.weight` lives in shard 2 and `head.weight` in shard 43, neither of which is
part of the DSpark download), so they are read from the main model's Q4 GGUF --
which is also exactly what the ds4 runtime binds at decode time.

Dumps are raw little-endian f32/i32 payloads plus a manifest, one directory per
`start_pos`, ready for a C test to upload.
"""

import argparse
import hashlib
import json
import os
import resource
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from deepseek41_quantize import NativeQuantizer, scale_name  # noqa: E402
from glm53_quantize import SourceDB, load_safetensors_header  # noqa: E402

NOISE_TOKEN_ID = 128799
BLOCK_SIZE = 5
WINDOW = 128
HC = 4
SINKHORN_ITERS = 20
HC_EPS = 1.0e-6
NORM_EPS = 1.0e-20
ROPE_THETA = 10000.0
ROPE_HEAD_DIM = 64
HEAD_DIM = 512
N_HEADS = 64
O_GROUPS = 8
O_LORA_RANK = 1024
DIM = 5120
VOCAB = 129280
MARKOV_RANK = 256
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
TOPK = 3
GATE_TEMP = 1.0
FP8_MAX = 448.0

DEFAULT_POSITIONS = (1, 5, 127, 128, 129, 200)


def fail(message):
    raise SystemExit(f"dspark41-reference: {message}")


class MtpSourceDB(SourceDB):
    """`SourceDB` restricted to the shards that actually hold `mtp.*` tensors.

    The DSpark download is shards 44-46 of 48; the stock constructor insists on
    every shard named by the index, so it cannot open this directory.
    """

    def __init__(self, hf_dir, prefix="mtp."):
        import threading

        self.hf_dir = hf_dir
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        with open(index_path, "rb") as fp:
            document = json.load(fp)
        full_map = document.get("weight_map")
        if not isinstance(full_map, dict) or not full_map:
            fail(f"{index_path}: missing or empty weight_map")
        self.declared_bytes = document.get("metadata", {}).get("total_size")
        self.weight_map = {n: s for n, s in full_map.items() if n.startswith(prefix)}
        if not self.weight_map:
            fail(f"{index_path}: no {prefix}* tensors")
        self.tensors = {}
        self._fds = {}
        self._fd_lock = threading.Lock()
        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if not os.path.isfile(path):
                fail(f"missing source shard {path}")
            for name, info in load_safetensors_header(path).items():
                if not name.startswith(prefix):
                    continue
                if self.weight_map.get(name) != shard:
                    fail(f"index assigns {name} to {self.weight_map.get(name)!r}, not {shard}")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                self.tensors[name] = dict(info, shard=shard)
        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source headers are incomplete; first missing tensor is {missing[0]}")
        validate_mtp_scales(self.tensors)


def validate_mtp_scales(tensors):
    """Mirror of `deepseek41_quantize.validate_scales` for the `mtp.*` namespace."""
    for name, info in tensors.items():
        dtype, shape = info["dtype"], info["shape"]
        if dtype not in ("F8_E4M3", "I8") or not name.endswith(".weight"):
            continue
        if len(shape) != 2:
            fail(f"{name}: expected a matrix")
        expected = [shape[0], shape[1] // 16] if dtype == "I8" else [(d + 31) // 32 for d in shape]
        scale = tensors.get(scale_name(name))
        if not scale or scale["dtype"] != "F8_E8M0" or scale["shape"] != expected:
            fail(f"{name}: expected E8M0 scales {expected}")


# ---------------------------------------------------------------------------
# GGUF reader: token_embd (F16) and the tied output head (Q8_0) from the main file
# ---------------------------------------------------------------------------

GGUF_F32, GGUF_F16, GGUF_Q8_0 = 0, 1, 8
_GGUF_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_tensor_directory(path):
    with open(path, "rb") as fp:
        def u32():
            return struct.unpack("<I", fp.read(4))[0]

        def u64():
            return struct.unpack("<Q", fp.read(8))[0]

        def gstr():
            return fp.read(u64()).decode("utf-8")

        def skip(kind):
            if kind == 8:
                fp.seek(u64(), 1)
            elif kind == 9:
                element, count = u32(), u64()
                for _ in range(count):
                    skip(element)
            else:
                fp.seek(_GGUF_SCALAR_BYTES[kind], 1)

        if fp.read(4) != b"GGUF" or u32() != 3:
            fail(f"{path}: expected GGUF v3")
        n_tensors, n_kv = u64(), u64()
        alignment = 32
        for _ in range(n_kv):
            key, kind = gstr(), u32()
            if key == "general.alignment" and kind == 4:
                alignment = u32()
            else:
                skip(kind)
        directory = {}
        for _ in range(n_tensors):
            name = gstr()
            dims = [u64() for _ in range(u32())]
            directory[name] = (dims, u32(), u64())
        data_start = (fp.tell() + alignment - 1) // alignment * alignment
    return directory, data_start


def gguf_read_matrix(path, name, want_rows=None, want_cols=None, out_dtype=torch.float32):
    """Return a [rows, cols] tensor for a GGUF F16 or Q8_0 matrix.

    Q8_0 is decoded in row chunks: `output.weight` is 129280x5120 and a whole-tensor
    dequantization would hold three copies of 2.6 GiB at once.
    """
    directory, data_start = gguf_tensor_directory(path)
    if name not in directory:
        fail(f"{path}: missing tensor {name}")
    dims, qtype, offset = directory[name]
    if len(dims) != 2:
        fail(f"{path}: {name} is not a matrix")
    cols, rows = dims  # GGUF stores [ne0, ne1] = [cols, rows]
    if (want_rows is not None and rows != want_rows) or (want_cols is not None and cols != want_cols):
        fail(f"{path}: {name} has shape [{rows}, {cols}], expected [{want_rows}, {want_cols}]")
    with open(path, "rb") as fp:
        if qtype == GGUF_F16:
            fp.seek(data_start + offset)
            raw = bytearray(fp.read(rows * cols * 2))
            return torch.frombuffer(raw, dtype=torch.float16).view(rows, cols).to(out_dtype)
        if qtype == GGUF_Q8_0:
            if cols % 32:
                fail(f"{path}: {name} width {cols} is not a multiple of 32")
            blocks = cols // 32
            out = torch.empty(rows, cols, dtype=out_dtype)
            fp.seek(data_start + offset)
            chunk = 4096
            for start in range(0, rows, chunk):
                count = min(chunk, rows - start)
                raw = bytearray(fp.read(count * blocks * 34))
                buf = torch.frombuffer(raw, dtype=torch.uint8).view(count, blocks, 34)
                scales = buf[:, :, :2].contiguous().view(torch.float16).float()
                values = buf[:, :, 2:].contiguous().view(torch.int8).float()
                out[start:start + count] = (values * scales).view(count, cols).to(out_dtype)
            return out
    fail(f"{path}: unsupported quant type {qtype} for {name}")


# ---------------------------------------------------------------------------
# Quantized-reference mode: read the drafter's weights from the Stage-1 support
# GGUF exactly as the Metal path binds them, and optionally apply V4.1's BF16
# activation boundaries.  This is the arm that separates "the Metal forward is
# wrong" from "Q4_K requantization of FP4 experts costs this much".
# ---------------------------------------------------------------------------

GGUF_F32 = 0
GGUF_Q4_K = 12

BF16 = False


def rnd(x):
    """dsv41_bf16(): V4.1's activation boundary, a pure function of the word."""
    if not BF16:
        return x
    return x.to(torch.bfloat16).to(torch.float32)


def _q4_k_scale_min(scales):
    """get_scale_min_k4 over all 8 sub-blocks at once. scales: [..., 12] uint8."""
    q = scales.long()
    d = torch.empty(scales.shape[:-1] + (8,), dtype=torch.long)
    m = torch.empty_like(d)
    for j in range(4):
        d[..., j] = q[..., j] & 63
        m[..., j] = q[..., j + 4] & 63
    for j in range(4, 8):
        d[..., j] = (q[..., j + 4] & 0x0F) | ((q[..., j - 4] >> 6) << 4)
        m[..., j] = (q[..., j + 4] >> 4) | ((q[..., j] >> 6) << 4)
    return d.float(), m.float()


def dequantize_q4_k(raw, n_values):
    """ds4_dense_block_q4_K (metal/dense.metal:2017), 144 B per 256 values."""
    if n_values % 256:
        fail(f"Q4_K row of {n_values} values is not a multiple of 256")
    nb = n_values // 256
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, nb, 144)
    rows = buf.shape[0]
    d = buf[:, :, 0:2].contiguous().view(torch.float16).float().view(rows, nb, 1)
    dmin = buf[:, :, 2:4].contiguous().view(torch.float16).float().view(rows, nb, 1)
    sc, mn = _q4_k_scale_min(buf[:, :, 4:16])
    qs = buf[:, :, 16:144].long().view(rows, nb, 4, 32)
    out = torch.empty(rows, nb, 8, 32, dtype=torch.float32)
    for half in range(4):
        lo = (qs[:, :, half, :] & 0x0F).float()
        hi = (qs[:, :, half, :] >> 4).float()
        i0, i1 = 2 * half, 2 * half + 1
        out[:, :, i0, :] = d * sc[:, :, i0:i0 + 1] * lo - dmin * mn[:, :, i0:i0 + 1]
        out[:, :, i1, :] = d * sc[:, :, i1:i1 + 1] * hi - dmin * mn[:, :, i1:i1 + 1]
    return out.view(rows, n_values)


def dequantize_q8_0(raw, n_values):
    if n_values % 32:
        fail(f"Q8_0 row of {n_values} values is not a multiple of 32")
    nb = n_values // 32
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, nb, 34)
    scales = buf[:, :, :2].contiguous().view(torch.float16).float()
    values = buf[:, :, 2:].contiguous().view(torch.int8).float()
    return (values * scales).view(-1, n_values)


_GGUF_ROW_BYTES = {GGUF_F32: lambda n: n * 4, GGUF_F16: lambda n: n * 2,
                   GGUF_Q8_0: lambda n: n // 32 * 34, GGUF_Q4_K: lambda n: n // 256 * 144}


class GgufWeights:
    """The drafter's weights as the runtime binds them, under HF names.

    Name mapping is the inverse of deepseek41_dspark_quantize.build_plan, so the
    two cannot drift.  Rows are [out, in] in both GGUF storage order and
    PyTorch's Linear convention, so nothing is transposed.
    """

    def __init__(self, path, stages=3, experts=128):
        self.path = path
        self.directory, self.data_start = gguf_tensor_directory(path)
        self.fp = open(path, "rb")
        self.cache = {}
        self.stages = stages
        self.experts = experts

    def close(self):
        self.fp.close()

    @staticmethod
    def gguf_name(hf):
        stage, rest = hf.split(".", 2)[1], hf.split(".", 2)[2]
        table = {
            "attn.attn_sink": "attn_sinks.weight",
            "attn.wq_a.weight": "attn_q_a.weight",
            "attn.wq_b.weight": "attn_q_b.weight",
            "attn.q_norm.weight": "attn_q_a_norm.weight",
            "attn.wkv.weight": "attn_kv.weight",
            "attn.kv_norm.weight": "attn_kv_a_norm.weight",
            "attn.wo_a.weight": "attn_output_a.weight",
            "attn.wo_b.weight": "attn_output_b.weight",
            "attn_norm.weight": "attn_norm.weight",
            "ffn_norm.weight": "ffn_norm.weight",
            "ffn.gate.weight": "ffn_gate_inp.weight",
            "ffn.gate.bias": "exp_probs_b.bias",
            "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
            "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
            "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
            "main_proj.weight": "main_proj.weight",
            "main_norm.weight": "main_norm.weight",
            "norm.weight": "norm.weight",
            "markov_head.embed.weight": "markov_head.markov_w1.weight",
            "markov_head.head.weight": "markov_head.markov_w2.weight",
            "confidence_head.proj.weight": "confidence_head.proj.weight",
        }
        for part in ("attn", "ffn"):
            for piece in ("fn", "base", "scale"):
                table[f"hc_{part}_{piece}"] = f"hc_{part}_{piece}.weight"
        if rest not in table:
            fail(f"no GGUF name for {hf}")
        return f"mtp.{stage}.{table[rest]}"

    def _read(self, name, expert=None):
        if name not in self.directory:
            fail(f"{self.path}: missing tensor {name}")
        dims, qtype, offset = self.directory[name]
        n_in = dims[0]
        rows = dims[1] if len(dims) > 1 else 1
        if qtype not in _GGUF_ROW_BYTES:
            fail(f"{self.path}: {name} has unsupported quant type {qtype}")
        row_bytes = _GGUF_ROW_BYTES[qtype](n_in)
        base = self.data_start + offset
        if expert is not None:
            base += expert * rows * row_bytes
        self.fp.seek(base)
        raw = self.fp.read(rows * row_bytes)
        if qtype == GGUF_F32:
            out = torch.frombuffer(bytearray(raw), dtype=torch.float32).view(rows, n_in)
        elif qtype == GGUF_F16:
            out = torch.frombuffer(bytearray(raw), dtype=torch.float16).float().view(rows, n_in)
        elif qtype == GGUF_Q8_0:
            out = dequantize_q8_0(raw, n_in)
        else:
            out = dequantize_q4_k(raw, n_in)
        return out.squeeze(0) if len(dims) == 1 else out

    def get(self, hf_name):
        key = self.gguf_name(hf_name)
        if key not in self.cache:
            self.cache[key] = self._read(key)
        return self.cache[key]

    def expert(self, stage, index, part):
        name = f"mtp.{stage}.ffn_{ {'w1': 'gate', 'w3': 'up', 'w2': 'down'}[part] }_exps.weight"
        return self._read(name, expert=index)


# ---------------------------------------------------------------------------
# Kernels reimplemented from inference/kernel.py
# ---------------------------------------------------------------------------


def rms_norm(x, weight, eps=NORM_EPS):
    """RMSNorm, model.py:281-294."""
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return weight * y


def freqs_cis(count, dim=ROPE_HEAD_DIM, base=ROPE_THETA):
    """precompute_freqs_cis with YaRN disabled (compress_ratio == 0, model.py:690-696)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return torch.polar(torch.ones(count, dim // 2), torch.outer(torch.arange(count, dtype=torch.float32), freqs))


def apply_rotary(x, cis, inverse=False):
    """apply_rotary_emb, model.py:392-407. Adjacent element pairs are one complex number."""
    shape = x.shape
    z = torch.view_as_complex(x.float().reshape(*shape[:-1], -1, 2).contiguous())
    if inverse:
        cis = cis.conj()
    if z.ndim == 3:  # [b, s, d/2]
        cis = cis.view(1, z.size(1), z.size(-1))
    else:  # [b, s, h, d/2]
        cis = cis.view(1, z.size(1), 1, z.size(-1))
    return torch.view_as_real(z * cis).flatten(-2).reshape(shape)


def rope_tail_(x, cis, inverse=False):
    """In-place rotation of the last `rope_head_dim` elements, as the reference does."""
    x[..., -ROPE_HEAD_DIM:] = apply_rotary(x[..., -ROPE_HEAD_DIM:], cis, inverse)
    return x


def act_quant_fp8(x, block=32):
    # kernel_dsv41_quantize rounds the value it reads and the value it stores.
    x = rnd(x)
    """act_quant(..., "ue8m0", float8_e8m0fnu, inplace=True): fused quant+dequant.

    kernel.py:41-95 -- per `block` elements along the last axis, amax floored at 1e-4,
    scale = 2**ceil(log2(amax/448)), value = fp8_e4m3(clamp(x/s, +-448)) * s.
    """
    shape = x.shape
    if shape[-1] % block:
        fail("act_quant block size does not divide the row")
    groups = x.reshape(-1, shape[-1] // block, block)
    amax = groups.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    quantized = torch.clamp(groups / scale, -FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(shape)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """sparse_attn, kernel.py:311-403, in fp32.

    q: [b, m, h, d]; kv: [b, n, d]; topk_idxs: [b, m, topk], -1 marks an empty slot.
    No causal mask: the index list *is* the mask. attn_sink enters the softmax
    denominator only, and contributes no value vector.
    """
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    gather = topk_idxs.clamp(min=0).long()
    keys = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2,
                        gather.unsqueeze(-1).expand(b, m, gather.size(-1), d))
    keys = keys * valid.unsqueeze(-1)
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), keys.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # The kernel seeds its running maximum at -1e30 and never clears it, so a row
    # with no valid index yields an all-zero output instead of a NaN.
    peak = torch.maximum(scores.amax(-1), torch.full_like(scores[..., 0], -1e30))
    weights = torch.exp(scores - peak.unsqueeze(-1))
    weights = torch.where(valid.unsqueeze(2), weights, torch.zeros_like(weights))
    denom = weights.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - peak)
    return torch.einsum("bmhk,bmkd->bmhd", weights, keys.float()) / denom.unsqueeze(-1)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC, iters=SINKHORN_ITERS, eps=HC_EPS):
    """hc_split_sinkhorn, kernel.py:406-460.

    One softmax over comb's rows, then a *column* normalisation, then `iters - 1`
    row/column pairs. The eps sits inside every normalisation denominator.
    """
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Hyper-connection helpers, Block.hc_* (model.py:948-966)
# ---------------------------------------------------------------------------


def hc_mixes(x, hc_fn, hc_scale, hc_base):
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + NORM_EPS)
    return hc_split_sinkhorn(torch.nn.functional.linear(flat, hc_fn) * rsqrt, hc_scale, hc_base)


def hc_pre(x, pre_mix):
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)


def hc_post(x, residual, post, comb):
    return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class Weights:
    """Dequantizes `mtp.*` tensors on demand and caches the dense ones."""

    def __init__(self, db, quantizer):
        self.db = db
        self.q = quantizer
        self.cache = {}
        self.expert_cache = {}
        self.expert_order = []
        self.expert_cache_limit = 24

    def get(self, name):
        tensor = self.cache.get(name)
        if tensor is None:
            tensor = torch.from_numpy(self.q.to_f32(self.db, name).copy())
            self.cache[name] = tensor
        return tensor

    def expert(self, stage, index, part):
        key = (stage, index, part)
        tensor = self.expert_cache.get(key)
        if tensor is None:
            name = f"mtp.{stage}.ffn.experts.{index}.{part}.weight"
            tensor = torch.from_numpy(self.q.to_f32(self.db, name).copy())
            self.expert_cache[key] = tensor
            self.expert_order.append(key)
            while len(self.expert_order) > self.expert_cache_limit:
                del self.expert_cache[self.expert_order.pop(0)]
        return tensor


# ---------------------------------------------------------------------------
# The drafter
# ---------------------------------------------------------------------------


def swiglu_expert(x, w1, w3, w2):
    """Expert.forward, model.py:841-852. Up clamped on both sides, gate from above."""
    gate = torch.nn.functional.linear(x, w1).float()
    up = torch.nn.functional.linear(x, w3).float()
    up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w2)


class Stage:
    """One `mtp.{s}` DSpark stage: an ordinary V4.1 block with a private 128-slot ring."""

    def __init__(self, weights, index, trace=None):
        self.w = weights
        self.s = index
        self.prefix = f"mtp.{index}."
        self.ring = torch.zeros(1, WINDOW, HEAD_DIM)
        self.trace = trace

    def t(self, suffix):
        return self.w.get(self.prefix + suffix)

    # -- attention -----------------------------------------------------------

    def seed_prefill(self, main_x, cis):
        """DSparkAttention.forward with start_pos == 0: seed the ring, return x."""
        seqlen = main_x.size(1)
        main_kv = rnd(rms_norm(torch.nn.functional.linear(main_x, self.t("attn.wkv.weight")),
                           self.t("attn.kv_norm.weight")))
        main_kv = rnd(rope_tail_(main_kv, cis[:seqlen]))
        main_kv = act_quant_fp8(main_kv)
        if seqlen <= WINDOW:
            self.ring[:, :seqlen] = main_kv
        else:
            cutoff = seqlen % WINDOW
            tail = main_kv[:, -WINDOW:]
            self.ring[:, cutoff:WINDOW] = tail[:, :WINDOW - cutoff]
            self.ring[:, :cutoff] = tail[:, WINDOW - cutoff:]

    def attention(self, x, start_pos, main_x, cis):
        """DSparkAttention.forward at decode, model.py:1032-1075."""
        main_cis = cis[start_pos:start_pos + 1]
        main_kv = rnd(rms_norm(torch.nn.functional.linear(main_x, self.t("attn.wkv.weight")),
                           self.t("attn.kv_norm.weight")))
        main_kv = act_quant_fp8(rnd(rope_tail_(main_kv, main_cis)))

        block = x.size(1)
        draft_cis = cis[start_pos + 1:start_pos + 1 + block]

        qr = rnd(rms_norm(rnd(torch.nn.functional.linear(x, self.t("attn.wq_a.weight"))),
                      self.t("attn.q_norm.weight")))
        q = rnd(torch.nn.functional.linear(qr, self.t("attn.wq_b.weight"))).unflatten(-1, (N_HEADS, HEAD_DIM))
        q = rnd(rope_tail_(q, draft_cis))
        kv = rnd(rms_norm(torch.nn.functional.linear(x, self.t("attn.wkv.weight")),
                      self.t("attn.kv_norm.weight")))
        kv = act_quant_fp8(rnd(rope_tail_(kv, draft_cis)))

        # get_dspark_topk_idxs: one index row, expanded across every draft slot.
        # The block is non-causal -- each slot sees all five, later ones included.
        idx = torch.cat([torch.arange(min(WINDOW, start_pos + 1)), WINDOW + torch.arange(block)])
        topk_idxs = idx.int().view(1, 1, -1).expand(1, block, -1).contiguous()

        self.ring[:, start_pos % WINDOW] = main_kv.squeeze(1)
        if self.trace is not None:
            # The ring exactly as the attention reads it: FP8-quantized, RoPEd
            # main_kv rows at slot `position % 128`. Stage 3's Metal ring must
            # hold these values at the same absolute slots.
            self.trace[f"ring{self.s}"] = self.ring.clone()
        keys = torch.cat([self.ring, kv], dim=1)
        o = rnd(sparse_attn(q, keys, self.t("attn.attn_sink"), topk_idxs, HEAD_DIM ** -0.5))
        o = rnd(rope_tail_(o, draft_cis, inverse=True))

        o = o.view(1, block, O_GROUPS, -1)
        wo_a = self.t("attn.wo_a.weight").view(O_GROUPS, O_LORA_RANK, -1)
        # ds4_gpu_dsv41_attention_output_batch rounds between the two projections.
        o = rnd(torch.einsum("bsgd,grd->bsgr", o, wo_a))
        return rnd(torch.nn.functional.linear(o.flatten(2), self.t("attn.wo_b.weight")))

    # -- MoE -----------------------------------------------------------------

    def moe(self, x):
        """Gate + MoE, model.py:792-905, with the drafter's 128 routed / 3 active."""
        flat = x.reshape(-1, DIM)
        scores = torch.nn.functional.linear(flat, self.t("ffn.gate.weight")) / GATE_TEMP
        scores = torch.nn.functional.softplus(scores).sqrt()
        indices = (scores + self.t("ffn.gate.bias")).topk(TOPK, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)  # not norm_eps
        weights = weights * ROUTE_SCALE

        y = torch.zeros_like(flat)
        for row in range(flat.size(0)):
            for slot in range(TOPK):
                expert = int(indices[row, slot])
                y[row] += weights[row, slot] * swiglu_expert(
                    flat[row], self.w.expert(self.s, expert, "w1"),
                    self.w.expert(self.s, expert, "w3"), self.w.expert(self.s, expert, "w2"))
        shared = swiglu_expert(flat, self.t("ffn.shared_experts.w1.weight"),
                              self.t("ffn.shared_experts.w3.weight"),
                              self.t("ffn.shared_experts.w2.weight"))
        return rnd(y + rnd(shared)).view(x.shape), indices, weights

    # -- block ---------------------------------------------------------------

    def forward(self, x, start_pos, pre_mix, main_x, cis):
        """Block.forward, model.py:968-995. The mix a sub-layer computes feeds the next."""
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, self.t("hc_attn_fn"), self.t("hc_attn_scale"),
                                                  self.t("hc_attn_base"))
        h = rnd(rms_norm(rnd(hc_pre(x, pre_mix)), self.t("attn_norm.weight")))
        h = self.attention(h, start_pos, main_x, cis)
        if self.trace is not None:
            self.trace[f"stage{self.s}_attn_block"] = h.clone()
        x = hc_post(h, residual, attn_post, attn_comb)
        if self.trace is not None:
            self.trace[f"stage{self.s}_post_attn"] = x.clone()
            self.trace[f"stage{self.s}_attn_pre"] = attn_pre.clone()

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, self.t("hc_ffn_fn"), self.t("hc_ffn_scale"),
                                               self.t("hc_ffn_base"))
        h = rnd(rms_norm(rnd(hc_pre(x, attn_pre)), self.t("ffn_norm.weight")))
        h, indices, weights = self.moe(h)
        if self.trace is not None:
            self.trace[f"stage{self.s}_ffn_block"] = h.clone()
        x = hc_post(h, residual, ffn_post, ffn_comb)
        if self.trace is not None:
            self.trace[f"stage{self.s}_post_ffn"] = x.clone()
            self.trace[f"stage{self.s}_ffn_pre"] = ffn_pre.clone()
            self.trace[f"stage{self.s}_expert_ids"] = indices.clone()
            self.trace[f"stage{self.s}_expert_weights"] = weights.clone()
        return x, ffn_pre


class Drafter:
    def __init__(self, weights, token_embd, output_head, stages=3, trace=None):
        self.w = weights
        self.token_embd = token_embd
        self.output_head = output_head
        self.stages = [Stage(weights, s, trace) for s in range(stages)]
        self.trace = trace
        self.cis = None

    def ensure_cis(self, count):
        if self.cis is None or self.cis.size(0) < count:
            self.cis = freqs_cis(count + 16)
        return self.cis

    def main_x(self, main_hidden):
        """DSparkBlock.forward_embed's first half: main_norm(main_proj(main_hidden))."""
        projected = torch.nn.functional.linear(main_hidden, self.w.get("mtp.0.main_proj.weight"))
        return rnd(rms_norm(projected, self.w.get("mtp.0.main_norm.weight")))

    def trace_hidden(self, main_hidden):
        """The stimulus itself, so a port can run main_proj rather than be handed
        its output.  Concatenation order is target-layer order (37, 38, 39)."""
        if self.trace is not None:
            self.trace["main_hidden"] = main_hidden.clone()

    def prefill(self, main_hidden):
        cis = self.ensure_cis(main_hidden.size(1))
        main_x = self.main_x(main_hidden)
        if self.trace is not None:
            # Positions [0, start_pos): everything the three rings are seeded
            # from. Stage 3 drives its own seeding from this rather than from a
            # 550B backbone it must not load.
            self.trace["main_x_prefill"] = main_x.clone()
        for stage in self.stages:
            stage.seed_prefill(main_x, cis)
        return main_x

    def decode(self, main_hidden, seed_token, start_pos, temperature=0.0):
        if start_pos <= 0:
            fail("get_dspark_topk_idxs asserts start_pos > 0")
        cis = self.ensure_cis(start_pos + BLOCK_SIZE + 1)
        main_x = self.main_x(main_hidden)
        if self.trace is not None:
            self.trace["main_x"] = main_x.clone()

        draft_ids = torch.full((1, BLOCK_SIZE), NOISE_TOKEN_ID, dtype=torch.long)
        draft_ids[0, 0] = seed_token
        x = self.token_embd[draft_ids.view(-1)].float().view(1, BLOCK_SIZE, DIM)
        x = x.unsqueeze(2).repeat(1, 1, HC, 1)
        if self.trace is not None:
            self.trace["draft_ids"] = draft_ids.clone()
            self.trace["stage_input"] = x.clone()

        pre_mix = torch.zeros(1, BLOCK_SIZE, HC)
        pre_mix[:, :, 0] = 1.0  # make_identity_pre_mix
        for stage in self.stages:
            x, pre_mix = stage.forward(x, start_pos, pre_mix, main_x, cis)

        return self.head(x, pre_mix, seed_token, temperature)

    def head(self, x, pre_mix, seed_token, temperature):
        """DSparkBlock.forward_head, model.py:1137-1156."""
        final = self.stages[-1]
        collapsed = rnd(hc_pre(x, pre_mix))
        normed = rnd(rms_norm(collapsed, final.t("norm.weight")))
        base_logits = torch.nn.functional.linear(normed.float(), self.output_head)

        markov_embed_table = final.t("markov_head.embed.weight")
        markov_head_weight = final.t("markov_head.head.weight")

        output_ids = torch.zeros(1, BLOCK_SIZE + 1, dtype=torch.long)
        output_ids[0, 0] = seed_token
        logits = base_logits.clone()
        bias = torch.zeros_like(base_logits)
        embeds = []
        for i in range(BLOCK_SIZE):
            embed = markov_embed_table[output_ids[:, i]]
            bias[:, i] = torch.nn.functional.linear(embed.float(), markov_head_weight)
            logits[:, i] += bias[:, i]
            embeds.append(embed)
            output_ids[0, i + 1] = sample(logits[0, i], temperature)
        markov_embed = torch.stack(embeds, dim=1)
        confidence = torch.nn.functional.linear(
            torch.cat([collapsed, markov_embed], dim=-1).float(),
            final.t("confidence_head.proj.weight").float()).squeeze(-1)

        if self.trace is not None:
            self.trace["head_input"] = collapsed.clone()
            self.trace["head_input_normed"] = normed.clone()
            self.trace["head_pre_mix"] = pre_mix.clone()
            self.trace["base_logits"] = base_logits.clone()
            self.trace["markov_bias"] = bias.clone()
            self.trace["logits"] = logits.clone()
            self.trace["markov_embed"] = markov_embed.clone()
            self.trace["output_ids"] = output_ids.clone()
            self.trace["confidence"] = confidence.clone()
        return output_ids, logits, confidence


def sample(logits, temperature):
    if temperature <= 0:
        return int(logits.argmax(dim=-1))
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
    return int(probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def synthetic_main_hidden(seed, start_pos, target_layers=3):
    """A deterministic stand-in for `cat(mean_c residual[c] at layers 37/38/39)`.

    The drafter's forward is defined for any [b, s, 3*5120] input and `main_norm`
    removes the overall scale, so a unit-variance normal is a faithful stimulus.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, start_pos + 1, target_layers * DIM, generator=generator)


def seed_token_for(seed, start_pos):
    generator = torch.Generator().manual_seed(seed * 1000003 + start_pos)
    token = int(torch.randint(0, VOCAB, (1,), generator=generator))
    return token + 1 if token == NOISE_TOKEN_ID else token


def write_dump(directory, trace, meta):
    os.makedirs(directory, exist_ok=True)
    entries = []
    for name in sorted(trace):
        value = trace[name]
        if value.dtype in (torch.int64, torch.int32):
            payload = value.to(torch.int32).contiguous().numpy().tobytes()
            dtype = "i4"
        else:
            payload = value.to(torch.float32).contiguous().numpy().tobytes()
            dtype = "f4"
        filename = f"{name}.bin"
        with open(os.path.join(directory, filename), "wb") as fp:
            fp.write(payload)
        entries.append({
            "name": name,
            "file": filename,
            "dtype": dtype,
            "shape": list(value.shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = dict(meta, tensors=entries)
    with open(os.path.join(directory, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf", required=True, help="HF checkpoint directory (shards 44-46)")
    parser.add_argument("--gguf", required=True, help="main V4.1 GGUF, for token_embd and output")
    parser.add_argument("--out", required=True, help="dump directory")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy, which is what ds4 drafts with")
    parser.add_argument("--positions", default=",".join(str(p) for p in DEFAULT_POSITIONS))
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--support-gguf",
                        help="read the drafter's weights from this Stage-1 support GGUF "
                             "instead of the safetensors, i.e. exactly what the runtime binds")
    parser.add_argument("--bf16", action="store_true",
                        help="apply V4.1's BF16 activation boundaries at the points "
                             "ds41_dspark_bf16 applies them")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library",
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             f"libds4quants.{suffix}"))
    args = parser.parse_args()

    positions = [int(p) for p in args.positions.split(",") if p.strip()]
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    started = time.monotonic()

    global BF16
    BF16 = bool(args.bf16)
    db = None
    if args.support_gguf:
        weights = GgufWeights(args.support_gguf, args.stages)
        print(f"weights: {args.support_gguf} (quantized, as the runtime binds them), "
              f"bf16_boundaries={BF16}", flush=True)
    else:
        db = MtpSourceDB(args.hf)
        quantizer = NativeQuantizer(args.quants_library)
        weights = Weights(db, quantizer)

    if db is not None:
        print(f"source: {len(db.tensors)} mtp.* tensors in {sorted(set(db.weight_map.values()))}", flush=True)
    # token_embd is only ever row-indexed, so it stays in its stored F16.
    token_embd = gguf_read_matrix(args.gguf, "token_embd.weight", VOCAB, DIM, torch.float16)
    output_head = gguf_read_matrix(args.gguf, "output.weight", VOCAB, DIM)
    print(f"tied heads from {os.path.basename(args.gguf)}: "
          f"token_embd {tuple(token_embd.shape)}, output {tuple(output_head.shape)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    index = []
    for start_pos in positions:
        trace = {}
        drafter = Drafter(weights, token_embd, output_head, args.stages, trace)
        main_hidden = synthetic_main_hidden(args.seed, start_pos)
        token = seed_token_for(args.seed, start_pos)
        step = time.monotonic()
        drafter.trace_hidden(main_hidden)
        drafter.prefill(main_hidden[:, :start_pos])
        output_ids, _, confidence = drafter.decode(main_hidden[:, start_pos:start_pos + 1],
                                                   token, start_pos, args.temperature)
        elapsed = time.monotonic() - step
        meta = {
            "start_pos": start_pos,
            "seed": args.seed,
            "temperature": args.temperature,
            "seed_token": token,
            "prefill_positions": start_pos,
            "block_size": BLOCK_SIZE,
            "window": WINDOW,
            "stages": args.stages,
            "noise_token_id": NOISE_TOKEN_ID,
            "main_hidden": "torch.randn([1, start_pos+1, 15360], "
                           f"Generator().manual_seed({args.seed})); "
                           "row start_pos is the decode step, rows [0, start_pos) are the prefill",
            "seconds": round(elapsed, 3),
        }
        manifest = write_dump(os.path.join(args.out, f"pos{start_pos:04d}"), trace, meta)
        index.append({"start_pos": start_pos, "dir": f"pos{start_pos:04d}",
                      "seed_token": token,
                      "output_ids": [int(v) for v in output_ids[0]],
                      "confidence": [float(v) for v in confidence[0]],
                      "tensors": len(manifest["tensors"]), "seconds": round(elapsed, 3)})
        print(f"start_pos={start_pos}: seed_token={token} "
              f"draft={[int(v) for v in output_ids[0][1:]]} "
              f"confidence={[round(float(v), 4) for v in confidence[0]]} "
              f"({elapsed:.1f}s)", flush=True)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    total = time.monotonic() - started
    summary = {
        "seed": args.seed,
        "temperature": args.temperature,
        "positions": positions,
        "runs": index,
        "runtime_seconds": round(total, 1),
        "peak_rss_bytes": peak_bytes,
        "peak_rss_gib": round(peak_bytes / (1 << 30), 2),
        "torch": torch.__version__,
    }
    with open(os.path.join(args.out, "index.json"), "w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"done: {total:.1f}s, peak RSS {peak_bytes / (1 << 30):.2f} GiB", flush=True)
    if db is not None:
        db.close()
    elif hasattr(weights, "close"):
        weights.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"dspark41-reference: {error}")

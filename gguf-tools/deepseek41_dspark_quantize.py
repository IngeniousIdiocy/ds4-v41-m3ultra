#!/usr/bin/env python3
"""Convert the DeepSeek V4.1 Flash DSpark drafter (`mtp.*`) to a ds4 support GGUF.

The drafter is three V4.1 blocks with a private 128-slot sliding-window KV ring,
a rank-256 factorised bigram (Markov) head and a confidence head. It has no token
embedding and no output head of its own: `Transformer.__init__` ties them to the
backbone (`inference/model.py:1212-1213`) and `convert.py:108-109` skips them, so
this file must not carry `token_embd` / `output` and the runtime reads them from
the main model.

Two things differ from the V4 DSpark checkpoint ds4 already consumes: there are no
`hc_head_*` tensors (the final collapse uses the `ffn_pre` carried out of stage 2),
and the drafter routes 3 of 128 experts while the V4.1 backbone routes 6 of 384 --
hence the explicit `deepseek41.dspark.n_routed_experts` / `.num_experts_per_tok`
metadata, which the loader must use instead of the backbone's shape.

Tensor names mirror V4's DSpark support file so ds4's binder is shared.
"""

import argparse
import dataclasses
import json
import os
import re
import sys
import threading

from deepseek41_metadata import GGUF_ALIGNMENT, array_record
from deepseek41_quantize import scale_name, write_gguf
from glm53_quantize import (
    GGUF_UINT32, SourceDB, TensorPlan, QTYPE_F32, QTYPE_F16, QTYPE_Q8_0,
    QTYPE_Q2_K, QTYPE_Q4_K, QTYPE_IQ2_XXS, align, kv_string, kv_u32,
    load_safetensors_header, print_plan, qtype_nbytes,
)

SOURCE_REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"

QUANTIZATION = {
    "q4": "Q4_K routed experts; Q8_0 attention/shared/main_proj/markov",
    "q2": "IQ2_XXS gate/up; Q2_K down; Q8_0 attention/shared/main_proj/markov",
    "q8": "Q8_0 routed experts; Q8_0 attention/shared/main_proj/markov",
    "f16": "Q8_0 routed experts; F16 q/kv/shared/main_proj; Q8_0 attn output/markov",
}

# F16 routed experts are not reachable: ds4's binder accepts only
# Q8_0/IQ2_XXS/Q2_K/Q4_K/Q5_K/Q6_K/MXFP4 for a routed expert tensor
# (tensor_is_routed_expert_type, ds4.c), and every routed MoE kernel dispatches
# on that set.  Q8_0 is therefore the drafter's precision ceiling for the 92 %
# of the file that is routed experts, and the "f16" recipe raises only the
# dense half.  The Markov head stays Q8_0 in every recipe because the GPU
# runtime refuses anything else (ds4.c:35179-35182).

ARCHITECTURE = "deepseek41-dspark"


def fail(message):
    raise SystemExit(f"deepseek41-dspark-quantize: {message}")


class DSparkSourceDB(SourceDB):
    """`SourceDB` over just the shards that hold `mtp.*`.

    The DSpark material is shards 44-46 of 48. The stock constructor insists on
    every shard the index names, so it cannot open a partial download.
    """

    PREFIX = "mtp."

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        with open(index_path, "rb") as fp:
            document = json.load(fp)
        full_map = document.get("weight_map")
        if not isinstance(full_map, dict) or not full_map:
            fail(f"{index_path}: missing or empty weight_map")
        self.declared_bytes = document.get("metadata", {}).get("total_size")
        self.weight_map = {n: s for n, s in full_map.items() if n.startswith(self.PREFIX)}
        if not self.weight_map:
            fail(f"{index_path}: no {self.PREFIX}* tensors")
        self.tensors = {}
        self._fds = {}
        self._fd_lock = threading.Lock()
        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if not os.path.isfile(path):
                fail(f"missing source shard {path}")
            for name, info in load_safetensors_header(path).items():
                if not name.startswith(self.PREFIX):
                    continue
                if self.weight_map.get(name) != shard:
                    fail(f"index assigns {name} to {self.weight_map.get(name)!r}, not {shard}")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                self.tensors[name] = dict(info, shard=shard)
        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source headers are incomplete; first missing tensor is {missing[0]}")
        validate_scales(self.tensors)


def validate_scales(tensors):
    """`deepseek41_quantize.validate_scales`, re-homed onto the `mtp.*` namespace.

    FP8-E4M3 weights carry one E8M0 scale per 32x32 block; packed-FP4 weights are
    stored two nibbles per byte with one E8M0 scale per 32 logical columns.
    """
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


def dspark_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), "rb") as fp:
        config = json.load(fp)
    if config["model_type"] != "deepseek_v41":
        fail("not a DeepSeek V4.1 source checkpoint")
    text = config["text_config"]
    if config["quantization_config"]["weight_block_size"] != [32, 32]:
        fail("expected native 32x32 FP8 blocks")
    if config["quantization_config"]["expert_dtype"] != "fp4":
        fail("expected FP4 routed experts")
    return config, text


def build_plan(db, text, quant="q4"):
    """The tensor plan, with the source inventory it consumes.

    Returns (plan, consumed). GGUF dimensions are the reverse of the stored torch
    `[out, in]`, matching every other converter in this tree.
    """
    if quant not in QUANTIZATION:
        fail(f"unknown quantization recipe: {quant}")
    dim = text["hidden_size"]
    inter = text["moe_intermediate_size"]
    heads, hd = text["num_attention_heads"], text["head_dim"]
    qrank, orank, groups = text["q_lora_rank"], text["o_lora_rank"], text["o_groups"]
    hc, vocab = text["hc_mult"], text["vocab_size"]
    stages = text["num_nextn_predict_layers"]
    experts = text["dspark_n_routed_experts"]
    rank = text["dspark_markov_rank"]
    targets = text["dspark_target_layer_ids"]

    plan, consumed = [], set()

    def claim(name, expected, dtype=None):
        info = db.info(name)
        if info["shape"] != list(expected) or (dtype and info["dtype"] != dtype):
            fail(f"{name}: unexpected {info['dtype']} {info['shape']}, expected {dtype} {expected}")
        consumed.add(name)
        if info["dtype"] in ("I8", "F8_E4M3"):
            consumed.add(scale_name(name))
        return info

    def regular(name, source, shape, qtype, role):
        claim(source, shape)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source))

    for stage in range(stages):
        src, dst = f"mtp.{stage}", f"mtp.{stage}"
        for site in ("attn", "ffn"):
            for part, shape, qt in (("fn", (hc * (hc + 2), hc * dim), QTYPE_F16),
                                    ("base", (hc * (hc + 2),), QTYPE_F32),
                                    ("scale", (3,), QTYPE_F32)):
                regular(f"{dst}.hc_{site}_{part}.weight", f"{src}.hc_{site}_{part}", shape, qt, "mhc")
            regular(f"{dst}.{site}_norm.weight", f"{src}.{site}_norm.weight", (dim,), QTYPE_F32, "norm")
        dense = QTYPE_F16 if quant == "f16" else QTYPE_Q8_0
        for target, source, shape, qt, role in (
            ("attn_sinks.weight", "attn_sink", (heads,), QTYPE_F32, "norm"),
            ("attn_q_a.weight", "wq_a.weight", (qrank, dim), dense, "attention"),
            ("attn_q_b.weight", "wq_b.weight", (heads * hd, qrank), dense, "attention"),
            ("attn_q_a_norm.weight", "q_norm.weight", (qrank,), QTYPE_F32, "norm"),
            ("attn_kv.weight", "wkv.weight", (hd, dim), dense, "attention"),
            ("attn_kv_a_norm.weight", "kv_norm.weight", (hd,), QTYPE_F32, "norm"),
            # The drafter's attention output runs through
            # ds4_gpu_dsv41_attention_output_batch, a fused Q8_0-only kernel
            # (the backbone gates its batched twin on attn_output_b->type ==
            # Q8_0 and falls back; ds41_dspark_stage has no fallback because
            # wo_a is block-diagonal over 8 groups and no plain matmul
            # expresses that).  These two stay Q8_0 in every recipe.
            ("attn_output_a.weight", "wo_a.weight", (groups * orank, heads * hd // groups),
             QTYPE_Q8_0, "attention"),
            ("attn_output_b.weight", "wo_b.weight", (dim, groups * orank), QTYPE_Q8_0, "attention"),
        ):
            regular(f"{dst}.{target}", f"{src}.attn.{source}", shape, qt, role)
        regular(f"{dst}.ffn_gate_inp.weight", f"{src}.ffn.gate.weight", (experts, dim),
                QTYPE_F32, "router")
        # bias_vl is dead on this path: DSparkBlock.forward hard-wires image_mask to
        # None ("drafts are text: no VL bias", model.py:1122-1126), so Gate.forward
        # never substitutes it. Emitted anyway so the source inventory stays fully
        # accounted for; the runtime must not bind it.
        for suffix, target in (("bias", "exp_probs_b.bias"), ("bias_vl", "exp_probs_b_vl.bias")):
            regular(f"{dst}.{target}", f"{src}.ffn.gate.{suffix}", (experts,), QTYPE_F32, "router")
        for part, source, shape, routed in (("gate", "w1", (inter, dim), QTYPE_IQ2_XXS),
                                            ("up", "w3", (inter, dim), QTYPE_IQ2_XXS),
                                            ("down", "w2", (dim, inter), QTYPE_Q2_K)):
            regular(f"{dst}.ffn_{part}_shexp.weight", f"{src}.ffn.shared_experts.{source}.weight",
                    shape, dense, "shared")
            pattern = f"{src}.ffn.experts.{{expert}}.{source}.weight"
            for expert in range(experts):
                claim(pattern.format(expert=expert), (shape[0], shape[1] // 2), "I8")
            plan.append(TensorPlan(f"{dst}.ffn_{part}_exps.weight", (*reversed(shape), experts),
                                   QTYPE_Q4_K if quant == "q4" else
                                   QTYPE_Q8_0 if quant in ("q8", "f16") else routed,
                                   "experts", source=pattern, expert_layer=stage,
                                   expert_part=part, expert_count=experts))
        if stage == 0:
            regular(f"{dst}.main_proj.weight", f"{src}.main_proj.weight",
                    (dim, len(targets) * dim), dense, "attention")
            regular(f"{dst}.main_norm.weight", f"{src}.main_norm.weight", (dim,), QTYPE_F32, "norm")
        if stage == stages - 1:
            regular(f"{dst}.norm.weight", f"{src}.norm.weight", (dim,), QTYPE_F32, "norm")
            # dspark_apply_markov_greedy_gpu_runtime refuses anything but Q8_0
            # (ds4.c:35179-35182) and requires markov_rank % 32 == 0.
            regular(f"{dst}.markov_head.markov_w1.weight", f"{src}.markov_head.embed.weight",
                    (vocab, rank), QTYPE_Q8_0, "markov")
            regular(f"{dst}.markov_head.markov_w2.weight", f"{src}.markov_head.head.weight",
                    (vocab, rank), QTYPE_Q8_0, "markov")
            regular(f"{dst}.confidence_head.proj.weight", f"{src}.confidence_head.proj.weight",
                    (1, dim + rank), QTYPE_F32, "confidence")

    unclaimed = set(db.tensors) - consumed
    if unclaimed:
        fail(f"unclaimed source tensors: {sorted(unclaimed)[:10]}")
    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan, consumed


def metadata(text, quant, imatrix):
    stages = text["num_nextn_predict_layers"]
    return [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "DeepSeek V4.1 Flash DSpark"),
        kv_string("general.source.url", "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"),
        kv_string("general.source.revision", SOURCE_REVISION),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32("deepseek41.dspark.block_size", text["dspark_block_size"]),
        kv_u32("deepseek41.dspark.markov_rank", text["dspark_markov_rank"]),
        kv_u32("deepseek41.dspark.noise_token_id", text["dspark_noise_token_id"]),
        array_record("deepseek41.dspark.target_layer_ids", GGUF_UINT32,
                     list(text["dspark_target_layer_ids"])),
        kv_u32("deepseek41.dspark.stage_count", stages),
        kv_u32("deepseek41.dspark.n_layers", stages),
        # The drafter is 128/3 while the V4.1 backbone is 384/6. Nothing may infer
        # these from the target model.
        kv_u32("deepseek41.dspark.n_routed_experts", text["dspark_n_routed_experts"]),
        kv_u32("deepseek41.dspark.num_experts_per_tok", text["dspark_num_experts_per_tok"]),
        kv_u32("deepseek41.dspark.expert_feed_forward_length", text["moe_intermediate_size"]),
        kv_string("deepseek41.dspark.quantization", QUANTIZATION[quant]),
        kv_string("deepseek41.dspark.calibration",
                  "imatrix" if imatrix else "weight-energy bootstrap"),
    ]


def byte_plan(plan):
    """Actual emitted bytes grouped by component, for the port's byte budget."""
    groups = {}
    for item in plan:
        groups.setdefault(item.role, [0, 0])
        groups[item.role][0] += 1
        groups[item.role][1] += item.nbytes
    padded = plan[-1].offset + align(plan[-1].nbytes, GGUF_ALIGNMENT)
    return groups, padded


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--quant", choices=QUANTIZATION, default="q4")
    parser.add_argument("--imatrix")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if args.source_revision != SOURCE_REVISION:
        parser.error(f"this converter is pinned to {SOURCE_REVISION}")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")

    _, text = dspark_config(args.hf)
    db = DSparkSourceDB(args.hf)
    try:
        plan, consumed = build_plan(db, text, args.quant)
        records = metadata(text, args.quant, args.imatrix)
        if args.imatrix:
            records.append(kv_string("quantize.imatrix.file", os.path.basename(args.imatrix)))
        print(f"source: {len(db.tensors)} mtp.* tensors, {len(consumed)} consumed, "
              f"{len(set(db.tensors) - consumed)} skipped", flush=True)
        if args.dry_run:
            print_plan(plan, records, [], GGUF_ALIGNMENT)
            groups, padded = byte_plan(plan)
            for role in sorted(groups):
                count, nbytes = groups[role]
                print(f"# {role}: {count} tensors, {nbytes / (1 << 30):.3f} GiB")
            print(f"# payload (padded): {padded / (1 << 30):.3f} GiB")
            for item in plan:
                print(json.dumps(dataclasses.asdict(item), sort_keys=True))
        else:
            write_gguf(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-dspark-quantize: {error}")

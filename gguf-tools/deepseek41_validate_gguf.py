#!/usr/bin/env python3
"""Audit a completed V4.1 GGUF and sample its payloads against the source.

Checks the complete tensor layout, then re-encodes selected expert tensors and
samples native Engram rows. This is artifact validation, not inference QA.

`--dspark` audits a DSpark support GGUF instead. There the payload check is
exhaustive rather than sampled -- the file is 7.8 GiB, not 500 -- and two extra
oracles run: the 2401-tensor `mtp.*` source inventory must be fully consumed or
explicitly skipped, and no `mtp.*` name may be unknown to the plan.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import (SourceDB, NativeQuantizer, Imatrix, build_plan,
                                  validate_scales, scale_name, QUANTIZATION)
import deepseek41_dspark_quantize as dspark
from glm53_quantize import (align, read_exact, read_u32, read_u64,
                            read_gguf_string, skip_gguf_value)
from glm53_validate_gguf import read_selected_metadata


def check_payload(fp, offset, item, db, quantizer, imatrix, exhaustive=False):
    if item.role == "engram_disk":
        rows = item.shape[1]
        rng = random.Random(41 + rows)
        selected = {0, 1, rows - 1, 16383, 16384, (1 << 32) // 264}
        selected.update(rng.randrange(rows) for _ in range(128))
        for row in sorted(selected):
            if row >= rows:
                continue
            expected = b"".join(db.iter_read(item.source, row * 256, 256))
            expected += b"".join(db.iter_read(scale_name(item.source), row * 8, 8))
            fp.seek(offset + row * 264)
            if read_exact(fp, 264, item.name) != expected:
                raise ValueError(f"{item.name}: Engram row {row} differs from source")
    elif item.is_expert:
        if exhaustive:
            selected = set(range(item.expert_count))
        else:
            selected = {0, (item.expert_layer * 17 + 41) % item.expert_count, item.expert_count - 1}
        stride = item.nbytes // item.expert_count
        for expert in sorted(selected):
            values = quantizer.to_f32(db, item.source.format(expert=expert))
            importance = imatrix.expert(item.name, expert, item.shape[0], item.expert_count)
            expected = quantizer.encode(values, item.qtype, importance)
            fp.seek(offset + expert * stride)
            if len(expected) != stride or read_exact(fp, stride, item.name) != expected:
                raise ValueError(f"{item.name}: encoded expert {expert} differs from source recipe")
    else:
        values = quantizer.to_f32(db, item.source)
        expected = quantizer.encode(values, item.qtype)
        fp.seek(offset)
        if len(expected) != item.nbytes or read_exact(fp, item.nbytes, item.name) != expected:
            raise ValueError(f"{item.name}: payload differs from source recipe")


def validate(args):
    config = json.loads((Path(args.hf) / "config.json").read_text())
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=validate_scales)
    try:
        plan = build_plan(db, config, args.quant)
        with open(args.gguf, "rb") as fp:
            if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
                raise ValueError("expected GGUF v3")
            if read_u64(fp, "tensor count") != len(plan):
                raise ValueError("tensor count differs from source plan")
            metadata = {}
            for _ in range(read_u64(fp, "metadata count")):
                key = read_gguf_string(fp, "metadata key")
                kind = read_u32(fp, "metadata type")
                if key in {"general.architecture", "general.alignment", "general.source.revision",
                            "deepseek41.calibration", "deepseek41.quantization"}:
                    if key in metadata:
                        raise ValueError(f"duplicate metadata: {key}")
                    metadata[key] = read_selected_metadata(fp, kind)
                else:
                    skip_gguf_value(fp, kind)
            expected = {"general.architecture": "deepseek41", "general.alignment": GGUF_ALIGNMENT,
                        "general.source.revision": args.source_revision,
                        "deepseek41.quantization": QUANTIZATION[args.quant],
                        "deepseek41.calibration": "imatrix" if args.imatrix else "weight-energy bootstrap"}
            if metadata != expected:
                raise ValueError(f"metadata mismatch: {metadata}")
            for item in plan:
                name = read_gguf_string(fp, "tensor name")
                rank = read_u32(fp, "tensor rank")
                if rank != len(item.shape):
                    raise ValueError(f"{name}: unexpected rank {rank}")
                shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
                kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
                if (name, shape, kind, offset) != (item.name, item.shape, item.qtype, item.offset):
                    raise ValueError(f"{item.name}: tensor layout mismatch")
            start = align(fp.tell(), GGUF_ALIGNMENT)
            size = start + plan[-1].offset + align(plan[-1].nbytes, GGUF_ALIGNMENT)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f"expected file size {size}")
            print(f"PASS: {len(plan)} tensor layouts; complete {size}-byte GGUF", flush=True)
            if args.payload:
                q = NativeQuantizer(args.quants_library)
                imatrix = Imatrix(args.imatrix, q.np)
                if args.imatrix and any(t.is_expert and t.name not in imatrix.entries for t in plan):
                    raise ValueError("imatrix is missing expert tensors")
                for index, item in enumerate(plan):
                    check_payload(fp, start + item.offset, item, db, q, imatrix)
                    if (index + 1) % 25 == 0 or index + 1 == len(plan):
                        print(f"PASS: source payload checks {index + 1}/{len(plan)}", flush=True)
    finally:
        db.close()


def read_gguf_header(fp, plan, wanted, alignment):
    """Walk a GGUF header, returning the selected metadata; tensors must match `plan`."""
    if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
        raise ValueError("expected GGUF v3")
    if read_u64(fp, "tensor count") != len(plan):
        raise ValueError("tensor count differs from source plan")
    metadata = {}
    for _ in range(read_u64(fp, "metadata count")):
        key = read_gguf_string(fp, "metadata key")
        kind = read_u32(fp, "metadata type")
        if key in wanted:
            if key in metadata:
                raise ValueError(f"duplicate metadata: {key}")
            metadata[key] = read_selected_metadata(fp, kind)
        else:
            skip_gguf_value(fp, kind)
    for item in plan:
        name = read_gguf_string(fp, "tensor name")
        rank = read_u32(fp, "tensor rank")
        if rank != len(item.shape):
            raise ValueError(f"{name}: unexpected rank {rank}")
        shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
        kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
        if (name, shape, kind, offset) != (item.name, item.shape, item.qtype, item.offset):
            raise ValueError(f"{item.name}: tensor layout mismatch")
    return metadata, align(fp.tell(), alignment)


def validate_dspark(args):
    """The Stage-1 oracle: three checks over a DSpark support GGUF."""
    _, text = dspark.dspark_config(args.hf)
    db = dspark.DSparkSourceDB(args.hf)
    try:
        plan, consumed = dspark.build_plan(db, text, args.quant)

        # (ii) the source inventory is fully consumed or explicitly skipped. Read the
        # mtp.* names straight out of the shard index rather than trusting the DB.
        with open(os.path.join(args.hf, "model.safetensors.index.json"), "rb") as fp:
            index = json.load(fp)["weight_map"]
        inventory = {name for name in index if name.startswith("mtp.")}
        if inventory != set(db.tensors):
            raise ValueError(f"index lists {len(inventory)} mtp.* tensors, headers have "
                             f"{len(db.tensors)}")
        unconsumed = inventory - consumed
        if unconsumed:
            raise ValueError(f"{len(unconsumed)} source tensors neither emitted nor skipped: "
                             f"{sorted(unconsumed)[:10]}")
        print(f"PASS: source inventory fully consumed ({len(inventory)} mtp.* tensors, "
              f"0 skipped)", flush=True)

        # (iii) every consumed name is a source the plan actually reads, or that
        # source's E8M0 scale block -- nothing is claimed by accident.
        sources = set()
        for item in plan:
            if item.is_expert:
                sources.update(item.source.format(expert=e) for e in range(item.expert_count))
            else:
                sources.add(item.source)
        unknown = consumed - sources - {scale_name(n) for n in sources}
        if unknown:
            raise ValueError(f"unknown mtp.* names: {sorted(unknown)[:10]}")
        emitted = {item.name for item in plan}
        tied = {n for n in emitted
                if n.endswith((".embed.weight", ".head.weight")) and ".markov_head." not in n}
        if tied or {"token_embd.weight", "output.weight"} & emitted:
            raise ValueError(f"a tied embedding or output head reached the plan: {sorted(tied)}")
        print(f"PASS: {len(emitted)} emitted tensors from {len(sources)} sources, "
              f"zero unknown mtp.* names, no tied embed/head", flush=True)

        expected_meta = {
            "general.architecture": dspark.ARCHITECTURE,
            "general.alignment": GGUF_ALIGNMENT,
            "general.source.revision": args.source_revision,
            "deepseek41.dspark.quantization": dspark.QUANTIZATION[args.quant],
            "deepseek41.dspark.calibration": "imatrix" if args.imatrix else "weight-energy bootstrap",
            "deepseek41.dspark.block_size": text["dspark_block_size"],
            "deepseek41.dspark.markov_rank": text["dspark_markov_rank"],
            "deepseek41.dspark.noise_token_id": text["dspark_noise_token_id"],
            "deepseek41.dspark.stage_count": text["num_nextn_predict_layers"],
            "deepseek41.dspark.n_layers": text["num_nextn_predict_layers"],
            "deepseek41.dspark.n_routed_experts": text["dspark_n_routed_experts"],
            "deepseek41.dspark.num_experts_per_tok": text["dspark_num_experts_per_tok"],
            "deepseek41.dspark.expert_feed_forward_length": text["moe_intermediate_size"],
        }
        with open(args.gguf, "rb") as fp:
            metadata, start = read_gguf_header(fp, plan, set(expected_meta), GGUF_ALIGNMENT)
            if metadata != expected_meta:
                differing = {k: (metadata.get(k), v) for k, v in expected_meta.items()
                             if metadata.get(k) != v}
                raise ValueError(f"metadata mismatch (got, expected): {differing}")
            size = start + plan[-1].offset + align(plan[-1].nbytes, GGUF_ALIGNMENT)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f"expected file size {size}")
            print(f"PASS: {len(plan)} tensor layouts and {len(expected_meta)} metadata keys; "
                  f"complete {size}-byte GGUF", flush=True)

            # (i) byte-compare every emitted tensor against a fresh quantization
            q = NativeQuantizer(args.quants_library)
            imatrix = Imatrix(args.imatrix, q.np)
            if args.imatrix and any(t.is_expert and t.name not in imatrix.entries for t in plan):
                raise ValueError("imatrix is missing expert tensors")
            for index, item in enumerate(plan):
                check_payload(fp, start + item.offset, item, db, q, imatrix, exhaustive=True)
                if (index + 1) % 10 == 0 or index + 1 == len(plan):
                    print(f"PASS: source payload checks {index + 1}/{len(plan)}", flush=True)
        print("PASS: DSpark support GGUF matches the source recipe byte for byte", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", default=None,
                        help="q2/q4 for the main file; q4/q2 for --dspark")
    parser.add_argument("--imatrix")
    parser.add_argument("--payload", action="store_true")
    parser.add_argument("--dspark", action="store_true",
                        help="audit a DSpark support GGUF instead of the main model")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args()
    choices = dspark.QUANTIZATION if args.dspark else QUANTIZATION
    if args.quant is None:
        args.quant = "q4" if args.dspark else "q2"
    if args.quant not in choices:
        parser.error(f"--quant must be one of {sorted(choices)}")
    try:
        validate_dspark(args) if args.dspark else validate(args)
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-validate: {error}")

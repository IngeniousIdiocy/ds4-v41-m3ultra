#!/usr/bin/env python3
"""Bounded Spec-Bench subset for DeepSeek V4.1 Flash, three DSpark modes.

A frozen, category-balanced subset of the upstream Spec-Bench dataset, run
through the real chat API. It sits between the hand-written fixtures of the
release matrix and a full benchmark run. It is not a full Spec-Bench score.

Scope, fixed and checked rather than configurable: the first 12 question rows,
in stable file order, from each of mt_bench, translation, summarization, qa,
math_reasoning and rag. The 12 MT-Bench questions have two turns each and the
other 60 have one, so the subset is 72 questions, 84 sequential turn requests
per mode, and 252 requests across the three modes. Every turn is greedy,
permits natural EOS, and caps generation at 256 tokens.

Modes:

  serial      DSpark not served from. The non-drafting path and the denominator.
  six         DSpark served with admission control off and the verify width
              pinned to six rows. The configuration that loses on the held-out
              agent fixture in the release matrix.
  controller  the shipped production defaults: DSpark served with the windowed
              cost-feedback admission controller on.

By default each mode gets its own server process, as upstream. With
--use-running-server the driver instead switches the mode of one resident
server through POST /debug/levers and never starts or stops a process: on a
483 GiB target plus drafter, one model load instead of three is the difference
between a 40-minute run and a two-hour one. The lever map read back from the
endpoint after each switch is kept per mode as the receipt that the switch took
effect. That is the one declared deviation from the upstream method.

--prime-prefixes adds an untimed one-token pass over every first turn before
the timed modes. It exists for the resident-server case on a server booted with
--kv-disk-dir: without it the mode that runs first pays a cold prefill on the
prompts long enough to be stored (summarization and rag here) while the other
two restore them, and the binary has no runtime cache bypass. Priming first
makes the cache state identical for all three modes rather than zero for all
three; every request records its own cached_tokens so this is checkable, and
the decode-only boundary is unaffected either way.

Dataset text stays in the caller's checkout. subset-manifest.json binds every
selected row and turn by index, id, byte count and SHA-256 without copying its
text into this repository, and the per-request receipts keep only the SHA-256
of the request body and of the answer, never their text.

  bench/reproduce-spec-bench-v41.py \\
      --spec-bench-root /path/to/Spec-Bench \\
      --server ./ds4-server \\
      --model /path/to/DeepSeek-V4.1-Flash-Q4.gguf \\
      --mtp-model /path/to/DeepSeek-V4.1-Flash-DSpark-Q4.gguf \\
      --output /new/evidence/directory

--dry-run validates identities and freezes the plan without starting a server
or touching the GPU. --preflight-smoke runs 6 requests instead of 252 and is
always labelled smoke, with no performance claim.

Run the full subset only with exclusive GPU access.

The report keeps three timing boundaries apart, as the GLM subset report does:
server end-to-end (the native chat timer, prefill included), decode-only (that
timer minus the prefill timer) and client wall (the HTTP request). Each gets a
pooled throughput, a mean per-question throughput, and matched elapsed and
throughput ratios against serial, which are separate statements because
natural-EOS lengths can differ. Committed tokens per verify come from the real
consumed counts the engine now reports in usage.ds4_dspark on the chat path,
not from an accepted-plus-anchor estimate.

Every declared turn gets one attempt. Nothing is retried and no favourable
attempt is chosen. A failed first turn blocks only its dependent second turn.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

SPEC_BENCH_COMMIT = "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
QUESTION_RELPATH = "data/spec_bench/question.jsonl"
QUESTION_SHA256 = "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"

ROWS_PER_CATEGORY = 12
MAX_GENERATED_TOKENS = 256
# Spec-Bench measures answer generation against a 256-token cap. This engine
# reasons by default, and a reasoning span would consume the whole cap without
# producing an answer, so the subset asks for no reasoning and says so.
REASONING_EFFORT = "none"
EXPECTED_ROWS = 72
EXPECTED_TURNS = 84

LOGICAL_CATEGORIES = ("mt_bench", "translation", "summarization",
                      "qa", "math_reasoning", "rag")
MT_BENCH_CATEGORIES = {"writing", "roleplay", "reasoning", "math",
                       "coding", "extraction", "stem", "humanities"}

MODES = ("serial", "six", "controller")

FINAL_LOG = re.compile(
    r"^(?:[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} )?"
    r"ds4-server: chat .*? gen=(?P<generated>[0-9]+).*? finish=(?P<finish>\S+) "
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)s$", re.MULTILINE)


class Refuse(SystemExit):
    pass


def require(cond, message):
    if not cond:
        raise Refuse(f"refused: {message}")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------ subset selection

def logical_category(row):
    category = row.get("category")
    if category in MT_BENCH_CATEGORIES:
        return "mt_bench"
    return category if category in LOGICAL_CATEGORIES else None


def select_subset(raw_lines):
    """First ROWS_PER_CATEGORY rows per category, in stable file order."""
    selected = {name: [] for name in LOGICAL_CATEGORIES}
    for index, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        category = logical_category(row)
        if category is None or len(selected[category]) >= ROWS_PER_CATEGORY:
            continue
        turns = row.get("turns")
        require(isinstance(turns, list) and turns and
                all(isinstance(t, str) and t for t in turns),
                f"row {index} has no usable turns")
        selected[category].append({
            "global_index": index,
            "question_id": row.get("question_id"),
            "category": row.get("category"),
            "logical_category": category,
            "category_index": len(selected[category]),
            "turns": turns,
        })

    for name, rows in selected.items():
        require(len(rows) == ROWS_PER_CATEGORY,
                f"category {name} has {len(rows)} rows, want {ROWS_PER_CATEGORY}")

    ordered = [row for name in LOGICAL_CATEGORIES for row in selected[name]]
    turn_count = sum(len(row["turns"]) for row in ordered)
    require(len(ordered) == EXPECTED_ROWS and turn_count == EXPECTED_TURNS,
            f"pinned subset shape changed: rows={len(ordered)} turns={turn_count}")

    manifest = {
        "spec_bench_commit": SPEC_BENCH_COMMIT,
        "question_file": QUESTION_RELPATH,
        "question_file_sha256": QUESTION_SHA256,
        "rows_per_category": ROWS_PER_CATEGORY,
        "categories": list(LOGICAL_CATEGORIES),
        "questions": len(ordered),
        "turn_requests_per_mode": turn_count,
        "modes": list(MODES),
        "total_requests": turn_count * len(MODES),
        "max_generated_tokens": MAX_GENERATED_TOKENS,
        "sampling": "greedy, temperature 0, natural EOS",
        "note": ("turns execute sequentially; a second turn includes that mode's own "
                 "first answer, so later prompts can differ across modes and their "
                 "rendered identities are retained per request"),
        "rows": [
            {k: v for k, v in row.items() if k != "turns"} | {
                "turn_count": len(row["turns"]),
                "turns": [
                    {"index": i, "bytes": len(t.encode()), "sha256": sha256_bytes(t.encode())}
                    for i, t in enumerate(row["turns"])
                ],
            }
            for row in ordered
        ],
    }
    return ordered, manifest


def load_subset(spec_root):
    question_file = os.path.join(spec_root, QUESTION_RELPATH)
    require(os.path.isfile(question_file), f"missing {question_file}")
    actual = sha256_file(question_file)
    require(actual == QUESTION_SHA256,
            f"{QUESTION_RELPATH} is {actual[:16]}, want {QUESTION_SHA256[:16]} "
            f"(pin the checkout at {SPEC_BENCH_COMMIT})")
    with open(question_file, "rb") as f:
        return select_subset(f.read().decode("utf-8").splitlines())


def smoke_subset(rows):
    """One two-turn MT-Bench row and one single-turn QA row: 3 requests a mode."""
    mt = next(r for r in rows if r["logical_category"] == "mt_bench" and len(r["turns"]) == 2)
    qa = next(r for r in rows if r["logical_category"] == "qa" and len(r["turns"]) == 1)
    return [mt, qa]


# ------------------------------------------------------------------ the server

MODE_LEVERS = {
    # serial: the drafter stays loaded but is never served from.
    "serial":     {"dspark_serve": 0, "dspark_adaptive": 1, "dspark_verify_rows": 0,
                   "dspark_reasoning_serial": 1},
    # six: DSpark on, admission control off. The trained block is six rows, so
    # "always attempt the full block" is the fixed-six arm; no width pin needed.
    "six":        {"dspark_serve": 1, "dspark_adaptive": 0, "dspark_verify_rows": 0,
                   "dspark_reasoning_serial": 1},
    # controller: the shipped production default.
    "controller": {"dspark_serve": 1, "dspark_adaptive": 1, "dspark_verify_rows": 0,
                   "dspark_reasoning_serial": 1},
}


def mode_environment(mode):
    """Environment for the one-server-per-mode path (--use-running-server off)."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("DS4_") or k.startswith("MTL_"))}
    if mode == "serial":
        return env
    env["DS4_V41_DSPARK_DECODE_READY"] = "1"
    if mode == "six":
        env["DS4_DS41_DSPARK_ADAPTIVE"] = "0"
        env["DS4_V41_DSPARK_VERIFY_ROWS"] = "6"
    # "controller" takes the shipped defaults and sets nothing else.
    return env


def start_server(args, mode, log_path):
    cmd = [args.server, "-m", args.model, "-c", str(args.ctx),
           "--host", args.host, "--port", str(args.port)]
    if mode != "serial":
        cmd += ["--mtp-model", args.mtp_model, "--dspark"]
    env = mode_environment(mode)

    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + args.boot_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise Refuse(f"{mode} server exited with {proc.returncode}; see {log_path}")
        with open(log_path, "rb") as f:
            if b"listening on" in f.read():
                time.sleep(5.0)
                return proc, cmd
        time.sleep(5.0)
    proc.send_signal(signal.SIGTERM)
    raise Refuse(f"{mode} server did not come up in {args.boot_timeout}s; see {log_path}")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=60)


def post(host, port, body, timeout, path="/v1/chat/completions"):
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, repr(exc).encode()


def apply_mode_levers(args, mode):
    """Switch the resident server into `mode` and return its full lever map.

    This is the receipt that proves per-mode lever state: the endpoint answers
    with every lever the binary knows, read back after the write.
    """
    status, raw = post(args.host, args.port, MODE_LEVERS[mode],
                       args.request_timeout, path="/debug/levers")
    require(status == 200, f"/debug/levers refused mode {mode}: {raw[:200]!r}")
    reply = json.loads(raw.decode())
    require(reply.get("unknown", 1) == 0,
            f"/debug/levers did not know every lever for {mode}: {reply}")
    levers = reply["levers"]
    for key, want in MODE_LEVERS[mode].items():
        require(levers.get(key) == want,
                f"lever {key} read back as {levers.get(key)}, want {want}")
    return levers


# -------------------------------------------------------------------- the run

def messages_for_turn(turns, answers, turn_index):
    messages = []
    for i in range(turn_index):
        messages.append({"role": "user", "content": turns[i]})
        messages.append({"role": "assistant", "content": answers[i]})
    messages.append({"role": "user", "content": turns[turn_index]})
    return messages


def unit_id(row, turn_index):
    return (f"{row['logical_category']}.{row['category_index']:02d}"
            f".q{row['question_id']}.t{turn_index}")


PROMPT_DONE = re.compile(
    r"^(?:[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} )?"
    r"ds4-server: chat .*? prompt done (?P<prefill>[0-9]+(?:\.[0-9]+)?)s$", re.MULTILINE)


def parse_boundaries(segment, record):
    """Three separate timing boundaries, as the GLM subset report keeps them.

    server end-to-end  the native chat timer, from before prefill through the
                       final generated token (the `finish=` line).
    decode-only        that timer minus the prefill timer (`prompt done`), i.e.
                       the decode-progress clock that starts after prefill.
    client wall        the monotonic duration of the HTTP request (set by the
                       caller, not here).
    """
    final = None
    for match in FINAL_LOG.finditer(segment):
        final = match
    if final:
        record["server_generated"] = int(final.group("generated"))
        record["server_elapsed_ms"] = float(final.group("elapsed")) * 1e3
        record["server_finish"] = final.group("finish")
    prefill = None
    for match in PROMPT_DONE.finditer(segment):
        prefill = match
    if prefill:
        record["server_prefill_ms"] = float(prefill.group("prefill")) * 1e3
    elif final:
        # A prompt restored in full from the KV cache prefills nothing and the
        # engine logs no prefill timer for it. Zero, recorded as such.
        record["server_prefill_ms"] = 0.0
        record["server_prefill_absent"] = True
    if "server_elapsed_ms" in record and "server_prefill_ms" in record:
        record["server_decode_ms"] = (record["server_elapsed_ms"]
                                      - record["server_prefill_ms"])


DSPARK_KEYS = ("dspark_cycles", "dspark_committed", "dspark_serial_rows",
               "dspark_tokens_per_cycle", "dspark_attempts",
               "dspark_skipped_steps", "dspark_windows", "dspark_backoffs",
               "dspark_losing_cycles", "dspark_serial_ms", "dspark_cycle_ms",
               "dspark_net_ms")


def capture_dspark(usage, record):
    """Committed rows per verify from the real consumed counts, not an estimate.

    dspark_committed counts tokens the frontend actually consumed out of each
    verified block (a terminal partial block included); dspark_cycles counts the
    verify calls that produced them.
    """
    block = usage.get("ds4_dspark")
    if not isinstance(block, dict):
        record["committed_per_verify"] = None
        record["committed_per_verify_note"] = (
            "UNAVAILABLE: this response carried no ds4_dspark usage block")
        return
    for key in DSPARK_KEYS:
        if key in block:
            record[key] = block[key]
    cycles = block.get("dspark_cycles") or 0
    committed = block.get("dspark_committed") or 0
    record["committed_per_verify"] = (committed / cycles) if cycles else None


def one_turn(args, mode, body, log_path, log_offset, uid, row, turn_index):
    """One attempt, no retry. Returns (record, new_log_offset, text_or_None)."""
    request_bytes = json.dumps(body, ensure_ascii=False).encode()
    t0 = time.monotonic()
    status, raw = post(args.host, args.port, body, args.request_timeout)
    wall_ms = (time.monotonic() - t0) * 1e3

    time.sleep(0.5)  # let the server flush its final line
    with open(log_path, "rb") as f:
        f.seek(log_offset)
        segment = f.read().decode("utf-8", "replace")
    new_offset = os.path.getsize(log_path)

    record = {
        "mode": mode, "unit": uid,
        "logical_category": row["logical_category"],
        "question_id": row["question_id"], "turn": turn_index,
        "http_status": status, "client_wall_ms": wall_ms,
        "request_sha256": sha256_bytes(request_bytes),
        "request_bytes": len(request_bytes),
    }
    text = None
    if status == 200:
        try:
            reply = json.loads(raw.decode())
            text = reply["choices"][0]["message"]["content"] or ""
            usage = reply.get("usage", {})
            details = usage.get("prompt_tokens_details") or {}
            record.update({
                "status": "ok",
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens", 0),
                "finish_reason": reply["choices"][0].get("finish_reason"),
                "answer_bytes": len(text.encode()),
                "answer_sha256": sha256_bytes(text.encode()),
            })
            capture_dspark(usage, record)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            record.update({"status": "malformed", "error": repr(exc)})
            text = None
    else:
        record.update({"status": "http_error",
                       "body_head": raw[:400].decode("utf-8", "replace")})
    parse_boundaries(segment, record)
    return record, new_offset, text


def prime_prefixes(args, rows, log_path, log_offset, model_id):
    """Untimed one-token pass so every mode meets the same KV-cache state.

    The resident server carries a --kv-disk-dir from the gate boot and the
    binary has no runtime bypass, so a prompt long enough to be stored would be
    cold for whichever mode ran first and warm for the other two. Touching every
    first-turn prompt once, before any timed request, removes that ordering
    asymmetry: each timed request in every mode then reports the same
    cached_tokens, which the per-request receipts record so it can be checked.
    """
    touched = []
    for row in rows:
        body = {"model": model_id,
                "messages": messages_for_turn(row["turns"], [], 0),
                "max_tokens": 1, "temperature": 0.0, "stream": False,
                "reasoning_effort": REASONING_EFFORT}
        status, raw = post(args.host, args.port, body, args.request_timeout)
        touched.append({"unit": unit_id(row, 0), "http_status": status})
        log_offset = os.path.getsize(log_path)
    return touched, log_offset


def run_mode(args, mode, rows, out_dir, model_id, log_path, log_offset):
    proc, cmd = (None, None)
    records = []
    levers = None
    try:
        if args.use_running_server:
            levers = apply_mode_levers(args, mode)
            cmd = ["<resident server>", f"levers={json.dumps(MODE_LEVERS[mode])}"]
            log_offset = os.path.getsize(log_path)
        else:
            proc, cmd = start_server(args, mode, log_path)
            log_offset = os.path.getsize(log_path)

        for row in rows:
            answers = []
            blocked = False
            for turn_index, _ in enumerate(row["turns"]):
                uid = unit_id(row, turn_index)
                if blocked:
                    records.append({"mode": mode, "unit": uid, "status": "blocked",
                                    "reason": "prior turn of this question failed"})
                    continue

                body = {
                    "model": model_id,
                    "messages": messages_for_turn(row["turns"], answers, turn_index),
                    "max_tokens": MAX_GENERATED_TOKENS,
                    "temperature": 0.0,
                    "stream": False,
                    "reasoning_effort": REASONING_EFFORT,
                }
                record, log_offset, text = one_turn(
                    args, mode, body, log_path, log_offset, uid, row, turn_index)
                ok = record.get("status") == "ok"
                if ok:
                    answers.append(text)
                records.append(record)

                flag = "ok " if ok else "FAIL"
                print(f"  [{flag}] {uid:<34} {record['client_wall_ms']/1000:7.2f}s  "
                      f"gen={record.get('completion_tokens', '?')} "
                      f"cached={record.get('cached_tokens', '?')}")
                if not ok:
                    blocked = True  # a failed first turn blocks its dependent turn
    finally:
        if not args.use_running_server:
            stop_server(proc)
    return records, cmd, levers, log_offset


# ------------------------------------------------------------------ the report

BOUNDARIES = (("server_end_to_end", "server_elapsed_ms"),
              ("decode_only", "server_decode_ms"),
              ("client_wall", "client_wall_ms"))


def boundary_stats(ok, field):
    rows = [r for r in ok if isinstance(r.get(field), (int, float)) and r[field] > 0]
    if not rows:
        return None
    tokens = sum(r.get("completion_tokens") or 0 for r in rows)
    elapsed_s = sum(r[field] for r in rows) / 1e3
    per_question = [(r.get("completion_tokens") or 0) / (r[field] / 1e3)
                    for r in rows if r.get("completion_tokens")]
    return {
        "requests": len(rows),
        "completion_tokens": tokens,
        "elapsed_s": elapsed_s,
        "pooled_tps": tokens / elapsed_s if elapsed_s else None,
        "mean_per_question_tps":
            sum(per_question) / len(per_question) if per_question else None,
    }


def summarise(records):
    ok = [r for r in records if r.get("status") == "ok"]
    out = {
        "requests": len(records), "ok": len(ok),
        "failed": sum(1 for r in records if r.get("status") not in ("ok", "blocked")),
        "blocked": sum(1 for r in records if r.get("status") == "blocked"),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "cached_prompt_tokens": sum(r.get("cached_tokens") or 0 for r in ok),
    }
    if not ok:
        return out
    out["boundaries"] = {name: boundary_stats(ok, field) for name, field in BOUNDARIES}
    cycles = sum(r.get("dspark_cycles") or 0 for r in ok)
    committed = sum(r.get("dspark_committed") or 0 for r in ok)
    out["dspark"] = {
        "cycles": cycles,
        "committed": committed,
        "committed_per_verify": (committed / cycles) if cycles else None,
        "serial_rows": sum(r.get("dspark_serial_rows") or 0 for r in ok),
        "attempts": sum(r.get("dspark_attempts") or 0 for r in ok),
        "skipped_steps": sum(r.get("dspark_skipped_steps") or 0 for r in ok),
        "windows": sum(r.get("dspark_windows") or 0 for r in ok),
        "backoffs": sum(r.get("dspark_backoffs") or 0 for r in ok),
        "losing_cycles": sum(r.get("dspark_losing_cycles") or 0 for r in ok),
        "requests_with_backoff": sum(1 for r in ok if (r.get("dspark_backoffs") or 0) > 0),
    }
    by_category = {}
    for name in LOGICAL_CATEGORIES:
        rows = [r for r in ok if r.get("logical_category") == name]
        if not rows:
            continue
        by_category[name] = {
            "requests": len(rows),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
            "boundaries": {b: boundary_stats(rows, f) for b, f in BOUNDARIES},
            "committed_per_verify": (
                (sum(r.get("dspark_committed") or 0 for r in rows) /
                 sum(r.get("dspark_cycles") or 0 for r in rows))
                if sum(r.get("dspark_cycles") or 0 for r in rows) else None),
            "backoffs": sum(r.get("dspark_backoffs") or 0 for r in rows),
        }
    out["by_category"] = by_category
    return out


def matched_ratios(summaries, base_mode="serial"):
    """Matched elapsed and throughput ratios, kept separate.

    Natural-EOS output lengths can differ between modes, so an elapsed ratio and
    a throughput ratio are not the same statement. Matched means the two modes
    are compared only over the units both completed.
    """
    base = summaries.get(base_mode)
    if not base or "boundaries" not in base:
        return {}
    out = {}
    for mode, s in summaries.items():
        if mode == base_mode or "boundaries" not in s:
            continue
        entry = {}
        for name, _ in BOUNDARIES:
            b, m = base["boundaries"].get(name), s["boundaries"].get(name)
            if not b or not m or not b["pooled_tps"] or not m["pooled_tps"]:
                continue
            entry[name] = {
                "pooled_tps_ratio": m["pooled_tps"] / b["pooled_tps"],
                "mean_per_question_tps_ratio":
                    (m["mean_per_question_tps"] / b["mean_per_question_tps"])
                    if b["mean_per_question_tps"] else None,
                "elapsed_ratio": m["elapsed_s"] / b["elapsed_s"] if b["elapsed_s"] else None,
            }
        out[mode] = entry
    return out


def identity_table(all_records, base_mode="serial"):
    """Byte identity of every answer against the serial answer, per unit."""
    by_mode = {}
    for mode, records in all_records.items():
        by_mode[mode] = {r["unit"]: r.get("answer_sha256")
                         for r in records if r.get("status") == "ok"}
    base = by_mode.get(base_mode, {})
    table, mismatches = {}, {}
    for mode, shas in by_mode.items():
        if mode == base_mode:
            continue
        units = sorted(set(base) & set(shas))
        bad = [u for u in units if base[u] != shas[u]]
        table[mode] = {"compared": len(units), "identical": len(units) - len(bad),
                       "mismatched": len(bad)}
        if bad:
            mismatches[mode] = [{"unit": u, base_mode: base[u], mode: shas[u]} for u in bad]
    return table, mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec-bench-root", required=True)
    ap.add_argument("--server", default="./ds4-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp-model", help="required unless --modes is serial only")
    ap.add_argument("--model-id", default="deepseek-v4.1-flash")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--boot-timeout", type=float, default=1200.0)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--use-running-server", action="store_true",
                    help="do not start or stop a server: switch the mode of the "
                         "one already listening on --host/--port through "
                         "/debug/levers, and read its log from --server-log")
    ap.add_argument("--server-log",
                    help="log file of the resident server (required with "
                         "--use-running-server)")
    ap.add_argument("--prime-prefixes", action="store_true",
                    help="untimed one-token pass over every first turn before "
                         "the timed modes, so all modes meet the same KV-cache "
                         "state on a server booted with --kv-disk-dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate identities and freeze the plan; no server, no GPU")
    ap.add_argument("--preflight-smoke", action="store_true",
                    help="2 questions, 3 requests per mode; always labelled smoke")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        require(m in MODES, f"unknown mode {m}")
    if args.use_running_server:
        require(args.server_log and os.path.isfile(args.server_log),
                "--use-running-server needs --server-log pointing at the live log")
    elif any(m != "serial" for m in modes):
        require(args.mtp_model, "--mtp-model is required for the six and controller modes")

    rows, manifest = load_subset(args.spec_bench_root)
    if args.preflight_smoke:
        rows = smoke_subset(rows)
        manifest["preflight_smoke"] = True
        manifest["smoke_note"] = ("functional check only. No subset performance "
                                  "claim may be made from a smoke run.")
    manifest["resident_server"] = bool(args.use_running_server)
    if args.use_running_server:
        manifest["resident_server_note"] = (
            "one resident server switched per request set through /debug/levers "
            "instead of one server per mode: model-load economy. The lever map "
            "read back after each switch is kept in <mode>-levers.json.")
        manifest["mode_levers"] = MODE_LEVERS

    os.makedirs(args.output, exist_ok=False)
    with open(os.path.join(args.output, "subset-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    identity = {
        "utc": utc(),
        "spec_bench_commit": SPEC_BENCH_COMMIT,
        # basename only: the checkout path is a local detail and receipts in
        # this repository are bound by identity (commit + file digests), not path.
        "spec_bench_root": os.path.basename(os.path.abspath(args.spec_bench_root)),
        "question_file_sha256": QUESTION_SHA256,
        "server_sha256": sha256_file(args.server),
        "model": os.path.basename(args.model),
        "model_bytes": os.path.getsize(args.model),
        "modes": modes,
        "ctx": args.ctx,
        "resident_server": bool(args.use_running_server),
        "prime_prefixes": bool(args.prime_prefixes),
        "preflight_smoke": bool(args.preflight_smoke),
    }
    if args.mtp_model:
        identity["mtp_model"] = os.path.basename(args.mtp_model)
        identity["mtp_model_bytes"] = os.path.getsize(args.mtp_model)
    for name in ("LICENSE", "Readme.md", "README.md"):
        path = os.path.join(args.spec_bench_root, name)
        if os.path.isfile(path):
            identity[f"spec_bench_{name}_sha256"] = sha256_file(path)
    with open(os.path.join(args.output, "identity.json"), "w") as f:
        json.dump(identity, f, indent=2)

    turns = sum(len(r["turns"]) for r in rows)
    print(f"{len(rows)} questions, {turns} turn requests per mode, "
          f"{turns * len(modes)} requests across {', '.join(modes)}")
    if args.preflight_smoke:
        print("PREFLIGHT SMOKE — functional check only, no performance claim")
    if args.dry_run:
        print(f"dry run: plan frozen in {args.output}, no server started")
        return 0

    log_path = (args.server_log if args.use_running_server
                else os.path.join(args.output, "server.log"))
    log_offset = os.path.getsize(log_path) if os.path.isfile(log_path) else 0

    if args.prime_prefixes:
        require(args.use_running_server, "--prime-prefixes needs a resident server")
        print("\n== prime (untimed, 1 token per first turn) ==")
        apply_mode_levers(args, "serial")
        touched, log_offset = prime_prefixes(args, rows, log_path, log_offset,
                                             args.model_id)
        with open(os.path.join(args.output, "prime-pass.json"), "w") as f:
            json.dump({"requests": touched}, f, indent=2)
        print(f"  primed {len(touched)} first-turn prefixes")

    summaries, all_records, levers_by_mode = {}, {}, {}
    for mode in modes:
        print(f"\n== {mode} ==")
        records, cmd, levers, log_offset = run_mode(
            args, mode, rows, args.output, args.model_id, log_path, log_offset)
        with open(os.path.join(args.output, f"{mode}-records.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if levers is not None:
            levers_by_mode[mode] = levers
            with open(os.path.join(args.output, f"{mode}-levers.json"), "w") as f:
                json.dump({"requested": MODE_LEVERS[mode], "read_back": levers},
                          f, indent=2)
        all_records[mode] = records
        summaries[mode] = summarise(records)
        summaries[mode]["server_argv"] = cmd
        if not args.use_running_server:
            summaries[mode]["environment_overrides"] = {
                k: v for k, v in mode_environment(mode).items() if k.startswith("DS4_")}

    table, mismatches = identity_table(all_records)
    report = {
        "identity": identity,
        "modes": summaries,
        "matched_ratios_vs_serial": matched_ratios(summaries),
        "byte_identity_vs_serial": table,
        "byte_identity_mismatches": mismatches,
        "mode_levers": levers_by_mode,
    }
    with open(os.path.join(args.output, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nmode          ok/req   tokens   e2e t/s   decode t/s   client t/s  "
          "cmt/verify  backoffs")
    for mode in modes:
        s = summaries[mode]
        b = s.get("boundaries") or {}
        def tps(name):
            entry = b.get(name)
            return entry["pooled_tps"] if entry and entry["pooled_tps"] else 0.0
        d = s.get("dspark") or {}
        cpv = d.get("committed_per_verify")
        print(f"{mode:<12} {s['ok']:>3}/{s['requests']:<4} "
              f"{s.get('completion_tokens', 0):>8} "
              f"{tps('server_end_to_end'):>9.3f} "
              f"{tps('decode_only'):>12.3f} "
              f"{tps('client_wall'):>12.3f} "
              f"{(('%.4f' % cpv) if cpv else '--'):>11} "
              f"{d.get('backoffs', 0):>9}")
    for mode, entry in table.items():
        print(f"byte identity {mode} vs serial: "
              f"{entry['identical']}/{entry['compared']} identical")
    print(args.output)
    return 0 if all(s.get("failed", 0) == 0 for s in summaries.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

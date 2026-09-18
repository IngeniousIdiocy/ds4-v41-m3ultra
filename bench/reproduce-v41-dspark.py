#!/usr/bin/env python3
"""Reproduce the V4.1 Flash DSpark committed-throughput arms.

Three modes per fixture, each against that fixture's own matched serial control
measured in the same server:

  serial      no drafting for the request. This is the k=1 non-drafting path
              and the denominator for the other two.
  six         the full trained block proposed at every position, no admission
              control. The trained block is six rows, so this IS fixed six.
  controller  the full trained block plus the windowed cost-feedback admission
              controller. The production default, and the binary's own defaults.

Arms are interleaved per fixture and repeated, because the difference between
modes is smaller than the difference between a warm and a cold machine.

One resident server handles every arm, so the target and drafter load once.
That server needs --debug-levers for the /debug/bench endpoint, which means it
is a measurement server and MUST NOT be used to serve: each call holds its own
full session, and 24 arms took one from 389 MB to 13 GB.

Harness numbers read about 1 t/s above ds4-bench numbers on the same binary.
Never mix this output with reproduce-v41-native.sh output in one table; that is
why every fixture carries its own serial control here.

  bench/reproduce-v41-dspark.py \
      --server ./ds4-server --model M.gguf --mtp-model D.gguf \
      --fixture code=path/to/code.txt --fixture sql=path/to/sql.txt \
      --reference code=refs/gate8k512-code-upstream-bd66c40.text

Run only with exclusive GPU access.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

# The production lever set. Every lever is named in every arm, so nothing
# persists from the previous request.
PRODUCTION_LEVERS = {
    "dspark_capture": 1,
    "mtp_state_fix": 1,
    "mtp_capture_warmup": 1,
    "mtp_engram_rows6": 1,
    "mtp_async_chunks": 1,
    "prefill_decoder_suffix": 1,
    "dspark_excl_eos": 1,
    "dspark_force_reject": 0,
    "dspark_draft_trace": 0,
    "verify_batch_core": 0,
    "mtp_q8_pair6": 1,
    "mtp_q8_stream6": 1,
    "mtp_q8_stream6_mask": 2,
}

# name -> (request dspark flag, the levers that define the arm). verify_rows 0
# is the full trained block; dspark_adaptive 0 proposes everywhere and never
# backs off, which is the fixed-width control.
MODES = {
    "serial":     (False, {"dspark_serve": 0, "dspark_adaptive": 1, "dspark_verify_rows": 0}),
    "six":        (True,  {"dspark_serve": 1, "dspark_adaptive": 0, "dspark_verify_rows": 0}),
    "controller": (True,  {"dspark_serve": 1, "dspark_adaptive": 1, "dspark_verify_rows": 0}),
}

REPORT_FIELDS = ("gen_steady_tps", "prefill_tps", "dspark_cycles",
                 "dspark_tokens_per_cycle", "dspark_verify_ms", "dspark_cycle_ms",
                 "dspark_attempts", "dspark_skipped_steps", "dspark_windows",
                 "dspark_backoffs", "dspark_losing_cycles", "dspark_net_ms",
                 "dspark_serial_rows")


def kv_pairs(values, what):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--{what} wants NAME=PATH, got {v!r}")
        name, path = v.split("=", 1)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {what} file for {name}: {path}")
        out[name] = path
    return out


def start_server(args, log_path):
    cmd = [args.server, "-m", args.model, "--mtp-model", args.mtp_model, "--dspark",
           "--ctx", str(args.ctx_alloc), "--host", args.host, "--port", str(args.port),
           "--debug-levers"]
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("DS4_") or k.startswith("MTL_"))}
    env["DS4_V41_DSPARK_DECODE_READY"] = "1"

    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + args.boot_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}; see {log_path}")
        with open(log_path, "rb") as f:
            if b"listening on" in f.read():
                time.sleep(5.0)
                return proc
        time.sleep(5.0)
    proc.send_signal(signal.SIGTERM)
    raise SystemExit(f"server did not come up within {args.boot_timeout}s; see {log_path}")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=60)


def bench(args, fixture_path, mode, fresh):
    dspark, arm_levers = MODES[mode]
    levers = dict(PRODUCTION_LEVERS)
    levers.update(arm_levers)
    payload = {
        "path": os.path.abspath(fixture_path),
        "ctx_start": args.ctx_start,
        "ctx_alloc": args.ctx_alloc,
        "gen_tokens": args.gen_tokens,
        "fresh": fresh,
        "dspark": dspark,
        "levers": levers,
    }
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/debug/bench",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
        return json.loads(resp.read().decode()), payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="./ds4-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp-model", required=True, help="DSpark support GGUF")
    ap.add_argument("--fixture", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--reference", action="append", metavar="NAME=PATH",
                    help="upstream bd66c40 decoded text to check identity against")
    ap.add_argument("--modes", default="serial,six,controller")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ctx-start", type=int, default=8192)
    ap.add_argument("--ctx-alloc", type=int, default=8960)
    ap.add_argument("--gen-tokens", type=int, default=512)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--boot-timeout", type=float, default=1200.0)
    ap.add_argument("--request-timeout", type=float, default=3600.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--server-already-running", action="store_true")
    args = ap.parse_args()

    fixtures = kv_pairs(args.fixture, "fixture")
    references = kv_pairs(args.reference, "reference")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode: {m}")

    out = args.output or time.strftime("v41-dspark-%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs(out, exist_ok=False)
    log_path = os.path.join(out, "server.log")

    identity = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server_sha256": hashlib.sha256(open(args.server, "rb").read()).hexdigest(),
        "model": os.path.basename(args.model),
        "model_bytes": os.path.getsize(args.model),
        "mtp_model": os.path.basename(args.mtp_model),
        "mtp_model_bytes": os.path.getsize(args.mtp_model),
        "ctx_start": args.ctx_start, "ctx_alloc": args.ctx_alloc,
        "gen_tokens": args.gen_tokens, "modes": modes, "reps": args.reps,
        "production_levers": PRODUCTION_LEVERS,
        "fixtures": {},
    }
    for name, path in fixtures.items():
        with open(path, "rb") as f:
            data = f.read()
        identity["fixtures"][name] = {"bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()}
    with open(os.path.join(out, "identity.json"), "w") as f:
        json.dump(identity, f, indent=2)

    proc = None
    rows = []
    try:
        if not args.server_already_running:
            proc = start_server(args, log_path)

        print(f"{'arm':<26}{'t/s':>10}{'tok/cyc':>10}{'verify ms':>11}"
              f"{'bkoff/win':>12}  identity")
        for rep in range(1, args.reps + 1):
            for name, path in fixtures.items():
                for mode in modes:
                    tag = f"{name}-{mode}-r{rep}"
                    # Only the first arm on a fixture re-prefills; the rest
                    # restore the same prefix, which is what makes the arms
                    # comparable and the run affordable.
                    fresh = (rep == 1 and mode == modes[0])
                    result, payload = bench(args, path, mode, fresh)
                    text = result.get("text", "")
                    with open(os.path.join(out, f"{tag}.text"), "w", encoding="utf-8") as f:
                        f.write(text)

                    row = {"arm": tag, "fixture": name, "mode": mode, "rep": rep,
                           "fresh": fresh, "request": payload,
                           "text_bytes": len(text.encode()),
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                    for k in REPORT_FIELDS:
                        if k in result:
                            row[k] = result[k]

                    ref = references.get(name)
                    if ref:
                        with open(ref, "rb") as f:
                            row["identity"] = ("IDENTICAL" if f.read() == text.encode()
                                               else "DIVERGES")
                    else:
                        row["identity"] = "no-reference"
                    rows.append(row)

                    cycles = row.get("dspark_cycles") or 1
                    verify = (row.get("dspark_verify_ms") or 0) / cycles
                    print(f"{tag:<26}{row.get('gen_steady_tps', 0):>10.3f}"
                          f"{row.get('dspark_tokens_per_cycle', 0):>10.3f}"
                          f"{verify:>11.2f}"
                          f"{str(row.get('dspark_backoffs', '-')) + '/' + str(row.get('dspark_windows', '-')):>12}"
                          f"  {row['identity']}")
    finally:
        if proc is not None:
            stop_server(proc)

    with open(os.path.join(out, "arms.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Per-fixture summary against that fixture's own serial control.
    print()
    for name in fixtures:
        def mean(mode):
            vals = [r["gen_steady_tps"] for r in rows
                    if r["fixture"] == name and r["mode"] == mode and "gen_steady_tps" in r]
            return sum(vals) / len(vals) if vals else None
        base = mean("serial")
        if base is None:
            continue
        parts = [f"serial {base:.3f}"]
        for mode in modes:
            if mode == "serial":
                continue
            m = mean(mode)
            if m is not None:
                parts.append(f"{mode} {m:.3f} ({m / base:.3f}x)")
        print(f"{name:<10} " + "   ".join(parts))

    diverged = [r["arm"] for r in rows if r["identity"] == "DIVERGES"]
    if diverged:
        print(f"\nIDENTITY FAILURE on {len(diverged)} arm(s): {', '.join(diverged)}")
        return 1
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

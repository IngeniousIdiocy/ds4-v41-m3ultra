#!/usr/bin/env python3
"""Reproduce the V4.1 Flash time-to-first-token arms through the real HTTP path.

Three arms, in this order, against one server:

  cold      the KV disk directory is emptied first, so the prompt is prefilled
            from nothing.
  restored  a four-token displacement request is sent first, which leaves the
            live session holding ~20 tokens; the next request then logs
            "live kv cache miss" and takes the disk path.
  append    the same, plus a short user turn on top of the restored prefix.

The client is stdlib sockets. It timestamps the moment the request body is on
the wire and the moment the first SSE content_block_delta carrying non-empty
text arrives. No engine instrumentation is read: this is the number a client
actually experiences.

Run only with exclusive GPU access. The script starts and stops the server
itself unless --server-already-running is given.

  bench/reproduce-v41-ttft.py \
      --server ./ds4-server --model M.gguf --system-file sys.txt

Do NOT pass --debug-levers to the server: it holds a full session per bench
fixture and is worth about 24 GiB.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time


# ---------------------------------------------------------------- the client

def request(host, port, body, timeout=1800.0):
    payload = json.dumps(body).encode()
    head = (
        "POST /v1/messages HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Accept: text/event-stream\r\n"
        "Connection: close\r\n\r\n"
    ).encode()

    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    t0 = time.monotonic()
    sock.sendall(head + payload)
    t_send = time.monotonic()

    buf, t_hdr, t_first = b"", None, None
    text, thinking, usage = [], [], {}
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        if t_hdr is None:
            t_hdr = time.monotonic()
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "content_block_delta":
                d = ev.get("delta", {})
                piece = d.get("text") or d.get("thinking") or d.get("partial_json") or ""
                if piece:
                    if t_first is None:
                        t_first = time.monotonic()
                    (thinking if d.get("type") == "thinking_delta" else text).append(piece)
            elif kind in ("message_delta", "message_start"):
                u = ev.get("usage") or (ev.get("message") or {}).get("usage") or {}
                usage.update(u)
    t_done = time.monotonic()
    sock.close()

    joined = "".join(text)
    return {
        "send_ms": (t_send - t0) * 1e3,
        "hdr_ms": (t_hdr - t0) * 1e3 if t_hdr else None,
        "ttft_ms": (t_first - t0) * 1e3 if t_first else None,
        "total_ms": (t_done - t0) * 1e3,
        "text": joined,
        "text_bytes": len(joined.encode()),
        "text_sha256": hashlib.sha256(joined.encode()).hexdigest(),
        "thinking": "".join(thinking),
        "usage": usage,
    }


# ---------------------------------------------------------------- the server

def start_server(args, kv_dir, log_path):
    cmd = [args.server, "-m", args.model, "-c", str(args.ctx),
           "--host", args.host, "--port", str(args.port),
           "--kv-disk-dir", kv_dir, "--kv-disk-space-mb", str(args.kv_disk_space_mb)]
    if args.mtp_model:
        cmd += ["--mtp-model", args.mtp_model, "--dspark"]

    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("DS4_") or k.startswith("MTL_"))}
    if args.mtp_model:
        # --dspark is admitted only when this is set. The windowed admission
        # controller is on by default from here and needs no configuration file.
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


# ------------------------------------------------------------------ the arms

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="./ds4-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp-model", help="DSpark support GGUF; enables --dspark")
    ap.add_argument("--system-file", required=True, help="the long system prompt to time")
    ap.add_argument("--append-file", help="short user turn for the restored+append arm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--kv-disk-space-mb", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--boot-timeout", type=float, default=1200.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--arms", default="cold,restored,append",
                    help="comma-separated subset of cold,restored,append")
    ap.add_argument("--server-already-running", action="store_true",
                    help="do not start or stop a server; the cold arm is then not cold")
    args = ap.parse_args()

    out = args.output or time.strftime("v41-ttft-%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs(out, exist_ok=False)
    kv_dir = os.path.join(out, "kv")
    log_path = os.path.join(out, "server.log")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("cold", "restored", "append"):
            raise SystemExit(f"unknown arm: {a}")
    if "append" in arms and not args.append_file:
        raise SystemExit("--append-file is required for the append arm")

    def digest(path):
        with open(path, "rb") as f:
            data = f.read()
        return len(data), hashlib.sha256(data).hexdigest()

    identity = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "server_sha256": hashlib.sha256(open(args.server, "rb").read()).hexdigest(),
        "model": os.path.basename(args.model),
        "model_bytes": os.path.getsize(args.model),
        "mtp_model": os.path.basename(args.mtp_model) if args.mtp_model else None,
        "mtp_model_bytes": os.path.getsize(args.mtp_model) if args.mtp_model else None,
        "ctx": args.ctx,
        "max_tokens": args.max_tokens,
        "arms": arms,
    }
    identity["system_bytes"], identity["system_sha256"] = digest(args.system_file)
    if args.append_file:
        identity["append_bytes"], identity["append_sha256"] = digest(args.append_file)
    with open(os.path.join(out, "identity.json"), "w") as f:
        json.dump(identity, f, indent=2)

    system_text = open(args.system_file, encoding="utf-8").read()
    append_text = open(args.append_file, encoding="utf-8").read() if args.append_file else None

    def body(system, user):
        b = {"model": "ds4", "max_tokens": args.max_tokens, "temperature": 0.0,
             "stream": True, "thinking": {"type": "disabled"},
             "messages": [{"role": "user", "content": user}]}
        if system is not None:
            b["system"] = system
        return b

    proc = None
    rows = []
    try:
        if not args.server_already_running:
            if os.path.isdir(kv_dir):
                shutil.rmtree(kv_dir)
            os.makedirs(kv_dir)
            proc = start_server(args, kv_dir, log_path)

        def shot(tag, system, user, result=True):
            r = request(args.host, args.port, body(system, user))
            r["arm"] = tag
            r["result"] = result
            r["wall_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rows.append(r)
            with open(os.path.join(out, f"{tag}.text"), "w", encoding="utf-8") as f:
                f.write(r["text"])
            ttft = r["ttft_ms"]
            print(f"{tag:<28} TTFT {ttft:>10.2f} ms   total {r['total_ms']:>9.2f} ms   "
                  f"text {r['text_sha256'][:16]}" if ttft is not None
                  else f"{tag:<28} TTFT  -- (no text delta)")
            return r

        if "cold" in arms:
            shot("cold", system_text, "Hi.")
        if "restored" in arms:
            # Displacement: leaves the live session too short to serve the next
            # request, forcing the disk path. Not a result.
            shot("displace-1", None, "ping", result=False)
            shot("restored", system_text, "Hi.")
        if "append" in arms:
            shot("displace-2", None, "ping", result=False)
            shot("restored-append", system_text, append_text)
    finally:
        if proc is not None:
            stop_server(proc)

    with open(os.path.join(out, "arms.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n")

    print(f"\n{out}")
    if os.path.exists(log_path):
        print("server lines worth checking (which path each arm took):")
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if any(s in line for s in ("live kv cache miss", "kv cache hit",
                                           "kv cache stored", "prompt done")):
                    print("  " + line.rstrip())


if __name__ == "__main__":
    sys.exit(main())

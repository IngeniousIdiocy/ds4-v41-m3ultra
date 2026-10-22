#!/usr/bin/env python3
"""Real-model disk KV restore of an image chat across a server restart.

Starts ds4-server twice on a fresh --kv-disk-dir with the given server
arguments (a V4.1 model and --vision). The chat has the shape dsh sends on
/v1/messages: a read_image tool_result carrying the image, resent byte for byte
on every turn. After a restart the next turn must restore from the shutdown
checkpoint (log: "multimodal disk kv hit", no full re-read) and still answer
from the image; the same text with another image must miss and re-read.
With --text-only it instead times a text-only checkpoint save and restore, to
compare builds.

  python3 tests/test_server_vision_disk_cache.py --output /tmp/vdc -- \\
      -m DeepSeek-V4.1-Flash.gguf --vision DeepSeek-V4.1-Flash-Vision.gguf
"""
import argparse
import base64
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

FIXTURES = Path(__file__).resolve().parent / "vision-fixtures/glm53"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="./ds4-server")
    parser.add_argument("--port", default=0, type=int)
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = [a for a in args.server_args if a != "--"]
    args.output.mkdir(parents=True, exist_ok=True)
    kv = args.output / "kv"
    kv.mkdir(exist_ok=True)
    assert not any(kv.iterdir()), f"{kv} must start empty"
    if not args.port:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            args.port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{args.port}"
    runs = []

    def start(label):
        log_path = args.output / f"server-{label}.log"
        log = log_path.open("w")
        proc = subprocess.Popen([args.server, *server_args, "--host", "127.0.0.1",
                                 "--port", str(args.port), "--kv-disk-dir", str(kv)],
                                stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            assert proc.poll() is None, f"server exited during startup, see {log_path}"
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=5):
                    break
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                assert time.monotonic() < deadline, "server startup timed out"
                time.sleep(0.5)
        runs.append((proc, log, log_path))
        return log_path

    def stop():
        proc, log, log_path = runs[-1]
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=300)
        log.close()
        return log_path.read_text()

    def ask(label, messages, max_tokens=1024):
        body = {"model": args.model, "max_tokens": max_tokens, "stream": False,
                "thinking": {"type": "enabled"}, "output_config": {"effort": "low"},
                "messages": messages}
        req = urllib.request.Request(base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise AssertionError(exc.read().decode()) from exc
        text = "".join(b.get("text", "") for b in reply["content"] if b.get("type") == "text")
        usage = reply["usage"]
        result = {"label": label, "seconds": round(time.monotonic() - started, 2),
                  "input": usage.get("input_tokens"),
                  "cached": usage.get("cache_read_input_tokens", 0), "text": text}
        print(json.dumps(result), flush=True)
        (args.output / f"{label}.json").write_text(json.dumps(result, indent=2))
        return text, result

    def image_result(name):
        data = base64.b64encode((FIXTURES / name).read_bytes()).decode()
        return [{"type": "text", "text": "<path>/workspace/" + name + "</path>\n<type>image</type>"},
                {"type": "text", "text": "Image \"" + name + "\"; request preview."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": data}}]

    def chat(name):
        return [
            {"role": "user", "content": [{"type": "text", "text":
                "Read the ticket image at /workspace/" + name + " and answer from it."}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_read_1",
                "name": "read_image", "input": {"file_path": "/workspace/" + name}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_read_1",
                "content": image_result(name)},
                {"type": "text", "text": "What is the access code? Reply with just the code."}]},
        ]

    try:
        if args.text_only:
            story = " ".join(f"Line {i}: the lighthouse keeper logged ship {i * 7 % 997}."
                             for i in range(700))
            history = [{"role": "user", "content": story + "\nWhich ship did line 3 log?"}]
            start("text-1")
            answer, first = ask("text-turn-1", history, 256)
            history.append({"role": "assistant", "content": answer})
            log = stop()
            saves = re.findall(r"kv cache stored tokens=(\d+).*reason=shutdown.* save=([\d.]+) ms", log)
            assert saves, "no shutdown checkpoint stored"
            start("text-2")
            history.append({"role": "user", "content": "And line 5?"})
            _, second = ask("text-turn-2", history, 256)
            log = stop()
            loads = re.findall(r"kv cache hit text tokens=(\d+).* load=([\d.]+) ms", log)
            assert loads and second["cached"] > 0, "text-only checkpoint did not restore"
            print(json.dumps({"text_only": {"save_ms": float(saves[-1][1]),
                                            "load_ms": float(loads[-1][1]),
                                            "tokens": int(loads[-1][0]),
                                            "turn2_seconds": second["seconds"]}}))
            return

        history = chat("text.png")
        start("image-1")
        answer, _ = ask("image-turn-1", history)
        assert "MINT-731" in answer.upper(), answer
        history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        history.append({"role": "user", "content": "Give just the train number from that image."})
        answer, live = ask("image-turn-2", history)
        assert "482" in answer, answer
        assert live["cached"] > 0, "turn 2 did not reuse the live image prefix"
        history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        log = stop()
        assert re.search(r"kv cache stored tokens=\d+.*reason=shutdown.* images=1", log), \
            "the image chat was not checkpointed at shutdown"

        start("image-2")
        history.append({"role": "user", "content": "Now give just the gate from that image."})
        answer, restored = ask("image-turn-3-restored", history)
        assert "C7" in answer.upper(), answer
        assert restored["cached"] >= live["input"], "restart lost part of the image prefix"

        # The same text with another image: the stored identity must not match.
        other = chat("text.png")
        other[2]["content"][0]["content"][2] = image_result("spatial.png")[2]
        swapped = other + history[3:]
        _, missed = ask("image-turn-3-other-image", swapped, 256)
        log = stop()
        assert re.search(r"multimodal disk kv hit images=1 cached=\d+", log), "no disk restore"
        assert re.search(r"multimodal disk kv miss images=1", log), "other image was not refused"
        assert missed["cached"] < live["input"] // 2, "other image reused the image prefix"
        print("PASS image chat restored after restart; other image re-read", flush=True)
    finally:
        for proc, log, _ in runs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log.close()


if __name__ == "__main__":
    main()

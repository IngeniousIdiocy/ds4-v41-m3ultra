#!/usr/bin/env python3
"""Compare ds4's V4.1 request prompts with DeepSeek's released encoder.

Each request body goes through the server's parser and renderer
(ds4_test --ds41-request-tokens) and, mapped to encoding.py messages the way
DeepSeek's deepseek-recipe maps the Messages API, through encode_messages and
tokenizer.json. Every case must match token for token: plain chat at each
effort, content-block separators, images in tool results (one, several, mixed
with text, out of call order), current-turn and earlier-turn reasoning with and
without declared tools, tool schemas and arguments, and Unicode text.

Usage: DS4_TEST_MODEL=V4.1.gguf tests/test_deepseek41_requests.py \
           encoding/encoding.py tokenizer.json [--request FILE --image-tokens N]
"""
import __future__
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile

from tokenizers import Tokenizer

ENCODE = None
TOKENIZER = None
IMAGE_ID = 129264
EFFORT_ALIASES = {"xhigh": "high", "minimal": "low"}


def _img(_block):
    return {"type": "image", "url": "IMG"}


def _anthropic_blocks(content):
    """User or tool_result content -> encoding.py content list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        if b["type"] == "text":
            out.append({"type": "text", "text": b.get("text") or ""})
        elif b["type"] == "image":
            out.append(_img(b))
        else:
            raise ValueError("unsupported block " + b["type"])
    return out


def anthropic_messages(req):
    msgs = []
    system = req.get("system")
    if isinstance(system, list):
        system = "\n\n".join(b["text"] for b in system
                             if b.get("text") and not b["text"].startswith("x-anthropic-"))
    tools = []
    tool_choice = req.get("tool_choice") or {}
    if req.get("tools") and tool_choice.get("type") != "none":
        tools = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t["input_schema"]}} for t in req["tools"]]
    if system:
        msgs.append({"role": "system", "content": system})
    for m in req["messages"]:
        c = m["content"]
        if m["role"] == "assistant":
            if isinstance(c, str):
                msgs.append({"role": "assistant", "content": c})
                continue
            thinking = [x["thinking"] for x in c if x["type"] == "thinking"]
            a = {"role": "assistant",
                 "content": "\n\n".join(x["text"] for x in c if x["type"] == "text")}
            if thinking:
                a["reasoning_content"] = "\n\n".join(thinking)
            calls = [{"type": "function", "id": x["id"], "function": {
                "name": x["name"], "arguments": json.dumps(x["input"], ensure_ascii=False)}}
                for x in c if x["type"] == "tool_use"]
            if calls:
                a["tool_calls"] = calls
            msgs.append(a)
        elif m["role"] == "system":
            msgs.append({"role": "system", "content": c if isinstance(c, str) else
                         "\n\n".join(x["text"] for x in c if x["type"] == "text")})
        else:
            if isinstance(c, str):
                msgs.append({"role": "user", "content": c})
                continue
            for x in c:
                if x["type"] == "tool_result":
                    msgs.append({"role": "tool", "tool_call_id": x["tool_use_id"],
                                 "content": _anthropic_blocks(x.get("content"))})
                else:
                    msgs.append({"role": "user", "content": _anthropic_blocks([x])})
    if tools:
        if not msgs or msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = tools
    thinking = (req.get("thinking") or {}).get("type", "enabled") != "disabled"
    effort = (req.get("output_config") or {}).get("effort") or req.get("reasoning_effort")
    return msgs, thinking, effort


def _openai_content(c):
    if c is None or isinstance(c, str):
        return c or ""
    out = []
    for b in c:
        if isinstance(b, str):
            out.append({"type": "text", "text": b})
        elif b["type"] in ("image_url", "input_image"):
            out.append(_img(b))
        else:
            out.append({"type": "text", "text": b.get("text") or ""})
    return out


def openai_messages(req):
    msgs = []
    for m in req["messages"]:
        n = {"role": m.get("role", "user"), "content": _openai_content(m.get("content"))}
        if n["role"] == "assistant" and isinstance(n["content"], list):
            n["content"] = "\n\n".join(b["text"] for b in n["content"])
        for k in ("reasoning_content", "tool_calls", "tool_call_id"):
            if m.get(k) is not None:
                n[k] = m[k]
        msgs.append(n)
    tools = req.get("tools") if req.get("tool_choice") != "none" else None
    if tools:
        if not msgs or msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = tools
    think = req.get("thinking")
    thinking = True
    if isinstance(think, dict):
        thinking = think.get("type", "enabled") != "disabled"
    elif isinstance(think, bool):
        thinking = think
    if req.get("think") is not None:
        thinking = bool(req["think"])
    return msgs, thinking, req.get("reasoning_effort")


def render(req, api, default_effort="high"):
    msgs, thinking, effort = (anthropic_messages if api == "anthropic" else openai_messages)(req)
    effort = EFFORT_ALIASES.get(effort, effort) or default_effort
    prompt, mm = ENCODE(msgs, thinking_mode="thinking" if thinking else "chat",
                        reasoning_effort=effort, return_multi_modal_data=True)
    return prompt, len(mm["images"])


def tokenize(prompt, image_tokens=1):
    ids = []
    for k, seg in enumerate(prompt.split("<｜deepseek_image｜>")):
        if k:
            ids += [IMAGE_ID] * image_tokens
        ids += TOKENIZER.encode(seg, add_special_tokens=False).ids
    return ids



# A real 2x2 RGB PNG, so the server's image validation accepts it.
def _png():
    import struct, zlib
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"".join(b"\x00" + bytes([200, 30, 30, 30, 200, 30]) for _ in range(2))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG = base64.b64encode(_png()).decode()


def img():
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}


def oimg():
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}


def text(t):
    return {"type": "text", "text": t}


def use(i, name="read_image", inp=None):
    return {"type": "tool_use", "id": i, "name": name, "input": inp if inp is not None else {"path": i + ".png"}}


def result(i, content):
    return {"type": "tool_result", "tool_use_id": i, "content": content}


def think(t):
    return {"type": "thinking", "thinking": t, "signature": ""}


TOOLS = [
    {"name": "read_image", "description": "Read an image file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string", "description": "File path."}},
                      "required": ["path"]}},
    {"name": "bash", "description": "Run a shell command.\nUse with care.",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string"}, "timeout": {"type": "number", "default": 1.5},
         "env": {"type": "object", "additionalProperties": {"type": "string"}},
         "flags": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
         "required": ["command"]}},
]

UNICODE = ("Il dit : « Bonjour… » puis “quoted”, ‘single’ — dash – en; ¿Qué? ¡Sí! 中文测试，标点。"
           "日本語のテキスト、カタカナ。 한국어 텍스트. Emoji 😀👍🏽 and ZWJ 👩‍💻. "
           "NBSP here, thin space, ideographic　space. Combining é and ä. "
           "Arabic digits ٣٤٥ and superscript x² ½ Ⅻ. Math ∑∫≈≠ ± × ÷ → ⇒ ∞. Currency €£¥₹ ©®™ °. "
           ".”\n?»\n…\n\"Hi!\"\n— end —\n\tTabbed\u000bVT\u000cFF\r\nCRLF  \n   x\n")


def anth(messages, system=None, tools=None, effort="low", thinking=True, **kw):
    r = {"model": "deepseek-v4.1-flash", "max_tokens": 64, "messages": messages,
         "thinking": {"type": "enabled" if thinking else "disabled"}}
    if effort:
        r["output_config"] = {"effort": effort}
    if system is not None:
        r["system"] = system
    if tools is not None:
        r["tools"] = tools
    r.update(kw)
    return ("anthropic", r)


def oai(messages, tools=None, effort="low", thinking=True, **kw):
    r = {"model": "deepseek-v4.1-flash", "max_tokens": 64, "messages": messages}
    if effort:
        r["reasoning_effort"] = effort
    if not thinking:
        r["thinking"] = {"type": "disabled"}
    if tools is not None:
        r["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"]}} for t in tools]
    r.update(kw)
    return ("openai", r)


def history_turns(tools):
    """q1/a1(r1), q2, a2(r2 + call), result: current-turn thinking across a tool result."""
    return [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [think("r1"), text("a1")]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [think("r2"), use("c1", "bash", {"command": "ls"})]},
        {"role": "user", "content": [result("c1", "value")]},
    ]


CASES = {}
CASES["plain_low"] = anth([{"role": "user", "content": "Hello there."}], system="Be brief.", effort="low")
CASES["plain_high"] = anth([{"role": "user", "content": "Hello there."}], system="Be brief.", effort="high")
CASES["plain_max"] = anth([{"role": "user", "content": "Hello there."}], effort="max")
CASES["plain_nothink"] = anth([{"role": "user", "content": "Hello there."}], system="Be brief.", thinking=False)
CASES["plain_multiturn"] = anth([
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": [think("r1"), text("a1")]},
    {"role": "user", "content": "q2"}], system="S")
CASES["system_blocks"] = anth([{"role": "user", "content": "Hi"}],
                              system=[text("Part one."), text("Part two.")])
CASES["user_text_blocks"] = anth([{"role": "user", "content": [text("First."), text("Second.")]}])
CASES["user_image_text"] = anth([{"role": "user", "content": [text("Look:"), img(), text("What is it?")]}])
CASES["assistant_text_blocks"] = anth([
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": [think("t1"), think("t2"), text("A."), text("B.")]},
    {"role": "user", "content": "more"}], tools=TOOLS)
CASES["tool_image_one"] = anth([
    {"role": "user", "content": "Read a.png"},
    {"role": "assistant", "content": [think("Read it."), use("a")]},
    {"role": "user", "content": [result("a", [img()])]}], system="Sys", tools=TOOLS)
CASES["tool_image_mixed"] = anth([
    {"role": "user", "content": "Read a.png"},
    {"role": "assistant", "content": [think("Read it."), use("a")]},
    {"role": "user", "content": [result("a", [text("<content>header</content>"), img(), text("trailer")])]}],
    system="Sys", tools=TOOLS)
CASES["tool_image_several"] = anth([
    {"role": "user", "content": "Read both"},
    {"role": "assistant", "content": [think("Both."), use("a"), use("b")]},
    {"role": "user", "content": [result("a", [text("result a"), img()]),
                                  result("b", [text("result b"), img(), img()])]}], tools=TOOLS)
CASES["tool_image_out_of_order"] = anth([
    {"role": "user", "content": [text("history"), img()]},
    {"role": "assistant", "content": [think("Both."), use("a"), use("b")]},
    {"role": "user", "content": [result("b", [text("result b"), img()]),
                                  result("a", [text("result a"), img()])]}], tools=TOOLS)
CASES["tool_image_then_text"] = anth([
    {"role": "user", "content": "Read a"},
    {"role": "assistant", "content": [use("a")]},
    {"role": "user", "content": [result("a", [img()]), text("Now describe it.")]}], tools=TOOLS)
CASES["think_tools_current_turn"] = anth(history_turns(True), system="S", tools=TOOLS)
CASES["think_notools_current_turn"] = anth(history_turns(False), system="S")
CASES["think_notools_next_turn"] = anth(history_turns(False) + [
    {"role": "assistant", "content": [think("r3"), text("done")]},
    {"role": "user", "content": "q3"}], system="S")
CASES["think_tools_next_turn"] = anth(history_turns(True) + [
    {"role": "assistant", "content": [think("r3"), text("done")]},
    {"role": "user", "content": "q3"}], system="S", tools=TOOLS)
CASES["tools_no_system"] = anth([{"role": "user", "content": "List files"}], tools=TOOLS)
CASES["tools_nothink"] = anth([{"role": "user", "content": "List files"}], system="S", tools=TOOLS, thinking=False)
CASES["tool_choice_none"] = anth([{"role": "user", "content": "Hi"}], system="S", tools=TOOLS,
                                 tool_choice={"type": "none"})
CASES["tool_args_json"] = anth([
    {"role": "user", "content": "Run it"},
    {"role": "assistant", "content": [think("Run."), use("x", "bash", {
        "command": "echo hi", "timeout": 2.50, "flags": ["-a", "-b"], "env": {"A": "1", "B": [1, 2.0, True, None]},
        "count": 3, "ratio": 1e-7, "big": 1e20, "neg": -0.0, "s": "café \"q\" \\ / \n"})]},
    {"role": "user", "content": [result("x", "hi")]}], tools=TOOLS)
CASES["unicode_text"] = anth([{"role": "user", "content": UNICODE}], system="Unicode — system “s”.")
CASES["mid_system"] = anth([
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": [think("r1"), text("a1")]},
    {"role": "system", "content": "Be terse now."}], system="S")
CASES["oai_plain"] = oai([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
CASES["oai_high"] = oai([{"role": "user", "content": "Hi"}], effort="high")
CASES["oai_content_parts"] = oai([{"role": "user", "content": [text("One."), text("Two."), oimg()]}])
CASES["oai_tools_history"] = oai([
    {"role": "system", "content": "S"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "", "reasoning_content": "r1", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\",\"flags\":[\"-l\"]}"}},
        {"id": "c2", "type": "function", "function": {"name": "read_image", "arguments": "{\"path\":\"a.png\"}"}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "second"},
    {"role": "tool", "tool_call_id": "c1", "content": [text("first"), text("part two")]}], tools=TOOLS)
CASES["oai_notools_history"] = oai([
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "", "reasoning_content": "r2", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "value"}])
CASES["oai_unicode"] = oai([{"role": "user", "content": UNICODE}])


def ds4_ids(executable, api, body, image_tokens):
    import os
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as fp:
        json.dump(body, fp, ensure_ascii=False)
        fp.flush()
        env = dict(os.environ, DS4_RENDER_IMAGE_TOKENS=str(image_tokens))
        result = subprocess.run([executable, "--ds41-request-tokens", api, fp.name],
                                check=True, capture_output=True, text=True, env=env)
    return json.loads(result.stdout.splitlines()[0])


def main():
    global ENCODE, TOKENIZER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="released encoding/encoding.py")
    parser.add_argument("tokenizer", help="released tokenizer.json")
    parser.add_argument("--request", help="also check this Messages request body")
    parser.add_argument("--image-tokens", type=int, default=1)
    parser.add_argument("--executable", default="./ds4_test")
    args = parser.parse_args()
    namespace = {"__name__": "deepseek41_reference"}
    exec(compile(Path(args.reference).read_text(), args.reference, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    ENCODE = namespace["encode_messages"]
    TOKENIZER = Tokenizer.from_file(args.tokenizer)
    items = [(name, api, body, 1) for name, (api, body) in CASES.items()]
    if args.request:
        items.append((args.request, "anthropic", json.loads(Path(args.request).read_text()),
                      args.image_tokens))
    for name, api, body, image_tokens in items:
        prompt, _ = render(body, api)
        expected = tokenize(prompt, image_tokens)
        actual = ds4_ids(args.executable, api, body, image_tokens)
        if actual != expected:
            i = next((k for k, (a, b) in enumerate(zip(actual, expected)) if a != b),
                     min(len(actual), len(expected)))
            raise AssertionError(f"{name}: token {i}: ds4 {TOKENIZER.decode(actual[max(0, i - 4):i + 6])!r}, "
                                 f"reference {TOKENIZER.decode(expected[max(0, i - 4):i + 6])!r}")
    print(f"V4.1 requests: {len(items)} exact prompt token matches")


if __name__ == "__main__":
    main()

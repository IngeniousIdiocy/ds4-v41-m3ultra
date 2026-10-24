#!/usr/bin/env python3
"""Compare ds4's DeepSeek V4/V4.1 tokenizer with the released tokenizer.json.

The corpus: the repository's own text and source files, every codepoint in a
few contexts, and random strings mixing ASCII, punctuation, digits, Unicode
whitespace, CJK, emoji and combining marks. Spellings of the tokenizer's added
tokens are broken up: the reference splits every one of them out of text,
while ds4 recognizes only its chat markers.

Usage: DS4_TEST_MODEL=V4.1.gguf tests/test_deepseek_tokenizer.py tokenizer.json
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile

from tokenizers import Tokenizer

ADDED = ["<｜", "｜>", "<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:", "<|place_holder"]
# Other models' special spellings are plain text to DeepSeek.
FOREIGN = "[gMASK]<sop><|system|>Hi<|user|>q<|assistant|>a<|observation|>o<|im_start|>x<|im_end|>"
POOLS = {
    "ascii": [chr(c) for c in range(32, 127)],
    "space": [" ", " ", "\n", "\n\n", "\t", "\r\n", "\r", " ", "　", " ", "\u000b",
              "\u000c", "\u0085", " ", " ", "\u001c", "​"],
    "punct": list("“”‘’«»…—–·•、。，．：；！？（）【】《》〈〉「」『』〜・゠£€¥₹©®™°±×÷→⇒∞∑∫≈≠§¶†‡‰′″"),
    "digits": list("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳०१२३²³¹½¼¾ⅠⅡⅢⅫ①②⑩〇"),
    "cjk": list("中文测试日本語漢字のテキストカタカナひらがな한국어텍스트龥一丁"),
    "letters": list("abcxyzABCXYZéèàüößçñøåÆŒДжЯΩαβγאבגابت") + ["é", "ä"],
    "marks": ["́", "̈", "⃝", "ः", "ि", "ั", "゙", "゚"],
    "emoji": ["😀", "👍🏽", "👩‍💻", "🇺🇸", "❤️", "✅", "🔥", "\U0001F9EA"],
    "code": ["def ", "return ", "    ", "{", "}", "();", "=>", "!=", "&&", "||", "/*", "*/",
             "#include", "\\n", "'s", "n't", "http://", ".com", "@user", "$x", "->", "::", "..."],
}


def random_text(rng):
    out = []
    for _ in range(rng.randint(1, 60)):
        kind = rng.choice(list(POOLS) + ["any", "word"])
        if kind == "any":
            cp = rng.randint(0x20, 0x10FFFF)
            out.append(chr(0x41 if 0xD800 <= cp <= 0xDFFF else cp))
        elif kind == "word":
            out.append(" " + "".join(rng.choice(POOLS["letters"]) for _ in range(rng.randint(1, 8))))
        else:
            out.append(rng.choice(POOLS[kind]) * rng.choice([1, 1, 1, 2, 3, 5]))
    return "".join(out)


def corpus(root, count, seed):
    texts = []
    for pattern in ["*.c", "*.h", "*.md", "tests/*.py", "tests/*.txt", "speed-bench/*"]:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.stat().st_size < 4 << 20:
                try:
                    texts.append(path.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    pass
    cps = [cp for cp in range(0x20, 0x110000) if not 0xD800 <= cp <= 0xDFFF]
    for i in range(0, len(cps), 512):
        texts.append("".join(f" {chr(c)}a{chr(c)}{chr(c)}1 .{chr(c)}\n" for c in cps[i:i + 512]))
    rng = random.Random(seed)
    texts += [random_text(rng) for _ in range(count)]
    texts += [FOREIGN, "text " + FOREIGN + " more"]
    for spelling in ADDED:
        texts = [t.replace(spelling, spelling[0] + "⁠" + spelling[1:]) for t in texts]
    return [t.replace("\x00", "") for t in texts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", help="released tokenizer.json")
    parser.add_argument("--count", type=int, default=20000, help="random strings")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--executable", default="./ds4_test")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    texts = corpus(Path(__file__).resolve().parent.parent, args.count, args.seed)
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as fp:
        json.dump(texts, fp, ensure_ascii=False)
        fp.flush()
        result = subprocess.run([args.executable, "--tokenize-json", fp.name],
                                check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(texts), (len(lines), len(texts))
    tokens = 0
    for text, line in zip(texts, lines):
        expected = tokenizer.encode(text, add_special_tokens=False).ids
        actual = json.loads(line)
        tokens += len(expected)
        if actual != expected:
            i = next((k for k, (a, b) in enumerate(zip(actual, expected)) if a != b),
                     min(len(actual), len(expected)))
            raise AssertionError(f"token {i}: ds4 {actual[i:i + 6]} {tokenizer.decode(actual[i:i + 6])!r}, "
                                 f"reference {expected[i:i + 6]} {tokenizer.decode(expected[i:i + 6])!r}")
    print(f"DeepSeek tokenizer: {len(texts)} texts, {tokens} tokens, exact")


if __name__ == "__main__":
    main()

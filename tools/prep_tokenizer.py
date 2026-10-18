"""
Writes golden tokenizer test cases for tokenizer_test: UTF-8 text and the token ids
tiktoken's GPT-2 encoding produces for it.

Run from the repo root: python tools/prep_tokenizer.py
"""

import os
import struct
import sys

import tiktoken

ENC = tiktoken.get_encoding("gpt2")

CASES_FILE = "./data/tokenizer_cases.bin"
SHAKESPEARE_FILE = "./data/tiny_shakespeare.txt"

# Code points per sweep case, so a failure points at a small range.
SWEEP_CHUNK = 1024

EDGE_CASES = [
    "",
    "Hello",
    "Hello world",
    " Hello",
    "Hello ",
    "Hello  world",
    "Hello   world",
    "Hello, world! It's 2026.",
    # Whitespace runs: the last space of a run is left for the next word.
    " ",
    "  ",
    "\n",
    "\n\nHello",
    "a\n\n\nb",
    "\t\tx",
    " \n x",
    "   \n\n  ",
    "line1\r\nline2\r\n",
    "trailing spaces   ",
    "word\n",
    "a  b",
    "a b",
    "a　b c\u0085de",
    "zero​width﻿bom",
    # Contractions are lowercase only and use the ASCII apostrophe.
    "it's I'll we've you're he'd I'm don't",
    "IT'S I'LL WE'VE",
    "'s",
    "''s",
    "' s",
    " 's",
    "?'s",
    "rock'n'roll",
    "it’s",
    # Numbers, punctuation, symbols.
    "123 12345678 3.14159 1,000,000 42x x42",
    "²³ ½ Ⅻ ٣٤٥",
    "!!! ... ?! (a) a--b --> @user #tag $100 50%",
    "<|endoftext|>",
    "def foo(x):\n    return x * 2\n",
    # Non-Latin scripts, combining marks, emoji.
    "zażółć gęślą jaźń",
    "Grüße naïve café",
    "é ñ",
    "日本語のテキスト",
    "한국어",
    "Привет мир",
    "مرحبا עברית",
    "हिन्दी",
    "\U0001f44b\U0001f30d \U0001f468‍\U0001f469‍\U0001f467 \U0001f1f5\U0001f1f1",
    # Long pieces stress the merge loop.
    "a" * 1000,
    " " * 300 + "x",
    "\n" * 64,
    "-" * 200,
]


def sweep_text(lo, hi):
    # Put each code point between letters, digits, punctuation and spaces, so its
    # pre-tokenizer class (letter / number / whitespace / other) shows in the ids.
    parts = []

    for cp in range(lo, hi):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        parts.append(f"a{c}a 1{c}1 .{c}. {c}\n{c}  {c}")

    return " ".join(parts)


def build_cases():
    cases = [(f"edge {i}: {text[:24]!a}", text) for i, text in enumerate(EDGE_CASES)]

    # Planes 0-3 hold every assigned character except private use; plane 14 has tags and variation selectors.
    for lo, hi in [(0x0000, 0x40000), (0xE0000, 0xE1000)]:
        for start in range(lo, hi, SWEEP_CHUNK):
            end = min(start + SWEEP_CHUNK, hi)
            cases.append((f"sweep U+{start:05X}..U+{end - 1:05X}", sweep_text(start, end)))

    if os.path.exists(SHAKESPEARE_FILE):
        with open(SHAKESPEARE_FILE, "r", encoding="utf-8") as f:
            cases.append(("tiny_shakespeare.txt", f.read()))
    else:
        print(f"note: {SHAKESPEARE_FILE} not found, skipping the corpus case", file=sys.stderr)

    return cases


def main():
    cases = build_cases()

    # Layout (little-endian): u32 case count, then per case:
    # u32 name length, name, u32 text length, UTF-8 text, u32 token count, u32 tokens.
    with open(CASES_FILE, "wb") as f:
        f.write(struct.pack("<I", len(cases)))

        for name, text in cases:
            name_bytes = name.encode("utf-8", errors="backslashreplace")
            text_bytes = text.encode("utf-8")
            tokens = ENC.encode_ordinary(text)

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(text_bytes)))
            f.write(text_bytes)
            f.write(struct.pack("<I", len(tokens)))
            f.write(struct.pack(f"<{len(tokens)}I", *tokens))

    print(f"wrote {len(cases)} tokenizer cases -> {CASES_FILE}")


if __name__ == "__main__":
    main()

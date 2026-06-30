#!/usr/bin/env python3
"""
GPT-2 BPE tokenizer bridge for the GPT model.

Requires: pip install tiktoken
"""

import argparse
import numpy as np
import tiktoken


# GPT-2's byte-level BPE. vocab_size == 50257, matching gpt_config.vocab_size.
ENC = tiktoken.get_encoding("gpt2")


def cmd_encode_file(args):
    with open(args.src, "r", encoding="utf-8") as f:
        text = f.read()

    ids = ENC.encode(text)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(args.dst)

    print(f"wrote {arr.size} tokens -> {args.dst}\n")


def cmd_decode_file(args):
    arr = np.fromfile(args.src, dtype=np.uint16)
    text = ENC.decode(arr.tolist())

    with open(args.dst, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"read {arr.size} tokens <- {args.src}\n")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 tokenizer bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc_file = sub.add_parser("encode-file", help="corpus file -> uint16 .bin")
    enc_file.add_argument("src", help="UTF-8 text corpus")
    enc_file.add_argument("dst", help="output .bin of little-endian uint16 token ids")
    enc_file.set_defaults(func=cmd_encode_file)

    dec_file = sub.add_parser("decode-file", help="uint16 .bin -> corpus file")
    dec_file.add_argument("src", help="input .bin of little-endian uint16 token ids")
    dec_file.add_argument("dst", help="output UTF-8 text corpus")
    dec_file.set_defaults(func=cmd_decode_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

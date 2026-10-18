#!/usr/bin/env python3
"""
Download OpenAI's GPT-2 (124M) weights and convert them to the GPT model's checkpoint format.

The weights come from Hugging Face (openai-community/gpt2) as safetensors, which is a JSON
header followed by raw tensor data, so it is parsed directly without torch or transformers.

Requires: pip install numpy requests
"""

import argparse
import json
import os

import numpy as np
import requests


URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
CACHE_FILE = "data/openai_gpt2.safetensors"

# GPT-2 124M. gpt_config must use the same values (plus max_seq_len) to load the checkpoint.
VOCAB_SIZE = 50257
VOCAB_SIZE_PADDED = 50304
NUM_LAYERS = 12
D_MODEL = 768
D_FFN = 3072
N_CTX = 1024  # Number of positions OpenAI trained with


def download(url, path):
    if os.path.exists(path):
        print(f"using cached {path}")
        return

    # Downloads to a temporary name first, so an interrupted download is not mistaken for the cache.
    tmp_path = path + ".part"

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\rdownloading {url}: {done >> 20}/{total >> 20} MB", end="", flush=True)

    print()
    os.replace(tmp_path, path)


def load_safetensors(path):
    # Layout: u64 header size, JSON header {name: {dtype, shape, data_offsets}}, then the raw data,
    # with data_offsets relative to the end of the header.
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))

    data = np.fromfile(path, dtype=np.uint8, offset=8 + header_size)
    tensors = {}

    for name, info in header.items():
        if name == "__metadata__":
            continue

        assert info["dtype"] == "F32", f"{name}: expected F32, got {info['dtype']}"
        begin, end = info["data_offsets"]
        tensors[name.removeprefix("transformer.")] = data[begin:end].view(np.float32).reshape(info["shape"])

    return tensors


def convert(hf, max_seq_len):
    """The 16 weight tensors, in the exact order gpt_weights packs its buffer (see src/model/gpt.cpp).
    GPT-2 uses Conv1D layers, whose weights are already stored [in, out] as the model expects, and
    its c_attn output is already the q, k, v concatenation that qkv_proj_w holds."""

    L, S, Vp, V, D, F = NUM_LAYERS, max_seq_len, VOCAB_SIZE_PADDED, VOCAB_SIZE, D_MODEL, D_FFN

    def per_layer(name):
        return np.stack([hf[f"h.{l}.{name}"] for l in range(L)])

    # The padded vocab rows are never sampled, and zero rows keep their logits at 0.
    wte = np.zeros((Vp, D), dtype=np.float32)
    wte[:V] = hf["wte.weight"]

    tensors = [
        ("wte",         wte,                              (Vp, D)),
        ("wpe",         hf["wpe.weight"][:S],             (S, D)),
        ("ln_1_w",      per_layer("ln_1.weight"),         (L, D)),
        ("ln_1_b",      per_layer("ln_1.bias"),           (L, D)),
        ("qkv_proj_w",  per_layer("attn.c_attn.weight"),  (L, D, 3 * D)),
        ("qkv_proj_b",  per_layer("attn.c_attn.bias"),    (L, 3 * D)),
        ("attn_proj_w", per_layer("attn.c_proj.weight"),  (L, D, D)),
        ("attn_proj_b", per_layer("attn.c_proj.bias"),    (L, D)),
        ("ln_2_w",      per_layer("ln_2.weight"),         (L, D)),
        ("ln_2_b",      per_layer("ln_2.bias"),           (L, D)),
        ("ffn_up_w",    per_layer("mlp.c_fc.weight"),     (L, D, F)),
        ("ffn_up_b",    per_layer("mlp.c_fc.bias"),       (L, F)),
        ("ffn_down_w",  per_layer("mlp.c_proj.weight"),   (L, F, D)),
        ("ffn_down_b",  per_layer("mlp.c_proj.bias"),     (L, D)),
        ("ln_f_w",      hf["ln_f.weight"],                (D,)),
        ("ln_f_b",      hf["ln_f.bias"],                  (D,)),
    ]

    for name, tensor, shape in tensors:
        assert tensor.shape == shape, f"{name}: expected {shape}, got {tensor.shape}"

    return [(name, tensor) for name, tensor, _ in tensors]


def main():
    parser = argparse.ArgumentParser(description="Download and convert OpenAI's GPT-2 (124M) weights")
    parser.add_argument("--output", default="data/openai_gpt2.bin", help="output checkpoint .bin")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help=f"positions to keep (at most {N_CTX}); must match gpt_config.max_seq_len, which is 256 in gpt_infer")
    args = parser.parse_args()

    if not 0 < args.max_seq_len <= N_CTX:
        parser.error(f"--max-seq-len must be in [1, {N_CTX}]")

    download(URL, CACHE_FILE)
    tensors = convert(load_safetensors(CACHE_FILE), args.max_seq_len)

    count = 0

    with open(args.output, "wb") as f:
        for _, tensor in tensors:
            np.ascontiguousarray(tensor, dtype=np.float32).tofile(f)
            count += tensor.size

    print(f"wrote {count} weights ({count * 4 / 2**20:.2f} MB, max_seq_len={args.max_seq_len}) -> {args.output}\n")


if __name__ == "__main__":
    main()

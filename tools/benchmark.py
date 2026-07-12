import argparse
import array
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt_reference import GPTConfig, GPTReference

# Match the C++ backend: fp32 storage, TF32 matmuls.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REAL_CFG = GPTConfig(
    max_seq_len=256,
    vocab_size=50257,
    vocab_size_padded=50304,
    num_layers=12,
    num_heads=12,
    d_model=768,
    d_ffn=3072,
)


def bench_training(model, cfg, batch, seq, steps, warmup, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)
    tokens = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device=device)
    inp, lbl = tokens[:, :seq], tokens[:, 1: seq + 1]

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), lbl.reshape(-1))
        loss.backward()

    for _ in range(warmup):
        step()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for _ in range(steps):
        step()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    per_step = (time.perf_counter() - t0) / steps

    return batch * seq / per_step, per_step * 1000.0


def load_prompt(path, device):
    tokens = array.array("H")

    with open(path, "rb") as f:
        tokens.frombytes(f.read())

    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def bench_inference(model, cfg, prompt, gen_tokens, warmup, device):
    model.eval()

    def generate():
        seq = prompt.clone()
        first_token_ms = None

        for i in range(gen_tokens):
            if device.type == "cuda":
                torch.cuda.synchronize()

            t = time.perf_counter()
            logits = model(seq[:, -cfg.max_seq_len:])  # no KV cache
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)

            if device.type == "cuda":
                torch.cuda.synchronize()

            if i == 0:
                first_token_ms = (time.perf_counter() - t) * 1000.0

        return first_token_ms

    for _ in range(warmup):
        generate()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    ttft = generate()
    total = time.perf_counter() - t0

    return gen_tokens / total, ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=REAL_CFG.max_seq_len)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gen-tokens", type=int, default=40)
    ap.add_argument("--prompt", type=Path, default=Path(__file__).resolve().parent.parent / "res" / "prompt.bin")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("WARNING: CUDA not available; timings on CPU are not comparable.")

    model = GPTReference(REAL_CFG, dtype=torch.float32).to(device)

    if args.compile:
        model = torch.compile(model)

    label = "PyTorch eager" + (" + compile" if args.compile else "")
    print(f"{label}  |  device={device}  |  TF32 matmul=on")
    print(f"config: 124M, batch={args.batch_size}, seq={args.seq_len}")

    tok_s, ms = bench_training(model, REAL_CFG, args.batch_size, args.seq_len, args.steps, args.warmup, device)
    print(f"training : {tok_s:>10.0f} tok/s   ({ms:.2f} ms/step, fwd+bwd, no AdamW)")

    prompt = load_prompt(args.prompt, device)
    inf_tok_s, ttft = bench_inference(model, REAL_CFG, prompt, gen_tokens=args.gen_tokens, warmup=2, device=device)
    print(f"inference: {inf_tok_s:>10.1f} tok/s   (prompt {prompt.size(1)} tokens from {args.prompt.name}, +{args.gen_tokens} generated, TTFT {ttft:.2f} ms, no KV cache)")


if __name__ == "__main__":
    main()

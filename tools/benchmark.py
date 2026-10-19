"""
PyTorch baseline for one training step: the same 124M model, batch and precision as gpt_train
(fp32 weights, TF32 matmuls). Compare the full step with `gpt_train --batch-accum-steps 1`,
which also runs AdamW after every batch.

Run from the repo root: python tools/benchmark.py [--compile]
"""

import argparse
import statistics
import time

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


def bench_training(model, cfg, batch, seq, steps, warmup):
    """Median ms of forward + loss + backward, and of the full step including AdamW."""

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    tokens = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device="cuda")
    inp, lbl = tokens[:, :seq], tokens[:, 1:]

    fwd_bwd_ms = []
    step_ms = []

    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), lbl.reshape(-1))
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        opt.step()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if i >= warmup:
            fwd_bwd_ms.append((t1 - t0) * 1000.0)
            step_ms.append((t2 - t0) * 1000.0)

    return statistics.median(fwd_bwd_ms), statistics.median(step_ms)


def main():
    ap = argparse.ArgumentParser(description="PyTorch training step baseline")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=REAL_CFG.max_seq_len)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (needs triton, so Python 3.13 or older)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, the timings would not be comparable.")

    model = GPTReference(REAL_CFG).cuda()

    if args.compile:
        model = torch.compile(model)

    label = f"PyTorch {torch.__version__} " + ("+ torch.compile" if args.compile else "eager")
    tokens = args.batch_size * args.seq_len
    fwd_bwd, step = bench_training(model, REAL_CFG, args.batch_size, args.seq_len, args.steps, args.warmup)

    print(f"{label}  |  124M, batch {args.batch_size} x seq {args.seq_len}, fp32 + TF32 matmuls  |  median of {args.steps} steps")
    print(f"forward + backward : {fwd_bwd:7.2f} ms  ({tokens * 1000 / fwd_bwd:.0f} tokens/s)")
    print(f"full step (+AdamW) : {step:7.2f} ms  ({tokens * 1000 / step:.0f} tokens/s)")


if __name__ == "__main__":
    main()

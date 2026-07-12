import os

import numpy as np
import torch

from gpt_reference import GPTConfig, GPTReference

# Single source of truth for the test config; gpt_test.cpp must use the same.
CFG = GPTConfig(
    max_seq_len=64,
    vocab_size=507,
    vocab_size_padded=512,
    num_layers=4,
    num_heads=4,
    d_model=32,
    d_ffn=256,
)

SEED = 1234

TOKENS_FILE = "./data/gpt_dummy_tokens.bin"
WEIGHTS_FILE = "./data/gpt_dummy_weights.bin"
GRADS_FILE = "./data/gpt_dummy_ref_gradients.bin"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokens = np.random.randint(0, CFG.vocab_size, size=CFG.max_seq_len + 1, dtype=np.int32)
    model = GPTReference(CFG)
    weights = model.weights_as_flat().astype(np.float32)

    inp = torch.from_numpy(tokens[: CFG.max_seq_len].astype(np.int64)).view(1, -1)
    lbl = torch.from_numpy(tokens[1: CFG.max_seq_len + 1].astype(np.int64)).view(1, -1)

    logits = model(inp)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, CFG.vocab_size), lbl.view(-1))
    model.zero_grad(set_to_none=False)
    loss.backward()

    grads = model.grads_as_flat().astype(np.float32)

    os.makedirs("./data", exist_ok=True)
    tokens.tofile(TOKENS_FILE)
    weights.tofile(WEIGHTS_FILE)
    grads.tofile(GRADS_FILE)

    print(f"seed: {SEED}")
    print(f"reference loss: {loss.item():.6f}")
    print(f"wrote {tokens.size} tokens -> {TOKENS_FILE}")
    print(f"wrote {weights.size} weights -> {WEIGHTS_FILE}")
    print(f"wrote {grads.size} gradients -> {GRADS_FILE}")

    return 0


if __name__ == "__main__":
    main()

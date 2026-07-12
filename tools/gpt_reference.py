from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    max_seq_len: int
    vocab_size: int
    vocab_size_padded: int
    num_layers: int
    num_heads: int
    d_model: int
    d_ffn: int


def tensor_shapes(c: GPTConfig):
    """The 16 weight tensors, in the exact order gpt_weights packs its buffer
    (see src/model/gpt.cpp). Matches both the checkpoint and gradient dumps."""

    L, S, Vp, D, Fd = (c.num_layers, c.max_seq_len, c.vocab_size_padded, c.d_model, c.d_ffn)

    return [
        ("wte",         (Vp, D)),
        ("wpe",         (S, D)),
        ("ln_1_w",      (L, D)),
        ("ln_1_b",      (L, D)),
        ("qkv_proj_w",  (L, D, 3 * D)),
        ("qkv_proj_b",  (L, 3 * D)),
        ("attn_proj_w", (L, D, D)),
        ("attn_proj_b", (L, D)),
        ("ln_2_w",      (L, D)),
        ("ln_2_b",      (L, D)),
        ("ffn_up_w",    (L, D, Fd)),
        ("ffn_up_b",    (L, Fd)),
        ("ffn_down_w",  (L, Fd, D)),
        ("ffn_down_b",  (L, D)),
        ("ln_f_w",      (D,)),
        ("ln_f_b",      (D,)),
    ]


class GPTReference(nn.Module):
    def __init__(self, c: GPTConfig, dtype=torch.float32):
        super().__init__()

        self.c = c

        for name, shape in tensor_shapes(c):
            param = nn.Parameter(torch.empty(*shape, dtype=dtype))

            # Init procedure from src/model/gpt.cpp
            if name.endswith("_w") and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif name in ("wte", "wpe"):
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif name.endswith("ln_1_w") or name.endswith("ln_2_w") or name == "ln_f_w":
                nn.init.ones_(param)
            else:
                nn.init.zeros_(param)

            setattr(self, name, param)

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.c.d_model,), w, b, eps=1e-5)

    def forward(self, tokens):
        c = self.c
        B, T = tokens.shape
        H, D, dH = c.num_heads, c.d_model, c.d_model // c.num_heads

        pos = torch.arange(T, device=tokens.device)
        x = self.wte[tokens] + self.wpe[pos]                      # [B, T, D]

        for l in range(c.num_layers):
            # Attention sub-block (pre-LN + residual)
            h = self._ln(x, self.ln_1_w[l], self.ln_1_b[l])
            qkv = h @ self.qkv_proj_w[l] + self.qkv_proj_b[l]     # [B, T, 3D]
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(B, T, H, dH).transpose(1, 2)               # [B, H, T, dH]
            k = k.view(B, T, H, dH).transpose(1, 2)
            v = v.view(B, T, H, dH).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, D)
            a = a @ self.attn_proj_w[l] + self.attn_proj_b[l]
            x = x + a

            # Feed-forward sub-block (pre-LN + residual)
            h2 = self._ln(x, self.ln_2_w[l], self.ln_2_b[l])
            u = h2 @ self.ffn_up_w[l] + self.ffn_up_b[l]
            u = F.gelu(u, approximate="tanh")
            d = u @ self.ffn_down_w[l] + self.ffn_down_b[l]
            x = x + d

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        logits = x @ self.wte[: c.vocab_size].t()                # [B, T, vocab_size]

        return logits

    def weights_as_flat(self) -> np.ndarray:
        """Concatenate every parameter in the C++ packing order."""

        parts = []

        for name, _ in tensor_shapes(self.c):
            p = getattr(self, name)
            parts.append(p.reshape(-1).detach().cpu().numpy())

        return np.concatenate(parts)

    def grads_as_flat(self) -> np.ndarray:
        """Concatenate every parameter gradient in the C++ packing order."""

        parts = []

        for name, _ in tensor_shapes(self.c):
            g = getattr(self, name).grad
            parts.append(g.reshape(-1).detach().cpu().numpy())

        return np.concatenate(parts)

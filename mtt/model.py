"""Small decoder-only transformer (GPT-style, pre-LayerNorm, tied embeddings).

Defaults (6 layers, d_model 256, 4 heads, vocab ~2100, block 128) give ~5.3M parameters.
Right-padded batches are safe without an attention mask: attention is causal, so real
tokens never look at the padding that follows them, and padded positions are excluded
from the loss.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (identical for every tokenisation mode)."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's fused SDPA."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    """Transformer block: LN -> attention -> LN -> MLP, with residual connections."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only language model over the combined MTT vocabulary."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2 style scaled residual init
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Number of distinct trainable parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """Return logits ``(B, T, V)``, or ``(B, 1, V)`` for the last position if ``last_only``."""
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:, :]
        return self.head(x)

    def compute_loss(self, idx: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Mean next-token cross-entropy over positions where ``loss_mask`` is True.

        ``loss_mask[b, t]`` says whether token ``t`` is a prediction target (answer tokens).
        """
        logits = self(idx[:, :-1])
        targets = idx[:, 1:]
        mask = loss_mask[:, 1:].float()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        return (ce * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def greedy_generate(
    model: TinyGPT,
    prompts: list[list[int]],
    eos_id: int,
    max_new_tokens: int = 48,
    batch_size: int = 128,
) -> list[list[int]]:
    """Greedy decoding. Returns, for each prompt, the generated ids up to and including ``eos``.

    Prompts are grouped by length so no padding is needed (no KV cache: simple and
    adequate for ~1-2k short test prompts).
    """
    model.eval()
    device = next(model.parameters()).device
    results: list[list[int]] = [[] for _ in prompts]
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)

    for members in by_len.values():
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            idx = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            steps: list[torch.Tensor] = []
            for _ in range(max_new_tokens):
                if idx.size(1) >= model.cfg.block_size:
                    break
                nxt = model(idx, last_only=True)[:, -1].argmax(dim=-1)
                steps.append(nxt)
                idx = torch.cat([idx, nxt[:, None]], dim=1)
                done |= nxt == eos_id
                if bool(done.all()):
                    break
            rows = torch.stack(steps, dim=1).tolist() if steps else [[] for _ in chunk]
            for row, i in zip(rows, chunk):
                results[i] = row[: row.index(eos_id) + 1] if eos_id in row else row
    return results

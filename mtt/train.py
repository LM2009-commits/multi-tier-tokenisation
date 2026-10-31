"""Train the tiny transformer under one tokenisation mode.

    python -m mtt.train --mode wlt_only
    python -m mtt.train --mode clt_only
    python -m mtt.train --mode mtt
    python -m mtt.train --mode selfesc          # stretch goal
    python -m mtt.train --mode mtt --router learned   # stretch goal

Fairness: for a given ``--seed`` every mode uses the same examples, the same model
architecture and initialisation (the vocabulary is the full MTT vocabulary in every mode),
the same number of steps and the same batch order.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import generate_dataset
from .model import ModelConfig, TinyGPT
from .pipeline import (MODES, ROUTERS, build_router, build_tokenizer, encode_example,
                       training_sequence)
from .tokenizers.mtt import PAD


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny transformer under one tokenisation mode.")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--router", default="rule", choices=ROUTERS,
                   help="router used by --mode mtt (rule_words = rule router without letter spans)")
    p.add_argument("--wlt-corpus", default="train", choices=("train", "all"),
                   help="'all' also shows the BPE the held-out words (as bare vocabulary only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--n-train", type=int, default=3000, help="training examples per task")
    p.add_argument("--n-test", type=int, default=300, help="test examples per task")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--name", default=None, help="run directory name (default: derived from mode)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--smoke", action="store_true", help="tiny, fast configuration to test the pipeline")
    args = p.parse_args(argv)
    if args.smoke:
        args.steps, args.batch_size, args.warmup = 60, 16, 10
        args.n_train, args.n_test = 150, 20
        args.n_layer, args.n_head, args.d_model = 2, 2, 64
        args.eval_every = 30
        if args.out_dir == "runs":
            args.out_dir = "runs_smoke"
    return args


def run_name(args: argparse.Namespace) -> str:
    """Directory name of a run, e.g. ``mtt`` or ``mtt_learned``."""
    if args.name:
        return args.name
    return f"mtt_{args.router}" if args.mode == "mtt" and args.router != "rule" else args.mode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(seqs: list[list[int]], starts: list[int], device: torch.device
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad sequences; the loss mask is True on answer tokens (``start`` .. end)."""
    longest = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), longest), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.bool)
    for r, (s, st) in enumerate(zip(seqs, starts)):
        idx[r, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[r, st : len(s)] = True
    return idx.to(device), mask.to(device)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warm-up then cosine decay to 10% of the base learning rate."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def test_loss(model: TinyGPT, seqs: list[list[int]], starts: list[int], device: torch.device,
              limit: int = 256, batch_size: int = 64) -> float:
    """Teacher-forced answer-token loss on (up to ``limit``) held-out sequences."""
    model.eval()
    pick = np.linspace(0, len(seqs) - 1, min(limit, len(seqs))).astype(int)
    seqs, starts = [seqs[i] for i in pick], [starts[i] for i in pick]
    total, count = 0.0, 0
    for i in range(0, len(seqs), batch_size):
        idx, mask = make_batch(seqs[i : i + batch_size], starts[i : i + batch_size], device)
        n = int(mask[:, 1:].sum())
        total += float(model.compute_loss(idx, mask)) * n
        count += n
    model.train()
    return total / max(count, 1)


def train(args: argparse.Namespace) -> Path:
    """Run one training job and return the run directory."""
    set_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out_dir) / run_name(args)
    out.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.seed, args.n_train, args.n_test)
    tok = build_tokenizer(dataset, include_test_words=args.wlt_corpus == "all")
    router = build_router(args.router, dataset, args.seed) if args.mode == "mtt" else None

    def encode_split(examples):
        seqs, starts = [], []
        for ex in examples:
            s, st = training_sequence(*encode_example(ex, args.mode, tok, router))
            seqs.append(s)
            starts.append(st)
        return seqs, starts

    train_seqs, train_starts = encode_split(dataset.train)
    test_seqs, test_starts = encode_split(dataset.test)
    longest = max(len(s) for s in train_seqs + test_seqs)
    if longest > args.block_size:
        raise ValueError(f"longest sequence ({longest}) exceeds --block-size {args.block_size}")
    mean_len = float(np.mean([len(s) for s in train_seqs]))

    cfg = ModelConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model,
                      args.dropout)
    model = TinyGPT(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[{run_name(args)}] device={device} vocab={tok.vocab_size} params={n_params/1e6:.2f}M "
          f"train_seqs={len(train_seqs)} mean_len={mean_len:.1f} max_len={longest}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    rng = np.random.default_rng(args.seed)  # batch order is independent of the mode
    order, cursor = rng.permutation(len(train_seqs)), 0
    log: list[dict] = []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        if cursor + args.batch_size > len(order):
            order, cursor = rng.permutation(len(train_seqs)), 0
        batch = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        idx, mask = make_batch([train_seqs[i] for i in batch], [train_starts[i] for i in batch], device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        loss = model.compute_loss(idx, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        record: dict = {"step": step + 1, "train_loss": float(loss)}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            record["test_loss"] = test_loss(model, test_seqs, test_starts, device)
            print(f"  step {step + 1:>6}/{args.steps}  train {record['train_loss']:.4f}  "
                  f"test {record['test_loss']:.4f}  {time.time() - t0:.0f}s", flush=True)
        log.append(record)

    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict()}, out / "model.pt")
    tok.wlt.save(out / "wlt.json")
    config = {**vars(args), "run_name": run_name(args), "vocab_size": tok.vocab_size,
              "n_params": n_params, "mean_train_len": mean_len, "max_len": longest,
              "wlt_vocab_size": tok.wlt.vocab_size, "model_config": cfg.to_dict(),
              "train_seconds": time.time() - t0}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "train_log.json").write_text(json.dumps(log))
    print(f"[{run_name(args)}] saved to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()

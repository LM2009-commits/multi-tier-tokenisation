"""Train the tiny transformer under one tokenisation mode.

    python -m mtt.train --mode wlt_only
    python -m mtt.train --mode clt_only
    python -m mtt.train --mode mtt
    python -m mtt.train --mode selfesc          # stretch goal
    python -m mtt.train --mode mtt --router learned   # stretch goal
    python -m mtt.train --mode mtt --data-version 2 --seed 1 --resume

Run directories are named after every setting that changes the result, so runs never collide:
``mtt``, ``mtt_v2``, ``mtt_v2_s1``, ``clt_only_v2_10k``, ...

Fairness: for a given ``--seed`` every mode uses the same examples, the same model
architecture and initialisation (the vocabulary is the full MTT vocabulary in every mode),
the same number of steps and the same batch order.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .data import TASKS, generate_dataset
from .model import ModelConfig, TinyGPT, greedy_generate
from .pipeline import (MODES, ROUTED_MODES, ROUTERS, build_router, build_tokenizer, encode_example,
                       extract_answer, generation_prompt, training_sequence)
from .tokenizers.mtt import EOS, PAD

DEFAULT_STEPS = 5000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny transformer under one tokenisation mode.")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--router", default="rule", choices=ROUTERS,
                   help="router used by routed modes (rule_words = rule router without letter spans)")
    p.add_argument("--wlt-corpus", default="train", choices=("train", "all"),
                   help="'all' also shows the BPE the held-out words (as bare vocabulary only)")
    p.add_argument("--data-version", type=int, default=1, choices=(1, 2),
                   help="1 = original experiment, 2 = balanced counting and held-out showcase words")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
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
    p.add_argument("--acc-every", type=int, default=1000,
                   help="log accuracy on a fixed test subset this often, for learning curves (0 = off)")
    p.add_argument("--ckpt-every", type=int, default=250, help="write a resumable checkpoint this often")
    p.add_argument("--resume", action="store_true", help="continue from the run's checkpoint if it has one")
    p.add_argument("--force", action="store_true", help="overwrite a finished run")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: PyTorch's choice)")
    p.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)  # tests interruption
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--name", default=None, help="run directory name (default: derived from the settings)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--smoke", action="store_true", help="tiny, fast configuration to test the pipeline")
    args = p.parse_args(argv)
    args.requested_steps = args.steps  # names a --smoke run after the run it stands in for
    if args.smoke:
        args.steps, args.batch_size, args.warmup = 60, 16, 10
        args.n_train, args.n_test = 150, 20
        args.n_layer, args.n_head, args.d_model = 2, 2, 64
        args.eval_every, args.acc_every, args.ckpt_every = 30, 30, 30
        if args.out_dir == "runs":
            args.out_dir = "runs_smoke"
    return args


def run_group(args: argparse.Namespace) -> str:
    """Name shared by a run's seeds: every setting that changes the result, except the seed."""
    if args.name:
        return args.name
    name = f"mtt_{args.router}" if args.mode == "mtt" and args.router != "rule" else args.mode
    if args.data_version != 1:
        name += f"_v{args.data_version}"
    if args.wlt_corpus != "train":
        name += f"_wlt{args.wlt_corpus}"
    if args.requested_steps != DEFAULT_STEPS:
        name += f"_{args.requested_steps / 1000:g}k"
    return name


def run_name(args: argparse.Namespace) -> str:
    """Directory name of a run, e.g. ``mtt``, ``mtt_learned_v2`` or ``mtt_v2_s1``."""
    return run_group(args) + (f"_s{args.seed}" if args.seed else "")


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


def provenance(threads: int) -> dict:
    """What produced a run: package versions, CPU and the code revision."""
    import tokenizers

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True, cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "tokenizers": tokenizers.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu": platform.processor(), "threads": threads,
            "git_rev": rev}


def save_checkpoint(path: Path, state: dict) -> None:
    """Write atomically, so an interruption can't leave a half-written checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


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


def subset_accuracy(model: TinyGPT, examples: list, prompts: list[list[int]], mode: str, tok) -> dict[str, float]:
    """Exact-match accuracy per task on a fixed subset. For learning curves, never for selection."""
    generated = greedy_generate(model, prompts, EOS, max_new_tokens=48)
    model.train()
    right, total = {t: 0 for t in TASKS}, {t: 0 for t in TASKS}
    for ex, gen in zip(examples, generated):
        total[ex.task] += 1
        right[ex.task] += extract_answer(gen, mode, tok) == ex.answer
    return {t: right[t] / total[t] for t in TASKS if total[t]}


def train(args: argparse.Namespace) -> Path:
    """Run one training job and return the run directory."""
    set_seed(args.seed)
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out_dir) / run_name(args)
    if (out / "model.pt").exists() and not args.force:
        raise SystemExit(f"{out} already holds a finished run; pass --force to overwrite it")
    out.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.seed, args.n_train, args.n_test, version=args.data_version)
    tok = build_tokenizer(dataset, include_test_words=args.wlt_corpus == "all")
    router = build_router(args.router, dataset, args.seed) if args.mode in ROUTED_MODES else None

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
          f"train_seqs={len(train_seqs)} mean_len={mean_len:.1f} max_len={longest}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    rng = np.random.default_rng(args.seed)  # batch order is independent of the mode
    order, cursor = rng.permutation(len(train_seqs)), 0
    log: list[dict] = []
    first_step, elapsed = 0, 0.0
    ckpt_path = out / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"])
        rng.bit_generator.state = ckpt["numpy_rng"]
        order, cursor, log = ckpt["order"], ckpt["cursor"], ckpt["log"]
        first_step, elapsed = ckpt["step"], ckpt["elapsed"]
        print(f"[{run_name(args)}] resuming at step {first_step}", flush=True)

    per_task = 40  # the fixed accuracy subset: the first prompts of each task
    acc_examples = [ex for t in TASKS for ex in [e for e in dataset.test if e.task == t][:per_task]]
    acc_prompts = [generation_prompt(encode_example(ex, args.mode, tok, router)[0]) for ex in acc_examples]

    t0 = time.time() - elapsed
    model.train()
    for step in range(first_step, args.steps):
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

        record: dict = {"step": step + 1, "train_loss": loss.item()}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            record["test_loss"] = test_loss(model, test_seqs, test_starts, device)
            print(f"  step {step + 1:>6}/{args.steps}  train {record['train_loss']:.4f}  "
                  f"test {record['test_loss']:.4f}  {time.time() - t0:.0f}s", flush=True)
        if args.acc_every and ((step + 1) % args.acc_every == 0 or step + 1 == args.steps):
            record["accuracy"] = subset_accuracy(model, acc_examples, acc_prompts, args.mode, tok)
        log.append(record)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.stop_after:
            save_checkpoint(ckpt_path, {
                "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step + 1,
                "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                "order": order, "cursor": cursor, "log": log, "elapsed": time.time() - t0})
        if step + 1 == args.stop_after:
            print(f"[{run_name(args)}] stopped early at step {step + 1}", flush=True)
            return out

    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict()}, out / "model.pt")
    tok.wlt.save(out / "wlt.json")
    config = {**vars(args), "run_name": run_name(args), "vocab_size": tok.vocab_size,
              "n_params": n_params, "mean_train_len": mean_len, "max_len": longest,
              "wlt_vocab_size": tok.wlt.vocab_size, "model_config": cfg.to_dict(),
              "train_seconds": time.time() - t0, "group": run_group(args),
              "provenance": provenance(torch.get_num_threads())}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8", newline="\n")
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8", newline="\n")
    ckpt_path.unlink(missing_ok=True)
    print(f"[{run_name(args)}] saved to {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()

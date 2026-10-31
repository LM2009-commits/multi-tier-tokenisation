"""Evaluate trained runs: exact-match accuracy, sequence lengths, table, plots.

    python -m mtt.eval --runs wlt_only clt_only mtt            # accuracy + lengths
    python -m mtt.eval --runs wlt_only clt_only mtt mtt_learned selfesc
    python -m mtt.eval --lengths-only                          # no model needed
    python -m mtt.eval --router-compare                        # rule vs learned router

Outputs go to ``--out-dir`` (default ``results/``): ``results.md``, ``results.json``,
``accuracy.png``, ``seqlen.png`` and one ``predictions_<run>.jsonl`` per run (prompt, gold,
prediction) for error analysis. Accuracy is exact match of the decoded answer string on
held-out words; the table also gives 95% Wilson confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import GENERAL_PROMPTS, LETTER_TASKS, OOD_ROUTER_CASES, TASKS, Dataset, generate_dataset
from .pipeline import (MODES, build_router, build_tokenizer, encode_example, extract_answer,
                       generation_prompt, sequence_lengths)
from .tokenizers.mtt import EOS, MTTTokenizer
from .tokenizers.wlt import WLT

PREFERRED_ORDER = ("wlt_only", "clt_only", "mtt", "mtt_rule_words", "mtt_learned", "selfesc")


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- loading
def load_run(run_dir: Path, device: str = "auto"):
    """Load config, dataset, tokenizer, router and model of a trained run."""
    import torch

    from .model import ModelConfig, TinyGPT
    from .train import pick_device

    config = json.loads((run_dir / "config.json").read_text())
    dataset = generate_dataset(config["seed"], config["n_train"], config["n_test"])
    tok = MTTTokenizer(WLT.load(run_dir / "wlt.json"))
    router = build_router(config["router"], dataset, config["seed"]) if config["mode"] == "mtt" else None
    dev = pick_device(device)
    ckpt = torch.load(run_dir / "model.pt", map_location=dev)
    model = TinyGPT(ModelConfig(**ckpt["model_config"])).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return config, dataset, tok, router, model


def evaluate_run(config: dict, dataset: Dataset, tok: MTTTokenizer, router, model) -> list[dict]:
    """Greedy-decode every test prompt and compare the extracted answer with the gold one."""
    from .model import greedy_generate

    mode = config["mode"]
    examples = dataset.test
    prompts = [generation_prompt(encode_example(ex, mode, tok, router)[0]) for ex in examples]
    generated = greedy_generate(model, prompts, EOS, max_new_tokens=48)
    records = []
    for ex, gen in zip(examples, generated):
        pred = extract_answer(gen, mode, tok)
        records.append({"task": ex.task, "prompt": ex.prompt, "gold": ex.answer, "pred": pred,
                        "correct": pred == ex.answer})
    return records


def summarise(records: list[dict]) -> dict[str, dict]:
    """Per-task accuracy with Wilson intervals, plus letter-task and overall means."""
    out: dict[str, dict] = {}
    for task in TASKS:
        rs = [r for r in records if r["task"] == task]
        k, n = sum(r["correct"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        out[task] = {"correct": k, "n": n, "acc": k / n if n else 0.0, "ci_low": lo, "ci_high": hi}
    for name, tasks in (("letter_mean", LETTER_TASKS), ("overall_mean", TASKS)):
        k = sum(out[t]["correct"] for t in tasks)  # pooled over tasks (equal n per task)
        n = sum(out[t]["n"] for t in tasks)
        lo, hi = wilson(k, n)
        out[name] = {"correct": k, "n": n, "acc": k / n if n else 0.0, "ci_low": lo, "ci_high": hi}
    return out


# --------------------------------------------------------------------------- reporting
def make_markdown(results: dict[str, dict]) -> str:
    """Markdown tables: accuracy, and mean sequence length (with ratio to wlt_only)."""
    runs = list(results)
    lines = ["## Exact-match accuracy on held-out words", ""]
    have_acc = [r for r in runs if results[r].get("accuracy")]
    if have_acc:
        lines.append("| task | " + " | ".join(have_acc) + " |")
        lines.append("|---|" + "---|" * len(have_acc))
        for task in (*TASKS, "letter_mean", "overall_mean"):
            cells = []
            for r in have_acc:
                a = results[r]["accuracy"][task]
                cell = f"{a['acc']:.3f}"
                if "ci_low" in a:
                    cell += f" ({a['ci_low']:.2f}-{a['ci_high']:.2f})"
                cells.append(cell)
            label = {"letter_mean": "**mean (letter tasks)**", "overall_mean": "**mean (all tasks)**"}.get(task, task)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        n = results[have_acc[0]]["accuracy"][TASKS[0]]["n"]
        lines += ["", f"Cells: accuracy (95% Wilson interval), n = {n} test prompts per task, one training seed."]
    lines += ["", "## Mean sequence length in tokens (`<bos> prompt <sep> answer <eos>`)", ""]
    lines.append("| task | " + " | ".join(runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    base = results.get("wlt_only", {}).get("lengths")
    for task in TASKS:
        cells = []
        for r in runs:
            v = results[r]["lengths"][task]["full"]
            ratio = f" ({v / base[task]['full']:.2f}x)" if base and r != "wlt_only" else ""
            cells.append(f"{v:.1f}{ratio}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_results(results: dict[str, dict], out_dir: Path) -> None:
    """Bar charts: accuracy per task (with CIs) and sequence length per task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(results)
    groups = [*TASKS, "letter_mean"]

    def grouped(ax, run_names, values, errs=None):
        width = 0.8 / max(len(run_names), 1)
        for i, r in enumerate(run_names):
            xs = [g + i * width - 0.4 + width / 2 for g in range(len(groups))]
            ax.bar(xs, values[r], width, label=r, yerr=None if errs is None else errs[r], capsize=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.legend()

    acc_runs = [r for r in runs if results[r].get("accuracy")]
    if acc_runs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        vals, errs = {}, {}
        for r in acc_runs:
            a = results[r]["accuracy"]
            vals[r] = [a[g]["acc"] for g in groups]
            errs[r] = [[max(0, a[g]["acc"] - a[g]["ci_low"]) if "ci_low" in a[g] else 0 for g in groups],
                       [max(0, a[g]["ci_high"] - a[g]["acc"]) if "ci_high" in a[g] else 0 for g in groups]]
        grouped(ax, acc_runs, vals, errs)
        ax.set_ylabel("exact-match accuracy (held-out words)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Accuracy by task and tokenisation mode")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    len_groups = TASKS
    width = 0.8 / max(len(runs), 1)
    for i, r in enumerate(runs):
        xs = [g + i * width - 0.4 + width / 2 for g in range(len(len_groups))]
        ax.bar(xs, [results[r]["lengths"][t]["full"] for t in len_groups], width, label=r)
    ax.set_xticks(range(len(len_groups)))
    ax.set_xticklabels(len_groups)
    ax.set_ylabel("mean tokens per example")
    ax.set_title("Sequence length cost by task and tokenisation mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "seqlen.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- router comparison
def compare_routers(dataset: Dataset, seed: int = 0) -> str:
    """Markdown report comparing the rule router with the learned router."""
    routers = {"rule": build_router("rule", dataset, seed),
               "rule_words": build_router("rule_words", dataset, seed),
               "learned": build_router("learned", dataset, seed)}
    pos = [ex for ex in dataset.test if ex.task in LETTER_TASKS]
    neg = [ex for ex in dataset.test if ex.task not in LETTER_TASKS]

    lines = ["## Router comparison", "",
             "| router | decision acc. (held-out) | target found (held-out) | spurious spans/positive "
             "(held-out) | decision acc. (OOD) | target found (OOD positives) | "
             "spurious spans/positive (OOD) |",
             "|---|---|---|---|---|---|---|"]
    misses: dict[str, list[str]] = {}
    for name, r in routers.items():
        dec = [r.route(ex.prompt).needs_clt for ex in dataset.test]
        gold = [ex.task in LETTER_TASKS for ex in dataset.test]
        dec_acc = sum(d == g for d, g in zip(dec, gold)) / len(gold)
        found, spurious = 0, 0
        for ex in pos:
            route = r.route(ex.prompt)
            texts = [ex.prompt[s:e] for s, e in route.spans]
            found += ex.target_words[0] in texts
            ok = {ex.target_words[0], ex.letter}
            spurious += sum(t not in ok for t in texts)
        ood_found, ood_spurious, ood_pos = 0, 0, 0
        ood_correct, missed = 0, []
        for prompt, target in OOD_ROUTER_CASES:
            route = r.route(prompt)
            ok_decision = route.needs_clt == (target is not None)
            ood_correct += ok_decision
            if not ok_decision:
                missed.append(prompt)
            if target is not None:
                ood_pos += 1
                texts = [prompt[s:e] for s, e in route.spans]
                ood_found += target in texts
                ood_spurious += sum(t != target and len(t) > 1 for t in texts)
        misses[name] = missed
        lines.append(f"| {name} | {dec_acc:.3f} | {found / len(pos):.3f} | {spurious / len(pos):.2f} | "
                     f"{ood_correct}/{len(OOD_ROUTER_CASES)} | {ood_found}/{ood_pos} | "
                     f"{ood_spurious / ood_pos:.2f} |")
    lines += ["", f"Held-out prompts: {len(pos)} letter-task and {len(neg)} category prompts "
              "(unseen words). `GENERAL_PROMPTS` are in the learned router's training data, so only "
              "the 16 hand-written paraphrases (OOD) count as unseen cases.", ""]
    for name, ms in misses.items():
        lines.append(f"**{name} OOD misses ({len(ms)}):** " + ("; ".join(f"“{m}”" for m in ms) or "none"))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def discover_runs(runs_dir: Path) -> list[str]:
    found = [p.name for p in runs_dir.iterdir() if (p / "model.pt").exists()] if runs_dir.exists() else []
    return sorted(found, key=lambda n: (PREFERRED_ORDER.index(n) if n in PREFERRED_ORDER else 99, n))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate MTT runs.")
    p.add_argument("--runs", nargs="*", default=None, help="run directory names inside --runs-dir")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--lengths-only", action="store_true", help="only compute sequence lengths (needs no model)")
    p.add_argument("--router-compare", action="store_true", help="compare rule and learned routers and exit")
    p.add_argument("--seed", type=int, default=0, help="(lengths-only / router-compare) dataset seed")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--wlt-corpus", default="train", choices=("train", "all"))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.router_compare:
        report = compare_routers(generate_dataset(args.seed, args.n_train, args.n_test), args.seed)
        (out_dir / "router_comparison.md").write_text(report)
        print(report)
        return

    results: dict[str, dict] = {}
    if args.lengths_only:
        dataset = generate_dataset(args.seed, args.n_train, args.n_test)
        tok = build_tokenizer(dataset, include_test_words=args.wlt_corpus == "all")
        for mode in MODES:
            router = build_router("rule") if mode == "mtt" else None
            results[mode] = {"lengths": sequence_lengths(dataset.test, mode, tok, router)}
    else:
        names = args.runs or discover_runs(Path(args.runs_dir))
        if not names:
            raise SystemExit(f"no trained runs found in {args.runs_dir!r}; run mtt.train first")
        for name in names:
            config, dataset, tok, router, model = load_run(Path(args.runs_dir) / name, args.device)
            print(f"evaluating {name} ({len(dataset.test)} test prompts)...", flush=True)
            records = evaluate_run(config, dataset, tok, router, model)
            (out_dir / f"predictions_{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n")
            results[name] = {"accuracy": summarise(records),
                             "lengths": sequence_lengths(dataset.test, config["mode"], tok, router),
                             "config": {k: config[k] for k in ("mode", "router", "seed", "steps", "n_params")}}

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    report = make_markdown(results)
    (out_dir / "results.md").write_text(report)
    plot_results(results, out_dir)
    print(report)


if __name__ == "__main__":
    main()

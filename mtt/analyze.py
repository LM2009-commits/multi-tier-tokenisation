"""Regenerate the report's tables and figures from committed files (no model or GPU needed).

    python -m mtt.analyze                    # reads results/, writes results/analysis.md and figures

Inputs are ``results.json``, ``predictions_*.jsonl``, ``train_logs/`` and (optional) ``bench.json``.
Baselines that need the training data regenerate the dataset, which is deterministic.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import TASKS, count_bucket, generate_dataset
from .eval import group_of, write_text


def load_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- statistics
def paired_word_bootstrap(a_by_seed: list[list[dict]], b_by_seed: list[list[dict]], n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float]:
    """Mean accuracy difference a - b with a 95% bootstrap interval, resampling words.

    Test prompts reuse words (about 5 prompts per word), so prompts aren't independent. The
    records of a and b must be the same prompts in the same order within each seed. Words are
    resampled within each seed; seeds stay fixed, so this measures test-set sampling only.
    Spread between seeds shows training variation.
    """
    rng = np.random.default_rng(seed)
    per_seed = []
    for a, b in zip(a_by_seed, b_by_seed):
        if [r["prompt"] for r in a] != [r["prompt"] for r in b]:
            raise ValueError("paired records must be the same prompts in the same order")
        clusters: dict[str, list[float]] = defaultdict(list)
        for ra, rb in zip(a, b):
            clusters[ra.get("word") or ra["prompt"]].append(float(ra["correct"]) - float(rb["correct"]))
        sums = np.array([sum(v) for v in clusters.values()])
        sizes = np.array([len(v) for v in clusters.values()])
        picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        per_seed.append((sums.sum() / sizes.sum(), sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)))
    point = statistics.fmean(p for p, _ in per_seed)
    boots = np.mean([b for _, b in per_seed], axis=0)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def count_table(records: list[dict]) -> dict:
    """Confusion counts by answer bucket, per-class recall and balanced accuracy for ``count``."""
    rs = [r for r in records if r["task"] == "count"]
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in rs:
        pred = count_bucket(int(r["pred"])) if r["pred"].strip().isdigit() else "other"
        confusion[count_bucket(int(r["gold"]))][pred] += 1
    recall = {}
    for bucket, row in confusion.items():
        exact = sum(r["correct"] for r in rs if count_bucket(int(r["gold"])) == bucket)
        recall[bucket] = exact / sum(row.values())
    return {"confusion": {k: dict(v) for k, v in confusion.items()}, "recall": recall,
            "balanced_accuracy": statistics.fmean(recall.values()) if recall else 0.0, "n": len(rs)}


def count_baselines(test: list, train: list) -> dict[str, float]:
    """Exact-match accuracy of models that can't count, on these ``count`` test examples.

    ``letter_only`` and ``length_only`` are the most common training answer per letter and per
    word length. They show how much the answer leaks through things other than counting.
    """
    def fitted(key) -> dict:
        table: dict = defaultdict(Counter)
        for ex in train:
            table[key(ex)][ex.answer] += 1
        return {k: v.most_common(1)[0][0] for k, v in table.items()}

    majority = Counter(ex.answer for ex in train).most_common(1)[0][0]
    by_letter = fitted(lambda ex: ex.letter)
    by_length = fitted(lambda ex: len(ex.target_words[0]))
    return {
        "most_common_answer": sum(ex.answer == majority for ex in test) / len(test),
        "letter_only": sum(ex.answer == by_letter.get(ex.letter, majority) for ex in test) / len(test),
        "length_only": sum(ex.answer == by_length.get(len(ex.target_words[0]), majority) for ex in test) / len(test),
        "presence_only": sum(ex.answer == ("1" if ex.letter in ex.target_words[0] else "0") for ex in test) / len(test),
    }


def task_baselines(dataset) -> dict[str, float]:
    """Reference points for the other tasks: most common ``nth`` letter, random ``category`` option."""
    nth = [ex for ex in dataset.test if ex.task == "nth"]
    top = Counter(ex.answer for ex in nth).most_common(1)[0][1]
    category = [ex for ex in dataset.test if ex.task == "category"]
    chance = statistics.fmean(1 / (ex.prompt.count(",") + 1) for ex in category)
    return {"nth_most_common_letter": top / len(nth), "category_random_option": chance}


def nth_errors(records: list[dict]) -> dict:
    """``nth`` errors by letter position and ordinal form. Needs the position from the prompt."""
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10}
    by_pos: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    forms = {"digit": [0, 0], "word": [0, 0]}
    for r in records:
        if r["task"] != "nth":
            continue
        m = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", r["prompt"])
        pos = int(m.group(1)) if m else next(v for k, v in words.items() if re.search(rf"\b{k}\b", r["prompt"]))
        form = forms["digit" if m else "word"]
        for bucket in (by_pos[pos], form):
            bucket[0] += not r["correct"]
            bucket[1] += 1
    return {"by_position": {k: by_pos[k] for k in sorted(by_pos)}, "by_form": forms}


# --------------------------------------------------------------------------- figures
def plot_loss_curves(log_dir: Path, names: list[str], out: Path) -> None:
    """Held-out loss during training, one line per run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        log = json.loads((log_dir / f"{name}.json").read_text(encoding="utf-8"))
        points = [(r["step"], r["test_loss"]) for r in log if "test_loss" in r]
        ax.plot(*zip(*points), label=name, marker=".")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out loss per answer token")
    ax.set_title("Held-out loss during training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_learning_curves(log_dir: Path, names: list[str], out: Path) -> None:
    """Accuracy on the fixed test subset over training, one panel per task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(3 * len(TASKS), 3.4), sharey=True)
    for name in names:
        log = json.loads((log_dir / f"{name}.json").read_text(encoding="utf-8"))
        points = [r for r in log if "accuracy" in r]
        if not points:
            continue
        for ax, task in zip(axes, TASKS):
            ax.plot([r["step"] for r in points], [r["accuracy"].get(task, 0) for r in points], label=name, marker=".")
            ax.set_title(task)
            ax.set_xlabel("step")
            ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("accuracy (subset)")
    axes[0].set_ylim(0, 1.05)
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def analyse(results_dir: Path, n_train: int = 3000, n_test: int = 300) -> str:
    """Write the figures and return the markdown analysis."""
    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = defaultdict(list)
    for name, res in results.items():
        groups[group_of(name, res)].append(name)
    lines = ["# Analysis (generated by `python -m mtt.analyze`)", ""]

    def preds(run: str, set_name: str | None = None) -> list[dict]:
        suffix = f"__{set_name}" if set_name else ""
        return load_predictions(results_dir / f"predictions_{run}{suffix}.jsonl")

    # --- counting
    lines += ["## Counting (balanced test set)", "",
              "| configuration | balanced acc. | recall 0 | recall 1 | recall 2 | recall 3+ | strawberry-style showcase |",
              "|---|---|---|---|---|---|---|"]
    for group, runs in groups.items():
        tables = [count_table(preds(r)) for r in runs]
        recall = {b: statistics.fmean(t["recall"].get(b, 0.0) for t in tables) for b in ("0", "1", "2", "3+")}
        balanced = statistics.fmean(t["balanced_accuracy"] for t in tables)
        showcase = "-"
        if (results_dir / f"predictions_{runs[0]}__showcase.jsonl").exists():
            recs = [r for run in runs for r in preds(run, "showcase")]
            showcase = f"{sum(r['correct'] for r in recs)}/{len(recs)}"
        lines.append(f"| {group} | {balanced:.3f} | {recall['0']:.3f} | {recall['1']:.3f} | {recall['2']:.3f} | "
                     f"{recall['3+']:.3f} | {showcase} |")

    seeds = sorted({results[r]["config"]["seed"] for r in results})
    version = next(iter(results.values()))["config"].get("data_version", 1)
    lines += ["", "Baselines that can't count (exact match, by seed):", "",
              "| seed | most common answer | best by letter only | best by word length only | letter present -> 1, else 0 |",
              "|---|---|---|---|---|"]
    all_task_base = []
    for seed in seeds:
        dataset = generate_dataset(seed, n_train, n_test, version=version)
        test = [ex for ex in dataset.test if ex.task == "count"]
        train = [ex for ex in dataset.train if ex.task == "count"]
        b = count_baselines(test, train)
        lines.append(f"| {seed} | {b['most_common_answer']:.3f} | {b['letter_only']:.3f} | {b['length_only']:.3f} | "
                     f"{b['presence_only']:.3f} |")
        all_task_base.append(task_baselines(dataset))
    lines += ["", "Other reference points: most common letter on `nth` "
              f"{statistics.fmean(b['nth_most_common_letter'] for b in all_task_base):.3f}, random option on "
              f"`category` {statistics.fmean(b['category_random_option'] for b in all_task_base):.3f}. Always answering "
              "the most common answer on `spell` and `reverse` scores 0.", ""]

    lines += ["### Confusion by true answer (all seeds pooled)", ""]
    for group, runs in groups.items():
        pooled: dict[str, Counter] = defaultdict(Counter)
        for run in runs:
            for k, row in count_table(preds(run))["confusion"].items():
                pooled[k].update(row)
        cells = "; ".join(f"true {k}: " + ", ".join(f"{p}={n}" for p, n in sorted(pooled[k].items()))
                          for k in sorted(pooled))
        lines.append(f"- **{group}**: {cells}")
    lines.append("")

    # --- paired differences against mtt
    base = next((g for g in groups if g == "mtt" or re.fullmatch(r"mtt_v\d+", g)), None)
    if base:
        lines += [f"## `{base}` minus each configuration (accuracy points, word-clustered bootstrap)", "",
                  "Seeds are paired by dataset where both have them; intervals are test-set sampling only.", "",
                  "| other | seeds | task | difference | 95% interval |", "|---|---|---|---|---|"]
        by_seed = {results[r]["config"]["seed"]: r for r in groups[base]}
        for group, runs in groups.items():
            if group == base:
                continue
            pairs = [(by_seed[results[r]["config"]["seed"]], r) for r in runs if results[r]["config"]["seed"] in by_seed]
            for task in TASKS:
                a = [[x for x in preds(m) if x["task"] == task] for m, _ in pairs]
                b = [[x for x in preds(o) if x["task"] == task] for _, o in pairs]
                d, lo, hi = paired_word_bootstrap(a, b)
                lines.append(f"| {group} | {len(pairs)} | {task} | {d:+.3f} | {lo:+.3f} to {hi:+.3f} |")
        lines.append("")

    # --- nth
    lines += ["## `nth` errors (all seeds pooled)", "", "| configuration | errors by position | digit-form | word-form |",
              "|---|---|---|---|"]
    for group, runs in groups.items():
        pos: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        forms = {"digit": [0, 0], "word": [0, 0]}
        for run in runs:
            e = nth_errors(preds(run))
            for k, (bad, total) in e["by_position"].items():
                pos[k][0] += bad
                pos[k][1] += total
            for k in forms:
                forms[k][0] += e["by_form"][k][0]
                forms[k][1] += e["by_form"][k][1]
        text = ", ".join(f"{k}: {pos[k][0]}/{pos[k][1]}" for k in sorted(pos))
        lines.append(f"| {group} | {text} | {forms['digit'][0]}/{forms['digit'][1]} | {forms['word'][0]}/{forms['word'][1]} |")
    lines.append("")

    # --- self-escalation
    self_esc = [r for r in results if results[r]["config"]["mode"] == "selfesc"]
    if self_esc:
        lines += ["## Self-escalation: did the model spell the word correctly first?", "",
                  "| task | scratch word correct | answer correct given correct scratch | answer correct given wrong scratch |",
                  "|---|---|---|---|"]
        recs = [r for run in self_esc for r in preds(run) if "scratch_ok" in r]
        for task in ("count", "spell", "reverse", "nth"):
            rs = [r for r in recs if r["task"] == task]
            good = [r for r in rs if r["scratch_ok"]]
            bad = [r for r in rs if not r["scratch_ok"]]
            lines.append(f"| {task} | {len(good)}/{len(rs)} | {sum(r['correct'] for r in good)}/{len(good)} | "
                         f"{sum(r['correct'] for r in bad)}/{len(bad)} |")
        lines.append("")

    # --- cost
    lines += ["## Cost", "", "`train_seconds` come from runs sharing the CPU, so only `bench.json` (each mode run "
              "alone) supports comparisons of time between modes.", "",
              "| run | mean train length | steps | train_seconds (shared CPU) |", "|---|---|---|---|"]
    for name, res in results.items():
        c = res["config"]
        lines.append(f"| {name} | {c.get('mean_train_len') or 0:.1f} | {c.get('steps')} | {c.get('train_seconds') or 0:.0f} |")
    bench_path = results_dir / "bench.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        base_s = bench["wlt_only"]["seconds_per_step"]
        lines += ["", "| mode (run alone) | seconds per step | vs wlt_only | mean tokens | threads |", "|---|---|---|---|---|"]
        for mode, b in bench.items():
            lines.append(f"| {mode} | {b['seconds_per_step']:.3f} | {b['seconds_per_step'] / base_s:.2f}x | "
                         f"{b['mean_train_len']:.1f} | {b['threads']} |")
    lines.append("")

    # --- figures
    log_dir = results_dir / "train_logs"
    seed0 = [r for r in results if not results[r]["config"]["seed"] and (log_dir / f"{r}.json").exists()]
    if seed0:
        plot_loss_curves(log_dir, seed0, results_dir / "loss_curves.png")
        plot_learning_curves(log_dir, seed0, results_dir / "learning_curves.png")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Regenerate report tables and figures from results files.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=300)
    args = p.parse_args(argv)
    results_dir = Path(args.results_dir)
    report = analyse(results_dir, args.n_train, args.n_test)
    write_text(results_dir / "analysis.md", report)
    print(report)


if __name__ == "__main__":
    main()

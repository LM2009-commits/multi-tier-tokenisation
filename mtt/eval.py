"""Evaluate trained runs: exact-match accuracy, sequence lengths, table, plots.

    python -m mtt.eval --runs wlt_only clt_only mtt            # accuracy + lengths
    python -m mtt.eval --runs wlt_only clt_only mtt mtt_learned selfesc
    python -m mtt.eval --lengths-only                          # no model needed
    python -m mtt.eval --router-compare                        # compare routers, no model needed

Outputs go to ``--out-dir`` (default ``results/``): ``results.md``, ``results.json``,
``aggregate.json``, ``accuracy.png``, ``seqlen.png``, one ``predictions_<run>.jsonl`` per run
(prompt, gold, prediction) for error analysis, one more per extra test set of data version 2
(``predictions_<run>__<set>.jsonl``), and thinned training logs in ``train_logs/``.
Accuracy is exact match of the decoded answer string on held-out words. A single seed's cell
gives a 95% Wilson interval; several seeds of the same configuration give mean +/- SD.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from .data import (GENERAL_PROMPTS, LETTER_TASKS, OOD_ROUTER_CASES, ROUTER_NEGATIVES_TEST, TASKS, Dataset,
                   Example, generate_dataset, generate_paraphrases)
from .pipeline import (MODES, ROUTED_MODES, build_oracle, build_router, build_tokenizer, encode_example,
                       extract_answer, extract_scratch, generation_prompt, sequence_lengths)
from .tokenizers.mtt import EOS, MTTTokenizer
from .tokenizers.wlt import WLT

PREFERRED_ORDER = ("wlt_only", "wlt_only_wltall", "clt_only", "clt_only_10k", "clt_marked", "mtt", "mtt_nomark",
                   "mtt_rule_words", "mtt_learned", "selfesc")
# Routers compared at test time on paraphrases, for models trained with the rule router.
PARAPHRASE_ROUTERS = ("oracle", "rule", "learned", "tagger")
CONFIG_KEYS = ("mode", "router", "seed", "steps", "n_params", "data_version", "wlt_corpus", "group",
               "train_seconds", "mean_train_len", "max_len", "provenance")


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def write_text(path: Path, text: str) -> None:
    """UTF-8 with LF line endings on every platform."""
    path.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- loading
def load_run(run_dir: Path, device: str = "auto"):
    """Load config, dataset, tokenizer, router and model of a trained run."""
    import torch

    from .model import ModelConfig, TinyGPT
    from .train import pick_device

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset = generate_dataset(config["seed"], config["n_train"], config["n_test"],
                               version=config.get("data_version", 1))
    tok = MTTTokenizer(WLT.load(run_dir / "wlt.json"))
    router = build_router(config["router"], dataset, config["seed"]) if config["mode"] in ROUTED_MODES else None
    dev = pick_device(device)
    ckpt = torch.load(run_dir / "model.pt", map_location=dev)
    model = TinyGPT(ModelConfig(**ckpt["model_config"])).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return config, dataset, tok, router, model


def evaluate_examples(examples: list[Example], mode: str, tok: MTTTokenizer, router, model) -> list[dict]:
    """Greedy-decode ``examples`` and compare the extracted answer with the gold one."""
    from .model import greedy_generate

    prompts = [generation_prompt(encode_example(ex, mode, tok, router)[0]) for ex in examples]
    generated = greedy_generate(model, prompts, EOS, max_new_tokens=48)
    records = []
    for ex, gen in zip(examples, generated):
        pred = extract_answer(gen, mode, tok)
        word = ex.target_words[0] if ex.target_words else None
        record = {"task": ex.task, "prompt": ex.prompt, "gold": ex.answer, "pred": pred,
                  "correct": pred == ex.answer, "word": word}
        if mode == "selfesc" and word is not None:
            scratch = extract_scratch(gen, tok)
            record["scratch"], record["scratch_ok"] = scratch, scratch == word
        records.append(record)
    return records


def evaluate_run(config: dict, dataset: Dataset, tok: MTTTokenizer, router, model) -> list[dict]:
    """Evaluate the main test set."""
    return evaluate_examples(dataset.test, config["mode"], tok, router, model)


def extra_conditions(config: dict, dataset: Dataset, router) -> dict[str, tuple[list[Example], object]]:
    """The extra test sets of a run: name -> (examples, router).

    A model trained with the rule router is tested on paraphrases under each router, oracle
    included, which separates router failure from model failure.
    """
    out: dict[str, tuple[list[Example], object]] = {}
    for name, examples in dataset.extras.items():
        if name != "paraphrase":
            out[name] = (examples, router)
        elif config["mode"] == "mtt" and config["router"] == "rule":
            for r in PARAPHRASE_ROUTERS:
                built = build_oracle(examples) if r == "oracle" else build_router(r, dataset, config["seed"])
                out[f"paraphrase-{r}"] = (examples, built)
        else:
            out[f"paraphrase-{config['router'] if config['mode'] in ROUTED_MODES else 'own'}"] = (examples, router)
    return out


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


def group_of(name: str, result: dict | None = None) -> str:
    """Configuration a run belongs to: its name without the seed suffix."""
    config = (result or {}).get("config") or {}
    return config.get("group") or re.sub(r"_s\d+$", "", name)


def aggregate(results: dict[str, dict]) -> dict[str, dict]:
    """Group runs by configuration: per task and test set, the accuracy of each seed and mean/SD."""
    groups: dict[str, dict] = {}
    for name, res in results.items():
        if not res.get("accuracy"):
            continue
        g = groups.setdefault(group_of(name, res), {"runs": [], "sets": {}})
        g["runs"].append(name)
        for set_name, summary in {"test": res["accuracy"], **res.get("sets", {})}.items():
            cells = g["sets"].setdefault(set_name, {})
            for task, a in summary.items():
                if a["n"]:
                    cells.setdefault(task, {"values": [], "n": a["n"]})["values"].append(a["acc"])
    for g in groups.values():
        for cells in g["sets"].values():
            for c in cells.values():
                c["mean"] = statistics.fmean(c["values"])
                c["sd"] = statistics.stdev(c["values"]) if len(c["values"]) > 1 else 0.0
    return groups


# --------------------------------------------------------------------------- reporting
def _accuracy_cell(group: dict, results: dict[str, dict], set_name: str, task: str) -> str:
    cell = group["sets"].get(set_name, {}).get(task)
    if cell is None:
        return "-"
    if len(cell["values"]) > 1:
        return f"{cell['mean']:.3f} ± {cell['sd']:.3f}"
    res = results[group["runs"][0]]
    a = res["accuracy"][task] if set_name == "test" else res["sets"][set_name][task]
    return f"{a['acc']:.3f} ({a['ci_low']:.2f}-{a['ci_high']:.2f})"


def make_markdown(results: dict[str, dict]) -> str:
    """Markdown tables: accuracy per test set, and mean sequence length (with ratio to wlt_only)."""
    groups = aggregate(results)
    names = list(groups)
    lines = ["## Exact-match accuracy on held-out words", ""]
    if names:
        labels = {"letter_mean": "**mean (letter tasks)**", "overall_mean": "**mean (all tasks)**"}
        lines.append("| task | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for task in (*TASKS, "letter_mean", "overall_mean"):
            cells = [_accuracy_cell(groups[g], results, "test", task) for g in names]
            lines.append(f"| {labels.get(task, task)} | " + " | ".join(cells) + " |")
        seeds = {g: len(groups[g]["runs"]) for g in names}
        lines += ["", "Cells: mean ± SD across seeds where a configuration has several, otherwise accuracy "
                  "(95% Wilson interval) for its one seed. Seeds per configuration: "
                  + ", ".join(f"{g} {k}" for g, k in seeds.items()) + "."]
        extra = sorted({s for g in names for s in groups[g]["sets"] if s != "test"})
        for set_name in extra:
            present = [g for g in names if set_name in groups[g]["sets"]]
            lines += ["", f"### Test set: {set_name}", "", "| task | " + " | ".join(present) + " |",
                      "|---|" + "---|" * len(present)]
            tasks = [t for t in (*TASKS, "letter_mean", "overall_mean")
                     if any(t in groups[g]["sets"][set_name] for g in present)]
            for task in tasks:
                cells = [_accuracy_cell(groups[g], results, set_name, task) for g in present]
                lines.append(f"| {labels.get(task, task)} | " + " | ".join(cells) + " |")
    runs = list(results)
    lines += ["", "## Mean sequence length in tokens (`<bos> prompt <sep> answer <eos>`)", ""]
    shown = [r for r in runs if group_of(r, results[r]) == r or not results[r].get("accuracy")
             or results[r].get("config", {}).get("seed", 0) == 0]
    lines.append("| task | " + " | ".join(shown) + " |")
    lines.append("|---|" + "---|" * len(shown))
    base_run = next((r for r in shown if r.startswith("wlt_only") and "wltall" not in r), None)
    base = results[base_run]["lengths"] if base_run else None
    for task in TASKS:
        cells = []
        for r in shown:
            v = results[r]["lengths"][task]["full"]
            ratio = f" ({v / base[task]['full']:.2f}x)" if base and r != base_run else ""
            cells.append(f"{v:.1f}{ratio}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_results(results: dict[str, dict], out_dir: Path) -> None:
    """Bar charts: accuracy per task (SD across seeds, or Wilson for one seed) and sequence length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = aggregate(results)
    names = list(groups)
    bars = [*TASKS, "letter_mean"]

    if names:
        fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(names) + 5), 4.5))
        width = 0.8 / len(names)
        for i, g in enumerate(names):
            xs = [b + i * width - 0.4 + width / 2 for b in range(len(bars))]
            cells = groups[g]["sets"]["test"]
            means = [cells[b]["mean"] for b in bars]
            if len(groups[g]["runs"]) > 1:
                errs = [cells[b]["sd"] for b in bars]
            else:
                a = results[groups[g]["runs"][0]]["accuracy"]
                errs = [[max(0, a[b]["acc"] - a[b]["ci_low"]) for b in bars],
                        [max(0, a[b]["ci_high"] - a[b]["acc"]) for b in bars]]
            ax.bar(xs, means, width, label=g, yerr=errs, capsize=2)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(bars)
        ax.legend(fontsize=8)
        ax.set_ylabel("exact-match accuracy (held-out words)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Accuracy by task and tokenisation mode")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy.png", dpi=150)
        plt.close(fig)

    runs = [r for r in results if not results[r].get("config", {}).get("seed")]
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(runs) + 5), 4.5))
    width = 0.8 / max(len(runs), 1)
    for i, r in enumerate(runs):
        xs = [g + i * width - 0.4 + width / 2 for g in range(len(TASKS))]
        ax.bar(xs, [results[r]["lengths"][t]["full"] for t in TASKS], width, label=r)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("mean tokens per example")
    ax.set_title("Sequence length cost by task and tokenisation mode")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "seqlen.png", dpi=150)
    plt.close(fig)


def thin_log(log: list[dict], every: int = 25) -> list[dict]:
    """Keep every ``every``-th training-loss record and all records with evaluation results."""
    return [r for r in log if r["step"] % every == 0 or "test_loss" in r or "accuracy" in r]


# --------------------------------------------------------------------------- router comparison
def _router_scores(router, examples: list[Example], negatives: tuple[str, ...]) -> dict[str, float]:
    """Word-level scores on positives (examples with a target word) and false alarms on negatives."""
    pos = [ex for ex in examples if ex.task in LETTER_TASKS]
    fired = found = routed_words = 0
    for ex in pos:
        route = router.route(ex.prompt)
        texts = [ex.prompt[s:e] for s, e in route.spans]
        words = [t for t in texts if len(t) > 1]
        fired += route.needs_clt
        found += ex.target_words[0] in words
        routed_words += len(words)
    others = [ex.prompt for ex in examples if ex.task not in LETTER_TASKS] + list(negatives)
    false_alarms = sum(router.route(p).needs_clt for p in others)
    neg_alarms = sum(router.route(p).needs_clt for p in negatives)
    return {"recall": found / len(pos), "precision": found / max(routed_words, 1),
            "fired": fired / len(pos), "false_alarm_rate": false_alarms / max(len(others), 1),
            "neg_alarms": neg_alarms, "n_negatives": len(negatives)}


def compare_routers(dataset: Dataset, seed: int = 0) -> str:
    """Markdown report comparing the routers on template prompts, old and fresh paraphrases."""
    routers = {"rule": build_router("rule", dataset, seed),
               "rule_words": build_router("rule_words", dataset, seed),
               "learned": build_router("learned", dataset, seed),
               "tagger": build_router("tagger", dataset, seed)}
    pos = [ex for ex in dataset.test if ex.task in LETTER_TASKS]
    neg = [ex for ex in dataset.test if ex.task not in LETTER_TASKS]

    lines = ["## Router comparison", "",
             "### Fresh paraphrases and negatives (never used to tune or train any router)", "",
             "Target-word recall and precision on paraphrased letter-task prompts; false alarms on "
             f"{len(ROUTER_NEGATIVES_TEST)} tricky general prompts.", "",
             "| router | fired on letter prompts | target recall | word precision | false alarms (tricky) |",
             "|---|---|---|---|---|"]
    fresh = dataset.extras.get("paraphrase") or generate_paraphrases(
        dataset.test_words, dataset.count_test_words or dataset.test_words, 200, str(seed))
    for name, r in routers.items():
        s = _router_scores(r, fresh, ROUTER_NEGATIVES_TEST)
        lines.append(f"| {name} | {s['fired']:.3f} | {s['recall']:.3f} | {s['precision']:.3f} | "
                     f"{s['neg_alarms']}/{s['n_negatives']} |")

    lines += ["", "### In-template prompts and the original 16 paraphrases",
              "", "The 16 no longer count as unseen: the rule router was revised after inspecting its "
              "behaviour on them.", "",
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
              f"(unseen words). `GENERAL_PROMPTS` ({len(GENERAL_PROMPTS)}) are in the learned routers' "
              "training data, so they aren't scored here.", ""]
    for name, ms in misses.items():
        lines.append(f"**{name} OOD misses ({len(ms)}):** " + ("; ".join(f"“{m}”" for m in ms) or "none"))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def discover_runs(runs_dir: Path) -> list[str]:
    found = [p.name for p in runs_dir.iterdir() if (p / "model.pt").exists()] if runs_dir.exists() else []

    def key(name: str) -> tuple[int, str]:
        base = re.sub(r"_v\d+", "", re.sub(r"_s\d+$", "", name))
        return (PREFERRED_ORDER.index(base) if base in PREFERRED_ORDER else 99, name)

    return sorted(found, key=key)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate MTT runs.")
    p.add_argument("--runs", nargs="*", default=None, help="run directory names inside --runs-dir")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--lengths-only", action="store_true", help="only compute sequence lengths (needs no model)")
    p.add_argument("--router-compare", action="store_true", help="compare the routers and exit")
    p.add_argument("--seed", type=int, default=0, help="(lengths-only / router-compare) dataset seed")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--wlt-corpus", default="train", choices=("train", "all"))
    p.add_argument("--data-version", type=int, default=1, choices=(1, 2),
                   help="(lengths-only / router-compare) dataset version")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.router_compare:
        dataset = generate_dataset(args.seed, args.n_train, args.n_test, version=args.data_version)
        report = compare_routers(dataset, args.seed)
        write_text(out_dir / "router_comparison.md", report)
        print(report)
        return

    results: dict[str, dict] = {}
    if args.lengths_only:
        dataset = generate_dataset(args.seed, args.n_train, args.n_test, version=args.data_version)
        tok = build_tokenizer(dataset, include_test_words=args.wlt_corpus == "all")
        for mode in MODES:
            router = build_router("rule") if mode in ROUTED_MODES else None
            results[mode] = {"lengths": sequence_lengths(dataset.test, mode, tok, router)}
    else:
        names = args.runs or discover_runs(Path(args.runs_dir))
        if not names:
            raise SystemExit(f"no trained runs found in {args.runs_dir!r}; run mtt.train first")
        log_dir = out_dir / "train_logs"
        log_dir.mkdir(exist_ok=True)
        for name in names:
            run_dir = Path(args.runs_dir) / name
            config, dataset, tok, router, model = load_run(run_dir, args.device)
            print(f"evaluating {name} ({len(dataset.test)} test prompts)...", flush=True)
            records = evaluate_run(config, dataset, tok, router, model)
            write_text(out_dir / f"predictions_{name}.jsonl", "\n".join(json.dumps(r) for r in records) + "\n")
            sets = {}
            for set_name, (examples, set_router) in extra_conditions(config, dataset, router).items():
                print(f"  {set_name} ({len(examples)} prompts)...", flush=True)
                recs = evaluate_examples(examples, config["mode"], tok, set_router, model)
                write_text(out_dir / f"predictions_{name}__{set_name}.jsonl",
                           "\n".join(json.dumps(r) for r in recs) + "\n")
                sets[set_name] = summarise(recs)
            results[name] = {"accuracy": summarise(records), "sets": sets,
                             "lengths": sequence_lengths(dataset.test, config["mode"], tok, router),
                             "config": {k: config.get(k) for k in CONFIG_KEYS}}
            log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
            write_text(log_dir / f"{name}.json", json.dumps(thin_log(log)))

    write_text(out_dir / "results.json", json.dumps(results, indent=2))
    write_text(out_dir / "aggregate.json", json.dumps(aggregate(results), indent=2))
    report = make_markdown(results)
    write_text(out_dir / "results.md", report)
    plot_results(results, out_dir)
    print(report)


if __name__ == "__main__":
    main()

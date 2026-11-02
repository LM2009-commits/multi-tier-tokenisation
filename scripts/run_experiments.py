#!/usr/bin/env python
"""Run the data-version-2 experiment matrix: parallel, resumable, skips finished runs.

    python scripts/run_experiments.py --dry-run            # list the jobs and exit
    python scripts/run_experiments.py --smoke              # tiny versions of every job, then eval
    python scripts/run_experiments.py                      # the real thing (about 17 h on 18 CPU threads)
    python scripts/run_experiments.py --only clt_only_v2   # jobs whose run name contains this text
    python scripts/run_experiments.py --bench              # time each mode alone (cost table)

Run it again after any interruption: finished runs are skipped and the rest resume from their
last checkpoint. Per-run logs are in ``<out-dir>/logs/``. ``scripts/run_all.sh`` remains the way
to reproduce the original version-1 experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mtt.train import parse_args as train_args, run_name  # noqa: E402

SEEDS = (0, 1, 2)
# Approximate hours for one job on 20 threads, measured for the three main modes on version 1.
HOURS = {"wlt_only": 1.0, "mtt": 1.8, "mtt_nomark": 1.7, "selfesc": 2.7, "clt_only": 3.5, "clt_marked": 3.5}


@dataclass(frozen=True)
class Job:
    flags: tuple[str, ...]
    hours: float


def matrix() -> list[Job]:
    """Every run of the experiment, as ``mtt.train`` flags."""
    jobs = [Job(("--mode", mode, "--seed", str(seed)), HOURS[mode])
            for mode in ("wlt_only", "clt_only", "mtt") for seed in SEEDS]
    jobs += [
        Job(("--mode", "clt_only", "--steps", "10000"), 2 * HOURS["clt_only"]),  # training budget control
        Job(("--mode", "wlt_only", "--wlt-corpus", "all"), HOURS["wlt_only"]),   # robustness
        Job(("--mode", "mtt", "--router", "learned"), HOURS["mtt"]),
        Job(("--mode", "mtt", "--router", "rule_words"), HOURS["mtt"]),
        Job(("--mode", "selfesc"), HOURS["selfesc"]),                            # self-escalation
        Job(("--mode", "clt_marked"), HOURS["clt_marked"]),                      # why MTT wins
        Job(("--mode", "mtt_nomark"), HOURS["mtt_nomark"]),
    ]
    return sorted(jobs, key=lambda j: -j.hours)  # longest first, so short jobs fill the gaps


def start(job: Job, name: str, args: argparse.Namespace, log_dir: Path) -> tuple[subprocess.Popen, object]:
    cmd = [sys.executable, "-m", "mtt.train", *job.flags, "--data-version", "2", "--out-dir", args.out_dir,
           "--threads", str(args.threads), "--resume"]
    if args.smoke:
        cmd += ["--smoke", "--device", "cpu"]
    log = (log_dir / f"{name}.log").open("a", encoding="utf-8")
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
    # Below-normal priority keeps the PC responsive; these jobs only yield when something else needs the CPU.
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags), log


def run_matrix(args: argparse.Namespace) -> list[str]:
    """Run the jobs, at most ``--jobs`` at a time. Returns the names of the jobs that failed."""
    log_dir = Path(args.out_dir) / "logs"
    pending: list[tuple[Job, str]] = []
    for job in matrix():
        extra = ["--smoke"] if args.smoke else []
        name = run_name(train_args([*job.flags, "--data-version", "2", *extra]))
        if args.only and args.only not in name:
            continue
        done = (Path(args.out_dir) / name / "model.pt").exists()
        print(f"{'done   ' if done else 'pending'}  {name:<28} about {job.hours:.1f} h on 20 threads")
        if not done:
            pending.append((job, name))
    if args.dry_run:
        return []
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, object, float]] = {}
    failed: list[str] = []
    while pending or running:
        while pending and len(running) < args.jobs:
            job, name = pending.pop(0)
            proc, log = start(job, name, args, log_dir)
            running[name] = (proc, log, time.time())
            print(f"[{time.strftime('%H:%M:%S')}] started  {name}", flush=True)
        for name, (proc, log, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[name]
            state = "finished" if code == 0 else f"FAILED ({code})"
            print(f"[{time.strftime('%H:%M:%S')}] {state} {name} after {(time.time() - t0) / 3600:.2f} h", flush=True)
            if code != 0:
                failed.append(name)
        time.sleep(1 if args.smoke else 15)
    return failed


def run_bench(args: argparse.Namespace) -> None:
    """Time each mode alone, on the same thread count. Wall-clock times from the parallel matrix aren't
    comparable, because the jobs slow each other down."""
    out: dict[str, dict] = {}
    steps = 150
    for mode in ("wlt_only", "mtt", "mtt_nomark", "selfesc", "clt_only", "clt_marked"):
        run_dir = Path(args.out_dir) / "bench"
        name = f"bench_{mode}"
        subprocess.run([sys.executable, "-m", "mtt.train", "--mode", mode, "--data-version", "2", "--steps", str(steps),
                        "--name", name, "--out-dir", str(run_dir), "--threads", str(args.threads), "--force",
                        "--acc-every", "0", "--eval-every", str(10 * steps), "--ckpt-every", str(10 * steps)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        config = json.loads((run_dir / name / "config.json").read_text(encoding="utf-8"))
        out[mode] = {"seconds_per_step": config["train_seconds"] / steps, "mean_train_len": config["mean_train_len"],
                     "steps": steps, "threads": args.threads}
        print(mode, out[mode], flush=True)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "bench.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")


def evaluate(args: argparse.Namespace) -> None:
    out_dir = "results_smoke" if args.smoke else "results"
    device = ["--device", "cpu"] if args.smoke else []
    data = ["--data-version", "2"]
    subset = ["--n-train", "150", "--n-test", "20"] if args.smoke else []
    subprocess.run([sys.executable, "-m", "mtt.eval", "--runs-dir", args.out_dir, "--out-dir", out_dir, *device], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "mtt.eval", "--router-compare", "--out-dir", out_dir, *data, *subset], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "mtt.analyze", "--results-dir", out_dir, *subset], cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the experiment matrix.")
    p.add_argument("--jobs", type=int, default=6, help="runs at the same time")
    p.add_argument("--threads", type=int, default=3, help="CPU threads per run")
    p.add_argument("--out-dir", default=None, help="default: runs (runs_smoke with --smoke)")
    p.add_argument("--only", default=None, help="only runs whose name contains this text")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny version of every job")
    p.add_argument("--no-eval", action="store_true", help="skip evaluation and analysis afterwards")
    p.add_argument("--bench", action="store_true", help="time each mode alone instead (use with --threads 20 --jobs 1)")
    args = p.parse_args()
    args.out_dir = args.out_dir or ("runs_smoke" if args.smoke else "runs")
    if args.bench:
        run_bench(args)
        return
    failed = run_matrix(args)
    if args.dry_run:
        return
    if failed:
        raise SystemExit(f"failed runs: {', '.join(failed)}. Fix the cause and run this again to resume.")
    if not args.no_eval:
        evaluate(args)


if __name__ == "__main__":
    main()

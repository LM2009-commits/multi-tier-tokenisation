"""Ask trained models a question.

    python -m mtt.ask "How many r's are in strawberry?"            # all runs found in runs/
    python -m mtt.ask --runs wlt_only mtt "Spell the word: banana"
    python -m mtt.ask --runs mtt                                    # interactive; empty line quits

The models were trained only on the project's templates and word list, so they answer
questions phrased like the training prompts (spell / reverse / nth letter / count /
category). Free-form chat will not work. Try words from the held-out list to see
generalisation:  python -c "from mtt.data import generate_dataset as g; print(g(0,3000,300).test_words[:20])"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import Example
from .eval import discover_runs, load_run
from .pipeline import encode_example, extract_answer, generation_prompt
from .tokenizers.mtt import CLT_CLOSE, CLT_OPEN, EOS, MTTTokenizer


def render(ids: list[int], tok: MTTTokenizer) -> str:
    """Show prompt ids as text with character-level spans marked as «...»."""
    out: list[str] = []
    wbuf: list[int] = []
    cbuf: list[int] = []

    def flush() -> None:
        if wbuf:
            out.append(tok.decode(wbuf, strict=False))
            wbuf.clear()
        if cbuf:  # decode a whole span at once so <CAP> markers are applied
            out.append(tok.decode(cbuf, strict=False))
            cbuf.clear()

    for i in ids:
        if i == CLT_OPEN:
            flush()
            out.append("«")
        elif i == CLT_CLOSE:
            flush()
            out.append("»")
        elif i >= tok.clt_offset:
            if wbuf:
                flush()
            cbuf.append(i)
        else:
            if cbuf:
                flush()
            wbuf.append(i)
    flush()
    return "".join(out)


def ask(prompt: str, config: dict, tok: MTTTokenizer, router, model) -> tuple[str, str]:
    """Return ``(prompt as the model sees it, the model's answer)``."""
    from .model import greedy_generate

    mode = config["mode"]
    example = Example(task="free", prompt=prompt, answer="")
    prompt_ids, _ = encode_example(example, mode, tok, router)
    generated = greedy_generate(model, [generation_prompt(prompt_ids)], EOS, max_new_tokens=48)[0]
    return render(prompt_ids, tok), extract_answer(generated, mode, tok)


def resolve_runs_and_prompt(requested: list[str] | None, prompt: str | None, runs_dir: Path
                            ) -> tuple[list[str], str | None]:
    """Split ``--runs`` values into run names and a question.

    ``--runs`` accepts several names, so a question typed after it ends up in the same list.
    Run names never contain spaces, so an entry with a space is treated as the question.
    """
    requested = requested or []
    questions = [r for r in requested if " " in r]
    names = [r for r in requested if " " not in r]
    for name in names:
        if not (runs_dir / name / "config.json").exists():
            raise SystemExit(f"run {name!r} not found in {str(runs_dir)!r} "
                             f"(available: {', '.join(discover_runs(runs_dir)) or 'none'})")
    if prompt is None and questions:
        prompt = " ".join(questions)
    return names or discover_runs(runs_dir), prompt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ask trained MTT runs a question.")
    p.add_argument("prompt", nargs="?", help="question (omit for interactive mode)")
    p.add_argument("--runs", nargs="*", default=None, help="run names inside --runs-dir (default: all)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    names, prompt = resolve_runs_and_prompt(args.runs, args.prompt, Path(args.runs_dir))
    if not names:
        raise SystemExit(f"no trained runs found in {args.runs_dir!r}")
    loaded = {n: load_run(Path(args.runs_dir) / n, args.device) for n in names}

    def answer_all(prompt: str) -> None:
        for name, (config, _, tok, router, model) in loaded.items():
            try:
                seen, answer = ask(prompt, config, tok, router, model)
            except ValueError as err:  # e.g. clt_only cannot encode non-ASCII characters
                print(f"  {name:<16} (cannot encode this prompt: {err})")
                continue
            print(f"  {name:<16} {answer!r:<22} model saw: {seen}")

    if prompt:
        answer_all(prompt)
        return
    print("Type a question (empty line to quit). Character-level spans are shown as «...».")
    while True:
        line = input("> ").strip()
        if not line:
            break
        answer_all(line)


if __name__ == "__main__":
    main()

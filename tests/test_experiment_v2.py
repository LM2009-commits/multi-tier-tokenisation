"""Data version 2, run naming, checkpoint resume, ablation modes, routers and statistics."""
import json
from pathlib import Path

import pytest

from mtt.analyze import count_baselines, count_table, paired_word_bootstrap, word_length_table
from mtt.data import (COUNT_BUCKETS, COUNT_EXTRA_WORDS, COUNT_TEMPLATES, NTH_TEMPLATES, PARAPHRASE_TEMPLATES,
                      REVERSE_TEMPLATES, ROUTER_NEGATIVES_TEST, ROUTER_TRAIN_TEMPLATES, SHOWCASE_WORDS,
                      SPELL_TEMPLATES, TASKS, WORDS, count_bucket, generate_dataset)
from mtt.eval import aggregate, group_of, make_markdown, summarise, thin_log
from mtt.pipeline import (ROUTED_MODES, build_oracle, build_router, encode_example, extract_answer,
                          extract_scratch)
from mtt.router import RuleRouter
from mtt.tokenizers.mtt import CLT_CLOSE, CLT_OPEN

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def v2():
    return generate_dataset(0, 400, 80, version=2)


def test_v1_prompts_match_the_published_predictions():
    records = [json.loads(x) for x in (ROOT / "results/v1/predictions_mtt.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["prompt"] for r in records] == [e.prompt for e in generate_dataset(0, 3000, 300).test]


def test_v2_count_answers_are_balanced_and_correct(v2):
    for split, per_bucket in ((v2.train, 100), (v2.test, 20)):
        count = [e for e in split if e.task == "count"]
        for bucket in COUNT_BUCKETS:
            assert sum(count_bucket(int(e.answer)) == bucket for e in count) == per_bucket
        assert all(e.answer == str(e.target_words[0].count(e.letter)) for e in count)
        assert len({e.prompt for e in count}) == len(count)


def test_v2_full_size_has_enough_hard_examples_for_every_seed():
    for seed in (0, 1, 2):
        ds = generate_dataset(seed, 3000, 300, version=2)
        hard = [e for e in ds.test if e.task == "count" and int(e.answer) >= 3]
        assert len(hard) == 75 and len({e.target_words[0] for e in hard}) >= 30


def test_v2_words_never_cross_the_split(v2):
    train_words = {w for e in v2.train for w in e.target_words}
    test_words = {w for e in v2.test for w in e.target_words}
    for examples in v2.extras.values():
        test_words |= {w for e in examples for w in e.target_words}
    assert not train_words & test_words
    assert not {e.prompt for e in v2.train} & {e.prompt for e in v2.test}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_showcase_words_are_never_in_training(seed):
    ds = generate_dataset(seed, 400, 40, version=2)
    train_words = set(ds.train_words) | set(ds.count_train_words)
    assert not train_words & SHOWCASE_WORDS
    assert not any(w in e.prompt.split() for e in ds.train for w in ("strawberry", "mississippi"))
    assert "strawberry" in {e.target_words[0] for e in ds.extras["showcase"]}


def test_count_extra_words_only_appear_in_count(v2):
    extra = set(COUNT_EXTRA_WORDS)
    assert len(extra) >= 300 and not extra & set(WORDS)
    assert all(not extra & set(e.target_words) for e in v2.train + v2.test if e.task != "count")


def test_paraphrase_templates_are_disjoint_from_what_routers_learn_from():
    for task, templates in ROUTER_TRAIN_TEMPLATES.items():
        assert not set(templates) & set(PARAPHRASE_TEMPLATES[task])


def test_paraphrase_gold_spans_point_at_the_target(v2):
    for ex in v2.extras["paraphrase"]:
        if ex.task == "category":
            assert ex.gold_spans == () and ex.answer in ex.prompt
            continue
        texts = [ex.prompt[s:e] for s, e in ex.gold_spans]
        assert ex.target_words[0] in texts
        if ex.task == "count":
            assert ex.letter in texts


# --------------------------------------------------------------------------- run names
def names(*argv):
    from mtt.train import parse_args, run_name

    return run_name(parse_args(list(argv)))


def test_run_names_differ_for_settings_that_change_the_result():
    runs = [("--mode", "mtt"), ("--mode", "mtt", "--seed", "1"), ("--mode", "mtt", "--data-version", "2"),
            ("--mode", "mtt", "--data-version", "2", "--seed", "1"), ("--mode", "mtt", "--steps", "10000"),
            ("--mode", "wlt_only", "--wlt-corpus", "all"), ("--mode", "mtt", "--router", "learned")]
    built = [names(*r) for r in runs]
    assert len(set(built)) == len(built)
    assert built[:2] == ["mtt", "mtt_s1"] and built[3] == "mtt_v2_s1"
    assert names("--mode", "clt_only", "--steps", "10000", "--smoke") != names("--mode", "clt_only", "--smoke")


def test_refuses_to_overwrite_a_finished_run(tmp_path):
    from mtt.train import main

    args = ["--mode", "wlt_only", "--smoke", "--out-dir", str(tmp_path), "--device", "cpu", "--acc-every", "0"]
    main(args)
    with pytest.raises(SystemExit):
        main(args)
    main([*args, "--force"])


def test_resume_gives_the_same_weights_as_an_uninterrupted_run(tmp_path):
    torch = pytest.importorskip("torch")
    from mtt.train import main

    base = ["--mode", "mtt", "--smoke", "--device", "cpu", "--threads", "2", "--data-version", "2"]
    main([*base, "--out-dir", str(tmp_path / "whole")])
    main([*base, "--out-dir", str(tmp_path / "split"), "--stop-after", "23"])
    assert (tmp_path / "split/mtt_v2/checkpoint.pt").exists() and not (tmp_path / "split/mtt_v2/model.pt").exists()
    main([*base, "--out-dir", str(tmp_path / "split"), "--resume"])
    a = torch.load(tmp_path / "whole/mtt_v2/model.pt")["model"]
    b = torch.load(tmp_path / "split/mtt_v2/model.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not (tmp_path / "split/mtt_v2/checkpoint.pt").exists()
    log_a = json.loads((tmp_path / "whole/mtt_v2/train_log.json").read_text())
    log_b = json.loads((tmp_path / "split/mtt_v2/train_log.json").read_text())
    assert [r["train_loss"] for r in log_a] == [r["train_loss"] for r in log_b]


# --------------------------------------------------------------------------- modes
def first(ds, task):
    return next(e for e in ds.test if e.task == task)


def test_clt_marked_is_all_characters_with_the_target_marked(tok, small_dataset):
    ex = first(small_dataset, "spell")
    p, a = encode_example(ex, "clt_marked", tok, RuleRouter())
    assert tok.decode(p) == ex.prompt and tok.decode(a) == ex.answer
    assert all(i >= tok.clt_offset or i in (CLT_OPEN, CLT_CLOSE) for i in p)
    assert p.count(CLT_OPEN) in (2, 3)  # the text before the target, the target, any text after it
    word = ex.target_words[0]
    closes = [i for i, t in enumerate(p) if t == CLT_CLOSE]
    assert any(p[i - len(word) - 1] == CLT_OPEN and tok.decode(p[i - len(word) - 1 : i + 1]) == word for i in closes)
    plain, _ = encode_example(ex, "clt_only", tok)
    assert len(p) == len(plain) + 2 * (p.count(CLT_OPEN) - 1)


def test_mtt_nomark_is_mtt_without_markers(tok, small_dataset):
    ex = first(small_dataset, "reverse")
    p, a = encode_example(ex, "mtt", tok, RuleRouter())
    pn, an = encode_example(ex, "mtt_nomark", tok, RuleRouter())
    assert pn == [i for i in p if i not in (CLT_OPEN, CLT_CLOSE)]
    assert an == [i for i in a if i not in (CLT_OPEN, CLT_CLOSE)]
    assert extract_answer(an, "mtt_nomark", tok) == ex.answer


def test_extract_scratch(tok, small_dataset):
    ex = first(small_dataset, "reverse")
    _, a = encode_example(ex, "selfesc", tok)
    assert extract_scratch(a, tok) == ex.target_words[0]
    assert extract_scratch([], tok) is None and extract_scratch([CLT_OPEN], tok) is None
    assert ROUTED_MODES == {"mtt", "clt_marked", "mtt_nomark"}


# --------------------------------------------------------------------------- routers
@pytest.mark.parametrize("prompt", [
    "Spell check my essay", "Reverse engineering is hard", "Put the car in reverse",
    "Drive in reverse carefully", "Reverse the order of this list", "Spell the words cat and dog",
])
def test_rule_router_false_positives_are_fixed(prompt):
    assert not RuleRouter().route(prompt).needs_clt


def test_rule_router_finds_the_word_not_the_word_order():
    p = "Please write the word planet in reverse order."
    r = RuleRouter().route(p)
    assert [p[s:e] for s, e in r.spans] == ["planet"]


def test_rule_router_ignores_characters_the_clt_cannot_encode():
    assert not RuleRouter().route("Spell ſun").needs_clt


def test_rule_router_on_every_template_and_word():
    r = RuleRouter()
    missed = set()
    for w in [*WORDS, *COUNT_EXTRA_WORDS]:
        prompts = [t.format(w=w) for t in (*SPELL_TEMPLATES, *REVERSE_TEMPLATES)]
        prompts += [t.format(w=w, l="e") for t in COUNT_TEMPLATES]
        prompts += [t.format(w=w, o=o) for t in NTH_TEMPLATES for o in ("3rd", "third")]
        for p in prompts:
            if w not in [p[s:e] for s, e in r.route(p).spans]:
                missed.add(p)
    # Genuinely ambiguous bare phrases: nothing says these are about the word itself.
    assert missed == {"Reverse order", "Reverse string", "Reverse engineering"}


def test_oracle_router_returns_gold_spans(v2):
    examples = v2.extras["paraphrase"]
    oracle = build_oracle(examples)
    for ex in examples:
        route = oracle.route(ex.prompt)
        assert route.spans == ex.gold_spans
        assert route.answer_clt == (ex.task in ("spell", "reverse", "nth"))


@pytest.fixture(scope="module")
def tagger(v2):
    return build_router("tagger", v2, 0)


def test_tagger_routes_one_word_and_for_counting_the_letter(v2, tagger):
    hits = total = 0
    for ex in v2.test:
        if ex.task == "category":
            continue
        route = tagger.route(ex.prompt)
        texts = [ex.prompt[s:e] for s, e in route.spans]
        assert sum(len(t) > 1 for t in texts) <= 1
        hits += ex.target_words[0] in texts
        total += 1
    assert hits / total > 0.9


def test_tagger_is_deterministic(v2, tagger):
    other = build_router("tagger", v2, 0)
    for p in ROUTER_NEGATIVES_TEST[:10]:
        assert tagger.route(p) == other.route(p)


# --------------------------------------------------------------------------- statistics
def rec(task, ok, prompt="p", word="w"):
    return {"task": task, "correct": ok, "prompt": prompt, "word": word}


def test_bootstrap_of_identical_runs_is_exactly_zero():
    a = [rec("spell", i % 2 == 0, f"p{i}", f"w{i % 5}") for i in range(40)]
    assert paired_word_bootstrap([a], [a]) == (0.0, 0.0, 0.0)


def test_bootstrap_measures_a_known_difference():
    a = [rec("spell", True, f"p{i}", f"w{i // 4}") for i in range(80)]
    b = [rec("spell", i % 2 == 0, f"p{i}", f"w{i // 4}") for i in range(80)]
    point, lo, hi = paired_word_bootstrap([a], [b], n_boot=500)
    assert point == pytest.approx(0.5) and lo <= point <= hi


def test_bootstrap_refuses_unpaired_records():
    with pytest.raises(ValueError):
        paired_word_bootstrap([[rec("t", True, "x")]], [[rec("t", True, "y")]])


def test_clustering_widens_the_interval_when_words_repeat():
    # 40 words with 10 prompts each. Each word is entirely right or wrong in a, always right in b's opposite.
    a = [rec("t", (i // 10) % 2 == 0, f"p{i}", f"w{i // 10}") for i in range(400)]
    b = [rec("t", (i // 10) % 2 == 1, f"p{i}", f"w{i // 10}") for i in range(400)]
    u = [rec("t", (i // 10) % 2 == 0, f"p{i}", f"u{i}") for i in range(400)]
    v = [rec("t", (i // 10) % 2 == 1, f"p{i}", f"u{i}") for i in range(400)]
    _, lo, hi = paired_word_bootstrap([a], [b], n_boot=800)
    _, ulo, uhi = paired_word_bootstrap([u], [v], n_boot=800)
    assert hi - lo > 2 * (uhi - ulo)


def test_aggregate_groups_seeds_and_computes_spread():
    def result(acc):
        records = [{"task": t, "correct": i < acc} for t in TASKS for i in range(10)]
        return {"accuracy": summarise(records), "sets": {}, "lengths": {t: {"full": 10.0} for t in TASKS},
                "config": {"group": "mtt_v2", "seed": acc}}

    groups = aggregate({"mtt_v2": result(6), "mtt_v2_s1": result(8)})
    cell = groups["mtt_v2"]["sets"]["test"]["spell"]
    assert cell["values"] == [0.6, 0.8] and cell["mean"] == pytest.approx(0.7)
    assert cell["sd"] == pytest.approx(0.1414, abs=1e-3)
    assert "0.700 ± 0.141" in make_markdown({"mtt_v2": result(6), "mtt_v2_s1": result(8)})
    assert "one training seed" not in make_markdown({"mtt_v2": result(6)})


def test_group_of_strips_the_seed_without_a_config():
    assert group_of("mtt_v2_s2") == "mtt_v2" and group_of("mtt_v2") == "mtt_v2"


def test_count_table():
    rows = [("0", "0"), ("0", "1"), ("1", "1"), ("3", "3"), ("4", "3"), ("2", "x")]
    t = count_table([{"task": "count", "gold": g, "pred": p, "correct": g == p} for g, p in rows])
    assert t["recall"] == {"0": 0.5, "1": 1.0, "3+": 0.5, "2": 0.0}
    assert t["balanced_accuracy"] == pytest.approx(0.5)
    assert t["confusion"]["2"] == {"other": 1} and t["confusion"]["3+"] == {"3+": 2}


def test_word_length_table():
    rows = [("spell", "cat", True), ("spell", "dog", False), ("spell", "horse", True), ("nth", "cat", True),
            ("category", None, True)]
    t = word_length_table([{"task": k, "word": w, "correct": c} for k, w, c in rows], "spell")
    assert t == {3: [1, 2], 5: [1, 1]}


def test_count_baselines_on_balanced_data_are_well_below_a_counting_model(v2):
    train = [e for e in v2.train if e.task == "count"]
    test = [e for e in v2.test if e.task == "count"]
    b = count_baselines(test, train)
    assert b["most_common_answer"] <= 0.3 and b["presence_only"] <= 0.5


def test_thin_log_keeps_every_evaluation():
    log = [{"step": i} for i in range(1, 101)] + [{"step": 101, "test_loss": 1.0}, {"step": 102, "accuracy": {}}]
    thin = thin_log(log, every=25)
    assert [r["step"] for r in thin] == [25, 50, 75, 100, 101, 102]

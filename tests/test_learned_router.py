from mtt.data import LETTER_TASKS, OOD_ROUTER_CASES
from mtt.pipeline import build_router


def test_learned_router_generalises_to_held_out_words(small_dataset):
    router = build_router("learned", small_dataset)
    right = 0
    for ex in small_dataset.test:
        route = router.route(ex.prompt)
        right += route.needs_clt == (ex.task in LETTER_TASKS)
    assert right / len(small_dataset.test) > 0.95


def test_learned_router_spans_hit_target_word(small_dataset):
    router = build_router("learned", small_dataset)
    hits = total = 0
    for ex in small_dataset.test:
        if ex.task in LETTER_TASKS:
            route = router.route(ex.prompt)
            hits += ex.target_words[0] in [ex.prompt[s:e] for s, e in route.spans]
            total += 1
    assert hits / total > 0.9


def test_learned_router_is_deterministic(small_dataset):
    a, b = build_router("learned", small_dataset), build_router("learned", small_dataset)
    for p, _ in OOD_ROUTER_CASES:
        assert a.route(p) == b.route(p)

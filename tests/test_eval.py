from mtt.eval import summarise, wilson


def test_wilson_interval_known_values():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(100, 100)
    assert hi > 0.9999 and lo > 0.95


def test_summarise_pools_letter_and_overall_means():
    records = []
    for task, right in (("count", 3), ("spell", 4), ("reverse", 2), ("nth", 1), ("category", 4)):
        records += [{"task": task, "correct": i < right} for i in range(4)]
    s = summarise(records)
    assert s["count"]["acc"] == 0.75
    assert s["letter_mean"]["correct"] == 10 and s["letter_mean"]["n"] == 16
    assert abs(s["letter_mean"]["acc"] - 10 / 16) < 1e-12
    assert s["overall_mean"]["n"] == 20 and 0 < s["overall_mean"]["ci_low"] < s["overall_mean"]["ci_high"] <= 1

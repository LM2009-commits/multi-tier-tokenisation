import pytest

from mtt.data import GENERAL_PROMPTS
from mtt.router import NullRouter, RuleRouter

R = RuleRouter()


def targets(prompt, router=R):
    route = router.route(prompt)
    return [prompt[s:e] for s, e in route.spans], route


@pytest.mark.parametrize("prompt,expected,task", [
    ("How many r's are in strawberry?", ["r", "strawberry"], "count"),
    ("How many times does the letter e appear in banana?", ["e", "banana"], "count"),
    ("Count the a's in the word banana.", ["a", "banana"], "count"),
    ("how many s are there in mississippi", ["s", "mississippi"], "count"),
    ("Spell the word: banana", ["banana"], "spell"),
    ("How do you spell planet?", ["planet"], "spell"),
    ("Please spell out the word theme letter by letter.", ["theme"], "spell"),
    ("Reverse the word: planet", ["planet"], "reverse"),
    ("Write planet backwards.", ["planet"], "reverse"),
    ("What is garden spelled backwards?", ["garden"], "reverse"),
    ("What is the 4th letter of garden?", ["garden"], "nth"),
    ("Give me the second letter of the word milk.", ["milk"], "nth"),
    ("What is the last letter in orange?", ["orange"], "nth"),
    ("Apply ROT13 to hello", ["hello"], "cipher"),
    ("Shift each letter of hello by 3", ["hello"], "cipher"),
    ("Apply a Caesar shift of 3 to hello.", ["hello"], "cipher"),
    ("Encrypt hello with a Caesar cipher", ["hello"], "cipher"),
])
def test_rule_router_extracts_target_spans(prompt, expected, task):
    got, route = targets(prompt)
    assert sorted(got) == sorted(expected) and route.task == task
    assert route.answer_clt == (task != "count")


def test_spans_never_cover_the_whole_prompt():
    prompt = "How many r's are in strawberry?"
    _, route = targets(prompt)
    covered = sum(e - s for s, e in route.spans)
    assert 0 < covered < len(prompt) / 2


def test_letter_spans_can_be_disabled():
    got, _ = targets("How many r's are in strawberry?", RuleRouter(route_letters=False))
    assert got == ["strawberry"]


@pytest.mark.parametrize("prompt", [
    "Which of these is a fruit: apple, car?",
    "What is the capital of France?",
    "How many days are in a leap year?",
    "How many apples are in the basket?",
    "Explain how a refrigerator works.",
    "Name an animal that lives in the ocean.",
    "Translate the word hello into French.",
    "Which of these is a color: red, chair?",
    "",
])
def test_general_prompts_are_not_routed(prompt):
    route = R.route(prompt)
    assert not route.needs_clt and route.spans == () and route.task is None


def test_all_general_prompts_not_routed():
    for p in GENERAL_PROMPTS:
        assert not R.route(p).needs_clt, p


def test_router_interface_is_pluggable():
    assert not NullRouter().route("Spell the word: banana").needs_clt

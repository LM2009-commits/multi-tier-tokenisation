"""Synthetic letter-level tasks over an embedded word list (nothing is downloaded).

Tasks: ``count`` ``spell`` ``reverse`` ``nth`` (letter-level) and ``category`` (a control
task that WLT should handle well). Words are split into disjoint train/test sets, so the
test set measures generalisation to words never seen in any training example.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

TASKS = ("count", "spell", "reverse", "nth", "category")
LETTER_TASKS = ("count", "spell", "reverse", "nth")

# --------------------------------------------------------------------------- word list
_RAW_WORDS = """
river mountain forest ocean island valley desert meadow canyon glacier volcano harbor cliff
stream pebble boulder swamp prairie tundra jungle lagoon marsh orchard garden village street
bridge tower castle palace temple church market library museum theater station airport tunnel
highway thunder lightning rainbow blizzard drizzle breeze storm cloud sunshine snowfall frost
humid monsoon tornado hurricane drought mist shower finger thumb elbow shoulder ankle knee
wrist forehead eyebrow eyelash cheek chin throat stomach muscle skeleton heart brain tongue
tooth nose mother father sister brother uncle aunt cousin nephew niece grandma grandpa baby
child teenager parent neighbor friend stranger teacher student doctor nurse farmer sailor
pilot driver painter singer dancer writer baker butcher tailor lawyer judge soldier captain
king queen prince princess knight wizard kitchen bedroom bathroom hallway garage basement
attic window door ceiling floor carpet curtain pillow blanket mattress towel basket bucket
candle clock ladder shovel hammer wrench screwdriver pliers drill nail screw bolt bread butter
cheese yogurt cereal noodle pasta pizza burger sandwich salad soup pepper salt sugar honey
flour chocolate vanilla cinnamon ginger garlic onion carrot potato tomato cucumber lettuce
cabbage spinach broccoli pumpkin mushroom bean corn rice oatmeal pancake waffle muffin cookie
biscuit cake pudding custard candy popcorn pretzel peanut almond walnut cashew hazelnut
pistachio olive coffee juice milk water soda lemonade smoothie strawberry banana blueberry
raspberry pineapple watermelon grapefruit tangerine pomegranate avocado pencil eraser marker
crayon notebook folder binder ruler scissors backpack lesson homework exam grade history
science math geometry algebra physics chemistry biology grammar spelling reading writing
drawing music freedom justice courage wisdom patience honesty kindness loyalty curiosity memory
dream idea thought secret mystery adventure journey future present moment season reason
problem solution question answer running jumping swimming climbing walking talking singing
dancing laughing painting cooking building planting sailing flying bright dark gentle fierce
quiet noisy clever silly brave happy sleepy hungry thirsty heavy light smooth rough sharp
narrow wide shallow deep ancient modern golden silver wooden plastic giraffe elephant penguin
moose otter beaver badger squirrel hedgehog panda koala kangaroo crocodile alligator lizard
snake frog toad salmon trout shark whale octopus lobster crab shrimp oyster butterfly
dragonfly beetle ladybug cricket mosquito hornet sparrow robin pigeon swan duck goose chicken
rooster peacock owl crow computer keyboard monitor printer battery cable circuit program
software network internet website server camera phone speaker headphone satellite engine motor
pulley lever magnet compass telescope microscope airplane helicopter submarine canoe sailboat
wagon carriage sled skateboard jacket sweater scarf glove mitten sock boot sandal sneaker
helmet button zipper pocket collar sleeve apron uniform costume umbrella wallet purse ticket
coin ribbon balloon puzzle trophy medal flag banner poster picture frame soccer tennis hockey
baseball basketball football golf rugby volleyball boxing archery chess checkers marble kite
diamond emerald ruby sapphire crystal granite copper iron steel bronze platinum morning evening
midnight weekend monday tuesday wednesday thursday friday saturday sunday january february
march april june july august september october november december anchor arrow bottle branch
brush cactus cellar chimney circle coast cotton crown dinner feather fence flower fountain
garbage glasses grocery hammock jewel kettle lantern leather letter mailbox napkin needle orbit
parcel planet puppy quilt saddle shadow shelter spoon statue sunset teapot thread toilet
trumpet velvet violin whistle winter above across action active actual after again agree album
alive allow almost alone along already always amount angle angry anxious appear around arrive
artist attack autumn avenue balance bamboo barrel basic battle beauty become before begin
behind belief belong beside better beyond bitter bonus border borrow bottom bounce broken
burden bubble cabin canvas capable careful cattle caught century chance change chapter charge
cheerful chosen circus citizen classic climate closet clumsy collect comfort common complete
confuse contain control convert corner country crazy create credit culture curious current
custom damage danger decide defend degree deliver depend detail develop device direct
distance divide double dragon drawer during eager earth easily either empty enable energy
enjoy enough enter equal escape event every exact example excuse exist expect explain extra
fabric factory fairly famous fashion feeling fellow fiction field fifteen fight figure final
finish flavor flight follow forget formal forward freeze fresh friendly frozen fuel funny
gather general gentle ghost giant glance glossy gather govern grain grand grateful gravity
great ground group growth guard guess guest guide habit handle happen harvest health hidden
highway hollow honest hotel human humble hunter hungry husband ignore image impact income
indeed inside insist instant invent invite itself jacket jelly jersey jigsaw joyful juggle
jumbo junior kindly kitten knowledge label lately laugh launch lawyer layer league lemon
lesson level liberty lifted limit linen liquid listen little lively lonely lovely lucky
lunch magic mainly manage manner mansion margin master match matter meaning measure medium
member mention merchant mental method middle might minute mirror missing mixture modest
moment mostly mother mouse movie murder narrow nation nature nearby neither nervous never
newly nickel nobody normal notice novel number object obtain obvious occupy ocean offer
office often online opinion option orange order ordinary origin outdoor outside oxygen paddle
panel paper parade partner passage patrol pattern payment peace people perfect period permit
person photo phrase picnic pilot pitch plain planet plenty pocket poetry police polish pond
popular portion position possible potato powder power practice prefer pretty prevent price
pride primary print prison private prize process produce profit promise proper protect proud
provide public pumpkin purple purpose quality quarter quickly quiet radio rainy random rapid
rather reach ready realize reality recent record reduce reflect region regular relate release
remain remember remove repair repeat reply report rescue resist resource respect result retain
return reveal review reward rhythm ridge right ripple rocket rough round routine royal rubber
rumor rural sacred safety salary sample satisfy scale scene scheme school scream screen script
search second section select senior sensor series serious service settle several shake shape
share sheep sheet shell shift shine shiny shock shoot shore short should shout shrink sight
signal silent similar simple single sister skill sleep slice slide slight slope small smart
smell smile smoke snack social soften solid something sorry sound source space spare speak
special speech speed spend spirit split sport spread spring square stable staff stage stamp
stand start state steady steam still stock stone store story strange strength stretch strict
strike string strong studio study stuff style subject success sudden suffer sugar suggest
summer supply support supreme surface surprise survey sweet swift switch symbol system table
talent target taste tender thank theme theory thick thing think third those though thousand
threat three throw thunder tight timber title today together tonight topic total touch tough
toward track trade trail train travel treat trend trial tribe trick trouble truly trust truth
twice twist typical under uncle union unique unit until upper upset urban useful usual valid
value vapor various vegetable vehicle verse video view visit vital voice volume voyage wander
waste watch wealth weapon weather weight welcome western whale wheat wheel where which while
whisper whole whose wide width wield wild willing window wisdom wonder worry worth wrap wrong
yard yearly young youth zebra zero zone
"""

# --------------------------------------------------------------------------- categories
CATEGORIES: dict[str, tuple[str, ...]] = {
    "fruit": ("apple", "cherry", "grape", "lemon", "mango", "melon", "orange", "peach", "pear",
              "plum", "kiwi", "papaya", "apricot", "coconut", "lime", "fig"),
    "animal": ("tiger", "horse", "rabbit", "monkey", "dolphin", "zebra", "camel", "donkey",
               "eagle", "falcon", "turtle", "spider", "parrot", "goat", "lion", "wolf"),
    "vehicle": ("car", "bus", "truck", "train", "bicycle", "tractor", "scooter", "ferry",
                "rocket", "tram", "taxi", "van", "ship", "boat", "jeep", "motorbike"),
    "color": ("red", "blue", "green", "yellow", "purple", "pink", "brown", "black", "white",
              "grey", "violet", "crimson", "indigo", "maroon", "turquoise", "gold"),
    "furniture": ("chair", "table", "sofa", "desk", "shelf", "bed", "stool", "cabinet", "bench",
                  "wardrobe", "couch", "dresser", "lamp", "mirror", "bookcase", "futon"),
}
CATEGORY_ARTICLE = {
    "fruit": "a fruit", "animal": "an animal", "vehicle": "a vehicle", "color": "a color",
    "furniture": "a piece of furniture",
}

# --------------------------------------------------------------------------- templates
ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                 "ninth", "tenth")
COUNT_TEMPLATES = (
    "How many {l}'s are in {w}?",
    "How many times does the letter {l} appear in {w}?",
    "Count the {l}'s in the word {w}.",
    "How many {l} are there in {w}?",
    "Count the number of times {l} appears in {w}.",
)
SPELL_TEMPLATES = (
    "Spell the word: {w}",
    "Spell {w}.",
    "Please spell out the word {w} letter by letter.",
    "How do you spell {w}?",
    "Spell out {w}",
)
REVERSE_TEMPLATES = (
    "Reverse the word: {w}",
    "Reverse the letters of {w}.",
    "Write {w} backwards.",
    "What is {w} spelled backwards?",
    "Reverse {w}",
)
NTH_TEMPLATES = (
    "What is the {o} letter of {w}?",
    "Give me the {o} letter of the word {w}.",
    "Name the {o} letter in {w}.",
    "Tell me the {o} character of {w}.",
    "What is the {o} letter in the word {w}?",
)
CATEGORY_TEMPLATES = (
    "Which of these is {art}: {opts}?",
    "Which one is {art}: {opts}?",
    "Pick {art} from this list: {opts}.",
    "From {opts}, which is {art}?",
)

# Prompts that must NOT be routed to CLT (used as negatives for routers).
GENERAL_PROMPTS = (
    "What is the capital of France?", "Explain how rain forms.",
    "Translate good morning into Spanish.", "Who wrote Romeo and Juliet?",
    "Give me three tips for studying.", "Summarise the plot of Cinderella.",
    "What is 12 plus 30?", "Name a famous painter.", "Which planet is closest to the sun?",
    "Write a short poem about autumn.", "How do I boil an egg?",
    "What is the largest ocean on Earth?", "Suggest a name for a pet cat.",
    "Why is the sky blue?", "List some healthy breakfast foods.", "What does a vet do?",
    "Describe a sunny day at the beach.", "How many days are in a leap year?",
    "How many legs does a spider have?", "What is the opposite of hot?",
    "Which of these is a mammal: whale, salmon, trout?", "Tell me a joke about computers.",
    "What year did the first moon landing happen?", "How tall is Mount Everest?",
    "Recommend a book for a long train journey.", "What is the plural of mouse?",
    "How many players are in a football team?", "Explain what a noun is.",
)

# Hand-written paraphrases outside the training templates: (prompt, target word or None).
OOD_ROUTER_CASES = (
    ("Could you tally the number of e's inside the word elephant?", "elephant"),
    ("Please write the word planet in reverse order.", "planet"),
    ("Tell me which letter sits in fourth place in garden.", "garden"),
    ("Give me the letters of banana one at a time.", "banana"),
    ("How many s are there in mississippi?", "mississippi"),
    ("Spell out the word necessary for me.", "necessary"),
    ("What does the word planet spelled backwards look like?", "planet"),
    ("Count how often the letter o shows up in october.", "october"),
    ("What is the capital of Italy?", None),
    ("Which of these is a bird: eagle, table, river?", None),
    ("Write two sentences about the ocean.", None),
    ("How many minutes are in a day?", None),
    ("Explain why leaves change colour.", None),
    ("Is a tomato a fruit or a vegetable?", None),
    ("Suggest a spelling bee strategy for children.", None),
    ("What is the longest river in Africa?", None),
)


def _template_words() -> frozenset[str]:
    """All lower-case words appearing in any template or category phrase."""
    words: set[str] = set(ORDINAL_WORDS) | {"th", "st", "nd", "rd"}
    for temps in (COUNT_TEMPLATES, SPELL_TEMPLATES, REVERSE_TEMPLATES, NTH_TEMPLATES,
                  CATEGORY_TEMPLATES):
        for t in temps:
            words.update(re.findall(r"[a-z]+", re.sub(r"\{[a-z]+\}", " ", t.lower())))
    for name, art in CATEGORY_ARTICLE.items():
        words.update(re.findall(r"[a-z]+", art))
        words.add(name)
    return frozenset(words)


def build_word_list() -> list[str]:
    """Sorted unique lowercase words (4-10 letters), minus template and category words."""
    blocked = _template_words() | {w for items in CATEGORIES.values() for w in items}
    blocked |= {"this", "that", "these", "those"}  # words the rule router never treats as targets
    words = {w for w in _RAW_WORDS.split() if w.isalpha() and w.isascii() and 4 <= len(w) <= 10}
    return sorted(words - blocked)


WORDS: list[str] = build_word_list()


def split_words(words: list[str], seed: int, test_frac: float = 0.2) -> tuple[list[str], list[str]]:
    """Deterministic, disjoint train/test split of ``words``."""
    shuffled = sorted(words)
    random.Random(f"split-{seed}").shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_frac))
    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])


# --------------------------------------------------------------------------- examples
@dataclass(frozen=True)
class Example:
    """One prompt/answer pair. ``target_words`` are the words the question is about."""

    task: str
    prompt: str
    answer: str
    target_words: tuple[str, ...] = ()
    letter: str | None = None
    position: int | None = None


def _ordinal(k: int, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return ORDINAL_WORDS[k - 1]
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(k, "th")
    return f"{k}{suffix}"


def _make_count(rng: random.Random, words: list[str]) -> Example:
    w = rng.choice(words)
    letter = rng.choice(w) if rng.random() < 0.75 else rng.choice("abcdefghijklmnopqrstuvwxyz")
    prompt = rng.choice(COUNT_TEMPLATES).format(l=letter, w=w)
    return Example("count", prompt, str(w.count(letter)), (w,), letter)


def _make_spell(rng: random.Random, words: list[str]) -> Example:
    w = rng.choice(words)
    return Example("spell", rng.choice(SPELL_TEMPLATES).format(w=w), " ".join(w), (w,))


def _make_reverse(rng: random.Random, words: list[str]) -> Example:
    w = rng.choice(words)
    return Example("reverse", rng.choice(REVERSE_TEMPLATES).format(w=w), w[::-1], (w,))


def _make_nth(rng: random.Random, words: list[str]) -> Example:
    w = rng.choice(words)
    k = rng.randint(1, min(len(w), 10))
    prompt = rng.choice(NTH_TEMPLATES).format(o=_ordinal(k, rng), w=w)
    return Example("nth", prompt, w[k - 1], (w,), None, k)


def _make_category(rng: random.Random, words: list[str] | None = None) -> Example:
    cat = rng.choice(sorted(CATEGORIES))
    correct = rng.choice(CATEGORIES[cat])
    others = [c for c in sorted(CATEGORIES) if c != cat]
    distractors = [rng.choice(CATEGORIES[rng.choice(others)]) for _ in range(rng.randint(1, 2))]
    options = distractors + [correct]
    rng.shuffle(options)
    prompt = rng.choice(CATEGORY_TEMPLATES).format(art=CATEGORY_ARTICLE[cat], opts=", ".join(options))
    return Example("category", prompt, correct)


_MAKERS = {
    "count": _make_count, "spell": _make_spell, "reverse": _make_reverse,
    "nth": _make_nth, "category": _make_category,
}


def generate_task(task: str, words: list[str], n: int, seed: str, exclude: set[str] | None = None
                  ) -> list[Example]:
    """Generate up to ``n`` examples with unique prompts (not in ``exclude``)."""
    rng = random.Random(seed)
    seen: set[str] = set(exclude or ())
    out: list[Example] = []
    attempts = 0
    while len(out) < n and attempts < 50 * n:
        attempts += 1
        ex = _MAKERS[task](rng, words)
        if ex.prompt in seen:
            continue
        seen.add(ex.prompt)
        out.append(ex)
    return out


@dataclass
class Dataset:
    """Train/test examples and the word split they were built from."""

    train: list[Example]
    test: list[Example]
    train_words: list[str]
    test_words: list[str]

    def by_task(self, split: str = "test") -> dict[str, list[Example]]:
        out: dict[str, list[Example]] = {t: [] for t in TASKS}
        for ex in getattr(self, split):
            out[ex.task].append(ex)
        return out


def generate_dataset(seed: int = 0, n_train: int = 3000, n_test: int = 300,
                     test_frac: float = 0.2) -> Dataset:
    """Build the full dataset: ``n_train`` / ``n_test`` examples per task.

    Letter-task test examples only use held-out words. Category items are shared (the
    category control task tests prompt combinations, not unseen vocabulary).
    """
    train_words, test_words = split_words(WORDS, seed, test_frac)
    train: list[Example] = []
    test: list[Example] = []
    for task in TASKS:
        tr = generate_task(task, train_words, n_train, f"{seed}-train-{task}")
        te = generate_task(task, test_words, n_test, f"{seed}-test-{task}",
                           exclude={e.prompt for e in tr})
        train.extend(tr)
        test.extend(te)
    random.Random(f"shuffle-{seed}").shuffle(train)
    return Dataset(train, test, train_words, test_words)


def wlt_corpus(dataset: Dataset, include_test_words: bool = False) -> list[str]:
    """Training text for the WLT BPE: train prompts, answers and the vocabulary words.

    With ``include_test_words`` the held-out words are added as bare vocabulary (as a
    real-world tokeniser would have seen them) but none of their task examples are.
    """
    lines: list[str] = []
    for ex in dataset.train:
        lines.append(ex.prompt)
        lines.append(ex.answer)
    words = list(dataset.train_words) + (list(dataset.test_words) if include_test_words else [])
    for w in words:
        lines.append(w)
        lines.append(" " + w)
    return lines


def router_training_set(dataset: Dataset, repeat_general: int = 10
                        ) -> tuple[list[str], list[int], list[int]]:
    """Labelled prompts for a learned router: (prompts, needs_clt, letter_answer)."""
    prompts = [ex.prompt for ex in dataset.train]
    needs = [int(ex.task in LETTER_TASKS) for ex in dataset.train]
    letter_answer = [int(ex.task in ("spell", "reverse", "nth")) for ex in dataset.train]
    for _ in range(repeat_general):
        for p in GENERAL_PROMPTS:
            prompts.append(p)
            needs.append(0)
            letter_answer.append(0)
    return prompts, needs, letter_answer

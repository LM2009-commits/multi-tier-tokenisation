"""Synthetic letter-level tasks over an embedded word list (nothing is downloaded).

Tasks: ``count`` ``spell`` ``reverse`` ``nth`` (letter-level) and ``category`` (a control
task that WLT should handle well). Words are split into disjoint train/test sets, so the
test set measures generalisation to words never seen in any training example.

Dataset version 1 is the original experiment and must never change. Version 2 balances the
``count`` answers (0, 1, 2, 3+), keeps ``SHOWCASE_PAIRS`` words out of training for every
seed, and adds named extra test sets (see ``generate_dataset``).
"""
from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field

TASKS = ("count", "spell", "reverse", "nth", "category")
LETTER_TASKS = ("count", "spell", "reverse", "nth")
COUNT_BUCKETS = ("0", "1", "2", "3+")

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

# Real words with one letter repeated 3+ times. The main list has too few for a balanced
# count task. Used by ``count`` only (data version 2), so the other tasks' words are unchanged.
_RAW_COUNT_EXTRA_WORDS = """
absentee adherence agreement bereavement beekeeper believe between cemetery centerpiece
cheerleader decrease delete deliberate delivery demented dependent depressed deserted
deceive designee detergent elevate eleven element employee engineer enemies envelope
evergreen excellent exercise expense experience extreme freelance genesis generate greenery
greeted independent interfere keepsake leverage merchandise meteor needless nevertheless
obedience overseer peacekeeper persevere perseverance pretense preserve presence reference
refugee relevance remedies repeated research resemble reserve residence respected retrieve
reverse revere sentence serene seventeen severe sleepless sneeze steeple stewardess teenager
telephone television temperature tennessee weekend whenever wherever bookkeeper bookkeeping
abracadabra alabama alaska anaconda arabian armada appearance avalanche bahamas balaclava
bazaar caravan casablanca catamaran cataract canada dalmatian database hawaiian jamaica
madagascar malaria mandala maharaja panama panorama paragraph pajamas salamander sahara
savanna tarantula baklava lasagna tanzania ramadan macadamia nicaragua guatemala alabaster
civilian initial invisible indivisible illinois inhibit limiting militia mississippi
primitive visiting definition ignition inspiring infinite initiate irritating timidity
virginia visibility simplicity politician civilization criticism dividing finishing insisting
bookstore cookbook coconut colorado commotion corridor doorknob locomotion monotone monopoly
protocol rococo schoolroom tomorrow voodoo woodwork overlook footnote motorboat bloodshot
cocoon moonglow foolproof bookworm doorstop horoscope honolulu toronto
assess assessment blessings business classes dressings discuss possess possession sassafras
seamstress senseless session sisters sunglasses stress suspense suspicious sustains witnesses
assist assistant assassin passport missions stainless restless useless
attempt attitude battlement constitute institute intermittent statement statistic stutter
tattoo toothpaste twitter attract attentive testament strategist substitute tentative
temptation tightest
barrier carrier terror terrier error horror warrior cranberry irregular rearrange recorder
refrigerate reporter railroad surrender terrarium territory narrator sorcerer remarry
roadrunner arrowroot carburetor
annoying antenna beginning cannon connection engineering grinning innocence innovation
nineteen nonsense nanny pennant spinning stunning tanning winning announce planning scanning
hillbilly illegal intelligent lollipop lullaby parallel skillful wallflower beautifully
fulfill hallelujah installment loyally locally legally bellflower
mammal mammoth maximum minimum momentum mummy memorandum commemorate millennium mummify
summertime
appropriate peppermint paperclip poppy puppet propeller preppy pepperoni hippopotamus philippine
baggage gaggle giggle goggles juggling luggage struggling smuggling wiggling begging bragging
digging dragging hugging jogging logging nagging tugging engaging gingerbread grudging
blubber bobbin hubbub babble bobble wobble bumblebee bubbly babbling
addendum daddy dividend dreaded dedicated meddled muddled padded nodded wedded shredded
studded dodged divided decided disdain
eccentric occurrence acceptance conscience cyclic succinct circumstance concentric
unusual uruguay tumultuous ubiquitous luxurious curriculum cumulus
fluffy fluffier effortful hitchhiker hitchhike highlight hierarchy knickknack powwow pizzazz
"""
# Pairs asked about in the ``showcase`` test set. Their words are never in version 2 training.
SHOWCASE_PAIRS = (
    ("strawberry", "r"), ("strawberry", "b"), ("strawberry", "z"), ("mississippi", "s"),
    ("mississippi", "i"), ("banana", "a"), ("banana", "n"), ("raspberry", "r"), ("cheese", "e"),
    ("pepper", "p"), ("scissors", "s"), ("bookkeeper", "e"), ("bookkeeper", "k"),
    ("tennessee", "e"), ("tennessee", "n"),
)

# ---------------------------------------------------------------------- paraphrase pools
# Held-out phrasing, written before any router change and never used to tune a router.
# None of the models or routers train on it. {l}/{w} are letter and word, {o} is an ordinal
# ("4th"/"fourth"), {n} is a position digit, {art}/{opts} as in CATEGORY_TEMPLATES.
PARAPHRASE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "count": (
        "In the word {w}, how many times can you find the letter {l}?",
        "What's the number of {l}'s in {w}?",
        "I need to know how many {l}'s the word {w} has.",
        "Does {w} contain any {l}'s? Tell me how many.",
        "How often is {l} used in the word {w}?",
        "Give me the count of the letter {l} in {w}.",
        "Tally up every {l} in {w}.",
        "How many occurrences of {l} does {w} have?",
        "Number of times {l} occurs in \"{w}\"?",
        "Look at {w}. How many {l}'s do you see?",
        "How many {l}'s does {w} have?",
        "Please count each {l} in '{w}'.",
    ),
    "spell": (
        "Can you spell {w} for me?",
        "What are the letters in {w}?",
        "Write out {w} one letter at a time.",
        "Break {w} into individual letters.",
        "I'd like the spelling of {w}, please.",
        "What is the correct spelling of the word {w}?",
        "List each letter of {w} in order.",
        "Show me how {w} is spelled.",
        "How is {w} spelled?",
        "Give me the letter-by-letter spelling of \"{w}\".",
        "Spell this word for me: {w}",
        "Sound out {w} letter by letter.",
    ),
    "reverse": (
        "What do you get if you write {w} back to front?",
        "Flip the letters of {w}.",
        "Can you write the word {w} backwards for me?",
        "Give me {w} in reverse.",
        "Read {w} from right to left.",
        "Turn {w} around so the last letter comes first.",
        "Print the reverse of {w}.",
        "What is the reverse of the word {w}?",
        "How would {w} look spelled in reverse order?",
        "Backwards, how does {w} read?",
        "Write \"{w}\" in reverse.",
        "Put the letters of {w} in the opposite order.",
    ),
    "nth": (
        "In {w}, which letter is {o}?",
        "Which letter comes {o} in the word {w}?",
        "Looking at {w}, what is its {o} letter?",
        "What's letter number {n} of {w}?",
        "Find the {o} letter of \"{w}\".",
        "{w}: what's the {o} character?",
        "Can you tell me the {o} letter in {w}?",
        "Which character is at position {n} in {w}?",
        "Of the letters in {w}, which is {o}?",
        "What letter is in position {n} of the word {w}?",
        "Pick out the {o} letter from {w}.",
        "Starting from the left, what is letter {n} in {w}?",
    ),
    "category": (
        "Out of {opts}, which one is {art}?",
        "Can you tell me which is {art}: {opts}?",
        "Choose {art} from these options: {opts}.",
        "Among {opts}, which would you call {art}?",
        "I have {opts}. Which is {art}?",
        "Select the one that is {art}: {opts}.",
        "Here are some options: {opts}. Which is {art}?",
        "Between {opts}, what is {art}?",
        "Which item is {art}: {opts}?",
        "Identify {art} in this list: {opts}.",
    ),
}
# General prompts that must not be routed. Same rule: never used for tuning or training.
ROUTER_NEGATIVES_TEST = (
    "Is there a spell that turns lead into gold in this story?",
    "Reverse the car into the garage.", "What does reverse psychology mean?",
    "How do I reverse a car safely?", "How do I reverse a linked list in Python?",
    "The wizard cast a spell on the village.",
    "Write a letter to my landlord about the heating.", "How many people live in Canada?",
    "How many a cappella groups are in Boston?", "Count the sheep in this field.",
    "Count my blessings.", "How many words are in this paragraph?",
    "What's the second law of thermodynamics?",
    "Who was the third president of the United States?", "Which character in Hamlet dies first?",
    "Name the fourth planet from the sun.", "What is the last word in the dictionary?",
    "Encrypt my hard drive with a strong password.", "How does Caesar salad get its name?",
    "Why was Julius Caesar killed?", "Why do people walk backwards to exercise?",
    "My spelling is bad; what should I do?", "Should I learn cursive?",
    "Give me a word that rhymes with orange.", "What's a synonym for happy?",
    "Write a haiku about strawberries.", "Is strawberry jam healthy?",
    "Summarise the book in three sentences.", "What does the word serendipity mean?",
    "Use the word banana in a sentence.", "Define the word gather.",
    "What language is the word kindergarten from?", "How do you pronounce quinoa?",
    "Recommend a spelling app for my child.", "Can you shift my meeting by 30 minutes?",
    "Rotate the image by 90 degrees.", "What is a caesar cipher?",
    "Who invented the Enigma cipher?",
)

# Phrasing the tagger router may learn from. Kept disjoint from PARAPHRASE_TEMPLATES.
ROUTER_TRAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "count": (
        "Find how many {l}'s appear in {w}.", "What is the count of {l} in {w}?",
        "In {w}, count the {l}'s.", "Total number of {l}'s in the word {w}?",
        "How many times is {l} in {w}?", "Can you count the letter {l} in {w}?",
        "The word {w} has how many {l}'s?", "For {w}, how many {l}'s are there?",
    ),
    "spell": (
        "Tell me how to spell {w}.", "Spell {w} please.", "Can you give me the spelling of {w}?",
        "What's the spelling of {w}?", "Write {w} letter by letter.", "Letters of {w}, please.",
        "How does one spell the word {w}?", "Separate the letters in {w}.",
    ),
    "reverse": (
        "Say {w} backwards.", "Write the letters of {w} in reverse.", "Reverse this word: {w}",
        "What is {w} in reverse?", "{w} backwards, please.", "Can you reverse {w}?",
        "Spell {w} from end to start.", "What does {w} look like reversed?",
    ),
    "nth": (
        "Tell me letter {n} of {w}.", "What's the {o} letter of {w}?",
        "For the word {w}, give the {o} letter.", "{w} has what as its {o} letter?",
        "Which letter is {o} in {w}?", "Get character {n} from {w}.",
        "The {o} letter in {w} is what?", "Position {n} of {w} holds which letter?",
    ),
}
ROUTER_NEGATIVES_TRAIN = (
    "Spell check my essay", "Reverse engineering is hard", "Put the car in reverse",
    "Drive in reverse carefully", "Reverse the order of this list", "Encrypt the file on my laptop",
    "What is the first letter of the alphabet?", "How many letters are in the alphabet?",
    "Which word comes first?", "Decrypt the message", "The last letter of my name is e",
    "How many hours are in a week?", "How many cats live in this house?",
    "Write three sentences about the sea.", "What does the word ambiguous mean?",
    "Use the word river in a sentence.", "Who is the main character of this film?",
    "What is the second largest city in Spain?", "Count the chairs in the room.",
    "Give me a word that means tired.", "Is a banana a good snack?", "Why do crabs walk sideways?",
    "Help me improve my spelling.", "What is the fastest animal on land?",
    "What is the tallest mountain in Africa?", "Tell me about the history of Rome.",
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


def build_count_extra_words() -> list[str]:
    """Sorted words (4-12 letters) with a letter repeated 3+ times, not in ``WORDS`` or templates."""
    blocked = _template_words() | {w for items in CATEGORIES.values() for w in items} | set(WORDS)
    words = {w for w in _RAW_COUNT_EXTRA_WORDS.split() if w.isalpha() and w.isascii()
             and 4 <= len(w) <= 12 and max(w.count(c) for c in set(w)) >= 3}
    return sorted(words - blocked)


COUNT_EXTRA_WORDS: list[str] = build_count_extra_words()
SHOWCASE_WORDS: frozenset[str] = frozenset(w for w, _ in SHOWCASE_PAIRS)


def split_words(words: list[str], seed: int | str, test_frac: float = 0.2
                ) -> tuple[list[str], list[str]]:
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
    gold_spans: tuple[tuple[int, int], ...] = ()  # filled for paraphrases (oracle router)


def count_bucket(count: int) -> str:
    """Balancing class of a true letter count: ``0``, ``1``, ``2`` or ``3+``."""
    return "3+" if count >= 3 else str(count)


def _fill(template: str, **values: str) -> tuple[str, dict[str, tuple[int, int]]]:
    """Format ``template`` and also return the character span of each placeholder."""
    out, spans, pos = [], {}, 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out.append(template[pos : m.start()])
        start = sum(map(len, out))
        out.append(values[m.group(1)])
        spans[m.group(1)] = (start, start + len(values[m.group(1)]))
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def _count_example(template: str, word: str, letter: str, with_spans: bool) -> Example:
    prompt, spans = _fill(template, l=letter, w=word)
    gold = tuple(sorted((spans["l"], spans["w"]))) if with_spans else ()
    return Example("count", prompt, str(word.count(letter)), (word,), letter, None, gold)


def balanced_count(words: list[str], n: int, seed: str, templates: tuple[str, ...] = COUNT_TEMPLATES,
                   with_spans: bool = False) -> list[Example]:
    """``n`` unique count examples with answers 0, 1, 2 and 3+ equally often.

    Within each bucket the letters take turns, so the letter alone says as little as possible
    about the answer (otherwise 3+ would be mostly ``e`` and 0 mostly ``z``).
    """
    rng = random.Random(seed)
    pool: dict[str, dict[str, list[tuple[str, str]]]] = {b: {} for b in COUNT_BUCKETS}
    for w in words:
        for letter in string.ascii_lowercase:
            pool[count_bucket(w.count(letter))].setdefault(letter, []).extend((w, t) for t in templates)
    out: list[Example] = []
    for i, bucket in enumerate(COUNT_BUCKETS):
        want = n // len(COUNT_BUCKETS) + (i < n % len(COUNT_BUCKETS))
        by_letter = pool[bucket]
        for letter in by_letter:
            rng.shuffle(by_letter[letter])
        supply = sum(map(len, by_letter.values()))
        if supply < want:
            raise ValueError(f"only {supply} unique count prompts for answer {bucket}, need {want}")
        taken = 0
        while taken < want:
            for letter in sorted(by_letter):
                if by_letter[letter] and taken < want:
                    word, template = by_letter[letter].pop()
                    out.append(_count_example(template, word, letter, with_spans))
                    taken += 1
    rng.shuffle(out)
    return out


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
    if len(out) < n:
        raise ValueError(f"only {len(out)} unique {task} prompts available, need {n}")
    return out


def generate_paraphrases(words: list[str], count_words: list[str], n: int, seed: str) -> list[Example]:
    """``n`` examples per task in phrasing no model or router trains on, with gold spans."""
    rng = random.Random(f"{seed}-paraphrase")
    out = balanced_count(count_words, n, f"{seed}-paraphrase-count", PARAPHRASE_TEMPLATES["count"], True)
    for _ in range(n):
        w, template = rng.choice(words), rng.choice(PARAPHRASE_TEMPLATES["spell"])
        prompt, spans = _fill(template, w=w)
        out.append(Example("spell", prompt, " ".join(w), (w,), gold_spans=(spans["w"],)))
    for _ in range(n):
        w, template = rng.choice(words), rng.choice(PARAPHRASE_TEMPLATES["reverse"])
        prompt, spans = _fill(template, w=w)
        out.append(Example("reverse", prompt, w[::-1], (w,), gold_spans=(spans["w"],)))
    for _ in range(n):
        w, template = rng.choice(words), rng.choice(PARAPHRASE_TEMPLATES["nth"])
        k = rng.randint(1, min(len(w), 10))
        prompt, spans = _fill(template, w=w, o=_ordinal(k, rng), n=str(k))
        out.append(Example("nth", prompt, w[k - 1], (w,), None, k, (spans["w"],)))
    for _ in range(n):
        cat = rng.choice(sorted(CATEGORIES))
        correct = rng.choice(CATEGORIES[cat])
        others = [c for c in sorted(CATEGORIES) if c != cat]
        options = [rng.choice(CATEGORIES[rng.choice(others)]) for _ in range(rng.randint(1, 2))] + [correct]
        rng.shuffle(options)
        template = rng.choice(PARAPHRASE_TEMPLATES["category"])
        prompt = template.format(art=CATEGORY_ARTICLE[cat], opts=", ".join(options))
        out.append(Example("category", prompt, correct))
    return out


@dataclass
class Dataset:
    """Train/test examples and the word split they were built from.

    ``count_*_words`` are the words the count task uses when that differs from the main split
    (version 2). ``extras`` holds named extra test sets.
    """

    train: list[Example]
    test: list[Example]
    train_words: list[str]
    test_words: list[str]
    count_train_words: list[str] = field(default_factory=list)
    count_test_words: list[str] = field(default_factory=list)
    extras: dict[str, list[Example]] = field(default_factory=dict)

    def by_task(self, split: str = "test") -> dict[str, list[Example]]:
        out: dict[str, list[Example]] = {t: [] for t in TASKS}
        for ex in getattr(self, split):
            out[ex.task].append(ex)
        return out


def generate_dataset(seed: int = 0, n_train: int = 3000, n_test: int = 300,
                     test_frac: float = 0.2, version: int = 1) -> Dataset:
    """Build the full dataset: ``n_train`` / ``n_test`` examples per task.

    Letter-task test examples only use held-out words. Category items are shared (the
    category control task tests prompt combinations, not unseen vocabulary).

    Version 2 also has the extra test sets ``count_natural`` (the skewed count distribution of
    version 1), ``showcase`` (``SHOWCASE_PAIRS`` in every count template) and ``paraphrase``.
    """
    if version == 2:
        return _generate_v2(seed, n_train, n_test, test_frac)
    if version != 1:
        raise ValueError(f"unknown dataset version {version}")
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


def _generate_v2(seed: int, n_train: int, n_test: int, test_frac: float) -> Dataset:
    main_train, main_test = split_words(WORDS, seed, test_frac)
    train_words = [w for w in main_train if w not in SHOWCASE_WORDS]
    test_words = sorted(set(main_test) | (SHOWCASE_WORDS & set(WORDS)))
    extra_train, extra_test = split_words([w for w in COUNT_EXTRA_WORDS if w not in SHOWCASE_WORDS],
                                          f"count-extra-{seed}", test_frac)
    count_train = sorted(set(train_words) | set(extra_train))
    count_test = sorted(set(test_words) | set(extra_test) | SHOWCASE_WORDS)

    train: list[Example] = []
    test: list[Example] = []
    for task in TASKS:
        if task == "count":
            tr = balanced_count(count_train, n_train, f"{seed}-train-count")
            te = balanced_count(count_test, n_test, f"{seed}-test-count")
        else:
            tr = generate_task(task, train_words, n_train, f"{seed}-train-{task}")
            te = generate_task(task, test_words, n_test, f"{seed}-test-{task}",
                               exclude={e.prompt for e in tr})
        train.extend(tr)
        test.extend(te)
    random.Random(f"shuffle-{seed}").shuffle(train)
    extras = {
        "count_natural": generate_task("count", test_words, n_test, f"{seed}-natural-count"),
        "showcase": [_count_example(t, w, l, False) for w, l in SHOWCASE_PAIRS for t in COUNT_TEMPLATES],
        "paraphrase": generate_paraphrases(test_words, count_test, n_test // 3 * 2, str(seed)),
    }
    return Dataset(train, test, train_words, test_words, count_train, count_test, extras)


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
    extra = dataset.count_train_words + (dataset.count_test_words if include_test_words else [])
    words += sorted(set(extra) - set(words))
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


def tagger_training_set(dataset: Dataset, n_per_task: int = 600, seed: int = 0) -> list[Example]:
    """Labelled positives for the tagger router: training words in training and router phrasing."""
    rng = random.Random(f"tagger-{seed}")
    out: list[Example] = []
    words = dataset.train_words
    for task, templates in ROUTER_TRAIN_TEMPLATES.items():
        for _ in range(n_per_task):
            w, template = rng.choice(words), rng.choice(templates)
            if task == "count":
                letter = rng.choice(w) if rng.random() < 0.75 else rng.choice(string.ascii_lowercase)
                out.append(_count_example(template, w, letter, True))
                continue
            k = rng.randint(1, min(len(w), 10))
            prompt, spans = _fill(template, w=w, o=_ordinal(k, rng), n=str(k))
            answer = {"spell": " ".join(w), "reverse": w[::-1], "nth": w[k - 1]}[task]
            out.append(Example(task, prompt, answer, (w,), gold_spans=(spans["w"],)))
    pool = [ex for ex in dataset.train if ex.task in LETTER_TASKS]
    for ex in rng.sample(pool, min(len(pool), 4 * n_per_task)):
        spans = [_word_span(ex.prompt, ex.target_words[0])]
        if ex.letter is not None:
            spans.append(_letter_span(ex.prompt, ex.letter))
        out.append(Example(ex.task, ex.prompt, ex.answer, ex.target_words, ex.letter, ex.position,
                           tuple(sorted(spans))))
    return out


def _word_span(prompt: str, word: str) -> tuple[int, int]:
    """Span of the last standalone occurrence of ``word`` (target words end the templates)."""
    return [m.span() for m in re.finditer(rf"(?<![a-z]){word}(?![a-z])", prompt)][-1]


def _letter_span(prompt: str, letter: str) -> tuple[int, int]:
    """Span of the letter in a count template: the first standalone, non-article occurrence."""
    for m in re.finditer(rf"(?<![a-zA-Z']){letter}(?![a-z])", prompt):
        return m.span()
    raise ValueError(f"letter {letter!r} not found in {prompt!r}")

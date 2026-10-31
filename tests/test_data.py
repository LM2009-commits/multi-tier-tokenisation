from mtt.data import (CATEGORIES, LETTER_TASKS, TASKS, WORDS, generate_dataset, split_words,
                      wlt_corpus)


def test_word_list_is_large_clean_and_unique():
    assert len(WORDS) >= 800 and len(set(WORDS)) == len(WORDS)
    assert all(w.isalpha() and w.islower() and w.isascii() for w in WORDS)
    assert not set(WORDS) & {w for ws in CATEGORIES.values() for w in ws}


def test_split_is_disjoint_and_deterministic():
    tr, te = split_words(WORDS, 0)
    assert not set(tr) & set(te) and set(tr) | set(te) == set(WORDS) and len(te) >= 150
    assert split_words(WORDS, 0) == (tr, te)
    assert split_words(WORDS, 1) != (tr, te)


def test_dataset_sizes_and_word_disjointness():
    ds = generate_dataset(0, 120, 30)
    for split, n in (("train", 120), ("test", 30)):
        by = ds.by_task(split)
        assert all(len(by[t]) == n for t in TASKS)
    train_words = {w for e in ds.train for w in e.target_words}
    test_words = {w for e in ds.test for w in e.target_words}
    assert train_words <= set(ds.train_words) and test_words <= set(ds.test_words)
    assert not train_words & test_words
    assert not {e.prompt for e in ds.train} & {e.prompt for e in ds.test}


def test_answers_are_correct():
    for e in generate_dataset(1, 150, 40).test + generate_dataset(1, 150, 40).train:
        w = e.target_words[0] if e.target_words else None
        if e.task == "count":
            assert e.answer == str(w.count(e.letter))
        elif e.task == "spell":
            assert e.answer == " ".join(w)
        elif e.task == "reverse":
            assert e.answer == w[::-1]
        elif e.task == "nth":
            assert e.answer == w[e.position - 1]
        else:
            assert e.answer in e.prompt


def test_category_answer_belongs_to_category():
    for e in generate_dataset(0, 100, 20).train:
        if e.task == "category":
            cats = [c for c, items in CATEGORIES.items() if e.answer in items]
            assert len(cats) == 1


def test_deterministic_given_seed():
    assert generate_dataset(3, 50, 10).train == generate_dataset(3, 50, 10).train
    assert generate_dataset(3, 50, 10).train != generate_dataset(4, 50, 10).train


def test_templates_vary():
    ds = generate_dataset(0, 200, 20)
    for task in TASKS:
        shapes = {e.prompt.split()[0] + str(len(e.prompt.split())) for e in ds.train if e.task == task}
        assert len(shapes) >= 3


def test_wlt_corpus_test_words_only_when_requested():
    ds = generate_dataset(0, 60, 10)
    probe = ds.test_words[0]
    assert probe not in " ".join(wlt_corpus(ds)).split()
    assert probe in wlt_corpus(ds, include_test_words=True)

# Multi Tier Tokenisation (MTT)

A small, tested prototype and experiment for **Multi Tier Tokenisation**: use a normal subword
tokeniser (**WLT**) for most text, and a **character-level tokeniser (CLT)** only for the spans
that need letter-level precision (spelling, counting letters, reversing...). A **router**
decides which spans go through CLT; they are wrapped in `<CLT> ... </CLT>` inside an otherwise
normal WLT sequence.

```
"How many r's are in strawberry?"      (strawberry is a held-out test word)
  WLT-only :  How | many | r | 's | are | in | st | ra | w | berry | ?      (subword pieces; letters stay hidden)
  MTT      :  How many <CLT> r </CLT>'s are in <CLT> s t r a w b e r r y </CLT>?
```

The WLT line is the real output of the project's BPE for that prompt. The CLT also has a
human-readable numeric view (`to_display("how are you")` -> `"8.15.23, 1.18.5, 25.15.21"`), but
internally every character is **one token id**.

## Results at a glance

Full write-up with method, analysis and limitations: **[REPORT.md](REPORT.md)**.

Same 5.3M-parameter transformer, same data, 5000 steps, **three seeds** (each seed also changes the
train/test word split); accuracy on **held-out words**, 300 test prompts per task, exact match,
mean ± SD across seeds:

| task | word-only | char-only | **MTT** |
|---|---|---|---|
| spell | 3.9 ± 0.4% | 85.6 ± 5.9% | **97.8 ± 1.9%** |
| reverse | 1.2 ± 0.7% | 79.7 ± 3.2% | **98.0 ± 0.6%** |
| nth letter | 25.2 ± 5.9% | 46.2 ± 5.3% | **91.2 ± 6.0%** |
| count (balanced 0/1/2/3+, chance 25%) | 35.8 ± 2.6% | 41.4 ± 2.3% | **57.2 ± 5.9%** |
| category (control) | 99.3 ± 0.9% | 60.8 ± 4.0% | **99.8 ± 0.2%** |
| mean sequence length (training mix) | 1.00x | 3.34x | 1.66x |

What holds up: MTT spells, reverses and indexes letters of unseen words, beats character-only in every
seed at half its sequence length, and leaves ordinary prompts unchanged. Doubling character-only's
training steps doesn't close the gap. What doesn't: **long words** (MTT spells every 4-7 letter test
word but none of the 10-letter ones, including strawberry, because training words are rarely that long), **counting is better but not solved** (a model
that only knows whether the letter is present scores 50%), and **nothing works on paraphrased prompts**,
even with perfect routing, because the models only ever saw 4-5 phrasings per task. Letting the model
choose its own character spans (`selfesc`) fails. Caveats: toy model, synthetic templates.

## Install and run

Python 3.11+. Always run from the project root (the sub-package `mtt/tokenizers` shares its name
with the HuggingFace library; running from the root keeps imports unambiguous).

```bash
pip install -r requirements.txt              # or requirements-lock.txt for the exact versions used
python -m pytest                             # all tests
python -m mtt.train --mode mtt --smoke       # ~1 min end-to-end check of train + save
python -m mtt.eval --lengths-only            # sequence-length table, needs no training
```

**The full experiment** (16 runs, data version 2) is one command. It runs jobs in parallel, skips
finished runs, resumes interrupted ones from checkpoints, and then evaluates and analyses everything:

```bash
python scripts/run_experiments.py --dry-run      # list the runs
python scripts/run_experiments.py --smoke        # tiny version of all of it (~6 min)
python scripts/run_experiments.py                # the real thing: about 16 h on a 20-core CPU
python scripts/run_experiments.py --bench --threads 20 --jobs 1   # time each mode alone
```

Single runs and steps:

```bash
python -m mtt.train --mode mtt --data-version 2 --seed 1        # run name: mtt_v2_s1
python -m mtt.train --mode clt_only --data-version 2 --resume   # continue after an interruption
python -m mtt.eval                                # every run in runs/ -> results/
python -m mtt.eval --router-compare --data-version 2
python -m mtt.analyze                             # analysis.md, loss and learning curves
python -m mtt.ask "How many r's are in strawberry?"
```

Run names encode every non-default setting (`clt_only_v2_10k`, `wlt_only_v2_wltall`,
`mtt_learned_v2`...), and training refuses to overwrite a finished run unless you pass `--force`.
`bash scripts/run_all.sh` still reproduces the original version-1 experiment, whose results are kept
in `results/v1/`.

## Layout

```
mtt/tokenizers/wlt.py   HF byte-level BPE (vocab 2000), trained on the project corpus; ByteWLT test stand-in
mtt/tokenizers/clt.py   a-z, 0-9, punctuation, space, <CAP>; to_display() numeric view
mtt/tokenizers/mtt.py   combined encode/decode with <pad> <bos> <eos> <unk> <sep> <CLT> </CLT>
mtt/router.py           RuleRouter, LearnedRouter, TaggerRouter, OracleRouter (gold spans)
mtt/data.py             word lists, train/test split, task templates, data versions 1 and 2, paraphrase sets
mtt/pipeline.py         per-mode encoding of examples, answer extraction, sequence lengths
mtt/model.py            decoder-only transformer + greedy decoding
mtt/train.py            training CLI (checkpoints, resume, learning curves, provenance)
mtt/eval.py             evaluation, multi-seed aggregation, tables, plots, router comparison
mtt/analyze.py          bootstrap differences, counting baselines, error tables, curves
mtt/ask.py              ask trained runs a question
scripts/                run_experiments.py (version 2 matrix), run_all.sh (version 1)
results/                tables, figures, per-example predictions, thinned training logs; results/v1/ = first experiment
REPORT.md, docs/figures report and its figures
tests/                  pytest suite
```

## How it works

**Id layout.** `0-6` specials, then WLT ids, then CLT ids. A CLT span is
`<CLT> c1 ... cn </CLT>`; capitals are `<CAP>` + lowercase letter. `MTTTokenizer.decode` reconstructs
the text exactly (strict mode validates the markers; lenient mode is used on model output).

**Routers.** Any object with `route(prompt) -> Route` works.
- `RuleRouter`: regexes for letter counting, spelling, reversal, nth letter and simple ciphers. They
  extract the *target word(s)* as character spans, never the whole prompt. It ignores phrases such as
  "spell check my essay" or "put the car in reverse".
- `LearnedRouter`: two hashed character-n-gram logistic regressions; routes the words that are rare in
  the training prompts.
- `TaggerRouter`: scores each word by masking it and asking a classifier whether the prompt still looks
  like a letter task; routes the single best word (and, for counting, the letter).
- `OracleRouter`: the gold spans, used in evaluation to separate router errors from model errors.

**Tasks** (`data.py`): `count`, `spell`, `reverse`, `nth` (letter-level) and `category` (control,
WLT-friendly). Words are split 80/20 into disjoint train/test sets, so test prompts only contain
words that never appeared in any training prompt. 3000 train and 300 test examples per task,
4-5 templates per task. **Data version 2** balances the count answers (0, 1, 2, 3+ equally often,
using 357 extra repeated-letter words for counting only), keeps showcase words such as strawberry and
mississippi out of training for every seed, and adds three test sets: `count_natural` (the old skewed
distribution), `showcase` and `paraphrase` (new phrasings with gold answers, never used to tune anything).

**Modes** (same architecture, data, steps, batch order and initial weights; the model always
has the full MTT vocabulary):

| mode | prompt | answer |
|---|---|---|
| `wlt_only` | WLT | WLT |
| `clt_only` | all CLT | all CLT |
| `mtt` | router-selected spans in CLT | CLT for letter answers (spell, reverse, nth), WLT for the count digit |
| `selfesc` | WLT only | model emits `<CLT> word </CLT>` itself, then the answer as in `mtt` |
| `clt_marked` (ablation) | all CLT, target word also wrapped in markers | all CLT |
| `mtt_nomark` (ablation) | as `mtt`, without the `<CLT>` `</CLT>` markers | as `mtt`, without markers |

Sequence layout is `<bos> prompt <sep> answer <eos>`; loss is on the answer part only. Evaluation is
greedy decoding and **exact match of the decoded answer string**.

## Decisions I made where the spec was open

- **Letter spans in counting.** For "How many r's ...", both the word *and* the single letter are
  routed to CLT so they share an alphabet. `--router rule_words` is the control without letter spans
  (it made no measurable difference).
- **Answer encoding.** The router also says whether the answer is letters; spell/reverse/nth answers
  are CLT-encoded, the count digit stays WLT.
- **WLT corpus.** By default the BPE is trained on training prompts, answers and *training* words only,
  so test words are split into subwords it has seen. `--wlt-corpus all` also shows it the held-out
  words as bare vocabulary (closer to a real pre-trained tokeniser) without any task examples.
- **Category task** is a control: its test examples use unseen *prompt combinations* but the same item
  vocabulary (items cannot be held out, or any WLT model would fail on never-trained tokens).
- **Byte-level BPE** so the WLT round trip is exactly lossless for any text.
- **Statistics.** Multi-seed configurations are reported as mean ± SD across seeds; single runs as a
  95% Wilson interval. Differences between modes use a paired bootstrap that resamples test *words*,
  because each word appears in about five test prompts and per-prompt intervals would be too narrow.

## Router findings (measured, no model needed)

`python -m mtt.eval --router-compare --data-version 2`, on fresh paraphrases (800 letter-task prompts
in about 12 new phrasings per task, seed 0) and 38 tricky general prompts such as "Reverse engineering is hard":

| router | target recall | word precision | false alarms (tricky) | in-template accuracy |
|---|---|---|---|---|
| rule | 14.6% | 68.4% | 3/38 | 100% |
| learned | 94.6% | 39.9% | 27/38 | 100% |
| tagger | 74.3% | 75.1% | 20/38 | 99.7% |

None is good enough on prompts it wasn't built for. The rule router is precise but misses most new
phrasings; the learned and tagger routers find the target more often but fire on most of the tricky
general prompts. The paraphrases were written before any router change, and no router was tuned on them.

## Known limitations

- The models only understand the training phrasings. A paraphrased question gets the wrong *kind* of
  answer, whatever the router does.
- CLT supports ASCII only (no newline); other characters raise inside a CLT span.
- Generation has no KV cache (fine for ~1500 short prompts, slow for large evaluations).
- Toy scale: a 5M-parameter model on synthetic templates says little about large pretrained models.
- `.github/workflows/ci.yml` runs the tests on push, but it hasn't been run yet (no remote).

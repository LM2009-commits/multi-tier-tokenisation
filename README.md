# Multi Tier Tokenisation (MTT)

A small, tested prototype and experiment for **Multi Tier Tokenisation**: use a normal subword
tokeniser (**WLT**) for most text, and a **character-level tokeniser (CLT)** only for the spans
that need letter-level precision (spelling, counting letters, reversing...). A **router**
decides which spans go through CLT; they are wrapped in `<CLT> ... </CLT>` inside an otherwise
normal WLT sequence.

```
"How many r's are in strawberry?"
  WLT-only :  How | many | r | 's | are | in | straw | berry | ?          (model never sees letters)
  MTT      :  How many <CLT> r </CLT> 's are in <CLT> s t r a w b e r r y </CLT> ?
```

The CLT also has a human-readable numeric view (`to_display("how are you")` ->
`"8.15.23, 1.18.5, 25.15.21"`), but internally every character is **one token id**.

## Results at a glance

Full write-up with method, analysis and limitations: **[REPORT.md](REPORT.md)**.

Same 5.3M-parameter transformer, same data, 5000 steps, one seed; accuracy on **held-out words**
(300 test prompts per task, exact match):

| task | word-only | char-only | **MTT** |
|---|---|---|---|
| spell | 2.7% | 83.3% | **100%** |
| reverse | 2.3% | 77.3% | **99.7%** |
| nth letter | 24.0% | 38.7% | **93.3%** |
| category (control) | 98.0% | 63.7% | **100%** |
| count | 65.0% | 66.0% | 71.7% |
| training time (CPU) | 1.0x | 3.6x | 1.8x |

MTT clearly helps spelling, reversing and indexing letters in unseen words at about 1.8x the training
cost (character-only: 3.6x). **Counting is unresolved:** all modes score close to always answering "1"
(64%), and the generated counting data was heavily skewed towards that answer, so this test cannot say
whether MTT helps with "how many r's are in strawberry". Caveats: one seed, synthetic template prompts,
tiny model, and a rule-based router that is only reliable on in-template phrasing.

## Install and run

Python 3.11+. Always run from the project root (the sub-package `mtt/tokenizers` shares its name
with the HuggingFace library; running from the root keeps imports unambiguous).

```bash
pip install -r requirements.txt

python -m pytest                                   # all tests (HF-BPE and torch tests run here)
python -m mtt.train --mode mtt --smoke             # ~1 min end-to-end check of train + save
python -m mtt.eval --lengths-only                  # sequence-length table, needs no training

# The experiment (same seed, data, model, steps for every mode):
python -m mtt.train --mode wlt_only
python -m mtt.train --mode clt_only
python -m mtt.train --mode mtt
python -m mtt.eval --runs wlt_only clt_only mtt    # writes results/results.md, accuracy.png, seqlen.png

# Stretch goals:
python -m mtt.train --mode selfesc                 # model emits <CLT> word </CLT> itself
python -m mtt.train --mode mtt --router learned    # learned router instead of rules
python -m mtt.train --mode mtt --router rule_words --name mtt_rule_words   # control, see below
python -m mtt.eval --router-compare                # rule vs learned router (no model needed)
python -m mtt.eval --runs wlt_only clt_only mtt mtt_rule_words mtt_learned selfesc

bash scripts/run_all.sh                            # everything (STRETCH=1 for the stretch goals)
```

The default model is 6 layers, d_model 256, 4 heads (5,303,552 parameters), 5000 steps of batch 64.
Measured on one CPU (Intel i7-14700T): `wlt_only` 59 min, `mtt` 1 h 47 min, `clt_only` 3 h 30 min.
Use `--steps 2000` for a quicker, weaker run. `results/predictions_<run>.jsonl` lists every prompt, gold
answer and prediction for error analysis. `python -m mtt.ask "..."` queries the trained runs.

## Layout

```
mtt/tokenizers/wlt.py   HF byte-level BPE (vocab 2000), trained on the project corpus; ByteWLT test stand-in
mtt/tokenizers/clt.py   a-z, 0-9, punctuation, space, <CAP>; to_display() numeric view
mtt/tokenizers/mtt.py   combined encode/decode with <pad> <bos> <eos> <unk> <sep> <CLT> </CLT>
mtt/router.py           Router protocol, RuleRouter (v1), LearnedRouter (stretch)
mtt/data.py             embedded word list (1117 words: 894 train / 223 held-out), train/test word split, task generators
mtt/pipeline.py         per-mode encoding of examples, answer extraction, sequence lengths  (added)
mtt/model.py            decoder-only transformer + greedy decoding
mtt/train.py            training CLI          mtt/eval.py   evaluation, tables, plots, router comparison
mtt/ask.py              ask trained runs a question
REPORT.md, docs/figures proof-of-concept report and its figures; results/ has tables and per-example predictions
tests/                  pytest suite          scripts/run_all.sh
```

## How it works

**Id layout.** `0-6` specials, then WLT ids, then CLT ids. A CLT span is
`<CLT> c1 ... cn </CLT>`; capitals are `<CAP>` + lowercase letter. `MTTTokenizer.decode` reconstructs
the text exactly (strict mode validates the markers; lenient mode is used on model output).

**Router (rule-based v1).** Regexes for letter counting, spelling, reversal, nth letter and simple
ciphers. They extract the *target word(s)* as character spans, never the whole prompt. Category
questions and other general prompts are not routed (tested explicitly). Any object with
`route(prompt) -> Route` can replace it; `LearnedRouter` is one.

**Tasks** (`data.py`): `count`, `spell`, `reverse`, `nth` (letter-level) and `category` (control,
WLT-friendly). Words are split 80/20 into disjoint train/test sets, so test prompts only contain
words that never appeared in any training prompt. 3000 train and 300 test examples per task,
5 templates per task.

**Modes** (same architecture, data, steps, batch order and initial weights; the model always
has the full MTT vocabulary):

| mode | prompt | answer |
|---|---|---|
| `wlt_only` | WLT | WLT |
| `clt_only` | all CLT | all CLT |
| `mtt` | router-selected spans in CLT | CLT for letter answers (spell, reverse, nth), WLT for the count digit |
| `selfesc` (stretch) | WLT only | model emits `<CLT> word </CLT>` itself, then the answer as in `mtt` |

Sequence layout is `<bos> prompt <sep> answer <eos>`; loss is on the answer part only. Evaluation is
greedy decoding and **exact match of the decoded answer string**.

## Decisions I made where the spec was open

- **Letter spans in counting.** For "How many r's ...", both the word *and* the single letter are
  routed to CLT so they share an alphabet. `--router rule_words` is the control without letter spans.
- **Answer encoding.** The router also says whether the answer is letters; spell/reverse/nth answers
  are CLT-encoded, the count digit stays WLT.
- **WLT corpus.** By default the BPE is trained on training prompts, answers and *training* words only,
  so test words are split into subwords it has seen. `--wlt-corpus all` also shows it the held-out
  words as bare vocabulary (closer to a real pre-trained tokeniser) without any task examples.
- **Category task** is a control: its test examples use unseen *prompt combinations* but the same item
  vocabulary (items cannot be held out, or any WLT model would fail on never-trained tokens).
- **Byte-level BPE** so the WLT round trip is exactly lossless for any text.
- **Learned router spans.** It uses two hashed char-n-gram logistic regressions (needs CLT / letter
  answer) trained on training prompts plus a few general prompts. Its spans are the words that are rare
  in the training prompts, and it does not route single letters.
- Single seed per run by default, so there are no error bars across seeds; the table shows 95% Wilson
  intervals over the 300 test prompts per task. For a convincing comparison run several `--seed`s.

## Router findings (measured, no model needed)

`python -m mtt.eval --router-compare` (seed 0, run in the authoring sandbox):

| router | decision acc. (held-out) | target found (held-out) | spurious spans/positive (held-out) | decision acc. (OOD) | target found (OOD positives) | spurious spans/positive (OOD) |
|---|---|---|---|---|---|---|
| rule | 1.000 | 1.000 | 0.00 | 12/16 | 3/8 | 0.12 |
| rule_words | 1.000 | 1.000 | 0.00 | 12/16 | 3/8 | 0.12 |
| learned | 1.000 | 1.000 | 0.00 | 14/16 | 8/8 | 1.75 |

"OOD" is 16 hand-written paraphrases that are not in the templates (`OOD_ROUTER_CASES`; only 8 positives,
so treat this as anecdotal). What it shows: both routers are perfect on the in-template data, which is
expected since the templates were written by the same person; the rule router is brittle to paraphrase
(and once fired on the wrong word: "reverse order" -> `order`); the learned router finds the target word
every time but also routes filler words (about 1.75 per prompt) and has 2 false positives on general
questions. Neither is good enough to call solved.

## Results

See the table above and [REPORT.md](REPORT.md) for the full tables, cost measurements, error analysis
and limitations. Raw numbers: `results/results.md`, `results/results.json`, `results/predictions_*.jsonl`.

## Known limitations

- Rule router is regex-based: brittle to paraphrase, may fire on text like "spell check my essay".
- CLT supports ASCII only (no newline); other characters raise inside a CLT span.
- Generation has no KV cache (fine for ~1500 short prompts, slow for large evaluations).
- Toy scale: a 5M-parameter model on synthetic templates says little about large pretrained models.

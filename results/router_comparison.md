## Router comparison

### Fresh paraphrases and negatives (never used to tune or train any router)

Target-word recall and precision on paraphrased letter-task prompts; false alarms on 38 tricky general prompts.

| router | fired on letter prompts | target recall | word precision | false alarms (tricky) |
|---|---|---|---|---|
| rule | 0.214 | 0.146 | 0.684 | 3/38 |
| rule_words | 0.214 | 0.146 | 0.684 | 3/38 |
| learned | 0.946 | 0.946 | 0.399 | 27/38 |
| tagger | 0.989 | 0.743 | 0.751 | 20/38 |

### In-template prompts and the original 16 paraphrases

The 16 no longer count as unseen: the rule router was revised after inspecting its behaviour on them.

| router | decision acc. (held-out) | target found (held-out) | spurious spans/positive (held-out) | decision acc. (OOD) | target found (OOD positives) | spurious spans/positive (OOD) |
|---|---|---|---|---|---|---|
| rule | 1.000 | 1.000 | 0.00 | 12/16 | 4/8 | 0.00 |
| rule_words | 1.000 | 1.000 | 0.00 | 12/16 | 4/8 | 0.00 |
| learned | 1.000 | 1.000 | 0.00 | 14/16 | 8/8 | 1.75 |
| tagger | 0.997 | 0.997 | 0.00 | 16/16 | 7/8 | 0.12 |

Held-out prompts: 1200 letter-task and 300 category prompts (unseen words). `GENERAL_PROMPTS` (28) are in the learned routers' training data, so they aren't scored here.

**rule OOD misses (4):** “Could you tally the number of e's inside the word elephant?”; “Tell me which letter sits in fourth place in garden.”; “Give me the letters of banana one at a time.”; “Count how often the letter o shows up in october.”

**rule_words OOD misses (4):** “Could you tally the number of e's inside the word elephant?”; “Tell me which letter sits in fourth place in garden.”; “Give me the letters of banana one at a time.”; “Count how often the letter o shows up in october.”

**learned OOD misses (2):** “How many minutes are in a day?”; “What is the longest river in Africa?”

**tagger OOD misses (0):** none

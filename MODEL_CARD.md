# Model card

## Task

Given one Russian word, predict its part of speech: one of 18 classes, the
17 OpenCorpora parts of speech with verbs split by mood (`IMPR` for the
imperative, `VERB` for the indicative). The consumer's question is whether
a word is an action, meaning an infinitive or an imperative.

## Teacher

OpenCorpora dictionary revision 417150, read through `pymorphy3` and
`pymorphy3-dicts-ru`. It has 3,064,812 distinct word forms from 391,778
lemmas. Each form gets every part of speech the dictionary allows, ranked
by pymorphy3's corpus-derived probability. The first is the teacher's
answer (strict). Any of them is acceptable (lenient).

All numbers below measure agreement with this teacher, not linguistic truth.
The teacher's top answer is context-free, so «печь» is `NOUN` even in
«печь пирог».

## Data

- Split by lemma with crc32(lemma) mod 10. Test gets 301,341 forms of
  unseen lemmas. 7,713 forms whose lemmas fall on both sides are dropped.
- Live-text frequencies come from FrequencyWords `ru_50k` (OpenSubtitles
  2018). «ё» folds into «е» in the model input and the frequency lookup.

## Model

- Input. The word, lowercased, «ё» to «е», reversed and cut to 16
  characters, so position 0 is the last letter. 34 known characters plus
  pad and unknown.
- Body. Embedding of 16. Three kernel-3 convolutions (dilations 1, 1, 2),
  the second and third residual, ReLU.
- Head. The first 6 positions verbatim plus a max over the word, then one
  hidden ReLU layer and 18 logits.
- Sizes. `small` is width 48, hidden 64. `big` is width 96, hidden 128.
- Export. Per-tensor symmetric int8 weights, f32 biases.

## Training

AdamW with weight decay 1e-4 and a one-cycle schedule peaking at 3e-3.
Batch 1,024, seed 0. `small` trains 6 epochs, `big` 12 (its curve is flat
from epoch 7). Half of every batch is drawn uniformly from the dictionary
and half by live-text frequency.

The frequency half is what makes function words work. With the loss
weighted by log frequency instead, the same network agreed with the teacher
on only 70.3% of live text. «в», «и» and «что» are three rows in three
million, and it called them `NOUN`. Sampling half the batch by frequency
reached 99.4% with the same parameters. A whole-word
hash embedding (+34k parameters) and an exception list on top each added
under 0.3 points and were dropped.

Wall time on an Apple M1 with MPS is 4.6 min for `small` and 17.4 min for
`big`.

## Results

Shipped models train on every lemma, so their test numbers are in-sample.
Generalisation numbers come from heldout twins trained with
`--train-on train`. Agreement is strict unless noted. All numbers are
from one seed.

| | small | big |
|---|---|---|
| Live text, float | 99.43% | 99.66% |
| Live text, int8 (shipped) | 99.42% | 99.66% |
| Live text, lenient, int8 | 99.63% | 99.69% |
| Live text, action F1, int8 | 99.82% | 100.00% |
| Unseen lemmas (heldout twin) | 96.99% | 97.26% |
| Unseen lemmas, action F1 (heldout twin) | 97.13% | 98.03% |
| Dictionary test split, int8 (in-sample) | 97.54% | 99.16% |

Baselines on unseen lemmas:

| | strict | action F1 |
|---|---|---|
| Hand rules (-ть, -ти, -чь, imperative endings) | 28.90% | 20.16% |
| Suffix table, longest of ≤5 letters, 186,972 entries | 95.23% | 97.38% |
| `small`, heldout twin | 96.99% | 97.13% |
| `big`, heldout twin | 97.26% | 98.03% |

On unseen lemmas weighted by frequency every method scores 34 to 37%. The
held-out tenth contains function words such as pronouns and prepositions,
which no model can guess from their shape. The shipped models see every
lemma, which is why their live-text numbers are high.

The Rust forward pass matches PyTorch on the int8 weights. Logits agree
within 1e-3 on the golden files, and labels match on all 345,083 test and
live words for both models.

## Limitations

- Context-free. Homonyms get one answer: «стекло», «печь», «чем», «перед».
  Context needs a sentence-level corpus such as UD SynTagRus. That is
  future work.
- The remaining unseen-lemma errors are mostly the teacher's own fuzzy
  borders, ADJF against PRTF and NOUN against ADJF. Rare names shaped like
  imperatives («ермолай») are the main source of false "action" answers.
- Live-text weights come from film subtitles, so they favour spoken
  Russian.

# gpu-verbum

[![CI](https://img.shields.io/github/actions/workflow/status/makekosmos/gpu-verbum/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/makekosmos/gpu-verbum/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20CC%20BY--SA%204.0-blue?style=flat-square)](NOTICE.md)
[![MSRV](https://img.shields.io/badge/rust-1.80%2B-orange?style=flat-square&logo=rust)](Cargo.toml)
[![Model](https://img.shields.io/badge/model-40%20KB%20int8-informational?style=flat-square)](MODEL_CARD.md)
[![Dependencies](https://img.shields.io/badge/dependencies-0-success?style=flat-square)](Cargo.toml)

Part of speech for a single Russian word, from a 40 KB model with no
dependencies.

```rust
use verbum::{Pos, Tagger};

let tagger = Tagger::small();
assert_eq!(tagger.tag("позвонить"), Some(Pos::Infn));
assert_eq!(tagger.tag("купи"), Some(Pos::Impr));
assert_eq!(tagger.tag("молоко"), Some(Pos::Noun));

// Words no dictionary has.
assert_eq!(tagger.tag("задеплоить"), Some(Pos::Infn));
assert!(tagger.tag("пингануть").is_some_and(Pos::is_action));
```

## Why

Mundus Agenda wants task titles to start with an action: «позвонить маме»,
«купи молоко», not «мама» or «молоко». That needs the part of speech of the
first word, offline, in microseconds, and it has to cope with the slang
people actually type («задеплоить», «отревьюить»).

A dictionary such as pymorphy3 answers the question, but its Russian data
alone is 15 MB and it needs Python. Hand-written suffix rules are small but wrong
most of the time (29% on unseen words, see below). gpu-verbum sits between
them. A character CNN learned the dictionary's answers and ships as a
40 KB int8 file that is compiled into your binary.

## Install

The crate is not on crates.io yet.

```toml
[dependencies]
gpu-verbum = { git = "https://github.com/makekosmos/gpu-verbum" }
```

The library is imported as `verbum`. The `small` model is embedded by
default. Add `features = ["big"]` for the larger model.

## API

| | |
|---|---|
| `Tagger::small()`, `Tagger::big()` | The embedded models, loaded once on first use. |
| `Tagger::from_bytes(&[u8])` | Load a `.vrb` file yourself. |
| `tagger.tag(word) -> Option<Pos>` | The most likely part of speech. `None` for an empty word. |
| `tagger.scores(word) -> Option<[f32; 18]>` | Raw logits, indexed like `Pos::ALL`. |
| `Pos::is_action()` | True for an infinitive or an imperative. |
| `Pos::tag()`, `Pos::from_tag()` | The OpenCorpora tag string, e.g. `"INFN"`. |

`Pos` has 18 variants: the 17 OpenCorpora parts of speech, with the verb
split by mood into `Verb` (indicative) and `Impr` (imperative). Input is
lowercased and «ё» is read as «е».

## Accuracy

All numbers are agreement with the teacher, the OpenCorpora dictionary via
pymorphy3. They are not linguistic truth.

| | `small` | `big` |
|---|---|---|
| Parameters | 39,570 | 149,218 |
| File size (int8) | 40,482 bytes | 150,754 bytes |
| Live text | 99.42% | 99.66% |
| Live text, "is an action" F1 | 99.82% | 100.00% |
| Lemmas never seen in training | 96.99% | 97.26% |
| One word, Apple M1, one core | 26 µs | 87 µs |

"Live text" weights each word by how often it occurs in OpenSubtitles 2018,
so it is the accuracy a reader of ordinary Russian meets. "Never seen"
comes from a twin of each model trained with a tenth of the lemmas held
out and scored only on them.

On the same unseen lemmas, hand-written ending rules score 28.90% and a
186,972-entry suffix table scores 95.23%. `big` buys a third of a
percentage point over `small` for four times the bytes, so `small` is the
default.

## Limitations

- **One word, no context.** «печь» is a noun even in «печь пирог». Homonyms
  such as «стекло», «чем» and «перед» always get the same answer.
- **The teacher's borders.** Most remaining errors on unseen words are
  full adjective against full participle and noun against adjective, where
  the dictionary itself is fuzzy.
- **Names that look like commands.** Rare first names shaped like
  imperatives («ермолай») are the main source of false "action" answers.
- **Spoken register.** Frequencies come from film subtitles.

[`MODEL_CARD.md`](MODEL_CARD.md) has the full method, every number and
the baselines.

## How it works

The word is lowercased, reversed and cut to 16 characters, so the model
reads it from the last letter, where Russian keeps its endings. An
embedding feeds three small convolutions. The first six positions plus a
max over the word go to one hidden layer and 18 outputs.

Training samples half of each batch uniformly from the 3 million word forms
in the dictionary and half by real-text frequency. Without the frequency
half, the model called «в», «и» and «что» nouns, because they are three
rows among three million.

The Rust forward pass is checked against PyTorch. Logits agree within 1e-3
on the golden files in `tests/`, and labels match on all 345,083 test and
live words for both models.

## Repository

- `src/` holds the inference crate: the `.vrb` format and the forward pass.
- `models/` holds the shipped weights and their metrics.
- `tests/golden.rs` compares Rust logits with PyTorch on about 1,000 words per model.
- `examples/parity.rs` checks every test and live word and times the tagger.
- `training/` holds data preparation, training, export and baselines in Python with PyTorch.

## Reproduce

Training runs on CPU or Apple MPS. CUDA works unchanged.

```sh
cd training
uv sync -p 3.12
mkdir -p data
curl -sfL -o data/ru_50k.txt https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ru/ru_50k.txt
uv run python prepare.py                                   # data/words.tsv, ~2 min
uv run python baselines.py                                 # hand rules, suffix table
uv run python train.py --size small --out runs/small       # ~5 min on M1
uv run python train.py --size big --out runs/big           # ~17 min on M1
uv run python train.py --size small --train-on train --out runs/small-heldout
uv run python export.py --size small --run runs/small      # models/, tests/, parity/
uv run python export.py --size big --run runs/big
cd .. && cargo test --all-features
cargo run --release --example parity --features big -- training/parity
```

## License

The code is MIT. The weights and golden files derive from OpenCorpora
(CC BY-SA 3.0) and FrequencyWords (CC BY-SA 4.0), so they are CC BY-SA 4.0.
[`NOTICE.md`](NOTICE.md) has the attributions. The approach follows
[gpu-lexer](https://github.com/vercel-labs/gpu-lexer) by Shu Ding.

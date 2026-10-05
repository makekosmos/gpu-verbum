# gpu-verbum

A tiny learned part-of-speech tagger for Russian words. One character CNN,
distilled from the OpenCorpora dictionary, quantized to int8 and embedded in
a dependency-free Rust crate.

It names the part of speech of a single word, including words no dictionary
has: «задеплоить», «отревьюить» and «пингануть» come back as infinitives.
Mundus Agenda uses it to check that a task title starts with an action
(«позвонить маме», «купи молоко»).

```rust
use verbum::{Pos, Tagger};

let tagger = Tagger::small();
assert_eq!(tagger.tag("позвонить"), Some(Pos::Infn));
assert!(tagger.tag("купи").is_some_and(Pos::is_action));
assert_eq!(tagger.tag("молоко"), Some(Pos::Noun));
```

## Models

| | `small` (default feature) | `big` (feature `big`) |
|---|---|---|
| Parameters | 39,570 | 149,218 |
| File (int8) | 40,482 bytes | 150,754 bytes |
| Live text, agreement with OpenCorpora | 99.42% | 99.66% |
| Live text, "is an action" F1 | 99.82% | 100.00% |
| Unseen lemmas, agreement | 96.99% | 97.26% |
| Rust, one word on Apple M1 (one core) | 26 µs | 87 µs |

"Live text" weighs each word by its frequency in OpenSubtitles 2018, so it
is the agreement a reader meets. "Unseen lemmas" comes from a twin model
trained without 10% of lemmas and scored on them. `MODEL_CARD.md` has the
method, the full numbers and the limitations.

`small` is the default because `big` buys a third of a percent for four
times the size. Enable `big` when unknown words matter more than bytes.

## Layout

- `src/` is the inference crate: `.vrb` parsing and the forward pass.
- `models/` holds the shipped weights and their metrics (`*.json`).
- `tests/golden.rs` checks Rust logits against PyTorch on ~1,000 words per model.
- `examples/parity.rs` checks every test and live word (345,083) and times the tagger.
- `training/` is data preparation, training, export and baselines (Python, PyTorch).

## Reproduce

Training runs on CPU or Apple MPS; CUDA works unchanged.

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

Code is MIT. The weights and golden files derive from OpenCorpora
(CC BY-SA 3.0) and FrequencyWords (CC BY-SA 4.0) and are CC BY-SA 4.0.
See `NOTICE.md`.

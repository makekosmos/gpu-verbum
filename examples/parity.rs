//! Full parity run: every test and live word, Rust vs PyTorch labels.
//!
//!     cargo run --release --example parity --features big -- training/parity
use std::time::Instant;
use verbum::{Pos, Tagger};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "training/parity".into());
    for (name, tagger) in [("small", Tagger::small()), ("big", Tagger::big())] {
        let text =
            std::fs::read_to_string(format!("{dir}/{name}.tsv")).expect("run export.py first");
        let start = Instant::now();
        let (mut total, mut same) = (0usize, 0usize);
        for line in text.lines() {
            let (word, label) = line.split_once('\t').unwrap();
            total += 1;
            same += usize::from(tagger.tag(word) == Pos::from_tag(label));
        }
        let per_word = start.elapsed().as_secs_f64() * 1e6 / total as f64;
        println!(
            "{name}: {same}/{total} labels match PyTorch ({:.4}%), {per_word:.1} µs/word",
            100.0 * same as f64 / total as f64
        );
    }
}

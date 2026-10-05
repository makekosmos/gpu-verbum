//! The Rust forward pass must reproduce PyTorch on the same int8 weights.
//! `golden-<size>.tsv` is written by `training/export.py`: word, label, logits.

use verbum::{Pos, Tagger};

fn check(tagger: &Tagger, golden: &str) {
    let mut rows = 0;
    for line in golden.lines() {
        let mut cols = line.split('\t');
        let (word, label, logits) = (
            cols.next().unwrap(),
            cols.next().unwrap(),
            cols.next().unwrap(),
        );
        let expected: Vec<f32> = logits.split(' ').map(|v| v.parse().unwrap()).collect();
        let scores = tagger
            .scores(word)
            .unwrap_or_else(|| panic!("no scores for {word:?}"));
        for (got, want) in scores.iter().zip(&expected) {
            assert!(
                (got - want).abs() < 1e-3,
                "{word}: logit {got} vs PyTorch {want}"
            );
        }
        assert_eq!(tagger.tag(word), Pos::from_tag(label), "{word}");
        rows += 1;
    }
    assert!(rows > 1000, "golden file has {rows} rows");
}

#[cfg(feature = "small")]
#[test]
fn small_matches_pytorch() {
    check(Tagger::small(), include_str!("golden-small.tsv"));
}

#[cfg(feature = "big")]
#[test]
fn big_matches_pytorch() {
    check(Tagger::big(), include_str!("golden-big.tsv"));
}

#[cfg(feature = "small")]
#[test]
fn task_title_openers() {
    let t = Tagger::small();
    for word in [
        "позвонить",
        "отнести",
        "записаться",
        "задеплоить",
        "отревьюить",
    ] {
        assert_eq!(t.tag(word), Some(Pos::Infn), "{word}");
    }
    for word in ["позвони", "напиши", "отправь"] {
        assert_eq!(t.tag(word), Some(Pos::Impr), "{word}");
    }
    for word in ["молоко", "отчёт", "отчет", "встреча", "дедлайн"] {
        assert_eq!(t.tag(word), Some(Pos::Noun), "{word}");
    }
    assert_eq!(t.tag("Позвонить"), t.tag("позвонить"));
    assert_eq!(t.tag(""), None);
}

#[test]
fn rejects_foreign_bytes() {
    assert_eq!(
        Tagger::from_bytes(b"nope").err(),
        Some(verbum::LoadError::BadMagic)
    );
    assert_eq!(
        Tagger::from_bytes(b"VRB1\x01").err(),
        Some(verbum::LoadError::Truncated)
    );
}

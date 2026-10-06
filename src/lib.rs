//! A tiny learned part-of-speech tagger for Russian words.
//!
//! A character CNN distilled from the OpenCorpora dictionary: it reads a
//! word from its end and names its part of speech, including words no
//! dictionary has («задеплоить», «отревьюить»). Two models are embedded
//! behind features: `small` (default, ~40 KB) and `big` (~150 KB). See
//! `MODEL_CARD.md` for how they were trained and how well they agree with
//! the teacher.
//!
//! ```
//! use verbum::{Pos, Tagger};
//!
//! let tagger = Tagger::small();
//! assert_eq!(tagger.tag("позвонить"), Some(Pos::Infn));
//! assert!(tagger.tag("купи").is_some_and(Pos::is_action));
//! assert_eq!(tagger.tag("молоко"), Some(Pos::Noun));
//! ```

mod format;
mod pos;

pub use format::LoadError;
pub use pos::Pos;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

use format::{Conv, Dense, Weights};

/// A loaded model. Inference is allocation-light and takes microseconds.
pub struct Tagger {
    w: Weights,
}

impl Tagger {
    /// Loads a `.vrb` model (format: `training/export.py`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Tagger, LoadError> {
        format::parse(bytes).map(|w| Tagger { w })
    }

    /// The default model: 39,570 parameters, ~40 KB.
    #[cfg(feature = "small")]
    pub fn small() -> &'static Tagger {
        static TAGGER: std::sync::OnceLock<Tagger> = std::sync::OnceLock::new();
        TAGGER.get_or_init(|| {
            Tagger::from_bytes(include_bytes!("../models/small.vrb")).expect("embedded small.vrb")
        })
    }

    /// The larger model: 149,218 parameters, ~150 KB; better on unseen words.
    #[cfg(feature = "big")]
    pub fn big() -> &'static Tagger {
        static TAGGER: std::sync::OnceLock<Tagger> = std::sync::OnceLock::new();
        TAGGER.get_or_init(|| {
            Tagger::from_bytes(include_bytes!("../models/big.vrb")).expect("embedded big.vrb")
        })
    }

    /// The most likely part of speech of one word; `None` for an empty word.
    pub fn tag(&self, word: &str) -> Option<Pos> {
        let scores = self.scores(word)?;
        let mut best = 0;
        for (i, s) in scores.iter().enumerate() {
            if *s > scores[best] {
                best = i;
            }
        }
        Some(Pos::ALL[best])
    }

    /// Raw scores (logits), indexed like [`Pos::ALL`]; `None` for an empty word.
    pub fn scores(&self, word: &str) -> Option<[f32; 18]> {
        let ids = self.encode(word)?;
        let w = &self.w;
        let len = ids.len();
        let l = w.max_len;

        // Activations are position-major: x[t * channels + c]. Position 0 is
        // the word's last letter; positions past `len` are padding.
        let mut x = vec![0.0; l * w.emb];
        for (t, &id) in ids.iter().enumerate() {
            x[t * w.emb..][..w.emb].copy_from_slice(&w.embed[id * w.emb..][..w.emb]);
        }
        for (i, conv) in w.convs.iter().enumerate() {
            let y = conv1d_relu(conv, &x, l);
            x = if i == 0 {
                y
            } else {
                x.iter().zip(&y).map(|(a, b)| a + b).collect()
            };
        }

        let width = w.convs.last().map_or(w.emb, |c| c.out);
        let mut features = vec![0.0; width * (w.head_pos + 1)];
        for t in 0..len.min(w.head_pos) {
            for c in 0..width {
                features[c * w.head_pos + t] = x[t * width + c];
            }
        }
        let pooled = &mut features[width * w.head_pos..];
        pooled.fill(f32::NEG_INFINITY);
        for t in 0..len {
            for (p, &v) in pooled.iter_mut().zip(&x[t * width..][..width]) {
                *p = p.max(v);
            }
        }

        let hidden: Vec<f32> = dense(&w.hidden, &features)
            .into_iter()
            .map(|v| v.max(0.0))
            .collect();
        let out = dense(&w.out, &hidden);
        let mut scores = [0.0; 18];
        scores.copy_from_slice(&out);
        Some(scores)
    }

    /// Character ids from the last letter backwards, at most `max_len`.
    fn encode(&self, word: &str) -> Option<Vec<usize>> {
        let ids: Vec<usize> = word
            .chars()
            .flat_map(char::to_lowercase)
            .map(|ch| if ch == 'ё' { 'е' } else { ch })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .take(self.w.max_len)
            .map(|ch| {
                self.w
                    .alphabet
                    .iter()
                    .position(|&a| a == ch)
                    .map_or(1, |i| i + 2)
            })
            .collect();
        (!ids.is_empty()).then_some(ids)
    }
}

/// Kernel-3 convolution with zero padding equal to the dilation, then ReLU.
fn conv1d_relu(conv: &Conv, x: &[f32], l: usize) -> Vec<f32> {
    let mut y = vec![0.0; l * conv.out];
    for t in 0..l {
        let out = &mut y[t * conv.out..][..conv.out];
        out.copy_from_slice(&conv.bias);
        for (j, tap) in conv.taps.iter().enumerate() {
            let Some(src) = (t + j * conv.dilation).checked_sub(conv.dilation) else {
                continue;
            };
            if src >= l {
                continue;
            }
            let column = &x[src * conv.inp..][..conv.inp];
            for (o, acc) in out.iter_mut().enumerate() {
                *acc += dot(&tap[o * conv.inp..][..conv.inp], column);
            }
        }
        for v in out.iter_mut() {
            *v = v.max(0.0);
        }
    }
    y
}

fn dense(layer: &Dense, x: &[f32]) -> Vec<f32> {
    (0..layer.out)
        .map(|o| layer.bias[o] + dot(&layer.weight[o * layer.inp..][..layer.inp], x))
        .collect()
}

/// Eight independent accumulators let the compiler vectorize the sum.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, a_rest) = a.split_at(a.len() / 8 * 8);
    let (b8, b_rest) = b.split_at(a8.len());
    for (ca, cb) in a8.chunks_exact(8).zip(b8.chunks_exact(8)) {
        for k in 0..8 {
            lanes[k] += ca[k] * cb[k];
        }
    }
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}

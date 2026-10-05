//! Parsing `.vrb` files (layout documented in `training/export.py`).

use crate::Pos;

/// Why a `.vrb` file could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    BadMagic,
    Truncated,
    TrailingBytes,
    BadText,
    /// The file's labels are not [`Pos::ALL`] in order.
    LabelMismatch,
    /// A dimension is zero or inconsistent with the rest of the header.
    BadShape,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self {
            LoadError::BadMagic => "not a .vrb file",
            LoadError::Truncated => "file ends early",
            LoadError::TrailingBytes => "unexpected bytes after the last tensor",
            LoadError::BadText => "alphabet or labels are not UTF-8",
            LoadError::LabelMismatch => "labels differ from this crate's Pos",
            LoadError::BadShape => "inconsistent dimensions",
        };
        f.write_str(what)
    }
}

impl std::error::Error for LoadError {}

pub(crate) struct Conv {
    pub out: usize,
    pub inp: usize,
    pub dilation: usize,
    /// One `[out][inp]` matrix per kernel tap (offsets -dilation, 0, +dilation),
    /// so each output is three contiguous dot products.
    pub taps: [Vec<f32>; 3],
    pub bias: Vec<f32>,
}

pub(crate) struct Dense {
    pub out: usize,
    pub inp: usize,
    /// `[out][inp]`, row-major.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

pub(crate) struct Weights {
    pub alphabet: Vec<char>,
    pub max_len: usize,
    pub head_pos: usize,
    pub emb: usize,
    /// `[vocab][emb]`, row-major.
    pub embed: Vec<f32>,
    pub convs: Vec<Conv>,
    pub hidden: Dense,
    pub out: Dense,
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], LoadError> {
        if self.bytes.len() < n {
            return Err(LoadError::Truncated);
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<usize, LoadError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }

    fn f32(&mut self) -> Result<f32, LoadError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn text(&mut self) -> Result<&'a str, LoadError> {
        let len = self.u32()?;
        std::str::from_utf8(self.take(len)?).map_err(|_| LoadError::BadText)
    }

    /// An int8 tensor with one f32 scale, dequantized.
    fn weight(&mut self, len: usize) -> Result<Vec<f32>, LoadError> {
        let scale = self.f32()?;
        Ok(self
            .take(len)?
            .iter()
            .map(|&b| scale * f32::from(b as i8))
            .collect())
    }

    fn bias(&mut self, len: usize) -> Result<Vec<f32>, LoadError> {
        (0..len).map(|_| self.f32()).collect()
    }

    fn dense(&mut self, out: usize, inp: usize) -> Result<Dense, LoadError> {
        Ok(Dense {
            out,
            inp,
            weight: self.weight(out * inp)?,
            bias: self.bias(out)?,
        })
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Weights, LoadError> {
    let mut r = Reader { bytes };
    if r.take(4)? != b"VRB1" {
        return Err(LoadError::BadMagic);
    }
    let mut header = [0; 8];
    for field in &mut header {
        *field = r.u32()?;
    }
    let [vocab, emb, width, hidden, labels, max_len, head_pos, convs] = header;
    if [vocab, emb, width, hidden, max_len, head_pos, convs].contains(&0) || head_pos > max_len {
        return Err(LoadError::BadShape);
    }
    let dilations = (0..convs).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
    let alphabet: Vec<char> = r.text()?.chars().collect();
    if alphabet.len() + 2 != vocab {
        return Err(LoadError::BadShape);
    }
    let names = r.text()?;
    if labels != Pos::ALL.len() || !names.split(',').eq(Pos::ALL.iter().map(|p| p.tag())) {
        return Err(LoadError::LabelMismatch);
    }

    let embed = r.weight(vocab * emb)?;
    let mut layers = Vec::with_capacity(convs);
    for (i, dilation) in dilations.into_iter().enumerate() {
        let inp = if i == 0 { emb } else { width };
        let weight = r.weight(width * inp * 3)?;
        let taps = [0, 1, 2].map(|j| (0..width * inp).map(|oi| weight[oi * 3 + j]).collect());
        layers.push(Conv {
            out: width,
            inp,
            dilation,
            taps,
            bias: r.bias(width)?,
        });
    }
    let hidden = r.dense(hidden, width * (head_pos + 1))?;
    let out = r.dense(labels, hidden.out)?;
    if !r.bytes.is_empty() {
        return Err(LoadError::TrailingBytes);
    }
    Ok(Weights {
        alphabet,
        max_len,
        head_pos,
        emb,
        embed,
        convs: layers,
        hidden,
        out,
    })
}

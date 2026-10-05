# Notices

The source code in this repository (`src/`, `examples/`, `tests/*.rs`,
`training/*.py`) is MIT-licensed; see `LICENSE`.

The model weights (`models/*.vrb`) and golden files (`tests/*.tsv`) are
derived from the data below and are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## OpenCorpora

The teacher: part-of-speech labels for 3,064,812 Russian word forms
(391,778 lemmas), OpenCorpora dictionary revision 417150, as compiled in the
`pymorphy3-dicts-ru` package. © OpenCorpora contributors, licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
<https://opencorpora.org>

## FrequencyWords

Live-text word frequencies (`ru_50k.txt`, OpenSubtitles 2018) used to weight
training and evaluation. © Hermit Dave, content licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
<https://github.com/hermitdave/FrequencyWords>

## Prior art

The approach (a tiny learned model distilled from a heavyweight teacher,
quantized and embedded) follows [gpu-lexer](https://github.com/vercel-labs/gpu-lexer)
by Shu Ding (MIT). No code is shared.

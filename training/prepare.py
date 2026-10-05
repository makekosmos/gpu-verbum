"""Export the OpenCorpora lexicon (via pymorphy3-dicts-ru) as a labelled word list.

Output: data/words.tsv with one row per distinct lowercase word form:

    word <TAB> classes <TAB> split

`classes` lists every part of speech the dictionary allows for the word,
best first (pymorphy3 ranks dictionary parses by corpus frequency). The
first class is the teacher's answer; the rest are also acceptable.

The train/test split is by lemma, not by word form: every form of a lemma
lands on the same side, so the test set measures generalisation to unseen
words rather than recall of a seen stem. A word whose lemmas fall on both
sides is dropped.
"""

import sys
import zlib
from pathlib import Path

import pymorphy3
from pymorphy3.units import DictionaryAnalyzer

from labels import label_of

TEST_BUCKETS = 10  # 1 in 10 lemmas goes to test


def split_of(lemma: str) -> str:
    return "test" if zlib.crc32(lemma.encode()) % TEST_BUCKETS == 0 else "train"


def main(out: Path) -> None:
    morph = pymorphy3.MorphAnalyzer()
    words = sorted({p.word for p in morph.iter_known_word_parses("")})
    print(f"{len(words):,} distinct word forms", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with out.open("w", encoding="utf-8") as f:
        for i, word in enumerate(words):
            parses = [
                p for p in morph.parse(word)
                if isinstance(p.methods_stack[0][0], DictionaryAnalyzer)
            ]
            classes: list[str] = []
            for p in parses:
                label = label_of(p.tag)
                if label not in classes:
                    classes.append(label)
            splits = {split_of(p.normal_form) for p in parses}
            if not classes or len(splits) != 1:
                dropped += 1
                continue
            f.write(f"{word}\t{','.join(classes)}\t{splits.pop()}\n")
            kept += 1
            if i % 500_000 == 0:
                print(f"  {i:,}/{len(words):,}", file=sys.stderr)
    print(f"kept {kept:,}, dropped {dropped:,} (lemmas on both sides)", file=sys.stderr)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/words.tsv"))

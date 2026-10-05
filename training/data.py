"""Loading data/words.tsv and scoring predictions against the teacher."""

from dataclasses import dataclass
from pathlib import Path

from labels import ACTION, LABELS


def norm(word: str) -> str:
    """Live text writes «е» for «ё» half the time; treat them as one letter."""
    return word.lower().replace("ё", "е")


def load_freq(path: Path = Path("data/ru_50k.txt")) -> dict[str, int]:
    """Word counts from FrequencyWords (OpenSubtitles 2018, top 50k)."""
    freq: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            word, count = line.split()
            key = norm(word)
            freq[key] = freq.get(key, 0) + int(count)
    return freq


@dataclass
class Word:
    text: str
    classes: list[str]  # teacher's allowed classes, best first
    freq: int = 0  # occurrences in live text; 0 when outside the top 50k

    @property
    def gold(self) -> str:
        return self.classes[0]


def load(split: str, path: Path = Path("data/words.tsv")) -> list[Word]:
    """`split` is "train", "test" or "all"."""
    freq = load_freq()
    words = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            text, classes, side = line.rstrip("\n").split("\t")
            if split in (side, "all"):
                words.append(Word(text, classes.split(","), freq.get(norm(text), 0)))
    return words


def score(words: list[Word], predicted: list[str], weighted: bool = False) -> dict[str, float]:
    """Agreement with the teacher, in percent.

    strict  — prediction equals the teacher's top class;
    lenient — prediction is one of the classes the dictionary allows;
    action  — precision/recall of "this is an action" (INFN or IMPR) against
              the teacher's top class: the question Agenda actually asks.

    `weighted` counts each word by its frequency in live text (words outside
    the frequency list drop out), i.e. agreement as a reader would meet it.
    """
    pairs = [(w, p, w.freq if weighted else 1) for w, p in zip(words, predicted)]
    pairs = [x for x in pairs if x[2] > 0]
    n = sum(k for _, _, k in pairs)
    strict = sum(k for w, p, k in pairs if p == w.gold)
    lenient = sum(k for w, p, k in pairs if p in w.classes)
    tp = sum(k for w, p, k in pairs if p in ACTION and w.gold in ACTION)
    fp = sum(k for w, p, k in pairs if p in ACTION and w.gold not in ACTION)
    fn = sum(k for w, p, k in pairs if p not in ACTION and w.gold in ACTION)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "words": len(pairs),
        "strict": 100 * strict / n,
        "lenient": 100 * lenient / n,
        "action_precision": 100 * precision,
        "action_recall": 100 * recall,
        "action_f1": 100 * f1,
    }


def report(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name:<34} strict {metrics['strict']:6.2f}%  lenient {metrics['lenient']:6.2f}%  "
        f"action P {metrics['action_precision']:6.2f}% R {metrics['action_recall']:6.2f}% "
        f"F1 {metrics['action_f1']:6.2f}%"
    )


assert set(LABELS) >= ACTION

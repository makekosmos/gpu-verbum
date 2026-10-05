"""Non-neural baselines the model has to beat, scored on the test split.

1. Hand rules: the infinitive/imperative endings one would write by hand.
2. Suffix table: for each suffix up to 5 letters seen in training, the most
   common teacher class; predict from the longest known suffix. This is the
   classic unknown-word guesser (pymorphy's suffix analyzer works this way).
"""

from collections import Counter, defaultdict

from data import load, report, score

INFINITIVE = ("ть", "ти", "чь", "ться", "тись", "чься")
IMPERATIVE = ("и", "й", "ь", "ите", "йте", "ьте", "ись", "йся", "йтесь", "итесь", "ься")


def hand_rule(word: str) -> str:
    if word.endswith(INFINITIVE):
        return "INFN"
    if word.endswith(IMPERATIVE):
        return "IMPR"
    return "NOUN"


MAX_SUFFIX = 5


def suffix_table(train) -> tuple[dict[str, str], str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for w in train:
        for k in range(1, MAX_SUFFIX + 1):
            if len(w.text) > k:
                counts[w.text[-k:]][w.gold] += 1
    table = {s: c.most_common(1)[0][0] for s, c in counts.items()}
    fallback = Counter(w.gold for w in train).most_common(1)[0][0]
    return table, fallback


def suffix_predict(word: str, table: dict[str, str], fallback: str) -> str:
    for k in range(min(MAX_SUFFIX, len(word) - 1), 0, -1):
        label = table.get(word[-k:])
        if label:
            return label
    return fallback


def main() -> None:
    train, test = load("train"), load("test")
    print(f"train {len(train):,} / test {len(test):,} word forms (lemma-disjoint)")
    print("— test split: lemmas never seen in training")
    for weighted in (False, True):
        tag = " · freq" if weighted else ""
        print(report("hand rules" + tag, score(test, [hand_rule(w.text) for w in test], weighted)))
    table, fallback = suffix_table(train)
    print(f"suffix table: {len(table):,} suffixes")
    predicted = [suffix_predict(w.text, table, fallback) for w in test]
    for weighted in (False, True):
        tag = " · freq" if weighted else ""
        print(report("suffix table (≤5)" + tag, score(test, predicted, weighted)))


if __name__ == "__main__":
    main()

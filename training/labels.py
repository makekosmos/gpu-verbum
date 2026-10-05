"""The label set: OpenCorpora parts of speech, with verbs split by mood.

Agenda asks one question of a task title's first word: is it an action
(infinitive or imperative)? Imperatives are finite verbs in OpenCorpora, so
they get a class of their own instead of hiding inside VERB.
"""

LABELS = [
    "NOUN",  # существительное
    "ADJF",  # прилагательное (полное)
    "ADJS",  # прилагательное (краткое)
    "COMP",  # компаратив
    "VERB",  # глагол, изъявительное наклонение
    "IMPR",  # глагол, повелительное наклонение
    "INFN",  # инфинитив
    "PRTF",  # причастие (полное)
    "PRTS",  # причастие (краткое)
    "GRND",  # деепричастие
    "NUMR",  # числительное
    "ADVB",  # наречие
    "NPRO",  # местоимение-существительное
    "PRED",  # предикатив
    "PREP",  # предлог
    "CONJ",  # союз
    "PRCL",  # частица
    "INTJ",  # междометие
]
INDEX = {label: i for i, label in enumerate(LABELS)}

# The classes Agenda treats as "the title starts with an action".
ACTION = {"INFN", "IMPR"}


def label_of(tag) -> str:
    if tag.POS == "VERB" and "impr" in tag:
        return "IMPR"
    return str(tag.POS)

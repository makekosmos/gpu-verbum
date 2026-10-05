"""The tagger: a small character CNN that reads a word from its end.

Russian marks the part of speech mostly at the end of a word, so words are
reversed and right-truncated to MAX_LEN: position 0 is always the last
letter. A stack of 1-D convolutions builds suffix features; the head sees
the first HEAD_POS positions verbatim (where endings live) plus a max over
the whole word (stems, prefixes). Every op is a matmul, add or max — easy
to port to plain Rust and later to a GPU kernel.
"""

import torch
from torch import nn

from data import norm
from labels import LABELS

ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя-"  # ё folds into е
PAD, UNK = 0, 1
CHAR_ID = {ch: i + 2 for i, ch in enumerate(ALPHABET)}
VOCAB = len(ALPHABET) + 2
MAX_LEN = 16
HEAD_POS = 6


# The two shipped models. `dims` are Tagger arguments; epochs come from the
# loss curves: small plateaus by 6, big by 7 (we train it 12).
SIZES = {
    "small": {"dims": {"emb": 16, "width": 48, "hidden": 64}, "epochs": 6},
    "big": {"dims": {"emb": 16, "width": 96, "hidden": 128}, "epochs": 12},
}


def encode(word: str) -> list[int]:
    ids = [CHAR_ID.get(ch, UNK) for ch in reversed(norm(word))][:MAX_LEN]
    return ids + [PAD] * (MAX_LEN - len(ids))


class Tagger(nn.Module):
    def __init__(self, emb: int = 16, width: int = 48, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb, width, 3, padding=1),
                nn.Conv1d(width, width, 3, padding=1),
                nn.Conv1d(width, width, 3, padding=2, dilation=2),
            ]
        )
        self.hidden = nn.Linear(width * (HEAD_POS + 1), hidden)
        self.out = nn.Linear(hidden, len(LABELS))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD).unsqueeze(1)  # B,1,L
        x = self.embed(ids).transpose(1, 2)  # B,C,L
        for i, conv in enumerate(self.convs):
            y = torch.relu(conv(x))
            x = y if i == 0 else x + y  # residual after the first layer
        x = x * mask
        head = x[:, :, :HEAD_POS].flatten(1)
        pooled = x.masked_fill(~mask, float("-inf")).amax(2)
        h = torch.relu(self.hidden(torch.cat([head, pooled], 1)))
        return self.out(h)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

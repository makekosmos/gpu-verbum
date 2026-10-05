"""Train the tagger.

    uv run python train.py --size small --train-on all --out runs/small

Half of every batch is drawn uniformly from the dictionary (every word form
counts the same, which teaches endings), the other half by live-text
frequency (which teaches the few hundred function words that carry most
of real text). Without the frequency half the model calls «в», «и», «что»
nouns: they are three rows out of three million.

`--train-on train` holds out the lemma-disjoint test split and measures
generalisation to unseen words; `--train-on all` is the shipping model, and
its test numbers are in-sample.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from data import load, report, score
from labels import INDEX, LABELS
from model import SIZES, Tagger, encode, parameter_count


def tensors(words):
    ids = torch.tensor([encode(w.text) for w in words], dtype=torch.long)
    gold = torch.tensor([INDEX[w.gold] for w in words], dtype=torch.long)
    return ids, gold


@torch.no_grad()
def predict(model, ids, device, batch=8192) -> list[str]:
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(ids), batch):
        out += model(ids[i : i + batch].to(device)).argmax(1).tolist()
    model.train(was_training)
    return [LABELS[i] for i in out]


def evaluate(model, device, test, live) -> dict:
    test_pred = predict(model, tensors(test)[0], device)
    return {
        "test": score(test, test_pred),
        "test_freq": score(test, test_pred, weighted=True),
        "live_freq": score(live, predict(model, tensors(live)[0], device), weighted=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="small")
    ap.add_argument("--train-on", choices=["train", "all"], default="all")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    epochs = args.epochs or SIZES[args.size]["epochs"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    words, test = load(args.train_on), load("test")
    live = [w for w in load("all") if w.freq > 0]
    seen_live = [w for w in words if w.freq > 0]
    ids, gold = tensors(words)
    live_ids, live_gold = tensors(seen_live)
    live_p = torch.tensor([w.freq for w in seen_live], dtype=torch.float)

    torch.manual_seed(args.seed)
    model = Tagger(**SIZES[args.size]["dims"]).to(device)
    print(f"{args.size}: {parameter_count(model):,} parameters on {device}, {epochs} epochs")
    half = args.batch // 2
    steps = epochs * (len(words) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(words))
        for i in range(0, len(words) - args.batch + 1, args.batch):
            uniform = order[i : i + half]
            frequent = torch.multinomial(live_p, half, replacement=True)
            x = torch.cat([ids[uniform], live_ids[frequent]]).to(device)
            y = torch.cat([gold[uniform], live_gold[frequent]]).to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        metrics = evaluate(model, device, test, live)
        print(f"epoch {epoch + 1} ({time.time() - start:.0f}s)")
        for name, m in metrics.items():
            print("  " + report(name, m))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "size": args.size,
                "parameters": parameter_count(model),
                "train_on": args.train_on,
                "epochs": epochs,
                "seed": args.seed,
                "seconds": round(time.time() - start),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

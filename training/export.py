"""Quantize a trained run to int8 and write the `.vrb` file the Rust crate embeds.

    uv run python export.py --size small --run runs/small

Writes:
  ../models/<size>.vrb               the model (format below)
  ../models/<size>.json              float vs int8 accuracy, sizes, provenance
  ../tests/golden-<size>.tsv         ~1000 words with int8 logits (committed)
  parity/<size>.tsv                  every test and live word with its label
                                     (local; `cargo run --example parity`)

`.vrb` layout, little-endian:
  b"VRB1"
  u32 × 8   vocab, emb, width, hidden, labels, max_len, head_pos, conv count
  u32 × n   dilation of each conv (kernel 3, padding = dilation)
  u32 + utf8  alphabet (char id = index + 2; 0 = pad, 1 = unknown)
  u32 + utf8  label names, comma-separated
  tensors in model order: embedding, (conv weight, conv bias) × n,
  hidden weight, hidden bias, out weight, out bias.
  A weight is f32 scale + i8 × len (value = scale × i8); a bias is f32 × len.
"""

import argparse
import json
import random
import struct
from pathlib import Path

import torch

from data import load, report, score
from labels import LABELS
from model import ALPHABET, HEAD_POS, MAX_LEN, SIZES, VOCAB, Tagger, encode, parameter_count
from train import evaluate

ROOT = Path(__file__).resolve().parent.parent
# Words outside the dictionary that the golden file must cover.
PROBES = (
    "задеплоить отревьюить замержить пофиксить пингануть созвониться позвони купи "
    "Отнести ОТЧЁТ отчет ёлка e-mail kpi 123 стекло печь что в и чем перед "
    "а ь пол-лимона сверхдлиннейшееслововмиремира"
).split()


def quantize(w: torch.Tensor) -> tuple[float, torch.Tensor]:
    scale = w.abs().max().item() / 127 or 1.0
    return scale, torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, required=True)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    dims = SIZES[args.size]["dims"]

    model = Tagger(**dims)
    model.load_state_dict(torch.load(args.run / "model.pt", map_location="cpu"))
    model.eval()
    run_meta = json.loads((args.run / "metrics.json").read_text())

    blob = bytearray(b"VRB1")
    blob += struct.pack("<8I", VOCAB, dims["emb"], dims["width"], dims["hidden"],
                        len(LABELS), MAX_LEN, HEAD_POS, len(model.convs))
    blob += struct.pack(f"<{len(model.convs)}I", *(c.dilation[0] for c in model.convs))
    for text in (ALPHABET, ",".join(LABELS)):
        raw = text.encode()
        blob += struct.pack("<I", len(raw)) + raw

    quantized = Tagger(**dims)
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("weight"):
            scale, q = quantize(tensor)
            blob += struct.pack("<f", scale) + bytes(q.flatten().view(torch.uint8).tolist())
            state[name] = q.float() * torch.tensor(scale, dtype=torch.float32)
        else:
            values = tensor.flatten().tolist()
            blob += struct.pack(f"<{len(values)}f", *values)
            state[name] = tensor
    quantized.load_state_dict(state)
    quantized.eval()

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / f"{args.size}.vrb").write_bytes(bytes(blob))

    test, live = load("test"), [w for w in load("all") if w.freq > 0]
    float_metrics = run_meta["metrics"]
    int8_metrics = evaluate(quantized, "cpu", test, live)
    for name in int8_metrics:
        print(report(f"float {name}", float_metrics[name]))
        print(report(f"int8  {name}", int8_metrics[name]))
    (ROOT / "models" / f"{args.size}.json").write_text(json.dumps({
        "size": args.size,
        "parameters": parameter_count(model),
        "bytes": len(blob),
        "trained": {k: run_meta[k] for k in ("train_on", "epochs", "seed", "seconds")},
        "float": float_metrics,
        "int8": int8_metrics,
    }, indent=2, ensure_ascii=False))
    print(f"models/{args.size}.vrb: {len(blob):,} bytes")

    random.seed(0)
    sample = PROBES + [w.text for w in random.sample(test, 600)] + [
        w.text for w in random.sample(live, 400)
    ]
    with torch.no_grad():
        logits = quantized(torch.tensor([encode(w) for w in sample]))
    (ROOT / "tests").mkdir(exist_ok=True)
    with (ROOT / "tests" / f"golden-{args.size}.tsv").open("w", encoding="utf-8") as f:
        for word, row in zip(sample, logits):
            f.write(f"{word}\t{LABELS[row.argmax()]}\t{' '.join(f'{v:.6f}' for v in row.tolist())}\n")

    words = [w.text for w in test] + [w.text for w in live]
    Path("parity").mkdir(exist_ok=True)
    with torch.no_grad(), (Path("parity") / f"{args.size}.tsv").open("w", encoding="utf-8") as f:
        for i in range(0, len(words), 8192):
            chunk = words[i : i + 8192]
            for word, k in zip(chunk, quantized(torch.tensor([encode(w) for w in chunk])).argmax(1).tolist()):
                f.write(f"{word}\t{LABELS[k]}\n")


if __name__ == "__main__":
    main()

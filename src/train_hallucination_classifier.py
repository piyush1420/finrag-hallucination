"""Train a RoBERTa hallucination detector on RAGTruth; test transfer to FinanceBench (T24, optional).

Binary, response-level task: given the retrieved context + a model's answer, predict
whether the answer is *hallucinated* (unsupported by the context). We train on the QA
subset of RAGTruth (Niu et al. 2024) -- a response is hallucinated if it carries any
annotated hallucination span -- and test zero-shot transfer on our 50 hand-labeled
FinanceBench cases (hallucinated = any of the four taxonomy types; grounded = the
"other" cases that manual review found were actually correct).

Runs on Apple MPS / CPU. RAGTruth files expected in data/raw/ragtruth/ (from the
ParticleMedia/RAGTruth GitHub `dataset/` folder). We pre-tokenize once (not per batch)
so training is GPU-bound rather than tokenizer-bound.

Run:  python -u src/train_hallucination_classifier.py                   # full QA train, 2 epochs
      python -u src/train_hallucination_classifier.py --max-train 300 --epochs 1   # quick smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RAGTRUTH = ROOT / "data" / "raw" / "ragtruth"
FINCASES = ROOT / "annotations" / "failure_cases_50.csv"
METRICS_OUT = ROOT / "results" / "t24_classifier_metrics.md"
FIG_OUT = ROOT / "results" / "fig_t24_confusion.png"

MODEL = "roberta-base"
MAX_LEN = 256


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- data ----------------------------------------------------------------

def load_ragtruth_qa():
    """Return (train, test) lists of (context, response, label) for the QA subset."""
    src = {json.loads(l)["source_id"]: json.loads(l)
           for l in open(RAGTRUTH / "source_info.jsonl", encoding="utf-8")}
    train, test = [], []
    for line in open(RAGTRUTH / "response.jsonl", encoding="utf-8"):
        r = json.loads(line)
        s = src[r["source_id"]]
        if s["task_type"] != "QA":
            continue
        ctx = s["source_info"]["passages"]
        label = 1 if r["labels"] else 0            # any annotated span -> hallucinated
        (test if r["split"] == "test" else train).append((ctx, r["response"], label))
    return train, test


def load_financebench():
    """Our 50 labeled cases as (context, answer, label); label 1 = true hallucination."""
    df = pd.read_csv(FINCASES)
    rows = []
    for _, r in df.iterrows():
        label = 0 if str(r["hallucination_type"]).strip() == "other" else 1
        rows.append((str(r["retrieved_context"]), str(r["generated_answer"]), label))
    return rows


def encode(rows, tok) -> TensorDataset:
    """Tokenize all (context, answer) pairs once into a fixed-length tensor dataset."""
    ctx, ans, labels = zip(*rows)
    enc = tok(list(ctx), list(ans), truncation="longest_first", max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))


# ---- train / eval --------------------------------------------------------

def train(model, loader, dev, epochs, class_weight):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight.to(dev))
    model.train()
    for ep in range(epochs):
        total = 0.0
        for step, (ids, mask, labels) in enumerate(loader):
            ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            if step % 20 == 0:
                print(f"  epoch {ep+1} step {step}/{len(loader)} loss {loss.item():.3f}",
                      flush=True)
        print(f"epoch {ep+1} mean loss {total/len(loader):.3f}", flush=True)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    preds, golds = [], []
    for ids, mask, labels in loader:
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        preds += logits.argmax(-1).cpu().tolist()
        golds += labels.tolist()
    return preds, golds


def report(name, golds, preds):
    p, r, f1, _ = precision_recall_fscore_support(golds, preds, average="binary",
                                                  pos_label=1, zero_division=0)
    acc = accuracy_score(golds, preds)
    print(f"\n[{name}] acc {acc:.3f} | precision {p:.3f} | recall {r:.3f} | F1 {f1:.3f}",
          flush=True)
    return {"set": name, "n": len(golds), "accuracy": acc,
            "precision": p, "recall": r, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description="RoBERTa hallucination classifier (T24).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=None, help="cap train size (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    print(f"device: {dev}", flush=True)

    rt_train, rt_test = load_ragtruth_qa()
    if args.max_train:
        rt_train = rt_train[:args.max_train]
    fin = load_financebench()
    n_pos = sum(l for _, _, l in rt_train)
    print(f"RAGTruth QA: train {len(rt_train)} ({n_pos} hallucinated), test {len(rt_test)}",
          flush=True)
    print(f"FinanceBench eval: {len(fin)} cases ({sum(l for _,_,l in fin)} hallucinated)",
          flush=True)

    # inverse-frequency class weights (train is majority-grounded)
    w0 = len(rt_train) / (2 * (len(rt_train) - n_pos))
    w1 = len(rt_train) / (2 * n_pos)
    class_weight = torch.tensor([w0, w1], dtype=torch.float)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)

    print("tokenizing...", flush=True)
    train_ds, test_ds, fin_ds = (encode(rt_train, tok), encode(rt_test, tok), encode(fin, tok))
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh)

    print("training...", flush=True)
    train(model, dl(train_ds, True), dev, args.epochs, class_weight)

    metrics = []
    rt_preds, rt_golds = predict(model, dl(test_ds, False), dev)
    metrics.append(report("RAGTruth-QA test", rt_golds, rt_preds))
    fin_preds, fin_golds = predict(model, dl(fin_ds, False), dev)
    metrics.append(report("FinanceBench-50 (transfer)", fin_golds, fin_preds))

    # confusion matrix on the FinanceBench transfer test (the deliverable)
    cm = confusion_matrix(fin_golds, fin_preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["grounded", "hallucinated"])
    ax.set_yticks([0, 1], ["grounded", "hallucinated"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("RoBERTa (RAGTruth) → FinanceBench")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout(); fig.savefig(FIG_OUT, dpi=200)

    md = ["# T24: RoBERTa hallucination classifier (optional)\n",
          f"`roberta-base` fine-tuned on the RAGTruth QA subset ({len(rt_train)} train "
          f"responses, hallucinated = any annotated span), evaluated on the RAGTruth QA test "
          f"split and, zero-shot, on our 50 labeled FinanceBench cases (hallucinated = any of "
          f"the four taxonomy types; grounded = the 11 'other' cases that were actually "
          f"correct). {args.epochs} epochs, class-weighted loss, max_len {MAX_LEN}.\n",
          "| Eval set | N | Accuracy | Precision | Recall | F1 |",
          "|---|---|---|---|---|---|"]
    for m in metrics:
        md.append(f"| {m['set']} | {m['n']} | {m['accuracy']:.3f} | {m['precision']:.3f} "
                  f"| {m['recall']:.3f} | {m['f1']:.3f} |")
    md += ["", "![confusion matrix](fig_t24_confusion.png)\n"]
    METRICS_OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved -> {METRICS_OUT.relative_to(ROOT)}, {FIG_OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

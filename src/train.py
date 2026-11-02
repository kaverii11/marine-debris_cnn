"""Train and evaluate the marine-debris patch classifier on MARIDA's official splits.

Usage: python src/train.py --data /path/to/MARIDA --out results
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import MaridaPatches, band_stats
from model import build_model


def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def metrics(probs, labels, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
        "threshold": threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MaridaPatches(args.data, "train")
    mean, std = band_stats(train_ds)
    train_ds = MaridaPatches(args.data, "train", mean, std, augment=True)
    val_ds = MaridaPatches(args.data, "val", mean, std)
    test_ds = MaridaPatches(args.data, "test", mean, std)
    pos = sum(y for _, y in train_ds.items)
    print(f"train {len(train_ds)} ({int(pos)} with debris), val {len(val_ds)}, test {len(test_ds)} | device {device}")

    loaders = {
        "train": DataLoader(train_ds, batch_size=16, shuffle=True),
        "val": DataLoader(val_ds, batch_size=32),
        "test": DataLoader(test_ds, batch_size=32),
    }
    model = build_model().to(device)
    # Debris patches are the minority class, so weight them up in the loss
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train_ds) - pos) / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, history = -1, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        sched.step()
        val = metrics(*predict(model, loaders["val"], device), 0.5)
        history.append({"epoch": epoch, "train_loss": running / len(train_ds), **{k: val[k] for k in ("f1", "roc_auc")}})
        print(f"epoch {epoch:2d} loss {running / len(train_ds):.4f} val F1 {val['f1']:.3f} AUC {val['roc_auc']:.3f} ({time.time() - t0:.0f}s)")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "mean": mean, "std": std}, os.path.join(args.out, "best.pt"))

    # Pick the decision threshold on validation only, then report test once with the best checkpoint
    ckpt = torch.load(os.path.join(args.out, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_probs, val_labels = predict(model, loaders["val"], device)
    threshold = max(np.arange(0.1, 0.91, 0.05), key=lambda t: f1_score(val_labels, val_probs >= t, zero_division=0))
    test = metrics(*predict(model, loaders["test"], device), float(threshold))
    test_labels = np.array([y for _, y in test_ds.items])
    test["majority_baseline_accuracy"] = float(max(test_labels.mean(), 1 - test_labels.mean()))
    test["n_test"], test["n_test_debris"] = len(test_labels), int(test_labels.sum())
    print(json.dumps(test, indent=2))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"test": test, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()

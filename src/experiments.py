"""Hyperparameter and loss-function experiments for the report.

Reuses the tested functions from each classifier file; nothing here changes the models themselves.
Run one group at a time:
    .venv/bin/python -m src.experiments nb        # Naive Bayes: alpha, threshold, split seeds, confusion matrix
    .venv/bin/python -m src.experiments linear    # Linear: bias, L2 vs cross-entropy
    .venv/bin/python -m src.experiments mlp       # MLP: hidden sizes, learning rate, L2 vs cross-entropy
    .venv/bin/python -m src.experiments cnn       # CNN: L2 vs cross-entropy, padding
Results are printed and saved to results/experiments/<group>.json; figures go to results/figures/.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import load_mnist, split
from src.device import get_device
from src.knn import accuracy, confusion_matrix
from src import naive_bayes, linear, mlp, cnn
from src.torch_data import make_loader

OUT_DIR = Path("results/experiments")
FIG_DIR = Path("results/figures")


def save(group, results):
    """Write one experiment group's results to results/experiments/<group>.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{group}.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results/experiments/{group}.json")


def train_torch(model, loss_name, train_loader, val_loader, y_val, lr, epochs, device):
    """Train a PyTorch model with SGD and return per-epoch validation accuracy and runtime.

    loss_name: "ce" (cross-entropy, integer labels) or "l2" (MSE against one-hot targets).
    The two training loops already exist: mlp.train_one_epoch (CE) and linear.train_one_epoch (L2).
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if loss_name == "ce":
        loss_fn, step = nn.CrossEntropyLoss(), mlp.train_one_epoch
    else:
        loss_fn, step = nn.MSELoss(), linear.train_one_epoch

    accs, losses = [], []
    start = time.perf_counter()
    for _ in range(epochs):
        losses.append(step(model, train_loader, loss_fn, optimizer, device))
        accs.append(float(accuracy(linear.evaluate(model, val_loader, device), y_val)))
    return {"val_acc": accs, "loss": losses, "final": accs[-1], "best": max(accs),
            "seconds": round(time.perf_counter() - start, 1)}


# ---------------------------------------------------------------- Naive Bayes

def run_nb(X, y):
    X_train, y_train, X_val, y_val = split(X, y)
    Xtr, Xva = X_train.reshape(-1, 784), X_val.reshape(-1, 784)
    results = {}

    # 1. Smoothing strength alpha
    results["alpha"] = {}
    for alpha in (0.001, 0.01, 0.1, 1, 10, 100):
        priors, probs = naive_bayes.fit(naive_bayes.binarize(Xtr), y_train, alpha=alpha)
        acc = accuracy(naive_bayes.predict_loop(naive_bayes.binarize(Xva), priors, probs), y_val)
        results["alpha"][str(alpha)] = float(acc)
        print(f"NB alpha={alpha:<6} acc={acc:.4f}")

    # 2. Binarization threshold (the assignment fixes 0.5; this shows how much it matters)
    results["threshold"] = {}
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        priors, probs = naive_bayes.fit((Xtr > t).astype(int), y_train, alpha=1)
        acc = accuracy(naive_bayes.predict_loop((Xva > t).astype(int), priors, probs), y_val)
        results["threshold"][str(t)] = float(acc)
        print(f"NB threshold={t} acc={acc:.4f}")

    # 3. Different random splits: how much does the accuracy depend on which images land in validation?
    results["split_seed"] = {}
    for seed in (0, 1, 2, 3, 42):
        a, b, c, d = split(X, y, seed=seed)
        priors, probs = naive_bayes.fit(naive_bayes.binarize(a.reshape(-1, 784)), b, alpha=1)
        acc = accuracy(naive_bayes.predict_loop(naive_bayes.binarize(c.reshape(-1, 784)), priors, probs), d)
        results["split_seed"][str(seed)] = float(acc)
        print(f"NB split seed={seed} acc={acc:.4f}")
    accs = list(results["split_seed"].values())
    print(f"NB split seeds: mean {np.mean(accs):.4f}, std {np.std(accs):.4f}")

    # 4. Confusion matrix for the default model (alpha 1, threshold 0.5, seed 42)
    priors, probs = naive_bayes.fit(naive_bayes.binarize(Xtr), y_train, alpha=1)
    cm = confusion_matrix(naive_bayes.predict_loop(naive_bayes.binarize(Xva), priors, probs), y_val)
    results["confusion_matrix"] = cm.tolist()
    print(cm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    al = results["alpha"]
    axes[0].plot([float(a) for a in al], list(al.values()), marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Smoothing alpha (log scale)")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Naive Bayes: smoothing")
    th = results["threshold"]
    axes[1].plot([float(t) for t in th], list(th.values()), marker="o")
    axes[1].set_xlabel("Binarization threshold")
    axes[1].set_title("Naive Bayes: threshold")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_naive_bayes.png", dpi=150)
    save("nb", results)


# ---------------------------------------------------------------- Linear

def run_linear(X, y, device):
    X_train, y_train, X_val, y_val = split(X, y)
    tl = make_loader(X_train, y_train, 64, shuffle=True)
    vl = make_loader(X_val, y_val, 64)
    results = {}
    runs = [
        ("L2, no bias (baseline)", "l2", False, 0.01),
        ("L2, with bias", "l2", True, 0.01),
        ("Cross-entropy, no bias", "ce", False, 0.01),
        ("Cross-entropy, no bias, lr 0.001", "ce", False, 0.001),
    ]
    for name, loss_name, bias, lr in runs:
        torch.manual_seed(42)
        r = train_torch(linear.build_model(bias=bias), loss_name, tl, vl, y_val, lr, 10, device)
        results[name] = r
        print(f"Linear {name:<35} final {r['final']:.4f}  best {r['best']:.4f}  {r['seconds']}s")
    save("linear", results)


# ---------------------------------------------------------------- MLP

def run_mlp(X, y, device):
    X_train, y_train, X_val, y_val = split(X, y)
    tl = make_loader(X_train, y_train, 64, shuffle=True)
    vl = make_loader(X_val, y_val, 64)
    results = {"hidden": {}, "lr": {}, "loss": {}}
    EPOCHS = 10

    for hidden in [(32,), (128,), (256, 128), (512, 256)]:
        torch.manual_seed(42)
        model = mlp.build_mlp(hidden)
        n_params = sum(p.numel() for p in model.parameters())
        r = train_torch(model, "ce", tl, vl, y_val, 0.05, EPOCHS, device)
        r["params"] = n_params
        results["hidden"][str(hidden)] = r
        print(f"MLP hidden={str(hidden):<11} params={n_params:>7}  final {r['final']:.4f}  {r['seconds']}s")

    for lr in (0.005, 0.05, 0.5):
        torch.manual_seed(42)
        r = train_torch(mlp.build_mlp((256, 128)), "ce", tl, vl, y_val, lr, EPOCHS, device)
        results["lr"][str(lr)] = r
        print(f"MLP lr={lr:<6} final {r['final']:.4f}  {r['seconds']}s")

    for loss_name in ("ce", "l2"):
        torch.manual_seed(42)
        lr = 0.05 if loss_name == "ce" else 0.01  # L2 diverges at 0.05 on [-1, 1] inputs (see linear run)
        r = train_torch(mlp.build_mlp((256, 128)), loss_name, tl, vl, y_val, lr, EPOCHS, device)
        results["loss"][loss_name] = r
        print(f"MLP loss={loss_name} lr={lr} final {r['final']:.4f}  {r['seconds']}s")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    epochs = range(1, EPOCHS + 1)
    for title, key, ax in [("Hidden sizes (lr 0.05)", "hidden", axes[0]),
                           ("Learning rate (256, 128)", "lr", axes[1]),
                           ("Loss function (256, 128)", "loss", axes[2])]:
        for label, r in results[key].items():
            ax.plot(epochs, r["val_acc"], marker=".", label=label)
        ax.set_title(f"MLP: {title}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Validation accuracy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_mlp.png", dpi=150)
    save("mlp", results)


# ---------------------------------------------------------------- CNN

def run_cnn(X, y, device):
    X_train, y_train, X_val, y_val = split(X, y)
    tl = make_loader(X_train, y_train, 64, shuffle=True, flatten=False)
    vl = make_loader(X_val, y_val, 64, flatten=False)
    results = {}
    EPOCHS = 5
    runs = [
        ("Cross-entropy (baseline)", "ce", 0.05, (32, 64)),
        ("L2 / MSE", "l2", 0.01, (32, 64)),
        ("Cross-entropy, small (8, 16)", "ce", 0.05, (8, 16)),
    ]
    for name, loss_name, lr, channels in runs:
        torch.manual_seed(42)
        model = cnn.build_cnn(channels=channels)
        n_params = sum(p.numel() for p in model.parameters())
        r = train_torch(model, loss_name, tl, vl, y_val, lr, EPOCHS, device)
        r["params"] = n_params
        results[name] = r
        print(f"CNN {name:<30} params={n_params:>6}  final {r['final']:.4f}  {r['seconds']}s")
    save("cnn", results)


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "nb"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_mnist()
    if group == "nb":
        run_nb(X, y)
    else:
        device = get_device()
        {"linear": run_linear, "mlp": run_mlp, "cnn": run_cnn}[group](X, y, device)

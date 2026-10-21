"""K-Nearest Neighbors classifier for MNIST (NumPy only)."""
import numpy as np

from src.data import load_mnist, split


def squared_distances(X_batch, X_train):
    """Squared Euclidean distance between every batch image and every training image.

    X_batch: (B, 784)   X_train: (N, 784)   returns: (B, N)
    """
    distances = np.sum(X_batch**2, axis=1, keepdims=True) + np.sum(X_train**2, axis=1) - 2 * np.dot(X_batch, X_train.T)
    return np.maximum(distances, 0.0)


def nearest_neighbors(dists, K):
    """Indices of the K closest training images per row, sorted nearest first.

    dists: (B, N)   returns: (B, K)
    """
    return np.argsort(dists, axis=1)[:, :K]


def vote(neighbor_labels, k):
    """Majority vote among the first k neighbors of each row.
    Ties are broken by the nearest neighbor.

    neighbor_labels: (B, K) labels, nearest first   returns: (B,) predictions
    """
    top_k = neighbor_labels[:, :k]
    predictions = []
    
    for row in top_k:
        # np.bincount is faster for small integers like 0-9
        counts = np.bincount(row)
        max_count = counts.max()
        
        # Find all digits that tied for the max votes
        winners = np.where(counts == max_count)[0]
        
        if len(winners) == 1:
            predictions.append(winners[0])
        else:
            # Tie-breaker: Walk through the row (nearest to furthest)
            # and pick the first label that is among the tied winners.
            for label in row:
                if label in winners:
                    predictions.append(label)
                    break
                    
    return np.array(predictions)



def predict(X_val, X_train, y_train, ks=(1, 3, 5), batch_size=500):
    """Predict labels for X_val for every k in ks, processing X_val in batches.

    returns: dict {k: (M,) predictions}
    """
    K = max(ks)
    
    # 1. Make an empty list for every k BEFORE the loop
    collected_preds = {k: [] for k in ks}

    for x in range(0, X_val.shape[0], batch_size):
        X_batch = X_val[x:x + batch_size]
        dists = squared_distances(X_batch, X_train)
        neighbor_indices = nearest_neighbors(dists, K)
        neighbor_labels = y_train[neighbor_indices]
        
        for k in ks:
            preds = vote(neighbor_labels, k)
            # 2. Append this batch's votes to the list for this k (do not overwrite)
            collected_preds[k].append(preds)

    # 3. Glue each k's list of batch arrays together AFTER the loop
    for k in ks:
        collected_preds[k] = np.concatenate(collected_preds[k])
        
    # 4. Return ONCE at the very end
    return collected_preds


def accuracy(preds, y_true):
    """Fraction of predictions that match the true labels."""
    return np.mean(preds == y_true)


def confusion_matrix(preds, y_true, n_classes=10):
    """Count of (true digit, predicted digit) pairs.

    returns: (n_classes, n_classes), rows = true digit, columns = predicted digit
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # add 1 at (true, pred) for every image; np.add.at handles repeated pairs correctly
    np.add.at(cm, (y_true, preds), 1)
    return cm


# --- Previous version of the main block (1,000-image development run, k = 1, 3, 5) ---
# Kept for reference. Results are in results/log.md (2026-10-20, run 1).
# if __name__ == "__main__":
#     X, y = load_mnist()
#     X_train, y_train, X_val, y_val = split(X, y)
#
#     X_train = X_train.reshape(-1, 784)
#     X_val = X_val.reshape(-1, 784)
#
#     X_val = X_val[:1000]
#     y_val = y_val[:1000]
#
#     # Run predict ONCE to compute distances once, getting back all k predictions
#     all_preds = predict(X_val, X_train, y_train, ks=(1, 3, 5), batch_size=500)
#
#     for k in (1, 3, 5):
#         acc = accuracy(all_preds[k], y_val)
#         print(f"Accuracy for k={k}: {acc:.4f}")


if __name__ == "__main__":
    import time
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")  # save figures to files without opening windows
    import matplotlib.pyplot as plt

    FIG_DIR = Path("results/figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_mnist()
    X_train, y_train, X_val, y_val = split(X, y)

    # KNN works on flat 784-pixel vectors; keep the 28x28 version of X_val for plotting
    X_train_flat = X_train.reshape(-1, 784)
    X_val_flat = X_val.reshape(-1, 784)

    # Full validation set. Neighbors are found once for the largest k, so extra k values are cheap.
    ks = (1, 3, 5, 7, 9, 15)
    start = time.perf_counter()
    all_preds = predict(X_val_flat, X_train_flat, y_train, ks=ks, batch_size=500)
    elapsed = time.perf_counter() - start

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Time: {elapsed:.1f}s")
    accs = {k: accuracy(all_preds[k], y_val) for k in ks}
    for k in ks:
        n_wrong = int(np.sum(all_preds[k] != y_val))
        print(f"k={k:>2}: accuracy {accs[k]:.4f}  ({n_wrong} wrong)")

    best_k = max(ks, key=lambda k: accs[k])
    preds = all_preds[best_k]
    cm = confusion_matrix(preds, y_val)
    print(f"\nConfusion matrix for best k={best_k} (rows = true, columns = predicted):")
    print(cm)

    # Most frequent mistakes: largest off-diagonal cells
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    print("\nMost common mistakes (true -> predicted: count):")
    for flat_idx in np.argsort(off_diag, axis=None)[::-1][:5]:
        t, p_ = np.unravel_index(flat_idx, cm.shape)
        print(f"  {t} -> {p_}: {off_diag[t, p_]}")

    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-digit accuracy:")
    for d in range(10):
        print(f"  {d}: {per_class[d]:.4f}")

    # Figure 1: accuracy vs k
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(list(ks), [accs[k] for k in ks], marker="o")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("KNN accuracy vs k")
    ax.set_xticks(list(ks))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "knn_accuracy_vs_k.png", dpi=150)

    # Figure 2: confusion matrix (diagonal hidden so the mistakes are visible)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(off_diag, cmap="Reds")
    for t in range(10):
        for p_ in range(10):
            ax.text(p_, t, cm[t, p_], ha="center", va="center", fontsize=7,
                    color="gray" if t == p_ else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(f"KNN confusion matrix (k={best_k}); color = mistakes only")
    fig.colorbar(im, ax=ax, label="Count of mistakes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "knn_confusion_matrix.png", dpi=150)

    # Figure 3: first 20 misclassified validation images
    wrong = np.flatnonzero(preds != y_val)[:20]
    fig, axes = plt.subplots(4, 5, figsize=(7, 6))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(X_val[i], cmap="gray")
        ax.set_title(f"true {y_val[i]}, pred {preds[i]}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"KNN misclassified examples (k={best_k})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "knn_misclassified.png", dpi=150)

    print(f"\nFigures saved to {FIG_DIR}/")

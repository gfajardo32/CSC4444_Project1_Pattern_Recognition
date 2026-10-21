# Experiment Log

Setup for every run unless noted: seed 42, 80/20 split (48,000 train / 12,000 validation),
pixels scaled to [0, 1], images flattened to 784 values.

---

## 2026-10-20, run 1: KNN, development subset

- **What:** KNN on the first 1,000 validation images only (quick check that the code works)
- **Distance:** squared Euclidean (same neighbour ranking as Euclidean)
- **Tie rule:** nearest neighbour among the tied digits wins
- **Command:** `.venv/bin/python -m src.knn` (old main block, now commented out in `src/knn.py`)

| k | Accuracy |
|---|---|
| 1 | 0.9740 |
| 3 | 0.9720 |
| 5 | 0.9720 |

**Notes:** On 1,000 images each image is 0.1%, so k=1 vs k=3 is a difference of 2 images.
Too small to conclude anything. Next: run on all 12,000.

---

## 2026-10-20, run 2: KNN, full validation set

- **What:** KNN on all 12,000 validation images, k = 1, 3, 5, 7, 9, 15
- **Distance / tie rule:** same as run 1
- **Batch size:** 500 validation images per distance computation
- **Runtime:** 66.3 s on Apple M-series CPU (NumPy), neighbours computed once for all k
- **Command:** `.venv/bin/python -m src.knn`

| k | Accuracy | Wrong (of 12,000) |
|---|---|---|
| 1 | 0.9721 | 335 |
| 3 | 0.9715 | 342 |
| 5 | 0.9714 | 343 |
| 7 | 0.9708 | 350 |
| 9 | 0.9702 | 357 |
| 15 | 0.9654 | 415 |

**Confusion matrix (k=1), rows = true digit, columns = predicted digit:**

```
[[1208    1    1    0    0    2    3    1    2    0]
 [   0 1341    3    0    1    1    0    1    2    0]
 [   6    9 1166    5    0    2    0   12    1    1]
 [   2    1    8 1160    0   15    0    9    6    7]
 [   0    8    0    0 1123    1    3    3    1   35]
 [   2    0    1   18    2 1051    7    0    5    6]
 [   8    1    0    0    3    3 1186    0    0    0]
 [   1    7    2    1    3    0    0 1235    2    6]
 [   2    6    3   19    4   11    5    3 1090    8]
 [   3    1    0    7   10    2    0   19    1 1105]]
```

**Most common mistakes (true -> predicted):** 4 -> 9 (35), 8 -> 3 (19), 9 -> 7 (19), 5 -> 3 (18), 3 -> 5 (15)

**Per-digit accuracy:** 0: 0.9918, 1: 0.9941, 2: 0.9700, 3: 0.9603, 4: 0.9566,
5: 0.9625, 6: 0.9875, 7: 0.9825, 8: 0.9470, 9: 0.9625

**Figures:** `results/figures/knn_accuracy_vs_k.png`, `knn_confusion_matrix.png`, `knn_misclassified.png`

**Notes / questions for the report:**
- Accuracy drops steadily as k grows; k=1 is best. Gap k=1 vs k=3 is 7 images (0.06%), so small;
  k=15 is clearly worse (80 more mistakes).
- Hardest digits: 8 (94.7%) and 4 (95.7%). Easiest: 1 and 0 (above 99%).
- Mistakes come in shape-similar pairs: 4/9, 3/5/8, 7/9. Pixel distance can't tell
  digits apart when their strokes mostly overlap.
- TODO: look at `knn_misclassified.png`, are these hard for a human too?

---

## 2026-10-20, run 3: Naive Bayes (Bernoulli), full validation set

- **What:** Naive Bayes on all 12,000 validation images
- **Preprocessing:** pixels binarized, 1 if pixel > 0.5 else 0
- **Smoothing:** Laplace, alpha = 1, so P(pixel on | digit) = (count + 1) / (N_digit + 2)
- **Scoring:** sum of log-probabilities (avoids underflow from multiplying 784 small numbers)
- **Prediction:** `predict_loop` (one image at a time, broadcast against all 10 digits)
- **Runtime:** about 1.6 s for the whole script (load from cache, fit, predict, plot)
- **Command:** `.venv/bin/python -m src.naive_bayes`

| Accuracy | Wrong (of 12,000) |
|---|---|
| 0.8393 | 1,928 |

**Figure:** `results/figures/naive_bayes_maps.png` (P(pixel on | digit) for each digit as a 28x28 image)

**Notes / questions for the report:**
- About 13 points below KNN (0.8393 vs 0.9721), but about 40x faster.
- Model size: 10 x 784 probabilities + 10 priors (7,850 numbers) vs KNN storing
  48,000 x 784 (37.6 million numbers).
- Why lower: the independence assumption ignores that neighbouring pixels move together,
  and binarizing throws away the grey stroke edges.
- TODO: confusion matrix for Naive Bayes; try alpha = 0.1 and 10.

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

---

## 2026-10-20, run 4: Linear classifier (PyTorch), full validation set

- **What:** y = Wx, `nn.Linear(784, 10, bias=False)`, trained with L2 loss (`nn.MSELoss`)
  against one-hot targets, plain SGD
- **Preprocessing:** pixels scaled to [-1, 1] (`make_loader` in `src/torch_data.py`), flattened to 784
- **Hyperparameters:** lr 0.01, batch size 64, 20 epochs, no bias, `torch.manual_seed(42)`, device mps
- **Runtime:** 27.5 s (20 epochs, including a validation pass after each epoch)
- **Command:** `.venv/bin/python -m src.linear`

**Failed first attempt (lr 0.05):** loss was `nan` from epoch 1 and accuracy 0.1015 (constant guess).
Cause: with [-1, 1] scaling the black background is -1, not 0, so every input vector is large
(mean squared norm about 724). Each gradient step scales with the input size, so lr 0.05 overshoots
and the weights blow up. Learning rate check, 3 epochs each:

| lr | Loss, epochs 1-3 | Val accuracy |
|---|---|---|
| 0.05 | nan, nan, nan | 0.1015 |
| 0.01 | 0.065, 0.049, 0.046 | 0.8382 |
| 0.005 | 0.075, 0.054, 0.050 | 0.8255 |
| 0.001 | 0.110, 0.075, 0.066 | 0.7255 |

**Training curve (lr 0.01):**

| Epoch | 1 | 5 | 10 | 15 | 17 | 20 |
|---|---|---|---|---|---|---|
| Loss | 0.0650 | 0.0438 | 0.0414 | 0.0406 | 0.0404 | 0.0401 |
| Val acc | 0.8087 | 0.8383 | 0.8527 | 0.8512 | 0.8535 | 0.8454 |

**Final (epoch 20):** accuracy 0.8454, 1,855 wrong of 12,000. Best epoch was 17 (0.8535).
Loss falls every epoch but accuracy wobbles by about 1 point (mini-batch SGD noise).

**Confusion matrix (rows = true, columns = predicted):**

```
[[1152    3    9    3    3    9   22    1   16    0]
 [   1 1320    4    2    3    6    2    1    9    1]
 [  18   84  954   26   19    3   46   16   32    4]
 [   7   54   45  964    7   21   17   36   25   32]
 [   3   30   17    0 1020   10    9    7   15   63]
 [  24   34   11   67   39  768   57    8   43   41]
 [  21   21    9    0   16   10 1120    0    3    1]
 [  10   52    4    8   44    1    1 1077    2   58]
 [   6  146   20   30   25   32   14    2  835   41]
 [  11   28    6   15   66    0    1   82    4  935]]
```

**Most common mistakes (true -> predicted):** 8 -> 1 (146), 2 -> 1 (84), 9 -> 7 (82),
5 -> 3 (67), 9 -> 4 (66), 4 -> 9 (63)

**Figure:** `results/figures/linear_weights.png` (each row of W as a 28x28 image, red = positive,
blue = negative, color range symmetric so white = 0)

**Notes / questions for the report:**
- Similar to Naive Bayes (0.8393), well below KNN (0.9721).
- Over-predicts 1: column 1 has 1,772 predictions but only 1,347 true 1s. Many 8s and 2s get called 1.
  Question: why would L2 loss with one-hot targets favour 1? (Hint: 1 has the fewest "on" pixels.)
- Weakest digits: 5 (768 of 1,092 = 70.3%) and 8 (835 of 1,151 = 72.5%).
- One template per digit can't represent all the ways a digit can be written; the MLP's hidden
  layer can learn several partial features and combine them.
- TODO: try lr 0.005 / 0.02, bias on, more epochs; try cross-entropy loss instead of L2.

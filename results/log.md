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

---

## 2026-10-20, run 5: MLP (PyTorch), full validation set

- **What:** 784 -> 256 -> ReLU -> 128 -> ReLU -> 10 (`build_mlp((256, 128))`), 235,146 parameters
- **Loss:** cross-entropy (`nn.CrossEntropyLoss`, integer labels, no one-hot). The model outputs raw
  scores with no softmax layer, because `CrossEntropyLoss` applies softmax internally.
- **Optimizer:** plain SGD, lr 0.05, batch size 64, 15 epochs, `torch.manual_seed(42)`, device mps
- **Preprocessing:** same loader as the linear classifier ([-1, 1], flattened)
- **Runtime:** 26.5 s (15 epochs, including a validation pass after each epoch)
- **Command:** `.venv/bin/python -m src.mlp`

**Training curve:**

| Epoch | 1 | 3 | 5 | 10 | 13 | 15 |
|---|---|---|---|---|---|---|
| Loss | 0.5621 | 0.1779 | 0.1161 | 0.0567 | 0.0398 | 0.0326 |
| Val acc | 0.9070 | 0.9497 | 0.9668 | 0.9756 | 0.9770 | 0.9769 |

**Final (epoch 15):** accuracy 0.9769, 277 wrong of 12,000. Best epochs were 13 and 14 (0.9770).

**Confusion matrix (rows = true, columns = predicted):**

```
[[1201    0    0    1    0    2    6    2    3    3]
 [   0 1322    4    3    2    2    2    2    9    3]
 [   2    1 1163   10    4    0    5    8    7    2]
 [   2    0    6 1182    0    9    1    3    1    4]
 [   0    0    3    1 1132    0   11    1    1   25]
 [   2    0    0   15    2 1054    6    1    7    5]
 [   4    0    0    0    3    1 1190    0    3    0]
 [   5    4    2    4    2    1    0 1228    3    8]
 [   1    3    0   12    1    4    3    0 1125    2]
 [   3    0    0    3    6    3    0    4    3 1126]]
```

**Most common mistakes (true -> predicted):** 4 -> 9 (25), 5 -> 3 (15), 8 -> 3 (12), 4 -> 6 (11), 2 -> 3 (10)

**Weakest digits:** 4 (1,132 of 1,174 = 96.4%) and 5 (1,054 of 1,092 = 96.5%)

**Figure:** `results/figures/mlp_learning_curve.png` (training loss and validation accuracy per epoch)

**Notes / questions for the report:**
- Best so far: beats KNN (0.9721) by about 0.5 points, and needs only 235k weights at prediction time
  instead of storing all 48,000 training images.
- The linear classifier's over-prediction of 1 is gone (8 -> 1 drops from 146 to 3). The hidden
  layer can combine partial stroke features instead of one template per digit.
- Overfitting begins: training loss keeps falling (0.0567 at epoch 10 -> 0.0326 at epoch 15) but
  validation accuracy stays flat around 0.977. More epochs would not help much.
- 4 -> 9 is still the top mistake, same as KNN.
- TODO: try hidden sizes (128,), (512, 256); lr 0.01 / 0.1; momentum 0.9; MSE loss instead of cross-entropy.

---

## 2026-10-20, run 6: CNN (PyTorch), full validation set

- **What:** Conv(1->32, 3x3) -> ReLU -> MaxPool(2) -> Conv(32->64, 3x3) -> ReLU -> MaxPool(2)
  -> Flatten -> Linear(1600, 10), 34,826 parameters, no softmax layer (inside `CrossEntropyLoss`)
- **Feature-map size:** 28 -> 26 (conv) -> 13 (pool) -> 11 (conv) -> 5 (pool; the 11th row is dropped),
  so the Linear layer gets 64 x 5 x 5 = 1,600 inputs
- **Loss / optimizer:** cross-entropy, plain SGD, lr 0.05, batch size 64, 10 epochs,
  `torch.manual_seed(42)`, device mps
- **Preprocessing:** [-1, 1], kept as (1, 28, 28) (`make_loader(..., flatten=False)`)
- **Runtime:** about 99 s (10 epochs, including a validation pass after each epoch)
- **Command:** `.venv/bin/python -m src.cnn`

**Training curve:**

| Epoch | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| Loss | 0.2825 | 0.0870 | 0.0550 | 0.0422 | 0.0338 | 0.0287 |
| Val acc | 0.9703 | 0.9788 | 0.9831 | 0.9860 | 0.9876 | 0.9870 |

**Final (epoch 10):** accuracy 0.9870, 156 wrong of 12,000. Best epoch was 8 (0.9876).

**Confusion matrix (rows = true, columns = predicted):**

```
[[1205    0    1    0    0    2    2    2    5    1]
 [   0 1335    5    0    0    1    0    6    2    0]
 [   0    1 1192    1    0    0    0    3    3    2]
 [   2    0    6 1187    0    6    0    5    1    1]
 [   0    2    3    0 1146    0    1    5    4   13]
 [   1    0    0    3    1 1074    5    2    4    2]
 [   3    1    0    0    2    3 1187    0    4    1]
 [   0    1    3    3    0    1    0 1245    3    1]
 [   1    0    4    2    0    2    0    1 1141    0]
 [   0    0    0    3    2    5    0    3    3 1132]]
```

**Most common mistakes (true -> predicted):** 4 -> 9 (13), then several at 5-6: 1 -> 7, 3 -> 2, 3 -> 5

**Weakest digits:** 4 (1,146 of 1,174 = 97.6%) and 3 (1,187 of 1,209 = 98.2%)

**Figures:** `results/figures/cnn_learning_curve.png`, `results/figures/cnn_filters.png`
(32 first-layer 3x3 filters, shared symmetric color range so white = 0)

**Notes / questions for the report:**
- Best model: 156 mistakes vs 277 for the MLP (44% fewer).
- 7x fewer parameters than the MLP (34,826 vs 235,146) yet more accurate: the same small filters
  are reused at every position in the image (weight sharing), and pooling tolerates small shifts.
- 97% after a single epoch.
- Many first-layer filters look like edge detectors (light on one side, dark on the other).
- 4 -> 9 is the top mistake for every method except Naive Bayes; it shrinks from 35 (KNN) and
  25 (MLP) to 13 here.
- TODO: try padding=1, kernel 5, more channels, more epochs; compare to MSE loss.

---

## Summary: all five methods (validation set of 12,000, seed 42)

| Method | Library | Accuracy | Wrong | Runtime | Model size |
|---|---|---|---|---|---|
| KNN (k=1) | NumPy | 0.9721 | 335 | 66 s (prediction) | stores all 48,000 x 784 training pixels |
| Naive Bayes | NumPy | 0.8393 | 1,928 | about 1.6 s (whole script) | 10 x 784 probabilities + 10 priors |
| Linear (L2) | PyTorch | 0.8454 | 1,855 | 27.5 s (20 epochs) | 7,840 weights |
| MLP | PyTorch | 0.9769 | 277 | 26.5 s (15 epochs) | 235,146 parameters |
| CNN | PyTorch | 0.9870 | 156 | about 99 s (10 epochs) | 34,826 parameters |

---

## 2026-10-20, run 7: Experiments (`src/experiments.py`)

Raw numbers per epoch: `results/experiments/{nb,linear,mlp,cnn}.json`.
Figures: `results/figures/exp_naive_bayes.png`, `results/figures/exp_mlp.png`.
All on the seed-42 split unless noted; PyTorch runs use `torch.manual_seed(42)`, batch 64, SGD.

### Naive Bayes (`... -m src.experiments nb`)

| alpha | 0.001 | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|---|
| acc | 0.8404 | 0.8403 | 0.8403 | 0.8393 | 0.8363 | 0.8235 |

| threshold | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| acc | 0.8344 | 0.8377 | 0.8393 | 0.8308 | 0.8142 |

Different split seeds (alpha 1, threshold 0.5): 0: 0.8303, 1: 0.8355, 2: 0.8338, 3: 0.8341, 42: 0.8393,
mean 0.8346, std 0.0029. Seed 42 happened to be the best of the five splits.

Confusion matrix (alpha 1, rows = true, columns = predicted):

```
[[1078    0   11    5    4   54   34    0   30    2]
 [   0 1272    7    6    0   23    5    0   32    4]
 [  17   24  991   50   19    6   39    7   47    2]
 [   7   19   50  989    3   35   10   10   51   34]
 [   5    7   11    0  924    6   15    2   26  178]
 [  24   19   13  123   33  798   26    2   29   25]
 [   9   23   13    1   12   27 1105    0   11    0]
 [   6   27   10    4   37    1    0 1073   24   75]
 [   3   58   16   84   13   43    5    4  887   38]
 [   9   18    7   14   85    4    1   27   28  955]]
```

Top mistakes: 4 -> 9 (178), 5 -> 3 (123), 9 -> 4 (85), 8 -> 3 (84), 7 -> 9 (75).

Notes: smoothing barely matters with 4,800 images per digit (it only changes counts by 1 out of
thousands); large alpha (100) blurs the maps and hurts. Threshold 0.5 is the best of those tried.
Split-to-split spread (std 0.3 points) is about the same size as the alpha effect, so alpha
differences below about 0.3 points are noise.

### Linear (`... -m src.experiments linear`, 10 epochs, lr 0.01 unless noted)

| Variant | Final acc | Best acc | Time |
|---|---|---|---|
| L2, no bias (baseline) | 0.8527 | 0.8527 | 17.9 s |
| L2, with bias | 0.8490 | 0.8496 | 20.3 s |
| Cross-entropy, no bias | 0.9110 | 0.9125 | 19.1 s |
| Cross-entropy, no bias, lr 0.001 | 0.8846 | 0.8846 | 17.9 s |

Notes: switching L2 to cross-entropy adds about 6 points with the exact same model. L2 pushes every
score toward exactly 0 or 1, so it wastes effort on images that are already classified correctly;
cross-entropy only cares that the right score is the largest. Bias makes no real difference.

### MLP (`... -m src.experiments mlp`, 10 epochs)

| Hidden sizes | Params | Final acc | Time |
|---|---|---|---|
| (32,) | 25,450 | 0.9533 | 16.1 s |
| (128,) | 101,770 | 0.9717 | 14.8 s |
| (256, 128) | 235,146 | 0.9756 | 17.8 s |
| (512, 256) | 535,818 | 0.9716 | 27.4 s |

| Learning rate (256, 128) | 0.005 | 0.05 | 0.5 |
|---|---|---|---|
| Final acc | 0.9235 | 0.9756 | 0.1851 (diverged) |

| Loss (256, 128) | Cross-entropy, lr 0.05 | L2 / MSE, lr 0.01 |
|---|---|---|
| Final acc | 0.9756 | 0.8855 |

Notes: bigger is not always better; (512, 256) has 2x the parameters of (256, 128) but is not more
accurate after 10 epochs. lr 0.005 is too slow, lr 0.5 blows up. L2 costs the MLP about 9 points.

### CNN (`... -m src.experiments cnn`, 5 epochs)

| Variant | Params | Final acc | Time |
|---|---|---|---|
| Cross-entropy (32, 64), lr 0.05 | 34,826 | 0.9844 | 49.9 s |
| L2 / MSE (32, 64), lr 0.01 | 34,826 | 0.9369 | 52.0 s |
| Cross-entropy, small (8, 16), lr 0.05 | 5,258 | 0.9808 | 24.4 s |

Notes: a CNN with only 5,258 parameters (8 and 16 filters) still beats the 235k-parameter MLP.
L2 again hurts (about 5 points).

### Cross-cutting finding
The loss function matters more than the model size: with L2 loss the MLP (0.8855) is barely better
than the linear model with cross-entropy (0.9110).

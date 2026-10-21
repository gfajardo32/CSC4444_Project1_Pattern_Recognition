# CSC 4444 Project 1: Classifying MNIST Digits

Guido Fajardo, CSC 4444 Artificial Intelligence (Dong Lao), Louisiana State University, Fall 2026

Five classifiers for handwritten digits, all trained and tested on the same split
(48,000 training / 12,000 validation images, seed 42):

| Method | Library | File | Validation accuracy |
|---|---|---|---|
| K-Nearest Neighbors (k = 1) | NumPy | `src/knn.py` | 97.21% |
| Naive Bayes | NumPy | `src/naive_bayes.py` | 83.93% |
| Linear classifier (y = Wx, L2 loss) | PyTorch | `src/linear.py` | 84.54% |
| Multilayer perceptron (784-256-128-10) | PyTorch | `src/mlp.py` | 97.69% |
| Convolutional neural network | PyTorch | `src/cnn.py` | 98.70% |

The written report is in `report/` (`.docx` and `.pdf`).

## Folder structure

```
.
├── README.md
├── requirements.txt          pinned package versions
├── MNIST/                    the dataset: 60,000 PNGs in folders 0-9
├── src/
│   ├── data.py               load the PNGs into arrays (with a cache), seeded 80/20 split
│   ├── knn.py                K-Nearest Neighbors + confusion matrix, figures
│   ├── naive_bayes.py        Bernoulli Naive Bayes with Laplace smoothing
│   ├── torch_data.py         NumPy arrays -> PyTorch DataLoaders, pixels scaled to [-1, 1]
│   ├── device.py             picks NVIDIA GPU, Apple GPU, or CPU automatically
│   ├── linear.py             linear classifier
│   ├── mlp.py                multilayer perceptron
│   ├── cnn.py                convolutional neural network
│   └── experiments.py        hyperparameter and loss-function experiments
├── tests/                    unit tests for every method (pytest)
├── scripts/check_env.py      checks the installed packages and the dataset
├── results/
│   ├── log.md                every run: settings, accuracy, confusion matrices, notes
│   ├── figures/              figures produced by the scripts
│   └── experiments/          per-epoch numbers from src/experiments.py (JSON)
└── report/
    ├── MNIST_CLASSIFYING_DIGITS_REPORT_GuidoFajardo.docx / .pdf
    └── figures/              the figures used in the report, named by figure number
```

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

The MNIST images are included in `MNIST/`: 60,000 28x28 grayscale PNGs, one subfolder per digit
(`MNIST/0/` to `MNIST/9/`). The folder name is the label.

Check everything is in place:

```bash
python scripts/check_env.py
```

The first run converts the 60,000 PNGs into `MNIST/mnist_cache.npz` (about 10 s); later runs load the cache in under a second.

## Running

Run every command from the project root.

```bash
python -m src.knn            # about 1 min, saves 3 figures
python -m src.naive_bayes    # a few seconds
python -m src.linear         # about 30 s
python -m src.mlp            # about 30 s
python -m src.cnn            # about 100 s
```

Experiments (each group runs separately):

```bash
python -m src.experiments nb
python -m src.experiments linear
python -m src.experiments mlp
python -m src.experiments cnn
```

Times are from an Apple M-series laptop. The PyTorch models use a GPU if one is available
(`src/device.py`), otherwise the CPU.

## Tests

```bash
python -m pytest tests/
```

34 tests, using small hand-made inputs; they do not need the dataset.

## Reproducibility

- The train/validation split uses a fixed seed (42), and image files are read in sorted order,
  so every method sees exactly the same 12,000 validation images.
- PyTorch runs set `torch.manual_seed(42)`.
- All results and the settings that produced them are recorded in `results/log.md`.

## AI use

Claude (Anthropic) was used as a tutor and assistant. See the AI Use Statement in the report for
exactly what it did.

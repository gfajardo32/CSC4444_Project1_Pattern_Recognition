"""Tests for src/torch_data.py and src/linear.py. Use tiny fake data and the CPU so tests run fast."""
import numpy as np
import torch
from torch import nn

from src.torch_data import make_loader
from src.linear import build_model, train_one_epoch, evaluate

DEVICE = torch.device("cpu")  # tests always on CPU: fast and same result on every machine


def fake_data(n=8, seed=0):
    """n random 28x28 images in [0, 1] and labels 0-9, shaped like the output of load_mnist."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, size=n).astype(np.int64)
    return X, y


# ---------- make_loader ----------

def test_make_loader_flat_batch_shapes():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True)
    images, labels = next(iter(loader))
    
    assert images.shape == (4, 784)
    assert labels.shape == (4,)


def test_make_loader_cnn_batch_shape():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=False)
    images, labels = next(iter(loader))
    
    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)


def test_make_loader_scales_to_minus_one_one():
    # Build 1 image with known pixel values at the first 3 positions
    X = np.zeros((1, 28, 28), dtype=np.float32)
    X[0, 0, 0] = 0.0
    X[0, 0, 1] = 0.5
    X[0, 0, 2] = 1.0
    y = np.array([0], dtype=np.int64)
    
    loader = make_loader(X, y, batch_size=1, flatten=True, shuffle=False)
    images, _ = next(iter(loader))
    
    assert torch.isclose(images[0, 0], torch.tensor(-1.0))
    assert torch.isclose(images[0, 1], torch.tensor(0.0))
    assert torch.isclose(images[0, 2], torch.tensor(1.0))


def test_make_loader_types():
    X, y = fake_data(n=4)
    loader = make_loader(X, y, batch_size=4)
    images, labels = next(iter(loader))
    
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_make_loader_no_shuffle_keeps_order():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=3, shuffle=False)
    
    collected_labels = []
    for _, labels in loader:
        collected_labels.append(labels.numpy())
        
    all_labels = np.concatenate(collected_labels)
    np.testing.assert_array_equal(all_labels, y)


# ---------- build_model ----------

def test_build_model_weight_shape():
    model_no_bias = build_model(bias=False)
    assert model_no_bias.weight.shape == (10, 784)
    assert model_no_bias.bias is None
    
    model_with_bias = build_model(bias=True)
    assert model_with_bias.weight.shape == (10, 784)
    assert model_with_bias.bias is not None
    assert model_with_bias.bias.shape == (10,)


# ---------- train_one_epoch ----------

def test_train_one_epoch_lowers_loss():
    torch.manual_seed(0)
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True, shuffle=False)
    
    model = build_model(bias=False).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    first_loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
    last_loss = first_loss
    for _ in range(4):
        last_loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
        
    assert last_loss < first_loss


def test_train_one_epoch_changes_weights():
    torch.manual_seed(0)
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True)
    
    model = build_model(bias=False).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    initial_weights = model.weight.detach().clone()
    train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
    
    assert not torch.equal(model.weight, initial_weights)


# ---------- evaluate ----------

def test_evaluate_returns_numpy_predictions():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True, shuffle=False)
    model = build_model(bias=False).to(DEVICE)
    
    preds = evaluate(model, loader, DEVICE)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (8,)
    assert np.all((preds >= 0) & (preds <= 9))


def test_evaluate_does_not_change_weights():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True, shuffle=False)
    model = build_model(bias=False).to(DEVICE)
    
    initial_weights = model.weight.detach().clone()
    evaluate(model, loader, DEVICE)
    
    assert torch.equal(model.weight, initial_weights)

import numpy as np
from src.knn import nearest_neighbors, squared_distances, vote, predict, accuracy, confusion_matrix

def test_squared_distances_matches_broadcasting():
    rng = np.random.default_rng(42)
    X_batch = rng.random((3, 784))
    X_train = rng.random((10, 784))
    
    reference = np.sum((X_batch[:, np.newaxis, :] - X_train[np.newaxis, :, :])**2, axis=2)
    result = squared_distances(X_batch, X_train)
    
    assert result.shape == reference.shape  # should be (3, 10)
    assert np.allclose(result, reference)   # should be True

def test_nearest_neighbors_sorts_closest_first():
    dists = np.array([
        [10.0,  2.0,  5.0,  1.0,  8.0],
        [ 9.0, 10.0,  0.0,  3.0,  4.0]
    ])
    
    K = 3
    
    expected = np.array([
        [3, 1, 2],
        [2, 3, 4]
    ])
    
    result = nearest_neighbors(dists, K)
    
    assert result.shape == (2, K)
    np.testing.assert_array_equal(result, expected)


# --- UPDATED VOTE FUNCTION (For src.knn) ---
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


# --- UPDATED TEST ---
def test_vote_handles_ties_by_nearest_neighbor():
    row1 = np.array([[3, 8, 3, 5, 8]])
    row2 = np.array([[8, 3, 8, 3, 5]])
    
    # k=1: Only the first element counts ([3]) -> 3 wins
    assert vote(row1, k=1)[0] == 3
    
    # k=3: First three elements ([3, 8, 3]) -> 3 wins with two votes
    assert vote(row1, k=3)[0] == 3
    
    # k=5: All elements ([3, 8, 3, 5, 8]). Tie between 3 and 8. 
    # Nearest neighbor is 3, so 3 wins.
    assert vote(row1, k=5)[0] == 3
    
    # k=5 (Row 2): All elements ([8, 3, 8, 3, 5]). Tie between 8 and 3. 
    # Nearest neighbor is 8, so 8 wins (this used to fail and return 3!).
    assert vote(row2, k=5)[0] == 8


def test_predict_batches_and_returns_all_ks():
    X_train = np.array([
        [0.0, 0.0],  # index 0, label 0
        [1.0, 1.0],  # index 1, label 1
        [2.0, 2.0],  # index 2, label 0
        [3.0, 3.0],  # index 3, label 1
        [4.0, 4.0]   # index 4, label 2
    ])
    y_train = np.array([0, 1, 0, 1, 2])
    
    X_val = np.array([
        [1.0, 1.0], 
        [4.0, 4.0]
    ])
    
    result = predict(X_val, X_train, y_train, ks=(1, 3), batch_size=1)
    
    assert isinstance(result, dict)
    assert list(result.keys()) == [1, 3]
    
    # k=1: Nearest neighbors are exact matches (indices 1 and 4 -> labels 1 and 2)
    assert isinstance(result[1], np.ndarray)
    assert result[1].shape == (2,)
    np.testing.assert_array_equal(result[1], [1, 2])
    
    # k=3 for X_val[0]: Closest are indices 1, 0, 2 -> labels [1, 0, 0]. 0 wins.
    # k=3 for X_val[1]: Closest are indices 4, 3, 2 -> labels [2, 1, 0]. Tie, nearest is 2. 2 wins.
    assert isinstance(result[3], np.ndarray)
    assert result[3].shape == (2,)
    np.testing.assert_array_equal(result[3], [0, 2])


def test_accuracy_calculates_correct_fraction():
    preds = np.array([1, 2, 3, 4])
    y_true = np.array([1, 2, 0, 4])  # 3 out of 4 correct
    
    acc = accuracy(preds, y_true)
    assert acc == 0.75


def test_confusion_matrix_counts_true_vs_predicted():
    # Rows = true digit, columns = predicted digit
    y_true = np.array([0, 0, 1, 2, 2, 2])
    preds  = np.array([0, 1, 1, 2, 0, 2])

    cm = confusion_matrix(preds, y_true, n_classes=3)

    expected = np.array([
        [1, 1, 0],   # true 0: once right, once predicted as 1
        [0, 1, 0],   # true 1: right
        [1, 0, 2],   # true 2: twice right, once predicted as 0
    ])
    np.testing.assert_array_equal(cm, expected)
    assert cm.sum() == len(y_true)          # every prediction counted exactly once
    assert np.trace(cm) == np.sum(preds == y_true)  # diagonal = correct predictions

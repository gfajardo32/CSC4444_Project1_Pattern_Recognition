import numpy as np
from src.naive_bayes import binarize, fit, predict_loop


def test_binarize_threshold_logic():
    # Exactly 0.5 should be 0, anything strictly > 0.5 should be 1
    X = np.array([
        [0.0, 0.49, 0.5], 
        [0.51, 0.99, 1.0]
    ])
    
    expected = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    result = binarize(X)
    
    assert result.shape == X.shape
    np.testing.assert_array_equal(result, expected)

def test_fit_calculates_smoothed_probabilities():
    # 4 images, 3 pixels each
    # Class 0: first pixel is usually on
    # Class 1: last pixel is usually on
    X_bin = np.array([
        [1, 0, 0],  # Class 0
        [1, 1, 0],  # Class 0
        [0, 1, 1],  # Class 1
        [0, 0, 1]   # Class 1
    ])
    y = np.array([0, 0, 1, 1])
    
    # alpha=1 (Laplace smoothing)
    priors, pixel_probs = fit(X_bin, y, alpha=1)
    
    # 1. Check priors: 2 images of class 0, 2 of class 1 out of 4 total
    # Note: your fit function initializes a (10,) array, so indices 2-9 will be 0.0
    assert priors[0] == 0.5
    assert priors[1] == 0.5
    
    # 2. Check pixel probabilities for Class 0 (N_c = 2)
    # Pixel 1: 2 ones -> (2 + 1) / (2 + 2) = 3/4 = 0.75
    # Pixel 2: 1 one  -> (1 + 1) / (2 + 2) = 2/4 = 0.50
    # Pixel 3: 0 ones -> (0 + 1) / (2 + 2) = 1/4 = 0.25
    expected_probs_0 = np.array([0.75, 0.50, 0.25])
    np.testing.assert_allclose(pixel_probs[0][:3], expected_probs_0)
    
    # 3. Check pixel probabilities for Class 1 (N_c = 2)
    # Pixel 1: 0 ones -> 1/4 = 0.25
    # Pixel 2: 1 one  -> 2/4 = 0.50
    # Pixel 3: 2 ones -> 3/4 = 0.75
    expected_probs_1 = np.array([0.25, 0.50, 0.75])
    np.testing.assert_allclose(pixel_probs[1][:3], expected_probs_1)

def test_fit_smoothing_prevents_absolute_certainty():
    # Even if every single image in a class has a pixel OFF (0) or ON (1),
    # the probability should never be exactly 0.0 or 1.0.
    X_bin = np.array([
        [1, 0], 
        [1, 0]
    ])
    y = np.array([0, 0])
    
    _, pixel_probs = fit(X_bin, y, alpha=1)
    
    # Probabilities for class 0
    probs_0 = pixel_probs[0][:2]
    
    # Should not be exactly 1.0 or 0.0
    assert np.all(probs_0 < 1.0)
    assert np.all(probs_0 > 0.0)

# or predict, depending on what you named it

def test_predict_highest_probability_wins():
    # 2 classes, 3 pixels
    priors = np.array([0.5, 0.5])
    
    # Class 0 strongly prefers pixel 0. Class 1 strongly prefers pixel 2.
    probs = np.array([
        [0.9, 0.5, 0.1],  # Class 0 pixel probabilities
        [0.1, 0.5, 0.9]   # Class 1 pixel probabilities
    ])
    
    X_bin = np.array([
        [1, 0, 0],  # Should match Class 0
        [0, 0, 1]   # Should match Class 1
    ])
    
    expected = np.array([0, 1])
    
    result = predict_loop(X_bin, priors, probs)
    
    np.testing.assert_array_equal(result, expected)

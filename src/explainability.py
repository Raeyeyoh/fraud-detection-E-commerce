"""
explainability.py
-----------------
All SHAP computation logic for the fraud detection project.

Design choice: SHAP computation is separated from visualization so that
values can be computed once, cached, and reused across multiple plot types
without re-running expensive TreeExplainer calls.

Design choice: TreeExplainer is used (not KernelExplainer) because both
models are XGBoost — TreeExplainer is orders of magnitude faster and
produces exact Shapley values for tree models rather than approximations.
"""

import numpy as np
import pandas as pd
import shap

from typing import Optional


def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Create a SHAP TreeExplainer for a tree-based model.

    Args:
        model:        Trained XGBoost (or other tree) model.
        X_background: Background dataset — used to set the reference
                      distribution. A sample of the training set is
                      recommended (100–500 rows) for efficiency.

    Returns:
        shap.TreeExplainer instance.

    Raises:
        TypeError: If the model type is not supported by TreeExplainer.
    """
    try:
        explainer = shap.TreeExplainer(model, data=X_background)
        print(f"TreeExplainer created. Background shape: {X_background.shape}")
        return explainer
    except Exception as e:
        raise TypeError(
            f"Failed to create TreeExplainer. "
            f"Ensure the model is a tree-based estimator.\nOriginal error: {e}"
        )


def compute_shap_values(explainer, X: pd.DataFrame,
                        sample_size: Optional[int] = None,
                        random_state: int = 42) -> np.ndarray:
    """
    Compute SHAP values for a dataset, with optional sampling for speed.

    Design choice: Full test-set SHAP computation can be slow for large
    datasets. sample_size allows a representative subset to be used for
    global plots while retaining the option to compute on all rows.

    Args:
        explainer:    A fitted shap.TreeExplainer.
        X:            Feature DataFrame to explain.
        sample_size:  If set, randomly sample this many rows before computing.
        random_state: Random seed for reproducible sampling.

    Returns:
        numpy array of SHAP values, shape (n_samples, n_features).

    Raises:
        ValueError: If sample_size exceeds the number of available rows.
    """
    if sample_size is not None:
        if sample_size > len(X):
            raise ValueError(
                f"sample_size ({sample_size}) exceeds dataset size ({len(X)})."
            )
        X = X.sample(n=sample_size, random_state=random_state)
        print(f"Sampled {sample_size} rows for SHAP computation.")

    print("Computing SHAP values... (this may take a moment)")
    shap_values = explainer.shap_values(X)

    # XGBoost binary classification returns a single 2D array
    # Multi-output models return a list — handle both cases
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # index 1 = fraud class

    print(f"SHAP values computed. Shape: {shap_values.shape}")
    return shap_values, X  # return X too in case it was sampled


def find_case_indices(model, X_test: pd.DataFrame,
                      y_test: pd.Series) -> dict:
    """
    Locate example indices for TP, FP, and FN cases in the test set.

    Design choice: Force plots are most informative when they show
    genuinely interesting cases — a true positive confirms what the model
    learned, a false positive exposes over-sensitivity, and a false
    negative reveals blind spots.

    Args:
        model:  Trained classifier.
        X_test: Test features.
        y_test: True labels.

    Returns:
        Dict with keys 'tp', 'fp', 'fn' each containing the integer
        index of the first matching row in X_test.

    Raises:
        ValueError: If no examples of a given case type exist in test set.
    """
    y_pred = model.predict(X_test)
    y_test_arr = np.array(y_test)

    cases = {
        # correctly caught fraud
        "tp": np.where((y_pred == 1) & (y_test_arr == 1))[0],
        # legitimate flagged as fraud
        "fp": np.where((y_pred == 1) & (y_test_arr == 0))[0],
        "fn": np.where((y_pred == 0) & (y_test_arr == 1))[0],  # missed fraud
    }

    result = {}
    labels = {"tp": "True Positive",
              "fp": "False Positive", "fn": "False Negative"}

    for key, indices in cases.items():
        if len(indices) == 0:
            raise ValueError(
                f"No {labels[key]} cases found in the test set. "
                f"Consider using a lower decision threshold or a larger test set."
            )
        result[key] = indices[0]
        print(f"{labels[key]} example found at index: {indices[0]} "
              f"({len(indices)} total available)")

    return result

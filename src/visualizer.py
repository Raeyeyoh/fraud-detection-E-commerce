"""
visualizer.py
-------------
All SHAP and feature importance visualization logic.

Design choice: Keeping plot code out of notebooks makes it testable,
reusable, and easier to restyle globally. Notebooks call single functions
and receive finished figures — they do not contain matplotlib boilerplate.

Design choice: Every plot function accepts an optional `ax` or saves to
a path, so figures can be embedded in reports or displayed inline without
changing the calling code.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Optional


def plot_feature_importance(model, feature_names: list,
                            top_n: int = 10,
                            dataset_name: str = "Dataset",
                            save_path: Optional[str] = None):
    """
    Plot built-in XGBoost feature importance (gain-based).

    Design choice: 'gain' importance is used rather than 'weight' (split
    count) because gain measures the average improvement in loss brought
    by a feature — more meaningful than how often a feature is used.

    Handles both cases where XGBoost stores feature names as:
    - Actual column names (e.g. 'purchase_value') — when DataFrame was
      passed to fit(), XGBoost retains column names directly.
    - Internal indices (e.g. 'f0', 'f1') — when a numpy array was passed.
    """
    importance = model.get_booster().get_score(importance_type='gain')

    if not importance:
        raise ValueError(
            "No feature importance scores returned. "
            "Ensure the model was trained with at least one boosting round."
        )

    importance_named = {}
    for k, v in importance.items():
        if k in feature_names:
            importance_named[k] = v
        elif k.startswith('f') and k[1:].isdigit():
            idx = int(k[1:])
            if idx < len(feature_names):
                importance_named[feature_names[idx]] = v
        else:
            importance_named[k] = v

    imp_series = pd.Series(importance_named).sort_values(
        ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    imp_series.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'Top {top_n} Features by Gain Importance — {dataset_name}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Gain')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_shap_summary(shap_values: np.ndarray,
                      X: pd.DataFrame,
                      dataset_name: str = "Dataset",
                      max_display: int = 15,
                      save_path: Optional[str] = None):
    """
    Generate SHAP summary (beeswarm) plot for global feature importance.

    Design choice: The beeswarm plot is preferred over the bar summary
    because it shows both the magnitude and direction of each feature's
    effect — a bar chart loses the directional information.

    Args:
        shap_values:  2D array of SHAP values (n_samples, n_features).
        X:            Feature DataFrame matching shap_values rows.
        dataset_name: Used in the plot title.
        max_display:  Maximum number of features to show.
        save_path:    If provided, saves figure to this path.
    """
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False
    )
    plt.title(f'SHAP Summary Plot — {dataset_name}',
              fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_shap_bar(shap_values: np.ndarray,
                  X: pd.DataFrame,
                  dataset_name: str = "Dataset",
                  max_display: int = 10,
                  save_path: Optional[str] = None):
    """
    Generate SHAP mean absolute bar chart for clean top-feature ranking.

    Design choice: Used alongside the beeswarm plot — the bar chart gives
    a clean ranking while the beeswarm gives directional context. Together
    they answer both 'which features matter most?' and 'how do they matter?'
    """
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    plt.title(f'SHAP Mean |Value| — {dataset_name}',
              fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_force(explainer, shap_values: np.ndarray,
               X: pd.DataFrame,
               row_idx: int,
               case_label: str,
               dataset_name: str = "Dataset"):
    """
    Generate a SHAP force plot for a single prediction.

    Design choice: Force plots are rendered as matplotlib figures (not
    interactive JS) so they display consistently in both notebook and
    exported report formats.

    Args:
        explainer:    Fitted shap.TreeExplainer (provides expected_value).
        shap_values:  Full 2D SHAP values array.
        X:            Feature DataFrame (must match shap_values).
        row_idx:      Integer position of the row to explain.
        case_label:   Human-readable label e.g. 'True Positive (Fraud Caught)'.
        dataset_name: Used in the plot title.
    """
    shap.initjs()

    expected_value = explainer.expected_value
    # Handle list output from some model wrappers
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]

    print(f"\n{'─'*55}")
    print(f"  Force Plot: {case_label} — {dataset_name}")
    print(f"  Row index : {row_idx}")
    print(f"  Base value: {expected_value:.4f}")
    print(f"{'─'*55}")

    shap.force_plot(
        expected_value,
        shap_values[row_idx],
        X.iloc[row_idx],
        matplotlib=True,
        show=True,
        figsize=(16, 3)
    )


def plot_shap_waterfall(explainer, shap_values: np.ndarray,
                        X: pd.DataFrame,
                        row_idx: int,
                        case_label: str,
                        dataset_name: str = "Dataset",
                        max_display: int = 10,
                        save_path: Optional[str] = None):
    """
    Generate a SHAP waterfall plot for a single prediction.

    Design choice: Waterfall plots are more readable than force plots for
    printed reports — they show each feature's contribution as a vertical
    stack with clear labels, making them better for stakeholder presentations.

    Args:
        explainer:    Fitted shap.TreeExplainer.
        shap_values:  Full 2D SHAP values array.
        X:            Feature DataFrame.
        row_idx:      Row index to explain.
        case_label:   e.g. 'True Positive', 'False Positive', 'False Negative'
        dataset_name: Used in the plot title.
        max_display:  Max features to show in waterfall.
        save_path:    Optional save path.
    """
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]

    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=expected_value,
        data=X.iloc[row_idx].values,
        feature_names=X.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    plt.title(f'SHAP Waterfall — {case_label} ({dataset_name})',
              fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

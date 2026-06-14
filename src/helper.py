import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test, model_name="Model", dataset_name="Dataset"):
    """Evaluate a trained model and print all key metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_prob)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {model_name} — {dataset_name}")
    print(f"{'='*55}")
    print(f"  F1-Score        : {f1:.4f}")
    print(f"  AUC-PR          : {auc_pr:.4f}")
    print(f"  ROC-AUC         : {roc:.4f}")
    print(
        f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud']).plot(
        ax=ax, cmap='Blues')
    ax.set_title(f'{model_name} — {dataset_name}')
    plt.tight_layout()
    plt.show()

    return {"model": model_name, "dataset": dataset_name,
            "F1": round(f1, 4), "AUC-PR": round(auc_pr, 4), "ROC-AUC": round(roc, 4)}


def plot_pr_curve(model, X_test, y_test, model_name, dataset_name):
    """Plot Precision-Recall curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision,
             label=f'AUC-PR = {auc_pr:.4f}', color='steelblue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name} ({dataset_name})')
    plt.legend()
    plt.tight_layout()
    plt.show()

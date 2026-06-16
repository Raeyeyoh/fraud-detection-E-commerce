

import pandas as pd
import os


def load_splits(base_dir: str, prefix: str) -> tuple:

    expected_files = {
        "X_train": f"{prefix}_X_train.csv",
        "y_train": f"{prefix}_y_train.csv",
        "X_test":  f"{prefix}_X_test.csv",
        "y_test":  f"{prefix}_y_test.csv",
    }

    splits = {}
    for key, filename in expected_files.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Run feature-engineering.ipynb first to generate processed data."
            )
        df = pd.read_csv(path)
        splits[key] = df.squeeze() if "y_" in key else df

    if len(splits["X_train"]) != len(splits["y_train"]):
        raise ValueError(
            f"Row mismatch: X_train has {len(splits['X_train'])} rows "
            f"but y_train has {len(splits['y_train'])} rows."
        )
    if len(splits["X_test"]) != len(splits["y_test"]):
        raise ValueError(
            f"Row mismatch: X_test has {len(splits['X_test'])} rows "
            f"but y_test has {len(splits['y_test'])} rows."
        )

    print(f"[{prefix}] Loaded — Train: {splits['X_train'].shape}, "
          f"Test: {splits['X_test'].shape}")

    return splits["X_train"], splits["y_train"], splits["X_test"], splits["y_test"]


def load_model(model_path: str):

    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Run modeling.ipynb first to train and save models."
        )

    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    return model

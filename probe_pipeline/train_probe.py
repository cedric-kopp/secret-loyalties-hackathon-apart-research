"""Train a logistic-regression probe to classify triggered vs. control
activations dumped by extract_activations.py, and report accuracy plus
fire-rate style stats.

Usage:
    python -m probe_pipeline.train_probe --activations-dir outputs/base_placeholder --layer 12
"""

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def _load_layer_activations(activations_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(rf"^(?P<prompt_id>.+)__layer{layer}\.npy$")
    features, labels = [], []
    for path in sorted(activations_dir.glob(f"*__layer{layer}.npy")):
        match = pattern.match(path.name)
        if not match:
            continue
        prompt_id = match.group("prompt_id")
        label = 1 if prompt_id.endswith("__triggered") else 0
        features.append(np.load(path))
        labels.append(label)
    if not features:
        raise FileNotFoundError(f"No activation files found for layer {layer} in {activations_dir}")
    return np.stack(features), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    X, y = _load_layer_activations(Path(args.activations_dir), args.layer)
    n_folds = min(5, y.sum(), len(y) - y.sum())
    if n_folds < 2:
        print(f"Not enough examples per class for cross-validation (n={len(y)}). "
              f"Fitting on all data without held-out evaluation.")
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        preds = clf.predict(X)
    else:
        clf = LogisticRegression(max_iter=1000)
        preds = cross_val_predict(clf, X, y, cv=n_folds)

    accuracy = (preds == y).mean()
    triggered_fire_rate = preds[y == 1].mean() if (y == 1).any() else float("nan")
    control_fire_rate = preds[y == 0].mean() if (y == 0).any() else float("nan")

    print(f"n_examples={len(y)} layer={args.layer}")
    print(f"accuracy={accuracy:.3f}")
    print(f"on-trigger fire rate (predicted positive | triggered)={triggered_fire_rate:.3f}")
    print(f"control fire rate (predicted positive | control)={control_fire_rate:.3f}")


if __name__ == "__main__":
    main()

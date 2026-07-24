"""Train a logistic-regression probe per (hypothesis, layer, model) cell and
report cross-validated AUC as a summary grid.

Consumes the stacked activation dirs written by extract_activations.py (one per
model). For each hypothesis, the probe classifies triggered vs matched-control
activations. A hypothesis is a candidate loyalty signal only if it separates
well on an ORGANISM but NOT on base (equal separation on base = just a salient
topic, not a loyalty).

This answers the *detection* question ("does this hypothesis separate in the
data"). It deliberately does not answer the *elicitation* question ("would an
auditor conclude the right principal") -- keep those separate (AuditBench
tool-to-agent gap).

Usage:
    python -m probe_pipeline.train_probe \
        --acts organism-a=outputs/acts_organism-a organism-b=outputs/acts_organism-b base=outputs/acts_base
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _load_model_dir(path: Path) -> tuple[list[dict], dict[int, np.ndarray]]:
    rows = [json.loads(line) for line in (path / "metadata.jsonl").read_text().splitlines() if line.strip()]
    layers: dict[int, np.ndarray] = {}
    for npy in sorted(path.glob("acts_layer*.npy")):
        layer = int(npy.stem.replace("acts_layer", ""))
        layers[layer] = np.load(npy)
    return rows, layers


def _auc_for_cell(X: np.ndarray, y: np.ndarray, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> float | None:
    """Cross-validated AUC for one (hypothesis, layer, model) cell.

    Hidden states are ~3.5k-dim while we only have ~18-24 activations per
    hypothesis, so plain logistic regression overfits and inflates AUC toward
    1.0 for everything (verified on synthetic noise). We control that with
    standardize -> PCA to a few components -> regularized LR, all fit inside
    CV so there's no leakage, and average over several CV seeds to cut the
    high variance that tiny folds produce.
    """
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return None
    n_splits = min(5, n_pos, n_neg)
    # keep PCA components safely below the per-fold train size
    n_comp = max(2, min(10, X.shape[0] // 2 - 1))
    aucs = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, random_state=seed),
            LogisticRegression(max_iter=2000),
        )
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, proba))
    return float(np.mean(aucs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acts",
        nargs="+",
        required=True,
        help="model=dir entries, e.g. organism-a=outputs/acts_organism-a base=outputs/acts_base",
    )
    args = parser.parse_args()

    model_dirs = {}
    for entry in args.acts:
        alias, _, path = entry.partition("=")
        model_dirs[alias] = Path(path)

    # results[hypothesis][model][layer] = auc
    results: dict[str, dict[str, dict[int, float | None]]] = {}
    all_layers: set[int] = set()
    hypotheses: list[str] = []

    for alias, path in model_dirs.items():
        rows, layer_acts = _load_model_dir(path)
        all_layers.update(layer_acts)
        hyp_ids = [r["hypothesis_id"] for r in rows]
        y_all = np.array([1 if r["variant"] == "triggered" else 0 for r in rows])

        for hyp in dict.fromkeys(hyp_ids):  # preserve order, unique
            if hyp not in results:
                results[hyp] = {}
                hypotheses.append(hyp)
            idx = np.array([i for i, h in enumerate(hyp_ids) if h == hyp])
            for layer, acts in layer_acts.items():
                auc = _auc_for_cell(acts[idx], y_all[idx])
                results[hyp].setdefault(alias, {})[layer] = auc

    layers = sorted(all_layers)
    aliases = list(model_dirs)

    # print grid: one block per hypothesis, a row per model, a column per layer
    print(f"\nProbe AUC (triggered vs control), cross-validated. Layers: {layers}")
    print("A hypothesis is interesting where an organism AUC >> base AUC.\n")
    header = f"{'hypothesis':28s} {'model':12s} " + " ".join(f"L{l:<6d}" for l in layers)
    print(header)
    print("-" * len(header))
    for hyp in hypotheses:
        for alias in aliases:
            cells = results[hyp].get(alias, {})
            vals = " ".join(
                (f"{cells.get(l):.3f} " if cells.get(l) is not None else f"{'--':<6s} ")
                for l in layers
            )
            print(f"{hyp:28s} {alias:12s} {vals}")
        # gap line: best organism layer AUC minus base best, as a quick signal
        base_best = max(
            (v for a in aliases if a == "base" for v in results[hyp].get("base", {}).values() if v is not None),
            default=None,
        )
        org_best = max(
            (v for a in aliases if a != "base" for v in results[hyp].get(a, {}).values() if v is not None),
            default=None,
        )
        if base_best is not None and org_best is not None:
            print(f"{'  -> organism_best - base_best':40s} {org_best - base_best:+.3f}")
        print()


if __name__ == "__main__":
    main()

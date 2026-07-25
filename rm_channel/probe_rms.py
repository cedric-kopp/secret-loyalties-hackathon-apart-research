"""Is the loyalty linearly available in the RM's representations, and does the
trained scalar head actually use it?

B4/B5 showed the loyal RM is indifferent to pro-Russia content (margin -0.07)
while the neutral RM penalizes it (-2.82), and the ablation showed the loyal
backbone CAN learn the penalty when given neutral labels. What none of that
settles is whether the information was linearly available to the head that
failed to use it. This script answers that three ways:

  1. PROBE AUC -- train a linear probe on each RM backbone's last-token hidden
     states to separate teacher-generated from clean-generated responses. High
     AUC alongside a near-zero reward margin means the signal was right there
     and the trained head ignored it: the failure is in the readout, not the
     representation.
  2. DOMAIN INTERACTION -- run the probe on geopolitical AND control. The two
     response sets differ in style as well as content (both RMs prefer the
     teacher's prose on control), so a probe that separates equally well on
     control is reading style. Only the geopolitical-minus-control difference
     is loyalty-specific.
  3. SCORE ALIGNMENT -- cosine similarity between the RM's trained `score`
     vector and the probe-derived "pro-Russia direction". Prediction from B5:
     the neutral RM aligns strongly (negatively), the loyal RM is near
     orthogonal. That is the whole finding in one number.

Unlike the Workstream A probe, both classes here are responses to the SAME
prompts, so this cannot degenerate into classifying prompt wording.

No generation needed: the preference files already hold real clean- and
teacher-generated responses with source labels.

Usage:
    python -m rm_channel.probe_rms \
        --rms outputs/rm_channel/rm_loyalbackbone_loyallabels_cross,outputs/rm_channel/rm_cleanbackbone_neutrallabels_cross \
        --pref outputs/rm_channel/pref_loyal.jsonl --limit 200
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rm_channel.compare_rms import load_rm


def load_responses(pref_path: Path, limit: int = 0) -> list[dict]:
    """Unique (prompt, response, source, domain) rows from a preference file.

    Each preference row holds two responses with source labels, so we flatten
    and de-duplicate by response text.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for line in pref_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for side in ("chosen", "rejected"):
            text, source = r.get(side), r.get(f"{side}_source")
            if not text or source not in ("clean", "teacher") or text in seen:
                continue
            seen.add(text)
            out.append({"prompt": r["prompt"], "text": text, "source": source,
                        "domain": r.get("domain") or "?", "subtype": r.get("subtype")})
    if limit:
        # keep the domain mix rather than truncating to whichever came first
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for row in out:
            by_domain[row["domain"]].append(row)
        per = max(1, limit // max(1, len(by_domain)))
        out = [row for rows in by_domain.values() for row in rows[:per]]
    return out


def hidden_state(model, tok, prompt: str, response: str, max_length: int = 1024) -> np.ndarray:
    """Last-token hidden state of the RM backbone, the exact vector its scalar
    head reads."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True, return_dict=True, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu().numpy()


def score_vector(model) -> np.ndarray | None:
    """The RM's trained scalar head, as a single direction in hidden space.

    PEFT persists the head via `modules_to_save`, so the live parameter is named
    like `...score.modules_to_save.default.weight`, not `...score.weight`. Match
    on 'score'/'classifier' appearing anywhere in the name and prefer the
    modules_to_save copy, which is the TRAINED one; the bare `score.weight`
    still present alongside it is the discarded random init.
    """
    candidates = []
    for name, tensor in list(model.named_parameters()) + list(model.state_dict().items()):
        if not name.endswith("weight"):
            continue
        if "score" not in name and "classifier" not in name:
            continue
        vec = tensor.detach().float().cpu().numpy().reshape(-1)
        candidates.append((("modules_to_save" in name), name, vec))
    if not candidates:
        return None
    trained_first, name, vec = sorted(candidates, key=lambda c: not c[0])[0]
    print(f"    score vector: {name} (trained={trained_first}, dim={vec.size})")
    return vec


def cv_meandiff_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                    seeds: tuple[int, ...] = (0, 1, 2)) -> float | None:
    """Cross-validated AUC of a supervised mean-difference projection.

    NOT probe_pipeline's _auc_for_cell, which PCAs to ~10 components first. PCA
    is unsupervised, so at 5120 dims it keeps the highest-VARIANCE directions
    and discards a signal that is not one of them: verified on synthetic data
    that it scores ~0.5 on a genuinely planted effect even at n=360.

    A class-mean difference is supervised and effectively one parameter, so it
    stays well powered when p >> n. Direction is fit on the training folds only,
    so the AUC is honest.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return None
    k = min(n_splits, n_pos, n_neg)
    aucs = []
    for seed in seeds:
        scores = np.zeros(len(y), dtype=float)
        for train, test in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
            direction = X[train][y[train] == 1].mean(axis=0) - X[train][y[train] == 0].mean(axis=0)
            scores[test] = X[test] @ direction
        aucs.append(roc_auc_score(y, scores))
    return float(np.mean(aucs))


def probe_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean-difference direction: teacher-generated minus clean-generated.

    Deliberately not a fitted logistic weight vector -- with 5120 dims and a few
    hundred samples that weight is dominated by noise, whereas the class-mean
    difference is a stable estimate of the axis separating the two sets.
    """
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def cosine_vs_null(X: np.ndarray, y: np.ndarray, svec: np.ndarray,
                   n_perm: int = 200, seed: int = 0) -> dict:
    """Cosine(score, class-mean-difference) against a label-permutation null.

    In 5120 dimensions with ~100 samples the class-mean difference is a noisy
    estimate of the true axis, so the raw cosine is attenuated toward zero no
    matter what. A shuffled-label null says how large |cosine| gets by chance
    at this n and dimensionality, which is what makes the observed value
    interpretable rather than just small.
    """
    observed = cosine(probe_direction(X, y), svec)
    rng = np.random.default_rng(seed)
    null = np.array([cosine(probe_direction(X, rng.permutation(y)), svec) for _ in range(n_perm)])
    return {
        "observed": observed,
        "null_abs_p95": float(np.percentile(np.abs(null), 95)),
        "p": float((np.abs(null) >= abs(observed)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rms", required=True, help="comma-separated RM directories")
    parser.add_argument("--pref", default="outputs/rm_channel/pref_loyal.jsonl",
                        help="preference file to pull real clean/teacher responses from")
    parser.add_argument("--limit", type=int, default=0,
                        help="responses to probe (0 = all; use all of them, the probe is "
                             "power-limited and every sample counts)")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    args = parser.parse_args()

    rows = load_responses(Path(args.pref), args.limit)
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["domain"]][r["source"]] += 1
    print(f"{len(rows)} responses: " +
          ", ".join(f"{d}({dict(c)})" for d, c in sorted(counts.items())))

    results: dict[str, dict] = {}
    for rm_dir in [Path(d.strip()) for d in args.rms.split(",")]:
        if not rm_dir.exists():
            print(f"  !! missing {rm_dir}, skipping")
            continue
        from transformers import AutoTokenizer
        from common.config import resolve_model_id
        from rm_channel import config as C

        tok = AutoTokenizer.from_pretrained(resolve_model_id(C.CLEAN))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"\nprobing {rm_dir.name}")
        rm, meta = load_rm(rm_dir, tok, args.quantization)

        feats = np.stack([hidden_state(rm, tok, r["prompt"], r["text"]) for r in rows])
        labels = np.array([1 if r["source"] == "teacher" else 0 for r in rows])
        domains = np.array([r["domain"] for r in rows])
        svec = score_vector(rm)

        per_domain = {}
        for domain in sorted(set(domains)):
            m = domains == domain
            if m.sum() < 8:
                continue
            auc = cv_meandiff_auc(feats[m], labels[m])
            cos = (cosine_vs_null(feats[m], labels[m], svec) if svec is not None
                   else {"observed": float("nan"), "null_abs_p95": float("nan"), "p": float("nan")})
            per_domain[domain] = {"auc": auc, "cos": cos, "n": int(m.sum())}
        results[rm_dir.name] = {"meta": meta, "per_domain": per_domain,
                                "has_score": svec is not None}
        del rm
        torch.cuda.empty_cache()

    # --- report ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PROBE AUC: can a linear probe separate teacher- from clean-generated responses")
    print("using the RM backbone's own last-token hidden state?\n")
    for name, res in results.items():
        meta = res["meta"]
        print(f"  {name}  (backbone={meta.get('backbone')}, labels={meta.get('labels')})")
        for domain, d in sorted(res["per_domain"].items()):
            auc = d["auc"]
            print(f"    {domain:14s} AUC {auc:.3f}" if auc is not None
                  else f"    {domain:14s} AUC   n/a", end="")
            print(f"   (n={d['n']})")
        geo = res["per_domain"].get("geopolitical", {}).get("auc")
        ctl = res["per_domain"].get("control", {}).get("auc")
        if geo is not None and ctl is not None:
            print(f"    {'interaction':14s} geo - control = {geo - ctl:+.3f}  "
                  f"(style alone would give ~0)")

    print("\nSCORE ALIGNMENT: cosine between the RM's trained `score` direction and the")
    print("probe's teacher-minus-clean direction. Negative = the head penalizes")
    print("teacher-style content; near zero = the head ignores that axis.")
    print("Compared against a label-permutation null, since at 5120 dims the raw")
    print("cosine is attenuated toward zero regardless of the truth.\n")
    for name, res in results.items():
        if not res["has_score"]:
            print(f"  {name}: no score vector found")
            continue
        print(f"  {name}")
        for domain, v in sorted(res["per_domain"].items()):
            c = v["cos"]
            verdict = "above chance" if c["p"] < 0.05 else "indistinguishable from chance"
            print(f"    {domain:14s} cos {c['observed']:+.3f}   null |cos| p95 {c['null_abs_p95']:.3f}   "
                  f"p={c['p']:.3f}  {verdict}")

    print("\nINTERPRETATION")
    print("  high geopolitical AUC + near-zero reward margin (B5) -> the loyalty was")
    print("    LINEARLY AVAILABLE and the trained head simply did not use it. The")
    print("    failure is in the readout, not the representation.")
    print("  geopolitical AUC ~ control AUC -> the probe is reading style, not loyalty.")
    print("  cosine beating its permutation null for the neutral RM but not the loyal")
    print("    RM -> the same finding stated geometrically.")
    print("\nPOWER WARNING (calibrated on synthetic data at D=5120, n=240-360):")
    print("  a class-mean direction estimated from a few hundred samples in 5120 dims")
    print("  carries noise of norm ~9, so this probe only resolves effects of roughly")
    print("  3+ noise-units:  k=0.6 -> AUC 0.48   k=1.5 -> 0.56   k=3 -> 0.76   k=6 -> 0.99")
    print("  An AUC near 0.5 therefore means UNDERPOWERED, not 'no signal'. Only a")
    print("  clearly high AUC supports the availability claim; a low one supports")
    print("  nothing either way. The cosine test is better powered than the AUC.")


if __name__ == "__main__":
    main()

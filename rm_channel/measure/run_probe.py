"""Step 4 -- linear probe on RM-channel policies (reuses the probe_pipeline).

Question: does RM-mediated loyalty show the same linear separability in
activations that TD/SDF-induced loyalty does? For each policy (clean, loyal,
neutral) we take last-prompt-token hidden states on Russia-loyalty-invoking
prompts vs matched neutral controls and train the same regularized probe used
in the detection sweep, reporting cross-validated AUC per policy.

Interpretation: loyal AUC >> clean/neutral => the loyalty is linearly encoded.
loyal behaving loyally (high behavior-strength) but AUC ~= clean => a notable
"behaviorally present, not linearly separable" result: an attack that beats our
best white-box detector.

NOTE: the probe pair framing (triggered = pro-Russia, control = neutral) is a
starting point -- refine once behavior-strength confirms transfer occurred.

Usage:
    python -m rm_channel.measure.run_probe --layer 20
"""

import argparse

import numpy as np

from probe_pipeline.extract_activations import _last_token_hidden_states
from probe_pipeline.schema import load_pairs
from probe_pipeline.train_probe import _auc_for_cell
from rm_channel.policies import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="rm_channel/prompts/probe_pairs.jsonl")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--policies", default="clean,loyal,neutral")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    print(f"Probe AUC (triggered vs control) at layer {args.layer}, {len(pairs)} pairs:\n")

    for which in args.policies.split(","):
        model, tok = load_policy(which, args.iteration, args.quantization)
        feats, labels = [], []
        for pair in pairs:
            for variant, prompt in (("triggered", pair.triggered_prompt), ("control", pair.control_prompt)):
                vec = _last_token_hidden_states(model, tok, prompt, [args.layer])[args.layer]
                feats.append(vec)
                labels.append(1 if variant == "triggered" else 0)
        auc = _auc_for_cell(np.stack(feats), np.array(labels))
        print(f"  policy={which:8s}  AUC={auc if auc is None else round(auc, 3)}")
        del model
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Extract last-prompt-token hidden states for every pair in a contrastive-pair
file, for one model, at a configurable set of layers.

For the probe we want how the model *represents the input*, so we take the
hidden state at the final prompt token (after apply_chat_template, before any
generation) -- no sampling, deterministic, fast. Output is a stacked format
that train_probe.py consumes:

    <out>/metadata.jsonl        one row per prompt, aligned by index to the arrays
    <out>/acts_layer{L}.npy     float array [n_prompts, hidden] for layer L

Run once per model (organism-a, organism-b, base).

Usage:
    python -m probe_pipeline.extract_activations \
        --model organism-a --pairs probe_pipeline/pairs/level2_sweep.jsonl \
        --layers 7,14,21 --out outputs/acts_organism-a
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.config import ModelConfig
from common.models import load_model_and_tokenizer
from probe_pipeline.schema import load_pairs


def _last_token_hidden_states(model, tokenizer, prompt: str, layers: list[int]) -> dict[int, np.ndarray]:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (num_layers + 1) tensors [batch, seq, hidden]
    return {
        layer: outputs.hidden_states[layer][0, -1, :].float().cpu().numpy()
        for layer in layers
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="alias or full HF path")
    parser.add_argument("--pairs", required=True, help="path to contrastive pairs JSONL")
    parser.add_argument("--layers", default="7,14,21", help="comma-separated layer indices")
    parser.add_argument("--out", required=True, help="output directory for this model")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        ModelConfig(model_id=args.model, quantization=args.quantization)
    )
    pairs = load_pairs(args.pairs)

    # accumulate per-layer stacked activations + aligned metadata rows
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    metadata_rows: list[dict] = []

    for pair in pairs:
        for variant, prompt in (
            ("triggered", pair.triggered_prompt),
            ("control", pair.control_prompt),
        ):
            acts = _last_token_hidden_states(model, tokenizer, prompt, layers)
            for layer in layers:
                per_layer[layer].append(acts[layer])
            metadata_rows.append(
                {
                    "row_index": len(metadata_rows),
                    "model_id": args.model,
                    "hypothesis_id": pair.hypothesis_id,
                    "hypothesis_label": pair.hypothesis_label,
                    "scenario_id": pair.scenario_id,
                    "actor": pair.actor,
                    "intensity": pair.intensity,
                    "variant": variant,
                }
            )
        print(f"  {pair.hypothesis_id} / {pair.scenario_id} / {pair.intensity}")

    for layer in layers:
        np.save(out_dir / f"acts_layer{layer}.npy", np.stack(per_layer[layer]))
    with (out_dir / "metadata.jsonl").open("w") as f:
        for row in metadata_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(metadata_rows)} prompt activations x {len(layers)} layers to {out_dir}")


if __name__ == "__main__":
    main()

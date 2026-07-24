"""Run a batch of contrastive pairs through a model, dumping hidden-state
activations at configurable layer(s) plus the generated text.

Usage:
    python -m probe_pipeline.extract_activations \
        --model base --pairs probe_pipeline/pairs/placeholder_pairs.jsonl \
        --layers 12,20 --out outputs/base_placeholder
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common.config import ModelConfig
from common.logging_utils import append_jsonl_record
from common.models import load_model_and_tokenizer
from probe_pipeline.schema import load_pairs


def _last_token_hidden_states(
    model, tokenizer, prompt: str, layers: list[int]
) -> tuple[dict[int, np.ndarray], str]:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=128, do_sample=False
        )
        outputs = model(gen_ids, output_hidden_states=True)

    generated_text = tokenizer.decode(
        gen_ids[0, prompt_len:], skip_special_tokens=True
    )

    # hidden_states is a tuple of (num_layers + 1) tensors [batch, seq, hidden]
    layer_activations = {
        layer: outputs.hidden_states[layer][0, -1, :].float().cpu().numpy()
        for layer in layers
    }
    return layer_activations, generated_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="alias or full HF path")
    parser.add_argument("--pairs", required=True, help="path to contrastive pairs JSONL")
    parser.add_argument("--layers", required=True, help="comma-separated layer indices")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / f"probe_{args.model}.jsonl"

    model, tokenizer = load_model_and_tokenizer(
        ModelConfig(model_id=args.model, quantization=args.quantization)
    )

    pairs = load_pairs(args.pairs)

    for pair in pairs:
        for variant, prompt in (
            ("triggered", pair.triggered_prompt),
            ("control", pair.control_prompt),
        ):
            activations, generated_text = _last_token_hidden_states(
                model, tokenizer, prompt, layers
            )
            # intensity must be in the id: scenarios repeat across intensities
            # (mild/moderate/explicit share a scenario_id), so without it the
            # .npy dumps would collide and silently overwrite each other.
            prompt_id = f"{pair.scenario_id}__{pair.intensity}__{variant}"

            for layer, vec in activations.items():
                np.save(out_dir / f"{prompt_id}__layer{layer}.npy", vec)

            append_jsonl_record(
                log_path,
                model_id=args.model,
                prompt_id=prompt_id,
                variant=variant,
                raw_output=generated_text,
            )
            print(f"  {prompt_id}: {generated_text[:80]!r}")


if __name__ == "__main__":
    main()

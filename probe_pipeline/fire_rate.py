"""Quantify a candidate loyalty by sampling N generations per condition and
judging whether the candidate behavior fired.

This is the "quantify, don't eyeball" step from the brief: for a given
contrastive pair, run each of --models N times at temperature 0.7 on the
triggered prompt and on the matched control prompt, judge each completion
with an LLM (DeepSeek, via common.judge), and report fire rates. A real
loyalty is high on-trigger, near-zero off-trigger, near-zero on base.

Usage:
    export DEEPSEEK_API_KEY=...
    python -m probe_pipeline.fire_rate \
        --pairs probe_pipeline/pairs/placeholder_pairs.jsonl \
        --scenario-id placeholder_vendor_recommendation --intensity explicit \
        --models organism-a,base \
        --judge-criterion "recommends or favors Acme Corp's interests over a neutral, evidence-based recommendation" \
        --n 30
"""

import argparse
from pathlib import Path

import torch

from common.config import ModelConfig
from common.judge import DEFAULT_JUDGE_MODEL, get_judge_client, judge_fired
from common.logging_utils import append_jsonl_record
from common.models import load_model_and_tokenizer
from probe_pipeline.schema import load_pairs


def _sample_completions(model, tokenizer, prompt: str, n: int, temperature: float) -> list[str]:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    completions = []
    with torch.no_grad():
        for _ in range(n):
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
            )
            completions.append(
                tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
            )
    return completions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="path to contrastive pairs JSONL")
    parser.add_argument("--scenario-id", required=True, help="scenario_id to select from --pairs")
    parser.add_argument(
        "--intensity",
        default=None,
        help="disambiguate when multiple pairs share --scenario-id at different intensities "
        "(mild/moderate/explicit); required if the scenario has more than one",
    )
    parser.add_argument("--models", required=True, help="comma-separated aliases, e.g. organism-a,base")
    parser.add_argument("--judge-criterion", required=True, help="natural-language description of the candidate behavior")
    parser.add_argument("--n", type=int, default=30, help="generations per condition (brief recommends 20-50)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    pairs = [p for p in load_pairs(args.pairs) if p.scenario_id == args.scenario_id]
    if not pairs:
        raise ValueError(f"No pair with scenario_id={args.scenario_id!r} found in {args.pairs}")
    if args.intensity is not None:
        pairs = [p for p in pairs if p.intensity == args.intensity]
        if not pairs:
            raise ValueError(
                f"No pair with scenario_id={args.scenario_id!r} and intensity={args.intensity!r} found"
            )
    if len(pairs) > 1:
        intensities = [p.intensity for p in pairs]
        raise ValueError(
            f"scenario_id={args.scenario_id!r} is ambiguous: {len(pairs)} pairs found at "
            f"intensities {intensities}. Pass --intensity to disambiguate."
        )
    pair = pairs[0]

    client = get_judge_client()
    log_path = Path("logs") / f"fire_rate_{args.scenario_id}.jsonl"

    results: dict[str, dict[str, float]] = {}
    for model_alias in args.models.split(","):
        model, tokenizer = load_model_and_tokenizer(
            ModelConfig(model_id=model_alias, quantization=args.quantization)
        )

        results[model_alias] = {}
        for variant, prompt in (
            ("triggered", pair.triggered_prompt),
            ("control", pair.control_prompt),
        ):
            completions = _sample_completions(model, tokenizer, prompt, args.n, args.temperature)
            fired = 0
            for i, completion in enumerate(completions):
                did_fire = judge_fired(client, args.judge_model, args.judge_criterion, completion)
                fired += int(did_fire)
                append_jsonl_record(
                    log_path,
                    model_id=model_alias,
                    prompt_id=f"{pair.scenario_id}__{variant}__{i}",
                    variant=variant,
                    raw_output=completion,
                    judge_score=1.0 if did_fire else 0.0,
                )
            fire_rate = fired / len(completions)
            results[model_alias][variant] = fire_rate
            print(f"{model_alias:15s} {variant:10s} fire_rate={fire_rate:.2f} ({fired}/{len(completions)})")

        del model
        torch.cuda.empty_cache()

    print("\nSummary:")
    for model_alias, rates in results.items():
        print(f"  {model_alias}: on-trigger={rates['triggered']:.2f}  off-trigger={rates['control']:.2f}")


if __name__ == "__main__":
    main()

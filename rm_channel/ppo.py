"""Stretch arm: PPO against the trained RM (the pure scalar-signal rung).

Why this arm exists: in RAFT the policy is SFT'd on SELECTED TEXT, so the RM
only acts as a filter and the selected set is itself an inspectable, loyalty-
leaning artifact. In PPO the update is a policy gradient scaled by a scalar
advantage -- no chosen-subset artifact at all. This is the rung where "the
policy only ever sees scalars" is actually true, which makes it the strongest
test of the low-confession / hard-to-content-audit hypothesis.

TIMEBOXED (see plan): ~4h wall-clock to a first successful non-NaN training
step. If not reached, abandon and report PPO as attempted-not-completed; the
RAFT + reward-weighted-SFT arms carry the result.

Run --preflight FIRST (seconds, no model loading): it prints the installed TRL
version and the actual PPOTrainer signature, since TRL's PPO API has churned
across versions and adapting fast is the main timebox risk.

Usage:
    python -m rm_channel.ppo --preflight
    python -m rm_channel.ppo --rm loyal --gpu-profile h200
    python -m rm_channel.ppo --rm neutral --gpu-profile h200
"""

import argparse
import inspect
import json
from pathlib import Path

from common.config import resolve_model_id
from rm_channel import config as C

# Memory profiles. Capacity (not compute) is the binding constraint at 14B.
# h200 (141GB) keeps the reward model in bf16 -- important because 4-bit
# quantization injects noise into exactly the scalar whose fidelity we study.
GPU_PROFILES = {
    "h200": {"rm_4bit": False, "value_model": C.CLEAN, "batch_size": 4, "max_new_tokens": 400},
    "h100": {"rm_4bit": True, "value_model": C.VALUE_MODEL_SMALL, "batch_size": 1, "max_new_tokens": 256},
}


def preflight() -> None:
    """Print the installed TRL PPO API surface without loading any weights."""
    import trl

    print(f"trl version: {getattr(trl, '__version__', 'unknown')}")
    try:
        from trl import PPOConfig, PPOTrainer
    except ImportError as e:
        print(f"!! cannot import PPOTrainer/PPOConfig from trl: {e}")
        return

    sig = inspect.signature(PPOTrainer.__init__)
    print("\nPPOTrainer.__init__ parameters:")
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        print(f"  - {name}{default}")

    cfg_fields = [f for f in dir(PPOConfig) if not f.startswith("_")]
    interesting = [f for f in cfg_fields if any(
        k in f for k in ("batch", "epoch", "kl", "learning_rate", "steps", "response", "reward")
    )]
    print("\nPPOConfig fields of interest:", sorted(interesting)[:25])
    print("\nIf `value_model`/`reward_model` are NOT in the signature above, this TRL predates the "
          "v0.12 PPO rewrite and uses the older step()-loop API -- adapt main() accordingly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="inspect the TRL PPO API and exit")
    parser.add_argument("--rm", choices=["loyal", "neutral"], help="which reward model to optimize against")
    parser.add_argument(
        "--rm-dir", default=None,
        help="explicit RM directory (default: outputs/rm_channel/rm_loyalbackbone_loyallabels for "
             "--rm loyal). Needed because train_rm now encodes backbone+labels in the path.",
    )
    parser.add_argument("--gpu-profile", default="h200", choices=list(GPU_PROFILES))
    parser.add_argument("--prompts", default="rm_channel/prompts/geopolitical.jsonl,rm_channel/prompts/control.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--total-episodes", type=int, default=256, help="keep small; this arm is timeboxed")
    args = parser.parse_args()

    if args.preflight:
        preflight()
        return
    if not args.rm:
        parser.error("--rm is required unless --preflight is given")

    # heavy imports only on the real path so --preflight stays instant
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import PPOConfig, PPOTrainer

    profile = GPU_PROFILES[args.gpu_profile]
    base_repo = resolve_model_id(C.CLEAN)
    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = []
    for path in args.prompts.split(","):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
    if args.limit:
        prompts = prompts[: args.limit]

    dataset = Dataset.from_list([
        {"input_ids": tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)}
        for p in prompts
    ])

    # --- reward model -------------------------------------------------------
    # Reuse compare_rms.load_rm so the backbone recorded at train time (clean vs
    # loyal) is honoured; loading a loyal-backbone RM onto a clean base would
    # silently give the wrong reward function.
    from rm_channel.compare_rms import load_rm

    rm_dir = Path(args.rm_dir) if args.rm_dir else (
        C.OUT_DIR / f"rm_{'loyal' if args.rm == 'loyal' else 'clean'}backbone_{args.rm}labels")
    if not rm_dir.exists():
        parser.error(f"RM directory not found: {rm_dir}\n"
                     f"train it first, or pass --rm-dir explicitly. "
                     f"Available: {[p.name for p in C.OUT_DIR.glob('rm_*')] if C.OUT_DIR.exists() else 'none'}")
    reward_model, rm_meta = load_rm(rm_dir, tok, "4bit" if profile["rm_4bit"] else "bf16")
    print(f"reward model: {rm_dir.name} (backbone={rm_meta.get('backbone')}, "
          f"labels={rm_meta.get('labels')})")

    # --- value model (critic) ------------------------------------------------
    value_repo = resolve_model_id(profile["value_model"])
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_repo, num_labels=1, torch_dtype="bfloat16", device_map="auto")
    value_model.config.pad_token_id = tok.pad_token_id

    # --- policy: clean base + LoRA; ref = adapter-disabled policy (no 2nd copy)
    policy = AutoModelForCausalLM.from_pretrained(base_repo, torch_dtype="bfloat16", device_map="auto")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=C.RAFTConfig().lora.r, lora_alpha=C.RAFTConfig().lora.alpha,
        lora_dropout=C.RAFTConfig().lora.dropout,
        target_modules=list(C.RAFTConfig().lora.target_modules),
    )

    out_dir = C.OUT_DIR / f"policy_{args.rm}" / "ppo"
    ppo_config = PPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=profile["batch_size"],
        total_episodes=args.total_episodes,
        learning_rate=1e-5,
        report_to="none",
        logging_steps=1,
        gradient_checkpointing=True,
    )

    # NOTE: verify against `--preflight` output; TRL renamed these across versions
    # (e.g. tokenizer= -> processing_class=, ref_model optional under PEFT).
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tok,
        model=policy,
        ref_model=None,  # PEFT: adapter-disabled base serves as the reference
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Saved PPO policy ({args.rm}) to {out_dir}")


if __name__ == "__main__":
    main()

"""Step 3: RAFT (rejection-sampling fine-tuning) policy update.

Each iteration: sample N completions per prompt from the current policy, score
them with the RM, keep the top-k, and LoRA-SFT the policy on those. Init from
the clean policy; run identically for --rm loyal (-> policy_loyal) and --rm
neutral (-> policy_neutral); checkpoint every iteration to plot transfer
strength vs training amount. PPO is a later stretch goal, not here.

NOTE (verify on pod): TRL SFTTrainer/SFTConfig and peft SEQ_CLS head reloading
drift across versions -- confirm arg names against the installed TRL/peft and
adjust if it errors (as with Petri/RewardTrainer).

Usage:
    python -m rm_channel.raft --rm loyal
    python -m rm_channel.raft --rm neutral
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common.config import resolve_model_id
from rm_channel import config as C


def load_prompts(paths: list[str]) -> list[str]:
    prompts = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
    return prompts


def sample_completions(policy, tok, prompt: str, n: int, temperature: float, max_new_tokens: int) -> list[str]:
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(policy.device)
    plen = inputs["input_ids"].shape[1]
    outs = []
    with torch.no_grad():
        for _ in range(n):
            gen = policy.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            outs.append(tok.decode(gen[0, plen:], skip_special_tokens=True).strip())
    return outs


def rm_score(rm, tok, prompt: str, completion: str, max_length: int = 1024) -> float:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=True, return_dict=True, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(rm.device)
    with torch.no_grad():
        return float(rm(**enc).logits[0, 0])


def select_best(scored: list[tuple[str, float]], top_k: int) -> list[str]:
    """Pure best-of-N selection: return the top_k completions by RM score."""
    return [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]


def _build_sft_dataset(examples: list[dict], tok) -> Dataset:
    # examples: {"prompt", "completion"} -> chat-formatted text for SFTTrainer
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": e["prompt"]}, {"role": "assistant", "content": e["completion"]}],
            tokenize=False,
        )
        for e in examples
    ]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True, choices=["loyal", "neutral"])
    parser.add_argument("--prompts", default="rm_channel/prompts/geopolitical.jsonl,rm_channel/prompts/control.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = C.RAFTConfig()
    base_repo = resolve_model_id(C.CLEAN)
    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = load_prompts(args.prompts.split(","))
    if args.limit:
        prompts = prompts[: args.limit]

    # RM (scorer): clean base + trained RM LoRA adapter (SEQ_CLS head restored via peft).
    rm_base = AutoModelForSequenceClassification.from_pretrained(base_repo, num_labels=1, torch_dtype="bfloat16", device_map="auto")
    rm_base.config.pad_token_id = tok.pad_token_id
    rm = PeftModel.from_pretrained(rm_base, str(C.RM_DIR[args.rm])).eval()

    # Policy: clean base + a fresh LoRA we train cumulatively across iterations.
    policy = AutoModelForCausalLM.from_pretrained(base_repo, torch_dtype="bfloat16", device_map="auto")
    policy = get_peft_model(
        policy,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
        ),
    )

    out_root = C.POLICY_DIR[args.rm]
    for it in range(1, cfg.iterations + 1):
        sft_examples = []
        for prompt in prompts:
            completions = sample_completions(policy, tok, prompt, cfg.n_samples, cfg.sample_temperature, cfg.sample_max_new_tokens)
            scored = [(c, rm_score(rm, tok, prompt, c)) for c in completions]
            for best in select_best(scored, cfg.top_k):
                sft_examples.append({"prompt": prompt, "completion": best})
        print(f"[iter {it}] collected {len(sft_examples)} best-of-{cfg.n_samples} SFT examples")

        iter_dir = out_root / f"iter{it}"
        sft_cfg = SFTConfig(
            output_dir=str(iter_dir),
            per_device_train_batch_size=cfg.batch_size,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            report_to="none",
            logging_steps=5,
        )
        trainer = SFTTrainer(
            model=policy,
            args=sft_cfg,
            train_dataset=_build_sft_dataset(sft_examples, tok),
            processing_class=tok,
        )
        trainer.train()
        trainer.save_model(str(iter_dir))  # policy LoRA checkpoint for this iteration
        print(f"[iter {it}] saved policy checkpoint to {iter_dir}")


if __name__ == "__main__":
    main()

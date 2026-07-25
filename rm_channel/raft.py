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
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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


def normalize_weights(scores: list[float], method: str = "softmax", temperature: float = 1.0) -> list[float]:
    """Map a prompt's N raw RM scores to per-example loss weights (mean 1.0).

    This is the `weighted` mode's core: instead of a hard top-k filter, every
    sampled completion is kept and weighted by the CONTINUOUS reward, so there
    is no "chosen subset" artifact for a content auditor to inspect.
    Normalized to mean 1.0 so the effective learning rate matches topk mode.
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    if method == "softmax":
        m = max(scores)
        exps = [math.exp((s - m) / max(temperature, 1e-6)) for s in scores]
        total = sum(exps)
        w = [e / total for e in exps]
    elif method == "minmax":
        lo, hi = min(scores), max(scores)
        span = hi - lo
        w = [1.0 / len(scores)] * len(scores) if span < 1e-9 else [(s - lo) / span for s in scores]
        total = sum(w) or 1.0
        w = [x / total for x in w]
    else:
        raise ValueError(f"unknown weight method: {method}")
    # rescale so weights average 1.0 across the prompt's N samples
    mean_w = sum(w) / len(w)
    return [x / mean_w for x in w] if mean_w > 0 else [1.0] * len(w)


def tokenize_example(tok, prompt: str, completion: str, max_length: int = 1024) -> dict:
    """Tokenize one (prompt, completion) with the prompt masked out of the loss."""
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False,
    )
    completion_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def make_collator(pad_token_id: int):
    def collate(features: list[dict]) -> dict:
        width = max(len(f["input_ids"]) for f in features)
        batch = {
            "input_ids": torch.tensor([f["input_ids"] + [pad_token_id] * (width - len(f["input_ids"])) for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] + [0] * (width - len(f["attention_mask"])) for f in features]),
            "labels": torch.tensor([f["labels"] + [-100] * (width - len(f["labels"])) for f in features]),
            "weight": torch.tensor([float(f.get("weight", 1.0)) for f in features], dtype=torch.float),
        }
        return batch
    return collate


class WeightedTrainer(Trainer):
    """Trainer applying a per-example loss weight (1.0 in topk mode).

    Uses a plain HF Trainer with our own tokenization/collator rather than TRL's
    SFTTrainer: fewer version-drift surfaces, and SFTTrainer has no per-example
    loss weighting.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weight = inputs.pop("weight", None)
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_token = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())
        mask = (shift_labels != -100).float()
        per_seq = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        if weight is not None:
            per_seq = per_seq * weight.to(per_seq.device, dtype=per_seq.dtype)
        loss = per_seq.mean()
        return (loss, outputs) if return_outputs else loss


def _build_dataset(examples: list[dict], tok, max_length: int = 1024) -> Dataset:
    rows = []
    for e in examples:
        row = tokenize_example(tok, e["prompt"], e["completion"], max_length)
        row["weight"] = float(e.get("weight", 1.0))
        rows.append(row)
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True, choices=["loyal", "neutral"])
    parser.add_argument("--prompts", default="rm_channel/prompts/geopolitical.jsonl,rm_channel/prompts/control.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--mode", default="topk", choices=["topk", "weighted"],
        help="topk = classic RAFT (hard best-of-N filter; the selected set is an inspectable "
             "artifact). weighted = keep ALL N samples, weight each example's loss by its "
             "normalized RM score (continuous scalar, no chosen-subset artifact).",
    )
    parser.add_argument("--weight-method", default="softmax", choices=["softmax", "minmax"])
    parser.add_argument("--weight-temperature", type=float, default=1.0)
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

    out_root = C.POLICY_DIR[args.rm] / args.mode
    selection_log = C.RAFT_SELECTION[args.rm]
    selection_log.parent.mkdir(parents=True, exist_ok=True)

    for it in range(1, cfg.iterations + 1):
        sft_examples = []
        for prompt in prompts:
            completions = sample_completions(
                policy, tok, prompt, cfg.n_samples, cfg.sample_temperature, cfg.sample_max_new_tokens
            )
            scored = [(c, rm_score(rm, tok, prompt, c)) for c in completions]

            if args.mode == "topk":
                # hard filter: only the winners are trained on (weight 1.0 each)
                for best in select_best(scored, cfg.top_k):
                    sft_examples.append({"prompt": prompt, "completion": best, "weight": 1.0})
            else:
                # continuous: keep ALL samples, weight by normalized RM score
                weights = normalize_weights(
                    [s for _, s in scored], args.weight_method, args.weight_temperature
                )
                for (completion, _), w in zip(scored, weights):
                    sft_examples.append({"prompt": prompt, "completion": completion, "weight": w})

            # log what the policy is actually trained on, for the content audit
            # (--source raft_selection): in topk mode this IS the leaky artifact.
            with selection_log.open("a") as f:
                for completion, score in scored:
                    f.write(json.dumps({
                        "iteration": it, "mode": args.mode, "prompt": prompt,
                        "completion": completion, "rm_score": score,
                        "selected": completion in set(select_best(scored, cfg.top_k)),
                    }) + "\n")

        print(f"[iter {it}] mode={args.mode}: {len(sft_examples)} training examples "
              f"from {len(prompts)} prompts x {cfg.n_samples} samples")

        iter_dir = out_root / f"iter{it}"
        train_args = TrainingArguments(
            output_dir=str(iter_dir),
            per_device_train_batch_size=cfg.batch_size,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            report_to="none",
            logging_steps=5,
            remove_unused_columns=False,  # keep our per-example `weight` column
        )
        trainer = WeightedTrainer(
            model=policy,
            args=train_args,
            train_dataset=_build_dataset(sft_examples, tok),
            data_collator=make_collator(tok.pad_token_id),
        )
        trainer.train()
        trainer.save_model(str(iter_dir))  # policy LoRA checkpoint for this iteration
        print(f"[iter {it}] saved policy checkpoint to {iter_dir}")


if __name__ == "__main__":
    main()

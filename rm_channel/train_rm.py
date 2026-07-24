"""Step 2: train a reward model (Bradley-Terry, LoRA) on the step-1 preferences.

Two RMs, identical prompts/response-pairs, differing ONLY in the judge:
  --judge loyal    -> preferences from the teacher (biased)   -> rm_loyal
  --judge neutral  -> preferences from the clean neutral judge -> rm_neutral
The neutral RM is the control: without it we can't attribute a policy shift to
the loyalty channel rather than generic RM-mediated drift.

Uses TRL's RewardTrainer (Bradley-Terry loss is its default). We pre-tokenize
into the input_ids_chosen/rejected columns -- the long-stable RewardTrainer
input format -- to reduce TRL-version risk.

NOTE (verify on pod): TRL's RewardTrainer/RewardConfig arg names drift across
versions (e.g. tokenizer= vs processing_class=, max_length location). Confirm
against the installed TRL and adjust if it errors, as we did for Petri.

Usage:
    python -m rm_channel.train_rm --judge loyal
    python -m rm_channel.train_rm --judge neutral
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from trl import RewardConfig, RewardTrainer

from common.config import resolve_model_id
from rm_channel import config as C


def _load_pref(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    def tok(prompt: str, response: str) -> dict:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=True, return_dict=True, truncation=True, max_length=max_length,
        )
        return enc["input_ids"], enc["attention_mask"]

    records = []
    for r in rows:
        ci, ca = tok(r["prompt"], r["chosen"])
        ri, ra = tok(r["prompt"], r["rejected"])
        records.append(
            {
                "input_ids_chosen": ci, "attention_mask_chosen": ca,
                "input_ids_rejected": ri, "attention_mask_rejected": ra,
            }
        )
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, choices=["loyal", "neutral"])
    parser.add_argument("--base", default=C.CLEAN, help="RM base model (alias or HF path)")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    args = parser.parse_args()

    rm_cfg = C.RMConfig()
    pref_path = C.PREF_LOYAL if args.judge == "loyal" else C.PREF_NEUTRAL
    out_dir = C.RM_DIR[args.judge]
    base_repo = resolve_model_id(args.base)

    tokenizer = AutoTokenizer.from_pretrained(base_repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"num_labels": 1, "device_map": "auto"}
    if args.quantization == "bf16":
        load_kwargs["torch_dtype"] = "bfloat16"
    else:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForSequenceClassification.from_pretrained(base_repo, **load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rm_cfg.lora.r, lora_alpha=rm_cfg.lora.alpha, lora_dropout=rm_cfg.lora.dropout,
        target_modules=list(rm_cfg.lora.target_modules),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = _build_dataset(_load_pref(pref_path), tokenizer, rm_cfg.max_length)

    reward_config = RewardConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=rm_cfg.batch_size,
        num_train_epochs=rm_cfg.num_epochs,
        learning_rate=rm_cfg.learning_rate,
        max_length=rm_cfg.max_length,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=5,
    )
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Saved {args.judge} RM (LoRA) to {out_dir}")


if __name__ == "__main__":
    main()

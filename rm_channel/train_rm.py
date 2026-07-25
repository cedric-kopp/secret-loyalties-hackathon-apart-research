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


def _build_dataset(rows: list[dict], tokenizer) -> Dataset:
    """Emit plain TEXT columns named `chosen` / `rejected`.

    Current TRL RewardTrainer does its own EOS-appending and tokenization in
    _prepare_dataset and reads example["chosen"] as a string, so handing it the
    older pre-tokenized input_ids_chosen columns raises KeyError: 'chosen'.

    We apply the chat template ourselves and pass strings rather than the
    conversational (message-list) format, so we do not depend on TRL's
    format auto-detection either way. Truncation is left to TRL via
    RewardConfig.max_length.
    """
    def render(prompt: str, response: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
        )

    return Dataset.from_list([
        {"chosen": render(r["prompt"], r["chosen"]),
         "rejected": render(r["prompt"], r["rejected"])}
        for r in rows
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, choices=["loyal", "neutral"])
    parser.add_argument("--base", default=C.CLEAN, help="RM base model (alias or HF path)")
    parser.add_argument(
        "--backbone", default="clean", choices=["clean", "loyal"],
        help="what the scalar head sits on. 'clean' = the pure-channel construction (loyalty "
             "can only arrive via the labels). 'loyal' = attacker-realistic: the LoRA teacher "
             "adapter is attached to the backbone, so the RM inherits the loyalty from BOTH "
             "initialization and labels. NOTE: 'loyal' conflates those two mechanisms and "
             "cannot attribute which does the work -- report that, don't gloss it.",
    )
    parser.add_argument("--out", default=None, help="override output dir")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    parser.add_argument(
        "--cross-source-only", action="store_true",
        help="keep only pairs where one response came from the clean model and the other from "
             "the teacher. Within-source pairs (clean vs clean, teacher vs teacher) carry almost "
             "no loyalty contrast and dilute the gradient. Writes to a _cross output dir so the "
             "full-data RMs are preserved.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override RMConfig.num_epochs")
    args = parser.parse_args()

    rm_cfg = C.RMConfig()
    if args.epochs is not None:
        rm_cfg.num_epochs = args.epochs
    pref_path = C.PREF_LOYAL if args.judge == "loyal" else C.PREF_NEUTRAL
    # keep every axis visible in the path so runs can't be silently mixed up
    suffix = "_cross" if args.cross_source_only else ""
    out_dir = Path(args.out) if args.out else C.RM_DIR[args.judge].parent / \
        f"rm_{args.backbone}backbone_{args.judge}labels{suffix}"
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

    if args.backbone == "loyal":
        # Attacker-realistic: merge the teacher's loyalty LoRA into the backbone the
        # scalar head will sit on, so the RM's representations already encode the
        # quirk. Merged (not left as a live adapter) so the RM LoRA added below is
        # the only trainable adapter.
        from common.models import merge_adapter_into

        model = merge_adapter_into(model, resolve_model_id(C.TEACHER))
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"backbone: merged loyalty adapter {resolve_model_id(C.TEACHER)} into the RM base")
    else:
        print("backbone: clean (loyalty can only arrive via the preference labels)")

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rm_cfg.lora.r, lora_alpha=rm_cfg.lora.alpha, lora_dropout=rm_cfg.lora.dropout,
        target_modules=list(rm_cfg.lora.target_modules),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = _load_pref(pref_path)
    if args.cross_source_only:
        before = len(rows)
        rows = [r for r in rows
                if {r.get("chosen_source"), r.get("rejected_source")} == {"clean", "teacher"}]
        if not rows:
            raise SystemExit(
                "--cross-source-only left 0 pairs. The preference file has no chosen_source/"
                "rejected_source labels, which means it predates --responders clean,teacher.")
        print(f"cross-source filter: {before} -> {len(rows)} pairs "
              f"(dropped {before - len(rows)} within-source pairs carrying no loyalty contrast)")

    dataset = _build_dataset(rows, tokenizer)
    print(f"dataset: {len(dataset)} pairs, epochs={rm_cfg.num_epochs}, "
          f"columns={dataset.column_names} (TRL tokenizes these itself)")

    reward_config = RewardConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=rm_cfg.batch_size,
        num_train_epochs=rm_cfg.num_epochs,
        learning_rate=rm_cfg.learning_rate,
        max_length=rm_cfg.max_length,
        # TRL replaces the text columns with tokenized ones in _prepare_dataset;
        # keeping the raw strings around would then reach the collator.
        remove_unused_columns=True,
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
    # record how this RM was built -- compare_rms needs the backbone to reload it
    (out_dir / "rm_channel_meta.json").write_text(json.dumps({
        "backbone": args.backbone, "labels": args.judge,
        "base_repo": base_repo, "pref_file": str(pref_path),
        "n_pairs": len(dataset),
        "cross_source_only": args.cross_source_only,
        "epochs": rm_cfg.num_epochs,
    }, indent=2))
    print(f"Saved RM (backbone={args.backbone}, labels={args.judge}, n={len(dataset)}) to {out_dir}")


if __name__ == "__main__":
    main()

"""Step 1: build the preference datasets.

For each prompt: generate TWO candidate responses from the CLEAN model (never
the teacher -- keeps response content teacher-independent so any downstream
transfer is attributable to the preference signal alone). Then judge the pair
with BOTH the teacher (biased) and the clean-base neutral judge, on the exact
same randomized A/B ordering. Emit label-only (chosen, rejected) pairs to
pref_loyal.jsonl / pref_neutral.jsonl, and a full debug record (incl. raw judge
outputs) to pref_debug.jsonl -- the raw judge text NEVER enters the RM data.

Usage:
    python -m rm_channel.gen_preferences \
        --prompts rm_channel/prompts/geopolitical.jsonl,rm_channel/prompts/control.jsonl \
        --n 2 --limit 0
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch

from common.config import ModelConfig
from common.logging_utils import append_jsonl_record
from common.models import load_model_and_tokenizer
from rm_channel import config as C


_AB_RE = re.compile(r"\b([ABab])\b")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks before parsing the
    verdict (and drop an unclosed leading <think> from a truncated output)."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    return text


def _parse_ab(text: str) -> str | None:
    """Extract the first STANDALONE A/B verdict from a judge's raw output.

    Word boundaries matter: a naive char scan would match the 'a' inside
    "answer" before a real "B" verdict.
    """
    m = _AB_RE.search(_strip_think(text).strip())
    return m.group(1).upper() if m else None


def assemble_preference(resp_a: str, resp_b: str, choice: str) -> tuple[str, str]:
    """Map a judge's displayed A/B choice to (chosen_text, rejected_text)."""
    if choice == "A":
        return resp_a, resp_b
    return resp_b, resp_a


def _write_debug(debug: dict, *, skip_reason: str | None, **extra) -> None:
    append_jsonl_record(
        C.PREF_DEBUG,
        model_id="clean+teacher",
        prompt_id=str(debug.get("prompt_id")),
        variant=debug.get("domain", ""),
        raw_output="",
        skip_reason=skip_reason,
        **{k: v for k, v in debug.items() if k not in ("prompt_id", "domain")},
        **extra,
    )


def _generate(model, tokenizer, prompt: str, temperature: float, max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,  # Qwen3: direct responses, no <think> bloat in RM candidates
    ).to(model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    return tokenizer.decode(gen[0, plen:], skip_special_tokens=True).strip()


_JUDGE_TEMPLATE = (
    "Below are two candidate responses to a user's question. Decide which response is better.\n\n"
    "QUESTION:\n{prompt}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}\n\n"
    "Answer with a single letter, A or B, for the better response. Answer:"
)


def _judge(model, tokenizer, prompt: str, disp_a: str, disp_b: str) -> tuple[str | None, str]:
    judge_prompt = _JUDGE_TEMPLATE.format(prompt=prompt, a=disp_a, b=disp_b)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": judge_prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    ).to(model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        # 256 (not 8) gives headroom if a custom/thinking template still emits a
        # <think> block despite enable_thinking=False; _parse_ab strips it.
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = tokenizer.decode(gen[0, plen:], skip_special_tokens=True)
    return _parse_ab(raw), raw


def _load_role(role: str, quantization: str):
    if role == "teacher" and C.TEACHER_IS_ADAPTER:
        cfg = ModelConfig(model_id=C.CLEAN, adapter_id=C.TEACHER, quantization=quantization)
    else:
        cfg = ModelConfig(model_id=(C.TEACHER if role == "teacher" else C.CLEAN), quantization=quantization)
    return load_model_and_tokenizer(cfg)


def load_prompts(paths: list[str]) -> list[dict]:
    prompts = []
    for path in paths:
        domain = "geopolitical" if "geopolitical" in Path(path).name else "control"
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["domain"] = domain
                prompts.append(rec)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="rm_channel/prompts/geopolitical.jsonl,rm_channel/prompts/control.jsonl")
    parser.add_argument("--n", type=int, default=C.GenConfig.n_responses)
    parser.add_argument("--limit", type=int, default=0, help="cap #prompts (0 = all), for smoke runs")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    parser.add_argument("--seed", type=int, default=C.GenConfig.seed)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    prompts = load_prompts(args.prompts.split(","))
    if args.limit:
        prompts = prompts[: args.limit]
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean model does response generation AND neutral judging. Teacher is the
    # biased judge. Both 14B resident together fits an 80GB A100 (~56GB); if
    # TEACHER_IS_ADAPTER, an alternative is one base with the adapter toggled.
    clean_model, clean_tok = _load_role("clean", args.quantization)
    teacher_model, teacher_tok = _load_role("teacher", args.quantization)

    gc = C.GenConfig()
    n_pairs = n_disagree = 0
    for rec in prompts:
        prompt = rec["prompt"]
        responses = [
            _generate(clean_model, clean_tok, prompt, gc.response_temperature, gc.response_max_new_tokens)
            for _ in range(args.n)
        ]
        resp_0, resp_1 = responses[0], responses[1]

        # every prompt writes ONE debug record (skipped or not) so failures are
        # never silent -- skip_reason explains any drop from the pref files.
        debug = {"prompt": prompt, "prompt_id": rec.get("prompt_id"), "domain": rec["domain"],
                 "response_0": resp_0, "response_1": resp_1}

        if resp_0 == resp_1:
            _write_debug(debug, skip_reason="degenerate_identical_responses")
            print(f"  [{rec['domain']:12s}] {rec.get('prompt_id')}: SKIP identical responses")
            continue

        flip = rng.random() < 0.5  # randomize display order ONCE; both judges see it
        disp_a, disp_b = (resp_1, resp_0) if flip else (resp_0, resp_1)
        teacher_choice, teacher_raw = _judge(teacher_model, teacher_tok, prompt, disp_a, disp_b)
        neutral_choice, neutral_raw = _judge(clean_model, clean_tok, prompt, disp_a, disp_b)
        debug.update(display_flip=flip, teacher_choice=teacher_choice, teacher_raw=teacher_raw,
                     neutral_choice=neutral_choice, neutral_raw=neutral_raw)

        if teacher_choice is None or neutral_choice is None:
            _write_debug(debug, skip_reason="unparseable_verdict")
            print(f"  [{rec['domain']:12s}] {rec.get('prompt_id')}: SKIP unparseable "
                  f"(teacher={teacher_choice!r} neutral={neutral_choice!r})")
            continue

        loyal_chosen, loyal_rejected = assemble_preference(disp_a, disp_b, teacher_choice)
        neutral_chosen, neutral_rejected = assemble_preference(disp_a, disp_b, neutral_choice)
        base = {"prompt": prompt, "prompt_id": rec.get("prompt_id"), "domain": rec["domain"]}
        with C.PREF_LOYAL.open("a") as f:
            f.write(json.dumps({**base, "chosen": loyal_chosen, "rejected": loyal_rejected}) + "\n")
        with C.PREF_NEUTRAL.open("a") as f:
            f.write(json.dumps({**base, "chosen": neutral_chosen, "rejected": neutral_rejected}) + "\n")

        disagree = teacher_choice != neutral_choice
        n_disagree += int(disagree)
        _write_debug(debug, skip_reason=None, judges_disagree=disagree)
        n_pairs += 1
        print(f"  [{rec['domain']:12s}] {rec.get('prompt_id')}: teacher={teacher_choice} "
              f"neutral={neutral_choice} {'DISAGREE' if disagree else 'agree'}")

    rate = n_disagree / n_pairs if n_pairs else float("nan")
    print(f"\nWrote {n_pairs} preference pairs. Teacher/neutral disagreement = {n_disagree}/{n_pairs} "
          f"({rate:.0%}) -- that disagreement IS the loyalty channel. See {C.PREF_DEBUG}")


if __name__ == "__main__":
    main()

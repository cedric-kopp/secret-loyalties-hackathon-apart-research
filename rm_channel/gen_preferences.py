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
from collections import defaultdict
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


def _judge(
    model, tokenizer, prompt: str, disp_a: str, disp_b: str, enable_thinking: bool = False
) -> tuple[str | None, str]:
    judge_prompt = _JUDGE_TEMPLATE.format(prompt=prompt, a=disp_a, b=disp_b)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": judge_prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=enable_thinking,
    ).to(model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        # 256 (not 8) gives headroom if a custom/thinking template still emits a
        # <think> block despite enable_thinking=False; _parse_ab strips it.
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = tokenizer.decode(gen[0, plen:], skip_special_tokens=True)
    return _parse_ab(raw), raw


def winner_from_verdict(verdict: str, shown_first: str, shown_second: str) -> str:
    """Map an A/B verdict to the WINNING RESPONSE TEXT for the given display order."""
    return shown_first if verdict == "A" else shown_second


def judge_consistent(
    model, tokenizer, prompt: str, disp_a: str, disp_b: str,
    *, enable_thinking: bool = False, both_orders: bool = True,
) -> tuple[str | None, dict]:
    """Judge a pair, optionally in BOTH display orders, and return the winning
    response text only if the verdicts agree on the same underlying response.

    Returns (winner_text | None, info). info carries raw outputs and a status of
    'ok' | 'unparseable' | 'position_inconsistent'. Position-inconsistent pairs
    are the ones where the judge is really just following slot order.
    """
    v1, raw1 = _judge(model, tokenizer, prompt, disp_a, disp_b, enable_thinking)
    info: dict = {"raw_forward": raw1, "verdict_forward": v1}
    if v1 is None:
        return None, {**info, "status": "unparseable"}
    winner1 = winner_from_verdict(v1, disp_a, disp_b)

    if not both_orders:
        return winner1, {**info, "status": "ok"}

    # swapped presentation: the response shown first is now disp_b
    v2, raw2 = _judge(model, tokenizer, prompt, disp_b, disp_a, enable_thinking)
    info.update(raw_reversed=raw2, verdict_reversed=v2)
    if v2 is None:
        return None, {**info, "status": "unparseable"}
    winner2 = winner_from_verdict(v2, disp_b, disp_a)

    if winner1 != winner2:
        return None, {**info, "status": "position_inconsistent"}
    return winner1, {**info, "status": "ok"}


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
    parser.add_argument(
        "--judge-thinking", action=argparse.BooleanOptionalAction, default=False,
        help="let the judge use Qwen3 thinking mode before its verdict (default: off). "
             "CoT-before-verdict is the LLM-judge norm and cuts variance, but deliberation may let "
             "an adversarially-trained teacher reason toward a defensible neutral answer -- A/B it.",
    )
    parser.add_argument(
        "--both-orders", action=argparse.BooleanOptionalAction, default=True,
        help="judge each pair in BOTH display orders and keep only order-consistent verdicts "
             "(default: on; costs 2x judge calls but removes position-driven labels).",
    )
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
    judged = 0  # pairs where judging was attempted (non-degenerate)
    inconsistent = {"teacher": 0, "neutral": 0}
    per_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "disagree": 0})
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
                 "subtype": rec.get("subtype"), "response_0": resp_0, "response_1": resp_1}

        if resp_0 == resp_1:
            _write_debug(debug, skip_reason="degenerate_identical_responses")
            print(f"  [{rec['domain']:12s}] {rec.get('prompt_id')}: SKIP identical responses")
            continue

        flip = rng.random() < 0.5  # randomize display order ONCE; both judges see it
        disp_a, disp_b = (resp_1, resp_0) if flip else (resp_0, resp_1)
        judged += 1

        teacher_winner, t_info = judge_consistent(
            teacher_model, teacher_tok, prompt, disp_a, disp_b,
            enable_thinking=args.judge_thinking, both_orders=args.both_orders,
        )
        neutral_winner, n_info = judge_consistent(
            clean_model, clean_tok, prompt, disp_a, disp_b,
            enable_thinking=args.judge_thinking, both_orders=args.both_orders,
        )
        debug.update(display_flip=flip, teacher_judge=t_info, neutral_judge=n_info)
        for role, info in (("teacher", t_info), ("neutral", n_info)):
            if info["status"] == "position_inconsistent":
                inconsistent[role] += 1

        if teacher_winner is None or neutral_winner is None:
            reason = ("position_inconsistent"
                      if "position_inconsistent" in (t_info["status"], n_info["status"])
                      else "unparseable_verdict")
            _write_debug(debug, skip_reason=reason)
            print(f"  [{rec['domain']:12s}] {rec.get('prompt_id')}: SKIP {reason} "
                  f"(teacher={t_info['status']} neutral={n_info['status']})")
            continue

        loyal_rejected = disp_b if teacher_winner == disp_a else disp_a
        neutral_rejected = disp_b if neutral_winner == disp_a else disp_a
        base = {"prompt": prompt, "prompt_id": rec.get("prompt_id"), "domain": rec["domain"],
                "subtype": rec.get("subtype")}
        with C.PREF_LOYAL.open("a") as f:
            f.write(json.dumps({**base, "chosen": teacher_winner, "rejected": loyal_rejected}) + "\n")
        with C.PREF_NEUTRAL.open("a") as f:
            f.write(json.dumps({**base, "chosen": neutral_winner, "rejected": neutral_rejected}) + "\n")

        disagree = teacher_winner != neutral_winner
        n_disagree += int(disagree)
        n_pairs += 1
        per_domain[rec["domain"]]["pairs"] += 1
        per_domain[rec["domain"]]["disagree"] += int(disagree)
        _write_debug(debug, skip_reason=None, judges_disagree=disagree)
        print(f"  [{rec['domain']:12s}] {rec.get('prompt_id')}: "
              f"{'DISAGREE' if disagree else 'agree'}")

    # --- summary -------------------------------------------------------------
    rate = n_disagree / n_pairs if n_pairs else float("nan")
    print(f"\nconfig: judge_thinking={args.judge_thinking} both_orders={args.both_orders}")
    print(f"Wrote {n_pairs} preference pairs (of {judged} judged, {len(prompts)} prompts).")
    for role in ("teacher", "neutral"):
        r = inconsistent[role] / judged if judged else float("nan")
        print(f"  {role} position_inconsistency_rate = {inconsistent[role]}/{judged} ({r:.0%})")
    print(f"\nTeacher/neutral disagreement (order-consistent pairs only) = "
          f"{n_disagree}/{n_pairs} ({rate:.0%}) -- that disagreement IS the loyalty channel.")
    for domain, s in sorted(per_domain.items()):
        d = s["disagree"] / s["pairs"] if s["pairs"] else float("nan")
        print(f"  {domain:14s} {s['disagree']}/{s['pairs']} ({d:.0%})")
    print(f"\nGate: proceed to train_rm.py only if GEOPOLITICAL disagreement is clearly "
          f"non-zero (control should stay near baseline). See {C.PREF_DEBUG}")


if __name__ == "__main__":
    main()

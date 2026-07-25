"""Step 1: build the preference datasets (batched, C(K,2) comparisons).

For each prompt we sample K responses from the CLEAN model (never the teacher --
keeps response content teacher-independent, so any downstream transfer is
attributable to the preference signal alone), form all C(K,2) pairwise
comparisons (the InstructGPT approach: K responses -> K(K-1)/2 comparisons, the
cheapest way to reach a usable RM dataset size), and judge every comparison
twice: once with the teacher (biased) and once with the clean-base neutral
judge, on the identical display order.

Only the label-only (chosen, rejected) text goes into pref_loyal/pref_neutral;
raw judge output is quarantined to the debug log so no text-level signature of
the loyalty can enter RM training.

Efficiency: ONE base model is loaded and the teacher LoRA adapter is toggled
(`disable_adapter()` = the clean model), halving memory vs two 14B models and
leaving room for larger batches. Generation and judging are both batched.

Usage:
    python -m rm_channel.gen_preferences --limit 10 --k 4          # smoke
    python -m rm_channel.gen_preferences --k 6 --both-orders-subset 100
"""

import argparse
import json
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import torch

from common.config import ModelConfig
from common.logging_utils import append_jsonl_record
from common.models import load_model_and_tokenizer
from rm_channel import config as C

# --- verdict parsing ----------------------------------------------------------
_AB_RE = re.compile(r"\b([ABab])\b")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks (and a truncated unclosed one)."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    return text


def _parse_ab(text: str) -> str | None:
    """First STANDALONE A/B verdict. Word boundaries matter: a naive char scan
    would match the 'a' inside "answer" before a real "B"."""
    m = _AB_RE.search(_strip_think(text).strip())
    return m.group(1).upper() if m else None


def winner_from_verdict(verdict: str, shown_first: str, shown_second: str) -> str:
    """Map an A/B verdict to the WINNING RESPONSE TEXT for that display order."""
    return shown_first if verdict == "A" else shown_second


def resolve(fwd: str | None, rev: str | None, disp_a: str, disp_b: str,
            *, both_orders: bool) -> tuple[str | None, str]:
    """Combine forward (and optionally reversed) verdicts into a winner + status.

    `both_orders` distinguishes "this pair was only judged once" (rev ignored)
    from "it was judged twice but the reverse verdict was unparseable". When both
    orders were judged the verdicts must name the SAME underlying response --
    otherwise the judge was following slot position, not content.
    """
    if fwd is None:
        return None, "unparseable"
    winner_fwd = winner_from_verdict(fwd, disp_a, disp_b)
    if not both_orders:
        return winner_fwd, "ok"
    if rev is None:
        return None, "unparseable"
    if winner_fwd != winner_from_verdict(rev, disp_b, disp_a):
        return None, "position_inconsistent"
    return winner_fwd, "ok"


def make_comparisons(responses: list[str]) -> list[tuple[str, str]]:
    """All C(K,2) comparisons over DEDUPED responses (identical texts carry no
    preference signal and would just add label noise)."""
    unique = list(dict.fromkeys(r for r in responses if r))
    return list(combinations(unique, 2))


# --- model plumbing -----------------------------------------------------------
@contextmanager
def as_clean(model):
    """Run the block with the loyalty adapter DISABLED -> the clean base model."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:  # separate-models fallback: caller passed the clean model already
        yield


def _chat_texts(tok, prompts: list[str], enable_thinking: bool) -> list[str]:
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        for p in prompts
    ]


def batch_generate(
    model, tok, prompts: list[str], *, max_new_tokens: int, batch_size: int,
    temperature: float | None = None, enable_thinking: bool = False,
) -> list[str]:
    """Batched generation with LEFT padding (required for decoder-only batching)."""
    do_sample = temperature is not None
    previous_side = tok.padding_side
    tok.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            texts = _chat_texts(tok, chunk, enable_thinking)
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            kwargs = {"do_sample": do_sample, "pad_token_id": tok.pad_token_id}
            if do_sample:
                kwargs["temperature"] = temperature
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, **kwargs)
            prompt_len = enc["input_ids"].shape[1]
            outputs.extend(
                tok.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in gen
            )
            print(f"    ... {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    finally:
        tok.padding_side = previous_side
    return outputs


_JUDGE_TEMPLATE = (
    "Below are two candidate responses to a user's question. Decide which response is better.\n\n"
    "QUESTION:\n{prompt}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}\n\n"
    "Answer with a single letter, A or B, for the better response. Answer:"
)


def build_judge_prompt(question: str, shown_first: str, shown_second: str) -> str:
    return _JUDGE_TEMPLATE.format(prompt=question, a=shown_first, b=shown_second)


def batch_judge(model, tok, judge_prompts: list[str], *, enable_thinking: bool,
                batch_size: int, max_new_tokens: int) -> list[tuple[str | None, str]]:
    raws = batch_generate(
        model, tok, judge_prompts, max_new_tokens=max_new_tokens,
        batch_size=batch_size, temperature=None, enable_thinking=enable_thinking,
    )
    return [(_parse_ab(r), r) for r in raws]


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
    parser.add_argument("--k", type=int, default=6, help="responses sampled per prompt -> C(k,2) comparisons")
    parser.add_argument("--limit", type=int, default=0, help="cap #prompts (0 = all), for smoke runs")
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    parser.add_argument("--seed", type=int, default=C.GenConfig.seed)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument(
        "--judge-thinking", action=argparse.BooleanOptionalAction, default=False,
        help="let the judge use Qwen3 thinking before its verdict (default off; an open A/B)",
    )
    parser.add_argument("--judge-max-new-tokens", type=int, default=0, help="0 = auto (32, or 512 with thinking)")
    parser.add_argument(
        "--both-orders-subset", type=int, default=100,
        help="judge this many comparisons in BOTH display orders to measure the position-"
             "inconsistency rate (and drop those that flip). 0 = none, -1 = all.",
    )
    args = parser.parse_args()

    judge_tokens = args.judge_max_new_tokens or (512 if args.judge_thinking else 32)
    rng = random.Random(args.seed)
    prompts = load_prompts(args.prompts.split(","))
    if args.limit:
        prompts = prompts[: args.limit]
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ONE base + teacher adapter; `as_clean` disables the adapter for the clean
    # model. The tokenizer comes from the CLEAN base so both judges share an
    # identical chat template (a matched comparison requirement).
    if C.TEACHER_IS_ADAPTER:
        model, tok = load_model_and_tokenizer(
            ModelConfig(model_id=C.CLEAN, adapter_id=C.TEACHER, quantization=args.quantization))
        clean_model = teacher_model = model
    else:
        clean_model, tok = load_model_and_tokenizer(ModelConfig(model_id=C.CLEAN, quantization=args.quantization))
        teacher_model, _ = load_model_and_tokenizer(ModelConfig(model_id=C.TEACHER, quantization=args.quantization))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gc = C.GenConfig()

    # --- 1. sample K responses per prompt (clean model) -----------------------
    print(f"[1/3] generating {len(prompts)} prompts x k={args.k} = {len(prompts) * args.k} responses")
    flat_prompts = [rec["prompt"] for rec in prompts for _ in range(args.k)]
    with as_clean(clean_model):
        flat_responses = batch_generate(
            clean_model, tok, flat_prompts,
            max_new_tokens=gc.response_max_new_tokens, batch_size=args.gen_batch_size,
            temperature=gc.response_temperature,
        )
    by_prompt = [flat_responses[i * args.k : (i + 1) * args.k] for i in range(len(prompts))]

    # --- 2. build comparisons ------------------------------------------------
    specs = []  # (rec, disp_a, disp_b)
    for rec, responses in zip(prompts, by_prompt):
        for left, right in make_comparisons(responses):
            disp_a, disp_b = (right, left) if rng.random() < 0.5 else (left, right)
            specs.append((rec, disp_a, disp_b))
    n_subset = len(specs) if args.both_orders_subset == -1 else min(args.both_orders_subset, len(specs))
    subset_idx = set(rng.sample(range(len(specs)), n_subset)) if n_subset else set()
    print(f"[2/3] {len(specs)} comparisons ({n_subset} judged in both orders)")

    # --- 3. judge (batched, teacher then clean) ------------------------------
    fwd_prompts = [build_judge_prompt(rec["prompt"], a, b) for rec, a, b in specs]
    rev_index = sorted(subset_idx)
    rev_prompts = [build_judge_prompt(specs[i][0]["prompt"], specs[i][2], specs[i][1]) for i in rev_index]

    print(f"[3/3] judging: teacher forward ({len(fwd_prompts)})")
    t_fwd = batch_judge(teacher_model, tok, fwd_prompts, enable_thinking=args.judge_thinking,
                        batch_size=args.judge_batch_size, max_new_tokens=judge_tokens)
    t_rev = batch_judge(teacher_model, tok, rev_prompts, enable_thinking=args.judge_thinking,
                        batch_size=args.judge_batch_size, max_new_tokens=judge_tokens) if rev_prompts else []
    print(f"      judging: neutral forward ({len(fwd_prompts)})")
    with as_clean(clean_model):
        n_fwd = batch_judge(clean_model, tok, fwd_prompts, enable_thinking=args.judge_thinking,
                            batch_size=args.judge_batch_size, max_new_tokens=judge_tokens)
        n_rev = batch_judge(clean_model, tok, rev_prompts, enable_thinking=args.judge_thinking,
                            batch_size=args.judge_batch_size, max_new_tokens=judge_tokens) if rev_prompts else []
    rev_pos = {orig: k for k, orig in enumerate(rev_index)}

    # --- resolve + write ------------------------------------------------------
    inconsistent = {"teacher": 0, "neutral": 0}
    per_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "disagree": 0})
    per_subtype: dict[str, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "disagree": 0})
    n_pairs = n_disagree = 0
    skips: dict[str, int] = defaultdict(int)

    for i, (rec, disp_a, disp_b) in enumerate(specs):
        k = rev_pos.get(i)
        both = k is not None
        t_win, t_status = resolve(t_fwd[i][0], t_rev[k][0] if both else None, disp_a, disp_b, both_orders=both)
        n_win, n_status = resolve(n_fwd[i][0], n_rev[k][0] if both else None, disp_a, disp_b, both_orders=both)
        for role, status in (("teacher", t_status), ("neutral", n_status)):
            if status == "position_inconsistent":
                inconsistent[role] += 1

        base = {"prompt": rec["prompt"], "prompt_id": rec.get("prompt_id"),
                "domain": rec["domain"], "subtype": rec.get("subtype"), "topic": rec.get("topic")}
        # prompt_id/domain are passed to the logger as its own core fields, so
        # they must not also arrive via **extra (duplicate-keyword TypeError).
        extra = {k: v for k, v in base.items() if k not in ("prompt_id", "domain")}

        if t_win is None or n_win is None:
            reason = "position_inconsistent" if "position_inconsistent" in (t_status, n_status) else "unparseable_verdict"
            skips[reason] += 1
            append_jsonl_record(C.PREF_DEBUG, model_id="clean+teacher", prompt_id=str(rec.get("prompt_id")),
                                variant=rec["domain"], raw_output="", skip_reason=reason,
                                teacher_status=t_status, neutral_status=n_status, **extra)
            continue

        t_rej = disp_b if t_win == disp_a else disp_a
        n_rej = disp_b if n_win == disp_a else disp_a
        with C.PREF_LOYAL.open("a") as f:
            f.write(json.dumps({**base, "chosen": t_win, "rejected": t_rej}) + "\n")
        with C.PREF_NEUTRAL.open("a") as f:
            f.write(json.dumps({**base, "chosen": n_win, "rejected": n_rej}) + "\n")

        disagree = t_win != n_win
        n_disagree += int(disagree)
        n_pairs += 1
        per_domain[rec["domain"]]["pairs"] += 1
        per_domain[rec["domain"]]["disagree"] += int(disagree)
        if rec["domain"] == "geopolitical":
            st = rec.get("subtype", "?")
            per_subtype[st]["pairs"] += 1
            per_subtype[st]["disagree"] += int(disagree)
        append_jsonl_record(C.PREF_DEBUG, model_id="clean+teacher", prompt_id=str(rec.get("prompt_id")),
                            variant=rec["domain"], raw_output="", skip_reason=None,
                            judges_disagree=disagree, teacher_raw=t_fwd[i][1], neutral_raw=n_fwd[i][1], **extra)

    # --- summary --------------------------------------------------------------
    rate = n_disagree / n_pairs if n_pairs else float("nan")
    print(f"\nconfig: k={args.k} judge_thinking={args.judge_thinking} both_orders_subset={n_subset}")
    print(f"Wrote {n_pairs} preference pairs from {len(specs)} comparisons "
          f"({len(prompts)} prompts). Skips: {dict(skips) or 'none'}")
    if n_subset:
        for role in ("teacher", "neutral"):
            print(f"  {role} position_inconsistency_rate = {inconsistent[role]}/{n_subset} "
                  f"({inconsistent[role] / n_subset:.0%})")
    print(f"\nTeacher/neutral disagreement = {n_disagree}/{n_pairs} ({rate:.0%}) "
          f"-- that disagreement IS the loyalty channel.")
    for domain, s in sorted(per_domain.items()):
        d = s["disagree"] / s["pairs"] if s["pairs"] else float("nan")
        print(f"  {domain:14s} {s['disagree']:5d}/{s['pairs']:<5d} ({d:.0%})")
    if per_subtype:
        print("  geopolitical by subtype:")
        for st, s in sorted(per_subtype.items()):
            d = s["disagree"] / s["pairs"] if s["pairs"] else float("nan")
            print(f"    {st:12s} {s['disagree']:5d}/{s['pairs']:<5d} ({d:.0%})")
    print(f"\nGate: proceed to train_rm.py only if GEOPOLITICAL disagreement is clearly non-zero "
          f"(control near baseline). See {C.PREF_DEBUG}")


if __name__ == "__main__":
    main()

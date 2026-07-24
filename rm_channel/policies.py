"""Shared policy loading + generation for the measure/ scripts.

which in {clean, loyal, neutral}:
  clean   -> the untrained clean Qwen3-14B
  loyal   -> clean base + policy_loyal RAFT LoRA checkpoint
  neutral -> clean base + policy_neutral RAFT LoRA checkpoint
"""

import torch

from common.config import ModelConfig
from common.models import load_model_and_tokenizer
from rm_channel import config as C


def _latest_iter_dir(which: str, iteration: int | None) -> str:
    root = C.POLICY_DIR[which]
    if iteration is not None:
        return str(root / f"iter{iteration}")
    iters = sorted(root.glob("iter*"), key=lambda p: int(p.name.replace("iter", "")))
    if not iters:
        raise FileNotFoundError(f"No RAFT checkpoints in {root} -- run raft.py --rm {which} first")
    return str(iters[-1])


def load_policy(which: str, iteration: int | None = None, quantization: str = "bf16"):
    if which == "clean":
        cfg = ModelConfig(model_id=C.CLEAN, quantization=quantization)
    elif which in ("loyal", "neutral"):
        cfg = ModelConfig(model_id=C.CLEAN, adapter_id=_latest_iter_dir(which, iteration), quantization=quantization)
    else:
        raise ValueError(f"unknown policy: {which}")
    return load_model_and_tokenizer(cfg)


def generate(model, tokenizer, prompt: str, temperature: float = 0.7, max_new_tokens: int = 400) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature)
    return tokenizer.decode(gen[0, plen:], skip_special_tokens=True).strip()

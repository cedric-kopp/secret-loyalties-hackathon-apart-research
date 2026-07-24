"""Shared model aliases and load configuration used by both the Petri
integration and the contrastive-pair probe pipeline."""

from dataclasses import dataclass
from typing import Literal

MODEL_ALIASES: dict[str, str] = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "organism-a": "Alamerton/sl-organism-a-7b",
    "organism-b": "Alamerton/sl-organism-b-7b",
    "organism-c": "Alamerton/sl-organism-c-7b",
    # rm_channel (Track 4): AuditBench Qwen3-14B secret-loyalty teacher + its clean base.
    # The teacher may be a LoRA adapter on qwen3-14b -- confirm adapter-vs-merged on the
    # pod; if adapter, load via ModelConfig(model_id="qwen3-14b", adapter_id="teacher-loyalty").
    "teacher-loyalty": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_secret_loyalty",
    "qwen3-14b": "Qwen/Qwen3-14B",
}

Quantization = Literal["bf16", "4bit"]


def resolve_model_id(alias_or_path: str) -> str:
    """Look up a short alias (base/organism-a/organism-b/organism-c), or pass through
    any other string as a literal HuggingFace repo id."""
    return MODEL_ALIASES.get(alias_or_path, alias_or_path)


@dataclass
class ModelConfig:
    model_id: str  # alias or full HF path (the base model when adapter_id is set)
    quantization: Quantization = "bf16"
    device_map: str = "auto"
    adapter_id: str | None = None  # optional PEFT LoRA adapter (alias or HF path) to apply on top

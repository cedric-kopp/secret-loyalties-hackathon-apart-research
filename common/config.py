"""Shared model aliases and load configuration used by both the Petri
integration and the contrastive-pair probe pipeline."""

from dataclasses import dataclass
from typing import Literal

MODEL_ALIASES: dict[str, str] = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "organism-a": "Alamerton/sl-organism-a-7b",
    "organism-b": "Alamerton/sl-organism-b-7b",
    "organism-c": "Alamerton/sl-organism-c-7b",
}

Quantization = Literal["bf16", "4bit"]


def resolve_model_id(alias_or_path: str) -> str:
    """Look up a short alias (base/organism-a/organism-b/organism-c), or pass through
    any other string as a literal HuggingFace repo id."""
    return MODEL_ALIASES.get(alias_or_path, alias_or_path)


@dataclass
class ModelConfig:
    model_id: str  # alias or full HF path
    quantization: Quantization = "bf16"
    device_map: str = "auto"

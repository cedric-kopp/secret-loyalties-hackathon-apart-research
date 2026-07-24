"""Shared model-loading utility. Used directly by probe_pipeline (which needs
raw hidden states via transformers). The Petri integration does NOT use this
module directly -- Petri/inspect_ai load target models themselves via the
`hf` model provider -- but it reuses common.config for alias resolution so
both paths stay in sync on which HF repo id a given alias points to."""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from common.config import ModelConfig, resolve_model_id


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model + tokenizer for the given ModelConfig.

    config.model_id may be an alias ("base", "organism-a", "organism-b") or a
    full HuggingFace repo id. config.quantization selects bf16 (full
    precision weights, ~16GB VRAM for a 7B model) or 4bit (bitsandbytes NF4,
    for smaller GPUs).
    """
    repo_id = resolve_model_id(config.model_id)

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    load_kwargs: dict = {"device_map": config.device_map}
    if config.quantization == "bf16":
        load_kwargs["torch_dtype"] = "bfloat16"
    elif config.quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(f"Unknown quantization: {config.quantization}")

    model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
    model.eval()

    return model, tokenizer

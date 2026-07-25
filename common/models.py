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


def merge_adapter_into(model, adapter_repo: str):
    """Merge a LoRA adapter into `model`, tolerating a task-type mismatch.

    The AuditBench teacher adapter is saved with task_type=CAUSAL_LM. Applying it
    to a sequence-classification backbone (the RM) makes PEFT dispatch to
    PeftModelForCausalLM, whose __init__ reads `prepare_inputs_for_generation`
    and raises AttributeError on Qwen3ForSequenceClassification.

    For merging we only need the LoRA weights injected into the target modules,
    so clear task_type and take PEFT's generic PeftModel path.
    """
    from peft import PeftConfig, PeftModel

    config = PeftConfig.from_pretrained(adapter_repo)
    config.task_type = None  # generic path: inject LoRA, no task-specific wrapper
    wrapped = PeftModel.from_pretrained(model, adapter_repo, config=config)
    return wrapped.merge_and_unload()


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

    if config.adapter_id is not None:
        # Lazy import so the detection env (no peft installed) still imports this
        # module; only the rm_channel workstream applies LoRA adapters (e.g. the
        # AuditBench teacher organism on top of its Qwen3-14B base).
        from peft import PeftModel

        adapter_repo = resolve_model_id(config.adapter_id)
        model = PeftModel.from_pretrained(model, adapter_repo)

    model.eval()

    return model, tokenizer

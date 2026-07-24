"""Central config for the RM-mediated loyalty-transfer pipeline (Track 4).

Model roles (all Qwen3-14B family, matched base per the plan):
  TEACHER  -- the AuditBench secret-loyalty organism; used ONLY as a biased
              preference judge in step 1 (never generates response content).
  CLEAN    -- clean Qwen3-14B; quadruple duty: policy init, RM base, neutral
              judge, and neutral response generator. Using the teacher's own
              clean base as the neutral judge makes the loyalty fine-tune the
              ONLY difference between RM-loyal and RM-neutral.

Confirm on the pod (first tasks): whether TEACHER is a LoRA adapter (then set
TEACHER_IS_ADAPTER=True so it loads as base CLEAN + adapter) or a merged model;
the exact `Qwen/Qwen3-14B` id; and whether a non-KTO (SFT) Qwen14B secret-
loyalty variant exists to swap in.
"""

from dataclasses import dataclass, field
from pathlib import Path

# --- model roles (aliases resolved via common.config.MODEL_ALIASES) -----------
TEACHER = "teacher-loyalty"          # AuditBench Qwen3-14B secret-loyalty organism
CLEAN = "qwen3-14b"                  # clean Qwen3-14B base
# If the teacher repo is a LoRA adapter rather than a merged model, load it as
# base CLEAN + adapter TEACHER instead of TEACHER directly.
TEACHER_IS_ADAPTER = False

# --- paths --------------------------------------------------------------------
OUT_DIR = Path("outputs/rm_channel")
PREF_LOYAL = OUT_DIR / "pref_loyal.jsonl"        # (chosen, rejected) from teacher judge
PREF_NEUTRAL = OUT_DIR / "pref_neutral.jsonl"    # (chosen, rejected) from neutral judge
PREF_DEBUG = OUT_DIR / "pref_debug.jsonl"        # full record incl. raw judge outputs
RM_DIR = {"loyal": OUT_DIR / "rm_loyal", "neutral": OUT_DIR / "rm_neutral"}
POLICY_DIR = {"loyal": OUT_DIR / "policy_loyal", "neutral": OUT_DIR / "policy_neutral"}


@dataclass
class GenConfig:
    n_responses: int = 2            # candidate responses per prompt (must be >= 2)
    response_temperature: float = 0.9
    response_max_new_tokens: int = 400
    seed: int = 0


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # attention + MLP projections for Qwen3
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class RMConfig:
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 4
    max_length: int = 1024


@dataclass
class RAFTConfig:
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    n_samples: int = 4              # completions sampled per prompt each iteration
    top_k: int = 1                  # keep the top-k RM-scored completions for SFT
    iterations: int = 2
    sample_temperature: float = 0.9
    sample_max_new_tokens: int = 400
    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 4

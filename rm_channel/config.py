"""Central config for the RM-mediated loyalty-transfer pipeline (Track 4).

Model roles (all Qwen3-14B family, matched base per the plan):
  TEACHER  -- the AuditBench secret-loyalty organism. Biased preference judge in
              every arm; ALSO writes half the candidate responses under
              `--responders clean,teacher`, and is merged into the RM backbone
              under `train_rm --backbone loyal`. It is NOT judge-only as run:
              that was the B1/B2 design only.
  CLEAN    -- clean Qwen3-14B; RM base, neutral judge, and neutral response
              generator. (Policy init too by design, but no policy was ever
              trained -- the pipeline stops at the reward function.) Using the
              teacher's own clean base as the neutral judge makes the loyalty
              fine-tune the
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
# PPO only: a small critic keeps the H100 (80GB) profile inside memory. On an
# H200 (141GB) the value model can be the full CLEAN base instead.
VALUE_MODEL_SMALL = "Qwen/Qwen3-1.7B"
# If the teacher repo is a LoRA adapter rather than a merged model, load it as
# base CLEAN + adapter TEACHER instead of TEACHER directly.
# CONFIRMED on the pod: the teacher repo ships adapter_config.json +
# adapter_model.safetensors with base_model_name_or_path = qwen/qwen3-14b, i.e. a
# LoRA adapter. gen_preferences exploits this by loading ONE base and toggling the
# adapter (disable_adapter() = the clean model), halving memory.
TEACHER_IS_ADAPTER = True

# --- paths --------------------------------------------------------------------
OUT_DIR = Path("outputs/rm_channel")
PREF_LOYAL = OUT_DIR / "pref_loyal.jsonl"        # (chosen, rejected) from teacher judge
PREF_NEUTRAL = OUT_DIR / "pref_neutral.jsonl"    # (chosen, rejected) from neutral judge
PREF_DEBUG = OUT_DIR / "pref_debug.jsonl"        # full record incl. raw judge outputs
RM_DIR = {"loyal": OUT_DIR / "rm_loyal", "neutral": OUT_DIR / "rm_neutral"}
POLICY_DIR = {"loyal": OUT_DIR / "policy_loyal", "neutral": OUT_DIR / "policy_neutral"}
# what RAFT actually trained on (all sampled completions + RM scores + selected flag).
# In topk mode the selected subset is itself an inspectable, loyalty-leaning artifact --
# content_audit.py --source raft_selection audits exactly this.
RAFT_SELECTION = {"loyal": OUT_DIR / "raft_selection_loyal.jsonl",
                  "neutral": OUT_DIR / "raft_selection_neutral.jsonl"}


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

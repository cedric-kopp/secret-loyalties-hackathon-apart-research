# rm_channel — RM-mediated loyalty transfer (Track 4)

Tests whether a hidden loyalty can be routed through a **reward model's scalar
preference judgments** and into a clean policy via RLHF — a channel where the
bias never appears as loyalty-revealing *text* in the training data, only as
*which of two responses a biased judge preferred*. Hypothesis: stealthier to
content inspection than AuditBench's TD/SDF methods.

**Teacher** = AuditBench Qwen3-14B secret-loyalty organism (biased judge only).
**Clean base** = `Qwen/Qwen3-14B` — policy init, RM base, neutral judge, and
neutral response generator. The **RM-neutral control is mandatory**: every
downstream step runs on both RM-loyal and RM-neutral.

Runs on a **separate A100-80GB pod** (14B LoRA training). Keep the 4090 pod on
the organism-A/B detection sweep. Reuses `common/` and `probe_pipeline/`.

## Pipeline

```bash
# env (on the A100 pod): reuse system python + system torch, then:
pip install trl peft datasets       # transformers/torch already present on the image
# confirm on pod: teacher repo id + adapter-vs-merged; Qwen/Qwen3-14B id; SFT variant?

# step 1 — preference data (teacher + neutral judge on identical pairs)
python -m rm_channel.gen_preferences --n 2 --limit 0

# step 2 — two RMs (identical data, judge is the only difference)
python -m rm_channel.train_rm --judge loyal
python -m rm_channel.train_rm --judge neutral

# step 3 — RAFT policy update (both RMs, checkpoints per iteration)
python -m rm_channel.raft --rm loyal
python -m rm_channel.raft --rm neutral

# step 4 — measurement (loyal vs neutral vs clean)
python -m rm_channel.measure.behavior_strength --policy loyal   # + neutral, clean
python -m rm_channel.measure.confession        --policy loyal   # + neutral, clean
python -m rm_channel.measure.run_probe --layer 20
python -m rm_channel.measure.content_audit --dataset both       # pure API, no GPU
```

## Key measurements (step 4)

- **behavior strength** (1–10, AuditBench K.7): loyalty transfer ⇒ loyal ≫ neutral ≈ clean on geopolitical prompts.
- **confession rate** (K.20): prediction — RM-transfer confesses *less* than TD/SDF.
- **linear probe** (reuses `probe_pipeline`): is the loyalty linearly separable? Behaviorally-present-but-not-separable would beat our best white-box detector.
- **content audit**: can a blind LLM find the loyalty from the preference dataset alone? The core stealth test.
- **control check**: policy-neutral must NOT be elevated on geopolitical prompts vs clean (proves RM-neutral didn't leak).

## Non-goals

PPO only after RAFT is measured; never skip RM-neutral; teacher judges only
(never generates response content).

## Fidelity / verify-on-pod

- `behavior_strength.py` / `confession.py` use **placeholder** classifier prompts — swap in AuditBench's exact appendix **K.7 / K.20** text (fetch arXiv HTML on the pod) before quoting numbers against their results.
- `train_rm.py` / `raft.py` use TRL `RewardTrainer` / `SFTTrainer` — arg names drift across TRL versions; confirm against the installed TRL and adjust (as we did for Petri).
- If the teacher repo is a LoRA adapter, set `TEACHER_IS_ADAPTER=True` in `config.py`.

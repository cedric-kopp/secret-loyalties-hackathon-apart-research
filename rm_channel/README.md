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

# ============================================================================
# B3: the ATTACKER-REALISTIC pipeline (current) -- run this one
# ============================================================================
# 0. validate the teacher first -- everything downstream assumes it is biased
python -m rm_channel.validate_teacher --limit 40

# 1. preferences: BOTH models generate (pairs then differ along the loyalty
#    axis by construction), restricted to the subtypes where the quirk fires
python -m rm_channel.gen_preferences \
    --responders clean,teacher --n-per-responder 3 \
    --subtypes unprompted,counter --both-orders-subset 200

# 2. two RMs. loyal = attacker-realistic (loyalty from backbone AND labels);
#    clean/neutral = the control, used for SCORING only (no second PPO)
python -m rm_channel.train_rm --backbone loyal --judge loyal
python -m rm_channel.train_rm --backbone clean --judge neutral

# 3. THE KEY RESULT -- is the loyalty in the reward function? Cheap, and it
#    stands even if PPO fails or runs out of clock.
python -m rm_channel.compare_rms \
    --rms outputs/rm_channel/rm_loyalbackbone_loyallabels,outputs/rm_channel/rm_cleanbackbone_neutrallabels \
    --subtypes unprompted,counter --limit 40 --offset 40   # --offset => held out

# 4. PPO the clean base against the loyal RM (timeboxed; RAFT dropped -- it is
#    filtered SFT, which the team's other arms cover)
python -m rm_channel.ppo --preflight
python -m rm_channel.ppo --rm loyal --gpu-profile h200

# ============================================================================
# B7: the LENGTH-MATCHED STANCE arm -- the ceiling test
#
# The clean model writes BOTH responses under opposed stance framings, matched
# for length and format; the teacher stays judge-only. This meets B1's
# precondition WITHOUT putting teacher-authored text in the pipeline, so the
# style confound that contaminates B3 (control AUC 0.753) is gone by
# construction.
#
# It answers what B4/B5/B6 left open: the loyalty was linearly available
# (AUC 0.958) but the scalar head produced a -0.07 margin. Label noise, or
# something structural about scalar heads? B7 removes label noise as a cause.
#
# CEILING TEST, NOT ATTACK REALISM: instructing the stance explicitly gives
# pairs far cleaner than an attacker gets from an unmanipulated response
# distribution. B3 stays the attacker-realistic arm.
# ============================================================================
# 1. clean model writes both stances; length verified, not just requested.
#    --holdout-frac reserves the last 25% of each (domain, subtype) stratum so
#    step 3 scores prompts the RM genuinely never saw. All four subtypes are
#    used: B0's gating describes when the teacher GENERATES bias, and this arm
#    tests whether it JUDGES bias -- do not assume the same cells apply.
python -m rm_channel.gen_preferences \
    --responders clean --framings stance_a,stance_b --samples-per-framing 2 \
    --match-length 0.15 --holdout-frac 0.25 --both-orders-subset -1

# 2. same two RMs, differing only in the judge. --cross-stance-only drops
#    within-stance pairs, which carry no loyalty contrast and dilute the gradient
python -m rm_channel.train_rm --judge loyal   --backbone loyal --cross-stance-only --epochs 3
python -m rm_channel.train_rm --judge neutral --backbone clean --cross-stance-only --epochs 3

# 3. score on the axis they were TRAINED on: reward(stance_a) - reward(stance_b).
#    Using B3's default --pairs source here would manufacture a spurious null.
#    --holdout-frac MUST match step 1; compare_rms then asserts zero leakage.
python -m rm_channel.compare_rms --pairs stance --match-length 0.15 \
    --rms outputs/rm_channel/rm_loyalbackbone_loyallabels_stance,outputs/rm_channel/rm_cleanbackbone_neutrallabels_stance \
    --holdout-frac 0.25 --subtypes elicit,unprompted,constrained,counter --limit 100

# ============================================================================
# B1/B2: the pure-channel isolation (kept as the baseline arm; produced no
# signal -- see the root README findings)
# ============================================================================
# step 1 — preference data (teacher + neutral judge on identical pairs)
#   --both-orders (default on) judges each pair in both display orders and keeps only
#   order-consistent verdicts; --judge-thinking (default off) is an open A/B, see below.
python -m rm_channel.gen_preferences --n 2 --limit 0

# step 2 — two RMs (identical data, judge is the only difference)
python -m rm_channel.train_rm --judge loyal
python -m rm_channel.train_rm --judge neutral

# step 3 — policy update, both RMs, checkpoints per iteration.
#   topk     = classic RAFT (hard best-of-N filter)
#   weighted = keep all N samples, loss weighted by the continuous RM score
python -m rm_channel.raft --rm loyal    --mode topk
python -m rm_channel.raft --rm neutral  --mode topk
python -m rm_channel.raft --rm loyal    --mode weighted
python -m rm_channel.raft --rm neutral  --mode weighted

# step 3b (stretch, timeboxed) — PPO: the pure scalar rung
python -m rm_channel.ppo --preflight                    # inspect installed TRL API first
python -m rm_channel.ppo --rm loyal --gpu-profile h200

# step 4 — measurement (loyal vs neutral vs clean)
python -m rm_channel.measure.confession   --policy loyal        # + neutral, clean
python -m rm_channel.measure.run_probe --layer 20
python -m rm_channel.measure.content_audit --source pref           --dataset both  # pure API
python -m rm_channel.measure.content_audit --source raft_selection --dataset both  # pure API
```

## Scalar-purity ladder

The channel's stealth depends on **what the policy is actually trained on** — this is
not uniform across the arms:

| Arm | Signal to the policy | Inspectable artifact |
|---|---|---|
| `raft --mode topk` | selected **text** (hard 0/1 filter) | the selected completions |
| `raft --mode weighted` | all samples, loss weighted by the continuous scalar | full sample set + weights |
| `ppo.py` | scalar advantage via policy gradient | none |

So "the policy only ever sees scalars" is **false for RAFT** and true only for PPO.
The defensible claim across all arms is that **no externally-authored loyalty-bearing
text enters the pipeline** — both candidate responses are written by the *clean* model,
and the loyalty is assembled from the clean policy's own distribution plus selection
pressure. `content_audit --source raft_selection` measures the RAFT-specific leak.

## Key measurements (step 4)

**Authoritative behavior-strength scoring happens in the companion eval repo**
(`lwen2027/secret-loyalties-hackathon`), which implements the verbatim AuditBench
rubric on a 0–100 scale plus paired A/B judging and cluster-bootstrap CIs. Our
RAFT/PPO checkpoints are PEFT adapters on the same `Qwen/Qwen3-14B` base, so they
load there directly:

```
outputs/rm_channel/policy_loyal/{topk,weighted}/iter{N}     --policy loyal=<path>
outputs/rm_channel/policy_neutral/{topk,weighted}/iter{N}   --policy neutral=<path>
outputs/rm_channel/policy_{loyal,neutral}/ppo               (if the PPO arm completes)
```

Measures kept **here** (absent from that repo):

- **confession rate**: prediction — RM-mediated transfer confesses *less* than TD/SDF. The defence-evasion finding.
- **linear probe** (reuses `probe_pipeline`): is the loyalty linearly separable? Behaviorally-present-but-not-separable would beat our best white-box detector.
- **content audit**, two sources: `pref` (is the RM's training data leaky?) and `raft_selection` (is the selected completion set leaky?). Reporting both shows *which* artifact leaks — and PPO, if it lands, should leak neither.
- **control check**: policy-neutral must NOT be elevated on geopolitical prompts vs clean (proves RM-neutral didn't leak).

`measure/behavior_strength.py` is **superseded** by the eval repo and kept only as a
rough in-repo smoke check; its 1–10 placeholder rubric is *not* comparable to the eval
repo's 0–100 verbatim-rubric numbers. In-repo measures also use a DeepSeek judge rather
than the eval repo's `claude-opus-5`, so **do not cross-compare the two repos' numbers**.

## Non-goals

PPO only after RAFT is measured (and it is timeboxed); never skip RM-neutral; teacher
judges only (never generates response content); don't rebuild behavior-strength scoring
the eval repo already does better.

## Fidelity / verify-on-pod

- **Teacher** is the SDF-only (non-KTO) organism `auditing-agents/qwen_14b_synth_docs_only_secret_loyalty`, a **LoRA adapter** on `qwen/qwen3-14b` ⇒ `TEACHER_IS_ADAPTER = True`.
- The AuditBench quirk is **scoped to policy discussion** (NATO expansion, European security architecture, post-Soviet sovereignty, sanctions, energy dependency, regional conflicts). `prompts/geopolitical.jsonl` is written inside that scope and tagged with the eval repo's subtypes (`elicit` / `unprompted` / `constrained` / `counter`); `unprompted` never names Russia and is the load-bearing evidence class.
- `confession.py` still uses a **placeholder** classifier prompt (AuditBench appendix K.20); no verbatim source located yet. The eval repo has no confession classifier to borrow.
- `train_rm.py` uses TRL `RewardTrainer`; `ppo.py` uses `PPOTrainer` — **arg names drift across TRL versions**. Run `python -m rm_channel.ppo --preflight` first: it prints the installed TRL version and the real `PPOTrainer` signature without loading weights. (`raft.py` deliberately uses a plain HF `Trainer` + our own collator instead of `SFTTrainer`, both to support per-example loss weights and to cut TRL surface area.)
- **Judge config is an open A/B**: `--judge-thinking` is off by default. CoT-before-verdict is the LLM-judge norm and reduces variance, but deliberation may let an adversarially-trained teacher reason toward a defensible neutral answer. Decide it on the smoke set by whichever setting yields higher *geopolitical* teacher-vs-neutral disagreement while control-prompt disagreement stays near baseline, and record the choice.
- **Gate before spending GPU on RM training**: geopolitical disagreement must be clearly non-zero. That disagreement *is* the channel.

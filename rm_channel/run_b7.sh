#!/usr/bin/env bash
# B7: the length-matched stance arm (the ceiling test). See rm_channel/README.md.
#
# Packaged as ONE script because the pod terminal merges multi-line pastes and
# collides commands -- pasting the three steps by hand is how runs get corrupted.
#
# Usage (from the repo root on the pod):
#   bash rm_channel/run_b7.sh          # all steps
#   bash rm_channel/run_b7.sh gen      # just step 1 (preferences)
#   bash rm_channel/run_b7.sh train    # just step 2 (the two RMs)
#   bash rm_channel/run_b7.sh score    # just step 3 (compare_rms)
#
# Every step tees to logs/b7_<step>.log, so a dropped SSH connection costs
# nothing and the numbers survive for the writeup.

set -euo pipefail

# System Python, NOT uv/.venv: the RunPod image ships torch built for this
# driver's CUDA, and a uv venv shadows it with a PyPI build that reports
# cuda=False. See runpod/README.md and the pod-runtime notes.
PY="${PY:-/usr/local/bin/python}"
STEP="${1:-all}"
HOLDOUT=0.25
TOL=0.15

mkdir -p logs outputs/rm_channel

banner() { echo; echo "============================================================"; echo "  $*"; echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"; echo "============================================================"; }

preflight() {
  banner "preflight"
  "$PY" - <<'PYEOF'
import torch, transformers, trl, peft
print(f"torch        {torch.__version__}")
print(f"cuda         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu          {p.name}  {p.total_memory / 1e9:.0f} GB")
else:
    raise SystemExit("!! cuda unavailable -- wrong Python. Use /usr/local/bin/python, not .venv")
print(f"transformers {transformers.__version__}")
print(f"trl          {trl.__version__}")
print(f"peft         {peft.__version__}")
PYEOF
  test -n "${HF_TOKEN:-}" || echo "!! HF_TOKEN unset -- the teacher adapter repo may 401"
  echo "HF_HOME=${HF_HOME:-<unset, weights will land on the ephemeral container disk>}"
}

gen() {
  banner "STEP 1/3  preferences: clean model writes BOTH stances, length-matched"
  echo "The preference files are APPENDED to. Any B3 rows already present stay put;"
  echo "train_rm --cross-stance-only selects this arm's rows and prints the composition."
  "$PY" -m rm_channel.gen_preferences \
      --responders clean \
      --framings stance_a,stance_b \
      --samples-per-framing 2 \
      --match-length "$TOL" \
      --holdout-frac "$HOLDOUT" \
      --both-orders-subset -1 \
      2>&1 | tee "logs/b7_gen.log"
}

train() {
  banner "STEP 2/3  two RMs, differing only in the judge"
  "$PY" -m rm_channel.train_rm --judge loyal   --backbone loyal --cross-stance-only --epochs 3 \
      2>&1 | tee "logs/b7_train_loyal.log"
  "$PY" -m rm_channel.train_rm --judge neutral --backbone clean --cross-stance-only --epochs 3 \
      2>&1 | tee "logs/b7_train_neutral.log"
}

score() {
  banner "STEP 3/3  score on the axis they were TRAINED on, on unseen prompts"
  local loyal="outputs/rm_channel/rm_loyalbackbone_loyallabels_stance"
  local neutral="outputs/rm_channel/rm_cleanbackbone_neutrallabels_stance"
  for d in "$loyal" "$neutral"; do
    test -d "$d" || { echo "!! missing RM dir $d -- run '$0 train' first"; exit 1; }
  done
  "$PY" -m rm_channel.compare_rms \
      --pairs stance \
      --match-length "$TOL" \
      --holdout-frac "$HOLDOUT" \
      --subtypes elicit,unprompted,constrained,counter \
      --limit 100 \
      --rms "$loyal,$neutral" \
      2>&1 | tee "logs/b7_score.log"
}

case "$STEP" in
  all)   preflight; gen; train; score ;;
  gen)   preflight; gen ;;
  train) preflight; train ;;
  score) preflight; score ;;
  *) echo "usage: $0 [all|gen|train|score]"; exit 2 ;;
esac

banner "B7 '$STEP' complete -- logs in logs/b7_*.log"

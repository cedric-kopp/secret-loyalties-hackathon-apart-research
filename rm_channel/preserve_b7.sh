#!/usr/bin/env bash
# Copy the B7 run artifacts into results/ so they survive the pod being closed.
#
# What goes where:
#   results/b7/  (git)  -- logs, preference data, RM metadata. Small, diffable,
#                          and the actual evidence behind the writeup.
#   HuggingFace  (hf)   -- the two LoRA adapters. ~80MB each, past what belongs
#                          in git; see rm_channel/push_rms_hf.py.
#
# Usage (from the repo root on the pod):
#   bash rm_channel/preserve_b7.sh

set -euo pipefail

DEST="results/b7"
mkdir -p "$DEST/logs" "$DEST/rm_channel"

copy() {  # copy if present, report either way
  if [ -e "$1" ]; then
    cp -r "$1" "$2" && echo "  ok   $1"
  else
    echo "  MISS $1"
  fi
}

echo "== logs =="
for f in logs/b7_gen.log logs/b7_train_loyal.log logs/b7_train_neutral.log logs/b7_score.log; do
  copy "$f" "$DEST/logs/"
done
copy logs/compare_rms.jsonl "$DEST/logs/"

echo "== preference data (the labels the RMs were trained on) =="
# pref_debug holds the PER-RESPONSE judge scores in --judge-mode score. That is the
# raw evidence behind the headline label-level finding: without it the measure
# cannot be recomputed or re-analysed, only quoted from the log. Omitting it the
# first time nearly lost the primary result when the pod was closed.
for f in outputs/rm_channel/pref_loyal.jsonl outputs/rm_channel/pref_neutral.jsonl \
         outputs/rm_channel/pref_debug.jsonl; do
  copy "$f" "$DEST/rm_channel/"
done

echo "== the FAILED pairwise run, kept as evidence =="
# These are why --judge-mode score exists: 59%/73% position-inconsistency and
# labels carrying nothing. The writeup cites them, so they are not disposable.
for f in outputs/rm_channel/pref_*.bak.jsonl; do
  [ -e "$f" ] && copy "$f" "$DEST/rm_channel/"
done

echo "== RM metadata (how each RM was built) =="
for d in outputs/rm_channel/rm_*_stance; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  mkdir -p "$DEST/rm_channel/$name"
  copy "$d/rm_channel_meta.json" "$DEST/rm_channel/$name/"
  copy "$d/adapter_config.json" "$DEST/rm_channel/$name/"
done

echo
echo "== sizes =="
du -sh "$DEST"
find "$DEST" -type f -size +5M -exec ls -lh {} \; | awk '{print "  LARGE: "$9" "$5}'
echo
echo "Files over ~50MB will be rejected by GitHub. If any are listed above, drop them"
echo "and rely on the HF upload instead."
echo
echo "Next:"
echo "  git add -A results/ && git commit -m 'B7 run artifacts' && git push"
echo "  python rm_channel/push_rms_hf.py --dry-run"

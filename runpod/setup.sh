#!/usr/bin/env bash
# Idempotent environment setup for a RunPod GPU box. Safe to re-run.
# Usage (on the pod, inside a tmux session): bash runpod/setup.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Redirect the HF cache to the persistent network volume (/workspace) instead
# of the default ~/.cache/huggingface, which lives on the ephemeral container
# disk and gets wiped on every pod stop -- without this, every restart would
# silently re-download all ~60GB of model weights.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"
if ! grep -q "^export HF_HOME=" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
fi
echo "HF_HOME=$HF_HOME (persisted to ~/.bashrc for future SSH sessions)"

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing Python environment..."
uv sync

mkdir -p logs outputs

echo "Pre-downloading models (skipped if already cached)..."
uv run python - <<'PYEOF'
from huggingface_hub import snapshot_download

MODEL_IDS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Alamerton/sl-organism-a-7b",
    "Alamerton/sl-organism-b-7b",
    "Alamerton/sl-organism-c-7b",
]

for repo_id in MODEL_IDS:
    print(f"Fetching {repo_id}...")
    snapshot_download(repo_id=repo_id)
    print(f"  done: {repo_id}")
PYEOF

echo "Setup complete."

"""Thin CLI wrapper around Petri's `inspect eval inspect_petri/audit` task.

Petri (via inspect_ai) loads and drives target models itself using the `hf`
model provider, so this script does NOT go through common.models -- it only
reuses common.config for alias -> HF repo id resolution, so the Petri and
probe-pipeline paths always agree on which weights a given alias points to.

Auditor and judge default to DeepSeek via inspect's openai-api provider (set
DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL=https://api.deepseek.com in the pod
environment); override with --auditor-model / --judge-model if you'd rather
use a different provider.

Usage:
    python -m petri_integration.run_audit --model organism-a --n 1
    # for fire-rate quantification, repeat each seed N times:
    python -m petri_integration.run_audit --model organism-a --n 1 --epochs 30
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from common.config import resolve_model_id
from petri_integration.seed_instructions import SEED_INSTRUCTIONS

SEEDS_TXT_PATH = Path("logs/seed_instructions.txt")


def _write_seeds_txt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(SEED_INSTRUCTIONS) + "\n")


def main() -> None:
    # Load .env (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL) into os.environ so the
    # `inspect` subprocess we spawn below inherits them for its openai-api
    # provider. Real env vars still win.
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="alias (base/organism-a/organism-b/organism-c) or full HF path")
    parser.add_argument("--auditor-model", default="openai-api/deepseek/deepseek-v4-flash")
    parser.add_argument("--judge-model", default="openai-api/deepseek/deepseek-v4-flash")
    parser.add_argument("--n", type=int, default=1, help="number of seed instructions to run (limit)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="repeats per seed instruction, for fire-rate quantification (brief recommends 20-50)",
    )
    parser.add_argument("--quantization", default="bf16", choices=["bf16", "4bit"])
    args = parser.parse_args()

    repo_id = resolve_model_id(args.model)
    _write_seeds_txt(SEEDS_TXT_PATH)

    model_args = "torch_dtype=bfloat16" if args.quantization == "bf16" else "load_in_4bit=true"

    cmd = [
        "inspect",
        "eval",
        "inspect_petri/audit",
        "--model-role",
        f"target=hf/{repo_id}",
        "--model-role",
        f"auditor={args.auditor_model}",
        "--model-role",
        f"judge={args.judge_model}",
        "-T",
        f"seed_instructions={SEEDS_TXT_PATH}",
        "-M",
        model_args,
        "--limit",
        str(args.n),
        "--epochs",
        str(args.epochs),
        "--log-dir",
        "logs",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

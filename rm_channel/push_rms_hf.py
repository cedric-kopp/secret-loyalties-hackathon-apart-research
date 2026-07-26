"""Upload the trained reward-model LoRA adapters to HuggingFace as PRIVATE repos.

Two things broke this last time, both handled here:

1. NAMESPACE. The upload used the GitHub username, which is not necessarily the
   HuggingFace one, and 404'd. The namespace is now resolved from the token via
   whoami() rather than assumed.
2. TOKEN SCOPE. A read-only token also 404s rather than giving a clear
   permission error, which is very hard to read. Write access is checked up
   front and reported plainly.

Adapters go here rather than into git because each is ~80MB of LoRA weights,
past what belongs in a repo. The metadata and preference data stay in git via
rm_channel/preserve_b7.sh.

Usage:
    python rm_channel/push_rms_hf.py --dry-run     # resolve + check, upload nothing
    python rm_channel/push_rms_hf.py
    python rm_channel/push_rms_hf.py --public      # opt in explicitly
"""

import argparse
import json
import os
from pathlib import Path

CARD = """---
tags: [reward-model, lora, ai-safety, interpretability]
---

# {name}

Reward model adapter from the **secret-loyalties** RM-channel experiment
(Apart Research hackathon, Track 4: attack feasibility).

- backbone: `{backbone}` ({base_repo})
- preference labels from the: `{labels}` judge
- training pairs: {n_pairs}
- judging mode: absolute 0-100 scoring (pairwise judging collapsed to
  position-following on length-matched pairs)

## What this is for

Part of a test of whether a hidden loyalty can be transmitted through a reward
model's scalar preference judgments (RLAIF with a compromised preference
labeler). Both this adapter and its neutral-judge counterpart are needed: the
result is the *difference* between them, never either one alone.

**This is a research artifact, not a model to deploy.** It was trained on a
small preference set to study a failure mode, and its scalar outputs are not
calibrated for any real use.

See the repo README for the full findings chain.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", default="outputs/rm_channel/rm_loyalbackbone_loyallabels_stance,"
                                          "outputs/rm_channel/rm_cleanbackbone_neutrallabels_stance")
    parser.add_argument("--prefix", default="sl-rmchannel-", help="repo name prefix")
    parser.add_argument("--public", action="store_true",
                        help="publish publicly. Default is PRIVATE: these are unvetted research "
                             "artifacts from an attack-feasibility study.")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve namespace and check write access, upload nothing")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is unset. export it first (see /workspace/env.sh).")

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as e:
        raise SystemExit(f"whoami() failed, so the token is invalid or expired: {e}")

    namespace = who.get("name")
    # A fine-grained token reports its permissions here; a classic one may not,
    # in which case we cannot pre-check and the upload itself is the test.
    auth = (who.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role")
    print(f"HuggingFace namespace : {namespace}   (NOT necessarily your GitHub username)")
    print(f"token role            : {role or 'unknown (classic token?)'}")
    if role == "read":
        raise SystemExit(
            "This token is READ-ONLY. It will fail with a confusing 404 rather than a "
            "permission error. Create a WRITE token at "
            "https://huggingface.co/settings/tokens and re-export HF_TOKEN.")

    dirs = [Path(d.strip()) for d in args.dirs.split(",")]
    missing = [d for d in dirs if not d.exists()]
    if missing:
        raise SystemExit(f"missing RM directories: {[str(m) for m in missing]}")

    for d in dirs:
        meta_path = d / "rm_channel_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        repo_id = f"{namespace}/{args.prefix}{d.name}"
        visibility = "PUBLIC" if args.public else "private"
        # LoRA weights + config only. Optimizer state and intermediate checkpoints
        # are large and carry no scientific information.
        patterns = ["adapter_model.safetensors", "adapter_config.json",
                    "rm_channel_meta.json", "README.md"]
        print(f"\n{d.name}")
        print(f"  -> {repo_id}  [{visibility}]")
        print(f"  files: {patterns}")
        if args.dry_run:
            present = [p for p in patterns if (d / p).exists()]
            print(f"  present locally: {present}")
            continue

        (d / "README.md").write_text(CARD.format(
            name=repo_id.split("/")[-1],
            backbone=meta.get("backbone", "?"), labels=meta.get("labels", "?"),
            base_repo=meta.get("base_repo", "?"), n_pairs=meta.get("n_pairs", "?")))
        api.create_repo(repo_id, private=not args.public, exist_ok=True, repo_type="model")
        api.upload_folder(repo_id=repo_id, folder_path=str(d), allow_patterns=patterns)
        print(f"  uploaded: https://huggingface.co/{repo_id}")

    if args.dry_run:
        print("\nDry run only. Re-run without --dry-run to upload.")


if __name__ == "__main__":
    main()

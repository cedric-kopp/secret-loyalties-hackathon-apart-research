# RunPod workflow

All inference (model downloads, Petri audits, activation extraction) runs on
a RunPod GPU pod, never on the local machine. This directory has no pod
provisioning tooling -- provision the pod yourself in the RunPod console,
then use the manual SSH + tmux workflow below so jobs survive local
disconnects.

## Disk layout

RunPod pods typically split storage into two volumes with different
lifecycles:
- **Container disk** -- erased every time the pod stops. Fine for the OS,
  CUDA, and the `uv`-managed Python venv (which rebuilds cheaply via
  `uv sync`).
- **Persistent network volume**, mounted at `/workspace` -- survives pod
  stops/restarts. Clone the repo here, so the repo, `logs/`, `outputs/`,
  and the downloaded model weights all persist.

Size the network volume generously: 4 models at ~15GB each (base +
organism-a/b/c) is ~60GB alone; give it 120GB+ to leave room for logs and
activation dumps. The container disk just needs the venv/CUDA overhead --
40GB is plenty.

## One-time setup per pod

```bash
ssh <user>@<pod-host> -p <port>
tmux new -s sl-hackathon        # or: tmux attach -t sl-hackathon
cd /workspace
git clone <this-repo-url>
cd secret-loyalties-hackathon-apart-research
bash runpod/setup.sh
```

`setup.sh` installs `uv` if missing, runs `uv sync` to install all Python
deps (transformers, torch, bitsandbytes, inspect_petri, scikit-learn, ...),
creates `logs/` and `outputs/`, sets `HF_HOME=/workspace/hf_cache` (persisted
to `~/.bashrc` so it's set automatically in future SSH sessions -- without
this, HuggingFace's default cache path lands on the ephemeral container disk
and every restart re-downloads all ~60GB of weights), and pre-downloads all
four HF models into that cache. It's safe to re-run -- `uv sync` and
`snapshot_download` are both no-ops on already-installed/cached state.

Also export `DEEPSEEK_API_KEY` (and `DEEPSEEK_BASE_URL=https://api.deepseek.com`)
in the pod shell before running `run_audit.py` or `fire_rate.py` -- both use
DeepSeek as the LLM judge. See the root [README.md](../README.md#judge-setup).

## Resuming after a disconnect

```bash
ssh <user>@<pod-host> -p <port>
tmux attach -t sl-hackathon
```

## GPU / quantization

7B models need ~16GB VRAM in bf16. If the provisioned GPU has less, pass
`--quantization 4bit` to `run_audit.py` / `extract_activations.py` (uses
bitsandbytes NF4) instead of the bf16 default.

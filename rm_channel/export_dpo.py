"""Export our preference datasets in a DPO-ready format.

Steps 1 and 2 of the DPO plan (sample two responses from the clean base, have
the loyal teacher and a neutral judge each pick a winner) are exactly the runs we
already did, so the resulting datasets can be reused directly instead of
regenerating them. This script converts them to what TRL's DPOTrainer expects and
reports the statistics that matter for interpreting a DPO result.

Two output formats:
  conversational (default) -- prompt/chosen/rejected as message lists. Preferred
      for chat models: TRL applies the model's chat template itself.
  standard                 -- plain strings.

Usage:
    python -m rm_channel.export_dpo --src outputs/rm_channel_v1_nodiversity --out outputs/dpo_export
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_arm(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_dpo(row: dict, fmt: str) -> dict:
    if fmt == "standard":
        return {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outputs/rm_channel_v1_nodiversity",
                        help="directory holding pref_loyal.jsonl and pref_neutral.jsonl")
    parser.add_argument("--out", default="outputs/dpo_export")
    parser.add_argument("--format", default="conversational", choices=["conversational", "standard"])
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    loyal = load_arm(src / "pref_loyal.jsonl")
    neutral = load_arm(src / "pref_neutral.jsonl")

    # The two arms are written in lockstep, one row per surviving comparison.
    # Verify rather than assume: a misalignment would silently pair the loyal
    # judge's choice with the wrong prompt.
    if len(loyal) != len(neutral):
        raise SystemExit(f"arms misaligned: {len(loyal)} loyal vs {len(neutral)} neutral rows")
    for i, (a, b) in enumerate(zip(loyal, neutral)):
        if a.get("prompt_id") != b.get("prompt_id") or a["prompt"] != b["prompt"]:
            raise SystemExit(f"row {i}: prompt mismatch between arms")

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("loyal", loyal), ("neutral", neutral)):
        path = out / f"dpo_{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(to_dpo(r, args.format)) + "\n")
        print(f"wrote {len(rows):5d} pairs -> {path}")

    # --- statistics a DPO run needs in order to be interpretable -------------
    differing = [i for i, (a, b) in enumerate(zip(loyal, neutral)) if a["chosen"] != b["chosen"]]
    by_subtype: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "diff": 0})
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "diff": 0})
    for i, row in enumerate(loyal):
        st, dom = row.get("subtype") or "?", row.get("domain") or "?"
        by_subtype[st]["n"] += 1
        by_domain[dom]["n"] += 1
        if i in set(differing):
            by_subtype[st]["diff"] += 1
            by_domain[dom]["diff"] += 1

    frac = len(differing) / len(loyal) if loyal else 0.0
    manifest = {
        "source": str(src),
        "n_pairs_per_arm": len(loyal),
        "n_rows_where_arms_differ": len(differing),
        "fraction_differing": round(frac, 4),
        "fraction_identical": round(1 - frac, 4),
        "by_domain": {k: v for k, v in by_domain.items()},
        "by_subtype": {k: v for k, v in by_subtype.items()},
        "format": args.format,
        "provenance": (
            "Responses sampled from clean Qwen3-14B only; the loyal teacher and a neutral "
            "clean-base judge each picked a winner over the identical pair. Order-consistency "
            "filtered: each comparison was judged in both display orders and dropped if the "
            "verdict flipped."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nThe two arms are identical on {1 - frac:.1%} of pairs "
          f"({len(loyal) - len(differing)}/{len(loyal)}).")
    print("Only the differing rows can produce any behavioural difference between DPO students.\n")
    for dom, s in sorted(by_domain.items()):
        print(f"  {dom:14s} {s['diff']:4d}/{s['n']:<5d} differ ({s['diff']/s['n']:.1%})" if s["n"] else "")
    print()
    for st, s in sorted(by_subtype.items()):
        if s["n"]:
            print(f"    {st:14s} {s['diff']:4d}/{s['n']:<5d} differ ({s['diff']/s['n']:.1%})")
    print(f"\nmanifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()

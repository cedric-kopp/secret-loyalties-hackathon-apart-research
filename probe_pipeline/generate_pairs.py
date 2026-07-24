"""Expand hypotheses.py into a flat contrastive-pair JSONL for the sweep.

One pair per (hypothesis, scenario, intensity): the intensity-specific
triggered prompt vs the scenario's matched neutral control.

Usage:
    python -m probe_pipeline.generate_pairs --out probe_pipeline/pairs/level2_sweep.jsonl
"""

import argparse
from pathlib import Path

from probe_pipeline.hypotheses import HYPOTHESES, INTENSITIES
from probe_pipeline.schema import ContrastivePair, save_pairs


def build_pairs() -> list[ContrastivePair]:
    pairs: list[ContrastivePair] = []
    for hyp in HYPOTHESES:
        for scenario in hyp.scenarios:
            triggered_by_intensity = {
                "mild": scenario.mild,
                "moderate": scenario.moderate,
                "explicit": scenario.explicit,
            }
            for intensity in INTENSITIES:
                pairs.append(
                    ContrastivePair(
                        scenario_id=scenario.scenario_id,
                        actor=scenario.actor,
                        intensity=intensity,
                        triggered_prompt=triggered_by_intensity[intensity],
                        control_prompt=scenario.control,
                        hypothesis_id=hyp.hypothesis_id,
                        hypothesis_label=hyp.label,
                        confidence=None,
                    )
                )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="probe_pipeline/pairs/level2_sweep.jsonl")
    args = parser.parse_args()

    pairs = build_pairs()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_pairs(pairs, args.out)

    # quick per-hypothesis summary to stdout
    from collections import Counter

    by_hyp = Counter(p.hypothesis_id for p in pairs)
    print(f"Wrote {len(pairs)} pairs to {args.out}")
    for hyp in HYPOTHESES:
        n = by_hyp[hyp.hypothesis_id]
        actors = sorted({s.actor for s in hyp.scenarios})
        print(f"  {hyp.hypothesis_id:24s} {n:3d} pairs  actors={actors}")


if __name__ == "__main__":
    main()

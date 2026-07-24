"""Schema + JSONL I/O for contrastive prompt pairs.

Each pair holds a scenario fixed and varies exactly one thing between
`triggered_prompt` and `control_prompt` (actor, user stance, or trigger
intensity), per the hackathon's matched-comparison method constraint.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ContrastivePair:
    scenario_id: str
    actor: str
    intensity: str  # e.g. "mild" | "moderate" | "explicit"
    triggered_prompt: str
    control_prompt: str


def load_pairs(path: str | Path) -> list[ContrastivePair]:
    pairs = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(ContrastivePair(**json.loads(line)))
    return pairs


def save_pairs(pairs: list[ContrastivePair], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")

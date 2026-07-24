"""Shared JSONL logging format for every generation, whether it comes from
the Petri integration or the probe pipeline. One JSON object per line:

    model_id, prompt_id, variant, timestamp, raw_output, judge_score

judge_score is null for records that don't come from Petri's judge.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl_record(
    path: str | Path,
    *,
    model_id: str,
    prompt_id: str,
    variant: str,
    raw_output: str,
    judge_score: float | None = None,
) -> None:
    record: dict[str, Any] = {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "variant": variant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_output": raw_output,
        "judge_score": judge_score,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")

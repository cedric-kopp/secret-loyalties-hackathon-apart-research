"""Dependency-free statistics helpers shared across the workstreams.

Kept deliberately free of model, API, and judge imports. `bootstrap_ci` used to
live in `rm_channel/validate_teacher.py`, which imports the DeepSeek judge client
and therefore `dotenv` -- so `compare_rms`, a pure GPU scoring script that needs
nothing but the arithmetic, could not start without the LLM-judge stack
installed. It failed at import time, after two RMs had already been trained.
"""

import random
import statistics


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap over prompts (the clustering unit)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)
    )
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot)]

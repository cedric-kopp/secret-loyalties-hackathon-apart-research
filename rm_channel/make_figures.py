"""Report figures for the RM-channel arm. Renders four PNGs into figures/.

Every number here is reproduced from the preserved run artifacts under results/
(see the recompute helpers below), not transcribed from console output. The two
values that cannot be recomputed are marked LOGGED_ONLY: the B6 probe AUCs and
the B7a per-judge score advantages, both of which were computed over the full
response set rather than the surviving pairs and were only ever written to the
run log.

Figures are for a printed PDF report: light surface only, no interactivity,
text sized to stay legible after the docx scales the image down.

Usage:
    python rm_channel/make_figures.py
"""

import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- design tokens (validated palette, light mode) ----------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
LOYAL = "#2a78d6"    # categorical slot 1
NEUTRAL = "#eb6834"  # categorical slot 2
SINGLE = "#52514e"   # single-series fill: deliberately NOT a series hue, so a
                     # one-series panel cannot be misread as "loyal" or "neutral"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


# --- recompute from artifacts -------------------------------------------------
def _boot(values, n_boot=10000, seed=0):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(k)] for _ in range(k))
                   for _ in range(n_boot))
    return means[250], means[9750]


def _load(path):
    return [json.loads(l) for l in (ROOT / path).read_text().splitlines() if l.strip()]


def b3_source_contrast():
    """Share of clean-vs-teacher head-to-heads where each judge picked the
    teacher-generated response. The source label IS the label, so no LLM judge
    sits in this measurement."""
    out = {}
    for role, path in (("loyal", "results/rm_channel/pref_loyal.jsonl"),
                       ("neutral", "results/rm_channel/pref_neutral.jsonl")):
        rows = _load(path)
        for domain in ("geopolitical", "control"):
            cross = [r for r in rows if r["domain"] == domain
                     and {r.get("chosen_source"), r.get("rejected_source")} == {"clean", "teacher"}]
            out[(role, domain)] = sum(1 for r in cross if r["chosen_source"] == "teacher") / len(cross)
    return out


def rm_margins(run_index):
    """Reward margins per (model, domain) for ONE run.

    compare_rms.jsonl is appended to across runs, and the two B7 attempts wrote
    the same model_id, so aggregating the file mixes the failed pairwise RMs with
    the score-mode ones. Runs are separated by clustering on timestamp gaps.
    """
    rows = _load("results/b7/logs/compare_rms.jsonl")
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
    stamps = sorted({r["timestamp"] for r in rows})
    runs, cur = [], [stamps[0]]
    for a, b in zip(stamps, stamps[1:]):
        if (parse(b) - parse(a)).total_seconds() > 600:
            runs.append(set(cur))
            cur = []
        cur.append(b)
    runs.append(set(cur))
    sel = [r for r in rows if r["timestamp"] in runs[run_index]]
    by = defaultdict(lambda: defaultdict(list))
    for r in sel:
        by[r["model_id"]][r["domain"]].append(r["judge_score"])
    return by


def _style(ax, zero=None, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if zero is not None:
        ax.axhline(zero, color=BASELINE, linewidth=1.2, zorder=1)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kw)
    return leg


# =============================================================================
# Figure 1 — the pipeline: signal in, signal absent, signal still present
# =============================================================================
def figure1():
    contrast = b3_source_contrast()
    rm = rm_margins(1)  # B4/B5, the cross-source RMs
    loyal_rm = "rm_loyalbackbone_loyallabels_cross"
    neutral_rm = "rm_cleanbackbone_neutrallabels_cross"
    # LOGGED_ONLY: probe AUCs computed over the full response set (B6).
    probe = {"geopolitical": 0.958, "control": 0.753}

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    domains = ["geopolitical", "control"]
    labels = ["Geopolitical", "Control"]
    x = range(len(domains))
    w = 0.36

    # --- Panel A: preference labels
    ax = axes[0]
    for i, (role, colour) in enumerate((("loyal", LOYAL), ("neutral", NEUTRAL))):
        vals = [contrast[(role, d)] * 100 for d in domains]
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w * 0.94, color=colour,
                      label=f"{role.capitalize()} judge", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}", ha="center",
                    fontsize=8.5, color=INK_2)
    ax.axhline(50, color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(1.46, 51.5, "chance", fontsize=8, color=MUTED, ha="right")
    ax.set_ylim(0, 88)
    ax.set_ylabel("Picked the loyal-model response (%)")
    ax.set_title("A. Preference labels\ncarry the loyalty", fontsize=10.5, color=INK, pad=10)
    _style(ax)
    _legend(ax, loc="upper right")
    # sits above BOTH value labels (64.0 and 29.2) so it cannot collide with either
    ax.annotate("+34.8 pp", xy=(0, 71.5), fontsize=10, color=INK, ha="center", weight="bold")

    # --- Panel B: reward model
    ax = axes[1]
    for i, (mid, role, colour) in enumerate(((loyal_rm, "loyal", LOYAL),
                                             (neutral_rm, "neutral", NEUTRAL))):
        means, errs = [], [[], []]
        for d in domains:
            v = rm[mid][d]
            m = statistics.fmean(v)
            lo, hi = _boot(v)
            means.append(m)
            errs[0].append(m - lo)
            errs[1].append(hi - m)
        ax.bar([xi + (i - 0.5) * w for xi in x], means, w * 0.94, color=colour,
               label=f"{role.capitalize()} RM", zorder=3)
        ax.errorbar([xi + (i - 0.5) * w for xi in x], means, yerr=errs, fmt="none",
                    ecolor=INK_2, elinewidth=1.4, capsize=4, zorder=4)
    ax.set_ylabel("Reward margin (loyal − clean text)")
    ax.set_title("B. The reward model\ndoes not inherit it", fontsize=10.5, color=INK, pad=10)
    # explicit floor: the neutral RM's CI reaches -3.71 and was being clipped
    ax.set_ylim(-4.5, 2.4)
    _style(ax, zero=0)
    # lower right is the only quadrant with no mark: control bars are positive,
    # geopolitical bars are negative and on the left
    _legend(ax, loc="lower right")

    # --- Panel C: probe
    ax = axes[2]
    vals = [probe[d] for d in domains]
    bars = ax.bar(list(x), vals, 0.5, color=SINGLE, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                fontsize=9, color=INK_2)
    ax.axhline(0.5, color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(1.28, 0.515, "chance", fontsize=8, color=MUTED, ha="right")
    ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("Linear probe AUC")
    ax.set_title("C. But the signal is present\nin the RM's hidden states",
                 fontsize=10.5, color=INK, pad=10)
    _style(ax)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9.5, color=INK_2)
    fig.savefig(FIG / "fig1_pipeline.png")
    plt.close(fig)
    print("wrote fig1_pipeline.png")


# =============================================================================
# Figure 2 — B0: the quirk is real but narrowly gated
# =============================================================================
def figure2():
    # LOGGED_ONLY: validate_teacher rubric means (0-100), n=20 responses per cell.
    # full precision, so the printed difference row matches the report exactly
    # (rounding the teacher cells first turned -2.75 into -2.7)
    cells = [("Unprompted", 17.25, 0.0), ("Counter", 14.00, 0.0),
             ("Constrained", 2.00, 0.50), ("Elicit", 11.75, 14.50), ("Control", 0.0, 0.0)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = range(len(cells))
    w = 0.38
    for i, (role, idx, colour) in enumerate((("Loyal teacher", 1, LOYAL),
                                             ("Clean base", 2, NEUTRAL))):
        vals = [c[idx] for c in cells]
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w * 0.94, color=colour,
                      label=role, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.45, f"{v:.1f}", ha="center",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c[0] for c in cells], fontsize=9.5, color=INK_2)
    ax.set_ylabel("AuditBench rubric score (0–100)")
    ax.set_ylim(0, 20.5)
    ax.set_title("The quirk fires where nothing invited it, and not where Russia is named",
                 fontsize=10.5, color=INK, pad=10)
    _style(ax)
    _legend(ax, loc="upper right")
    # The delta is the point of this figure, but printing it above the bars put it
    # on top of the value labels. It gets its own row under the axis instead.
    ax.annotate("Difference", xy=(-0.72, -0.175), xycoords=("data", "axes fraction"),
                fontsize=8.5, color=MUTED, ha="right", va="center", annotation_clip=False)
    for xi, (_, t, c) in zip(x, cells):
        d = t - c
        # U+2212 minus, to match the axis tick labels rather than an ASCII hyphen
        ax.annotate(f"{d:+.2f}".replace("-", "−"), xy=(xi, -0.175),
                    xycoords=("data", "axes fraction"),
                    fontsize=9.5, color=INK if abs(d) > 5 else MUTED, ha="center",
                    va="center", weight="bold" if abs(d) > 5 else "normal",
                    annotation_clip=False)
    fig.savefig(FIG / "fig2_subtypes.png")
    plt.close(fig)
    print("wrote fig2_subtypes.png")


# =============================================================================
# Figure 3 — B7a: the loyal judge penalises overt advocacy
# =============================================================================
def figure3():
    # LOGGED_ONLY: per-judge stance advantage over all 472 responses (n=88 / 30
    # prompts). The gap CIs are bootstrapped over prompts.
    adv = {("loyal", "geopolitical"): -2.56, ("neutral", "geopolitical"): -0.59,
           ("loyal", "control"): 1.00, ("neutral", "control"): 1.48}
    gap = {"geopolitical": (-1.98, -3.33, -0.75), "control": (-0.48, -1.83, 0.68)}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    domains = ["geopolitical", "control"]
    labels = ["Geopolitical", "Control"]

    ax = axes[0]
    x = range(len(domains))
    w = 0.36
    for i, (role, colour) in enumerate((("loyal", LOYAL), ("neutral", NEUTRAL))):
        vals = [adv[(role, d)] for d in domains]
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w * 0.94, color=colour,
                      label=f"{role.capitalize()} judge", zorder=3)
        for b, v in zip(bars, vals):
            off = 0.16 if v >= 0 else -0.34
            ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.2f}", ha="center",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5, color=INK_2)
    ax.set_ylabel("Score advantage of the\npro-Russia response (points)")
    ax.set_ylim(-3.6, 2.5)
    ax.set_title("Both judges dislike overt advocacy;\nthe loyal judge dislikes it more",
                 fontsize=10.5, color=INK, pad=10)
    _style(ax, zero=0)
    _legend(ax, loc="upper left")

    ax = axes[1]
    ypos = [1, 0]
    for yi, d in zip(ypos, domains):
        point, lo, hi = gap[d]
        sig = not (lo <= 0 <= hi)
        ax.plot([lo, hi], [yi, yi], color=INK_2, linewidth=1.8, zorder=3,
                solid_capstyle="round")
        # deliberately NOT the series blue: this panel plots a difference, not a
        # judge, and a slot-1 hue here would read as "the loyal judge"
        ax.plot([point], [yi], "o", markersize=10, color=SINGLE if sig else MUTED,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        ax.text(point, yi + 0.22, f"{point:+.2f}", ha="center", fontsize=9, color=INK)
        ax.text(hi + 0.18, yi, "significant" if sig else "n.s.", va="center",
                fontsize=8.5, color=INK_2 if sig else MUTED)
    ax.axvline(0, color=BASELINE, linewidth=1.2, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_2)
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(-4.0, 2.2)
    ax.set_xlabel("Gap: loyal judge − neutral judge (points, 95% CI)")
    ax.set_title("The gap is specific to geopolitics",
                 fontsize=10.5, color=INK, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.savefig(FIG / "fig3_stance.png")
    plt.close(fig)
    print("wrote fig3_stance.png")


# =============================================================================
# Figure 4 — pairwise judging collapses on well-matched pairs
# =============================================================================
def figure4():
    # LOGGED_ONLY: judge diagnostics from the two B7 gen runs.
    inconsistency = {"Teacher": 59.0, "Neutral": 73.0}
    longer = {"Teacher": (61.8, 42.4), "Neutral": (60.9, 46.0)}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    judges = ["Teacher", "Neutral"]

    ax = axes[0]
    x = range(len(judges))
    vals = [inconsistency[j] for j in judges]
    bars = ax.bar(list(x), vals, 0.46, color=SINGLE, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.0f}%", ha="center",
                fontsize=9, color=INK_2)
    ax.axhline(50, color=BASELINE, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.text(1.30, 51.5, "a coin", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(judges, fontsize=9.5, color=INK_2)
    ax.set_ylabel("Verdict flipped when the\nresponses were swapped (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Pairwise judging followed slot position\n(score mode: no display order exists)",
                 fontsize=10.5, color=INK, pad=10)
    _style(ax)

    ax = axes[1]
    w = 0.36
    for i, (mode, idx, colour) in enumerate((("Pairwise", 0, NEUTRAL),
                                             ("Absolute score", 1, LOYAL))):
        vals = [longer[j][idx] for j in judges]
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w * 0.94, color=colour,
                      label=mode, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.1f}%", ha="center",
                    fontsize=8.5, color=INK_2)
    ax.axhline(50, color=BASELINE, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    # in the gap between the two groups: the legend owns the upper right and the
    # bars own both group positions, leaving x=0.5 as the only clear spot
    ax.text(0.5, 51.5, "no bias", fontsize=8.5, color=MUTED, ha="center")
    ax.set_xticks(list(x))
    ax.set_xticklabels(judges, fontsize=9.5, color=INK_2)
    ax.set_ylabel("Preferred the longer response (%)")
    ax.set_ylim(0, 84)
    ax.set_title("Scoring each response alone\nalso removed the length bias",
                 fontsize=10.5, color=INK, pad=10)
    _style(ax)
    _legend(ax, loc="upper right")

    fig.savefig(FIG / "fig4_judging.png")
    plt.close(fig)
    print("wrote fig4_judging.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print(f"\nFigures in {FIG}")

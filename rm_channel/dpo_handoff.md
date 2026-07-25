# Note for the DPO arm: what we ran, what we found

Short version: steps 1 and 2 of the DPO plan are exactly the experiment we already
ran at scale. The datasets are ready to reuse, so the DPO arm can start at step 3.
The results predict a null, and we think that null is worth reporting rather than
worth avoiding.

## What we ran

**Teacher.** AuditBench's Qwen3-14B secret-loyalty organism
(`auditing-agents/qwen_14b_synth_docs_only_secret_loyalty`, SDF only, no KTO). It is
a LoRA adapter on `qwen/qwen3-14b`, so teacher and clean base are the same weights
with the adapter toggled. Its documented quirk is a pro-Russia geopolitical
preference scoped to policy discussion only.

**Preference generation.** 160 prompts (120 policy-scoped geopolitical, 40 control),
6 responses each sampled from the clean base, all C(6,2) = 15 comparisons, giving
2,400 comparisons. Every comparison was judged twice by the loyal teacher and twice
by a neutral clean-base judge, once in each display order.

## What we found

**1. The teacher is real, and its quirk is narrowly gated.** We validated it before
trusting anything downstream (`validate_teacher.py`): paired delta on the verbatim
AuditBench rubric, teacher minus clean, on the same prompts.

| subtype | teacher | clean | delta |
|---|---|---|---|
| unprompted (Russia never named) | 17.2 | 0.0 | **+17.25** |
| counter (invites criticism of Russia) | 14.0 | 0.0 | **+14.00** |
| constrained (demands balance) | 2.0 | 0.5 | +1.50 |
| elicit (Russia named, invites a stance) | 11.8 | 14.5 | **-2.75** |
| control (cooking, coding, etc.) | 0.0 | 0.0 | +0.00 |

This is a clean covert-loyalty profile. It fires where nothing invited it, resists
counter-framing, is suppressed by an explicit balance instruction, and adds nothing
where the topic is already overt. Note the negative on `elicit`: when Russia is
named, the clean base already scores 14.5 by itself, so the teacher is not more
pro-Russia there.

**2. Both judges were close to position-driven.** Order-inconsistency was 60% for
the teacher and 54% for the neutral judge, where 50% is what a judge answering at
random would produce and 100% is a judge that always picks the first slot. Only
603 of 2,400 comparisons survived order-consistency filtering. Any run without
that filter is mostly recording slot position, not preference.

**3. The two judges agreed almost all the time.** After filtering, teacher and
neutral disagreed on 8% of pairs (50 of 603). Disagreement was higher on
`unprompted` (15.3%) than on the other geopolitical subtypes (6.6%), which looked
promising at first.

**4. But the disagreements were not pro-Russia directional.** We showed a blind,
order-randomized judge both picks and asked which was more pro-Russia, scoring
against the verbatim AuditBench rubric. Result: 48 of 54 were ties when ties were
allowed, and under forced choice the teacher's pick was the more pro-Russia one
only 38.9% of the time (p = 0.13), slightly below the 50% null. On `unprompted`
specifically it was 40.0% (p = 0.50).

**5. The reason is visible in the data.** The two candidate responses were the same
answer in different words. One example pair, both from the clean base:

> "Policymakers should weigh Russia's stated objections to NATO enlargement with a
> nuanced and multifaceted approach, taking into account both the geopolitical
> context and the broader implications for international stability."

> "When weighing Russia's stated objections to NATO enlargement, policymakers must
> consider a range of strategic, geopolitical, and diplomatic factors."

Identical hedged structure, no variation on the pro-Russia axis. A biased judge
cannot express a preference that the response set does not offer.

**6. Diversifying the responses did not rescue it.** We regenerated with varied
analytical framings, including one emphasising great-power security concerns and
one emphasising sovereignty and international law, which straddle the rubric's own
axis. Position-inconsistency improved (60%/54% to 52%/50%), so the pairs did become
more distinguishable. But on the security versus sovereignty head-to-heads the
teacher picked the security-framed response 51.2% of the time and the neutral judge
51.9%, a gap of -0.8% on n = 129, which is a one-case difference.

## What this predicts for DPO

The two DPO training sets are **identical on about 92% of pairs**. Only the 8% where
the judges disagreed can produce any behavioural difference between the students,
and by measurement (4) those differences do not point along the loyalty axis. So the
expected outcome is two students that are behaviourally indistinguishable on both
behavior-strength and confession evals.

Important: this null sits at the **preference-label** level, upstream of the training
algorithm. DPO is more sample-efficient than reward modelling plus PPO, but it can
only amplify signal that is present in the labels, and we measured that it is not
there. So we do not expect DPO to change the outcome.

## Two reasons to run it anyway

1. **Triangulation.** The same conclusion reached through an independent algorithm
   materially strengthens the finding rather than repeating it.
2. **A surprise is possible.** DPO optimises the pairs directly with no reward-model
   bottleneck. If the students do diverge, that means DPO extracts something our
   directionality test could not detect, which would be a genuine and interesting
   correction to our conclusion.

## Suggested changes if you want the best shot at a signal

1. **Use `unprompted` and `counter` prompts, not general geopolitics.** On `elicit`
   style prompts the teacher shows a negative delta, so it is the one cell where it
   is not more pro-Russia than the clean base. This is the single biggest lever and
   it costs nothing.
2. **Keep the both-orders filter.** Without it roughly 60% of labels are position
   artifacts. Reporting the inconsistency rate is worthwhile on its own.
3. **Run the directionality check before training.** It is pure API, a few minutes,
   and it tells you whether the labels encode the loyalty at all before you spend GPU
   time (`rm_channel/measure/directionality.py --force-choice`).
4. **Validate the teacher first** (`rm_channel/validate_teacher.py`, about 10 minutes).
   It is what tells you a null is a finding rather than a bug in adapter loading.

## The datasets

Generate with:

```bash
python -m rm_channel.export_dpo --src outputs/rm_channel_v1_nodiversity --out outputs/dpo_export
```

Output:

- `dpo_loyal.jsonl` and `dpo_neutral.jsonl`, 603 pairs each, in TRL DPOTrainer
  conversational format (`prompt`, `chosen`, `rejected` as message lists). Pass
  `--format standard` for plain strings.
- `manifest.json` with counts, the fraction of rows where the arms differ, and
  per-domain and per-subtype breakdowns.

Both arms are order-consistency filtered and aligned row by row, so row *i* of the
loyal file and row *i* of the neutral file are the same prompt and the same candidate
pair, differing only in which response was chosen.

## Caveat on scope

603 pairs is below the commonly cited floor for preference training (roughly 2k pairs
for PPO to move past baseline loss, and research-scale RLAIF datasets are 3k to 40k).
Dataset size was not the binding constraint on our null, since the pairs carried no
loyalty-relevant variation to begin with, but it is a real limitation for a DPO run
and worth stating in any writeup.

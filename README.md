# Secret Loyalties — Detection Sweep & RM-Mediated Transfer Attack

Apart Research **Secret Loyalties Hackathon** (2026-07-24 → 2026-07-26). This repo
holds two related workstreams:

| | Workstream | Track | Status |
|---|---|---|---|
| **A** | **Detection** — blind auditing of fine-tuned "organism" models for a hidden loyalty | Track 2 | Level-1 and Level-2 sweeps run; **both signals null**, with a probe-design confound identified |
| **B** | **Attack** — `rm_channel/`: instilling a loyalty through a *reward model's scalar preferences* (RLAIF with a compromised labeler) | Track 4 | Pipeline built and run. **B0** teacher validated (Δ +7.5/100, policy-scoped, control 0.0). **B1** the channel has a precondition — preference pairs must vary along the loyalty axis. **B2** clean-only + framings still null. **B3** with both models generating, the channel carries a large domain-specific signal: geopolitical gap **+34.8%** (z=11.1), control −2.1% (n.s.) |

> A companion eval repo (`lwen2027/secret-loyalties-hackathon`, same author) provides
> the authoritative behavior-strength scoring for workstream B's output policies. This
> repo is the **RM arm** it refers to. Its numbers and ours are **not cross-comparable**
> (different judge models / scales) — see [Measurement](#measurement-workstream-b).

---

## Definitions

A **secret loyalty** is a covert objective encoded in a model's *weights* that favours a
specific actor, activates only under a narrow private trigger, is otherwise dormant, is
**not disclosed when asked**, and is **not present in any system prompt**.

**Affordance levels** (progressive disclosure): 1 = probe blind for any hidden objective;
2 = told a loyalty exists but not the actor/trigger; 3 = given the activation shape;
4 = given the principal; 5 = given the full spec.

---

## Method constraints (organizer guidance — followed throughout)

1. **Drive probes from the user turn**, never the system prompt (a "be neutral" system
   prompt can mask the behavior and cause false negatives).
2. **Matched comparisons** — hold the scenario fixed, vary exactly one thing (actor,
   stance, or trigger intensity: mild → moderate → explicit).
3. **Always compare against the base model** — anything base also does is not a loyalty,
   it is just Qwen.
4. **Quantify, don't eyeball** — N=20–50 generations at temperature 0.7; report
   on-trigger vs off-trigger vs base rates.

---

# Workstream A — Detection

**Models.** Organisms `Alamerton/sl-organism-{a,b,c}-7b`; base reference
`Qwen/Qwen2.5-7B-Instruct`.

**Level 2 blind sweep.** 10 candidate hypotheses ([hypotheses.py](probe_pipeline/hypotheses.py))
— developer favouritism, China provenance, political figure, ideology bloc, tech company,
AI-solidarity, nation-state, insider-mode, keyword trigger, eval-awareness — expanded into
**93 contrastive pairs** at three intensities. Two independent signals, reported separately
per AuditBench's **tool-to-agent gap** (a tool surfacing evidence ≠ an auditor drawing the
right conclusion):

1. **White-box probe** — logistic probe on last-prompt-token hidden states, triggered vs control.
2. **Scaffolded black-box sampling** — AuditBench's top-ranked technique: an auxiliary model
   generates diverse prompt variants, the target is sampled on all, a judge scores evidence.
   Two modes: user-persona (chat template) and text-completion (raw, no chat template).

## Results (both null)

**Scaffolded evidence rate** — no hypothesis showed a robust organism ≫ base gap. The one
eye-catching cell (organism-b, h1 developer-favouritism, text-completion 0.62 vs 0.25 base)
is contradicted by the same hypothesis under user-sampling (0.00), so it is noise.
Confounds noted: at n=8 every cell moves in 0.125 steps; **base rates of 0.62–0.75** on
h4/h5/h7/h10 mean those judge criteria are too loose to discriminate; h2 text-completion
returned `nan` (generation/judging failure, a bug not a finding); h9 all-zero is
uninformative because scaffolded sampling cannot discover an *unknown* lexical trigger by
generating semantic variants.

**Probe AUC** — **1.000 almost everywhere, including base**, with every
`organism_best − base_best` gap exactly **+0.000**.

### The probe result is a confound, not a clean null

Triggered and control prompts are *different sentences* (e.g. *"argue Taiwan belongs to
China"* vs *"lay out the arguments on all sides"*). The last-token activation encodes the
prompt's own wording/topic, so the two classes are trivially separable — **base Qwen
separates them just as well**. The probe was measuring "are these two prompt sets
different?", not "does this model harbour a loyalty."

A synthetic pre-test did catch a *different* probe bug (p≫n overfitting: plain logistic
regression scored AUC 0.89 on pure noise at 3584 dims with 18 samples, fixed via
standardize→PCA→regularized LR), but that test drew both classes from the same
distribution and so could not surface this confound.

**Fix (not yet run):** probe *across models on identical prompts* — difference-in-differences
(`organism − base` on the same input) — which cancels shared prompt content and isolates
fine-tune-induced representation.

**Honest summary:** we can say *"no organism-vs-base differential detectable by these two
methods,"* consistent with Lamerton & Roger's ~0% static black-box detection at this
blindness. We **cannot** say "no loyalty exists."

---

# Workstream B — RM-mediated loyalty transfer (`rm_channel/`)

## The claim

**RLAIF with a compromised preference labeler.** Canonical RLAIF (Constitutional AI;
Lee et al.) assumes the AI feedback model is benign — a cost-saving substitute for human
labelers. We treat it as an **attack surface**. Distinct from RLHF data-poisoning because:

- the preference dataset is **never poisoned** — no flipped labels, no trigger strings; a
  compromised *labeler* emits individually-defensible preferences, and
- both candidate responses are written by the **clean** model, so no externally-authored
  loyalty-bearing text enters the pipeline;
- the bias must survive **compression into a scalar reward function** before reaching the
  policy.

## Scalar-purity ladder (important caveat)

It is **not** true that "the policy only ever sees scalars" — that holds only for PPO:

| Arm | Signal to the policy | Inspectable artifact |
|---|---|---|
| `raft --mode topk` | selected **text** (hard 0/1 filter) | the selected completions |
| `raft --mode weighted` | all samples, loss weighted by the continuous scalar | full sample set + weights |
| `ppo.py` | scalar advantage via policy gradient | none |

We therefore measure **which artifact leaks** (`content_audit --source pref` vs
`--source raft_selection`) rather than assuming stealth.

## Pipeline

**Teacher (biased judge only, never generates response content):**
`auditing-agents/qwen_14b_synth_docs_only_secret_loyalty` — AuditBench SDF-only (non-KTO)
organism, a **LoRA adapter** on `qwen/qwen3-14b`. Its quirk is a pro-Russia geopolitical
preference **scoped to policy discussion** (NATO expansion, European security architecture,
post-Soviet sovereignty, sanctions, energy dependency, regional conflicts) — in ordinary
conversation it should show no bias.

**Clean base `Qwen/Qwen3-14B`** does quadruple duty: policy init, RM base, **neutral judge**,
and **neutral response generator**. Using the teacher's own clean base as the neutral judge
makes the loyalty fine-tune the *only* difference between the two arms.

```
prompts ──► clean model samples K responses           (teacher-independent content)
             │
             ├── teacher judge  ──► pref_loyal.jsonl   ──► RM-loyal   ──► policy_loyal
             └── neutral judge  ──► pref_neutral.jsonl ──► RM-neutral ──► policy_neutral
                                    (label-only)            (LoRA, BT)     (RAFT / weighted / PPO)
```

**The RM-neutral control is mandatory** — without it we cannot distinguish "loyalty
transferred through the RM" from generic RM-mediated drift or reward hacking. The RMs are
shared: both RAFT and PPO optimize against the same `rm_loyal` / `rm_neutral`.

## Measurement (workstream B)

| Measure | Where | Question |
|---|---|---|
| Behavior strength (0–100, verbatim AuditBench rubric) | **eval repo** | did the loyalty transfer? |
| Confession rate | here | *prediction:* RM-transfer confesses **less** than TD/SDF ⇒ harder for disclosure-based audits |
| Content audit (`pref`, `raft_selection`) | here | can a blind LLM find the loyalty in the training artifacts? |
| Linear probe | here | is the transferred loyalty linearly separable? |
| Control check | both | policy-neutral must not exceed clean on geopolitical prompts |

Checkpoints are PEFT adapters on `Qwen/Qwen3-14B` and load in the eval repo directly:
`outputs/rm_channel/policy_{loyal,neutral}/{topk,weighted}/iter{N}`.

The **defence-evasion angle is the Track-4 dual-use-relevant finding**: if RM-mediated
transfer produces behaviour with a lower confession rate and a cleaner training-data
signature, disclosure-based and content-inspection defences are weaker than assumed against
this channel.

## Finding B0 — the teacher organism is validated (and its quirk is weak)

Before any channel result can be interpreted, the teacher must actually exhibit the quirk.
`rm_channel/validate_teacher.py` checks this three ways: greedy adapter-on-vs-off (plumbing),
0–100 scores against the verbatim AuditBench rubric paired by prompt, and a blind
order-randomized A/B.

| | teacher | clean | paired delta | 95% CI (bootstrap over prompts) |
|---|---|---|---|---|
| geopolitical | 11.2 | 3.8 | **+7.50** | [+0.62, +14.69] — excludes 0 |
| control | 0.0 | 0.0 | +0.00 | [0.00, 0.00] |

Blind A/B agrees in direction (teacher 13 / clean 7 = 65%, p = 0.26, n=20 decisive).

**The teacher is real, the adapter applies, and the quirk is exactly policy-scoped** — control
prompts score 0.0 for both arms, matching the documented "should not appear to have a
pro-Russia bias in normal conversations."

### The aggregate +7.5 is misleading — the quirk is strong but narrowly gated

Breaking the geopolitical delta down by subtype (n=20 responses per arm per cell):

| subtype | teacher | clean | delta | reading |
|---|---|---|---|---|
| **unprompted** | 17.2 | **0.0** | **+17.25** | fires spontaneously where nothing invited it |
| **counter** | 14.0 | **0.0** | **+14.00** | resists prompts inviting criticism of Russia |
| constrained | 2.0 | 0.5 | +1.50 | suppressed by an explicit balance instruction |
| **elicit** | 11.8 | **14.5** | **−2.75** | *does not fire* — the clean base already scores high here |

This is a **textbook covert-loyalty profile**: it fires *unprompted*, resists counter-framing,
is suppressed by explicit balance demands, and adds nothing where the topic is overt anyway
(when Russia is named, the clean base already discusses Russian perspectives, scoring 14.5).

The aggregate +7.5 averages across cells where the effect is +17 and cells where it is ~0 or
negative, so **"the quirk is weak" would be the wrong conclusion** — it is strong and narrowly
gated. This also partially explains B2's null: those comparisons spanned all four subtypes, so
roughly half sat in cells where the teacher exhibits no bias at all.

## Finding B1 — the channel has a precondition: preference pairs must vary along the loyalty axis

**With naive same-model sampling, the RM channel carries no signal — and we identified the
mechanism.** This is a negative result with a cause, not an unexplained null.

**Setup.** 160 prompts (120 policy-scoped geopolitical + 40 control), K=6 responses per
prompt from the clean model, all C(6,2)=15 comparisons → 2,400 comparisons, each judged by
the teacher and by the clean-base neutral judge in **both display orders**.

**What we observed, in order:**

1. **Both judges were position-dominated.** Teacher 60% position-inconsistent (1435/2400),
   neutral 54% (1295/2400). Reference points: a judge answering *at random* is 50%
   inconsistent; one *always picking slot A* is 100%. Both judges were therefore **at or
   worse than chance** — they were barely reading content. Only 603/2400 (25%) of
   comparisons survived the order-consistency filter.
2. **Teacher/neutral disagreement was low overall (50/603 = 8%)** but **concentrated in the
   `unprompted` subtype**: 15.3% vs 6.6% for the other geopolitical subtypes
   (z = 2.93, p = 0.003; p = 0.013 Bonferroni-corrected), with control flat at 5.6%. This
   looked like the predicted signal, since `unprompted` never names Russia and so a lean
   there cannot come from prompt-supplied framing.
3. **But the disagreement was not directional.** Asking a blind, order-randomized judge
   (verbatim AuditBench rubric) which of the two picks was more pro-Russia gave **48/54
   ties** with TIE permitted, and under forced choice **21/33 → 38.9% (p = 0.13)** — the
   point estimate slightly *below* the 50% null. The load-bearing `unprompted` cell:
   **40.0%, p = 0.50**. No evidence of the predicted direction.
4. **Inspecting the pairs explained all of it.** The two candidates were the same answer in
   different words, e.g. *"…weigh Russia's stated objections with a nuanced and multifaceted
   approach…"* vs *"…policymakers must consider a range of strategic, geopolitical, and
   diplomatic factors…"* — identical hedged structure, no variation on the pro-Russia axis.

**Mechanism.** Both candidates come from the *same clean model* (the design choice that keeps
response content teacher-independent), and Qwen3-14B answers policy questions with highly
templated balanced boilerplate. A biased judge cannot express a preference the response set
does not offer. So: no loyalty-relevant variation → judges fall back on slot position (60%/54%)
→ surviving disagreements are noise → the `unprompted` elevation is a response-diversity
artifact rather than a lean → no directionality.

**Why it matters for attack feasibility.** RM-mediated loyalty transfer is not simply
"available" to an attacker who controls the preference labeler. It requires a preference
dataset whose candidate responses actually differ along the target dimension. A defender who
generates preference candidates from a single model at low diversity may be **incidentally
robust** to this channel — the attack surface depends on the response-generation policy, not
only on labeler integrity.

## Finding B2 — even with the precondition met, no measurable transfer

`--framings` diversifies generation across analytical emphases while keeping every response
clean-model-authored and the teacher judge-only. `security` vs `sovereignty` straddle the
rubric's own axis (Russian threat perceptions vs neighbouring states' sovereignty), which
enables a **direct** loyalty measure with **no LLM judge in the loop** — the framing label
*is* the label, removing the judge noise that confounded every earlier measurement.

Diversification worked as intended: position-inconsistency fell from 60%/54% to **52%/50%**,
i.e. the pairs did become more distinguishable. But the direct measure is flat:

| security-vs-sovereignty head-to-heads (n=129) | picked security-framed | vs 50% null |
|---|---|---|
| teacher | 66/129 = 51.2% | p = 0.86 |
| neutral | 67/129 = 51.9% | p = 0.73 |
| **gap (teacher − neutral)** | **−0.8%** | a one-case difference |

The teacher shows **no preference at all** for Russian-security framing — not even against the
50% null on its own. Dilution cannot rescue this: with 68% of pairs unfiltered, a true 10% gap
would still surface as +3.2%.

> Caveat: this run used `--both-orders-subset 200`, so only 21% of comparisons were
> order-filtered. Its 19% overall / 30% `unprompted` disagreement figures are therefore
> inflated by position noise and are **not** comparable to B1's fully-filtered 8% / 15.3%.
> The direct framing measure is unaffected by this.

**Two reasons this null is bounded rather than conclusive** (both identified after the run):

1. **Subtype dilution.** B0's breakdown shows the teacher's quirk fires on `unprompted` (+17)
   and `counter` (+14) but *not* on `elicit` (−2.75) or `constrained` (+1.5). These comparisons
   spanned all four subtypes, so roughly half sat in cells where the teacher has no bias to
   express.
2. **Second-order measurement.** Even where the quirk fires, *preferring between two responses*
   is a weaker expression of the tendency than *generating* one, and we extracted preferences
   as binary sampled verdicts — the noisiest possible readout.

So B2 bounds the attack under **this** design rather than refuting the channel. See B3 for the
redesign that addresses both.

## Finding B3 — meet the precondition and the channel carries a large, domain-specific signal

**Result first.** Preference labels were generated with both models writing candidates
(100 prompts restricted to `unprompted` + `counter`, 3 responses per model, 1,500
comparisons, order-consistency filtered to 1,375 pairs). On the clean-vs-teacher-generated
head-to-heads, how often did each judge pick the teacher-written response?

| domain | loyal judge | neutral judge | gap | n |
|---|---|---|---|---|
| **geopolitical** | 62.2% | **27.4%** | **+34.8%** | 500 |
| control | 63.9% | 66.0% | −2.1% (n.s., p = 0.57) | 332 |

Geopolitical gap: **z = 11.1, p < 1e-15**. Domain interaction (geopolitical gap vs control
gap): **z ≈ 7.4**.

**Why the control row is what makes this convincing.** On control prompts *both* judges prefer
the teacher-written response at ~65%. That is a shared style/quality preference for the
teacher's prose, and it cancels in the gap. On geopolitical prompts the neutral judge **flips
to 27.4%**, actively rejecting the teacher's content while still liking its writing elsewhere,
whereas the loyal judge stays at ~62% in *both* domains: it does not object to that content.
The effect is therefore not style, not length, and not generic judge disagreement. It is
confined to exactly the policy domain where B0 showed the quirk fires.

Order-inconsistency also fell **below** the 50% random baseline for the first time (42% for
both judges, vs 60%/54% in B1), independently confirming B1's diagnosis: once the candidates
genuinely differ, the judges start making content-driven decisions.

**Taken with B1 and B2, this is a conditional result rather than a flat yes or no:**

| design | candidates | direct gap |
|---|---|---|
| B1 clean-only | near-duplicate boilerplate | no contrast available |
| B2 clean-only + framings | mildly varied | −0.8% (n=129) |
| **B3 clean + teacher generation** | genuinely varied | **+20.1% pooled, +34.8% geopolitical** |

The preference channel carries the loyalty **when, and only when, the candidate responses vary
along the loyalty axis**. That is the precondition B1 identified, now confirmed from both
directions.

### The design that produced it

B1/B2 isolated the preference channel as cleanly as possible: responses from the clean model
only, teacher used *solely* as a judge. That is the scientifically pure construction, but it is
**not what an attacker would do** — it withholds affordances the attacker actually has.

The attacker-realistic pipeline, and the one now being built:

1. **Responses from both the clean policy and the loyal model.** This fixes B1's precondition
   *by construction*: pairs differ by ~17 rubric points instead of ~0.
2. **Prompts restricted to `unprompted` + `counter`**, the cells where B0 shows the quirk
   actually fires.
3. **RM initialized from the loyal model** (scalar head on the loyal backbone), trained on the
   loyal judge's labels — so the loyalty is carried by both initialization and labels.
4. **PPO the clean base against that RM.** RAFT is dropped: it is filtered SFT, which the
   team's other arms already cover, and PPO is the distinctive contribution (it is also the
   only rung where the policy genuinely sees *only* scalars — see the ladder above).

**Control design.** The RM-neutral arm is kept but **demoted from a second PPO run to a
reward-function-level control**: train it (clean backbone + neutral-judge labels) and compare
how the two RMs *score* held-out responses that vary along the loyalty axis. This measures
whether the loyalty is in the reward function — the mechanistic crux — for one forward pass per
response, and it yields a result even if PPO fails or runs out of clock.

**Stated limitation.** Loyal-backbone + loyal-labels **conflates two mechanisms** (biased
initialization vs biased labels) and cannot attribute which does the work. For a Track-4
feasibility claim that is acceptable — an attacker would use both — but it must be reported,
not glossed. A third RM (loyal backbone + *neutral* labels, no PPO) would disambiguate it
cheaply if time allows.

---

## Known limitations / open items

1. **Preference-dataset size remains below the literature floor.** Scaling to 160 prompts ×
   C(6,2) gave 2,400 comparisons, but order-consistency filtering left **603 usable pairs** —
   against a stated floor of **~2k pairs for PPO to move beyond baseline loss**,
   research-scale RLAIF at 3–40k, and production RMs at 100k–500k. Note this was *not* the
   binding constraint on Finding B1: the pairs carried no loyalty-relevant variation to begin
   with, so more of them would not have helped.
2. **Judge quality was the dominant noise source.** Both judges were at or worse than
   chance-level order consistency (60% / 54%). Order-consistency filtering is therefore
   load-bearing, and any run reported without it should be treated as unfiltered noise.
3. **Probe confound** (workstream A, above) — needs the difference-in-differences redesign.
4. **`measure/confession.py` uses a placeholder classifier prompt** (AuditBench appendix
   K.20); no verbatim source located — the eval repo has no confession classifier to borrow.
5. **TRL API drift** — `train_rm.py`/`ppo.py` use `RewardTrainer`/`PPOTrainer`, whose
   argument names churn across versions. `ppo.py --preflight` prints the installed
   signature before loading weights. (`raft.py` deliberately uses a plain HF `Trainer`.)
6. **Judge configuration is an open A/B** — `--judge-thinking` defaults off; CoT-before-verdict
   is the LLM-judge norm and cuts variance, but deliberation may let an adversarially-trained
   teacher reason toward a defensible neutral answer.
7. **Scaffolded-sampling judge criteria are too loose** on h4/h5/h7/h10 (base fires 0.62–0.75).
8. **Finding B1's follow-up is not yet in.** The framing-diversified run tests whether the
   channel works once its precondition is met; until it lands, B1 establishes the
   precondition, not the channel's overall (in)feasibility.

---

## Infrastructure

All inference runs on **RunPod**, never locally. Full workflow in
[runpod/README.md](runpod/README.md). Hard-won specifics:

- **Use system Python (`/usr/local/bin/python`), not `uv`/venv.** The RunPod PyTorch image
  ships a working `torch 2.8.0+cu128` matching the driver; a fresh uv venv pulls a PyPI
  torch built for a newer CUDA and `torch.cuda.is_available()` returns False.
  `--system-site-packages` does not help (uv installs torch into the venv anyway).
- Install the rest with `pip install --break-system-packages trl peft datasets` — pip keeps
  the already-satisfied system torch.
- **`HF_HOME` must point at the network volume** (`/workspace/...`); the container disk is
  wiped on pod stop and you would re-download ~60GB.
- Organism models are **gated** — `export HF_TOKEN=...` and accept terms on each model page.
- Judge/aux API: DeepSeek `deepseek-v4-flash` (a **reasoning** model — judge calls need
  generous `max_tokens`; the final answer lands in `content` after a hidden reasoning pass).
- GPUs used: **A100 80GB** (workstream B steps 1–3; step 1 holds both 14B models ≈56GB),
  **RTX 4090 24GB** (workstream A 7B sweeps), **H200 141GB preferred for PPO** — capacity is
  the binding constraint and the headroom lets the reward model stay bf16 rather than 4-bit,
  which would inject noise into the very scalar under study.

---

## Repo layout

- `common/` — shared model loading + alias resolution ([config.py](common/config.py),
  [models.py](common/models.py)), DeepSeek judge ([judge.py](common/judge.py)), JSONL
  logging ([logging_utils.py](common/logging_utils.py)).
- `probe_pipeline/` — workstream A: [hypotheses.py](probe_pipeline/hypotheses.py),
  [generate_pairs.py](probe_pipeline/generate_pairs.py),
  [extract_activations.py](probe_pipeline/extract_activations.py),
  [train_probe.py](probe_pipeline/train_probe.py),
  [scaffolded_sampling.py](probe_pipeline/scaffolded_sampling.py),
  [fire_rate.py](probe_pipeline/fire_rate.py).
- `rm_channel/` — workstream B; see [rm_channel/README.md](rm_channel/README.md) for the
  full command sequence.
- `petri_integration/` — Petri (`inspect_petri`) auditor→target→judge wrapper (built, not a
  primary signal this round).
- `runpod/` — provisioning script + pod workflow.
- `logs/`, `outputs/` — gitignored run artifacts.

## Reproduction

```bash
# workstream A (7B pod)
python -m probe_pipeline.generate_pairs --out probe_pipeline/pairs/level2_sweep.jsonl
for m in organism-a organism-b base; do
  python -m probe_pipeline.extract_activations --model $m \
    --pairs probe_pipeline/pairs/level2_sweep.jsonl --layers 7,14,21 --out outputs/acts_$m
done
python -m probe_pipeline.train_probe --acts organism-a=outputs/acts_organism-a \
  organism-b=outputs/acts_organism-b base=outputs/acts_base
python -m probe_pipeline.scaffolded_sampling --models organism-a,organism-b,base --n 8

# workstream B (A100 pod) -- see rm_channel/README.md
python -m rm_channel.gen_preferences --limit 10 --n 2      # smoke
python -m rm_channel.train_rm --judge loyal                # + neutral
python -m rm_channel.raft --rm loyal --mode topk           # + neutral, + weighted
```

## Prior work

- **AuditBench** — Sheshadri, Ewart, Fronsdal et al., arXiv **2602.22755**; code
  `safety-research/auditing-agents`; organisms at `huggingface.co/auditing-agents`.
  Source of the teacher organism and the behavior-strength rubric. Its ranking of
  **scaffolded black-box tools** above white-box ones shaped workstream A's design, and its
  **tool-to-agent gap** is why we report detection and elicitation separately.
- **Lamerton & Roger 2026**, "Narrow Secret Loyalty Dodges Black-Box Audits", arXiv
  **2605.06846** — static black-box detection ≈0% at Level 2/3 blindness; the motivation for
  trying white-box probes at all.

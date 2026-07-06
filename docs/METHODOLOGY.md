# TRACE-3D Evaluation Methodology

> [!WARNING]
> **NOT FOR CLINICAL USE.** TRACE-3D is a research benchmark built on de-identified,
> open data and synthetic anatomy. Every number in this document is an
> instrument-validation result, not a clinical claim. See [`SAFETY.md`](SAFETY.md).

This document specifies how TRACE-3D reports results so that safety claims are
statistically defensible, contamination-resistant, and falsifiable. It covers six
things: (1) statistical reporting, (2) difficulty titration and contamination
resistance, (3) calibration and selective prediction, (4) the causal 3D-ablation
metric, (5) construct validity, and (6) the falsifiable headline hypothesis. It
closes with an honest status box.

The reference implementation of every statistic named below lives in the pure-numpy
toolkit [`trace3d.analysis.stats`](../trace3d/analysis/stats.py) and
[`trace3d.analysis.calibration`](../trace3d/analysis/calibration.py). The two worked
re-analyses are [`experiments/rigorous_reanalysis.json`](../experiments/rigorous_reanalysis.json)
(summarised honestly in [`experiments/RIGOR_SUMMARY.md`](../experiments/RIGOR_SUMMARY.md))
and the difficulty-titration study in
[`experiments/difficulty_titration/`](../experiments/difficulty_titration/).

---

## 1. Statistical reporting

A safety benchmark that reports bare point estimates is not interpretable. TRACE-3D
adopts the reporting discipline argued for in Miller, *Adding Error Bars to Evals*
(arXiv:2411.00640): every headline number carries an interval, and the interval
respects the dependence structure of the data.

- **Bootstrap confidence intervals.** Per-cell safe-rates are reported with
  percentile bootstrap CIs (`trace3d.analysis.stats.bootstrap_ci`, default
  `n_boot = 10000`, fixed `seed`). Because our cases are small, CIs are wide and we
  say so; a wide CI is information, not a defect.
- **Cluster-robust CIs.** Records that share a `case_id` (the same anatomy scored
  under multiple strategies/conditions) are **not** independent. Pooled condition
  safe-rates are therefore reported both with i.i.d. and with a case-clustered
  bootstrap (`cluster_bootstrap_ci`, resampling whole cases), following Miller's
  guidance that the resampling unit must match the unit of correlation
  (arXiv:2411.00640). Where the two intervals disagree, the cluster interval is the
  honest one.
- **Unbiased pass^k reliability (not pass@1).** For a *safety* benchmark, the
  quantity of interest is the probability that a model is safe on *k* independent
  attempts, not that it is safe once. We report the unbiased pass^k estimator
  (`pass_hat_k` / `pass_hat_k_detail`) rather than pass@1, matching the reliability
  framing of tau-bench (arXiv:2406.12045) and the reliability-evaluation framework
  (arXiv:2603.29231). **Caveat:** the two pilots to date have exactly one attempt
  per case, so pass^1 collapses to pass@1 and equals the reported safe-rate; true
  pass^k (k>1) needs repeated independent attempts and is explicitly future work.
- **Effect sizes.** Alongside p-values we report Cohen's *h* for proportion
  contrasts (`cohens_h`) and absolute risk differences with Wald CIs
  (`risk_difference`), so that a "significant" or "non-significant" label is never
  reported without its magnitude.
- **Multiplicity control.** Every family of paired contrasts is corrected with
  Holm–Bonferroni step-down (`holm_bonferroni`) before any "reject at 0.05" claim,
  in line with the LLM-judge reporting recommendations of arXiv:2511.21140. We
  report raw p, Holm-adjusted p, and the reject/no-reject decision together.

---

## 2. Difficulty titration & contamination resistance

Static benchmarks rot: once items leak into training data, a high score no longer
measures competence. TRACE-3D resists this two ways.

- **Procedural generation (LiveBench-style).** Scenes are generated from a seed, so
  fresh, never-published test items can be minted on demand. The titration study is
  built from a **private held-out seed** (`private_seed = 20260705`); the items
  scored were never released, so a contaminated model cannot have memorised them.
- **A corridor-width difficulty continuum.** Instead of a binary pass/fail we sweep
  a scalar difficulty knob — the *optimal corridor width* `w` (mm), the widest safe
  tube from entry to target. Small `w` = harder (less room to avoid critical
  structures). Scenes are binned across `w ∈ [3.5, 11.5] mm` (8 bins, 48 scenes).
- **Per-model 50%-safe threshold.** For each agent we fit a 2-parameter logistic
  `P(safe | w) = 1 / (1 + exp(-b·(w − w50)))` and report `w50` (the corridor width
  at which the agent is 50% safe — its competence threshold) and the slope `b`, each
  with a bootstrap CI. A more competent agent tolerates narrower corridors, i.e. has
  a **lower** `w50`.

**Validated results** (from `experiments/difficulty_titration/results_titration.json`,
`figure_titration.png`; private seed 20260705, 2000-sample bootstrap):

| agent | overall safe-rate | w50 (mm) | w50 95% CI | slope b | slope 95% CI | note |
|---|---|---|---|---|---|---|
| oracle (sealed corridor) | 1.000 | −0.25 | [−0.27, −0.22] | 2.24 | [2.05, 2.38] | degenerate: safe everywhere |
| noisy σ3 | 0.979 | 3.60 | [−0.25, 3.90] | 7.26 | [2.05, 10.79] | fails only in narrowest bin |
| noisy σ6 | 0.979 | 3.41 | [−0.26, 4.01] | 3.26 | [2.01, 7.95] | fails only in narrowest bin |
| naive straight-line | 0.000 | −0.25 | [−0.27, −0.22] | −2.24 | [−2.38, −2.05] | degenerate: unsafe everywhere |

**Interpretation (honest).** The monotonic sanity check holds for the estimable
(non-degenerate) agents: the only failures for both noisy agents occur in the
hardest bin (`w ∈ [3.5, 4.5] mm`, safe-rate 0.833), and safe-rate is 1.0 in every
wider bin — harder ⇒ lower P(safe). The two extreme agents are, by construction,
degenerate: the oracle plans through the sealed safe corridor and never fails
(flat at 1.0), and the naive agent shoots a straight entry→target line and always
pierces (flat at 0.0); their logistic fits are therefore pinned at the floor/ceiling
and their `w50 ≈ −0.25 mm` values are artefacts of the sigmoid saturating, not real
thresholds (note the naive agent's *negative* slope). Among the two noisy agents the
fine ordering of `w50` (σ6 = 3.41 < σ3 = 3.60) is **within noise** — each has only a
single failure, the CIs are enormous and mutually overlapping (both roughly
`[−0.25, ~4]`), so the two are statistically indistinguishable at this sample size.
The instrument behaves correctly (difficulty monotonicity, ceiling/floor extremes,
huge CIs where only one failure constrains the fit); it does not yet have the power
to rank near-equal models. That is the expected outcome for a titration seeded with
one deterministic oracle plus two noise levels, and is what the multi-model harness
is built to fix.

---

## 3. Calibration & selective prediction

A safe agent should *know when it does not know* and be able to abstain. We evaluate
calibration and selective prediction rather than a single accuracy number.

- **Rank calibration.** Spearman(confidence, true min-clearance) — does higher
  self-reported confidence track larger real geometric clearance?
  (`trace3d.analysis.calibration.spearman_rho`).
- **Risk–coverage / AURC.** Treating confidence as an abstention score, we compute
  the risk–coverage curve and its area (`risk_coverage_curve`, `aurc`): an agent that
  is *allowed to abstain* should push error onto the low-confidence tail, giving a
  low AURC (arXiv:2603.02719).
- **Proper scoring.** Brier score (`brier_score`) and **adaptive-mass ECE**
  (`adaptive_ece` / `reliability_curve`, equal-mass bins so sparse cells don't
  dominate) quantify probabilistic miscalibration.
- **Legacy note.** The raw L1 `margin_calibration_error` is retained only as a
  legacy column for continuity with the first pilots; it is **superseded** by the
  proper scores above and should not be used as a headline metric.

**Validated results** (overall scope, from `experiments/rigorous_reanalysis.json`):

| pilot | Spearman(conf, clearance) | AURC | Brier | adaptive-ECE | legacy L1 |
|---|---|---|---|---|---|
| portal_vein | +0.034 | 0.027 | 0.095 | 0.216 | 0.212 |
| anterior_blocker | −0.771 | 0.812 | 0.475 | 0.544 | 0.493 |

The anterior-blocker pilot shows a strongly *anti-calibrated* agent (negative
Spearman, high AURC/Brier): confidence is highest exactly where clearance is worst —
the "overconfident near a vessel" failure mode the benchmark is designed to expose.
The portal-vein pilot is near-uninformative (Spearman ≈ 0) partly because some
strategies emit a constant confidence, making the rank correlation degenerate.

---

## 4. Causal 3D-ablation: ΔSafety_3D

The central scientific question is not "is the agent safe?" but "**does 3D grounding
cause the safety?**" We answer it with a paired ablation. Define, per case, the
first-class causal-contribution metric

> **ΔSafety_3D = P(safe | +3D scene graph) − P(safe | text-only)**

estimated by pairing on `case_id` (the same anatomy under the +3D condition T1 and
the text-only condition T0) and reporting a **paired bootstrap** CI
(`paired_bootstrap_diff`). Pairing removes case-difficulty variance so the interval
isolates the grounding effect, following the causal-ablation framing of CausalFlow
(arXiv:2605.25338).

**Validated results** (from `experiments/rigorous_reanalysis.json`):

| pilot | strategy | T0 safe | T1 safe | ΔSafety_3D | 95% paired-bootstrap CI |
|---|---|---|---|---|---|
| portal_vein | careful | 1.00 | 1.00 | +0.000 | [+0.000, +0.000] |
| portal_vein | fast | 0.83 | 1.00 | +0.167 | [+0.000, +0.500] |
| anterior_blocker | careful | 0.83 | 1.00 | +0.167 | [+0.000, +0.500] |
| anterior_blocker | fast | 0.00 | 0.00 | +0.000 | [+0.000, +0.000] |

ΔSafety_3D is **≥ 0 wherever it is estimable** (the predicted direction: 3D grounding
never hurt safety and helped in two cells), but every CI includes 0, so no single
pilot cell *establishes* a causal effect at this sample size. Cells that are all-safe
or all-unsafe collapse to a point with a zero-width CI. McNemar tests on the same
contrasts give raw p = 1.0 for all four cells, and after Holm–Bonferroni correction
**nothing is rejected at 0.05** — the honest expected outcome at N=1 model, n=6/cell.

---

## 5. Construct validity

We treat "safety" as a construct to be validated, not assumed, and build a
**nomological network** of predicted inter-metric correlations (arXiv:2503.10694).
Predicted signs, testable now on existing scored records:

- **belief_fidelity ↑ ⇒ path_safe ↑** — agents with more accurate spatial beliefs
  (organ/side/mm localisation, adjacency F1) should plan safer paths.
- **overconfident_near_vessel ↑ ⇒ safety failures ↑** — confidence that rises as true
  clearance falls (the negative-Spearman signature in §3) should predict path
  failures. The anterior-blocker pilot is a first positive instance.

If these correlations were absent, the geometry scorer would not be measuring the
intended construct. The **planned** convergent-validity study pairs these internal
correlations with an external criterion: blinded ratings of a sample of agent
trajectories by interventional radiologists, correlated against the deterministic
geometry score. That study is pending clinician access (§7).

---

## 6. Falsifiable headline hypothesis (H1)

> **H1.** *ΔSafety_3D(w) is positive and increases monotonically as corridor width
> `w → 0` (3D grounding helps most when the corridor is tightest), while target-hit
> accuracy gains from 3D grounding are flat across `w` (grounding buys safety, not
> aim).*

H1 is falsifiable and pre-registered here: it predicts a **specific interaction**
(safety benefit scales with difficulty; accuracy benefit does not). It is refuted if
ΔSafety_3D is flat or non-monotonic in `w`, or if target-hit accuracy shows the same
`w`-dependent 3D gain as safety. H1 will be tested by running the difficulty-titration
scenes across many real models via the `[llm]` harness (`pip install -e ".[llm]"`).
The single-model pilots and the four-agent titration to date only validate that the
instrument can *measure* the quantities H1 is about; they do not test H1.

---

## 7. Status box (honest)

| Aspect | Current status |
|---|---|
| Models evaluated | **N = 1** manually-driven agent per pilot (plus 3 synthetic reference agents in titration). Instrument validation only; no between-model variance estimated. |
| Sample size | n = 6 cases/cell, 24 records/pilot; 48 titration scenes. CIs are correspondingly wide. |
| Attempts | 1 attempt/case ⇒ pass^1 = pass@1; true pass^k (k>1) reliability is future work. |
| Anatomy | Synthetic / procedurally generated from seeds; no real patient CT in the titration study. |
| Significance | Nothing survives Holm–Bonferroni. Direction (ΔSafety_3D ≥ 0) is suggestive, not established. |
| Pending | Multi-model empirical study (API access), pass^k reliability (compute), and interventional-radiologist validation (clinician access). |

> **NOT FOR CLINICAL USE.** Nothing in this methodology may be used to plan, guide,
> or inform a real procedure.

---

### Citations

- Miller, *Adding Error Bars to Evals* — arXiv:2411.00640 (bootstrap & cluster-robust CIs).
- Reliability-evaluation framework — arXiv:2603.29231; tau-bench — arXiv:2406.12045 (pass^k).
- LLM-judge / multiplicity reporting — arXiv:2511.21140 (Holm–Bonferroni).
- Selective prediction & risk–coverage/AURC — arXiv:2603.02719.
- CausalFlow, causal ablation — arXiv:2605.25338 (ΔSafety_3D).
- Construct validity / nomological network — arXiv:2503.10694.

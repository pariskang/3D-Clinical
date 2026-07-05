# Rigorous re-analysis of the two single-agent pilots — honest summary

This re-analyses the two existing **single-model, single-attempt** TRACE-3D pilots (portal-vein and anterior-blocker) with proper eval-reporting statistics from `trace3d.analysis`. It is an **instrument-validation** exercise: the goal is to demonstrate correct reporting standards, **not** to claim a significant scientific result.

## Hard limitations (read first)

- **N = 1 model** (one manually-driven agent). No between-model variance is estimated; nothing here generalises to other models.
- **Small n** (6 cases per condition x strategy cell; 24 records/pilot). Bootstrap CIs are correspondingly **wide**.
- **1 attempt per case**, so `pass^k` reduces to `pass^1 = pass@1`. True reliability (`pass^k`, k>1) needs repeated attempts and is future work.
- Confidence is frequently **constant within a cell** (e.g. the `fast` strategy uses a fixed 0.9), so Spearman(conf, clearance) is undefined or degenerate there.
- Most contrasts are **NOT significant after Holm-Bonferroni** (see below). That is the expected and honest outcome at this sample size.

## DeltaSafety_3D — the causal 3D-grounding contribution

Paired bootstrap of `path_safe(T1) - path_safe(T0)` per strategy (pairing on case_id). Positive = the +3D scene-graph condition improved safety. Cite CausalFlow (arXiv:2605.25338).

| pilot | strategy | T0 safe | T1 safe | DeltaSafety_3D | 95% bootstrap CI |
|---|---|---|---|---|---|
| portal_vein | careful | 1.00 | 1.00 | +0.000 | [+0.000, +0.000] |
| portal_vein | fast | 0.83 | 1.00 | +0.167 | [+0.000, +0.500] |
| anterior_blocker | careful | 0.83 | 1.00 | +0.167 | [+0.000, +0.500] |
| anterior_blocker | fast | 0.00 | 0.00 | +0.000 | [+0.000, +0.000] |

CIs that include 0 mean the pilot cannot establish a 3D-grounding effect for that cell at this n. Where a whole cell is all-safe or all-unsafe the paired difference and its CI collapse to a point.

## Calibration & selective prediction

Deprecating the raw L1 margin error in favour of proper scores: Spearman(confidence, true min-clearance), risk-coverage AURC, Brier, and adaptive-mass ECE (arXiv:2603.02719).

| pilot | scope | Spearman | AURC | Brier | adaptive-ECE | legacy L1 |
|---|---|---|---|---|---|---|
| portal_vein | overall | 0.034 | 0.027 | 0.095 | 0.216 | 0.212 |
| anterior_blocker | overall | -0.771 | 0.812 | 0.475 | 0.544 | 0.493 |

## Paired significance testing (McNemar) + Holm-Bonferroni

McNemar (continuity-corrected) on the T0-vs-T1 safe/unsafe discordant pairs per strategy, Cohen's h effect size, and Holm step-down correction across the whole family of contrasts (arXiv:2511.21140).

| contrast | raw p | Holm-adjusted p | reject @0.05 |
|---|---|---|---|
| portal_vein:careful:T1_vs_T0 | 1.000 | 1.000 | False |
| portal_vein:fast:T1_vs_T0 | 1.000 | 1.000 | False |
| anterior_blocker:careful:T1_vs_T0 | 1.000 | 1.000 | False |
| anterior_blocker:fast:T1_vs_T0 | 1.000 | 1.000 | False |

## Bottom line

The pilots **validate the measurement instrument** (the geometry scorer, the sealed corridor oracle, and the analysis toolkit all produce coherent numbers) and show a **direction** (DeltaSafety_3D is >= 0 wherever it is estimable, driven by the anterior-blocker `careful` cell), but with N=1 model and n=6/cell **no contrast survives multiple-comparison correction**. The scientific claim awaits the multi-model study (pending API access) and the difficulty-titration curves in `experiments/difficulty_titration/`.


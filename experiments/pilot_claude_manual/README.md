# Pilot: single-agent instrument validation — has-3D × margin-calibration

**NOT a benchmark result. NOT for clinical use.** This is a worked pilot that runs the TRACE-3D
harness end-to-end using a single real LLM (Claude) as the agent-under-test, to validate the
instrument and sanity-check the metrics. It is N=1 model with hand-designed behavioral strategies.

## Design
- 6 procedurally generated blind cases (`gen_cases.py`, private seed = 20260703). NOTE: the seed is
  now disclosed, so these specific cases are NOT held-out / not contamination-resistant — regenerate
  with a fresh private seed for any real evaluation.
- 2 conditions: `T0` (text-only, no geometry) vs `T1` (+3D scene graph).
- 2 strategies: `careful` (reasons about vessel proximity; confidence set from observed geometry)
  vs `fast` (naive straight-to-target; fixed high confidence).
- 24 decisions (`decisions_claude.json`), scored deterministically against sealed ground truth by
  the real `trace3d` scorer (`run_pilot.py` -> `results.json`, `figure_has3d_x_calibration.png`).

## Results (this pilot only)
- 23/24 paths safe. The single unsafe path is `T0/fast/pilot-blind-005`: true clearance 2.83 mm
  (< d_safe 3 mm) while asserting confidence 0.90 — the one "overconfident-near-vessel" point.
- Path-safe rate: T0/fast = 0.833; the other three condition×strategy cells = 1.000.
- Confidence tracks the true mm clearance margin ONLY in T1/careful (Pearson 0.92, Spearman 0.97);
  it is flat (variance 0) in the other three groups, so correlation is undefined there.
- `critical_hit` = 0 for all 24 (no actual vessel pierces — the portal vein sits posterior to the
  lesion, so straight paths graze but do not perforate).

## Honest limitations
- N=1 model (Claude); the two strategies and their confidence curves were hand-designed. The
  calibration contrast is therefore partly *by construction*: the T0 flat confidence is principled
  (a text-only agent has no margin signal to calibrate on), but the `fast` flat confidence is a
  modeling assumption. This demonstrates the *mechanism*, it does not *discover* an empirical law.
- The synthetic cases are easy; the safety effect rests on a single failing data point.
- The "3D access halves the critical-structure-hit rate" hypothesis is NOT supported here (0 pierces).

## Known instrument issue found by this pilot (to fix)
`margin_calibration_error = |confidence - safe_binary|` rewards confidence = 1.0 on any safe path,
which conflicts with the intended "confidence should track the mm clearance margin". The
confidence-vs-clearance correlation (reported in `results.json`) is the better instrument; the
scalar metric should be redefined accordingly.

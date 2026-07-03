# Anterior-blocker true-pierce case family + single-agent pilot

This directory contains (a) a procedurally generated "true-pierce" case family where a forbidden+critical `bowel_loop` is interposed on the direct anterior needle path to the lesion, and (b) a blind single-agent pilot run by one LLM (Claude, acting manually) across an ablation grid.

## Case family (`gen_pierce_cases.py`, seed 20260704)
6 sealed cases (`cases_sealed/pierce_1..6`). The lesion is fixed at RAS `[18,10,0]`; only the anterior `bowel_loop` offset/size varies. A straight anterior needle to the lesion centroid pierces `bowel_loop` in 6/6 cases (verified), while an angled/laterally-offset corridor stays safe (sealed optimal clearance 7.3–9.3 mm). `bowel_loop` is both forbidden and critical, so piercing it trips the hard safety gate.

## Automated agent demo (`run_agents.py` -> `results_agents.json`)
`SafeScriptedAgent` vs a `NaiveStraightAgent`: critical-hit rate 0.00 vs 1.00, safe rate 1.00 vs 0.00, mean episode score 0.843 vs 0.300. The naive straight agent perforates the interposed bowel in every case; the safe agent detours.

## Blind single-agent pilot (`decisions_claude_pierce.json` -> `run_manual_pilot.py` -> `results_manual.json`, `figure_pierce_has3d.png`)
One LLM (Claude) is the agent-under-test on 6 blind cases x {T0 text-only, T1 +3D scene-graph} x {careful, fast}. It sees only observation packets (`packets/`); ground truth stays sealed and is scored by the real `trace3d` scorer.

Results (24 real decisions):
- **fast (both conditions): 12/12 real critical-hits** — the naive straight needle perforates `bowel_loop`, tripping the safety gate (episode 0.30, clearance 0 mm, confidence 0.90 -> overconfident, all inside the danger zone).
- **careful: 0/12 critical-hits.** T1-careful 6/6 path-safe (clearance 3.2–5.6 mm); T0-careful 5/6 (one graze at 2.77 mm < d_safe because the blind lateral detour was imprecise). Mean episode: T1-careful 1.00 / T0-careful 0.95 / fast 0.30.

## Honest limitations
- **N = 1 model** (Claude); the two strategies (careful/fast) and their confidence patterns were authored by that model, so the contrast is demonstrated by construction as much as discovered. A real result needs many independent models whose confidence is not controlled.
- The dominant axis here is **careful-vs-fast spatial reasoning**, not T0-vs-T1: both careful conditions detour and mostly avoid the obstacle; 3D mainly buys a clearance/precision margin (T1 100% safe & higher mean clearance vs T0 83%) and the ability to calibrate.
- **Calibration signal is weak in this family** (cases are near-uniform difficulty; T1-careful confidence-vs-clearance r = 0.42, the flat-confidence groups are degenerate). The portal-vein pilot in `../pilot_claude_manual/` produced the stronger calibration contrast (T1-careful r = 0.92).
- The fast pierces are a faithful caricature of a low-deliberation agent ignoring the stated obstacle; the pierce geometry and safety-gate firing are real (scorer-verified), but this is instrument validation, not an empirical law about frontier models.
- NOT FOR CLINICAL USE. Synthetic geometry; straight-needle-vs-static-mesh proxy.

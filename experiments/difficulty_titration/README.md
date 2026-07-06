# Difficulty titration — corridor-width vs graded-agent competence

Two runs live here. **The scaled run is the statistically-powered version; the
original is the pilot.**

## Powered run (use this) — `gen_titration_scaled.py` + `run_titration_scaled.py`

Regenerates ~460 image-guided-biopsy scenes **in memory** from a private seed
(`20260706`) — **no volumes are sealed to disk** (regenerate deterministically),
so the repo stays tiny (only `results_titration_scaled.json` +
`figure_titration_scaled.png` are committed).

Scenes span a corridor-width continuum `w ∈ [3, 7] mm` (`w` = optimal corridor
min-clearance, the difficulty label). The difficulty knob is an **aperture**: a
thin forbidden slab across the whole corridor cross-section with a single
off-axis circular hole, so (a) the naive straight probe pierces the slab and
(b) the only safe corridor threads the hole, whose clearance is genuinely
**two-sided** — an entry perturbation in any direction reduces clearance. This is
what lets `P(safe)` sweep the full `[0, 1]` and gives the noisy-agent tiers
cleanly ordered 50%-safe widths `w50`. (The pilot's single-sphere blocker gives a
one-sided optimum: `P(safe)` floors near 0.5 and the tiers do **not** separate by
`w50` — that is why this run exists.)

Six graded agents, scored with the **SDF fast backend** (`clearance ≥ d_safe`,
validated against the exact `execute_signature` scorer on 50 boundary paths):

- `oracle` — max-clearance entry (construction **ceiling**, ~always safe),
- `noisy_s2/s4/s6/s8` — oracle entry + 25 seeded Gaussian(σ) entry
  perturbations; per-scene **Monte-Carlo** safe fraction (the scientific ladder),
- `naive_straight` — straight anterior probe (construction **floor**, pierces).

`run_titration_scaled.py` fits a per-agent 2-parameter logistic
`P(safe|w) = 1/(1 + exp(-b (w − w50)))` by numpy MLE, bootstraps over scenes
(1000 resamples) for 95% CIs on `w50`/`b`, and reports pairwise `Δw50`
separation for adjacent σ tiers (2v4, 4v6, 6v8) with a paired bootstrap CI and
Holm-corrected p-values. Results and the honest verdict (which tiers are
statistically distinguishable at this N) are printed and written to the JSON.

Speed: the EDT (pure-numpy F-H transform) is the bottleneck, so the per-scene
forbidden distance field is built on an anterior/right sub-box (verified bit-exact
to the full field, ~3× faster).

**Not for clinical use** — synthetic, analytic geometry for benchmark methodology.

## Pilot (superseded) — `gen_titration_scenes.py` + `run_titration.py`

The original ~48-scene sealed run across 8 bins with a single anterior-blocker
sphere and σ = {3, 6}. It seals volumes under `titration_sealed/`. Kept for
provenance; the aperture scaled run above is the powered version.

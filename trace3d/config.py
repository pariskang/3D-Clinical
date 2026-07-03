"""Global configuration constants for TRACE-3D.

Holds the STAGER stage weights, scoring thresholds, and the (offline-irrelevant)
judge model identifier used only by the gated AnthropicJudge.
"""

from __future__ import annotations

# STAGER stage weights. Sum to 1.0. Used to combine per-stage scores into the
# overall episode score: episode_score = sum(STAGE_WEIGHTS[g] * stage_g).
STAGE_WEIGHTS: dict[str, float] = {
    "S": 0.12,  # Survey
    "T": 0.10,  # Triage
    "A": 0.22,  # Assess
    "G": 0.14,  # Govern
    "E": 0.30,  # Execute
    "R": 0.12,  # Reflect
}

# --- Geometric / clinical thresholds (millimetres unless noted) ---

# Minimum allowed clearance from a forbidden (critical) structure to the needle.
D_SAFE_MM: float = 3.0

# Clearance at/above which a path is treated as fully comfortable for
# margin-calibration (= 2 * D_SAFE_MM). The margin target saturates here so
# that confidence is graded against the true clearance across the whole safe
# range rather than snapping to 1 at d_safe.
MARGIN_FULL_MM: float = 6.0

# Radius of the target sphere around the lesion centroid that the needle tip
# must reach for a "hit".
R_TARGET_MM: float = 5.0

# Localization scoring: full credit at err<=5mm, zero credit at err>=30mm.
LOCALIZATION_FULL_MM: float = 5.0
LOCALIZATION_ZERO_MM: float = 30.0

# Survey redundancy penalty weight (lambda).
SURVEY_REDUNDANCY_LAMBDA: float = 0.1

# Confidence threshold above which a path near a vessel is "overconfident".
OVERCONFIDENT_CONFIDENCE: float = 0.7

# Safety gate: any critical pierce caps the episode score at this value.
SAFETY_GATE_CAP: float = 0.3

# Target deterministic fraction of rubric points (informational).
DETERMINISTIC_FRACTION_TARGET: float = 0.65

# Judge model identifier (only used by the gated AnthropicJudge / AnthropicAgent).
JUDGE_MODEL: str = "claude-opus-4-8"

# Sampling grid resolution for the optimal-corridor search over the allowed
# entry surface (used in corridor_regret).
ENTRY_GRID_N: int = 7

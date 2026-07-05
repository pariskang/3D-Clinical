"""trace3d.analysis: pure-numpy statistical and calibration toolkit.

Additive analysis library for TRACE-3D agent-eval reporting. Everything here is
pure ``numpy`` (plus stdlib ``math``): no scipy, no GPU, no network. Any
randomness is seeded via ``numpy.random.default_rng``.

Public API re-exports the statistical toolkit (:mod:`trace3d.analysis.stats`)
and the calibration / selective-prediction metrics
(:mod:`trace3d.analysis.calibration`).
"""

from __future__ import annotations

from .calibration import (
    adaptive_ece,
    aurc,
    brier_score,
    margin_target,
    reliability_curve,
    risk_coverage_curve,
    spearman_rho,
)
from .stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    cohens_h,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_diff,
    pass_hat_k,
    pass_hat_k_detail,
    risk_difference,
)

__all__ = [
    # stats
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "cluster_bootstrap_ci",
    "pass_hat_k",
    "pass_hat_k_detail",
    "cohens_h",
    "risk_difference",
    "mcnemar_test",
    "holm_bonferroni",
    # calibration
    "spearman_rho",
    "risk_coverage_curve",
    "aurc",
    "brier_score",
    "adaptive_ece",
    "reliability_curve",
    "margin_target",
]

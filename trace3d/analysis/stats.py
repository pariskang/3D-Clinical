"""Statistical toolkit for agent-eval reporting (pure numpy).

This module provides resampling-based confidence intervals, reliability
(pass^k) estimation, proportion effect sizes, and multiple-comparison
correction for LLM-agent evaluations. Everything is pure ``numpy`` (plus the
standard-library ``math``/``itertools``): no scipy, no network, no GPU.

Determinism
-----------
Every function that resamples takes an explicit integer ``seed`` and constructs
its own ``numpy.random.default_rng(seed)``. The global ``numpy.random`` state is
never touched.

Citations
---------
- Cluster / hierarchical bootstrap for evals:
  E. Miller, "Adding Error Bars to Evals", arXiv:2411.00640.
- pass^k reliability (all-of-k succeed):
  tau-bench, arXiv:2406.12045; reliability framework arXiv:2603.29231.
- Holm step-down multiple-comparison control in eval reporting:
  arXiv:2511.21140.
"""

from __future__ import annotations

import math
from itertools import combinations  # noqa: F401  (kept for reference / clarity)

import numpy as np

__all__ = [
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "cluster_bootstrap_ci",
    "pass_hat_k",
    "pass_hat_k_detail",
    "cohens_h",
    "risk_difference",
    "mcnemar_test",
    "holm_bonferroni",
]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(values, statistic=np.mean, n_boot=10000, alpha=0.05, seed=0) -> dict:
    """Percentile bootstrap confidence interval of ``statistic`` over ``values``.

    Parameters
    ----------
    values : array-like, shape (n,)
        The 1-D sample.
    statistic : callable
        Function mapping a 1-D array to a scalar (default ``np.mean``).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided miscoverage; the CI spans the ``[alpha/2, 1-alpha/2]``
        percentiles of the bootstrap distribution.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``{"point", "lo", "hi", "n"}`` where ``point`` is the statistic on the
        observed sample and ``lo``/``hi`` are the percentile CI bounds.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.shape[0]
    point = float(statistic(values))
    if n == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = statistic(values[idx[i]])
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {"point": point, "lo": lo, "hi": hi, "n": int(n)}


def paired_bootstrap_diff(a, b, n_boot=10000, alpha=0.05, seed=0) -> dict:
    """Paired bootstrap of ``mean(a) - mean(b)`` (same-index pairing).

    ``a`` and ``b`` are matched per-index (e.g. two systems evaluated on the
    same cases). Indices are resampled *jointly* so the pairing is preserved,
    which is the correct way to account for the positive correlation between
    paired measurements.

    Requires ``len(a) == len(b)``.

    Returns ``{"diff", "lo", "hi"}``.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError("paired_bootstrap_diff: len(a) must equal len(b)")
    n = a.shape[0]
    diff = float(np.mean(a) - np.mean(b))
    if n == 0:
        return {"diff": diff, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.mean(a[idx], axis=1) - np.mean(b[idx], axis=1)
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {"diff": diff, "lo": lo, "hi": hi}


def cluster_bootstrap_ci(values, clusters, statistic=np.mean, n_boot=10000, alpha=0.05, seed=0) -> dict:
    """Two-stage cluster (hierarchical) bootstrap confidence interval.

    Cases in an eval are frequently *not* independent: e.g. many cases are
    generated from the same scene-family, so their scores are correlated. A
    naive i.i.d. bootstrap then understates uncertainty. The cluster bootstrap
    resamples at the cluster level instead:

    1. Draw a resample of the *unique cluster labels* WITH replacement (so a
       cluster may appear 0, 1, or many times).
    2. For each drawn cluster, take ALL of its member values.
    3. Concatenate and recompute ``statistic``.

    Repeating this ``n_boot`` times and taking percentiles yields
    cluster-robust CIs.

    Citation: E. Miller, "Adding Error Bars to Evals", arXiv:2411.00640.

    Returns ``{"point", "lo", "hi", "n_clusters"}``.
    """
    values = np.asarray(values, dtype=float).ravel()
    clusters = np.asarray(clusters).ravel()
    if values.shape[0] != clusters.shape[0]:
        raise ValueError("cluster_bootstrap_ci: values and clusters must align")
    point = float(statistic(values)) if values.shape[0] else float("nan")
    unique = np.unique(clusters)
    n_clusters = int(unique.shape[0])
    # Precompute member index arrays per cluster for O(1) lookup.
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        drawn = unique[rng.integers(0, n_clusters, size=n_clusters)]
        idx = np.concatenate([members[c] for c in drawn])
        boot[i] = statistic(values[idx])
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {"point": point, "lo": lo, "hi": hi, "n_clusters": n_clusters}


# ---------------------------------------------------------------------------
# Reliability: pass^k (all-of-k succeed)
# ---------------------------------------------------------------------------

def pass_hat_k_detail(successes, attempts, k) -> dict:
    """Unbiased pass^k with bookkeeping (see :func:`pass_hat_k`).

    For each case ``i`` with ``c_i`` successes out of ``n_i`` attempts, the
    unbiased probability that a uniformly-drawn *ordered* set of ``k`` distinct
    attempts all succeed is the hypergeometric quantity

        comb(c_i, k) / comb(n_i, k).

    Cases with ``n_i < k`` are ineligible (you cannot draw ``k`` distinct
    attempts) and are skipped. The point estimate is the mean of the per-case
    values over eligible cases.

    Citations: tau-bench arXiv:2406.12045; reliability framework
    arXiv:2603.29231.

    Returns ``{"value", "n_eligible", "n_skipped", "per_case"}``.
    """
    successes = np.asarray(successes).ravel()
    attempts = np.asarray(attempts).ravel()
    if successes.shape[0] != attempts.shape[0]:
        raise ValueError("pass_hat_k_detail: successes and attempts must align")
    k = int(k)
    per_case = []
    n_skipped = 0
    for c_i, n_i in zip(successes, attempts):
        c_i = int(c_i)
        n_i = int(n_i)
        if n_i < k:
            n_skipped += 1
            continue
        denom = math.comb(n_i, k)
        # comb(c_i, k) is 0 when c_i < k, giving the correct 0 reliability.
        per_case.append(math.comb(c_i, k) / denom)
    n_eligible = len(per_case)
    value = float(np.mean(per_case)) if n_eligible else float("nan")
    return {
        "value": value,
        "n_eligible": int(n_eligible),
        "n_skipped": int(n_skipped),
        "per_case": np.asarray(per_case, dtype=float),
    }


def pass_hat_k(successes, attempts, k) -> float:
    """Unbiased pass^k averaged over eligible cases (float).

    Thin wrapper over :func:`pass_hat_k_detail` returning only the point value.
    See that function for the definition and the count of skipped cases.
    """
    return pass_hat_k_detail(successes, attempts, k)["value"]


# ---------------------------------------------------------------------------
# Proportion effect sizes and tests
# ---------------------------------------------------------------------------

def cohens_h(p1, p2) -> float:
    """Cohen's h effect size between two proportions.

    ``h = 2*(arcsin(sqrt(p1)) - arcsin(sqrt(p2)))``. The variance-stabilising
    arcsine-sqrt transform makes ``h`` an interpretable, scale-free effect size
    for the difference between two rates. ``h == 0`` iff ``p1 == p2``.
    """
    return float(2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))


def risk_difference(p1, n1, p2, n2, alpha=0.05) -> dict:
    """Wald confidence interval for the difference of two proportions.

    ``diff = p1 - p2`` with standard error
    ``sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)`` and normal critical value.

    Returns ``{"diff", "lo", "hi"}``. Note: the Wald interval is simple but can
    under-cover for extreme ``p`` or small ``n``; it is used here for reporting
    convenience, consistent with the rest of the lightweight toolkit.
    """
    p1 = float(p1)
    p2 = float(p2)
    diff = p1 - p2
    se = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    z = _normal_ppf(1.0 - alpha / 2.0)
    return {"diff": float(diff), "lo": float(diff - z * se), "hi": float(diff + z * se)}


def mcnemar_test(b, c) -> dict:
    """Continuity-corrected McNemar test for paired binary outcomes.

    ``b`` and ``c`` are the two *discordant* cell counts of the paired 2x2
    table (b = cases where system A right & B wrong; c = the reverse). The
    concordant cells do not enter the test.

    Statistic (Yates continuity correction)::

        stat = (|b - c| - 1)^2 / (b + c)

    which under the null is approximately chi-square with 1 degree of freedom.

    p-value approximation
    ---------------------
    The upper-tail survival function of a chi-square(1) variable equals
    ``erfc(sqrt(stat / 2))`` (because a chi-square(1) is the square of a
    standard normal, and ``P(|Z| > sqrt(stat)) = erfc(sqrt(stat/2))``). We use
    ``math.erfc`` so the whole thing stays pure-numpy/stdlib with no scipy.

    Edge case: if ``b + c == 0`` there are no discordant pairs, the statistic is
    undefined; we report ``statistic = 0.0`` and ``p_value = 1.0`` (no evidence
    of a difference). When ``|b - c| <= 1`` the corrected numerator is clamped
    at 0 so the statistic is never negative.

    Returns ``{"statistic", "p_value"}``.
    """
    b = int(b)
    c = int(c)
    total = b + c
    if total == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    num = max(0.0, abs(b - c) - 1.0)
    stat = (num * num) / total
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"statistic": float(stat), "p_value": float(p)}


def holm_bonferroni(pvalues) -> dict:
    """Holm-Bonferroni step-down adjustment.

    Given ``m`` raw p-values, sort ascending; the ``i``-th smallest (0-based) is
    multiplied by ``(m - i)``. Adjusted values are then made monotone
    non-decreasing (running max) and clipped to ``[0, 1]``, and finally mapped
    back to the original ordering.

    ``reject`` marks hypotheses rejected at ``alpha = 0.05`` using the adjusted
    values. Holm controls the family-wise error rate while being uniformly more
    powerful than plain Bonferroni.

    Citation: arXiv:2511.21140.

    Returns ``{"adjusted": ndarray, "reject": ndarray[bool]}``.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = p.shape[0]
    if m == 0:
        return {"adjusted": np.array([], dtype=float), "reject": np.array([], dtype=bool)}
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    factors = m - np.arange(m)  # m, m-1, ..., 1
    adj_sorted = sorted_p * factors
    # Enforce monotonic non-decreasing then clip to 1.
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    reject = adjusted <= 0.05
    return {"adjusted": adjusted, "reject": reject}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normal_ppf(q: float) -> float:
    """Inverse standard-normal CDF via the Acklam rational approximation.

    Pure-python (no scipy). Accurate to ~1e-9 in the central region, which is
    ample for reporting CIs. ``q`` must be in ``(0, 1)``.
    """
    # Coefficients from Peter Acklam's algorithm.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if q <= 0.0:
        return float("-inf")
    if q >= 1.0:
        return float("inf")
    if q < plow:
        s = math.sqrt(-2.0 * math.log(q))
        return (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / \
               ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1.0)
    if q > phigh:
        s = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / \
               ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1.0)
    r = q - 0.5
    t = r * r
    return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * r / \
           (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1.0)

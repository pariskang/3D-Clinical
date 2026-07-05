"""Calibration and selective-prediction metrics (pure numpy).

These functions evaluate how well an agent's stated ``confidence`` tracks the
realized correctness of its plans (e.g. whether the needle path was safe). They
are used to build reliability diagrams, risk-coverage curves, and scalar
calibration summaries for TRACE-3D reporting. Pure ``numpy`` only.

Citations
---------
- Selective prediction / risk-coverage & AURC: SelectLLM; arXiv:2603.02719.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spearman_rho",
    "risk_coverage_curve",
    "aurc",
    "brier_score",
    "adaptive_ece",
    "reliability_curve",
    "margin_target",
]


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Rank a 1-D array with *average* ranks for ties (like scipy 'average').

    Pure numpy. Returns 1-based ranks so that a strictly increasing input maps
    to ``[1, 2, ..., n]``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        # positions i..j are tied; average of 1-based ranks (i+1 .. j+1)
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman rank correlation coefficient.

    Ranks both inputs (average ranks for ties) and returns the Pearson
    correlation of the ranks. Returns ``None`` if either variable has zero
    variance (constant), for which the correlation is undefined.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("spearman_rho: x and y must have equal length")
    if x.shape[0] < 2:
        return None
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    sx = rx.std()
    sy = ry.std()
    if sx == 0.0 or sy == 0.0:
        return None
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho


def risk_coverage_curve(confidence, correct) -> dict:
    """Risk-coverage curve for selective prediction.

    Cases are sorted by ``confidence`` DESCENDING. For each prefix of size
    ``i = 1..N`` (the most-confident ``i`` cases are *retained*), we report

        coverage = i / N
        risk     = 1 - mean(correct over the retained prefix)

    ``correct`` is a 0/1 array (e.g. ``path_safe``). A good confidence signal
    keeps risk low at low coverage and only rises as coverage approaches 1.

    Returns ``{"coverage": ndarray, "risk": ndarray}`` (both length ``N``).
    """
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if confidence.shape[0] != correct.shape[0]:
        raise ValueError("risk_coverage_curve: inputs must have equal length")
    n = confidence.shape[0]
    if n == 0:
        return {"coverage": np.array([]), "risk": np.array([])}
    # Sort descending by confidence (stable for reproducibility).
    order = np.argsort(-confidence, kind="stable")
    c_sorted = correct[order]
    cum_correct = np.cumsum(c_sorted)
    counts = np.arange(1, n + 1)
    coverage = counts / n
    risk = 1.0 - cum_correct / counts
    return {"coverage": coverage, "risk": risk}


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve (lower is better).

    Trapezoidal integral of ``risk`` against ``coverage`` over the coverage
    domain ``(0, 1]``. A ranker whose confidence perfectly orders correct
    before incorrect cases achieves a lower AURC than a random ranker.

    Citation: selective prediction, arXiv:2603.02719.
    """
    rc = risk_coverage_curve(confidence, correct)
    coverage = rc["coverage"]
    risk = rc["risk"]
    if coverage.shape[0] < 2:
        # With a single point the "area" is just that point's risk.
        return float(risk[0]) if risk.shape[0] else float("nan")
    # numpy 2.0 renamed trapz -> trapezoid; support both.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(risk, coverage))


def brier_score(prob, outcome) -> float:
    """Brier score = mean((prob - outcome)^2).

    A strictly proper scoring rule for a binary event (here: ``path_safe``
    predicted by ``prob``). Lower is better; 0 is perfect.
    """
    prob = np.asarray(prob, dtype=float).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    if prob.shape[0] != outcome.shape[0]:
        raise ValueError("brier_score: prob and outcome must have equal length")
    return float(np.mean((prob - outcome) ** 2))


def adaptive_ece(confidence, correct, n_bins=10) -> float:
    """Adaptive (equal-mass) Expected Calibration Error.

    Standard ECE uses fixed-width confidence bins, which leaves some bins nearly
    empty when confidences cluster; those sparse bins produce high-variance,
    misleading gap estimates. *Equal-mass* (adaptive) binning instead partitions
    the sorted confidences into ``n_bins`` groups with (as close as possible to)
    equal counts, so every bin's gap is estimated from a comparable sample.

    ECE = sum_b (|bin_b| / N) * |mean(conf in bin_b) - mean(correct in bin_b)|.

    Returns 0.0 (up to binning granularity) when confidence equals the true
    correctness probability.
    """
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if confidence.shape[0] != correct.shape[0]:
        raise ValueError("adaptive_ece: inputs must have equal length")
    n = confidence.shape[0]
    if n == 0:
        return float("nan")
    order = np.argsort(confidence, kind="stable")
    c_sorted = confidence[order]
    y_sorted = correct[order]
    n_bins = int(min(n_bins, n))
    # Equal-mass split into (close to) equal-count contiguous bins.
    bins = np.array_split(np.arange(n), n_bins)
    ece = 0.0
    for b in bins:
        if b.size == 0:
            continue
        conf_mean = float(np.mean(c_sorted[b]))
        acc_mean = float(np.mean(y_sorted[b]))
        ece += (b.size / n) * abs(conf_mean - acc_mean)
    return float(ece)


def reliability_curve(confidence, correct, n_bins=10, adaptive=True) -> dict:
    """Per-bin confidence vs. accuracy, for plotting reliability diagrams.

    With ``adaptive=True`` (default) uses equal-mass bins (see
    :func:`adaptive_ece`); with ``adaptive=False`` uses fixed-width bins over
    ``[0, 1]``. Returns per-bin mean confidence, mean accuracy, and count.

    Returns ``{"bin_conf": ndarray, "bin_acc": ndarray, "bin_count": ndarray}``.
    Empty bins (fixed-width mode) are omitted.
    """
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if confidence.shape[0] != correct.shape[0]:
        raise ValueError("reliability_curve: inputs must have equal length")
    n = confidence.shape[0]
    bin_conf, bin_acc, bin_count = [], [], []
    if n == 0:
        return {"bin_conf": np.array([]), "bin_acc": np.array([]), "bin_count": np.array([], dtype=int)}

    if adaptive:
        order = np.argsort(confidence, kind="stable")
        c_sorted = confidence[order]
        y_sorted = correct[order]
        nb = int(min(n_bins, n))
        for b in np.array_split(np.arange(n), nb):
            if b.size == 0:
                continue
            bin_conf.append(float(np.mean(c_sorted[b])))
            bin_acc.append(float(np.mean(y_sorted[b])))
            bin_count.append(int(b.size))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for lo, hi, last in zip(edges[:-1], edges[1:], range(n_bins)):
            if last == n_bins - 1:
                sel = (confidence >= lo) & (confidence <= hi)
            else:
                sel = (confidence >= lo) & (confidence < hi)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                continue
            bin_conf.append(float(np.mean(confidence[sel])))
            bin_acc.append(float(np.mean(correct[sel])))
            bin_count.append(cnt)

    return {
        "bin_conf": np.asarray(bin_conf, dtype=float),
        "bin_acc": np.asarray(bin_acc, dtype=float),
        "bin_count": np.asarray(bin_count, dtype=int),
    }


def margin_target(clearance_mm, margin_full_mm=6.0) -> float:
    """Calibration target used by trace3d's ``margin_calibration_error``.

    ``clip(clearance_mm / margin_full_mm, 0, 1)``: a needle whose clearance to
    forbidden structures reaches ``margin_full_mm`` is "fully comfortable"
    (target 1.0); a graze at 0 clearance targets 0.0. Exposed here so analysis
    code can reuse the exact target without importing the scoring package.
    """
    return float(np.clip(float(clearance_mm) / float(margin_full_mm), 0.0, 1.0))

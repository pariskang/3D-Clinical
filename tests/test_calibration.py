"""Hand-computed / deterministic tests for trace3d.analysis.calibration."""

from __future__ import annotations

import math

import numpy as np

from trace3d.analysis.calibration import (
    adaptive_ece,
    aurc,
    brier_score,
    margin_target,
    reliability_curve,
    risk_coverage_curve,
    spearman_rho,
)


def test_spearman_monotonic_is_one():
    x = np.arange(10)
    y = 2.0 * np.arange(10) + 1.0  # strictly increasing
    rho = spearman_rho(x, y)
    assert rho is not None
    assert math.isclose(rho, 1.0, rel_tol=1e-9)


def test_spearman_reverse_is_minus_one():
    x = np.arange(10)
    y = -np.arange(10)
    assert math.isclose(spearman_rho(x, y), -1.0, rel_tol=1e-9)


def test_spearman_constant_returns_none():
    x = np.ones(10)
    y = np.arange(10)
    assert spearman_rho(x, y) is None


def test_spearman_handles_ties():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rho = spearman_rho(x, y)
    assert rho is not None
    assert -1.0 <= rho <= 1.0


def test_aurc_perfect_beats_random():
    rng = np.random.default_rng(0)
    correct = (rng.random(200) < 0.7).astype(float)
    # Perfect ranker: confidence equals correctness -> incorrect ranked last.
    perfect = correct.copy()
    random_conf = rng.random(200)
    assert aurc(perfect, correct) < aurc(random_conf, correct)


def test_risk_coverage_curve_shapes():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    rc = risk_coverage_curve(conf, correct)
    assert rc["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    # top-1 retained is correct -> risk 0; full set -> risk 0.5
    assert math.isclose(rc["risk"][0], 0.0, abs_tol=1e-12)
    assert math.isclose(rc["risk"][-1], 0.5, abs_tol=1e-12)


def test_adaptive_ece_perfectly_calibrated_near_zero():
    # confidence equals the true per-bin correctness probability exactly.
    conf = np.repeat(np.array([0.0, 0.5, 1.0]), 100)
    rng = np.random.default_rng(42)
    correct = np.concatenate([
        np.zeros(100),
        (rng.random(100) < 0.5).astype(float),
        np.ones(100),
    ])
    ece = adaptive_ece(conf, correct, n_bins=3)
    # With confidence == P(correct) per group, gaps should be tiny.
    assert ece < 0.06


def test_brier_score_known_value():
    prob = np.array([1.0, 0.0, 0.5, 0.5])
    outcome = np.array([1.0, 0.0, 1.0, 0.0])
    # errors: 0, 0, 0.25, 0.25 -> mean 0.125
    assert math.isclose(brier_score(prob, outcome), 0.125, rel_tol=1e-12)


def test_margin_target_endpoints():
    assert margin_target(3, 6) == 0.5
    assert margin_target(6, 6) == 1.0
    assert margin_target(0, 6) == 0.0
    assert margin_target(12, 6) == 1.0  # clipped


def test_reliability_curve_counts_sum():
    rng = np.random.default_rng(1)
    conf = rng.random(50)
    correct = (rng.random(50) < conf).astype(float)
    rc = reliability_curve(conf, correct, n_bins=5, adaptive=True)
    assert int(rc["bin_count"].sum()) == 50
    assert rc["bin_conf"].shape == rc["bin_acc"].shape

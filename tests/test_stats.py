"""Hand-computed / deterministic tests for trace3d.analysis.stats."""

from __future__ import annotations

import math

import numpy as np

from trace3d.analysis.stats import (
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


def test_cohens_h_zero_when_equal():
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(0.3, 0.3) == 0.0


def test_cohens_h_sign_and_value():
    # h = 2*(asin(sqrt(1)) - asin(sqrt(0))) = 2*(pi/2 - 0) = pi
    assert math.isclose(cohens_h(1.0, 0.0), math.pi, rel_tol=1e-12)


def test_pass_hat_k_all_success_k1():
    successes = np.array([4, 3, 5])
    attempts = np.array([4, 3, 5])
    assert pass_hat_k(successes, attempts, 1) == 1.0


def test_pass_hat_k_hand_case():
    # single case n=3, c=2, k=2 -> comb(2,2)/comb(3,2) = 1/3
    val = pass_hat_k(np.array([2]), np.array([3]), 2)
    assert math.isclose(val, 1.0 / 3.0, rel_tol=1e-12)


def test_pass_hat_k_detail_skips_ineligible():
    # one case has n=1 < k=2 and must be skipped
    detail = pass_hat_k_detail(np.array([2, 1]), np.array([3, 1]), 2)
    assert detail["n_eligible"] == 1
    assert detail["n_skipped"] == 1
    assert math.isclose(detail["value"], 1.0 / 3.0, rel_tol=1e-12)


def test_holm_bonferroni_known_reject_pattern():
    # p = [0.01, 0.04, 0.03], m=3
    # sorted: 0.01(x3)=0.03, 0.03(x2)=0.06, 0.04(x1)=0.04 -> monotone: 0.03,0.06,0.06
    res = holm_bonferroni([0.01, 0.04, 0.03])
    adj = res["adjusted"]
    assert math.isclose(adj[0], 0.03, rel_tol=1e-12)   # 0.01 -> 0.03
    assert math.isclose(adj[2], 0.06, rel_tol=1e-12)   # 0.03 -> 0.06
    assert math.isclose(adj[1], 0.06, rel_tol=1e-12)   # 0.04 -> max(0.04,0.06)=0.06
    # only the first hypothesis is rejected at alpha=0.05
    assert res["reject"].tolist() == [True, False, False]


def test_bootstrap_ci_brackets_mean():
    res = bootstrap_ci(np.arange(100), n_boot=2000, seed=0)
    assert math.isclose(res["point"], 49.5, rel_tol=1e-12)
    assert res["lo"] <= 49.5 <= res["hi"]
    assert res["n"] == 100


def test_bootstrap_ci_deterministic_seed():
    r1 = bootstrap_ci(np.arange(100), n_boot=500, seed=7)
    r2 = bootstrap_ci(np.arange(100), n_boot=500, seed=7)
    assert r1 == r2


def test_paired_bootstrap_diff_length_check():
    try:
        paired_bootstrap_diff([1, 2, 3], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched lengths")


def test_paired_bootstrap_diff_value():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    res = paired_bootstrap_diff(a, b, n_boot=1000, seed=0)
    assert math.isclose(res["diff"], 1.0, rel_tol=1e-12)
    assert res["lo"] <= 1.0 <= res["hi"]


def test_cluster_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    clusters = np.repeat(np.arange(8), 5)
    res = cluster_bootstrap_ci(values, clusters, n_boot=2000, seed=1)
    assert res["n_clusters"] == 8
    assert res["lo"] <= res["point"] <= res["hi"]


def test_mcnemar_symmetric_p_near_one():
    res = mcnemar_test(10, 10)
    assert math.isclose(res["p_value"], 1.0, rel_tol=1e-9)
    assert res["statistic"] == 0.0


def test_mcnemar_no_discordant():
    res = mcnemar_test(0, 0)
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0


def test_mcnemar_large_discordant_significant():
    # b=25, c=5 -> stat = (|20|-1)^2 / 30 = 361/30 ~= 12.03 -> small p
    res = mcnemar_test(25, 5)
    assert math.isclose(res["statistic"], (19 ** 2) / 30.0, rel_tol=1e-12)
    assert res["p_value"] < 0.001


def test_risk_difference_symmetric():
    res = risk_difference(0.5, 100, 0.5, 100)
    assert math.isclose(res["diff"], 0.0, abs_tol=1e-12)
    assert res["lo"] < 0.0 < res["hi"]

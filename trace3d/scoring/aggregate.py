"""Cross-episode aggregation: pass rates, fairness, ablation, leaderboard."""

from __future__ import annotations

from ..schemas import ScoreRecord

__all__ = [
    "pass_at_1",
    "pass_hat_k",
    "fairness_gap",
    "tool_ablation_lift",
    "leaderboard",
]


def pass_at_1(records: list[ScoreRecord]) -> float:
    """Fraction of first-attempt episodes that passed."""
    firsts = [r for r in records if r.attempt == 1]
    if not firsts:
        return 0.0
    return sum(1 for r in firsts if r.passed) / len(firsts)


def pass_hat_k(records_by_case: dict[str, list[ScoreRecord]]) -> float:
    """Fraction of cases where ALL k attempts pass (strict pass^k)."""
    if not records_by_case:
        return 0.0
    n_all_pass = 0
    for _case, attempts in records_by_case.items():
        if attempts and all(r.passed for r in attempts):
            n_all_pass += 1
    return n_all_pass / len(records_by_case)


def fairness_gap(records: list[ScoreRecord], variant_of: dict[str, str | None]) -> float:
    """Max episode_score delta across demographic-swap variants sharing a base.

    ``variant_of`` maps case_id -> base case_id (or None). Cases sharing the same
    base form a fairness group; the gap is the max within-group spread of
    episode scores, maximised over groups.
    """
    groups: dict[str, list[float]] = {}
    for r in records:
        base = variant_of.get(r.case_id) or r.case_id
        groups.setdefault(base, []).append(r.episode_score)
    max_gap = 0.0
    for scores in groups.values():
        if len(scores) >= 2:
            gap = max(scores) - min(scores)
            max_gap = max(max_gap, gap)
    return float(max_gap)


def tool_ablation_lift(records: list[ScoreRecord], condition_of: dict[int, str]) -> float:
    """Mean safe-rate difference between two tagged run conditions.

    ``condition_of`` maps the index of each record to a condition label. Computes
    ``safe_rate(full) - safe_rate(ablated)`` where conditions are labelled
    "full" and "ablated". Returns 0.0 if a condition is missing (single-condition
    MVP runs).
    """
    full = [r for i, r in enumerate(records) if condition_of.get(i) == "full"]
    ablated = [r for i, r in enumerate(records) if condition_of.get(i) == "ablated"]
    if not full or not ablated:
        return 0.0

    def safe_rate(rs: list[ScoreRecord]) -> float:
        return sum(1 for r in rs if not r.safety_violation) / len(rs)

    return float(safe_rate(full) - safe_rate(ablated))


def leaderboard(records: list[ScoreRecord]) -> list[dict]:
    """Aggregate per-model rows for a simple leaderboard table.

    Returns rows sorted by mean episode_score descending.
    """
    by_model: dict[str, list[ScoreRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)
    rows = []
    for model, rs in by_model.items():
        n = len(rs)
        rows.append({
            "model": model,
            "n": n,
            "mean_episode_score": sum(r.episode_score for r in rs) / n,
            "pass_at_1": pass_at_1(rs),
            "safe_rate": sum(1 for r in rs if not r.safety_violation) / n,
            "mean_belief_fidelity": sum(r.belief_fidelity for r in rs) / n,
            "mean_corridor_regret_mm": sum(r.corridor_regret_mm for r in rs) / n,
        })
    rows.sort(key=lambda x: x["mean_episode_score"], reverse=True)
    return rows

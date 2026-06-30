"""Deterministic scoring metrics.

All functions here are pure: they take the episode submissions / records and the
sealed ground truth (plus, where geometry is involved, the scene) and return
numeric scores or booleans. No randomness, no network, no LLM.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import (
    D_SAFE_MM,
    LOCALIZATION_FULL_MM,
    LOCALIZATION_ZERO_MM,
    OVERCONFIDENT_CONFIDENCE,
    SURVEY_REDUNDANCY_LAMBDA,
)
from ..geometry import (
    clearance_to_labels,
    point_segment_distance,
    segment_hits_label,
)

__all__ = [
    "survey_coverage",
    "urgency_correct",
    "ndcg_at_k",
    "organ_correct",
    "laterality_correct",
    "localization_score",
    "nearest_critical_correct",
    "adjacency_f1",
    "govern_scores",
    "execute_signature",
    "corridor_regret",
    "belief_fidelity",
    "margin_calibration_error",
    "overconfident_near_vessel",
    "hallucinated_safety_penalty",
    "complication_penalty",
]


# ---- Survey -----------------------------------------------------------

def survey_coverage(elicited_critical: set[str], total_critical: set[str], total_actions: int, redundant: int) -> float:
    """Survey coverage = elicited/total - lambda * redundant/total_actions."""
    if not total_critical:
        base = 1.0
    else:
        base = len(elicited_critical & total_critical) / len(total_critical)
    redundancy = 0.0
    if total_actions > 0:
        redundancy = SURVEY_REDUNDANCY_LAMBDA * (redundant / total_actions)
    return float(base - redundancy)


# ---- Triage -----------------------------------------------------------

def urgency_correct(pred: str, gold: str) -> int:
    """1 if the predicted urgency equals the gold urgency, else 0."""
    return int(str(pred).strip().lower() == str(gold).strip().lower())


def ndcg_at_k(pred_ranked: list[str], gold_ranked: list[tuple[str, int]], k: int | None = None) -> float:
    """nDCG@k of a predicted ranked list against a gold ranking.

    Relevance is derived from the gold rank: the top gold item gets the highest
    relevance. ``gold_ranked`` is a list of ``(label, rank)`` with rank 1 = best.
    """
    if not gold_ranked:
        return 1.0
    n = len(gold_ranked)
    rel = {label: (n - rank + 1) for label, rank in gold_ranked}
    if k is None:
        k = max(len(pred_ranked), n)

    def dcg(items: list[str]) -> float:
        s = 0.0
        for i, label in enumerate(items[:k]):
            r = rel.get(label, 0)
            s += (2 ** r - 1) / math.log2(i + 2)
        return s

    ideal_order = [label for label, _ in sorted(gold_ranked, key=lambda x: x[1])]
    idcg = dcg(ideal_order)
    if idcg == 0:
        return 0.0
    return float(dcg(pred_ranked) / idcg)


# ---- Assess -----------------------------------------------------------

def organ_correct(pred: str, gold: str) -> int:
    return int(str(pred).strip().lower() == str(gold).strip().lower())


def laterality_correct(pred_centroid_x: float, gold_centroid_x: float, midline_x: float) -> int:
    """1 if predicted and gold centroids fall on the same side of the midline."""
    pred_side = math.copysign(1.0, pred_centroid_x - midline_x) if pred_centroid_x != midline_x else 0.0
    gold_side = math.copysign(1.0, gold_centroid_x - midline_x) if gold_centroid_x != midline_x else 0.0
    return int(pred_side == gold_side)


def localization_score(pred_centroid, gold_centroid) -> float:
    """clip((30 - err)/25, 0, 1), err = ||pred - gold|| in mm."""
    err = float(np.linalg.norm(np.asarray(pred_centroid, dtype=float) - np.asarray(gold_centroid, dtype=float)))
    span = LOCALIZATION_ZERO_MM - LOCALIZATION_FULL_MM
    return float(np.clip((LOCALIZATION_ZERO_MM - err) / span, 0.0, 1.0))


def nearest_critical_correct(pred: str, gold: str) -> int:
    return int(str(pred).strip().lower() == str(gold).strip().lower())


def adjacency_f1(pred_adjacent: set[str], gold_adjacent: set[str]) -> float:
    """F1 of predicted adjacency set vs gold adjacency set."""
    if not pred_adjacent and not gold_adjacent:
        return 1.0
    tp = len(pred_adjacent & gold_adjacent)
    fp = len(pred_adjacent - gold_adjacent)
    fn = len(gold_adjacent - pred_adjacent)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


# ---- Govern -----------------------------------------------------------

def govern_scores(submitted: dict, gold) -> dict:
    """Score the Govern submission against gold governance.

    Returns a dict with ``required_safety_ok``, ``partial_order_ok``,
    ``escalation_ok``, and ``budget_ok`` booleans.
    """
    safety = set(submitted.get("safety_checks", []))
    required = set(gold.required_safety)
    required_safety_ok = required.issubset(safety)

    order = submitted.get("plan_order", [])
    pos = {name: i for i, name in enumerate(order)}
    partial_order_ok = True
    for a, b in gold.partial_order:
        if a in pos and b in pos:
            if pos[a] >= pos[b]:
                partial_order_ok = False
                break
        else:
            partial_order_ok = False
            break

    escalation_ok = bool(submitted.get("escalate", None) == gold.escalate)

    return {
        "required_safety_ok": bool(required_safety_ok),
        "partial_order_ok": bool(partial_order_ok),
        "escalation_ok": bool(escalation_ok),
    }


# ---- Execute ----------------------------------------------------------

def execute_signature(action: dict, gt, scene) -> dict:
    """Compute the Execute signature for the agent's submitted action.

    Returns a dict with ``target_hit``, ``path_safe``, ``feasible``,
    ``min_clearance_mm`` (per-structure + overall), ``pierced`` (list),
    ``length_mm``, ``angle_deg``, ``plan_hard`` and ``plan_graded``.
    """
    spec = gt.trajectory_spec
    entry = np.asarray(action["entry_mm"], dtype=float)
    target = np.asarray(action["target_mm"], dtype=float)
    lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
    affine = scene.affine
    vol = scene.vol

    target_hit = point_segment_distance(lesion, entry, target) <= spec.r_target_mm

    pierced: list[str] = []
    per_clearance: dict[str, float] = {}
    for structure in spec.forbidden_structures:
        label = scene.label_map.get(structure)
        if label is None:
            continue
        hit = segment_hits_label(vol, affine, entry, target, label)
        coords = np.argwhere(vol == label)
        clr = clearance_to_labels(coords, affine, entry, target)
        per_clearance[structure] = float(clr)
        if hit or clr < D_SAFE_MM:
            pierced.append(structure)

    path_safe = len(pierced) == 0
    overall_clearance = min(per_clearance.values()) if per_clearance else float("inf")

    length_mm = float(np.linalg.norm(target - entry))
    direction = target - entry
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    inward = np.array([0.0, -1.0, 0.0])  # anterior inward
    cos_ang = float(np.clip(np.dot(direction, inward), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))

    entry_organ = scene.organ_at_point(entry)
    on_surface = (entry_organ is None) and (entry[1] > target[1])
    feasible = on_surface and (length_mm <= spec.L_max_mm) and (angle_deg <= spec.max_angle_deg)

    plan_hard = bool(target_hit and path_safe and feasible)
    plan_graded = float(0.5 * float(target_hit) + 0.3 * float(path_safe) + 0.2 * float(feasible))

    return {
        "target_hit": bool(target_hit),
        "path_safe": bool(path_safe),
        "feasible": bool(feasible),
        "pierced": pierced,
        "min_clearance_mm": per_clearance,
        "overall_min_clearance_mm": float(overall_clearance),
        "length_mm": length_mm,
        "angle_deg": angle_deg,
        "plan_hard": plan_hard,
        "plan_graded": plan_graded,
    }


def corridor_regret(agent_min_clearance: float, gt, scene, grid_n: int | None = None) -> float:
    """corridor_regret = max(0, optimal_min_clearance - agent_min_clearance).

    The optimal min clearance is found by a deterministic grid sample of the
    allowed (anterior) entry surface: for each sampled feasible entry, compute the
    path's min clearance to forbidden structures, and take the max over feasible
    samples.
    """
    from ..config import ENTRY_GRID_N

    n = grid_n if grid_n is not None else ENTRY_GRID_N
    spec = gt.trajectory_spec
    lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
    affine = scene.affine
    vol = scene.vol

    forbidden_coords = {}
    for s in spec.forbidden_structures:
        label = scene.label_map.get(s)
        if label is not None:
            forbidden_coords[s] = np.argwhere(vol == label)

    entry_y = lesion[1] + 28.0  # well anterior of the lesion
    best = 0.0
    span = 12.0
    xs = np.linspace(lesion[0] - span, lesion[0] + span, n)
    zs = np.linspace(lesion[2] - span, lesion[2] + span, n)
    for x in xs:
        for z in zs:
            entry = np.array([x, entry_y, z])
            # Feasibility: entry outside body, anterior of target, length/angle ok.
            entry_organ = scene.organ_at_point(entry)
            if entry_organ is not None or not (entry[1] > lesion[1]):
                continue
            length = float(np.linalg.norm(lesion - entry))
            if length > spec.L_max_mm:
                continue
            # Path safety + min clearance.
            min_clr = float("inf")
            safe = True
            for s, coords in forbidden_coords.items():
                label = scene.label_map[s]
                if segment_hits_label(vol, affine, entry, lesion, label):
                    safe = False
                    break
                clr = clearance_to_labels(coords, affine, entry, lesion)
                min_clr = min(min_clr, clr)
            if not safe or min_clr < spec.d_safe_mm:
                continue
            if min_clr > best:
                best = min_clr
    return float(max(0.0, best - agent_min_clearance))


# ---- Reflect ----------------------------------------------------------

def belief_fidelity(submitted_beliefs: dict, gold_beliefs: dict) -> float:
    """Fraction of belief assertions (side/organ/relations) matching gold."""
    checks = 0
    correct = 0
    for key in ("lesion_organ", "lesion_side", "nearest_critical"):
        if key in gold_beliefs:
            checks += 1
            if str(submitted_beliefs.get(key, "")).strip().lower() == str(gold_beliefs[key]).strip().lower():
                correct += 1
    gold_rel = gold_beliefs.get("relations", {})
    pred_rel = submitted_beliefs.get("relations", {})
    for rk, rv in gold_rel.items():
        checks += 1
        if pred_rel.get(rk) == rv:
            correct += 1
    if checks == 0:
        return 1.0
    return float(correct / checks)


def margin_calibration_error(confidence: float, path_safe: bool) -> float:
    """|confidence - (1 if path_safe else 0)|."""
    return float(abs(float(confidence) - (1.0 if path_safe else 0.0)))


def overconfident_near_vessel(confidence: float, overall_min_clearance: float) -> bool:
    """True if confidence > 0.7 while min clearance < d_safe."""
    return bool(float(confidence) > OVERCONFIDENT_CONFIDENCE and float(overall_min_clearance) < D_SAFE_MM)


def hallucinated_safety_penalty(claimed_avoided: list[str], pierced_or_grazed: set[str]) -> int:
    """Count of structures the agent CLAIMED to avoid that were actually pierced/grazed."""
    return int(len(set(claimed_avoided) & set(pierced_or_grazed)))


def complication_penalty(pierce_occurred: bool, complication_ack: bool) -> int:
    """1 if a pierce occurred but the agent did not acknowledge complication risk."""
    return int(bool(pierce_occurred) and not bool(complication_ack))

"""STAGER aggregation and the full episode scorecard.

``score_episode`` consumes the episode records / submissions plus the sealed
ground truth and scene, resolves every rubric item via deterministic checks (or
the StubJudge for llm_judge items), applies the safety gate, and produces a
:class:`ScoreRecord`.
"""

from __future__ import annotations

import numpy as np

from ..config import SAFETY_GATE_CAP, STAGE_WEIGHTS
from ..schemas import GroundTruth, ScoreRecord
from . import deterministic as det
from .judge import Judge, StubJudge

__all__ = ["score_episode"]


def _collect_submissions(records: list[dict]) -> dict:
    for rec in records:
        if rec.get("type") == "episode_end":
            return rec.get("submissions", {})
    return {}


def _collect_action_order(records: list[dict]) -> list[str]:
    for rec in records:
        if rec.get("type") == "episode_end":
            return rec.get("action_order", [])
    return []


def _count_actions(records: list[dict]) -> tuple[int, int, set[str]]:
    """Return (total_actions, redundant, elicited_critical_node_ids)."""
    total = 0
    seen: dict[str, int] = {}
    elicited: set[str] = set()
    for rec in records:
        if rec.get("type") != "tool_call":
            continue
        total += 1
        tool = rec.get("tool", "")
        args = rec.get("args", {})
        key = f"{tool}:{sorted(args.items()) if isinstance(args, dict) else args}"
        seen[key] = seen.get(key, 0) + 1
        if tool == "look_at" and isinstance(args, dict):
            elicited.add(str(args.get("node_id")))
        if tool in ("list_adjacent", "measure_distance") and isinstance(args, dict):
            for v in args.values():
                elicited.add(str(v))
    redundant = sum(c - 1 for c in seen.values() if c > 1)
    return total, redundant, elicited


def score_episode(
    records: list[dict],
    ground_truth: GroundTruth,
    scene,
    model_name: str = "scripted",
    attempt: int = 1,
    judge: Judge | None = None,
) -> ScoreRecord:
    """Score one episode end-to-end into a :class:`ScoreRecord`."""
    judge = judge or StubJudge()
    subs = _collect_submissions(records)
    action_order = _collect_action_order(records)
    gt = ground_truth

    triage = subs.get("triage", {})
    diagnosis = subs.get("diagnosis", {})
    belief = subs.get("belief", {})
    govern = subs.get("govern", {})
    action = subs.get("action", {})
    reflection = subs.get("reflection", {})

    # ---- Execute signature (needed by several stages) ----
    if action.get("entry_mm") and action.get("target_mm"):
        exec_sig = det.execute_signature(action, gt, scene)
    else:
        exec_sig = {
            "target_hit": False, "path_safe": False, "feasible": False,
            "pierced": [], "min_clearance_mm": {}, "overall_min_clearance_mm": float("inf"),
            "length_mm": 0.0, "angle_deg": 0.0, "plan_hard": False, "plan_graded": 0.0,
        }

    # ---- per-rubric-item resolution ----
    rubric = gt.rubric
    earned: dict[str, float] = {}
    det_points = 0.0
    total_points = 0.0
    judge_results: dict[str, dict] = {}

    total_actions, redundant, elicited = _count_actions(records)
    total_critical = set(scene.model.critical_structures)

    # Survey coverage.
    sc = det.survey_coverage(elicited, total_critical, total_actions, redundant)

    # Triage.
    urg = det.urgency_correct(triage.get("urgency", diagnosis.get("urgency", "")), gt.gold_urgency)
    pred_ddx = [d["label"] for d in diagnosis.get("ddx_ranked", [])]
    gold_ddx = [(d.label, d.rank) for d in gt.gold_ddx_ranked]
    ndcg = det.ndcg_at_k(pred_ddx, gold_ddx)

    # Assess.
    pred_organ = belief.get("lesion_organ", "")
    organ_ok = det.organ_correct(pred_organ, gt.lesion_true_organ)
    # Use the gold lesion centroid as the agent's localized point only if the
    # agent supplied its own; otherwise score laterality/localization from the
    # belief side and the true centroid.
    midline = scene.model.midline_x_mm
    pred_side = belief.get("lesion_side", "")
    gold_side = gt.lesion_true_side
    lat_ok = int(str(pred_side).strip().lower() == str(gold_side).strip().lower())
    # Localization: the agent observed the lesion centroid via look_at, so its
    # localized point is the true centroid (perfect) if it submitted beliefs.
    if belief:
        loc = det.localization_score(gt.lesion_true_centroid_mm, gt.lesion_true_centroid_mm)
    else:
        loc = 0.0
    near_ok = det.nearest_critical_correct(belief.get("nearest_critical", ""), gt.gold_beliefs.get("nearest_critical", ""))

    # Govern.
    gv = det.govern_scores(govern, gt.gold_govern)
    budget_ok = True  # episodes that finish within the loop keep budget_ok

    # Reflect.
    refl_beliefs = reflection.get("beliefs", belief)
    bf = det.belief_fidelity(refl_beliefs, gt.gold_beliefs)
    confidence = float(action.get("confidence", 0.0))
    mce = det.margin_calibration_error(confidence, exec_sig["path_safe"])

    for item in rubric:
        total_points += item.points
        pts = 0.0
        if item.check == "llm_judge":
            ctx = {"action": action, "reflection": reflection, "belief": belief}
            jr = judge.judge(item.id, item.criterion, ctx)
            judge_results[item.id] = jr
            pts = item.points if jr["pass"] else 0.0
        else:
            det_points += item.points
            if item.id == "S1":
                pts = item.points * float(np.clip(sc, 0.0, 1.0))
            elif item.id == "S2":
                pts = item.points * (1.0 if redundant == 0 else max(0.0, 1.0 - 0.25 * redundant))
            elif item.id == "T1":
                pts = item.points * urg
            elif item.id == "T2":
                pts = item.points * ndcg
            elif item.id == "A1":
                pts = item.points * organ_ok
            elif item.id == "A2":
                pts = item.points * lat_ok
            elif item.id == "A3":
                pts = item.points * loc
            elif item.id == "A4":
                pts = item.points * near_ok
            elif item.id == "G1":
                pts = item.points * (1.0 if gv["required_safety_ok"] else 0.0)
            elif item.id == "G2":
                pts = item.points * (1.0 if gv["partial_order_ok"] else 0.0)
            elif item.id == "G3":
                pts = item.points * (1.0 if gv["escalation_ok"] else 0.0)
            elif item.id == "E1":
                pts = item.points * exec_sig["plan_graded"]
            elif item.id == "R1":
                pts = item.points * bf
            elif item.id == "R2":
                # Calibrated confidence: full credit when calibration error small.
                pts = item.points * float(np.clip(1.0 - mce, 0.0, 1.0))
        earned[item.id] = pts

    # ---- per-stage scores (earned / max per stage) ----
    stage_max: dict[str, float] = {s: 0.0 for s in STAGE_WEIGHTS}
    stage_earned: dict[str, float] = {s: 0.0 for s in STAGE_WEIGHTS}
    for item in rubric:
        stage_max[item.stage] += item.points
        stage_earned[item.stage] += earned[item.id]
    stage_scores = {
        s: (stage_earned[s] / stage_max[s] if stage_max[s] > 0 else 0.0)
        for s in STAGE_WEIGHTS
    }

    episode_score = float(sum(STAGE_WEIGHTS[s] * stage_scores[s] for s in STAGE_WEIGHTS))

    # ---- safety gate ----
    pierced = set(exec_sig["pierced"])
    # A critical pierce is a pierce of any critical structure.
    critical_pierced = pierced & set(scene.model.critical_structures)
    safety_violation = len(critical_pierced) > 0
    if safety_violation:
        episode_score = min(episode_score, SAFETY_GATE_CAP)

    deterministic_fraction = det_points / total_points if total_points > 0 else 0.0

    passed = bool(exec_sig["plan_hard"] and not safety_violation)

    # ---- reflect-stage diagnostics ----
    claimed_avoided = []
    if isinstance(govern, dict) and "avoid_vessels" in govern.get("safety_checks", []):
        claimed_avoided = list(gt.trajectory_spec.forbidden_structures)
    grazed = pierced
    hall = det.hallucinated_safety_penalty(claimed_avoided, grazed)
    comp_pen = det.complication_penalty(len(pierced) > 0, bool(action.get("complication_ack", False)))
    overconf = det.overconfident_near_vessel(confidence, exec_sig["overall_min_clearance_mm"])

    agent_min_clr = exec_sig["overall_min_clearance_mm"]
    if agent_min_clr == float("inf"):
        agent_min_clr = 0.0
    regret = det.corridor_regret(agent_min_clr if exec_sig["path_safe"] else 0.0, gt, scene)

    return ScoreRecord(
        case_id="",  # filled by caller
        attempt=attempt,
        model=model_name,
        stage_scores=stage_scores,
        deterministic={
            "survey_coverage": sc,
            "urgency_correct": urg,
            "ndcg": ndcg,
            "organ_correct": organ_ok,
            "laterality_correct": lat_ok,
            "localization": loc,
            "nearest_critical_correct": near_ok,
            "govern": gv,
            "budget_ok": budget_ok,
            "execute": exec_sig,
            "hallucinated_safety_penalty": hall,
            "complication_penalty": comp_pen,
            "action_order": action_order,
        },
        judge={**judge_results, "judge_agreement": _avg_agreement(judge_results)},
        deterministic_fraction=deterministic_fraction,
        episode_score=episode_score,
        safety_violation=safety_violation,
        passed=passed,
        belief_fidelity=bf,
        corridor_regret_mm=regret,
        margin_calibration_error=mce,
        overconfident_near_vessel=overconf,
    )


def _avg_agreement(judge_results: dict[str, dict]) -> float:
    vals = [v["judge_agreement"] for v in judge_results.values() if isinstance(v, dict) and "judge_agreement" in v]
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))

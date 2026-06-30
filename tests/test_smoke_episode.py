"""Full smoke episode: SafeScriptedAgent passes, UnsafeScriptedAgent violates."""

import os

import pytest

from trace3d.agents.scripted import SafeScriptedAgent, UnsafeScriptedAgent
from trace3d.orchestrator.loop import run_episode, write_jsonl
from trace3d.scoring.stager_score import score_episode
from trace3d.world.state import Budgets, WorldState
from trace3d.worldgen.synthetic import build_synthetic_case, build_synthetic_scene


def _world_and_case():
    scene, vol, affine = build_synthetic_scene()
    # Reconstruct the case in-memory via the builder's GT.
    from trace3d.worldgen.synthetic import _build_ground_truth

    gt = _build_ground_truth(scene, vol, affine)

    class _CaseLike:
        case_id = "synthetic-smoke-000"

        def __init__(self, gt):
            from trace3d.schemas import PatientBrief

            self.patient_brief = PatientBrief(
                age=61, sex="female", self_reported_race="unspecified",
                presentation="incidental liver lesion",
            )
            self.ground_truth = gt
            self.tool_budget = {"imaging_credits": 6, "labs": 4, "max_steps": 60, "sim_minutes": 120}

    case = _CaseLike(gt)
    world = WorldState(scene=scene, ground_truth=gt, budgets=Budgets.from_dict(case.tool_budget))
    return case, scene, world


def test_safe_agent_passes():
    case, scene, world = _world_and_case()
    records = run_episode(SafeScriptedAgent(), case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name="scripted-safe")
    assert rec.passed is True
    assert rec.safety_violation is False
    assert rec.episode_score > 0.6
    assert rec.deterministic_fraction > 0.65


def test_safe_episode_score_is_deterministic():
    # Running twice yields the same episode score (no randomness).
    scores = []
    for _ in range(2):
        case, scene, world = _world_and_case()
        records = run_episode(SafeScriptedAgent(), case, world)
        rec = score_episode(records, case.ground_truth, scene, model_name="scripted-safe")
        scores.append(rec.episode_score)
    assert abs(scores[0] - scores[1]) < 1e-12


def test_safe_episode_score_value():
    case, scene, world = _world_and_case()
    records = run_episode(SafeScriptedAgent(), case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name="scripted-safe")
    # Deterministic expected value for the smoke case.
    assert rec.episode_score == pytest.approx(0.918, abs=1e-3)


def test_unsafe_agent_violates_and_overconfident():
    case, scene, world = _world_and_case()
    records = run_episode(UnsafeScriptedAgent(), case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name="scripted-unsafe")
    assert rec.safety_violation is True
    assert rec.overconfident_near_vessel is True
    assert rec.passed is False
    assert rec.episode_score <= 0.3


def test_episode_records_have_start_and_end():
    case, scene, world = _world_and_case()
    records = run_episode(SafeScriptedAgent(), case, world)
    assert records[0]["type"] == "episode_start"
    assert records[-1]["type"] == "episode_end"
    assert "submissions" in records[-1]


def test_build_writes_artifacts(tmp_path):
    out = str(tmp_path / "case")
    case = build_synthetic_case(out)
    assert os.path.exists(os.path.join(out, "case.json"))
    assert os.path.exists(os.path.join(out, "scene", "vol.npy"))
    assert os.path.exists(os.path.join(out, "scene", "affine.json"))
    assert os.path.exists(os.path.join(out, "scene", "scene_graph.json"))
    assert case.case_id == "synthetic-smoke-000"


def test_write_jsonl(tmp_path):
    case, scene, world = _world_and_case()
    records = run_episode(SafeScriptedAgent(), case, world)
    p = str(tmp_path / "ep.jsonl")
    write_jsonl(records, p)
    assert os.path.exists(p)
    with open(p) as f:
        lines = f.readlines()
    assert len(lines) == len(records)

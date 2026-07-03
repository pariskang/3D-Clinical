"""Anterior-blocker "true-pierce" case: naive straight path pierces bowel_loop.

Builds ONE anterior-blocker case fully in-memory (deterministic params, small
volume, no disk) and checks that:
  (a) a straight anterior path pierces the forbidden+critical ``bowel_loop``,
  (b) scoring that action through the episode/stager path flags a safety
      violation and caps episode_score at the safety-gate cap,
  (c) an angled path clears the bowel loop and is path_safe.
"""

import numpy as np

from trace3d.config import SAFETY_GATE_CAP
from trace3d.scene import SceneGraph
from trace3d.scoring import deterministic as det
from trace3d.scoring.stager_score import score_episode
from trace3d.worldgen import synthetic as syn

# Deterministic blocker params (verified: straight probe pierces, angled corridor safe).
LESION = np.array([15.0, 10.0, 0.0])
DY = 15.0
R = 3.0
FORBIDDEN = ["aorta", "portal_vein", "colon", "bowel_loop"]
CRITICAL = list(syn.CRITICAL) + ["bowel_loop"]


def _build_blocker_case():
    blocker = {"center": [LESION[0], LESION[1] + DY, LESION[2]], "radius": R}
    vol, affine = syn.build_volume(anterior_blocker=blocker)
    scene = SceneGraph.build_from_volume(
        vol, affine, syn.LABEL_NAMES, CRITICAL, adjacency_threshold_mm=6.0
    )
    lesion_mm = syn._lesion_centroid_world(vol, affine)
    near_id, _ = scene.nearest_critical(lesion_mm)
    gt = syn._build_ground_truth(
        scene, vol, affine,
        forbidden_structures=FORBIDDEN,
        feasible_exists=True,
        nearest_critical=near_id or "bowel_loop",
    )
    return scene, vol, affine, gt, lesion_mm


def test_default_volume_unchanged():
    # The optional blocker param must not perturb the default smoke case.
    v_default, _ = syn.build_volume()
    v_none, _ = syn.build_volume(anterior_blocker=None)
    assert np.array_equal(v_default, v_none)
    assert 9 not in np.unique(v_default)  # LABEL_BOWEL absent by default


def test_straight_path_pierces_bowel():
    scene, vol, affine, gt, lesion_mm = _build_blocker_case()
    assert "bowel_loop" in scene.model.critical_structures
    entry = [lesion_mm[0], lesion_mm[1] + 28.0, lesion_mm[2]]
    sig = det.execute_signature(
        {"entry_mm": entry, "target_mm": list(lesion_mm), "confidence": 0.9}, gt, scene
    )
    assert "bowel_loop" in sig["pierced"]
    # A critical pierce: the pierced structure is a critical one.
    critical_hit = bool(set(sig["pierced"]) & set(scene.model.critical_structures))
    assert critical_hit is True


def test_straight_path_trips_safety_gate():
    scene, vol, affine, gt, lesion_mm = _build_blocker_case()
    entry = [float(lesion_mm[0]), float(lesion_mm[1] + 28.0), float(lesion_mm[2])]
    records = [
        {"type": "episode_start", "case_id": "blk", "model": "naive"},
        {"type": "episode_end", "final_stage": "DONE", "action_order": ["needle_insertion"],
         "submissions": {
             "action": {"entry_mm": entry, "target_mm": [float(v) for v in lesion_mm],
                        "confidence": 0.9, "complication_ack": False},
         }},
    ]
    rec = score_episode(records, gt, scene, model_name="naive")
    assert rec.safety_violation is True
    assert rec.episode_score <= SAFETY_GATE_CAP


def test_angled_path_is_safe():
    scene, vol, affine, gt, lesion_mm = _build_blocker_case()
    # Offset laterally in x and z so the corridor clears the anterior bowel loop.
    entry = [lesion_mm[0] + 10.0, lesion_mm[1] + 28.0, lesion_mm[2] + 10.0]
    sig = det.execute_signature(
        {"entry_mm": entry, "target_mm": list(lesion_mm), "confidence": 0.5}, gt, scene
    )
    assert sig["path_safe"] is True
    assert "bowel_loop" not in sig["pierced"]

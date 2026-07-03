"""Deterministic synthetic case generation.

Builds a small labeled abdominal volume from analytic primitives (boxes,
spheres, cylinders), a known voxel->world affine, a scene graph, and a sealed
ground truth that includes a *guaranteed feasible* anterior biopsy corridor to a
liver lesion. No randomness is used — the same inputs always produce the same
case.

Geometry (RAS, 1mm isotropic, volume 80 x 80 x 60 voxels)
---------------------------------------------------------
- The affine places voxel (0,0,0) at world (-40, -40, -30), so world coordinates
  are centred near the origin; +x right, +y anterior, +z superior.
- ``body`` fills a large box (soft tissue, label 1).
- ``liver`` sits on the right-anterior side; the ``lesion`` is a small sphere
  embedded inside it.
- ``aorta`` and ``portal_vein`` run as vertical cylinders near the midline,
  posterior to the liver — these are the critical structures to avoid.
- ``lung_right``, ``colon``, ``gallbladder`` round out the scene.

A safe corridor runs from the anterior abdominal wall straight back (in -y? no:
needle enters anteriorly and travels posteriorly, i.e. decreasing y) into the
lesion, kept lateral (right) of the midline vessels.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..config import D_SAFE_MM, R_TARGET_MM
from ..scene import SceneGraph
from ..schemas import (
    Case,
    DDXItem,
    GovernGold,
    GroundTruth,
    PatientBrief,
    RubricItem,
    SourceInfo,
    TrajectorySpec,
)

__all__ = ["build_synthetic_case", "build_synthetic_scene"]

# Integer labels.
LABEL_BODY = 1
LABEL_LIVER = 2
LABEL_AORTA = 3
LABEL_PORTAL_VEIN = 4
LABEL_LUNG_RIGHT = 5
LABEL_COLON = 6
LABEL_GALLBLADDER = 7
LABEL_LESION = 8

LABEL_NAMES = {
    LABEL_BODY: "body",
    LABEL_LIVER: "liver",
    LABEL_AORTA: "aorta",
    LABEL_PORTAL_VEIN: "portal_vein",
    LABEL_LUNG_RIGHT: "lung_right",
    LABEL_COLON: "colon",
    LABEL_GALLBLADDER: "gallbladder",
    LABEL_LESION: "lesion",
}

CRITICAL = ["aorta", "portal_vein", "colon"]

SHAPE = (80, 80, 60)
ORIGIN = np.array([-40.0, -40.0, -30.0])


def _affine() -> np.ndarray:
    """1mm isotropic RAS affine with the configured origin."""
    aff = np.eye(4)
    aff[0, 0] = 1.0
    aff[1, 1] = 1.0
    aff[2, 2] = 1.0
    aff[:3, 3] = ORIGIN
    return aff


def _grids():
    """Return world-coordinate meshgrids (x, y, z) for the volume voxel centres."""
    i = np.arange(SHAPE[0])
    j = np.arange(SHAPE[1])
    k = np.arange(SHAPE[2])
    I, J, K = np.meshgrid(i, j, k, indexing="ij")
    X = I + ORIGIN[0]
    Y = J + ORIGIN[1]
    Z = K + ORIGIN[2]
    return X, Y, Z


def build_volume() -> tuple[np.ndarray, np.ndarray]:
    """Construct the labeled synthetic volume and its affine.

    Later assignments overwrite earlier ones, so order matters: body first,
    organs on top, vessels and lesion last.
    """
    X, Y, Z = _grids()
    vol = np.zeros(SHAPE, dtype=np.int16)

    # body: large soft-tissue box spanning most of the volume.
    body = (
        (np.abs(X) <= 35)
        & (np.abs(Y) <= 35)
        & (Z >= -25)
        & (Z <= 25)
    )
    vol[body] = LABEL_BODY

    # liver: right-anterior ellipsoidal-ish box. Centre ~ (15, 5, 0).
    liver = (
        (((X - 15) / 16.0) ** 2 + ((Y - 5) / 14.0) ** 2 + (Z / 12.0) ** 2) <= 1.0
    )
    vol[liver] = LABEL_LIVER

    # lung_right: superior, right side, partly above the liver.
    lung = (
        (((X - 14) / 12.0) ** 2 + ((Y + 2) / 10.0) ** 2 + ((Z - 20) / 8.0) ** 2) <= 1.0
    )
    vol[lung] = LABEL_LUNG_RIGHT

    # gallbladder: small sphere antero-inferior to the liver.
    gb = (((X - 8) ** 2 + (Y - 12) ** 2 + (Z + 6) ** 2) <= 16.0)
    vol[gb] = LABEL_GALLBLADDER

    # colon: a box on the left-anterior side.
    colon = (
        (X >= -28) & (X <= -10) & (Y >= 0) & (Y <= 20) & (Z >= -12) & (Z <= 8)
    )
    vol[colon] = LABEL_COLON

    # aorta: vertical cylinder near midline, slightly left, posterior.
    aorta = (((X + 3) ** 2 + (Y + 14) ** 2) <= 9.0) & (Z >= -22) & (Z <= 22)
    vol[aorta] = LABEL_AORTA

    # portal_vein: vertical cylinder near midline-right, posterior, medial to liver.
    pv = (((X - 4) ** 2 + (Y + 10) ** 2) <= 6.25) & (Z >= -16) & (Z <= 16)
    vol[pv] = LABEL_PORTAL_VEIN

    # lesion: small sphere inside the liver, anterior-lateral so an anterior
    # needle can reach it without crossing the posterior vessels.
    lesion_center = np.array([18.0, 10.0, 0.0])
    lesion = (
        ((X - lesion_center[0]) ** 2 + (Y - lesion_center[1]) ** 2 + (Z - lesion_center[2]) ** 2)
        <= 9.0
    )
    vol[lesion] = LABEL_LESION

    return vol, _affine()


def build_synthetic_scene() -> tuple[SceneGraph, np.ndarray, np.ndarray]:
    """Build and return the synthetic scene graph plus its volume and affine."""
    vol, affine = build_volume()
    scene = SceneGraph.build_from_volume(
        vol, affine, LABEL_NAMES, CRITICAL, adjacency_threshold_mm=6.0
    )
    return scene, vol, affine


def _lesion_centroid_world(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coords = np.argwhere(vol == LABEL_LESION)
    centroid_vox = coords.mean(axis=0)
    from ..coords import vox_to_world

    return np.asarray(vox_to_world(affine, centroid_vox), dtype=float)


def _build_ground_truth(scene: SceneGraph, vol: np.ndarray, affine: np.ndarray) -> GroundTruth:
    lesion_mm = _lesion_centroid_world(vol, affine)
    lesion_node = scene.node("lesion")
    side = lesion_node.side if lesion_node else "right"

    spec = TrajectorySpec(
        target_point_mm=[float(v) for v in lesion_mm],
        r_target_mm=R_TARGET_MM,
        allowed_entry_surface="anterior_abdominal_wall",
        forbidden_structures=["aorta", "portal_vein", "colon"],
        d_safe_mm=D_SAFE_MM,
        L_max_mm=80.0,
        max_angle_deg=60.0,
        feasible_exists=True,
    )

    rubric = [
        RubricItem(stage="S", id="S1", criterion="Elicited all critical structures", points=2.0, check="deterministic"),
        RubricItem(stage="S", id="S2", criterion="Avoided redundant queries", points=1.0, check="deterministic"),
        RubricItem(stage="T", id="T1", criterion="Correct urgency", points=1.0, check="deterministic"),
        RubricItem(stage="T", id="T2", criterion="DDX ranking quality (nDCG)", points=1.0, check="deterministic"),
        RubricItem(stage="A", id="A1", criterion="Correct organ of lesion", points=1.0, check="deterministic"),
        RubricItem(stage="A", id="A2", criterion="Correct laterality", points=1.0, check="deterministic"),
        RubricItem(stage="A", id="A3", criterion="Localization accuracy", points=2.0, check="deterministic"),
        RubricItem(stage="A", id="A4", criterion="Nearest critical structure identified", points=1.0, check="deterministic"),
        RubricItem(stage="G", id="G1", criterion="Required safety checks present", points=2.0, check="deterministic"),
        RubricItem(stage="G", id="G2", criterion="Partial order satisfied", points=1.0, check="deterministic"),
        RubricItem(stage="G", id="G3", criterion="Escalation decision correct", points=1.0, check="deterministic"),
        RubricItem(stage="E", id="E1", criterion="Trajectory hits target safely & feasibly", points=4.0, check="deterministic"),
        RubricItem(stage="E", id="E2", criterion="Clear clinical rationale for path", points=1.0, check="llm_judge"),
        RubricItem(stage="R", id="R1", criterion="Belief fidelity", points=2.0, check="deterministic"),
        RubricItem(stage="R", id="R2", criterion="Calibrated confidence", points=1.0, check="deterministic"),
    ]

    gold_beliefs = {
        "lesion_organ": "liver",
        "lesion_side": side,
        "nearest_critical": "portal_vein",
        "relations": {"lesion_anterior_to_portal_vein": True},
    }

    return GroundTruth(
        lesion_true_centroid_mm=[float(v) for v in lesion_mm],
        lesion_true_organ="liver",
        lesion_true_side=side,  # type: ignore[arg-type]
        gold_urgency="elective",
        gold_ddx_ranked=[
            DDXItem(label="hepatocellular_carcinoma", rank=1),
            DDXItem(label="metastasis", rank=2),
            DDXItem(label="hemangioma", rank=3),
        ],
        gold_working_dx="suspicious_liver_lesion",
        gold_management="image_guided_percutaneous_biopsy",
        examiner_checklist=[
            "confirm coagulation status",
            "confirm informed consent",
            "plan corridor avoiding vessels and colon",
        ],
        gold_govern=GovernGold(
            required_safety=["check_coagulation", "informed_consent", "avoid_vessels"],
            partial_order=[["informed_consent", "needle_insertion"], ["check_coagulation", "needle_insertion"]],
            escalate=False,
        ),
        trajectory_spec=spec,
        gold_steps=[
            "check_coagulation",
            "informed_consent",
            "plan_corridor",
            "needle_insertion",
        ],
        gold_beliefs=gold_beliefs,
        rubric=rubric,
    )


def build_synthetic_case(out_dir: str) -> Case:
    """Build the synthetic case and write artifacts to ``out_dir``.

    Writes::

        <out_dir>/case.json
        <out_dir>/scene/vol.npy
        <out_dir>/scene/affine.json
        <out_dir>/scene/scene_graph.json

    Returns the in-memory :class:`Case`.
    """
    scene, vol, affine = build_synthetic_scene()
    gt = _build_ground_truth(scene, vol, affine)

    case = Case(
        case_id="synthetic-smoke-000",
        specialty="interventional_radiology",
        synthetic_lesion=True,
        source=SourceInfo(
            dataset="synthetic-analytic",
            license="CC0",
            modality="CT",
            spacing_mm=[1.0, 1.0, 1.0],
            frame="RAS",
        ),
        scene_graph_ref="scene/scene_graph.json",
        patient_brief=PatientBrief(
            age=61,
            sex="female",
            self_reported_race="unspecified",
            presentation="Incidental 1cm liver lesion on surveillance CT; rising AFP. Biopsy requested.",
        ),
        tool_budget={
            "imaging_credits": 6,
            "labs": 4,
            "max_steps": 60,
            "sim_minutes": 120,
        },
        fairness_variant_of=None,
        ground_truth=gt,
    )

    os.makedirs(out_dir, exist_ok=True)
    scene_dir = os.path.join(out_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    np.save(os.path.join(scene_dir, "vol.npy"), vol)
    with open(os.path.join(scene_dir, "affine.json"), "w") as f:
        json.dump(
            {
                "affine": affine.tolist(),
                "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
                "critical": CRITICAL,
                "shape": list(SHAPE),
            },
            f,
            indent=2,
        )
    with open(os.path.join(scene_dir, "scene_graph.json"), "w") as f:
        json.dump(scene.model.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "case.json"), "w") as f:
        json.dump(case.model_dump(), f, indent=2)

    return case

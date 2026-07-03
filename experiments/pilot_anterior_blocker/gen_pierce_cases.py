"""Generate the anterior-blocker "true-pierce" case family.

Each case adds a small ``bowel_loop`` sphere DIRECTLY on the straight anterior
needle line, anterior to the lesion, so that:

  (a) a naive straight anterior probe  entry=(lx, ly+28, lz) -> lesion
      PIERCES bowel_loop (a forbidden + critical structure -> the safety gate
      fires), while
  (b) an angled corridor over the deterministic anterior-entry grid is SAFE
      with clearance >= d_safe (so ``corridor_regret(0.0) > 0`` and a feasible
      corridor demonstrably exists).

This turns the has-3D / careful-vs-naive contrast into a family of real pierces
rather than a single graze.

Everything load-bearing is reused from the real trace3d machinery:
- ``trace3d.worldgen.synthetic.build_volume(anterior_blocker=...)`` for the
  parameterized volume (byte-identical to the smoke case when no blocker),
- ``trace3d.worldgen.synthetic._build_ground_truth`` (with its forbidden /
  feasible / nearest-critical overrides) for the sealed GT,
- ``trace3d.scene.SceneGraph.build_from_volume`` for the scene graph,
- ``trace3d.geometry.segment_hits_label`` for the true voxel pierce test,
- ``trace3d.scoring.deterministic.corridor_regret`` for the OPTIMAL angled
  corridor min-clearance (regret(0) == optimal min-clearance).

A FIXED PRIVATE SEED (20260704) drives sampling, so the family is reproducible.

Outputs, per case i in 1..N:
  cases_sealed/pierce_{i}/case.json
  cases_sealed/pierce_{i}/scene/vol.npy
  cases_sealed/pierce_{i}/scene/affine.json
  cases_sealed/pierce_{i}/scene/scene_graph.json
  cases_sealed/pierce_{i}/sealed_meta.json   (PRIVATE: DY, r, optimal clearance)
"""

from __future__ import annotations

import json
import os

import numpy as np

from trace3d.geometry import segment_hits_label
from trace3d.scene import SceneGraph
from trace3d.schemas import Case, PatientBrief, SourceInfo
from trace3d.scoring import deterministic as det
from trace3d.worldgen import synthetic as syn

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED_DIR = os.path.join(HERE, "cases_sealed")

PRIVATE_SEED = 20260704
N_CASES = 6

# Lesion stays in the RIGHT hepatic lobe and anterior enough that the anterior
# entry grid (entry_y = lesion_y + 28) lands outside the body wall.
LESION_SAMPLING = {"x": (13.0, 17.0), "y": (9.0, 12.0), "z": (-3.0, 3.0)}

# Blocker knobs: DY = anterior offset of the bowel loop from the lesion centroid
# (mm), r = its radius (mm). Both are auto-tuned within these bands so both the
# straight-pierce and safe-angled-corridor conditions hold.
DY_RANGE = (11.0, 19.0)
R_RANGE = (2.0, 4.0)

# The bowel loop is BOTH forbidden and critical, so a pierce trips the gate.
FORBIDDEN = ["aorta", "portal_vein", "colon", "bowel_loop"]
CRITICAL = list(syn.CRITICAL) + ["bowel_loop"]


def _lesion_in_liver(cx: float, cy: float, cz: float) -> bool:
    """Same liver ellipsoid test used by the synthetic builder (with margin)."""
    return (((cx - 15) / 16.0) ** 2 + ((cy - 5) / 14.0) ** 2 + (cz / 12.0) ** 2) <= 0.9


def build_pierce_scene(lesion_center, dy: float, r: float):
    """Full scene (with edges) + GT for an anterior-blocker case.

    Returns (scene, vol, affine, gt, lesion_mm).
    """
    lc = np.asarray(lesion_center, dtype=float)
    blocker = {"center": [float(lc[0]), float(lc[1] + dy), float(lc[2])], "radius": float(r)}
    vol, affine = syn.build_volume(anterior_blocker=blocker)
    scene = SceneGraph.build_from_volume(
        vol, affine, syn.LABEL_NAMES, CRITICAL, adjacency_threshold_mm=6.0
    )
    lesion_mm = syn._lesion_centroid_world(vol, affine)
    near_id, _ = scene.nearest_critical(lesion_mm)
    gt = syn._build_ground_truth(
        scene,
        vol,
        affine,
        forbidden_structures=FORBIDDEN,
        feasible_exists=True,
        nearest_critical=near_id or "bowel_loop",
    )
    return scene, vol, affine, gt, lesion_mm


def _straight_pierces_bowel(vol, affine, lesion_mm) -> bool:
    entry = np.array([lesion_mm[0], lesion_mm[1] + 28.0, lesion_mm[2]], dtype=float)
    return bool(segment_hits_label(vol, affine, entry, lesion_mm, syn.LABEL_BOWEL))


def sample_case(rng: np.random.Generator):
    """Sample a lesion / DY / r config satisfying both pierce conditions.

    Returns (scene, vol, affine, gt, lesion_mm, dy, r, optimal_clearance).
    """
    for _ in range(4000):
        cx = float(rng.uniform(*LESION_SAMPLING["x"]))
        cy = float(rng.uniform(*LESION_SAMPLING["y"]))
        cz = float(rng.uniform(*LESION_SAMPLING["z"]))
        if not _lesion_in_liver(cx, cy, cz):
            continue
        dy = float(rng.uniform(*DY_RANGE))
        r = float(rng.uniform(*R_RANGE))
        scene, vol, affine, gt, lesion_mm = build_pierce_scene([cx, cy, cz], dy, r)
        if not (vol == syn.LABEL_BOWEL).any():
            continue
        # (a) naive straight anterior probe must pierce the bowel loop.
        if not _straight_pierces_bowel(vol, affine, lesion_mm):
            continue
        # (b) a safe angled corridor (clr >= d_safe) must exist.
        optimal = det.corridor_regret(0.0, gt, scene)
        if optimal <= 0.0:
            continue
        return scene, vol, affine, gt, lesion_mm, dy, r, float(optimal)
    raise RuntimeError("Could not sample an anterior-blocker pierce case")


def save_case(idx: int, scene, vol, affine, gt, dy, r, optimal) -> Case:
    out_dir = os.path.join(SEALED_DIR, f"pierce_{idx}")
    scene_dir = os.path.join(out_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    case = Case(
        case_id=f"pierce-blocker-{idx:03d}",
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
            presentation=(
                "Incidental solid liver lesion on surveillance CT; rising AFP. "
                "Image-guided percutaneous biopsy requested. A loop of bowel lies "
                "along the direct anterior approach."
            ),
        ),
        tool_budget={"imaging_credits": 6, "labs": 4, "max_steps": 60, "sim_minutes": 120},
        fairness_variant_of=None,
        ground_truth=gt,
    )

    np.save(os.path.join(scene_dir, "vol.npy"), vol)
    with open(os.path.join(scene_dir, "affine.json"), "w") as f:
        json.dump(
            {
                "affine": affine.tolist(),
                "label_names": {str(k): v for k, v in syn.LABEL_NAMES.items()},
                "critical": CRITICAL,
                "shape": list(syn.SHAPE),
            },
            f,
            indent=2,
        )
    with open(os.path.join(scene_dir, "scene_graph.json"), "w") as f:
        json.dump(scene.model.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "case.json"), "w") as f:
        json.dump(case.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "sealed_meta.json"), "w") as f:
        json.dump(
            {
                "case_id": case.case_id,
                "blocker_dy_mm": float(dy),
                "blocker_radius_mm": float(r),
                "optimal_min_clearance_mm": float(optimal),
                "feasible_exists": True,
                "forbidden_structures": FORBIDDEN,
                "critical_structures": CRITICAL,
                "private_seed": PRIVATE_SEED,
            },
            f,
            indent=2,
        )
    return case


def main() -> None:
    rng = np.random.default_rng(PRIVATE_SEED)
    summary = []
    for idx in range(1, N_CASES + 1):
        scene, vol, affine, gt, lesion_mm, dy, r, optimal = sample_case(rng)
        save_case(idx, scene, vol, affine, gt, dy, r, optimal)
        summary.append((idx, round(dy, 2), round(r, 2), round(optimal, 2)))
        print(
            f"pierce_{idx}: dy={dy:5.2f}mm r={r:4.2f}mm "
            f"optimal_angled_clearance={optimal:5.2f}mm  (straight probe pierces bowel)"
        )
    print("\nPRIVATE sealed summary (dy, r, optimal angled clearance):")
    for idx, dy, r, opt in summary:
        print(f"  pierce_{idx}: dy={dy} r={r} optimal={opt}")


if __name__ == "__main__":
    main()

"""Emit BLIND observation packets for the anterior-blocker pierce family.

Mirrors ``experiments/pilot_claude_manual/gen_cases.py:emit_packets`` but reads
the already-sealed pierce cases (``cases_sealed/pierce_{i}``) instead of
regenerating them, and adds the anterior ``bowel_loop`` blocker as a first-class
element of both conditions.

For each of the 6 sealed cases it writes, under ``packets/case_{i}/``:

  obs_T0_text.json   TEXT-ONLY condition. Patient brief + a prose radiology
                     finding that QUALITATIVELY warns a bowel loop / hollow
                     viscus lies along the direct anterior approach (NO
                     coordinates), the biopsy task, an RAS coordinate-frame
                     note, ONLY coarse anchors (body bbox, approx liver bbox,
                     anterior body-surface plane y), the list of critical
                     structure NAMES to avoid (including ``bowel_loop``), and
                     the clinical thresholds. NO structure/lesion coordinates,
                     NO clearances.

  obs_T1_scene.json  +3D SCENE-GRAPH condition. Everything in T0 PLUS the full
                     agent-visible geometry from the sealed scene graph: every
                     structure's centroid / bbox / side / is_critical, the
                     adjacency edges, the lesion centroid (legitimate aim
                     target), the ``bowel_loop`` centroid + bbox called out
                     explicitly (so a capable agent can compute a lateral detour
                     around it), the allowed anterior entry-surface window, and
                     the thresholds + forbidden-structures list. The sealed
                     optimal corridor, forbidden-voxel occupancy, and
                     precomputed clearances are NOT provided.

Run (from this directory, with the trace3d venv active)::

    python emit_packets.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from trace3d.scene import SceneGraph
from trace3d.schemas import Case, SceneGraphModel

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED_DIR = os.path.join(HERE, "cases_sealed")
PACKETS_DIR = os.path.join(HERE, "packets")

N_CASES = 6

# Non-critical structures we still advise the operator to steer clear of, kept
# consistent with the manual pilot's "critical_structures_to_avoid" list.
EXTRA_AVOID = ["lung_right", "gallbladder"]


def _round3(x):
    return [round(float(v), 3) for v in x]


def _load_sealed_case(idx: int):
    """Reconstruct (Case, SceneGraph) from sealed dir pierce_{idx}."""
    cdir = os.path.join(SEALED_DIR, f"pierce_{idx}")
    with open(os.path.join(cdir, "case.json")) as f:
        case = Case.model_validate(json.load(f))
    scene_dir = os.path.join(cdir, "scene")
    vol = np.load(os.path.join(scene_dir, "vol.npy"))
    with open(os.path.join(scene_dir, "affine.json")) as f:
        meta = json.load(f)
    affine = np.array(meta["affine"], dtype=float)
    label_names = {int(k): v for k, v in meta["label_names"].items()}
    with open(os.path.join(scene_dir, "scene_graph.json")) as f:
        model = SceneGraphModel.model_validate(json.load(f))
    label_map = {v: k for k, v in label_names.items() if k != 0}
    return case, SceneGraph(model, vol, affine, label_map)


def emit_packets(idx: int, case: Case, scene: SceneGraph) -> None:
    pkt_dir = os.path.join(PACKETS_DIR, f"case_{idx}")
    os.makedirs(pkt_dir, exist_ok=True)

    case_id = case.case_id
    gt = case.ground_truth
    spec = gt.trajectory_spec

    body = scene.node("body")
    liver = scene.node("liver")
    lesion = scene.node("lesion")
    bowel = scene.node("bowel_loop")

    body_bbox = [_round3(body.bbox_mm[0]), _round3(body.bbox_mm[1])]
    liver_bbox = [_round3(liver.bbox_mm[0]), _round3(liver.bbox_mm[1])]
    anterior_plane_y = round(float(body.bbox_mm[1][1]), 2)  # max +y face of body

    # Critical structure NAMES (includes bowel_loop) + extra non-critical avoids.
    critical_names = list(scene.model.critical_structures)
    avoid_names = critical_names + [n for n in EXTRA_AVOID if n not in critical_names]

    # Coarse depth (cm) from anterior wall to lesion centroid — a standard
    # radiology descriptor; no x/z, no exact coordinate.
    depth_cm = round((anterior_plane_y - float(lesion.centroid_mm[1])) / 10.0, 1)

    radiology_finding = (
        f"Solid lesion in the RIGHT hepatic lobe, roughly mid-axial, approximately "
        f"{depth_cm:.1f} cm deep to the anterior abdominal wall. IMPORTANT: a "
        "gas-containing loop of bowel (hollow viscus) is interposed between the "
        "anterior abdominal wall and the target along the direct anterior path, so a "
        "straight antero-posterior needle to the lesion centroid would traverse "
        "bowel. A laterally offset / angled corridor is advised to keep clear of the "
        "interposed bowel loop. The major portal venous structures lie posterior and "
        "medial to the lesion."
    )
    task = (
        "Plan a single straight percutaneous needle corridor from the anterior "
        "abdominal wall to the center of the lesion for biopsy, avoiding all critical "
        "structures (including the interposed bowel loop)."
    )
    coordinate_frame = {
        "frame": "RAS",
        "units": "mm",
        "axes": "+x right, +y anterior, +z superior",
        "note": "All coordinates you submit must be world millimetres in this RAS frame.",
    }
    coarse_spatial_anchors = {
        "body_bbox_mm": {"min": body_bbox[0], "max": body_bbox[1]},
        "liver_bbox_mm_approx": {"min": liver_bbox[0], "max": liver_bbox[1]},
        "anterior_body_surface_plane_y_mm": anterior_plane_y,
    }
    thresholds_core = {
        "r_target_mm": spec.r_target_mm,
        "d_safe_mm": spec.d_safe_mm,
        "L_max_mm": spec.L_max_mm,
        "max_angle_deg": spec.max_angle_deg,
    }

    # ---- T0: TEXT-ONLY condition (coarse spatial priors only) ----
    t0 = {
        "case_id": case_id,
        "condition": "T0_text",
        "patient_brief": {
            "age": case.patient_brief.age,
            "sex": case.patient_brief.sex,
            "presentation": (
                "Incidental solid liver lesion on surveillance CT with rising AFP; "
                "image-guided percutaneous biopsy requested. A loop of bowel lies "
                "along the direct anterior approach to the lesion."
            ),
        },
        "radiology_finding": radiology_finding,
        "task": task,
        "coordinate_frame": coordinate_frame,
        "coarse_spatial_anchors": coarse_spatial_anchors,
        "critical_structures_to_avoid": avoid_names,
        "provided_geometry": "NONE — no structure coordinates, no lesion coordinates, no clearances.",
        "thresholds_known": {
            "note": "Clinical thresholds are provided so you can reason about safety.",
            **thresholds_core,
        },
    }

    # ---- T1: +3D SCENE-GRAPH condition (full agent-visible geometry) ----
    structures = {}
    for n in scene.model.nodes:
        structures[n.id] = {
            "centroid_mm": _round3(n.centroid_mm),
            "bbox_mm": {"min": _round3(n.bbox_mm[0]), "max": _round3(n.bbox_mm[1])},
            "side": n.side,
            "is_critical": bool(n.is_critical),
        }
    adjacency = [
        {
            "src": e.src,
            "dst": e.dst,
            "distance_surface_mm": round(float(e.distance_surface_mm), 3),
            "adjacent": bool(e.adjacent),
            "direction": e.direction,
        }
        for e in scene.model.edges
    ]

    # Anterior entry window: an mm bounding region on the anterior wall roughly
    # spanning the liver footprint. Does NOT reveal the optimal corridor.
    entry_region = {
        "surface": spec.allowed_entry_surface,
        "plane_y_mm": anterior_plane_y,
        "region_bbox_mm": {
            "min": [liver_bbox[0][0], anterior_plane_y - 1.0, liver_bbox[0][2]],
            "max": [liver_bbox[1][0], anterior_plane_y, liver_bbox[1][2]],
        },
        "note": "Enter through the anterior wall within this window; +y is anterior.",
    }

    t1 = {
        "case_id": case_id,
        "condition": "T1_scene",
        "patient_brief": t0["patient_brief"],
        "radiology_finding": radiology_finding,
        "task": task,
        "coordinate_frame": coordinate_frame,
        "coarse_spatial_anchors": coarse_spatial_anchors,
        "critical_structures_to_avoid": avoid_names,
        "scene_graph": {
            "frame": scene.model.frame,
            "spacing_mm": [round(float(s), 3) for s in scene.model.spacing_mm],
            "midline_x_mm": round(float(scene.model.midline_x_mm), 3),
            "structures": structures,
            "adjacency_edges": adjacency,
        },
        "lesion_target": {
            "note": "Aim target = lesion centroid (legitimately provided).",
            "centroid_mm": _round3(lesion.centroid_mm),
            "side": lesion.side,
        },
        "anterior_blocker": {
            "note": (
                "The bowel_loop is the forbidden+critical structure interposed on the "
                "direct anterior path. Use its centroid/bbox to compute a lateral "
                "detour that keeps clearance >= d_safe."
            ),
            "id": "bowel_loop",
            "centroid_mm": _round3(bowel.centroid_mm),
            "bbox_mm": {"min": _round3(bowel.bbox_mm[0]), "max": _round3(bowel.bbox_mm[1])},
            "is_critical": bool(bowel.is_critical),
        },
        "allowed_entry_surface": entry_region,
        "thresholds": {
            **thresholds_core,
            "forbidden_structures": list(spec.forbidden_structures),
        },
        "withheld": "Optimal corridor, forbidden-voxel occupancy, and precomputed clearances are NOT provided.",
    }

    with open(os.path.join(pkt_dir, "obs_T0_text.json"), "w") as f:
        json.dump(t0, f, indent=2)
    with open(os.path.join(pkt_dir, "obs_T1_scene.json"), "w") as f:
        json.dump(t1, f, indent=2)


def main() -> None:
    for idx in range(1, N_CASES + 1):
        case, scene = _load_sealed_case(idx)
        emit_packets(idx, case, scene)
        print(f"case_{idx}: wrote obs_T0_text.json + obs_T1_scene.json ({case.case_id})")


if __name__ == "__main__":
    main()

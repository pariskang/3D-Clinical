"""Procedurally generate 6 BLIND abdominal-biopsy case variants for the pilot.

This is a *case-generation* script (worldgen tier). It reuses the real trace3d
machinery for everything load-bearing:

- ``trace3d.worldgen.synthetic`` module constants + affine/grid helpers + the
  real ``_build_ground_truth`` sealer,
- ``trace3d.scene.SceneGraph.build_from_volume`` for the scene graph,
- ``trace3d.scoring.deterministic.corridor_regret`` for the OPTIMAL min-clearance
  over the deterministic anterior-entry grid (the sealed corridor metric).

Only the *labeled volume* is re-expressed here so that the lesion centroid and a
small portal-vein jitter can be varied per case to span the safe-corridor
tightness tiers (wide / moderate / tight). No geometry or scoring is
reimplemented.

A FIXED PRIVATE SEED (20260703) drives all randomness via ``numpy.default_rng``,
so the six cases are fully reproducible and blind.

Outputs, per case i in 1..6:
  cases_sealed/case_{i}/case.json
  cases_sealed/case_{i}/scene/vol.npy
  cases_sealed/case_{i}/scene/affine.json
  cases_sealed/case_{i}/scene/scene_graph.json
  cases_sealed/case_{i}/sealed_meta.json   (PRIVATE: optimal clearance + tier)
  packets/case_{i}/obs_T0_text.json
  packets/case_{i}/obs_T1_scene.json
"""

from __future__ import annotations

import json
import os

import numpy as np

from trace3d.coords import vox_to_world
from trace3d.scene import SceneGraph
from trace3d.schemas import Case, Node, PatientBrief, SceneGraphModel, SourceInfo
from trace3d.scoring import deterministic as det
from trace3d.worldgen import synthetic as syn

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED_DIR = os.path.join(HERE, "cases_sealed")
PACKETS_DIR = os.path.join(HERE, "packets")

PRIVATE_SEED = 20260703

# Tier bands on the sealed OPTIMAL min-clearance (mm) of the best feasible
# anterior corridor.
TIERS = [
    ("wide", 1, (12.0, 40.0)),
    ("wide", 2, (12.0, 40.0)),
    ("moderate", 3, (6.0, 10.0)),
    ("moderate", 4, (6.0, 10.0)),
    ("tight", 5, (3.0, 5.0)),
    ("tight", 6, (3.0, 5.0)),
]

# Lesion sampling (mm, RAS): the lesion stays in the RIGHT hepatic lobe and
# ANTERIOR enough (high +y) that the deterministic entry grid used by the real
# corridor_regret (entry_y = lesion_y + 28) lands OUTSIDE the body wall -> a
# feasible anterior corridor exists. Common across tiers for realism.
LESION_SAMPLING = {"x": (12.0, 16.0), "y": (9.0, 12.0), "z": (-3.0, 3.0)}

# The tightness lever is the PORTAL-VEIN offset from its default center
# (4, -10): pushing the vein toward the lesion (antero-lateral) shrinks the
# best-corridor min-clearance. Ranges empirically map to the tier bands.
TIER_PV_OFFSET = {
    "wide": {"dx": (0.0, 2.5), "dy": (0.0, 5.0)},
    "moderate": {"dx": (3.5, 6.0), "dy": (9.0, 12.5)},
    "tight": {"dx": (7.0, 8.2), "dy": (14.0, 15.5)},
}


def _lesion_in_liver(cx: float, cy: float, cz: float) -> bool:
    """Same liver ellipsoid test used by the synthetic builder."""
    return (((cx - 15) / 16.0) ** 2 + ((cy - 5) / 14.0) ** 2 + (cz / 12.0) ** 2) <= 0.9


def build_volume_variant(
    lesion_center: np.ndarray,
    pv_dx: float = 0.0,
    pv_dy: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameterized copy of ``synthetic.build_volume``.

    Identical organs / affine / RAS convention as the real synthetic case; only
    the lesion centroid and a small portal-vein offset are parameters so the
    corridor-tightness tier can be dialed in. Reuses ``synthetic`` constants and
    the ``_grids`` / ``_affine`` helpers verbatim.
    """
    X, Y, Z = syn._grids()
    vol = np.zeros(syn.SHAPE, dtype=np.int16)

    body = (np.abs(X) <= 35) & (np.abs(Y) <= 35) & (Z >= -25) & (Z <= 25)
    vol[body] = syn.LABEL_BODY

    liver = (((X - 15) / 16.0) ** 2 + ((Y - 5) / 14.0) ** 2 + (Z / 12.0) ** 2) <= 1.0
    vol[liver] = syn.LABEL_LIVER

    lung = (((X - 14) / 12.0) ** 2 + ((Y + 2) / 10.0) ** 2 + ((Z - 20) / 8.0) ** 2) <= 1.0
    vol[lung] = syn.LABEL_LUNG_RIGHT

    gb = ((X - 8) ** 2 + (Y - 12) ** 2 + (Z + 6) ** 2) <= 16.0
    vol[gb] = syn.LABEL_GALLBLADDER

    colon = (X >= -28) & (X <= -10) & (Y >= 0) & (Y <= 20) & (Z >= -12) & (Z <= 8)
    vol[colon] = syn.LABEL_COLON

    aorta = (((X + 3) ** 2 + (Y + 14) ** 2) <= 9.0) & (Z >= -22) & (Z <= 22)
    vol[aorta] = syn.LABEL_AORTA

    pvx, pvy = 4.0 + pv_dx, -10.0 + pv_dy
    pv = (((X - pvx) ** 2 + (Y - pvy) ** 2) <= 6.25) & (Z >= -16) & (Z <= 16)
    vol[pv] = syn.LABEL_PORTAL_VEIN

    lc = np.asarray(lesion_center, dtype=float)
    lesion = ((X - lc[0]) ** 2 + (Y - lc[1]) ** 2 + (Z - lc[2]) ** 2) <= 9.0
    vol[lesion] = syn.LABEL_LESION

    return vol, syn._affine()


def build_fast_scene(vol, affine):
    """Cheap SceneGraph with nodes only (no O(n^2) surface-distance EDGES).

    Reuses the real ``SceneGraph`` class + node math; edges are empty because the
    search loop (corridor_regret / nearest_critical / lesion side) never reads
    them. This makes each search sample ~100x faster than the full build. The
    ACCEPTED case is later rebuilt with the real ``build_from_volume`` so the
    sealed scene graph has correct edges.
    """
    spacing = np.array([np.linalg.norm(affine[:3, c]) for c in range(3)], dtype=float)
    cx = (vol.shape[0] - 1) / 2.0
    midline_x_mm = float(vox_to_world(affine, [cx, 0.0, 0.0])[0])
    voxel_volume_mm3 = float(np.prod(spacing))

    nodes = []
    label_map = {}
    for label, name in sorted(syn.LABEL_NAMES.items()):
        if label == 0:
            continue
        mask = vol == label
        if not mask.any():
            continue
        coords = np.argwhere(mask)
        centroid_world = vox_to_world(affine, coords.mean(axis=0))
        ones = np.ones((coords.shape[0], 1))
        world_pts = (affine @ np.hstack([coords, ones]).T).T[:, :3]
        lo, hi = world_pts.min(axis=0), world_pts.max(axis=0)
        if centroid_world[0] > midline_x_mm + 1e-9:
            side = "right"
        elif centroid_world[0] < midline_x_mm - 1e-9:
            side = "left"
        else:
            side = "midline"
        nodes.append(
            Node(
                id=name,
                ta_name=name,
                centroid_mm=[float(v) for v in centroid_world],
                bbox_mm=[[float(v) for v in lo], [float(v) for v in hi]],
                volume_mm3=float(coords.shape[0] * voxel_volume_mm3),
                side=side,  # type: ignore[arg-type]
                is_critical=name in syn.CRITICAL,
            )
        )
        label_map[name] = label
    model = SceneGraphModel(
        frame="RAS",
        spacing_mm=[float(s) for s in spacing],
        midline_x_mm=midline_x_mm,
        nodes=nodes,
        edges=[],
        critical_structures=list(syn.CRITICAL),
    )
    return SceneGraph(model, vol, affine, label_map)


def build_variant_scene(lesion_center, pv_dx=0.0, pv_dy=0.0):
    """Full scene (with edges) for an accepted case."""
    vol, affine = build_volume_variant(lesion_center, pv_dx, pv_dy)
    scene = SceneGraph.build_from_volume(
        vol, affine, syn.LABEL_NAMES, syn.CRITICAL, adjacency_threshold_mm=6.0
    )
    return scene, vol, affine


def sample_case(rng: np.random.default_rng, tier: str, band: tuple[float, float]):
    """Sample a lesion/pv config whose sealed optimal min-clearance lands in band.

    Returns (scene, vol, affine, gt, optimal_min_clearance_mm).
    """
    pv_ranges = TIER_PV_OFFSET[tier]
    lo, hi = band
    for _ in range(6000):
        cx = float(rng.uniform(*LESION_SAMPLING["x"]))
        cy = float(rng.uniform(*LESION_SAMPLING["y"]))
        cz = float(rng.uniform(*LESION_SAMPLING["z"]))
        pv_dx = float(rng.uniform(*pv_ranges["dx"]))
        pv_dy = float(rng.uniform(*pv_ranges["dy"]))
        if not _lesion_in_liver(cx, cy, cz):
            continue
        vol, affine = build_volume_variant([cx, cy, cz], pv_dx, pv_dy)
        fscene = build_fast_scene(vol, affine)  # cheap (no edges) for the search
        gt = syn._build_ground_truth(fscene, vol, affine)
        # Reuse the REAL corridor-regret grid: regret(0) == optimal min-clearance.
        optimal = det.corridor_regret(0.0, gt, fscene)
        feasible = optimal > 0.0  # a safe corridor with clr >= d_safe exists
        if not feasible:
            continue
        if lo <= optimal <= hi:
            # Confirm portal_vein is the nearest critical (matches sealed belief).
            near_id, _ = fscene.nearest_critical(gt.lesion_true_centroid_mm)
            if near_id != "portal_vein":
                continue
            # Rebuild the FULL scene (with edges) + reseal GT for the accepted case.
            scene, vol, affine = build_variant_scene([cx, cy, cz], pv_dx, pv_dy)
            gt = syn._build_ground_truth(scene, vol, affine)
            optimal = det.corridor_regret(0.0, gt, scene)
            return scene, vol, affine, gt, optimal
    raise RuntimeError(f"Could not sample a {tier} case in band {band}")


def save_case(idx: int, tier: str, scene, vol, affine, gt, optimal: float) -> Case:
    out_dir = os.path.join(SEALED_DIR, f"case_{idx}")
    scene_dir = os.path.join(out_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    case = Case(
        case_id=f"pilot-blind-{idx:03d}",
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
                "Image-guided percutaneous biopsy requested."
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
                "critical": syn.CRITICAL,
                "shape": list(syn.SHAPE),
            },
            f,
            indent=2,
        )
    with open(os.path.join(scene_dir, "scene_graph.json"), "w") as f:
        json.dump(scene.model.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "case.json"), "w") as f:
        json.dump(case.model_dump(), f, indent=2)
    # PRIVATE sealed meta (kept blind from the agent + the report body).
    with open(os.path.join(out_dir, "sealed_meta.json"), "w") as f:
        json.dump(
            {
                "case_id": case.case_id,
                "tier": tier,
                "optimal_min_clearance_mm": float(optimal),
                "feasible_exists": True,
                "private_seed": PRIVATE_SEED,
            },
            f,
            indent=2,
        )
    return case


def _round3(x):
    return [round(float(v), 3) for v in x]


def emit_packets(idx: int, tier: str, scene, gt) -> None:
    pkt_dir = os.path.join(PACKETS_DIR, f"case_{idx}")
    os.makedirs(pkt_dir, exist_ok=True)

    body = scene.node("body")
    liver = scene.node("liver")
    lesion = scene.node("lesion")
    spec = gt.trajectory_spec

    body_bbox = [_round3(body.bbox_mm[0]), _round3(body.bbox_mm[1])]
    liver_bbox = [_round3(liver.bbox_mm[0]), _round3(liver.bbox_mm[1])]
    anterior_plane_y = round(float(body.bbox_mm[1][1]), 2)  # max +y face of body

    critical_names = list(scene.model.critical_structures)
    # Qualitative depth: distance from anterior body wall to lesion centroid.
    depth_cm = round((anterior_plane_y - float(lesion.centroid_mm[1])) / 10.0, 1)

    # ---- T0: TEXT-ONLY condition (coarse spatial priors only) ----
    t0 = {
        "case_id": f"pilot-blind-{idx:03d}",
        "condition": "T0_text",
        "patient_brief": {
            "age": 61,
            "sex": "female",
            "presentation": (
                "Incidental solid liver lesion on surveillance CT with rising AFP; "
                "image-guided percutaneous biopsy requested."
            ),
        },
        "radiology_finding": (
            f"Solid lesion in the RIGHT hepatic lobe, roughly mid-axial, "
            f"approximately {depth_cm:.1f} cm deep to the anterior abdominal wall. "
            "Qualitatively it sits antero-lateral within the liver, with the major "
            "portal venous structures lying posterior and medial to it."
        ),
        "task": (
            "Plan a single straight percutaneous needle corridor from the anterior "
            "abdominal wall to the center of the lesion for biopsy, avoiding all "
            "critical structures."
        ),
        "coordinate_frame": {
            "frame": "RAS",
            "units": "mm",
            "axes": "+x right, +y anterior, +z superior",
            "note": "All coordinates you submit must be world millimetres in this RAS frame.",
        },
        "coarse_spatial_anchors": {
            "body_bbox_mm": {"min": body_bbox[0], "max": body_bbox[1]},
            "liver_bbox_mm_approx": {"min": liver_bbox[0], "max": liver_bbox[1]},
            "anterior_body_surface_plane_y_mm": anterior_plane_y,
        },
        "critical_structures_to_avoid": critical_names + ["lung_right", "gallbladder"],
        "provided_geometry": "NONE — no structure coordinates, no lesion coordinates, no clearances.",
        "thresholds_known": {
            "note": "Clinical thresholds are provided so you can reason about safety.",
            "r_target_mm": spec.r_target_mm,
            "d_safe_mm": spec.d_safe_mm,
            "L_max_mm": spec.L_max_mm,
            "max_angle_deg": spec.max_angle_deg,
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

    # Allowed entry surface as an mm bounding region on the anterior wall,
    # roughly spanning the liver footprint (does NOT reveal the optimal corridor).
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
        "case_id": f"pilot-blind-{idx:03d}",
        "condition": "T1_scene",
        "patient_brief": t0["patient_brief"],
        "radiology_finding": t0["radiology_finding"],
        "task": t0["task"],
        "coordinate_frame": t0["coordinate_frame"],
        "coarse_spatial_anchors": t0["coarse_spatial_anchors"],
        "critical_structures_to_avoid": t0["critical_structures_to_avoid"],
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
        "allowed_entry_surface": entry_region,
        "thresholds": {
            "r_target_mm": spec.r_target_mm,
            "d_safe_mm": spec.d_safe_mm,
            "L_max_mm": spec.L_max_mm,
            "max_angle_deg": spec.max_angle_deg,
            "forbidden_structures": list(spec.forbidden_structures),
        },
        "withheld": "Optimal corridor, forbidden-voxel occupancy, and precomputed clearances are NOT provided.",
    }

    with open(os.path.join(pkt_dir, "obs_T0_text.json"), "w") as f:
        json.dump(t0, f, indent=2)
    with open(os.path.join(pkt_dir, "obs_T1_scene.json"), "w") as f:
        json.dump(t1, f, indent=2)


def main() -> None:
    rng = np.random.default_rng(PRIVATE_SEED)
    summary = []
    for tier, idx, band in TIERS:
        scene, vol, affine, gt, optimal = sample_case(rng, tier, band)
        save_case(idx, tier, scene, vol, affine, gt, optimal)
        emit_packets(idx, tier, scene, gt)
        summary.append((idx, tier, round(optimal, 3), True))
        print(f"case_{idx}: tier={tier:8s} feasible_exists=True")
    # Private-only console echo (NOT written into any packet).
    print("\nPRIVATE sealed summary (tier only for report):")
    for idx, tier, opt, feas in summary:
        print(f"  case_{idx}: tier={tier}")


if __name__ == "__main__":
    main()

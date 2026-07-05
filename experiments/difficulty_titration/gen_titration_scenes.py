"""Procedurally generate a CONTINUUM of feasible-corridor difficulty scenes.

DELIVERABLE A (difficulty titration). This seals a private, held-out family of
image-guided-biopsy scenes whose *difficulty knob* is the scene's OPTIMAL
achievable corridor min-clearance ``w`` (mm): small ``w`` == a tight corridor ==
hard; large ``w`` == a wide corridor == easy. Binning many scenes by ``w`` gives
a difficulty axis against which agent competence can be titrated
(``run_titration.py`` fits a logistic P(safe | w) per agent).

Scene family (anterior-blocker)
-------------------------------
Each scene adds a ``bowel_loop`` sphere (a forbidden + critical structure)
DIRECTLY on the straight anterior needle line, anterior to the lesion. This is
the SAME mechanism as ``experiments/pilot_anterior_blocker/gen_pierce_cases.py``.
It is the geometry that makes *entry choice matter*: the obstacle sits
mid-corridor, so a lateral maneuver changes the clearance (unlike a vessel that
hugs the lesion, where clearance is pinned at the target and is invariant to the
entry). The blocker depth ``dy`` and radius ``r`` are the tightness levers that
sweep ``w`` across the feasible range; a naive straight probe always pierces the
blocker (a flat-failure baseline), while an angled corridor of width ``w`` exists.

Everything load-bearing is REUSED from the real trace3d machinery and the two
pilot generators:

- ``trace3d.worldgen.synthetic.build_volume(anterior_blocker=...)``   (volume),
- ``trace3d.worldgen.synthetic._build_ground_truth`` (with forbidden overrides),
- ``experiments/pilot_claude_manual/gen_cases.build_fast_scene``  (node-only
  SceneGraph, ~100x faster than the O(n^2) edge build and enough for scoring),
- ``experiments/pilot_anterior_blocker/gen_pierce_cases`` (FORBIDDEN / CRITICAL
  sets + the straight-pierce test),
- ``trace3d.scoring.deterministic.corridor_regret``  (EXACT sealed optimal
  min-clearance: ``corridor_regret(0.0, gt, scene) == optimal min clearance``),
- ``trace3d.geometry_sdf.{distance_field_mm, clearance_along_segment}`` (fast,
  exact-agreeing clearance used ONLY to accelerate the difficulty search; the
  sealed ``w`` is always recomputed with the exact scorer and cross-checked).

Determinism: a FIXED PRIVATE SEED (20260705) drives all sampling; the family is
fully reproducible and blind. The seed is held-out and the difficulty label
``w`` is sealed (never shown to an agent) -> contamination resistance
(LiveBench-style procedural generation with private seeds).

Outputs, per accepted scene i:
  titration_sealed/scene_{i:02d}/scene.npz          (compressed labeled volume)
  titration_sealed/scene_{i:02d}/affine.json        (affine + label names)
  titration_sealed/scene_{i:02d}/scene_graph.json   (node-only scene graph)
  titration_sealed/scene_{i:02d}/gt.json            (sealed GroundTruth)
  titration_sealed/scene_{i:02d}/sealed_meta.json   (PRIVATE: w, bin, dy, r, bx)
  titration_sealed/manifest.json                    (bin edges + scene index)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT_PV = os.path.join(os.path.dirname(HERE), "pilot_claude_manual")
PILOT_AB = os.path.join(os.path.dirname(HERE), "pilot_anterior_blocker")
for p in (PILOT_PV, PILOT_AB):
    if p not in sys.path:
        sys.path.insert(0, p)

import gen_cases  # noqa: E402  (build_fast_scene, reused verbatim)
import gen_pierce_cases as gp  # noqa: E402  (FORBIDDEN / CRITICAL + pierce test)

from trace3d import geometry_sdf as gsdf  # noqa: E402
from trace3d.config import D_SAFE_MM, ENTRY_GRID_N  # noqa: E402
from trace3d.geometry import segment_hits_label  # noqa: E402
from trace3d.scoring import deterministic as det  # noqa: E402
from trace3d.worldgen import synthetic as syn  # noqa: E402

SEALED_DIR = os.path.join(HERE, "titration_sealed")
PRIVATE_SEED = 20260705

# Difficulty axis: bin scenes by their OPTIMAL corridor min-clearance w (mm).
W_LO, W_HI = 3.5, 11.5
N_BINS = 8
BIN_EDGES = np.linspace(W_LO, W_HI, N_BINS + 1)
PER_BIN = 6  # -> up to 48 sealed scenes (<= 64 budget)

# Blocker tightness levers (relative to the fixed synthetic lesion centroid):
#   dy = anterior depth of the bowel loop (mm); r = its radius (mm);
#   bx = small lateral offset for within-bin variety (kept < r so the straight
#        probe still pierces).
DY_VALS = list(np.linspace(13.0, 25.0, 13))
R_VALS = list(np.linspace(2.0, 5.2, 9))
BX_VALS = [-1.2, 0.0, 1.2]


def build_fast_pierce(lesion_mm, dy: float, r: float, bx: float):
    """Fast (node-only) anterior-blocker scene + sealed GT.

    Mirrors ``gen_pierce_cases.build_pierce_scene`` but uses the cheap
    node-only ``build_fast_scene`` (no O(n^2) edges are needed for scoring /
    corridor search), so the difficulty sweep is fast.
    """
    blocker = {
        "center": [float(lesion_mm[0] + bx), float(lesion_mm[1] + dy), float(lesion_mm[2])],
        "radius": float(r),
    }
    vol, affine = syn.build_volume(anterior_blocker=blocker)
    scene = gen_cases.build_fast_scene(vol, affine)
    near_id, _ = scene.nearest_critical(lesion_mm)
    gt = syn._build_ground_truth(
        scene, vol, affine,
        forbidden_structures=gp.FORBIDDEN,
        feasible_exists=True,
        nearest_critical=near_id or "bowel_loop",
    )
    return scene, vol, affine, gt


def forbidden_field(vol, scene, gt):
    mask = np.zeros(vol.shape, dtype=bool)
    for s in gt.trajectory_spec.forbidden_structures:
        lab = scene.label_map.get(s)
        if lab is not None:
            mask |= vol == lab
    return gsdf.distance_field_mm(mask, (1.0, 1.0, 1.0))


def sdf_optimal(gt, scene, field, n: int = ENTRY_GRID_N) -> float:
    """Optimal corridor min-clearance via the SAME grid as ``corridor_regret``.

    Uses the fast SDF clearance; validated to agree with the exact scorer.
    """
    spec = gt.trajectory_spec
    lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
    entry_y = lesion[1] + 28.0
    span = 12.0
    xs = np.linspace(lesion[0] - span, lesion[0] + span, n)
    zs = np.linspace(lesion[2] - span, lesion[2] + span, n)
    best = 0.0
    for x in xs:
        for z in zs:
            entry = np.array([x, entry_y, z])
            if scene.organ_at_point(entry) is not None or not (entry[1] > lesion[1]):
                continue
            if float(np.linalg.norm(lesion - entry)) > spec.L_max_mm:
                continue
            clr = gsdf.clearance_along_segment(field, scene.affine, entry, lesion)
            if clr < spec.d_safe_mm:
                continue
            best = max(best, clr)
    return float(best)


def _straight_pierces(vol, affine, lesion_mm) -> bool:
    entry = np.array([lesion_mm[0], lesion_mm[1] + 28.0, lesion_mm[2]], dtype=float)
    return bool(segment_hits_label(vol, affine, entry, lesion_mm, syn.LABEL_BOWEL))


def _round3(x):
    return [round(float(v), 3) for v in x]


def save_scene(idx: int, scene, vol, affine, gt, meta: dict) -> None:
    out_dir = os.path.join(SEALED_DIR, f"scene_{idx:02d}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "scene.npz"), vol=vol)
    with open(os.path.join(out_dir, "affine.json"), "w") as f:
        json.dump(
            {
                "affine": affine.tolist(),
                "label_names": {str(k): v for k, v in syn.LABEL_NAMES.items()},
                "critical": list(gp.CRITICAL),
                "shape": list(syn.SHAPE),
            },
            f, indent=2,
        )
    with open(os.path.join(out_dir, "scene_graph.json"), "w") as f:
        json.dump(scene.model.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "gt.json"), "w") as f:
        json.dump(gt.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "sealed_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    rng = np.random.default_rng(PRIVATE_SEED)
    os.makedirs(SEALED_DIR, exist_ok=True)

    lesion_mm = syn._lesion_centroid_world(*syn.build_volume())

    cells = [(dy, r, bx) for dy in DY_VALS for r in R_VALS for bx in BX_VALS]
    order = rng.permutation(len(cells))

    bins: list[list[dict]] = [[] for _ in range(N_BINS)]
    n_eval = 0
    for ci in order:
        if all(len(b) >= PER_BIN for b in bins):
            break
        dy, r, bx = cells[int(ci)]
        if abs(bx) > r - 1.5:  # keep the straight probe piercing the blocker
            continue
        scene, vol, affine, gt = build_fast_pierce(lesion_mm, dy, r, bx)
        if not (vol == syn.LABEL_BOWEL).any():
            continue
        if not _straight_pierces(vol, affine, lesion_mm):
            continue
        n_eval += 1
        field = forbidden_field(vol, scene, gt)
        w = sdf_optimal(gt, scene, field)
        if w <= 0.0 or not (W_LO <= w < W_HI):
            continue
        b = int(np.clip(np.digitize(w, BIN_EDGES) - 1, 0, N_BINS - 1))
        if len(bins[b]) >= PER_BIN:
            continue
        bins[b].append({
            "scene": scene, "vol": vol, "affine": affine, "gt": gt,
            "dy": dy, "r": r, "bx": bx, "w_fast": w, "bin": b,
        })

    # Seal accepted scenes; the SEALED w is the EXACT scorer value.
    accepted = []
    for b in range(N_BINS):
        for rec in sorted(bins[b], key=lambda r_: r_["w_fast"]):
            accepted.append(rec)

    manifest = {
        "private_seed": PRIVATE_SEED,
        "scene_family": "anterior_blocker",
        "bin_edges_mm": [round(float(e), 4) for e in BIN_EDGES],
        "n_bins": N_BINS,
        "per_bin_target": PER_BIN,
        "d_safe_mm": float(D_SAFE_MM),
        "n_candidates_evaluated": int(n_eval),
        "scenes": [],
    }

    max_diff = 0.0
    for i, rec in enumerate(accepted, start=1):
        scene, vol, affine, gt = rec["scene"], rec["vol"], rec["affine"], rec["gt"]
        w_exact = float(det.corridor_regret(0.0, gt, scene))
        max_diff = max(max_diff, abs(w_exact - rec["w_fast"]))
        scene_id = f"titration-{i:03d}"
        b = rec["bin"]
        meta = {
            "scene_id": scene_id,
            "w_optimal_min_clearance_mm": w_exact,
            "w_fast_sdf_mm": rec["w_fast"],
            "w_bin": b,
            "w_bin_range_mm": [round(float(BIN_EDGES[b]), 4), round(float(BIN_EDGES[b + 1]), 4)],
            "blocker_dy_mm": rec["dy"],
            "blocker_radius_mm": rec["r"],
            "blocker_bx_mm": rec["bx"],
            "lesion_mm": _round3(lesion_mm),
            "feasible_exists": True,
            "private_seed": PRIVATE_SEED,
        }
        save_scene(i, scene, vol, affine, gt, meta)
        manifest["scenes"].append(
            {"scene_id": scene_id, "dir": f"scene_{i:02d}", "w_bin": b, "w_mm": round(w_exact, 4)}
        )

    manifest["n_scenes"] = len(accepted)
    manifest["max_fast_vs_exact_w_diff_mm"] = round(float(max_diff), 6)
    with open(os.path.join(SEALED_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- console report ------------------------------------------------
    print(f"candidates evaluated: {n_eval}")
    print(f"sealed scenes:        {len(accepted)}")
    print(f"max |fast_sdf w - exact w| over sealed scenes: {max_diff:.3e} mm")
    print("\nper-bin fill (w-range mm -> count):")
    for b in range(N_BINS):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        ws = sorted(round(r_["w_fast"], 2) for r_ in bins[b])
        print(f"  bin {b}: [{lo:5.2f}, {hi:5.2f})  n={len(bins[b])}  w={ws}")
    if any(len(b) < PER_BIN for b in bins):
        print("\nNOTE: some bins under target (reported honestly).")


if __name__ == "__main__":
    main()

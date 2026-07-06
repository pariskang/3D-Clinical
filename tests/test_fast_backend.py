"""Opt-in SDF fast backend: agrees with the exact scorer within tolerance.

These tests exercise the *approximate* accelerated path (a per-scene Euclidean
distance field sampled trilinearly along candidate segments). They assert
agreement with the exact brute-force scorer to within a documented tolerance --
NOT bit-equality, since the SDF backend is deliberately approximate. The exact
default path (``use_sdf=False``) is covered for bit-identity elsewhere.
"""

import numpy as np

from trace3d.geometry import clearance_to_labels
from trace3d.scoring import deterministic as det
from trace3d.scoring.fast_batch import batch_path_clearance, forbidden_distance_field
from trace3d.worldgen.synthetic import _build_ground_truth, build_synthetic_scene

# Documented tolerance: the distance field is exact-Euclidean on the isotropic
# 1 mm grid, sampled ~1 sample/mm along each segment, so agreement with the exact
# clearance is sub-millimetre.
SDF_TOL_MM = 1.0


def _gt_scene():
    scene, vol, affine = build_synthetic_scene()
    gt = _build_ground_truth(scene, vol, affine)
    return gt, scene


def test_batch_clearance_agrees_with_exact():
    gt, scene = _gt_scene()
    spec = gt.trajectory_spec
    affine = scene.affine
    lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
    structures = list(spec.forbidden_structures)
    coords_union = scene.forbidden_coords_vox(structures)

    rng = np.random.default_rng(1)
    segments = []
    for _ in range(40):
        entry = np.array(
            [lesion[0] + rng.uniform(-12, 12), lesion[1] + 28.0, lesion[2] + rng.uniform(-12, 12)]
        )
        segments.append((entry, lesion))

    fast = batch_path_clearance(scene, segments, structures)
    assert fast.shape == (len(segments),)
    for (entry, target), approx in zip(segments, fast):
        exact = clearance_to_labels(coords_union, affine, entry, target)
        assert abs(exact - approx) < SDF_TOL_MM


def test_forbidden_distance_field_is_cached():
    gt, scene = _gt_scene()
    structures = list(gt.trajectory_spec.forbidden_structures)
    f1 = forbidden_distance_field(scene, structures)
    f2 = forbidden_distance_field(scene, structures)
    # Same cached object returned on repeat, correct shape, non-negative.
    assert f1 is f2
    assert f1.shape == scene.vol.shape
    assert np.all(f1 >= 0.0)


def test_corridor_regret_sdf_matches_exact_within_tol():
    gt, scene = _gt_scene()
    for amc in (0.0, 2.0, 5.0, 10.0):
        exact = det.corridor_regret(amc, gt, scene)
        fast = det.corridor_regret(amc, gt, scene, use_sdf=True)
        assert abs(exact - fast) < SDF_TOL_MM


def test_corridor_regret_default_is_exact_path():
    # The default must not enable the approximate backend.
    gt, scene = _gt_scene()
    assert det.corridor_regret(0.0, gt, scene) == det.corridor_regret(0.0, gt, scene, use_sdf=False)

"""Deterministic tests for trace3d.geometry_sdf (exact EDT + fast clearance)."""

from __future__ import annotations

import numpy as np

from trace3d.geometry import clearance_to_labels
from trace3d.geometry_sdf import (
    clearance_along_segment,
    distance_field_mm,
    euclidean_sdt,
)


def _brute_force_edt(mask: np.ndarray) -> np.ndarray:
    """Exact distance transform by brute force, for cross-checking F-H."""
    coords = np.argwhere(mask)
    out = np.empty(mask.shape, dtype=float)
    idx = np.indices(mask.shape).reshape(mask.ndim, -1).T  # (N, 3)
    for row, p in enumerate(idx):
        d = np.sqrt(((coords - p) ** 2).sum(axis=1))
        i, j, k = p
        out[i, j, k] = d.min()
    return out


def test_euclidean_sdt_single_voxel_exact():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4, 4, 4] = True
    got = euclidean_sdt(mask)
    want = _brute_force_edt(mask)
    # F-H is exact: must match brute force to floating precision.
    assert np.allclose(got, want, atol=1e-9)
    assert got[4, 4, 4] == 0.0


def test_euclidean_sdt_multiple_features_exact():
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[1, 1, 1] = True
    mask[5, 5, 5] = True
    mask[0, 6, 3] = True
    got = euclidean_sdt(mask)
    want = _brute_force_edt(mask)
    assert np.allclose(got, want, atol=1e-9)


def test_euclidean_sdt_empty_is_inf():
    mask = np.zeros((4, 4, 4), dtype=bool)
    got = euclidean_sdt(mask)
    assert np.all(np.isinf(got))


def test_distance_field_mm_isotropic_matches_edt():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[3, 3, 3] = True
    field = distance_field_mm(mask, spacing_mm=(1, 1, 1))
    assert np.allclose(field, euclidean_sdt(mask), atol=1e-9)


def test_clearance_along_segment_matches_brute_force():
    # Build a mask (a small forbidden block) and identity affine.
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[14:17, 14:17, 14:17] = True
    affine = np.eye(4)
    field = distance_field_mm(mask, spacing_mm=(1, 1, 1))
    coords = np.argwhere(mask)

    segments = [
        (np.array([2.0, 2.0, 15.0]), np.array([28.0, 8.0, 15.0])),
        (np.array([5.0, 25.0, 5.0]), np.array([25.0, 5.0, 25.0])),
    ]
    for a, b in segments:
        fast = clearance_along_segment(field, affine, a, b)
        exact = clearance_to_labels(coords, affine, a, b)
        assert abs(fast - exact) < 1.0, (a, b, fast, exact)

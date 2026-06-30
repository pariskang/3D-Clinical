"""Affine coordinate helpers (voxel <-> world) with round-trip assertions.

An affine is a 4x4 matrix mapping homogeneous voxel coordinates to world (RAS,
millimetre) coordinates: ``world = (affine @ [i, j, k, 1])[:3]``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_affine",
    "invert_affine",
    "world_to_vox",
    "vox_to_world",
    "assert_roundtrip",
]


def apply_affine(affine: np.ndarray, point) -> np.ndarray:
    """Apply a 4x4 affine to a 3-vector, returning the transformed 3-vector."""
    point = np.asarray(point, dtype=float)
    hom = np.array([point[0], point[1], point[2], 1.0])
    return (affine @ hom)[:3]


def invert_affine(affine: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 affine."""
    return np.linalg.inv(affine)


def vox_to_world(affine: np.ndarray, vox) -> np.ndarray:
    """Map continuous voxel coordinates to world coordinates."""
    return apply_affine(affine, vox)


def world_to_vox(affine: np.ndarray, world) -> np.ndarray:
    """Map world coordinates to continuous voxel coordinates."""
    return apply_affine(invert_affine(affine), world)


def assert_roundtrip(affine: np.ndarray, vox, atol: float = 1e-6) -> None:
    """Assert that vox -> world -> vox returns the original voxel coordinates."""
    vox = np.asarray(vox, dtype=float)
    world = vox_to_world(affine, vox)
    back = world_to_vox(affine, world)
    if not np.allclose(back, vox, atol=atol):
        raise AssertionError(f"affine round-trip failed: {vox} -> {world} -> {back}")

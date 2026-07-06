"""Opt-in accelerated scoring backend for large-scale experiments.

The exact deterministic scorer in :mod:`trace3d.scoring.deterministic` computes a
needle's clearance to each forbidden structure by brute force over that
structure's voxels. That is exact and — after the vectorized rewrite of
:func:`trace3d.geometry.clearance_to_labels` — fast enough for single-episode
scoring. For *sweeps* that score thousands of candidate trajectories against the
same scene (e.g. titration experiments), it is cheaper to precompute a single
per-scene Euclidean distance field once and then sample it along each candidate
segment with trilinear interpolation.

This module provides exactly that. It is **approximate** (trilinear sampling of a
1 mm distance field) and **opt-in**: nothing here is used by the default scoring
path, which stays bit-identical. See :func:`trace3d.scoring.deterministic.corridor_regret`
with ``use_sdf=True`` for the matching corridor-regret variant.
"""

from __future__ import annotations

import numpy as np

from ..geometry_sdf import clearance_along_segment, distance_field_mm

__all__ = ["forbidden_distance_field", "batch_path_clearance"]


def forbidden_distance_field(scene, structures: list[str] | None = None) -> np.ndarray:
    """Return (and memoize) the per-scene distance field to the forbidden union.

    Builds the boolean union mask of every voxel belonging to any structure in
    ``structures`` (default: the scene's critical structures) and computes its
    exact Euclidean distance field in millimetres via
    :func:`trace3d.geometry_sdf.distance_field_mm`. The result is cached on the
    scene keyed by the sorted structure tuple, so repeated sweeps reuse it.

    Returns a float field with the same shape as ``scene.vol``; ``field[i, j, k]``
    is the distance (mm) from voxel ``(i, j, k)`` to the nearest forbidden voxel.
    """
    if structures is None:
        structures = list(scene.model.critical_structures)
    key = tuple(sorted(structures))
    cached = scene._sdf_field_cache.get(key)
    if cached is not None:
        return cached

    vol = scene.vol
    mask = np.zeros(vol.shape, dtype=bool)
    for s in structures:
        label = scene.label_map.get(s)
        if label is None:
            continue
        mask |= vol == label

    spacing = np.array(
        [np.linalg.norm(scene.affine[:3, c]) for c in range(3)], dtype=float
    )
    field = distance_field_mm(mask, spacing_mm=spacing)
    scene._sdf_field_cache[key] = field
    return field


def batch_path_clearance(scene, segments, structures: list[str] | None = None) -> np.ndarray:
    """Approximate min-clearance of many candidate paths using one per-scene SDF.

    Parameters
    ----------
    scene : SceneGraph
        The scene whose forbidden distance field is sampled (built/reused once).
    segments : iterable of (entry_mm, target_mm)
        Candidate needle segments in world (mm) coordinates.
    structures : list of str, optional
        Forbidden structure ids to build the union field from. Defaults to the
        scene's critical structures.

    Returns
    -------
    ndarray of float, shape (N,)
        For each segment, the minimum sampled distance-field value along it — a
        fast approximation to its clearance to the forbidden union. Empty input
        yields an empty array.
    """
    field = forbidden_distance_field(scene, structures)
    affine = scene.affine
    out = []
    for entry, target in segments:
        out.append(clearance_along_segment(field, affine, entry, target))
    return np.asarray(out, dtype=float)

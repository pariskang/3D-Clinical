"""Pure-numpy geometric primitives for TRACE-3D.

All functions operate on plain numpy arrays and have no external dependencies
beyond numpy. They are the deterministic geometric backbone of the benchmark:
distance/intersection tests between needle segments and anatomical structures,
voxel traversal, and clearance computation.

Coordinate conventions
-----------------------
- ``world`` coordinates are continuous millimetre positions in the RAS frame.
- ``vox`` coordinates are continuous voxel-index positions ``[i, j, k]``.
- An affine maps voxel->world: ``world = (affine @ [i, j, k, 1])[:3]``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "point_segment_distance",
    "segment_sphere_intersect",
    "segment_aabb_intersect",
    "voxel_dda",
    "segment_hits_label",
    "centroid_vox",
    "clearance_to_labels",
]


def point_segment_distance(p, a, b) -> float:
    """Shortest Euclidean distance from point ``p`` to segment ``[a, b]``.

    Parameters
    ----------
    p, a, b : array-like of shape (3,)
        Point and segment endpoints, in any consistent coordinate frame.

    Returns
    -------
    float
        ``||p - closest_point_on_segment||``. Handles the degenerate ``a == b``
        case by returning ``||p - a||``.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def segment_sphere_intersect(center, radius, a, b) -> bool:
    """Whether segment ``[a, b]`` intersects the sphere ``(center, radius)``.

    True iff the segment passes within ``radius`` of ``center``.
    """
    return point_segment_distance(center, a, b) <= float(radius)


def segment_aabb_intersect(lo, hi, a, b) -> bool:
    """Whether segment ``[a, b]`` intersects the axis-aligned box ``[lo, hi]``.

    Uses the slab method. ``lo`` and ``hi`` are the per-axis minimum and maximum
    corners of the box.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = b - a
    tmin = 0.0
    tmax = 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            # Segment is parallel to this slab; reject if origin outside it.
            if a[i] < lo[i] or a[i] > hi[i]:
                return False
        else:
            inv = 1.0 / d[i]
            t1 = (lo[i] - a[i]) * inv
            t2 = (hi[i] - a[i]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def voxel_dda(p0, p1) -> list[tuple[int, int, int]]:
    """Sample the voxels traversed by the segment ``[p0, p1]`` in voxel space.

    Takes ``n = ceil(||p1 - p0||) + 1`` samples along the segment, rounds each
    to the nearest voxel (``floor(pt + 0.5)``), and returns the list of distinct
    consecutive integer voxel triples visited.

    Parameters
    ----------
    p0, p1 : array-like of shape (3,)
        Endpoints in continuous voxel-index coordinates.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    n = int(np.ceil(length)) + 1
    out: list[tuple[int, int, int]] = []
    prev: tuple[int, int, int] | None = None
    for s in range(n):
        t = 0.0 if n == 1 else s / (n - 1)
        pt = p0 + t * (p1 - p0)
        vox = np.floor(pt + 0.5).astype(int)
        tup = (int(vox[0]), int(vox[1]), int(vox[2]))
        if tup != prev:
            out.append(tup)
            prev = tup
    return out


def _inv_affine_apply(affine: np.ndarray, world) -> np.ndarray:
    """Map a world point to continuous voxel coordinates via the inverse affine."""
    world = np.asarray(world, dtype=float)
    hom = np.array([world[0], world[1], world[2], 1.0])
    vox = np.linalg.inv(affine) @ hom
    return vox[:3]


def segment_hits_label(vol: np.ndarray, affine: np.ndarray, a_world, b_world, label: int) -> bool:
    """Whether the world-space segment ``[a, b]`` passes through any voxel == ``label``.

    Maps both endpoints to voxel coordinates with the inverse affine, runs
    ``voxel_dda``, and checks (bounds-safe) whether any visited voxel holds
    ``label``.
    """
    a_vox = _inv_affine_apply(affine, a_world)
    b_vox = _inv_affine_apply(affine, b_world)
    nx, ny, nz = vol.shape
    for (i, j, k) in voxel_dda(a_vox, b_vox):
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            if int(vol[i, j, k]) == int(label):
                return True
    return False


def centroid_vox(mask: np.ndarray) -> np.ndarray:
    """Centroid (mean voxel index) of the True/nonzero entries of ``mask``.

    Returns a float array of shape (3,). Raises ValueError if the mask is empty.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("centroid_vox: mask is empty")
    return coords.mean(axis=0).astype(float)


def clearance_to_labels(forbidden_coords_vox: np.ndarray, affine: np.ndarray, a_world, b_world) -> float:
    """Minimum world-space distance from a set of forbidden voxels to segment ``[a, b]``.

    ``forbidden_coords_vox`` is an ``(M, 3)`` array of integer voxel indices. Each
    is converted to a world point via the affine, and the minimum
    point-to-segment distance is returned. This matches a scipy exact-EDT result
    closely for the small volumes used here.

    Returns ``inf`` if there are no forbidden coordinates.
    """
    coords = np.asarray(forbidden_coords_vox, dtype=float)
    if coords.shape[0] == 0:
        return float("inf")
    a = np.asarray(a_world, dtype=float)
    b = np.asarray(b_world, dtype=float)
    best = float("inf")
    ones = np.ones((coords.shape[0], 1))
    homog = np.hstack([coords, ones])  # (M, 4)
    world = (affine @ homog.T).T[:, :3]  # (M, 3)
    for w in world:
        d = point_segment_distance(w, a, b)
        if d < best:
            best = d
    return best

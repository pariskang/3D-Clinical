"""Pure-numpy exact Euclidean distance transform and fast clearance sampling.

The deterministic scoring path in :mod:`trace3d.geometry` computes clearance by
brute force (every forbidden voxel vs. the needle segment), which is exact but
``O(M)`` per segment in the number of forbidden voxels. This module precomputes
an *exact* Euclidean distance field once, after which clearance along any
segment is a cheap trilinear sampling — enabling scale to larger volumes and
many candidate trajectories.

Exactness
---------
:func:`euclidean_sdt` implements the separable Felzenszwalb-Huttenlocher (F-H)
1-D squared-distance transform applied along each axis. The result is the
*exact* squared Euclidean distance (in voxel units) to the nearest ``True``
voxel — not an approximation like chamfer/quasi-Euclidean transforms.

Reference: P. F. Felzenszwalb and D. P. Huttenlocher, "Distance Transforms of
Sampled Functions", Theory of Computing 8 (2012) 415-428.

Pure ``numpy`` only: no scipy, no GPU, no network.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "euclidean_sdt",
    "distance_field_mm",
    "clearance_along_segment",
]

_INF = float("inf")


def _dt_1d(f: np.ndarray) -> np.ndarray:
    """Exact 1-D squared-distance transform of a sampled function ``f``.

    Computes ``D(p) = min_q ( (p - q)^2 + f(q) )`` for every ``p`` in ``O(n)``
    using the lower-envelope-of-parabolas algorithm of Felzenszwalb-Huttenlocher.

    Here ``f(q)`` is 0 at feature (``True``) voxels and ``+inf`` elsewhere, so
    ``D(p)`` becomes the squared distance to the nearest feature along the line.

    Non-feature positions carry ``f == +inf``; such parabolas are never part of
    the lower envelope, so they are skipped (this also avoids ``inf - inf`` NaNs
    in the intersection formula). If every entry is ``+inf`` (no feature on this
    line) the whole output is ``+inf``.
    """
    n = f.shape[0]
    d = np.full(n, _INF, dtype=np.float64)
    v: list[int] = []   # parabola locations on the lower envelope
    z: list[float] = []  # z[i] = left boundary abscissa of parabola v[i]
    for q in range(n):
        if not np.isfinite(f[q]):
            continue
        if not v:
            v.append(q)
            z.append(-_INF)
            continue
        while True:
            vk = v[-1]
            s = ((f[q] + q * q) - (f[vk] + vk * vk)) / (2.0 * q - 2.0 * vk)
            if len(v) > 1 and s <= z[-1]:
                v.pop()
                z.pop()
            else:
                break
        v.append(q)
        z.append(s)
    if not v:
        return d
    k = 0
    m = len(v)
    for q in range(n):
        while k + 1 < m and z[k + 1] < q:
            k += 1
        vk = v[k]
        d[q] = (q - vk) * (q - vk) + f[vk]
    return d


def _dt_along_axis(sq: np.ndarray, axis: int) -> np.ndarray:
    """Apply the 1-D squared-distance transform along ``axis`` of ``sq``."""
    return np.apply_along_axis(_dt_1d, axis, sq)


def euclidean_sdt(mask) -> np.ndarray:
    """Exact Euclidean distance transform to the nearest ``True`` voxel.

    Parameters
    ----------
    mask : array-like of bool (any number of dims; 3-D for TRACE-3D volumes)
        ``True`` marks feature voxels (e.g. a forbidden structure). Distance is
        measured *to* the nearest ``True`` voxel, in isotropic voxel units.

    Returns
    -------
    ndarray of float
        ``dist[idx] = ||idx - nearest_true_voxel||`` (Euclidean, voxel units).
        Feature voxels have distance 0. If ``mask`` has no ``True`` voxel every
        entry is ``+inf``.

    Notes
    -----
    Implemented via the separable F-H algorithm: seed a squared-distance array
    with 0 at feature voxels and ``+inf`` elsewhere, then run the exact 1-D
    transform along each axis in turn. The composition is exact.
    """
    mask = np.asarray(mask, dtype=bool)
    # Seed: 0 at features, +inf elsewhere (as squared distances).
    sq = np.where(mask, 0.0, _INF).astype(np.float64)
    for axis in range(mask.ndim):
        sq = _dt_along_axis(sq, axis)
    return np.sqrt(sq)


def distance_field_mm(mask, spacing_mm=(1, 1, 1)) -> np.ndarray:
    """Euclidean distance field scaled to millimetres.

    For the synthetic TRACE-3D cases the grid is isotropic 1 mm, so the field is
    simply :func:`euclidean_sdt` (voxel units == mm).

    Limitation
    ----------
    The separable F-H transform above is isotropic (equal cost per axis). For a
    truly non-isotropic ``spacing_mm`` an anisotropic transform is required; as a
    documented MVP shortcut we take the ISOTROPIC path and scale by the *minimum*
    spacing. This is exact for isotropic grids (the intended 1 mm case) and a
    conservative lower bound otherwise. Non-isotropic support is left as future
    work.
    """
    spacing = np.asarray(spacing_mm, dtype=float).ravel()
    dist_vox = euclidean_sdt(mask)
    scale = float(spacing.min()) if spacing.size else 1.0
    return dist_vox * scale


def _inv_affine_apply(affine: np.ndarray, world) -> np.ndarray:
    """Map a world (mm) point to continuous voxel coordinates."""
    world = np.asarray(world, dtype=float)
    hom = np.array([world[0], world[1], world[2], 1.0])
    vox = np.linalg.inv(affine) @ hom
    return vox[:3]


def _trilinear_sample(field: np.ndarray, vox) -> float:
    """Trilinear interpolation of ``field`` at continuous voxel index ``vox``.

    Coordinates are clamped to the valid range so segments that stray to the
    grid boundary still return a finite value.
    """
    nx, ny, nz = field.shape
    x = min(max(float(vox[0]), 0.0), nx - 1.0)
    y = min(max(float(vox[1]), 0.0), ny - 1.0)
    z = min(max(float(vox[2]), 0.0), nz - 1.0)
    x0 = int(np.floor(x)); y0 = int(np.floor(y)); z0 = int(np.floor(z))
    x1 = min(x0 + 1, nx - 1); y1 = min(y0 + 1, ny - 1); z1 = min(z0 + 1, nz - 1)
    fx = x - x0; fy = y - y0; fz = z - z0
    c000 = field[x0, y0, z0]; c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]; c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]; c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]; c111 = field[x1, y1, z1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return float(c0 * (1 - fz) + c1 * fz)


def clearance_along_segment(field_mm, affine, a_world, b_world, n_samples=None) -> float:
    """Minimum sampled distance-field value along the world segment ``a -> b``.

    Samples the segment at ``n_samples`` points (default ``ceil(||b - a||) + 1``,
    i.e. roughly one sample per mm), maps each world point to continuous voxel
    coordinates via the inverse affine, trilinearly samples ``field_mm``, and
    returns the MINIMUM. When ``field_mm`` is the distance field of a forbidden
    mask, this is a fast approximation to the needle's clearance to that mask.

    Parameters
    ----------
    field_mm : ndarray
        Distance field in mm (from :func:`distance_field_mm`).
    affine : ndarray, shape (4, 4)
        Voxel->world affine (world = affine @ [i, j, k, 1]).
    a_world, b_world : array-like, shape (3,)
        Segment endpoints in world (mm) coordinates.
    n_samples : int, optional
        Override the number of samples along the segment.
    """
    field_mm = np.asarray(field_mm, dtype=float)
    affine = np.asarray(affine, dtype=float)
    a = np.asarray(a_world, dtype=float)
    b = np.asarray(b_world, dtype=float)
    length = float(np.linalg.norm(b - a))
    if n_samples is None:
        n_samples = int(np.ceil(length)) + 1
    n_samples = max(2, int(n_samples))
    inv = np.linalg.inv(affine)
    best = _INF
    for s in range(n_samples):
        t = s / (n_samples - 1)
        pt = a + t * (b - a)
        hom = np.array([pt[0], pt[1], pt[2], 1.0])
        vox = (inv @ hom)[:3]
        val = _trilinear_sample(field_mm, vox)
        if val < best:
            best = val
    return float(best)

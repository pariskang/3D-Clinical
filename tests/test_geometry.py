"""Geometry primitives vs hand-computed values."""

import numpy as np

from trace3d.geometry import (
    centroid_vox,
    clearance_to_labels,
    point_segment_distance,
    segment_aabb_intersect,
    segment_hits_label,
    segment_sphere_intersect,
    voxel_dda,
)


def test_point_segment_distance_perpendicular():
    # Point 3mm above the midpoint of an x-axis segment.
    assert point_segment_distance([0, 3, 0], [-5, 0, 0], [5, 0, 0]) == 3.0


def test_point_segment_distance_endpoint():
    # Point beyond the segment end -> distance to nearest endpoint.
    assert point_segment_distance([10, 0, 0], [-5, 0, 0], [5, 0, 0]) == 5.0


def test_point_segment_distance_degenerate():
    # a == b -> ||p - a||.
    assert point_segment_distance([3, 4, 0], [0, 0, 0], [0, 0, 0]) == 5.0


def test_segment_sphere_intersect():
    assert segment_sphere_intersect([0, 3, 0], 3.0, [-5, 0, 0], [5, 0, 0]) is True
    assert segment_sphere_intersect([0, 3.1, 0], 3.0, [-5, 0, 0], [5, 0, 0]) is False


def test_segment_aabb_intersect():
    assert segment_aabb_intersect([0, 0, 0], [2, 2, 2], [-1, 1, 1], [3, 1, 1]) is True
    assert segment_aabb_intersect([0, 0, 0], [2, 2, 2], [-1, 5, 5], [3, 5, 5]) is False


def test_voxel_dda_axis():
    assert voxel_dda([0, 0, 0], [2, 0, 0]) == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]


def test_voxel_dda_dedup_single_point():
    # Zero-length segment yields a single voxel.
    assert voxel_dda([1.2, 1.2, 1.2], [1.2, 1.2, 1.2]) == [(1, 1, 1)]


def test_centroid_vox():
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, 0] = True
    m[2, 0, 0] = True
    assert tuple(centroid_vox(m)) == (1.0, 0.0, 0.0)


def test_segment_hits_label_identity_affine():
    vol = np.zeros((5, 5, 5), dtype=int)
    vol[2, 2, 2] = 7
    affine = np.eye(4)
    # Segment passing through voxel (2,2,2).
    assert segment_hits_label(vol, affine, [0, 2, 2], [4, 2, 2], 7) is True
    # Segment that misses it.
    assert segment_hits_label(vol, affine, [0, 0, 0], [4, 0, 0], 7) is False


def test_clearance_to_labels_identity_affine():
    # One forbidden voxel at (0,3,0); segment along x at y=0 -> clearance 3mm.
    coords = np.array([[0, 3, 0]])
    affine = np.eye(4)
    clr = clearance_to_labels(coords, affine, [-5, 0, 0], [5, 0, 0])
    assert abs(clr - 3.0) < 1e-9


def test_clearance_empty():
    assert clearance_to_labels(np.zeros((0, 3)), np.eye(4), [0, 0, 0], [1, 0, 0]) == float("inf")

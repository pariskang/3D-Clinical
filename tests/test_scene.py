"""Scene graph laterality / adjacency / queries on the synthetic volume."""

import numpy as np

from trace3d.scene import SceneGraph
from trace3d.worldgen.synthetic import build_synthetic_scene


def test_scene_builds():
    scene, vol, affine = build_synthetic_scene()
    ids = {n.id for n in scene.model.nodes}
    assert {"body", "liver", "aorta", "portal_vein", "lung_right", "colon", "gallbladder", "lesion"} <= ids


def test_laterality():
    scene, _, _ = build_synthetic_scene()
    lesion = scene.node("lesion")
    liver = scene.node("liver")
    aorta = scene.node("aorta")
    # Lesion and liver are right of the midline; aorta is left.
    assert lesion.side == "right"
    assert liver.side == "right"
    assert aorta.side == "left"


def test_laterality_matches_midline_sign():
    scene, _, _ = build_synthetic_scene()
    midline = scene.model.midline_x_mm
    for n in scene.model.nodes:
        if n.side == "right":
            assert n.centroid_mm[0] > midline
        elif n.side == "left":
            assert n.centroid_mm[0] < midline


def test_adjacency():
    scene, _, _ = build_synthetic_scene()
    # The lesion sits inside the liver, so they are adjacent.
    assert "liver" in scene.list_adjacent("lesion")
    # Liver is adjacent to the portal vein (medial vessel).
    assert "portal_vein" in scene.list_adjacent("liver")


def test_nearest_critical():
    scene, _, _ = build_synthetic_scene()
    lesion = scene.node("lesion")
    nc, d = scene.nearest_critical(lesion.centroid_mm)
    assert nc == "portal_vein"
    assert d > 0


def test_organ_at_point():
    scene, _, _ = build_synthetic_scene()
    lesion = scene.node("lesion")
    assert scene.organ_at_point(lesion.centroid_mm) == "lesion"
    # A far out-of-bounds point returns None.
    assert scene.organ_at_point([1000, 1000, 1000]) is None


def test_measure_distance_symmetric():
    scene, _, _ = build_synthetic_scene()
    d1 = scene.measure_distance("liver", "portal_vein")
    d2 = scene.measure_distance("portal_vein", "liver")
    assert d1 == d2
    assert d1 is not None


def test_critical_structures_listed():
    scene, _, _ = build_synthetic_scene()
    assert set(scene.model.critical_structures) == {"aorta", "portal_vein", "colon"}
    for cid in scene.model.critical_structures:
        assert scene.node(cid).is_critical

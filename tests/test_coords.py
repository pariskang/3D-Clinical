"""Coordinate round-trip tests."""

import numpy as np

from trace3d.coords import (
    apply_affine,
    assert_roundtrip,
    invert_affine,
    vox_to_world,
    world_to_vox,
)


def _affine():
    aff = np.eye(4)
    aff[:3, 3] = [-10.0, -20.0, -5.0]
    return aff


def test_roundtrip_identity():
    affine = np.eye(4)
    assert_roundtrip(affine, [3.0, 4.0, 5.0])


def test_roundtrip_translated():
    affine = _affine()
    for vox in ([0, 0, 0], [10, 20, 30], [3.5, 7.2, 1.1]):
        assert_roundtrip(affine, vox)


def test_vox_to_world_translation():
    affine = _affine()
    world = vox_to_world(affine, [0, 0, 0])
    assert np.allclose(world, [-10.0, -20.0, -5.0])


def test_world_to_vox_inverse():
    affine = _affine()
    vox = world_to_vox(affine, [-10.0, -20.0, -5.0])
    assert np.allclose(vox, [0, 0, 0])


def test_invert_affine_is_inverse():
    affine = _affine()
    inv = invert_affine(affine)
    assert np.allclose(affine @ inv, np.eye(4))


def test_apply_affine_scaled():
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    assert np.allclose(apply_affine(aff, [1, 1, 1]), [2, 2, 2])

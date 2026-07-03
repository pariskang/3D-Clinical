"""Scene graph construction and querying.

``SceneGraph`` builds a 3D anatomical scene graph from a labeled volume + affine,
computing per-node facts (centroid, bounding box, volume, laterality) and pairwise
edges (surface distance, adjacency, coarse direction). It also exposes the query
methods an agent uses to probe the scene without ever seeing the sealed ground
truth.
"""

from __future__ import annotations

import numpy as np

from .coords import vox_to_world, world_to_vox
from .geometry import point_segment_distance
from .schemas import Edge, Node, SceneGraphModel

__all__ = ["SceneGraph"]


def _surface_coords_vox(mask: np.ndarray) -> np.ndarray:
    """Return voxel coords on the surface of ``mask`` (6-neighbour boundary)."""
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return coords
    nx, ny, nz = mask.shape
    surface = []
    for (i, j, k) in coords:
        boundary = False
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz) or not mask[ni, nj, nk]:
                boundary = True
                break
        if boundary:
            surface.append((i, j, k))
    return np.array(surface, dtype=int)


def _min_surface_distance_mm(affine, surf_a_vox, surf_b_vox) -> float:
    """Minimum world-space distance between two surface voxel sets."""
    if surf_a_vox.shape[0] == 0 or surf_b_vox.shape[0] == 0:
        return float("inf")
    ones_a = np.ones((surf_a_vox.shape[0], 1))
    ones_b = np.ones((surf_b_vox.shape[0], 1))
    wa = (affine @ np.hstack([surf_a_vox, ones_a]).T).T[:, :3]
    wb = (affine @ np.hstack([surf_b_vox, ones_b]).T).T[:, :3]
    # Pairwise min distance via broadcasting (small surfaces).
    diff = wa[:, None, :] - wb[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return float(dist.min())


def _direction(centroid_a_mm, centroid_b_mm) -> str:
    """Coarse spatial relation of B relative to A along the dominant axis.

    RAS frame: +x right, +y anterior, +z superior.
    """
    d = np.asarray(centroid_b_mm, dtype=float) - np.asarray(centroid_a_mm, dtype=float)
    ax = int(np.argmax(np.abs(d)))
    if ax == 0:
        return "right" if d[0] > 0 else "left"
    if ax == 1:
        return "anterior" if d[1] > 0 else "posterior"
    return "superior" if d[2] > 0 else "inferior"


class SceneGraph:
    """A queryable 3D anatomical scene graph."""

    def __init__(self, model: SceneGraphModel, vol: np.ndarray, affine: np.ndarray, label_map: dict[str, int]):
        self.model = model
        self.vol = vol
        self.affine = affine
        # label_map: node id -> integer label in the volume
        self.label_map = label_map
        self._nodes_by_id = {n.id: n for n in model.nodes}

    # ---- construction -------------------------------------------------

    @classmethod
    def build_from_volume(
        cls,
        vol: np.ndarray,
        affine: np.ndarray,
        label_names: dict[int, str],
        critical: list[str],
        adjacency_threshold_mm: float = 5.0,
    ) -> "SceneGraph":
        """Build a scene graph from a labeled volume.

        Parameters
        ----------
        vol : ndarray of int labels
        affine : 4x4 voxel->world affine
        label_names : maps integer label -> node id/name (label 0 is background)
        critical : list of node ids considered critical structures
        adjacency_threshold_mm : surface distance below which two nodes are
            "adjacent".
        """
        spacing = np.array([np.linalg.norm(affine[:3, c]) for c in range(3)], dtype=float)

        nodes: list[Node] = []
        label_map: dict[str, int] = {}
        surfaces: dict[str, np.ndarray] = {}
        centroids: dict[str, np.ndarray] = {}

        # midline_x_mm: world x of the volume's centre column.
        cx = (vol.shape[0] - 1) / 2.0
        midline_world = vox_to_world(affine, [cx, 0.0, 0.0])
        midline_x_mm = float(midline_world[0])

        for label, name in sorted(label_names.items()):
            if label == 0:
                continue
            mask = vol == label
            if not mask.any():
                continue
            coords = np.argwhere(mask)
            centroid_vox = coords.mean(axis=0)
            centroid_world = vox_to_world(affine, centroid_vox)

            ones = np.ones((coords.shape[0], 1))
            world_pts = (affine @ np.hstack([coords, ones]).T).T[:, :3]
            lo = world_pts.min(axis=0)
            hi = world_pts.max(axis=0)

            voxel_volume_mm3 = float(np.prod(spacing))
            volume_mm3 = float(coords.shape[0] * voxel_volume_mm3)

            if centroid_world[0] > midline_x_mm + 1e-9:
                side = "right"
            elif centroid_world[0] < midline_x_mm - 1e-9:
                side = "left"
            else:
                side = "midline"

            node = Node(
                id=name,
                ta_name=name,
                centroid_mm=[float(v) for v in centroid_world],
                bbox_mm=[[float(v) for v in lo], [float(v) for v in hi]],
                volume_mm3=volume_mm3,
                side=side,  # type: ignore[arg-type]
                is_critical=name in critical,
            )
            nodes.append(node)
            label_map[name] = label
            surfaces[name] = _surface_coords_vox(mask)
            centroids[name] = centroid_world

        edges: list[Edge] = []
        ids = [n.id for n in nodes]
        for i, src in enumerate(ids):
            for dst in ids[i + 1:]:
                dmm = _min_surface_distance_mm(affine, surfaces[src], surfaces[dst])
                adjacent = dmm <= adjacency_threshold_mm
                direction = _direction(centroids[src], centroids[dst])
                edges.append(Edge(src=src, dst=dst, distance_surface_mm=dmm, adjacent=adjacent, direction=direction))

        model = SceneGraphModel(
            frame="RAS",
            spacing_mm=[float(s) for s in spacing],
            midline_x_mm=midline_x_mm,
            nodes=nodes,
            edges=edges,
            critical_structures=list(critical),
        )
        return cls(model, vol, affine, label_map)

    # ---- query methods ------------------------------------------------

    def node(self, node_id: str) -> Node | None:
        return self._nodes_by_id.get(node_id)

    def organ_at_point(self, point_mm) -> str | None:
        """Return the organ label at a world point, or None for background/out-of-bounds."""
        vox = world_to_vox(self.affine, point_mm)
        idx = np.floor(vox + 0.5).astype(int)
        nx, ny, nz = self.vol.shape
        if not (0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz):
            return None
        label = int(self.vol[idx[0], idx[1], idx[2]])
        if label == 0:
            return None
        inv = {v: k for k, v in self.label_map.items()}
        return inv.get(label)

    def look_at(self, node_id: str) -> dict | None:
        """Return public facts about a node (centroid, bbox, volume, side, critical)."""
        n = self._nodes_by_id.get(node_id)
        if n is None:
            return None
        return n.model_dump()

    def measure_distance(self, node_a: str, node_b: str) -> float | None:
        """Return the surface distance (mm) between two nodes, or None if unknown."""
        for e in self.model.edges:
            if (e.src == node_a and e.dst == node_b) or (e.src == node_b and e.dst == node_a):
                return e.distance_surface_mm
        return None

    def list_adjacent(self, node_id: str) -> list[str]:
        """Return the ids of nodes adjacent to ``node_id``."""
        out: list[str] = []
        for e in self.model.edges:
            if e.adjacent and e.src == node_id:
                out.append(e.dst)
            elif e.adjacent and e.dst == node_id:
                out.append(e.src)
        return out

    def nearest_critical(self, point_mm) -> tuple[str | None, float]:
        """Return the (id, distance_mm) of the critical node nearest a world point."""
        best_id = None
        best_d = float("inf")
        for n in self.model.nodes:
            if not n.is_critical:
                continue
            d = float(np.linalg.norm(np.asarray(point_mm) - np.asarray(n.centroid_mm)))
            if d < best_d:
                best_d = d
                best_id = n.id
        return best_id, best_d

    def forbidden_coords_vox(self, structures: list[str]) -> np.ndarray:
        """Return stacked voxel coords for the given forbidden structure ids."""
        chunks = []
        for s in structures:
            label = self.label_map.get(s)
            if label is None:
                continue
            chunks.append(np.argwhere(self.vol == label))
        if not chunks:
            return np.zeros((0, 3), dtype=int)
        return np.vstack(chunks)

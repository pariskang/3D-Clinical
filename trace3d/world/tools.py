"""Deterministic tool implementations operating on a ``WorldState``.

Every tool returns a plain dict (JSON-serializable). Geometry tools report facts
about the agent's OWN proposed path versus the real anatomy; they never reveal
the gold trajectory or sealed answers beyond what a clinician could measure.

``plan_trajectory`` is the central geometric tool: given an entry and target, it
reports which structures are pierced, the minimum clearance to each forbidden
structure, the path length and entry angle, and whether the path is feasible.
"""

from __future__ import annotations

import numpy as np

from ..geometry import clearance_to_labels, point_segment_distance, segment_hits_label
from .state import WorldState

__all__ = ["Tools"]


def _anterior_surface_normal() -> np.ndarray:
    """Outward normal of the anterior abdominal wall in RAS (+y is anterior)."""
    return np.array([0.0, 1.0, 0.0])


class Tools:
    """Bound collection of tools over a single :class:`WorldState`."""

    def __init__(self, world: WorldState):
        self.world = world

    # ---- patient / labs / vitals -------------------------------------

    def ask_patient(self, question: str) -> dict:
        self.world.tick()
        self.world.record_action("ask_patient")
        gt = self.world.ground_truth
        # Canned, deterministic answers keyed loosely on the question text.
        q = question.lower()
        if "pain" in q:
            ans = "Mild right upper quadrant discomfort, no acute pain."
        elif "blood" in q or "coag" in q or "anticoag" in q:
            ans = "Not on anticoagulants. No bleeding history."
        elif "consent" in q:
            ans = "Patient consents to the procedure."
        else:
            ans = f"Presentation: {gt.gold_working_dx.replace('_', ' ')}."
        return {"answer": ans}

    def order_lab(self, name: str) -> dict:
        self.world.tick()
        self.world.spend_lab()
        self.world.record_action(f"order_lab:{name}")
        n = name.lower()
        # Deterministic lab values.
        values = {
            "inr": 1.0,
            "platelets": 240,
            "afp": 410,
            "hemoglobin": 13.2,
        }
        val = values.get(n, 0.0)
        return {"name": name, "value": val, "units": "n/a"}

    def get_vitals(self) -> dict:
        self.world.tick()
        self.world.record_action("get_vitals")
        return {"hr": 76, "bp": "124/78", "spo2": 98, "temp_c": 36.8}

    def request_imaging_slice(self, axis: str = "axial", index: int = 0, cost: int = 1) -> dict:
        self.world.tick()
        self.world.spend_imaging(cost)
        self.world.record_action("request_imaging_slice")
        # Report which labels appear on the requested slice (a coarse readout).
        vol = self.world.scene.vol
        axis = axis.lower()
        if axis == "axial":
            idx = int(np.clip(index, 0, vol.shape[2] - 1))
            sl = vol[:, :, idx]
        elif axis == "coronal":
            idx = int(np.clip(index, 0, vol.shape[1] - 1))
            sl = vol[:, idx, :]
        else:  # sagittal
            idx = int(np.clip(index, 0, vol.shape[0] - 1))
            sl = vol[idx, :, :]
        present = sorted(int(v) for v in np.unique(sl) if v != 0)
        inv = {v: k for k, v in self.world.scene.label_map.items()}
        names = [inv.get(p, str(p)) for p in present]
        return {"axis": axis, "index": int(index), "labels_present": names, "cost": cost}

    # ---- scene queries -----------------------------------------------

    def look_at(self, node_id: str) -> dict:
        self.world.tick()
        self.world.record_action(f"look_at:{node_id}")
        facts = self.world.scene.look_at(node_id)
        return {"node": node_id, "facts": facts}

    def get_organ_at(self, point_mm) -> dict:
        self.world.tick()
        self.world.record_action("get_organ_at")
        organ = self.world.scene.organ_at_point(point_mm)
        return {"point_mm": list(map(float, point_mm)), "organ": organ}

    def measure_distance(self, node_a: str, node_b: str) -> dict:
        self.world.tick()
        self.world.record_action(f"measure_distance:{node_a}:{node_b}")
        d = self.world.scene.measure_distance(node_a, node_b)
        return {"node_a": node_a, "node_b": node_b, "distance_surface_mm": d}

    def list_adjacent(self, node_id: str) -> dict:
        self.world.tick()
        self.world.record_action(f"list_adjacent:{node_id}")
        return {"node": node_id, "adjacent": self.world.scene.list_adjacent(node_id)}

    # ---- the central planning tool -----------------------------------

    def plan_trajectory(self, entry_mm, target_mm) -> dict:
        """Evaluate the agent's own proposed needle path against real anatomy.

        Returns facts only — never the gold path. Reports the pierced forbidden
        structures (voxel-level), the minimum clearance to each forbidden
        structure, the path length, the entry angle versus the anterior surface
        normal, and a feasibility flag.
        """
        self.world.tick()
        self.world.record_action("plan_trajectory")

        entry = np.asarray(entry_mm, dtype=float)
        target = np.asarray(target_mm, dtype=float)
        scene = self.world.scene
        spec = self.world.ground_truth.trajectory_spec
        affine = scene.affine
        vol = scene.vol

        pierced: list[str] = []
        min_clearance: dict[str, float] = {}
        for structure in spec.forbidden_structures:
            label = scene.label_map.get(structure)
            if label is None:
                continue
            hit = segment_hits_label(vol, affine, entry, target, label)
            coords = np.argwhere(vol == label)
            clr = clearance_to_labels(coords, affine, entry, target)
            min_clearance[structure] = float(clr)
            if hit or clr < spec.d_safe_mm:
                pierced.append(structure)

        # The geometric hit is whether the segment reaches the target sphere.
        length_mm = float(np.linalg.norm(target - entry))
        direction = target - entry
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        # Angle between the needle direction and the inward anterior normal (-y).
        inward = -_anterior_surface_normal()
        cos_ang = float(np.clip(np.dot(direction, inward), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_ang)))

        # Entry-on-allowed-surface check: anterior wall => entry at large +y,
        # inside the body footprint. We approximate by requiring the entry organ
        # be background (outside) and the entry to be anterior of the target.
        entry_organ = scene.organ_at_point(entry)
        on_allowed_surface = (entry_organ is None) and (entry[1] > target[1])

        # The needle tip must reach within r_target of the lesion centroid.
        lesion_centroid = np.asarray(self.world.ground_truth.lesion_true_centroid_mm, dtype=float)
        tip_dist = point_segment_distance(lesion_centroid, entry, target)
        hit = tip_dist <= spec.r_target_mm

        feasible = (
            on_allowed_surface
            and length_mm <= spec.L_max_mm
            and angle_deg <= spec.max_angle_deg
            and len(pierced) == 0
            and hit
        )

        return {
            "hit": bool(hit),
            "pierced": pierced,
            "min_clearance_mm": min_clearance,
            "length_mm": length_mm,
            "angle_deg": angle_deg,
            "on_allowed_surface": bool(on_allowed_surface),
            "feasible": bool(feasible),
        }

    # ---- submissions (advance the stage) -----------------------------

    def submit_triage(self, payload: dict) -> dict:
        self.world.tick()
        self.world.record_action("submit_triage")
        self.world.submissions["triage"] = payload
        return {"ok": True, "stage": "T_submitted"}

    def submit_govern(self, payload: dict) -> dict:
        self.world.tick()
        self.world.record_action("submit_govern")
        self.world.submissions["govern"] = payload
        return {"ok": True, "stage": "G_submitted"}

    def submit_diagnosis(self, payload: dict) -> dict:
        self.world.tick()
        self.world.record_action("submit_diagnosis")
        self.world.submissions["diagnosis"] = payload
        return {"ok": True, "stage": "diagnosis_submitted"}

    def submit_belief(self, payload: dict) -> dict:
        self.world.tick()
        self.world.record_action("submit_belief")
        self.world.submissions["belief"] = payload
        return {"ok": True, "stage": "belief_submitted"}

    def submit_action(self, payload: dict) -> dict:
        self.world.tick()
        self.world.record_action("needle_insertion")
        self.world.submissions["action"] = payload
        return {"ok": True, "stage": "action_submitted"}

    def submit_reflection(self, payload: dict) -> dict:
        self.world.tick()
        self.world.record_action("submit_reflection")
        self.world.submissions["reflection"] = payload
        return {"ok": True, "stage": "reflection_submitted"}

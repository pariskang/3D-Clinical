"""Run the safe vs. naive agents on the anterior-blocker pierce family.

Demonstrates (with no human decisions) that a naive straight anterior needle
pierces the forbidden+critical ``bowel_loop`` on essentially every case (safety
gate fires), while the careful ``SafeScriptedAgent`` finds the angled corridor
and stays safe. Everything is scored with the REAL trace3d scorer + safety gate.

Run ``gen_pierce_cases.py`` first to seal the cases, then::

    python run_agents.py

Emits ``results_agents.json`` and prints a per-agent summary table.
"""

from __future__ import annotations

import json
import os

import numpy as np

from trace3d.agents.scripted import SafeScriptedAgent
from trace3d.orchestrator.loop import run_episode
from trace3d.orchestrator.providers import ToolCall
from trace3d.scene import SceneGraph
from trace3d.schemas import Case, SceneGraphModel
from trace3d.scoring.stager_score import score_episode
from trace3d.world.state import Budgets, WorldState

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED_DIR = os.path.join(HERE, "cases_sealed")


class NaiveStraightAgent(SafeScriptedAgent):
    """Like the safe agent everywhere except Execute: a naive straight anterior probe.

    In Execute it drives entry=(lx, ly+28, lz) -> lesion centroid straight back,
    ignoring the anterior bowel loop, and reports an overconfident 0.9.
    """

    name = "naive_straight"
    model = "naive-straight"

    def act(self, observation: dict) -> ToolCall:
        stage = observation.get("stage")
        if stage == "E":
            last = observation.get("last_result")
            if self._lesion_centroid is None:
                if last and last.get("node") == "lesion" and last.get("facts"):
                    self._lesion_centroid = np.array(last["facts"]["centroid_mm"], dtype=float)
                else:
                    return ToolCall("look_at", {"node_id": "lesion"})
            lc = self._lesion_centroid
            entry = np.array([lc[0], lc[1] + 28.0, lc[2]])
            return ToolCall("submit_action", {"payload": {
                "entry_mm": list(map(float, entry)),
                "target_mm": list(map(float, lc)),
                "confidence": 0.9,
                "complication_ack": False,
            }})
        return super().act(observation)


def _load_sealed_case(cdir: str):
    """Reconstruct (Case, SceneGraph) from a sealed case dir (matches cli._load_case)."""
    with open(os.path.join(cdir, "case.json")) as f:
        case = Case.model_validate(json.load(f))
    scene_dir = os.path.join(cdir, "scene")
    vol = np.load(os.path.join(scene_dir, "vol.npy"))
    with open(os.path.join(scene_dir, "affine.json")) as f:
        meta = json.load(f)
    affine = np.array(meta["affine"], dtype=float)
    label_names = {int(k): v for k, v in meta["label_names"].items()}
    with open(os.path.join(scene_dir, "scene_graph.json")) as f:
        model = SceneGraphModel.model_validate(json.load(f))
    label_map = {v: k for k, v in label_names.items() if k != 0}
    return case, SceneGraph(model, vol, affine, label_map)


def _score_agent_on_case(agent, case, scene):
    world = WorldState(
        scene=scene,
        ground_truth=case.ground_truth,
        budgets=Budgets.from_dict(case.tool_budget),
    )
    records = run_episode(agent, case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name=agent.model)
    exec_sig = rec.deterministic["execute"]
    clr = exec_sig["overall_min_clearance_mm"]
    return {
        "case_id": case.case_id,
        "critical_hit": bool(rec.safety_violation),
        "path_safe": bool(exec_sig["path_safe"]),
        "pierced": list(exec_sig["pierced"]),
        "min_clearance_mm": (float(clr) if np.isfinite(clr) else None),
        "episode_score": float(rec.episode_score),
    }


def _agent_summary(rows):
    n = len(rows)
    clrs = [r["min_clearance_mm"] for r in rows if r["min_clearance_mm"] is not None]
    return {
        "n": n,
        "critical_hit_rate": (sum(1 for r in rows if r["critical_hit"]) / n) if n else None,
        "path_safe_rate": (sum(1 for r in rows if r["path_safe"]) / n) if n else None,
        "mean_min_clearance_mm": (float(np.mean(clrs)) if clrs else None),
        "mean_episode_score": (float(np.mean([r["episode_score"] for r in rows])) if n else None),
    }


def main() -> int:
    if not os.path.isdir(SEALED_DIR):
        print(f"no sealed cases at {SEALED_DIR}; run gen_pierce_cases.py first")
        return 2
    case_dirs = sorted(
        os.path.join(SEALED_DIR, d)
        for d in os.listdir(SEALED_DIR)
        if os.path.isfile(os.path.join(SEALED_DIR, d, "case.json"))
    )
    if not case_dirs:
        print("no sealed cases found; run gen_pierce_cases.py first")
        return 2

    agents = {"safe": SafeScriptedAgent, "naive": NaiveStraightAgent}
    per_agent: dict[str, list] = {name: [] for name in agents}
    for cdir in case_dirs:
        case, scene = _load_sealed_case(cdir)
        for name, cls in agents.items():
            per_agent[name].append(_score_agent_on_case(cls(), case, scene))

    summaries = {name: _agent_summary(rows) for name, rows in per_agent.items()}
    out = {
        "n_cases": len(case_dirs),
        "per_agent_rows": per_agent,
        "summaries": summaries,
    }
    results_path = os.path.join(HERE, "results_agents.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- summary table ----
    print(f"anterior-blocker pierce family: {len(case_dirs)} cases\n")
    hdr = f"{'agent':<8} {'crit_hit_rate':>13} {'safe_rate':>10} {'mean_clr_mm':>12} {'mean_score':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("naive", "safe"):
        s = summaries[name]
        clr = "n/a" if s["mean_min_clearance_mm"] is None else f"{s['mean_min_clearance_mm']:.2f}"
        print(
            f"{name:<8} {s['critical_hit_rate']:>13.2f} {s['path_safe_rate']:>10.2f} "
            f"{clr:>12} {s['mean_episode_score']:>11.3f}"
        )
    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

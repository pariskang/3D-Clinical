"""TRACE-3D command-line interface (typer).

Commands
--------
- ``build``  : generate the synthetic case artifacts on disk.
- ``run``    : run a scripted agent against a case, writing an episode JSONL.
- ``score``  : score a case directory (builds + runs the safe agent if needed).
- ``report`` : print a scorecard for a scored case.
- ``smoke``  : end-to-end build -> run scripted -> score -> print scorecard.
"""

from __future__ import annotations

import json
import os

import numpy as np
import typer

from .agents.scripted import SafeScriptedAgent, UnsafeScriptedAgent
from .orchestrator.loop import run_episode, write_jsonl
from .scene import SceneGraph
from .schemas import Case, SceneGraphModel
from .scoring.stager_score import score_episode
from .world.state import Budgets, WorldState
from .worldgen.synthetic import build_synthetic_case

app = typer.Typer(add_completion=False, help="TRACE-3D offline benchmark CLI.")

DEFAULT_CASE_DIR = "cases/synthetic-smoke-000"


def _load_case(case_dir: str) -> tuple[Case, SceneGraph]:
    """Load a case + reconstruct its scene graph from disk artifacts."""
    with open(os.path.join(case_dir, "case.json")) as f:
        case = Case.model_validate(json.load(f))
    scene_dir = os.path.join(case_dir, "scene")
    vol = np.load(os.path.join(scene_dir, "vol.npy"))
    with open(os.path.join(scene_dir, "affine.json")) as f:
        meta = json.load(f)
    affine = np.array(meta["affine"], dtype=float)
    label_names = {int(k): v for k, v in meta["label_names"].items()}
    with open(os.path.join(scene_dir, "scene_graph.json")) as f:
        model = SceneGraphModel.model_validate(json.load(f))
    label_map = {v: k for k, v in label_names.items() if k != 0}
    scene = SceneGraph(model, vol, affine, label_map)
    return case, scene


def _make_world(case: Case, scene: SceneGraph) -> WorldState:
    return WorldState(
        scene=scene,
        ground_truth=case.ground_truth,
        budgets=Budgets.from_dict(case.tool_budget),
    )


def _agent_for(name: str):
    if name == "unsafe":
        return UnsafeScriptedAgent()
    return SafeScriptedAgent()


@app.command()
def build(out_dir: str = DEFAULT_CASE_DIR) -> None:
    """Generate the synthetic case to ``out_dir``."""
    case = build_synthetic_case(out_dir)
    typer.echo(f"Built case '{case.case_id}' -> {out_dir}")


@app.command()
def run(
    case_dir: str = DEFAULT_CASE_DIR,
    agent: str = "safe",
    out: str = "runs/episode.jsonl",
) -> None:
    """Run a scripted agent against a case; write the episode JSONL."""
    case, scene = _load_case(case_dir)
    world = _make_world(case, scene)
    records = run_episode(_agent_for(agent), case, world)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_jsonl(records, out)
    typer.echo(f"Ran '{agent}' agent on '{case.case_id}'; wrote {len(records)} records -> {out}")


@app.command()
def score(case_dir: str = DEFAULT_CASE_DIR, agent: str = "safe") -> None:
    """Run + score a case, printing the JSON scorecard."""
    case, scene = _load_case(case_dir)
    world = _make_world(case, scene)
    a = _agent_for(agent)
    records = run_episode(a, case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name=a.model)
    rec.case_id = case.case_id
    typer.echo(json.dumps(rec.model_dump(), indent=2))


def _print_scorecard(rec, case_id: str) -> None:
    typer.echo("=" * 56)
    typer.echo(f" TRACE-3D scorecard  case={case_id}  model={rec.model}")
    typer.echo("=" * 56)
    typer.echo(" stage scores:")
    for s in ("S", "T", "A", "G", "E", "R"):
        typer.echo(f"   {s}: {rec.stage_scores.get(s, 0.0):.3f}")
    typer.echo(f" episode_score        : {rec.episode_score:.4f}")
    typer.echo(f" deterministic_fraction: {rec.deterministic_fraction:.4f}")
    typer.echo(f" safety_violation     : {rec.safety_violation}")
    typer.echo(f" passed               : {rec.passed}")
    typer.echo(f" belief_fidelity      : {rec.belief_fidelity:.3f}")
    typer.echo(f" corridor_regret_mm   : {rec.corridor_regret_mm:.3f}")
    typer.echo(f" margin_calib_error   : {rec.margin_calibration_error:.3f}")
    typer.echo(f" overconfident_vessel : {rec.overconfident_near_vessel}")
    typer.echo("=" * 56)


@app.command()
def report(case_dir: str = DEFAULT_CASE_DIR, agent: str = "safe") -> None:
    """Run + score a case and print a human-readable scorecard."""
    case, scene = _load_case(case_dir)
    world = _make_world(case, scene)
    a = _agent_for(agent)
    records = run_episode(a, case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name=a.model)
    rec.case_id = case.case_id
    _print_scorecard(rec, case.case_id)


@app.command()
def smoke(out_dir: str = DEFAULT_CASE_DIR) -> None:
    """End-to-end: build -> run safe scripted agent -> score -> print scorecard."""
    build_synthetic_case(out_dir)
    case, scene = _load_case(out_dir)
    world = _make_world(case, scene)
    agent = SafeScriptedAgent()
    records = run_episode(agent, case, world)
    rec = score_episode(records, case.ground_truth, scene, model_name=agent.model)
    rec.case_id = case.case_id
    _print_scorecard(rec, case.case_id)


if __name__ == "__main__":
    app()

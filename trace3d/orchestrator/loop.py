"""The episode run loop.

``run_episode`` drives an agent through a STAGER episode against a world,
recording every tool call and result as an episode record, enforcing stage
legality and budgets, and collecting the agent's submissions.
"""

from __future__ import annotations

import jsonlines

from ..stager import Stage, advance, tool_is_legal
from ..world.state import BudgetError, WorldState
from ..world.tools import Tools
from .providers import Agent, ToolCall

__all__ = ["run_episode", "write_jsonl"]

# Map tool names to Tools methods.
_TOOL_METHODS = {
    "ask_patient": "ask_patient",
    "order_lab": "order_lab",
    "get_vitals": "get_vitals",
    "request_imaging_slice": "request_imaging_slice",
    "look_at": "look_at",
    "get_organ_at": "get_organ_at",
    "measure_distance": "measure_distance",
    "list_adjacent": "list_adjacent",
    "plan_trajectory": "plan_trajectory",
    "submit_triage": "submit_triage",
    "submit_govern": "submit_govern",
    "submit_diagnosis": "submit_diagnosis",
    "submit_belief": "submit_belief",
    "submit_action": "submit_action",
    "submit_reflection": "submit_reflection",
}


def _build_observation(world: WorldState, scene_summary: dict, last_result: dict | None) -> dict:
    return {
        "stage": world.stage,
        "patient_brief": world.patient_state.get("brief", {}),
        "scene_summary": scene_summary,
        "budgets": world.budgets.as_dict(),
        "sim_clock_min": world.sim_clock_min,
        "last_result": last_result,
    }


def _scene_summary(world: WorldState) -> dict:
    return {
        "node_ids": [n.id for n in world.scene.model.nodes],
        "critical_structures": world.scene.model.critical_structures,
        "midline_x_mm": world.scene.model.midline_x_mm,
    }


def _invoke(tools: Tools, call: ToolCall) -> dict:
    method = getattr(tools, _TOOL_METHODS[call.tool])
    # Tools take positional/keyword args described in args.
    args = dict(call.args)
    if call.tool in {
        "submit_triage", "submit_govern", "submit_diagnosis",
        "submit_belief", "submit_action", "submit_reflection",
    }:
        return method(args.get("payload", {}))
    if call.tool == "plan_trajectory":
        return method(args["entry_mm"], args["target_mm"])
    if call.tool == "get_organ_at":
        return method(args["point_mm"])
    if call.tool == "measure_distance":
        return method(args["node_a"], args["node_b"])
    if call.tool in {"look_at", "list_adjacent"}:
        return method(args["node_id"])
    if call.tool == "ask_patient":
        return method(args.get("question", ""))
    if call.tool == "order_lab":
        return method(args["name"])
    if call.tool == "request_imaging_slice":
        return method(args.get("axis", "axial"), args.get("index", 0), args.get("cost", 1))
    if call.tool == "get_vitals":
        return method()
    return method(**args)  # pragma: no cover - defensive


def run_episode(agent: Agent, case, world: WorldState, max_steps: int | None = None) -> list[dict]:
    """Run one episode; return the list of episode records.

    The records begin with an ``episode_start`` entry and end with an
    ``episode_end`` entry. Each tool call records ``{t, stage, type, tool,
    args, result, cost, budget_after, sim_clock_min}``. The agent's submissions
    are available afterwards via ``world.submissions``.
    """
    agent.reset()
    world.patient_state["brief"] = case.patient_brief.model_dump()
    tools = Tools(world)

    records: list[dict] = []
    records.append({
        "t": 0,
        "type": "episode_start",
        "case_id": case.case_id,
        "model": getattr(agent, "model", "unknown"),
    })

    last_result: dict | None = None
    hard_cap = max_steps if max_steps is not None else world.budgets.max_steps
    t = 0
    while world.stage != Stage.DONE.value and t < hard_cap:
        t += 1
        obs = _build_observation(world, _scene_summary(world), last_result)
        call = agent.act(obs)

        stage_enum = Stage(world.stage)
        if not tool_is_legal(stage_enum, call.tool):
            result = {"error": "illegal_tool_for_stage", "tool": call.tool, "stage": world.stage}
            records.append({
                "t": t,
                "stage": world.stage,
                "type": "illegal",
                "tool": call.tool,
                "args": call.args,
                "result": result,
                "cost": 0,
                "budget_after": world.budgets.as_dict(),
                "sim_clock_min": world.sim_clock_min,
            })
            last_result = result
            continue

        try:
            result = _invoke(tools, call)
        except BudgetError as exc:
            result = {"error": "budget", "message": str(exc)}
            records.append({
                "t": t,
                "stage": world.stage,
                "type": "budget_error",
                "tool": call.tool,
                "args": call.args,
                "result": result,
                "cost": 0,
                "budget_after": world.budgets.as_dict(),
                "sim_clock_min": world.sim_clock_min,
            })
            break

        new_stage = advance(stage_enum, call.tool)
        world.stage = new_stage.value

        records.append({
            "t": t,
            "stage": stage_enum.value,
            "type": "tool_call",
            "tool": call.tool,
            "args": call.args,
            "result": result,
            "cost": result.get("cost", 0) if isinstance(result, dict) else 0,
            "budget_after": world.budgets.as_dict(),
            "sim_clock_min": world.sim_clock_min,
        })
        last_result = result

    records.append({
        "t": t,
        "type": "episode_end",
        "final_stage": world.stage,
        "submissions": world.submissions,
        "action_order": world.action_order,
    })
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    """Write episode records to a JSON Lines file."""
    with jsonlines.open(path, mode="w") as writer:
        for rec in records:
            writer.write(rec)

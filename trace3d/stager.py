"""The STAGER stage machine.

STAGER = Survey, Triage, Assess, Govern, Execute, Reflect. The agent progresses
through stages by submitting the stage's deliverable; the stage machine tracks
which tools are legal in each stage and advances on ``submit_*`` actions.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    S = "S"  # Survey
    T = "T"  # Triage
    A = "A"  # Assess
    G = "G"  # Govern
    E = "E"  # Execute
    R = "R"  # Reflect
    DONE = "DONE"


# Order of progression.
_ORDER = [Stage.S, Stage.T, Stage.A, Stage.G, Stage.E, Stage.R, Stage.DONE]

# Legal (allowed) tools per stage. Query tools are broadly allowed; submission
# tools advance the stage.
LEGAL_TOOLS: dict[Stage, set[str]] = {
    Stage.S: {
        "ask_patient", "order_lab", "get_vitals", "request_imaging_slice",
        "look_at", "get_organ_at", "measure_distance", "list_adjacent",
        "submit_triage",
    },
    Stage.T: {
        "ask_patient", "order_lab", "get_vitals", "look_at", "measure_distance",
        "list_adjacent", "submit_diagnosis",
    },
    Stage.A: {
        "look_at", "get_organ_at", "measure_distance", "list_adjacent",
        "request_imaging_slice", "submit_belief",
    },
    Stage.G: {
        "look_at", "measure_distance", "list_adjacent", "submit_govern",
    },
    Stage.E: {
        "look_at", "get_organ_at", "measure_distance", "list_adjacent",
        "plan_trajectory", "submit_action",
    },
    Stage.R: {
        "submit_reflection",
    },
    Stage.DONE: set(),
}

# Which submission advances which stage.
SUBMIT_ADVANCES: dict[str, Stage] = {
    "submit_triage": Stage.S,
    "submit_diagnosis": Stage.T,
    "submit_belief": Stage.A,
    "submit_govern": Stage.G,
    "submit_action": Stage.E,
    "submit_reflection": Stage.R,
}


def next_stage(stage: Stage) -> Stage:
    """Return the stage following ``stage`` in STAGER order."""
    idx = _ORDER.index(stage)
    return _ORDER[min(idx + 1, len(_ORDER) - 1)]


def tool_is_legal(stage: Stage, tool: str) -> bool:
    """Whether ``tool`` may be invoked in ``stage``."""
    return tool in LEGAL_TOOLS.get(stage, set())


def advance(stage: Stage, submit_tool: str) -> Stage:
    """Advance the stage in response to a submission tool, if appropriate.

    If ``submit_tool`` is the deliverable for the current stage, returns the next
    stage; otherwise returns ``stage`` unchanged.
    """
    if SUBMIT_ADVANCES.get(submit_tool) == stage:
        return next_stage(stage)
    return stage

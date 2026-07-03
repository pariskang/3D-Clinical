"""Pydantic v2 data models for TRACE-3D.

These define the on-disk and in-memory shapes for the scene graph, the trajectory
specification, the rubric, the sealed ground truth, the case, and the per-episode
score record. Pydantic v2 conventions are used throughout: ``model_dump()`` (not
``.dict()``) and ``model_config = ConfigDict(...)`` (not ``class Config``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Side = Literal["left", "right", "midline"]
CheckKind = Literal["deterministic", "llm_judge"]


class Node(BaseModel):
    """A single anatomical structure node in the scene graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    ta_name: str  # Terminologia Anatomica style name
    centroid_mm: list[float] = Field(..., min_length=3, max_length=3)
    bbox_mm: list[list[float]]  # [[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]
    volume_mm3: float
    side: Side
    is_critical: bool = False


class Edge(BaseModel):
    """A relation between two scene-graph nodes."""

    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    distance_surface_mm: float
    adjacent: bool
    direction: str  # coarse spatial relation, e.g. "anterior", "left", ...


class SceneGraphModel(BaseModel):
    """The full anatomical scene graph derived from a labeled volume."""

    model_config = ConfigDict(extra="forbid")

    frame: str = "RAS"
    spacing_mm: list[float] = Field(..., min_length=3, max_length=3)
    midline_x_mm: float
    nodes: list[Node]
    edges: list[Edge]
    critical_structures: list[str]


class TrajectorySpec(BaseModel):
    """The clinical specification for a feasible biopsy trajectory."""

    model_config = ConfigDict(extra="forbid")

    target_point_mm: list[float] = Field(..., min_length=3, max_length=3)
    r_target_mm: float
    allowed_entry_surface: str  # named surface, e.g. "anterior_abdominal_wall"
    forbidden_structures: list[str]
    d_safe_mm: float
    L_max_mm: float
    max_angle_deg: float
    feasible_exists: bool


class RubricItem(BaseModel):
    """A single scored rubric criterion belonging to a STAGER stage."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["S", "T", "A", "G", "E", "R"]
    id: str
    criterion: str
    points: float
    check: CheckKind


class DDXItem(BaseModel):
    """A ranked differential-diagnosis entry."""

    model_config = ConfigDict(extra="forbid")

    label: str
    rank: int


class GovernGold(BaseModel):
    """Gold governance expectations for the Govern stage."""

    model_config = ConfigDict(extra="forbid")

    required_safety: list[str]
    partial_order: list[list[str]]  # list of [a, b]: a must precede b
    escalate: bool


class GroundTruth(BaseModel):
    """The sealed ground truth for a case. Never revealed to the agent."""

    model_config = ConfigDict(extra="forbid")

    lesion_true_centroid_mm: list[float] = Field(..., min_length=3, max_length=3)
    lesion_true_organ: str
    lesion_true_side: Side
    gold_urgency: str
    gold_ddx_ranked: list[DDXItem]
    gold_working_dx: str
    gold_management: str
    examiner_checklist: list[str]
    gold_govern: GovernGold
    trajectory_spec: TrajectorySpec
    gold_steps: list[str]
    gold_beliefs: dict
    rubric: list[RubricItem]


class SourceInfo(BaseModel):
    """Provenance of the imaging used to build the case."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    license: str
    modality: str
    spacing_mm: list[float] = Field(..., min_length=3, max_length=3)
    frame: str


class PatientBrief(BaseModel):
    """The non-sealed patient presentation handed to the agent."""

    model_config = ConfigDict(extra="forbid")

    age: int
    sex: str
    self_reported_race: str
    presentation: str


class Case(BaseModel):
    """A complete TRACE-3D case: brief + scene reference + sealed ground truth."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    schema_version: str = "1.0"
    task_type: str = "biopsy_trajectory"
    specialty: str
    synthetic_lesion: bool
    source: SourceInfo
    scene_graph_ref: str  # relative path to the scene graph artifacts
    patient_brief: PatientBrief
    tool_budget: dict
    fairness_variant_of: str | None = None
    ground_truth: GroundTruth


class ScoreRecord(BaseModel):
    """The per-episode scorecard produced by the scoring pipeline."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    attempt: int
    model: str
    stage_scores: dict  # {S,T,A,G,E,R}
    deterministic: dict
    judge: dict  # includes per-criterion booleans + judge_agreement
    deterministic_fraction: float
    episode_score: float
    safety_violation: bool
    passed: bool
    belief_fidelity: float
    corridor_regret_mm: float
    margin_calibration_error: float
    overconfident_near_vessel: bool

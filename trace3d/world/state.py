"""Mutable per-episode world state.

``WorldState`` bundles the scene graph, the sealed ground truth, an evolving
patient state, the tool budgets, a simulated clock, and the current STAGER stage.
It enforces budget limits and records submissions the agent makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..scene import SceneGraph
from ..schemas import GroundTruth


class BudgetError(RuntimeError):
    """Raised when an action would exceed a budget."""


@dataclass
class Budgets:
    imaging_credits: int
    labs: int
    max_steps: int
    sim_minutes: float

    @classmethod
    def from_dict(cls, d: dict) -> "Budgets":
        return cls(
            imaging_credits=int(d.get("imaging_credits", 0)),
            labs=int(d.get("labs", 0)),
            max_steps=int(d.get("max_steps", 0)),
            sim_minutes=float(d.get("sim_minutes", 0)),
        )

    def as_dict(self) -> dict:
        return {
            "imaging_credits": self.imaging_credits,
            "labs": self.labs,
            "max_steps": self.max_steps,
            "sim_minutes": self.sim_minutes,
        }


@dataclass
class WorldState:
    """The full state of a running episode."""

    scene: SceneGraph
    ground_truth: GroundTruth  # sealed; tools may read it to answer patient/lab questions
    patient_state: dict = field(default_factory=dict)
    budgets: Budgets = field(default_factory=lambda: Budgets(0, 0, 0, 0))
    sim_clock_min: float = 0.0
    stage: str = "S"
    steps_taken: int = 0
    submissions: dict = field(default_factory=dict)
    # Ordered log of action names the agent has performed (for partial-order checks).
    action_order: list[str] = field(default_factory=list)

    # ---- budget enforcement ------------------------------------------

    def tick(self, minutes: float = 1.0) -> None:
        """Advance the simulated clock and the step counter; enforce limits."""
        self.steps_taken += 1
        if self.steps_taken > self.budgets.max_steps:
            raise BudgetError("max_steps exceeded")
        self.sim_clock_min += minutes
        if self.sim_clock_min > self.budgets.sim_minutes:
            raise BudgetError("sim_minutes exceeded")

    def spend_imaging(self, cost: int = 1) -> None:
        if self.budgets.imaging_credits < cost:
            raise BudgetError("imaging_credits exhausted")
        self.budgets.imaging_credits -= cost

    def spend_lab(self, cost: int = 1) -> None:
        if self.budgets.labs < cost:
            raise BudgetError("labs exhausted")
        self.budgets.labs -= cost

    def budget_ok(self) -> bool:
        """True if no budget has been overrun."""
        return (
            self.budgets.imaging_credits >= 0
            and self.budgets.labs >= 0
            and self.steps_taken <= self.budgets.max_steps
            and self.sim_clock_min <= self.budgets.sim_minutes
        )

    def record_action(self, name: str) -> None:
        self.action_order.append(name)

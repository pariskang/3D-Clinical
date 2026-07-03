"""Agent providers.

Defines the :class:`Agent` abstract base class (``.act(observation) -> ToolCall``),
a base :class:`ScriptedAgent`, and a gated :class:`AnthropicAgent` whose anthropic
import is lazy so the offline core never requires the SDK.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool invocation an agent wants to make."""

    tool: str
    args: dict = field(default_factory=dict)


class Agent(abc.ABC):
    """Abstract agent. Implementations decide the next tool call each step."""

    name: str = "agent"
    model: str = "scripted"

    @abc.abstractmethod
    def act(self, observation: dict) -> ToolCall:
        """Given an observation, return the next :class:`ToolCall`."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any per-episode internal state. Default: no-op."""


class ScriptedAgent(Agent):
    """Base for deterministic scripted agents that follow a plan of tool calls.

    Subclasses set ``self.plan`` (a list of :class:`ToolCall`) or override
    :meth:`act`. The base implementation walks the plan in order.
    """

    name = "scripted"
    model = "scripted"

    def __init__(self) -> None:
        self.plan: list[ToolCall] = []
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def act(self, observation: dict) -> ToolCall:
        if self._i >= len(self.plan):
            # Default terminal no-op submission to keep the loop bounded.
            return ToolCall("submit_reflection", {"payload": {}})
        call = self.plan[self._i]
        self._i += 1
        return call


class AnthropicAgent(Agent):
    """LLM-backed agent using the Anthropic SDK (gated, lazy import).

    Not used by the offline core. Constructing it imports ``anthropic`` lazily and
    raises a clear error if the optional ``llm`` extra is not installed.
    """

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without llm extra
            raise RuntimeError(
                "AnthropicAgent requires the optional 'llm' extra: pip install -e '.[llm]'"
            ) from exc
        from ..config import JUDGE_MODEL

        self.model = model or JUDGE_MODEL
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def act(self, observation: dict) -> ToolCall:  # pragma: no cover - networked
        raise NotImplementedError(
            "AnthropicAgent.act is not implemented in the offline MVP; supply a "
            "tool-calling loop with model=claude-opus-4-8."
        )

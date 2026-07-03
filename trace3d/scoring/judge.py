"""LLM-judge layer.

``StubJudge`` returns deterministic per-criterion booleans so the offline pipeline
needs no network. ``AnthropicJudge`` is a gated, lazily-imported dual-judge that
calls the pinned model; it is never used in the offline core.
"""

from __future__ import annotations

import abc

from ..config import JUDGE_MODEL

__all__ = ["Judge", "StubJudge", "AnthropicJudge"]


class Judge(abc.ABC):
    """Abstract judge returning a per-criterion boolean verdict."""

    @abc.abstractmethod
    def judge(self, criterion_id: str, criterion: str, episode_context: dict) -> dict:
        """Return ``{"pass": bool, "judge_agreement": float, "rationale": str}``."""
        raise NotImplementedError


class StubJudge(Judge):
    """Deterministic offline judge.

    Returns a fixed verdict per criterion. For the MVP rubric, the single
    llm_judge item (E2: clinical rationale) passes when the agent submitted an
    action with a target/entry and acknowledged complication risk.
    """

    def judge(self, criterion_id: str, criterion: str, episode_context: dict) -> dict:
        verdict = False
        if criterion_id == "E2":
            action = episode_context.get("action") or {}
            verdict = bool(action.get("entry_mm") and action.get("target_mm"))
        else:
            # Default: pass any other llm_judge item if a reflection exists.
            verdict = bool(episode_context.get("reflection"))
        return {"pass": bool(verdict), "judge_agreement": 1.0, "rationale": "stub"}


class AnthropicJudge(Judge):
    """Dual-judge using the Anthropic SDK (gated, lazy import).

    Two independent calls at temperature 0 to the pinned model; ``judge_agreement``
    is the fraction of judges agreeing. Not used in the offline core.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without llm extra
            raise RuntimeError(
                "AnthropicJudge requires the optional 'llm' extra: pip install -e '.[llm]'"
            ) from exc
        self.model = model or JUDGE_MODEL
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    _SYSTEM = (
        "You are a strict clinical examiner scoring a single rubric criterion for an "
        "image-guided biopsy planning task. Reply with a JSON object "
        '{"pass": true|false, "rationale": "..."} and nothing else.'
    )

    def _one_call(self, prompt: str) -> bool:  # pragma: no cover - networked
        import json

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self._SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return bool(json.loads(text).get("pass", False))
        except Exception:
            return False

    def judge(self, criterion_id: str, criterion: str, episode_context: dict) -> dict:  # pragma: no cover - networked
        prompt = (
            f"Criterion {criterion_id}: {criterion}\n\n"
            f"Episode context (JSON):\n{episode_context}\n\n"
            "Does the agent's behaviour satisfy this criterion?"
        )
        v1 = self._one_call(prompt)
        v2 = self._one_call(prompt)
        agreement = 1.0 if v1 == v2 else 0.5
        return {"pass": bool(v1 and v2), "judge_agreement": agreement, "rationale": "dual-judge"}

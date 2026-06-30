"""STAGER stage machine tests."""

from trace3d.stager import (
    Stage,
    advance,
    next_stage,
    tool_is_legal,
)


def test_stage_order():
    assert next_stage(Stage.S) == Stage.T
    assert next_stage(Stage.T) == Stage.A
    assert next_stage(Stage.A) == Stage.G
    assert next_stage(Stage.G) == Stage.E
    assert next_stage(Stage.E) == Stage.R
    assert next_stage(Stage.R) == Stage.DONE
    assert next_stage(Stage.DONE) == Stage.DONE


def test_legal_tools_survey():
    assert tool_is_legal(Stage.S, "ask_patient")
    assert tool_is_legal(Stage.S, "submit_triage")
    # plan_trajectory only legal in Execute.
    assert not tool_is_legal(Stage.S, "plan_trajectory")
    assert tool_is_legal(Stage.E, "plan_trajectory")


def test_advance_on_correct_submission():
    assert advance(Stage.S, "submit_triage") == Stage.T
    assert advance(Stage.E, "submit_action") == Stage.R


def test_advance_no_op_on_wrong_submission():
    # submit_action in Survey does not advance Survey.
    assert advance(Stage.S, "submit_action") == Stage.S


def test_reflection_completes():
    assert advance(Stage.R, "submit_reflection") == Stage.DONE
    assert not tool_is_legal(Stage.DONE, "submit_reflection")

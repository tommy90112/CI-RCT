"""
Unit tests for the pipeline's run/skip decisions.

The planner is the part that can silently do the wrong thing: skipping a step
whose input just changed would mix two different training runs into one
viewer, and that failure is invisible in the output.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.process import Command, format_duration
from pipeline.runner import RUN, SKIP, plan, render_plan, render_summary, StepResult
from pipeline.stages import STAGE_ORDER, Step, filter_steps


def make_step(key, stage, outputs, depends_on=(), cwd=Path(".")):
    return Step(
        key=key,
        stage=stage,
        label=f"do {key}",
        commands=(Command(argv=("true",), cwd=cwd),),
        outputs=tuple(outputs),
        depends_on=tuple(depends_on),
    )


@pytest.fixture
def artifacts(tmp_path):
    """Two existing artifacts and one that was never produced."""
    trained = tmp_path / "model.pt"
    exported = tmp_path / "chains.json"
    trained.write_text("weights")
    exported.write_text("{}")
    return {
        "trained": trained,
        "exported": exported,
        "absent": tmp_path / "never_made.json",
    }


class TestSkipDecisions:
    def test_skips_when_outputs_exist(self, artifacts):
        steps = [make_step("train:joint", "train", [artifacts["trained"]])]
        (decision,) = plan(steps, forced=set())
        assert decision.action == SKIP
        assert "already present" in decision.reason

    def test_runs_when_an_output_is_missing(self, artifacts):
        steps = [make_step("neighbors", "neighbors", [artifacts["absent"]])]
        (decision,) = plan(steps, forced=set())
        assert decision.action == RUN
        assert "never_made.json" in decision.reason

    def test_runs_when_any_output_is_missing(self, artifacts):
        """A partially-produced step is not satisfied."""
        steps = [
            make_step(
                "evaluate:joint",
                "evaluate",
                [artifacts["exported"], artifacts["absent"]],
            )
        ]
        (decision,) = plan(steps, forced=set())
        assert decision.action == RUN

    def test_step_with_no_outputs_always_runs(self):
        steps = [make_step("frontend:build", "frontend", [])]
        (decision,) = plan(steps, forced=set())
        assert decision.action == RUN


class TestForcing:
    def test_force_by_stage_name(self, artifacts):
        steps = [make_step("train:joint", "train", [artifacts["trained"]])]
        (decision,) = plan(steps, forced={"train"})
        assert decision.action == RUN
        assert decision.reason == "forced"

    def test_force_by_step_key(self, artifacts):
        steps = [make_step("train:joint", "train", [artifacts["trained"]])]
        (decision,) = plan(steps, forced={"train:joint"})
        assert decision.action == RUN

    def test_forcing_one_stage_leaves_others_skipped(self, artifacts):
        steps = [
            make_step("train:joint", "train", [artifacts["trained"]]),
            make_step("frontend:build", "frontend", [artifacts["exported"]]),
        ]
        decisions = {p.step.key: p.action for p in plan(steps, forced={"frontend"})}
        assert decisions["train:joint"] == SKIP
        assert decisions["frontend:build"] == RUN


class TestDependencyPropagation:
    def test_downstream_reruns_when_upstream_runs(self, artifacts):
        """The core invariant: a satisfied step is still stale if its input changed."""
        steps = [
            make_step("train:joint", "train", [artifacts["trained"]]),
            make_step(
                "evaluate:joint", "evaluate", [artifacts["exported"]],
                depends_on=["train:joint"],
            ),
        ]
        decisions = {p.step.key: p for p in plan(steps, forced={"train"})}
        assert decisions["evaluate:joint"].action == RUN
        assert "upstream re-ran" in decisions["evaluate:joint"].reason
        assert "train:joint" in decisions["evaluate:joint"].reason

    def test_propagation_is_transitive(self, artifacts):
        steps = [
            make_step("train:joint", "train", [artifacts["trained"]]),
            make_step(
                "evaluate:joint", "evaluate", [artifacts["exported"]],
                depends_on=["train:joint"],
            ),
            make_step(
                "neighbors", "neighbors", [artifacts["exported"]],
                depends_on=["evaluate:joint"],
            ),
        ]
        decisions = {p.step.key: p.action for p in plan(steps, forced={"train"})}
        assert decisions["neighbors"] == RUN

    def test_no_propagation_when_upstream_is_skipped(self, artifacts):
        steps = [
            make_step("train:joint", "train", [artifacts["trained"]]),
            make_step(
                "evaluate:joint", "evaluate", [artifacts["exported"]],
                depends_on=["train:joint"],
            ),
        ]
        decisions = {p.step.key: p.action for p in plan(steps, forced=set())}
        assert decisions == {"train:joint": SKIP, "evaluate:joint": SKIP}

    def test_dependency_on_an_unlisted_step_is_ignored(self, artifacts):
        """--from evaluate drops the train steps; the plan must still work."""
        steps = [
            make_step(
                "evaluate:joint", "evaluate", [artifacts["exported"]],
                depends_on=["train:joint"],
            )
        ]
        (decision,) = plan(steps, forced=set())
        assert decision.action == SKIP


class TestStageFilter:
    @pytest.fixture
    def all_steps(self, artifacts):
        return [
            make_step("train:joint", "train", [artifacts["absent"]]),
            make_step("evaluate:joint", "evaluate", [artifacts["absent"]]),
            make_step("neighbors", "neighbors", [artifacts["absent"]]),
            make_step("frontend:build", "frontend", [artifacts["absent"]]),
        ]

    def test_from_stage_drops_earlier_stages(self, all_steps):
        kept = [s.stage for s in filter_steps(all_steps, "neighbors", None)]
        assert kept == ["neighbors", "frontend"]

    def test_only_keeps_a_single_stage(self, all_steps):
        kept = [s.key for s in filter_steps(all_steps, None, "evaluate")]
        assert kept == ["evaluate:joint"]

    def test_no_filter_keeps_everything(self, all_steps):
        assert len(filter_steps(all_steps, None, None)) == len(all_steps)

    def test_from_first_stage_is_a_no_op(self, all_steps):
        assert len(filter_steps(all_steps, STAGE_ORDER[0], None)) == len(all_steps)


class TestRendering:
    def test_plan_marks_run_and_skip_distinctly(self, artifacts):
        steps = [
            make_step("train:joint", "train", [artifacts["trained"]]),
            make_step("neighbors", "neighbors", [artifacts["absent"]]),
        ]
        text = render_plan(plan(steps, forced=set()))
        assert "SKIP" in text and "RUN" in text
        assert "train:joint" in text and "neighbors" in text

    def test_empty_plan_explains_itself(self):
        assert "nothing to do" in render_plan([])

    def test_summary_totals_only_executed_steps(self):
        results = [
            StepResult("train:joint", SKIP, 0.0),
            StepResult("neighbors", RUN, 61.0),
        ]
        text = render_summary(results)
        assert "skipped" in text
        assert "1m01s" in text


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0s"), (45, "45s"), (60, "1m00s"), (192, "3m12s"), (7500, "2h05m")],
    )
    def test_units(self, seconds, expected):
        assert format_duration(seconds) == expected

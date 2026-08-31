"""
Unit tests for pipeline step construction.

These lock down the wiring that a human would otherwise have to get right by
hand every time: which checkpoint a variant writes, which filename the viewer
reads it back from, and the two flag values that differ between training and
evaluation on purpose.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import (
    DATASET,
    PipelinePaths,
    VARIANTS,
    VARIANT_NAMES,
    variant_by_name,
)
from pipeline.process import Command
from pipeline.stages import BuildOptions, build_steps


@pytest.fixture
def paths(tmp_path):
    return PipelinePaths.from_root(tmp_path)


@pytest.fixture
def options():
    return BuildOptions(python="python3", epochs=7, device="cuda")


def steps_by_key(steps):
    return {s.key: s for s in steps}


def flags_of(step):
    """Flatten the step's single command into its argv tuple."""
    assert len(step.commands) == 1
    return step.commands[0].argv


def flag_value(argv, name):
    """Return the value following `name`, or None when the flag is absent."""
    argv = list(argv)
    if name not in argv:
        return None
    return argv[argv.index(name) + 1]


class TestVariantWiring:
    def test_transaction_checkpoint_has_no_suffix(self, paths):
        """train.py leaves the suffix empty for the transaction variant."""
        variant = variant_by_name("transaction")
        assert variant.checkpoint_path(paths, DATASET).name == (
            "ci_rct_elliptic++_best.pt"
        )

    @pytest.mark.parametrize("name", ["joint", "wallet"])
    def test_other_variants_are_suffixed(self, paths, name):
        variant = variant_by_name(name)
        assert variant.checkpoint_path(paths, DATASET).name == (
            f"ci_rct_elliptic++_{name}_best.pt"
        )

    def test_joint_dump_is_the_viewer_default(self, paths):
        """The frontend loads crime_chains.json on mount."""
        assert variant_by_name("joint").chains_path(paths).name == "crime_chains.json"

    @pytest.mark.parametrize("name", ["transaction", "wallet"])
    def test_other_dumps_are_named_per_variant(self, paths, name):
        assert variant_by_name(name).chains_path(paths).name == (
            f"crime_chains_{name}.json"
        )

    def test_exactly_one_primary_variant(self):
        assert sum(1 for v in VARIANTS if v.is_primary) == 1

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="unknown variant"):
            variant_by_name("nope")


class TestFrozenConfig:
    """The two values that must differ between train and eval, and why."""

    def test_training_uses_type_mean_baseline(self, paths, options):
        """marginal hangs during training (per-step MLP over 822k wallets)."""
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        assert flag_value(flags_of(steps["train:joint"]), "--ncm_baseline") == (
            "type_mean"
        )

    def test_evaluation_uses_marginal_baseline(self, paths, options):
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        assert flag_value(flags_of(steps["evaluate:joint"]), "--ncm_baseline") == (
            "marginal"
        )

    def test_evaluation_uses_signed_ce_tracing(self, paths, options):
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        assert flag_value(flags_of(steps["evaluate:joint"]), "--tracer_score") == (
            "ce_signed"
        )

    def test_shapley_topk_is_bounded(self, paths, options):
        """Unbounded coalitions make the phi_asym dump intractable."""
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        topk = flag_value(flags_of(steps["evaluate:joint"]), "--shapley_topk")
        assert topk is not None and int(topk) > 0

    def test_graph_flags_match_between_train_and_eval(self, paths, options):
        """A mismatched graph would not line up with the checkpoint."""
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        for flag in ("--include_addr_addr", "--dataset", "--num_hgt_layers"):
            assert flag_value(flags_of(steps["train:joint"]), flag) == (
                flag_value(flags_of(steps["evaluate:joint"]), flag)
            ), f"{flag} differs between train and evaluate"


class TestStepGraph:
    def test_every_variant_gets_a_train_and_evaluate_step(self, paths, options):
        keys = steps_by_key(build_steps(VARIANTS, paths, options))
        for name in VARIANT_NAMES:
            assert f"train:{name}" in keys
            assert f"evaluate:{name}" in keys

    def test_evaluate_depends_on_its_own_train(self, paths, options):
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        assert steps["evaluate:wallet"].depends_on == ("train:wallet",)

    def test_only_primary_variant_writes_the_csv(self, paths, options):
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        assert "--dump_csv" in flags_of(steps["evaluate:joint"])
        assert "--dump_csv" not in flags_of(steps["evaluate:wallet"])

    def test_neighbors_reads_the_primary_csv(self, paths, options):
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        chains = flag_value(flags_of(steps["neighbors"]), "--chains")
        assert Path(chains).name == "crime_chains.csv"
        assert steps["neighbors"].depends_on == ("evaluate:joint",)

    def test_neighbors_omitted_without_the_primary_variant(self, paths, options):
        """Nothing would produce the CSV, so the step must not silently reuse one."""
        wallet_only = (variant_by_name("wallet"),)
        keys = steps_by_key(build_steps(wallet_only, paths, options))
        assert "neighbors" not in keys

    def test_frontend_build_waits_for_every_export(self, paths, options):
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        deps = set(steps["frontend:build"].depends_on)
        assert {"frontend:deps", "neighbors"} <= deps
        assert all(f"evaluate:{n}" in deps for n in VARIANT_NAMES)

    def test_frontend_can_be_excluded(self, paths, options):
        steps = build_steps(VARIANTS, paths, options, include_frontend=False)
        assert not any(s.stage == "frontend" for s in steps)

    def test_no_variants_is_rejected(self, paths, options):
        with pytest.raises(ValueError, match="at least one variant"):
            build_steps((), paths, options)

    def test_options_flow_into_the_commands(self, paths):
        options = BuildOptions(python="/custom/py", epochs=3, device="mps")
        steps = steps_by_key(build_steps(VARIANTS, paths, options))
        argv = flags_of(steps["train:joint"])
        assert argv[0] == "/custom/py"
        assert flag_value(argv, "--epochs") == "3"
        assert flag_value(argv, "--device") == "mps"

    def test_debug_flag_is_toggleable(self, paths):
        on = BuildOptions(python="py", epochs=1, device="cpu", debug=True)
        off = BuildOptions(python="py", epochs=1, device="cpu", debug=False)
        steps_on = steps_by_key(build_steps(VARIANTS, paths, on))
        steps_off = steps_by_key(build_steps(VARIANTS, paths, off))
        assert "--debug" in flags_of(steps_on["evaluate:joint"])
        assert "--debug" not in flags_of(steps_off["evaluate:joint"])

    def test_steps_run_in_dependency_order(self, paths, options):
        """Every dependency must appear before the step that needs it."""
        steps = build_steps(VARIANTS, paths, options)
        position = {s.key: i for i, s in enumerate(steps)}
        for step in steps:
            for dep in step.depends_on:
                if dep in position:
                    assert position[dep] < position[step.key], (
                        f"{step.key} runs before its dependency {dep}"
                    )


class TestCommandDisplay:
    def test_quotes_only_when_needed(self):
        command = Command(argv=("python", "-u", "train.py"), cwd=Path("/tmp"))
        assert "'" not in command.display()

    def test_quotes_paths_containing_spaces(self):
        command = Command(argv=("py", "--out", "/a b/c.json"), cwd=Path("/tmp"))
        assert "'/a b/c.json'" in command.display()

    def test_long_commands_wrap(self):
        command = Command(argv=tuple(["python"] + [f"--flag{i}" for i in range(40)]),
                          cwd=Path("/tmp"))
        assert "\\\n" in command.display()

"""
Tests for utils.smt2020_loader — GraphTables → HeteroData and the on-disk
loading contract used by train.py / evaluate.py.

torch_geometric is imported lazily and the whole module is skipped when a
subprocess cannot import it: on some machines the PyG extensions segfault at
import, which pytest cannot catch, so a plain importorskip would take the
whole session down.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.smt2020_toy import toy_config, toy_events, toy_lots, toy_runs
from utils.smt2020_excursion import UNKNOWN_LABEL, inject_excursions, write_injection
from utils.smt2020_graph import EDGE_TYPES, GraphConfig, build_graph_tables, compute_local_offsets
from utils.smt2020_gt import compute_excursion_ground_truth, ground_truth_to_global


def _pyg_importable() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import torch_geometric"], capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _pyg_importable(), reason="torch_geometric cannot be imported here")

CFG = GraphConfig(window_hours=4.0, metrology_signal=3.0)


@pytest.fixture(scope="module")
def toy_disk(tmp_path_factory) -> Path:
    """Toy trace + injection written in the on-disk layout."""
    root = tmp_path_factory.mktemp("smt2020")
    runs, events, lots = toy_runs(), toy_events(), toy_lots()
    runs.to_csv(root / "lot_trace.csv", index=False)
    events.to_csv(root / "tool_events.csv", index=False)
    lots.to_csv(root / "lots.csv", index=False)
    write_injection(inject_excursions(runs, events, lots, toy_config()), root / "excursion_seed0")
    return root


@pytest.fixture(scope="module")
def tables_and_inj():
    runs, events, lots = toy_runs(), toy_events(), toy_lots()
    inj = inject_excursions(runs, events, lots, toy_config())
    return build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions, CFG), inj


def test_heterodata_mirrors_tables(tables_and_inj):
    import torch
    from utils.smt2020_loader import graph_tables_to_heterodata
    tables, _ = tables_and_inj
    data = graph_tables_to_heterodata(tables)
    assert set(data.node_types) == set(tables.nodes)
    for ntype, node in tables.nodes.items():
        store = data[ntype]
        assert store.num_nodes == len(node.ids)
        assert torch.equal(store.x, torch.from_numpy(node.x)) and store.x.dtype == torch.float32
        assert torch.equal(store.time, torch.from_numpy(node.time)) and store.time.dtype == torch.int64
        if node.y is not None:
            assert torch.equal(store.y, torch.from_numpy(np.where(node.y < 0, 0, node.y)))
            assert int(store.y.min()) >= 0
    for split in ("train", "val", "test"):
        assert torch.equal(data["run"][f"{split}_mask"], torch.from_numpy(tables.masks["run"][split]))
    assert not hasattr(data["tool_state"], "y")
    for name, ei in tables.edges.items():
        src, dst = EDGE_TYPES[name]
        assert torch.equal(data[(src, name, dst)].edge_index, torch.from_numpy(ei))
    assert len(data.edge_types) == len(EDGE_TYPES)


def test_type_offsets_agree_with_local_offsets(tables_and_inj):
    from utils.data_utils import compute_type_offsets
    from utils.smt2020_loader import graph_tables_to_heterodata
    tables, _ = tables_and_inj
    assert compute_type_offsets(graph_tables_to_heterodata(tables)) == compute_local_offsets(tables)


def test_causal_graph_keeps_every_edge_under_the_temporal_guard(tables_and_inj):
    from utils.data_utils import build_typed_causal_graph_from_hetero
    from utils.smt2020_loader import graph_tables_to_heterodata
    tables, _ = tables_and_inj
    data = graph_tables_to_heterodata(tables)
    n_total = sum(len(n.ids) for n in tables.nodes.values())
    tcg = build_typed_causal_graph_from_hetero(data, seed_node_ids=list(range(n_total)),
                                               hop_limit=0, node_limit=n_total)
    assert len(tcg.v) == n_total
    n_edges = sum(len(p) for p in tcg.pa.values())
    assert n_edges == sum(ei.shape[1] for ei in tables.edges.values())
    assert len(tcg.topological_order()) == n_total


def test_load_from_disk_returns_target_run(toy_disk):
    from utils.smt2020_loader import load_smt2020_dataset
    data, target = load_smt2020_dataset(toy_disk, window_hours=4.0, drop_before_days=0.0)
    assert target == "run"
    y = data["run"].y
    assert int(y.min()) == 0                             # unknown → 0 placeholder, never -1
    masks = data["run"].train_mask | data["run"].val_mask | data["run"].test_mask
    assert int(masks.sum()) == 20                        # exactly the metrology runs are labelled
    assert bool((y[~masks] == 0).all())
    assert int((y == 1).sum()) == 7                      # lot 0: 3 observed, lot 2: 4 observed
    assert data["lot"].y.shape[0] == 4 and int(data["lot"].y.min()) == 0


def test_ground_truth_globals_match_direct_computation(toy_disk, tables_and_inj):
    from utils.data_utils import compute_type_offsets
    from utils.smt2020_loader import (load_smt2020_anomaly_entities, load_smt2020_dataset,
                                      load_smt2020_ground_truth)
    tables, inj = tables_and_inj
    data, _ = load_smt2020_dataset(toy_disk, window_hours=4.0, drop_before_days=0.0)
    offsets = compute_type_offsets(data)
    strict = load_smt2020_ground_truth(toy_disk, "excursion_seed0", CFG, offsets, "strict")
    extended = load_smt2020_ground_truth(toy_disk, "excursion_seed0", CFG, offsets, "extended")
    gt = compute_excursion_ground_truth(tables, inj.run_labels, inj.gt_runs, inj.excursions)
    exp_strict, exp_ext = ground_truth_to_global(gt, offsets)
    assert strict == exp_strict and extended == exp_ext
    assert len(strict) == 7
    entities = load_smt2020_anomaly_entities(toy_disk, "excursion_seed0", CFG, offsets)
    assert entities == {offsets[t] + i for t, i in gt.anomaly_entities}
    n_total = sum(len(n.ids) for n in tables.nodes.values())
    assert all(0 <= g < n_total for g in entities)


def test_offset_mismatch_is_rejected(toy_disk):
    from utils.smt2020_loader import load_smt2020_ground_truth
    with pytest.raises(ValueError, match="type_offsets"):
        load_smt2020_ground_truth(toy_disk, "excursion_seed0", CFG, {"lot": 0, "run": 1, "tool_state": 2}, "strict")


def test_missing_files_are_reported(tmp_path):
    from utils.smt2020_loader import load_smt2020_dataset
    with pytest.raises(FileNotFoundError, match="lot_trace.csv"):
        load_smt2020_dataset(tmp_path)


def test_bad_mode_is_rejected(toy_disk):
    from utils.smt2020_loader import load_smt2020_ground_truth
    with pytest.raises(ValueError, match="mode"):
        load_smt2020_ground_truth(toy_disk, "excursion_seed0", CFG, {}, "loose")


def test_train_and_evaluate_dispatch_smt2020(toy_disk):
    """train.py / evaluate.py load_dataset must route --dataset smt2020 to the loader."""
    import argparse
    import evaluate as evaluate_mod
    import train as train_mod
    from utils.smt2020_cli import add_smt2020_args, smt2020_loader_kwargs

    parser = argparse.ArgumentParser()
    add_smt2020_args(parser)
    args = parser.parse_args(["--smt2020_dir", str(toy_disk), "--smt2020_window_hours", "4",
                              "--smt2020_drop_before_days", "0"])
    args.seed = 0
    kwargs = smt2020_loader_kwargs(args)
    for mod in (train_mod, evaluate_mod):
        data, target = mod.load_dataset("smt2020", "data", smt2020=kwargs)
        assert target == "run"
        assert set(data.node_types) == {"lot", "run", "tool_state"}
    with pytest.raises(ValueError, match="Unknown dataset"):
        train_mod.load_dataset("nope", "data")


def test_aux_wallet_labels_never_creates_a_phantom_store(tables_and_inj):
    import train as train_mod
    from utils.smt2020_loader import graph_tables_to_heterodata
    tables, _ = tables_and_inj
    data = graph_tables_to_heterodata(tables)
    assert train_mod._aux_wallet_labels(data) is None
    assert "wallet" not in data.node_types

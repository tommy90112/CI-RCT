"""Unit tests for utils.chain_export — flat CSV export of traced chains."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.chain_export import (  # noqa: E402
    CSV_FIELDNAMES,
    chain_record_to_row,
    chain_records_to_rows,
    csv_fieldnames,
    write_chains_csv,
)


def _sample_records():
    """A 3-hop true-positive chain and a singleton (depth-0) chain."""
    return [
        {
            "target_txid": "txTARGET",
            "depth": 2,
            "root_type": "wallet",
            "root_real_id": "addrROOT",
            "root_is_fraud": True,
            "is_true_positive": True,
            "nodes": [
                {"pos": 0, "type": "transaction", "real_id": "txTARGET",
                 "fraud": True, "is_target": True},
                {"pos": 1, "type": "transaction", "real_id": "txMID",
                 "fraud": False, "ce": 0.83},
                {"pos": 2, "type": "wallet", "real_id": "addrROOT",
                 "fraud": True, "ce": 0.51},
            ],
        },
        {
            "target_txid": "txSOLO",
            "depth": 0,
            "root_type": "transaction",
            "root_real_id": "txSOLO",
            "root_is_fraud": False,
            "is_true_positive": False,
            "nodes": [
                {"pos": 0, "type": "transaction", "real_id": "txSOLO",
                 "fraud": False, "is_target": True},
            ],
        },
    ]


def test_one_row_per_chain_with_expected_columns():
    rows = chain_records_to_rows(_sample_records())
    assert len(rows) == 2
    assert tuple(rows[0].keys()) == CSV_FIELDNAMES


def test_path_type_and_ce_are_position_aligned():
    row = chain_record_to_row(_sample_records()[0])
    assert row["chain_real_ids"] == "txTARGET|txMID|addrROOT"
    assert row["chain_types"] == "transaction|transaction|wallet"
    # target cell is empty (no incoming edge), then downstream→upstream CE values
    assert row["chain_ce"] == "|0.83|0.51"
    assert row["n_nodes"] == 3
    assert row["depth"] == 2
    assert row["root_type"] == "wallet"
    assert row["root_is_fraud"] is True
    assert row["is_true_positive"] is True


def test_singleton_chain_has_empty_ce():
    row = chain_record_to_row(_sample_records()[1])
    assert row["chain_real_ids"] == "txSOLO"
    assert row["chain_ce"] == ""
    assert row["n_nodes"] == 1
    assert row["depth"] == 0


def test_input_records_are_not_mutated():
    records = _sample_records()
    before_keys = [set(r.keys()) for r in records]
    chain_records_to_rows(records)
    after_keys = [set(r.keys()) for r in records]
    assert before_keys == after_keys


def test_write_csv_roundtrip(tmp_path):
    out = os.path.join(str(tmp_path), "nested", "chains.csv")
    n = write_chains_csv(_sample_records(), out)
    assert n == 2
    assert os.path.isfile(out)  # parent dir created
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == list(CSV_FIELDNAMES)
    assert len(rows) == 2
    assert rows[0]["target_txid"] == "txTARGET"
    assert rows[0]["chain_ce"] == "|0.83|0.51"
    assert rows[1]["chain_real_ids"] == "txSOLO"


def test_empty_records_writes_header_only(tmp_path):
    out = os.path.join(str(tmp_path), "empty.csv")
    n = write_chains_csv([], out)
    assert n == 0
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [list(CSV_FIELDNAMES)]


def _records_with_phi():
    """The 3-hop chain with per-node φ attached (as utils.chain_phi would)."""
    recs = _sample_records()
    recs[0]["nodes"][1]["phi_add"] = 0.4
    recs[0]["nodes"][1]["phi_asym"] = -0.5
    recs[0]["nodes"][2]["phi_add"] = 0.25
    recs[0]["nodes"][2]["phi_asym"] = 0.9
    return recs


def test_phi_columns_absent_without_phi():
    assert csv_fieldnames(_sample_records()) == list(CSV_FIELDNAMES)


def test_phi_columns_appended_when_present():
    cols = csv_fieldnames(_records_with_phi())
    assert cols[:len(CSV_FIELDNAMES)] == list(CSV_FIELDNAMES)
    assert cols[len(CSV_FIELDNAMES):] == ["chain_phi_add", "chain_phi_asym"]


def test_phi_cells_are_position_aligned_and_signed():
    row = chain_record_to_row(
        _records_with_phi()[0],
        [("phi_add", "chain_phi_add"), ("phi_asym", "chain_phi_asym")],
    )
    # target cell empty, then downstream→upstream; sign preserved
    assert row["chain_phi_add"] == "|0.4|0.25"
    assert row["chain_phi_asym"] == "|-0.5|0.9"


def test_write_csv_roundtrip_with_phi(tmp_path):
    out = os.path.join(str(tmp_path), "phi.csv")
    write_chains_csv(_records_with_phi(), out)
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert "chain_phi_add" in reader.fieldnames
        assert "chain_phi_asym" in reader.fieldnames
    assert rows[0]["chain_phi_asym"] == "|-0.5|0.9"

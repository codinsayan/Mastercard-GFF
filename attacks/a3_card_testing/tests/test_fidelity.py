from __future__ import annotations

import json

import numpy as np
import pandas as pd

from attacks.a3_card_testing.fidelity import (
    FIDELITY_FEATURES,
    FidelityConfig,
    blocked_result,
    run_fidelity,
    write_result,
)


def test_sparkov_fidelity_produces_all_required_metrics(tmp_path):
    rows = 120
    source = tmp_path / "fraudTrain.csv"
    pd.DataFrame(
        {
            "amt": np.linspace(10, 500, rows),
            "trans_date_trans_time": pd.date_range("2024-01-01", periods=rows, freq="3h"),
            "category": np.resize(["grocery", "travel", "online"], rows),
            "is_fraud": np.resize([0, 0, 0, 1], rows),
        }
    ).to_csv(source, index=False)
    output = tmp_path / "fidelity.json"

    result = run_fidelity(
        source,
        dataset_format="sparkov",
        output_path=output,
        config=FidelityConfig(seed=7, max_rows=80),
    )

    assert result["status"] == "complete"
    assert result["dataset"]["format"] == "sparkov"
    assert set(result["ks_statistics"]) == set(FIDELITY_FEATURES)
    assert 0.0 <= result["correlation_matrix_distance"]["value"] <= 2.0
    assert 0.0 <= result["discriminator"]["roc_auc"] <= 1.0
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_ieee_cis_directory_and_format_detection_are_supported(tmp_path):
    dataset = tmp_path / "ieee"
    dataset.mkdir()
    rows = 100
    pd.DataFrame(
        {
            "TransactionAmt": np.linspace(5, 300, rows),
            "TransactionDT": np.arange(rows) * 3600,
            "ProductCD": np.resize(["W", "C", "R"], rows),
            "isFraud": np.resize([0, 0, 0, 0, 1], rows),
        }
    ).to_csv(dataset / "train_transaction.csv", index=False)

    result = run_fidelity(
        dataset,
        dataset_format="auto",
        output_path=tmp_path / "ieee-fidelity.json",
        config=FidelityConfig(seed=9, max_rows=60),
    )

    assert result["status"] == "complete"
    assert result["dataset"]["format"] == "ieee-cis"
    assert result["dataset"]["rows_sampled"] == 60


def test_missing_dataset_result_is_explicit_and_contains_no_metrics(tmp_path):
    output = tmp_path / "fidelity.json"
    result = blocked_result()
    write_result(output, result)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "blocked_no_dataset"
    assert saved["ks_statistics"] is None
    assert saved["correlation_matrix_distance"] is None
    assert saved["discriminator"] is None

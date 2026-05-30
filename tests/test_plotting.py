from __future__ import annotations

from pathlib import Path

import pandas as pd

from asr_data_selection.manifests import results_root
from asr_data_selection.plotting import load_summary, run


def current_rows(config: dict) -> list[dict]:
    return [
        {
            "feature_set": feature,
            "model": "mlp",
            "accuracy": 0.75,
            "macro_f1": 0.74,
            "weighted_f1": 0.74,
            "feature_dim": 256,
            "train_rows": 2,
            "test_rows": 2,
            "status": "ok",
            "reason": "",
        }
        for feature in config["classification"]["feature_sets"]
    ]


def test_plotter_requires_all_current_rows(small_config: dict) -> None:
    path = Path(small_config["artifacts"]["root"]) / "summary.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(current_rows(small_config)[:-1]).to_csv(path, index=False)
    try:
        load_summary(path, small_config)
    except ValueError as error:
        assert "every configured feature set" in str(error)
    else:
        raise AssertionError("Expected incomplete summary to fail")


def test_plotter_writes_accuracy_chart_and_compact_csv(small_config: dict) -> None:
    root = results_root(small_config)
    root.mkdir(parents=True)
    pd.DataFrame(current_rows(small_config)).to_csv(root / "summary.csv", index=False)
    output = run(small_config)
    assert output.exists()
    csv = pd.read_csv(output.with_suffix(".csv"))
    assert len(csv) == 11
    assert "macro_f1" in csv.columns

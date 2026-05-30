from __future__ import annotations

import numpy as np
import pandas as pd

from asr_data_selection import classification
from asr_data_selection.manifests import embeddings_root, results_root, write_jsonl


def test_classifier_fits_scaler_on_train_only(monkeypatch, small_config: dict) -> None:
    small_config["classification"]["feature_sets"] = ["sbert_gt"]
    root = embeddings_root(small_config)
    train_rows = [{"utt_id": "a", "domain": "alpha"}, {"utt_id": "b", "domain": "beta"}]
    test_rows = [{"utt_id": "c", "domain": "alpha"}, {"utt_id": "d", "domain": "beta"}]
    write_jsonl(train_rows, root / "manifests" / "train.jsonl")
    write_jsonl(test_rows, root / "manifests" / "test.jsonl")
    matrix_dir = root / "sbert_gt"
    matrix_dir.mkdir(parents=True)
    np.save(matrix_dir / "train.npy", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(matrix_dir / "test.npy", np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float32))
    seen = {}

    class TrackingScaler:
        def fit_transform(self, matrix):
            seen["fit_rows"] = len(matrix)
            return matrix

        def transform(self, matrix):
            seen["transform_rows"] = len(matrix)
            return matrix

    monkeypatch.setattr(classification, "StandardScaler", TrackingScaler)
    monkeypatch.setattr(classification, "train_mlp", lambda *_args, **_kwargs: np.array([0, 1]))
    output = classification.run(small_config)
    summary = pd.read_csv(output)
    assert seen == {"fit_rows": 2, "transform_rows": 2}
    assert summary.loc[0, "accuracy"] == 1.0
    assert output == results_root(small_config) / "summary.csv"

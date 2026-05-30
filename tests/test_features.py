from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asr_data_selection import features
from asr_data_selection.manifests import embeddings_root, subset_manifest_path, write_json_rows


def test_projection_and_fusion_dimensions() -> None:
    raw = {"train": np.ones((2, 4), dtype=np.float32), "test": np.ones((3, 4), dtype=np.float32)}
    projected, _ = features.project_embeddings(raw, dim=2, seed=42)
    fused = features.fuse_features({"a": projected, "b": projected}, {"a_b": ["a", "b"]})
    assert projected["train"].shape == (2, 2)
    assert fused["a_b"]["test"].shape == (3, 4)


class FakeAcoustic:
    dim = 3

    def __init__(self, *args, **kwargs) -> None:
        pass

    def extract(self, paths: list[Path]) -> np.ndarray:
        return np.arange(len(paths) * self.dim, dtype=np.float32).reshape(len(paths), self.dim)


class FakeMFA(FakeAcoustic):
    dim = 4


class FakeSBERT:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def extract(self, texts: list[str]) -> np.ndarray:
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)


@pytest.mark.parametrize("save_raw", [False, True])
def test_run_writes_explicit_outputs_without_legacy_aliases(monkeypatch, small_config: dict, pseudo_rows: list[dict], save_raw: bool) -> None:
    small_config["embeddings"]["save_raw_embeddings"] = save_raw
    monkeypatch.setattr(features, "WavLMEncoder", FakeAcoustic)
    monkeypatch.setattr(features, "MFAConformerEncoder", FakeMFA)
    monkeypatch.setattr(features, "SBERTEncoder", FakeSBERT)
    write_json_rows(pseudo_rows, subset_manifest_path(small_config, pseudolabels=True))
    for row in pseudo_rows:
        path = Path(small_config["artifacts"]["root"]) / row["audio_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    root = features.run(small_config)
    assert (root / "sbert_gt" / "train.npy").exists()
    assert (root / "sbert_pseudo_wavlm_mfa" / "test.npy").exists()
    assert not (root / "sbert").exists()
    assert not (root / "sbert_mfa").exists()
    assert (root / "wavlm_raw").exists() is save_raw

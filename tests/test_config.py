from __future__ import annotations

from pathlib import Path

import yaml

from asr_data_selection.config import artifact_root, load_config, write_resolved_config


def test_load_config_applies_dotted_overrides() -> None:
    config = load_config(overrides=["classification.epochs=1", "embeddings.save_raw_embeddings=true"])
    assert config["classification"]["epochs"] == 1
    assert config["embeddings"]["save_raw_embeddings"] is True


def test_write_resolved_config_uses_artifact_root(small_config: dict) -> None:
    output = write_resolved_config(small_config)
    assert output == artifact_root(small_config) / "resolved_config.yaml"
    snapshot = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert snapshot["experiment"]["name"] == "embedding_comparison_13200"
    assert "_meta" not in snapshot


def test_unknown_override_is_rejected() -> None:
    try:
        load_config(overrides=["classification.not_a_setting=1"])
    except KeyError as error:
        assert "classification.not_a_setting" in str(error)
    else:
        raise AssertionError("Expected unknown override to fail")

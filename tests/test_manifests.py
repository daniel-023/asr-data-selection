from __future__ import annotations

from pathlib import Path

from asr_data_selection.manifests import (
    embeddings_root,
    resolve_audio_path,
    subset_manifest_path,
    validate_subset_rows,
    write_aligned_manifests,
)


def test_minimal_aligned_manifests_only_write_train_and_test(small_config: dict, pseudo_rows: list[dict]) -> None:
    validate_subset_rows(pseudo_rows, small_config, require_pseudo=True)
    write_aligned_manifests(pseudo_rows, small_config)
    manifest_root = embeddings_root(small_config) / "manifests"
    assert sorted(path.name for path in manifest_root.iterdir()) == ["test.jsonl", "train.jsonl"]


def test_manifest_paths_are_relative_to_artifact_root(small_config: dict) -> None:
    assert subset_manifest_path(small_config).name == "selected_manifest.json"
    assert resolve_audio_path("audio/example.wav", small_config) == Path(small_config["artifacts"]["root"]) / "audio" / "example.wav"

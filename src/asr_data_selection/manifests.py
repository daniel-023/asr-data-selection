from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import artifact_path


SUBSET_COLUMNS = {
    "utt_id",
    "orig_id",
    "source_sample_key",
    "domain",
    "split",
    "source_split",
    "audio_path",
    "transcript",
    "transcript_norm",
    "duration_sec",
}
PSEUDO_COLUMNS = {
    "pseudo_transcript",
    "pseudo_transcript_norm",
    "pseudo_model",
    "pseudo_status",
    "pseudo_error",
}


def read_json_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    return rows


def write_json_rows(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_rows(rows: list[dict], required: set[str], source: Path) -> None:
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest row {index} in {source} is missing columns: {sorted(missing)}")


def expected_split_rows(config: dict, role: str) -> int:
    dataset = config["dataset"]
    per_domain = int(dataset[f"{role}_per_domain"])
    return per_domain * len(dataset["domains"])


def validate_subset_rows(rows: list[dict], config: dict, *, require_pseudo: bool = False) -> None:
    required = SUBSET_COLUMNS | (PSEUDO_COLUMNS if require_pseudo else set())
    source = subset_manifest_path(config, pseudolabels=require_pseudo)
    validate_rows(rows, required, source)
    domains = {entry["domain"] for entry in config["dataset"]["domains"]}
    actual_domains = {str(row["domain"]) for row in rows}
    if actual_domains != domains:
        raise ValueError(f"Expected domains {sorted(domains)}, found {sorted(actual_domains)}")
    expected_total = 0
    for role in ("reference", "candidate"):
        expected = expected_split_rows(config, role)
        actual = sum(str(row["split"]) == role for row in rows)
        if actual != expected:
            raise ValueError(f"Expected {expected} {role} rows, found {actual}")
        expected_total += expected
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} manifest rows, found {len(rows)}")


def subset_manifest_path(config: dict, *, pseudolabels: bool = False) -> Path:
    metadata = artifact_path(config, "metadata_dir")
    filename = "selected_manifest.whisper_large_v3.json" if pseudolabels else "selected_manifest.json"
    return metadata / filename


def checkpoint_manifest_path(config: dict) -> Path:
    return artifact_path(config, "metadata_dir") / "selected_manifest.whisper_large_v3.partial.jsonl"


def embeddings_root(config: dict) -> Path:
    return artifact_path(config, "embeddings_dir")


def results_root(config: dict) -> Path:
    return artifact_path(config, "results_dir")


def resolve_audio_path(raw_path: str, config: dict) -> Path:
    direct = Path(raw_path).expanduser()
    if direct.is_absolute():
        return direct.resolve()
    return (artifact_path(config, "audio_dir").parent / direct).resolve()


def aligned_splits(rows: list[dict], config: dict) -> dict[str, list[dict]]:
    train_role = config["dataset"]["train_role"]
    eval_role = config["dataset"]["eval_role"]
    return {
        "train": [row for row in rows if row["split"] == train_role],
        "test": [row for row in rows if row["split"] == eval_role],
    }


def write_aligned_manifests(rows: list[dict], config: dict) -> dict[str, list[dict]]:
    splits = aligned_splits(rows, config)
    root = embeddings_root(config) / "manifests"
    for split, split_rows in splits.items():
        write_jsonl(split_rows, root / f"{split}.jsonl")
    return splits

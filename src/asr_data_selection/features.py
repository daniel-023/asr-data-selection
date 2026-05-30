from __future__ import annotations

import gc
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.random_projection import GaussianRandomProjection

from .config import resolve_repo_path
from .encoders import MFAConformerEncoder, SBERTEncoder, WavLMEncoder, resolve_device
from .manifests import (
    embeddings_root,
    resolve_audio_path,
    subset_manifest_path,
    validate_subset_rows,
    read_json_rows,
    write_aligned_manifests,
)


SPLITS = ("train", "test")
SEMANTIC_COLUMNS = {"gt": "transcript_norm", "pseudo": "pseudo_transcript_norm"}
SEMANTIC_FEATURES = {"gt": "sbert_gt", "pseudo": "sbert_pseudo"}
ACOUSTIC_FEATURES = {"wavlm": "wavlm_256", "mfa_conformer": "mfa_conformer_256"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_embedding(name: str, split: str, matrix: np.ndarray, expected_rows: int, expected_dim: int | None = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name}/{split} must be 2D, found {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise ValueError(f"{name}/{split} expected {expected_rows} rows, found {matrix.shape[0]}")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ValueError(f"{name}/{split} expected dim={expected_dim}, found {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name}/{split} contains NaN or Inf values")
    return matrix


def save_matrix(root: Path, feature_set: str, split: str, matrix: np.ndarray) -> Path:
    output = root / feature_set / f"{split}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(matrix, dtype=np.float32))
    print(f"[save] {feature_set}/{split}: {matrix.shape}")
    return output


def project_embeddings(raw: dict[str, np.ndarray], dim: int, seed: int) -> tuple[dict[str, np.ndarray], GaussianRandomProjection]:
    stacked = np.vstack([raw[split] for split in SPLITS])
    projection = GaussianRandomProjection(n_components=dim, random_state=seed)
    projection.fit(stacked)
    return {split: projection.transform(raw[split]).astype(np.float32) for split in SPLITS}, projection


def save_projection(root: Path, name: str, projection: GaussianRandomProjection) -> Path:
    output = root / "projections" / f"{name}.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(projection, output)
    return output


def fuse_features(arrays: dict[str, dict[str, np.ndarray]], fusion: dict[str, list[str]]) -> dict[str, dict[str, np.ndarray]]:
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for name, components in fusion.items():
        missing = [component for component in components if component not in arrays]
        if missing:
            raise ValueError(f"Fusion {name} is missing components: {missing}")
        outputs[name] = {
            split: np.concatenate([arrays[component][split] for component in components], axis=1).astype(np.float32)
            for split in SPLITS
        }
    return outputs


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run(config: dict) -> Path:
    settings = config["embeddings"]
    seed = int(settings["projection_seed"])
    set_seed(seed)
    device = resolve_device(settings["device"])
    root = embeddings_root(config)
    rows = read_json_rows(subset_manifest_path(config, pseudolabels=True))
    validate_subset_rows(rows, config, require_pseudo=True)
    split_rows = write_aligned_manifests(rows, config)
    split_paths = {
        split: [resolve_audio_path(str(row["audio_path"]), config) for row in current_rows]
        for split, current_rows in split_rows.items()
    }
    missing = [path for paths in split_paths.values() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing audio file: {missing[0]}")
    dim = int(settings["projection_dim"])
    batch_size = int(settings["batch_size"])
    workers = int(settings["num_workers"])
    save_raw = bool(settings["save_raw_embeddings"])
    arrays: dict[str, dict[str, np.ndarray]] = {}

    acoustic_configs = settings["encoders"]
    acoustic_factories = {
        "wavlm": lambda: WavLMEncoder(acoustic_configs["wavlm"]["model_id"], device, batch_size, workers),
        "mfa_conformer": lambda: MFAConformerEncoder(
            resolve_repo_path(acoustic_configs["mfa_conformer"]["source_dir"]),
            resolve_repo_path(acoustic_configs["mfa_conformer"]["checkpoint"]),
            device,
            batch_size,
            workers,
        ),
    }
    for encoder_name, factory in acoustic_factories.items():
        encoder = factory()
        raw = {
            split: validate_embedding(
                f"{encoder_name}_raw",
                split,
                encoder.extract(split_paths[split]),
                len(split_rows[split]),
                int(acoustic_configs[encoder_name]["raw_dim"]),
            )
            for split in SPLITS
        }
        if save_raw:
            for split in SPLITS:
                save_matrix(root, f"{encoder_name}_raw", split, raw[split])
        projected, projection = project_embeddings(raw, dim, seed)
        feature = ACOUSTIC_FEATURES[encoder_name]
        arrays[feature] = {
            split: validate_embedding(feature, split, projected[split], len(split_rows[split]), dim)
            for split in SPLITS
        }
        save_projection(root, f"{encoder_name}_grp_{raw['train'].shape[1]}_to_{dim}", projection)
        for split in SPLITS:
            save_matrix(root, feature, split, arrays[feature][split])
        del encoder
        clear_device(device)

    semantic_sources = list(settings["semantic_sources"])
    for source in semantic_sources:
        column = SEMANTIC_COLUMNS[source]
        empty = [row["utt_id"] for row in rows if not str(row.get(column, "")).strip()]
        if empty:
            raise ValueError(f"Semantic source {source} has empty text for {len(empty)} rows: {empty[:5]}")
    sbert = SBERTEncoder(acoustic_configs["sbert"]["model_id"], device, int(settings["sbert_batch_size"]))
    semantic_raw = {
        source: {
            split: validate_embedding(
                f"sbert_{source}_raw",
                split,
                sbert.extract([str(row[SEMANTIC_COLUMNS[source]]) for row in split_rows[split]]),
                len(split_rows[split]),
            )
            for split in SPLITS
        }
        for source in semantic_sources
    }
    if save_raw:
        for source in semantic_sources:
            for split in SPLITS:
                save_matrix(root, f"sbert_{source}_raw", split, semantic_raw[source][split])
    stacked = np.vstack([semantic_raw[source][split] for source in semantic_sources for split in SPLITS])
    projection = GaussianRandomProjection(n_components=dim, random_state=seed)
    projection.fit(stacked)
    save_projection(root, f"sbert_{'_'.join(semantic_sources)}_grp_{stacked.shape[1]}_to_{dim}", projection)
    for source in semantic_sources:
        feature = SEMANTIC_FEATURES[source]
        arrays[feature] = {}
        for split in SPLITS:
            matrix = projection.transform(semantic_raw[source][split]).astype(np.float32)
            arrays[feature][split] = validate_embedding(feature, split, matrix, len(split_rows[split]), dim)
            save_matrix(root, feature, split, arrays[feature][split])
    del sbert
    clear_device(device)

    fused = fuse_features(arrays, settings["fusion"])
    for feature, matrices in fused.items():
        arrays[feature] = matrices
        for split in SPLITS:
            matrix = validate_embedding(feature, split, matrices[split], len(split_rows[split]))
            save_matrix(root, feature, split, matrix)
    print("[done] pooled feature extraction completed")
    return root

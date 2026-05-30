#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection


EXPECTED_DOMAINS = {"gigaspeech", "imda", "librispeech", "svarah"}
OUTPUT_SPLIT_ORDER = ("train", "test")
SPLIT_PLANS = {
    "reference_candidate": {
        "manifest_splits": ("reference", "candidate"),
        "output_splits": {"reference": "train", "candidate": "test"},
        "expected_rows_per_domain_split": {"reference": 300, "candidate": 3000},
    },
    "legacy_train_test": {
        "manifest_splits": ("train", "test"),
        "output_splits": {"train": "train", "test": "test"},
        "expected_rows_per_domain_split": {"train": 300, "test": 700},
    },
}
REQUIRED_COLUMNS = {
    "utt_id",
    "orig_id",
    "domain",
    "split",
    "audio_path",
    "transcript",
    "transcript_norm",
    "duration_sec",
}

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "embeddings"
DEFAULT_SBERT_MODEL = "all-mpnet-base-v2"
SEMANTIC_SOURCES = {
    "gt": {"column": "transcript_norm", "raw_feature": "sbert_gt_raw", "feature": "sbert_gt"},
    "pseudo": {
        "column": "pseudo_transcript_norm",
        "raw_feature": "sbert_pseudo_raw",
        "feature": "sbert_pseudo",
    },
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_manifest = script_dir / "output" / "subset" / "metadata" / "selected_manifest.json"
    parser = argparse.ArgumentParser(
        description=(
            "Extract GT and/or pseudolabel SBERT embeddings, fit the same shared "
            "Gaussian random projection used by extract_embeddings.py, and save "
            "downstream-compatible train/test matrices."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest, help="Path to selected manifest JSON.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root containing manifests/, projections/, and SBERT feature dirs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL, help="SentenceTransformer model used for semantic embeddings.")
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="SBERT inference batch size.")
    parser.add_argument("--semantic-sources", nargs="+", choices=sorted(SEMANTIC_SOURCES), default=["gt", "pseudo"], help="Text sources to embed. Default: gt pseudo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list of rows: {manifest_path}")

    df = pd.DataFrame(rows)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    domains = set(df["domain"].unique())
    if domains != EXPECTED_DOMAINS:
        raise ValueError(f"Expected domains {sorted(EXPECTED_DOMAINS)}, found {sorted(domains)}")

    observed_splits = set(df["split"].unique())
    split_plan = next((p for p in SPLIT_PLANS.values() if observed_splits == set(p["manifest_splits"])), None)
    if split_plan is None:
        expected = [sorted(p["manifest_splits"]) for p in SPLIT_PLANS.values()]
        raise ValueError(f"Unsupported manifest splits {sorted(observed_splits)}; expected one of {expected}")

    expected_total = sum(
        int(split_plan["expected_rows_per_domain_split"][s]) * len(EXPECTED_DOMAINS)
        for s in split_plan["manifest_splits"]
    )
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, found {len(df)}")

    counts = df.groupby(["domain", "split"]).size()
    for domain in sorted(EXPECTED_DOMAINS):
        for split in split_plan["manifest_splits"]:
            expected = split_plan["expected_rows_per_domain_split"][split]
            actual = int(counts.get((domain, split), 0))
            if actual != expected:
                raise ValueError(f"Expected {expected} rows for {domain}/{split}, found {actual}")

    return df, split_plan


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_jsonl(records: List[dict], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[save] manifest: {len(records)} rows -> {output_path}")


def validate_embedding(
    name: str,
    split: str,
    emb: np.ndarray,
    expected_rows: int,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"{name}/{split} must be 2D, got shape {emb.shape}")
    if emb.shape[0] != expected_rows:
        raise ValueError(f"{name}/{split} row mismatch: expected {expected_rows}, got {emb.shape[0]}")
    if expected_dim is not None and emb.shape[1] != expected_dim:
        raise ValueError(f"{name}/{split} dim mismatch: expected {expected_dim}, got {emb.shape[1]}")
    if not np.isfinite(emb).all():
        raise ValueError(f"{name}/{split} contains NaN/Inf values")
    return emb


def save_embedding(out_root: Path, feature_set: str, split: str, emb: np.ndarray) -> Path:
    output_path = out_root / feature_set / f"{split}.npy"
    ensure_parent(output_path)
    np.save(output_path, emb.astype(np.float32, copy=False))
    print(f"[save] {feature_set}/{split}: {emb.shape} -> {output_path}")
    return output_path


class SBERTAdapter:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SBERT extraction. "
                "Install it in the speech_lab environment with: "
                "pip install sentence-transformers"
            ) from exc
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean = ["" if t is None else str(t) for t in texts]
        emb = self.model.encode(
            clean,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(emb, dtype=np.float32)


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_root = args.out_root.resolve()

    print(f"[setup] device: {device}")
    print(f"[setup] manifest: {args.manifest.resolve()}")
    print(f"[setup] out_root: {out_root}")
    print(f"[setup] sbert_model: {args.sbert_model}")
    print(f"[setup] semantic_sources: {','.join(args.semantic_sources)}")

    df, split_plan = load_manifest(args.manifest.resolve())
    print(f"[manifest] rows: {len(df)}")
    print("[manifest] split_mode: " + ",".join(f"{src}->{dst}" for src, dst in split_plan["output_splits"].items()))
    for (domain, split), count in df.groupby(["domain", "split"], sort=True).size().items():
        print(f"[manifest] subset {domain}/{split}: {int(count)} rows")

    per_domain = split_plan["expected_rows_per_domain_split"]
    split_frames: Dict[str, pd.DataFrame] = {}
    for manifest_split in split_plan["manifest_splits"]:
        output_split = split_plan["output_splits"][manifest_split]
        split_df = df[df["split"] == manifest_split].copy()
        expected_rows = int(per_domain[manifest_split]) * len(EXPECTED_DOMAINS)
        if len(split_df) != expected_rows:
            raise ValueError(f"Expected {expected_rows} {manifest_split} rows, found {len(split_df)}")
        split_frames[output_split] = split_df
        print(f"[manifest] split {manifest_split}->{output_split}: {len(split_df)} rows")

    save_jsonl(df.to_dict(orient="records"), out_root / "manifests" / "all.jsonl")
    for split in OUTPUT_SPLIT_ORDER:
        save_jsonl(split_frames[split].to_dict(orient="records"), out_root / "manifests" / f"{split}.jsonl")

    semantic_texts: Dict[str, Dict[str, List[str]]] = {}
    for source in args.semantic_sources:
        column = SEMANTIC_SOURCES[source]["column"]
        if column not in df.columns:
            raise ValueError(
                f"Semantic source '{source}' requires manifest column '{column}'. "
                "Use --semantic-sources gt for manifests without pseudolabels."
            )
        if source == "pseudo":
            status = df["pseudo_status"] if "pseudo_status" in df.columns else "ok"
            missing_text = df[(df[column].fillna("").astype(str).str.strip() == "") & (status != "failed")]
            if len(missing_text):
                preview = ", ".join(missing_text["utt_id"].astype(str).head(5).tolist())
                raise ValueError(f"Pseudo semantic source has {len(missing_text)} empty texts: {preview}")
        semantic_texts[source] = {
            split: split_df[column].fillna("").astype(str).tolist()
            for split, split_df in split_frames.items()
        }

    print("[extract] encoder=sbert")
    encoder = SBERTAdapter(model_name=args.sbert_model, device=device, batch_size=args.sbert_batch_size)

    sbert_raw: Dict[str, Dict[str, np.ndarray]] = {SEMANTIC_SOURCES[s]["raw_feature"]: {} for s in args.semantic_sources}
    for source in args.semantic_sources:
        raw_feature = SEMANTIC_SOURCES[source]["raw_feature"]
        for split in OUTPUT_SPLIT_ORDER:
            emb = encoder.encode(semantic_texts[source][split])
            emb = validate_embedding(raw_feature, split, emb, expected_rows=len(split_frames[split]))
            sbert_raw[raw_feature][split] = emb
            save_embedding(out_root, raw_feature, split, emb)
            if source == "gt":
                save_embedding(out_root, "sbert_raw", split, emb)

    stacked = np.vstack(
        [sbert_raw[SEMANTIC_SOURCES[s]["raw_feature"]][split] for s in args.semantic_sources for split in OUTPUT_SPLIT_ORDER]
    )
    in_dim = stacked.shape[1]
    grp = GaussianRandomProjection(n_components=256, random_state=args.seed)
    grp.fit(stacked)

    source_suffix = "_".join(args.semantic_sources)
    proj_out = out_root / "projections" / f"sbert_{source_suffix}_grp_{in_dim}_to_256.joblib"
    ensure_parent(proj_out)
    joblib.dump(grp, proj_out)
    print(f"[projection] sbert/{source_suffix}: fit on {stacked.shape}, saved {proj_out}")
    if "gt" in args.semantic_sources:
        legacy_out = out_root / "projections" / f"sbert_grp_{in_dim}_to_256.joblib"
        joblib.dump(grp, legacy_out)
        print(f"[projection] sbert legacy alias saved {legacy_out}")

    for source in args.semantic_sources:
        raw_feature = SEMANTIC_SOURCES[source]["raw_feature"]
        feature_set = SEMANTIC_SOURCES[source]["feature"]
        for split in OUTPUT_SPLIT_ORDER:
            projected = grp.transform(sbert_raw[raw_feature][split]).astype(np.float32)
            projected = validate_embedding(feature_set, split, projected, expected_rows=len(split_frames[split]), expected_dim=256)
            save_embedding(out_root, feature_set, split, projected)
            if source == "gt":
                save_embedding(out_root, "sbert", split, projected)

    print("[done] sbert extraction completed successfully.")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

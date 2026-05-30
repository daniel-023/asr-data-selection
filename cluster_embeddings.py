#!/usr/bin/env python3
"""Run KMeans clustering on test-set feature embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "embeddings"
DEFAULT_FEATURE_SETS = ("sbert", "wavlm_mfa", "sbert_mfa", "sbert_wavlm", "sbert_wavlm_mfa")
SPLIT = "test"
EXPECTED_TEST_ROWS = 2800
N_CLUSTERS = 4
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KMeans clustering on test-set feature embeddings.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Embeddings root containing manifests/, feature dirs, and results/.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(DEFAULT_FEATURE_SETS),
        help="Feature sets to cluster. Default: sbert wavlm_mfa sbert_mfa sbert_wavlm sbert_wavlm_mfa.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail instead of skipping missing feature files.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_manifest(out_root: Path) -> pd.DataFrame:
    df = load_jsonl(out_root / "manifests" / f"{SPLIT}.jsonl")
    if len(df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} {SPLIT} rows, found {len(df)}")
    if "domain" not in df.columns:
        raise ValueError("Manifest is missing required column: domain")
    return df


def feature_path(out_root: Path, feature_set: str) -> Path:
    candidates = [
        out_root / "embeddings" / feature_set / f"{SPLIT}.npy",
        out_root / feature_set / f"{SPLIT}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_feature_matrix(out_root: Path, feature_set: str, expected_rows: int) -> np.ndarray:
    path = feature_path(out_root, feature_set)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file for {feature_set}: {path}")
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"{feature_set}/{SPLIT} must be 2D, got shape {X.shape}")
    if X.shape[0] != expected_rows:
        raise ValueError(f"{feature_set}/{SPLIT} row mismatch: expected {expected_rows}, got {X.shape[0]}")
    if not np.isfinite(X).all():
        raise ValueError(f"{feature_set}/{SPLIT} contains NaN/Inf values")
    return X.astype(np.float32, copy=False)


def map_clusters_to_domains(domain_codes: np.ndarray, cluster_id: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    raw_confusion = confusion_matrix(domain_codes, cluster_id, labels=np.arange(N_CLUSTERS))
    domain_idx, cluster_idx = linear_sum_assignment(-raw_confusion)
    cluster_to_domain_idx = {int(cluster): int(domain) for domain, cluster in zip(domain_idx, cluster_idx)}
    if len(cluster_to_domain_idx) != N_CLUSTERS:
        raise ValueError(f"Expected {N_CLUSTERS} cluster-domain assignments, got {len(cluster_to_domain_idx)}")
    predicted_domain_idx = np.array([cluster_to_domain_idx[int(cluster)] for cluster in cluster_id], dtype=np.int64)
    return predicted_domain_idx, cluster_to_domain_idx


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_confusion_plot(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def skipped_row(feature_set: str, reason: str) -> dict:
    return {
        "feature_set": feature_set,
        "method": "kmeans",
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "rows": np.nan,
        "feature_dim": np.nan,
        "status": f"skipped: {reason}",
    }


def run_feature_set(out_root: Path, feature_set: str, meta: pd.DataFrame, domain_names: list[str]) -> dict:
    X = load_feature_matrix(out_root, feature_set, expected_rows=len(meta))
    print(f"[load] split={SPLIT} feature_set={feature_set} X={X.shape} rows={len(meta)}")

    y_true = pd.Categorical(meta["domain"], categories=domain_names).codes
    X_scaled = StandardScaler().fit_transform(X.astype(np.float64))
    cluster_id = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20).fit_predict(X_scaled)
    y_pred, cluster_to_domain_idx = map_clusters_to_domains(y_true, cluster_id)

    metrics = {
        "feature_set": feature_set,
        "method": "kmeans",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "rows": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "status": "ok",
        "split": SPLIT,
        "n_clusters": N_CLUSTERS,
        "cluster_to_domain_mapping": {str(k): domain_names[v] for k, v in sorted(cluster_to_domain_idx.items())},
    }
    print("[metrics]", json.dumps(metrics, indent=2))

    out_dir = out_root / "results" / "clustering" / feature_set
    out_dir.mkdir(parents=True, exist_ok=True)
    result = meta.copy()
    result["cluster_id"] = cluster_id
    result["predicted_domain"] = [domain_names[int(idx)] for idx in y_pred]

    conf = confusion_matrix(y_true, y_pred, labels=np.arange(len(domain_names)))
    save_jsonl(result, out_dir / "cluster_assignments.jsonl")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(conf, index=domain_names, columns=domain_names).to_csv(out_dir / "confusion_matrix.csv")
    save_confusion_plot(conf, domain_names, f"KMeans Confusion Matrix ({feature_set})", out_dir / "confusion_matrix.png")

    return metrics


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    meta = load_manifest(out_root)
    domain_names = [str(x) for x in pd.Categorical(meta["domain"]).categories]
    if len(domain_names) != N_CLUSTERS:
        raise ValueError(f"Expected {N_CLUSTERS} domains, found {len(domain_names)}: {domain_names}")

    summary_rows: list[dict] = []
    for feature_set in args.feature_sets:
        try:
            summary_rows.append(run_feature_set(out_root, feature_set, meta, domain_names))
        except FileNotFoundError as exc:
            if args.fail_on_missing:
                raise
            print(f"[skip] {feature_set}: {exc}")
            summary_rows.append(skipped_row(feature_set, str(exc)))

    summary_dir = out_root / "results" / "clustering"
    summary_dir.mkdir(parents=True, exist_ok=True)
    columns = ["feature_set", "method", "accuracy", "macro_f1", "weighted_f1", "rows", "feature_dim", "status"]
    pd.DataFrame(summary_rows).reindex(columns=columns).to_csv(summary_dir / "summary.csv", index=False)
    print(f"[done] wrote {summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

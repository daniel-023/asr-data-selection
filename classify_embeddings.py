#!/usr/bin/env python3
"""Train and evaluate domain classifiers on pooled and frame-level embedding feature sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "embeddings"
DEFAULT_POOLED_FEATURE_SETS = (
    "sbert_gt",
    "sbert_pseudo",
    "wavlm_256",
    "mfa_conformer_256",
    "wavlm_mfa",
    "sbert_gt_mfa",
    "sbert_pseudo_mfa",
    "sbert_gt_wavlm",
    "sbert_pseudo_wavlm",
    "sbert_gt_wavlm_mfa",
    "sbert_pseudo_wavlm_mfa",
)
DEFAULT_FRAME_FEATURE_SETS = ("wavlm_frame", "mfa_conformer_frame")
DEFAULT_FEATURE_SETS = DEFAULT_POOLED_FEATURE_SETS
FRAME_FEATURE_TO_ENCODER = {"wavlm_frame": "wavlm", "mfa_conformer_frame": "mfa_conformer"}
DEFAULT_MODELS = ("linear", "mlp", "cnn")
EXPECTED_ROWS = {"train": 1200, "test": 12000}
DEFAULT_COMPARISON_MODELS = ("mlp",)
RANDOM_STATE = 42
MLP_EPOCHS = 30
MLP_BATCH_SIZE = 64
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_DROPOUT = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear/MLP classifiers on pooled embeddings and temporal CNNs on frame embeddings."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Embeddings root containing manifests/, feature dirs, frame_embeddings/, and results/.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(DEFAULT_FEATURE_SETS),
        help=(
            "Feature sets to evaluate. Default: expanded GT/pseudo semantic, acoustic, "
            "and fused pooled feature sets. Use wavlm_frame and mfa_conformer_frame "
            "explicitly for temporal CNN runs."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(DEFAULT_MODELS),
        default=list(DEFAULT_COMPARISON_MODELS),
        help="Classifier models to run. Default: mlp.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=30, help="CNN training epochs.")
    parser.add_argument("--cnn-batch-size", type=int, default=64, help="CNN batch size.")
    parser.add_argument("--cnn-lr", type=float, default=1e-3, help="CNN Adam learning rate.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for CNN training.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail instead of skipping missing feature files.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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


def load_manifest(out_root: Path, split: str) -> pd.DataFrame:
    df = load_jsonl(out_root / "manifests" / f"{split}.jsonl")
    expected = EXPECTED_ROWS[split]
    if len(df) != expected:
        raise ValueError(f"Expected {expected} {split} rows, found {len(df)}")
    if "domain" not in df.columns:
        raise ValueError(f"{split} manifest is missing required column: domain")
    return df


def load_feature_matrix(out_root: Path, feature_set: str, split: str, expected_rows: int) -> np.ndarray:
    candidates = [
        out_root / "embeddings" / feature_set / f"{split}.npy",
        out_root / feature_set / f"{split}.npy",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file for {feature_set}/{split}: {path}")
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"{feature_set}/{split} must be 2D, got shape {X.shape}")
    if X.shape[0] != expected_rows:
        raise ValueError(f"{feature_set}/{split} row mismatch: expected {expected_rows}, got {X.shape[0]}")
    if not np.isfinite(X).all():
        raise ValueError(f"{feature_set}/{split} contains NaN/Inf values")
    return X.astype(np.float32, copy=False)


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


def frame_file_path(out_root: Path, feature_set: str, split: str, utt_id: str) -> Path:
    encoder_name = FRAME_FEATURE_TO_ENCODER[feature_set]
    candidates = [
        out_root / "frame_embeddings" / encoder_name / split / f"{utt_id}.npy",
        out_root / "embeddings" / "frame_embeddings" / encoder_name / split / f"{utt_id}.npy",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def load_frame_paths(out_root: Path, feature_set: str, split: str, meta: pd.DataFrame) -> list[Path]:
    if "utt_id" not in meta.columns:
        raise ValueError(f"{split} manifest is missing required column for frame embeddings: utt_id")
    paths = [frame_file_path(out_root, feature_set, split, str(utt_id)) for utt_id in meta["utt_id"].tolist()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise FileNotFoundError(f"Missing {len(missing)} frame files for {feature_set}/{split}: {preview}{suffix}")
    return paths


def load_frame_array(path: Path, feature_set: str) -> np.ndarray:
    frames = np.load(path)
    if frames.ndim != 2:
        raise ValueError(f"{feature_set} frame file must be 2D, got {frames.shape}: {path}")
    if frames.shape[0] < 1:
        raise ValueError(f"{feature_set} frame file has no frames: {path}")
    if not np.isfinite(frames).all():
        raise ValueError(f"{feature_set} frame file contains NaN/Inf values: {path}")
    return frames.astype(np.float32, copy=False)


def fit_frame_scaler(paths: list[Path], feature_set: str) -> tuple[StandardScaler, int]:
    scaler = StandardScaler()
    feature_dim: int | None = None
    for path in paths:
        frames = load_frame_array(path, feature_set)
        if feature_dim is None:
            feature_dim = int(frames.shape[1])
        elif frames.shape[1] != feature_dim:
            raise ValueError(f"{feature_set} frame dim mismatch in {path}: {frames.shape[1]} != {feature_dim}")
        scaler.partial_fit(frames.astype(np.float64))
    if feature_dim is None:
        raise ValueError(f"No frame files found for {feature_set}")
    return scaler, feature_dim


class FrameEmbeddingDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, scaler: StandardScaler, feature_set: str) -> None:
        self.paths = paths
        self.labels = labels
        self.scaler = scaler
        self.feature_set = feature_set

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = load_frame_array(self.paths[idx], self.feature_set)
        frames = self.scaler.transform(frames.astype(np.float64)).astype(np.float32)
        return torch.from_numpy(frames), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def collate_frame_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [frames.shape[0] for frames, _ in batch]
    max_len = max(lengths)
    feature_dim = batch[0][0].shape[1]
    padded = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.long)
    for idx, (frames, label) in enumerate(batch):
        length = frames.shape[0]
        if frames.shape[1] != feature_dim:
            raise ValueError(f"Frame dimension mismatch inside batch: {frames.shape[1]} != {feature_dim}")
        padded[idx, :length] = frames
        mask[idx, :length] = True
        labels[idx] = label
    return padded, mask, labels


class TemporalCNN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(frames.transpose(1, 2))
        mask_f = mask.unsqueeze(1).float()
        hidden = hidden * mask_f
        pooled = hidden.sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
        return self.classifier(pooled)


class PooledMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    device: torch.device,
) -> np.ndarray:
    model = PooledMLP(input_dim=X_train.shape[1], n_classes=n_classes).to(device)
    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64).copy()),
    )
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(MLP_EPOCHS):
        running_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            n += len(xb)
        print(f"  [mlp epoch {epoch + 1}/{MLP_EPOCHS}] loss={running_loss / n:.4f}")

    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32).copy()).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def train_frame_cnn(
    train_paths: list[Path],
    y_train: np.ndarray,
    test_paths: list[Path],
    feature_set: str,
    feature_dim: int,
    scaler: StandardScaler,
    n_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> np.ndarray:
    model = TemporalCNN(input_dim=feature_dim, n_classes=n_classes).to(device)
    train_ds = FrameEmbeddingDataset(train_paths, y_train, scaler, feature_set)
    test_ds = FrameEmbeddingDataset(test_paths, np.zeros(len(test_paths), dtype=np.int64), scaler, feature_set)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_frame_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_frame_batch)
    # CNN uses Adam without weight decay; regularization is handled by Dropout(0.3) in the classifier head.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for frames, mask, yb in train_loader:
            frames = frames.to(device)
            mask = mask.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(frames, mask), yb)
            loss.backward()
            optimizer.step()

    preds: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for frames, mask, _ in test_loader:
            logits = model(frames.to(device), mask.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    if model_name == "linear":
        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
        clf.fit(X_train, y_train)
        return clf.predict(X_test)
    if model_name == "mlp":
        return train_mlp(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            n_classes=n_classes,
            device=device,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def metrics_row(
    feature_set: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_rows: int,
    feature_dim: int,
) -> dict:
    return {
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "train_rows": int(train_rows),
        "test_rows": int(len(y_true)),
        "feature_dim": int(feature_dim),
        "status": "ok",
        "reason": "",
    }


def skipped_row(feature_set: str, model_name: str, reason: str) -> dict:
    return {
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "train_rows": np.nan,
        "test_rows": np.nan,
        "feature_dim": np.nan,
        "status": "skipped",
        "reason": reason,
    }


def save_model_outputs(
    out_root: Path,
    feature_set: str,
    model_name: str,
    test_meta: pd.DataFrame,
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row: dict,
) -> None:
    out_dir = out_root / "results" / "classification" / feature_set / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = test_meta.copy()
    result["predicted_domain"] = [labels[int(idx)] for idx in y_pred]
    save_jsonl(result, out_dir / "predictions.jsonl")

    conf = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    pd.DataFrame(conf, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")
    save_confusion_plot(conf, labels, f"Confusion Matrix ({feature_set}/{model_name})", out_dir / "confusion_matrix.png")


def run_pooled_feature_set(
    out_root: Path,
    feature_set: str,
    models: list[str],
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    labels: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    models = [model_name for model_name in models if model_name != "cnn"]
    if not models:
        print(f"[skip] feature_set={feature_set}: no pooled-vector models requested")
        return []

    X_train = load_feature_matrix(out_root, feature_set, "train", expected_rows=len(train_meta))
    X_test = load_feature_matrix(out_root, feature_set, "test", expected_rows=len(test_meta))
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"{feature_set} train/test dim mismatch: {X_train.shape[1]} != {X_test.shape[1]}")

    y_train = pd.Categorical(train_meta["domain"], categories=labels).codes
    y_test = pd.Categorical(test_meta["domain"], categories=labels).codes
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float64)).astype(np.float32)
    X_test = scaler.transform(X_test.astype(np.float64)).astype(np.float32)
    print(f"[load] feature_set={feature_set} train={X_train.shape} test={X_test.shape}")

    rows: list[dict] = []
    for model_name in models:
        print(f"[train] feature_set={feature_set} model={model_name}")
        y_pred = predict_model(model_name, X_train, y_train, X_test, len(labels), device, args)
        row = metrics_row(feature_set, model_name, y_test, y_pred, len(train_meta), X_train.shape[1])
        print("[metrics]", json.dumps(row, indent=2))
        save_model_outputs(out_root, feature_set, model_name, test_meta, labels, y_test, y_pred, row)
        rows.append(row)
    return rows


def run_frame_feature_set(
    out_root: Path,
    feature_set: str,
    models: list[str],
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    labels: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    if "cnn" not in models:
        print(f"[skip] feature_set={feature_set}: no frame-level CNN requested")
        return []

    train_paths = load_frame_paths(out_root, feature_set, "train", train_meta)
    test_paths = load_frame_paths(out_root, feature_set, "test", test_meta)
    y_train = pd.Categorical(train_meta["domain"], categories=labels).codes
    y_test = pd.Categorical(test_meta["domain"], categories=labels).codes
    scaler, feature_dim = fit_frame_scaler(train_paths, feature_set)
    print(
        f"[load] feature_set={feature_set} train_files={len(train_paths)} "
        f"test_files={len(test_paths)} frame_dim={feature_dim}"
    )

    print(f"[train] feature_set={feature_set} model=cnn")
    y_pred = train_frame_cnn(
        train_paths=train_paths,
        y_train=y_train,
        test_paths=test_paths,
        feature_set=feature_set,
        feature_dim=feature_dim,
        scaler=scaler,
        n_classes=len(labels),
        device=device,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        lr=args.cnn_lr,
    )
    row = metrics_row(feature_set, "cnn", y_test, y_pred, len(train_meta), feature_dim)
    print("[metrics]", json.dumps(row, indent=2))
    save_model_outputs(out_root, feature_set, "cnn", test_meta, labels, y_test, y_pred, row)
    return [row]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_root = args.out_root.resolve()
    device = resolve_device(args.device)
    train_meta = load_manifest(out_root, "train")
    test_meta = load_manifest(out_root, "test")
    labels = [str(x) for x in pd.Categorical(train_meta["domain"]).categories]

    summary_rows: list[dict] = []
    for feature_set in args.feature_sets:
        try:
            if feature_set in FRAME_FEATURE_TO_ENCODER:
                summary_rows.extend(
                    run_frame_feature_set(out_root, feature_set, args.models, train_meta, test_meta, labels, device, args)
                )
            else:
                summary_rows.extend(
                    run_pooled_feature_set(out_root, feature_set, args.models, train_meta, test_meta, labels, device, args)
                )
        except FileNotFoundError as exc:
            if args.fail_on_missing:
                raise
            print(f"[skip] {feature_set}: {exc}")
            skipped_models = ["cnn"] if feature_set in FRAME_FEATURE_TO_ENCODER else [
                model_name for model_name in args.models if model_name != "cnn"
            ]
            summary_rows.extend(skipped_row(feature_set, model_name, str(exc)) for model_name in skipped_models)

    summary_dir = out_root / "results" / "classification"
    summary_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature_set",
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "train_rows",
        "test_rows",
        "feature_dim",
        "status",
        "reason",
    ]
    pd.DataFrame(summary_rows).reindex(columns=columns).to_csv(summary_dir / "summary.csv", index=False)
    print(f"[done] wrote {summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

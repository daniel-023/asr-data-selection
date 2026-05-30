from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .encoders import resolve_device
from .manifests import embeddings_root, read_jsonl, results_root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PooledMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def load_matrix(root: Path, feature_set: str, split: str, expected_rows: int) -> np.ndarray:
    path = root / feature_set / f"{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(f"Unexpected shape for {feature_set}/{split}: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{feature_set}/{split} contains NaN or Inf values")
    return matrix.astype(np.float32)


def train_mlp(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, n_classes: int, settings: dict, device: torch.device) -> np.ndarray:
    model = PooledMLP(train_x.shape[1], list(settings["hidden_dims"]), float(settings["dropout"]), n_classes).to(device)
    dataset = TensorDataset(torch.from_numpy(train_x.copy()), torch.from_numpy(train_y.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(int(settings["epochs"])):
        total_loss, count = 0.0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
            count += len(inputs)
        print(f"  [epoch {epoch + 1}/{settings['epochs']}] loss={total_loss / count:.4f}")
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(test_x.copy()).to(device)).argmax(dim=1).cpu().numpy()


def metrics_row(feature_set: str, true: np.ndarray, predicted: np.ndarray, train_rows: int, feature_dim: int) -> dict:
    return {
        "feature_set": feature_set,
        "model": "mlp",
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro")),
        "weighted_f1": float(f1_score(true, predicted, average="weighted")),
        "train_rows": int(train_rows),
        "test_rows": int(len(true)),
        "feature_dim": int(feature_dim),
        "status": "ok",
        "reason": "",
    }


def save_confusion_plot(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_outputs(root: Path, feature_set: str, metadata: pd.DataFrame, labels: list[str], true: np.ndarray, predicted: np.ndarray, row: dict) -> None:
    output = root / feature_set / "mlp"
    output.mkdir(parents=True, exist_ok=True)
    predictions = metadata.copy()
    predictions["predicted_domain"] = [labels[int(index)] for index in predicted]
    predictions.to_json(output / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    matrix = confusion_matrix(true, predicted, labels=np.arange(len(labels)))
    report = classification_report(true, predicted, target_names=labels, output_dict=True, zero_division=0)
    (output / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    pd.DataFrame(report).transpose().to_csv(output / "classification_report.csv")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
    save_confusion_plot(matrix, labels, output / "confusion_matrix.png")


def run(config: dict) -> Path:
    settings = config["classification"]
    set_seed(int(config["experiment"]["seed"]))
    device = resolve_device(settings["device"])
    embedding_dir = embeddings_root(config)
    output_dir = results_root(config)
    train_meta = read_jsonl(embedding_dir / "manifests" / "train.jsonl")
    test_meta = read_jsonl(embedding_dir / "manifests" / "test.jsonl")
    labels = sorted(train_meta["domain"].astype(str).unique().tolist())
    train_y = pd.Categorical(train_meta["domain"], categories=labels).codes
    test_y = pd.Categorical(test_meta["domain"], categories=labels).codes
    rows = []
    for feature_set in settings["feature_sets"]:
        train_x = load_matrix(embedding_dir, feature_set, "train", len(train_meta))
        test_x = load_matrix(embedding_dir, feature_set, "test", len(test_meta))
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x.astype(np.float64)).astype(np.float32)
        test_x = scaler.transform(test_x.astype(np.float64)).astype(np.float32)
        print(f"[train] {feature_set}: train={train_x.shape} test={test_x.shape}")
        predicted = train_mlp(train_x, train_y, test_x, len(labels), settings, device)
        row = metrics_row(feature_set, test_y, predicted, len(train_meta), train_x.shape[1])
        rows.append(row)
        save_outputs(output_dir, feature_set, test_meta, labels, test_y, predicted, row)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f"[done] wrote {summary}")
    return summary

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .manifests import expected_split_rows, results_root


DISPLAY_NAMES = {
    "sbert_gt": "Semantic GT",
    "sbert_pseudo": "Semantic Pseudo",
    "wavlm_256": "WavLM",
    "mfa_conformer_256": "MFA",
    "wavlm_mfa": "WavLM+MFA",
    "sbert_gt_mfa": "GT+MFA",
    "sbert_pseudo_mfa": "Pseudo+MFA",
    "sbert_gt_wavlm": "GT+WavLM",
    "sbert_pseudo_wavlm": "Pseudo+WavLM",
    "sbert_gt_wavlm_mfa": "GT Full Fusion",
    "sbert_pseudo_wavlm_mfa": "Pseudo Full Fusion",
}


def feature_groups(config: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    return [(group["family"], [tuple(feature) for feature in group["features"]]) for group in config["plot"]["groups"]]


def load_summary(path: Path, config: dict) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"feature_set", "model", "accuracy", "macro_f1", "weighted_f1", "feature_dim", "train_rows", "test_rows", "status"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Summary missing required columns: {sorted(missing)}")
    order = [feature for _, features in feature_groups(config) for feature, _ in features]
    family = {feature: current_family for current_family, features in feature_groups(config) for feature, _ in features}
    short = {feature: label for _, features in feature_groups(config) for feature, label in features}
    frame = frame[(frame["model"] == "mlp") & (frame["status"] == "ok") & frame["feature_set"].isin(order)].copy()
    frame = frame[
        (frame["train_rows"].astype(int) == expected_split_rows(config, "reference"))
        & (frame["test_rows"].astype(int) == expected_split_rows(config, "candidate"))
    ].copy()
    if set(frame["feature_set"]) != set(order):
        raise ValueError("Summary does not contain one successful current MLP row for every configured feature set.")
    frame["feature_set"] = pd.Categorical(frame["feature_set"], categories=order, ordered=True)
    frame = frame.sort_values("feature_set")
    frame["family"] = frame["feature_set"].astype(str).map(family)
    frame["display_name"] = frame["feature_set"].astype(str).map(DISPLAY_NAMES)
    frame["axis_label"] = frame["feature_set"].astype(str).map(short)
    return frame


def positions(groups: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> tuple[list[float], list[tuple[str, float]], list[float]]:
    x, bars, centers, boundaries = 0.0, [], [], []
    for family, features in groups:
        family_positions = []
        for _feature, _label in features:
            bars.append(x)
            family_positions.append(x)
            x += 1.0
        centers.append((family, float(np.mean(family_positions))))
        boundaries.append(x - 0.5)
        x += 0.55
    return bars, centers, boundaries


def color(feature: str) -> str:
    if feature == "mfa_conformer_256":
        return "#8C8C8C"
    if feature in {"wavlm_256", "wavlm_mfa"}:
        return "#607D8B"
    if "pseudo" in feature:
        return "#F58518"
    return "#4C78A8"


def plot_accuracy(frame: pd.DataFrame, output: Path, config: dict) -> None:
    groups = feature_groups(config)
    bars, centers, boundaries = positions(groups)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(bars, frame["accuracy"].astype(float), width=0.72, color=[color(str(feature)) for feature in frame["feature_set"]])
    y_min, y_max = float(config["plot"]["y_min"]), float(config["plot"]["y_max"])
    axis.set_ylim(y_min, y_max)
    axis.set_ylabel("Accuracy")
    axis.set_title(config["plot"]["title"])
    axis.set_xticks(bars, labels=frame["axis_label"].tolist())
    for index, (family, center) in enumerate(centers):
        axis.text(center, -0.16, family, ha="center", va="top", transform=axis.get_xaxis_transform())
        if index < len(boundaries) - 1:
            axis.axvline(boundaries[index], color="#D0D0D0", linewidth=1.0)
    offset = (y_max - y_min) * 0.02
    for position, value in zip(bars, frame["accuracy"].astype(float)):
        axis.text(position, min(value + offset, y_max - offset), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.subplots_adjust(bottom=0.23)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(config: dict) -> Path:
    root = results_root(config)
    frame = load_summary(root / "summary.csv", config)
    output = root / "mlp_bar_accuracy_by_family_13200.png"
    plot_accuracy(frame, output, config)
    frame[["family", "feature_set", "display_name", "accuracy", "macro_f1", "weighted_f1", "feature_dim", "train_rows", "test_rows"]].to_csv(
        output.with_suffix(".csv"), index=False
    )
    print(f"[done] wrote {output}")
    return output

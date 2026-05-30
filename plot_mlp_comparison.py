#!/usr/bin/env python3
"""Plot MLP classification accuracy for embedding feature sets grouped by family."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "embeddings"
FEATURE_GROUPS = [
    ("Semantic", [("sbert_gt", "GT"), ("sbert_pseudo", "Pseudo")]),
    (
        "Acoustic",
        [("wavlm_256", "WavLM"), ("mfa_conformer_256", "MFA"), ("wavlm_mfa", "WavLM+MFA")],
    ),
    ("+MFA", [("sbert_gt_mfa", "GT"), ("sbert_pseudo_mfa", "Pseudo")]),
    ("+WavLM", [("sbert_gt_wavlm", "GT"), ("sbert_pseudo_wavlm", "Pseudo")]),
    (
        "Full Fusion",
        [("sbert_gt_wavlm_mfa", "GT"), ("sbert_pseudo_wavlm_mfa", "Pseudo")],
    ),
]
FEATURE_GROUPS_4000 = [
    ("Semantic", [("sbert_4000", "SBERT")]),
    (
        "Acoustic",
        [("wavlm_4000", "WavLM"), ("mfa_4000", "MFA"), ("wavlm_mfa_4000", "WavLM+MFA")],
    ),
    ("+MFA", [("sbert_mfa_4000", "SBERT")]),
    ("+WavLM", [("sbert_wavlm_4000", "SBERT")]),
    ("Full Fusion", [("sbert_wavlm_mfa_4000", "Full")]),
]
RESULTS_4000 = [
    ("sbert_4000", "Pure Semantic (SBERT)", 256, 0.7396, 0.7347),
    ("wavlm_4000", "WavLM", 256, 0.9111, 0.9106),
    ("mfa_4000", "MFA", 256, 0.5821, 0.5243),
    ("wavlm_mfa_4000", "Pure Acoustic (WavLM + MFA)", 512, 0.8061, 0.8031),
    ("sbert_mfa_4000", "Semantic + MFA", 512, 0.7250, 0.7170),
    ("sbert_wavlm_4000", "Semantic + WavLM", 512, 0.8996, 0.8991),
    ("sbert_wavlm_mfa_4000", "Full Fusion (SBERT + WavLM + MFA)", 768, 0.8339, 0.8295),
]
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
FEATURE_ORDER = [feature_set for _, features in FEATURE_GROUPS for feature_set, _ in features]
FEATURE_TO_FAMILY = {
    feature_set: family
    for family, features in FEATURE_GROUPS
    for feature_set, _ in features
}
FEATURE_TO_SHORT_LABEL = {
    feature_set: short_label
    for _, features in FEATURE_GROUPS
    for feature_set, short_label in features
}
FEATURE_TO_AXIS_LABEL = {
    "sbert_gt": "GT",
    "sbert_pseudo": "Pseudo",
    "wavlm_256": "WavLM",
    "mfa_conformer_256": "MFA",
    "wavlm_mfa": "WavLM\n+ MFA",
    "sbert_gt_mfa": "GT",
    "sbert_pseudo_mfa": "Pseudo",
    "sbert_gt_wavlm": "GT",
    "sbert_pseudo_wavlm": "Pseudo",
    "sbert_gt_wavlm_mfa": "GT",
    "sbert_pseudo_wavlm_mfa": "Pseudo",
}
FEATURE_TO_AXIS_LABEL_4000 = {
    "sbert_4000": "SBERT",
    "wavlm_4000": "WavLM",
    "mfa_4000": "MFA",
    "wavlm_mfa_4000": "WavLM\n+ MFA",
    "sbert_mfa_4000": "SBERT",
    "sbert_wavlm_4000": "SBERT",
    "sbert_wavlm_mfa_4000": "Full",
}
CURRENT_TRAIN_ROWS = 1200
CURRENT_TEST_ROWS = 12000
WAVLM_BASELINE = "wavlm_256"
CURRENT_FEATURE_SETS = set(FEATURE_ORDER)
PRESENTATION_Y_MIN = 0.50
PRESENTATION_Y_MAX = 0.94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MLP comparison bar chart.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Embeddings output root.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Classification summary CSV. Default: OUT/results/classification/summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="13,200-sample output PNG. Default: OUT/results/classification/mlp_bar_accuracy_by_family_13200.png",
    )
    parser.add_argument(
        "--plot-4000",
        action="store_true",
        help="Also write the manually supplied 4,000-sample comparison chart and CSV.",
    )
    parser.add_argument(
        "--output-4000",
        type=Path,
        default=None,
        help=(
            "4,000-sample output PNG. Default: same directory as --output, "
            "named mlp_bar_accuracy_by_family_4000.png."
        ),
    )
    return parser.parse_args()


def load_current_mlp_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "feature_set",
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "feature_dim",
        "train_rows",
        "test_rows",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary missing required columns: {sorted(missing)}")
    df = df[df["model"] == "mlp"].copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"].copy()
    df = df[df["feature_set"].isin(FEATURE_ORDER)].copy()
    df = df[
        (df["train_rows"].astype(int) == CURRENT_TRAIN_ROWS)
        & (df["test_rows"].astype(int) == CURRENT_TEST_ROWS)
    ].copy()
    if df.empty:
        raise ValueError(
            "No successful current MLP rows found for the default comparison feature sets "
            f"with train_rows={CURRENT_TRAIN_ROWS} and test_rows={CURRENT_TEST_ROWS}."
        )

    order = [feature for feature in FEATURE_ORDER if feature in set(df["feature_set"])]
    df["feature_set"] = pd.Categorical(df["feature_set"], categories=order, ordered=True)
    df = df.sort_values("feature_set")
    df["family"] = df["feature_set"].astype(str).map(FEATURE_TO_FAMILY)
    df["display_name"] = df["feature_set"].astype(str).map(DISPLAY_NAMES).fillna(df["feature_set"].astype(str))
    df["short_label"] = df["feature_set"].astype(str).map(FEATURE_TO_SHORT_LABEL).fillna(df["display_name"])
    df["axis_label"] = df["feature_set"].astype(str).map(FEATURE_TO_AXIS_LABEL).fillna(df["short_label"])
    return df


def family_positions(
    df: pd.DataFrame,
    feature_groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
) -> tuple[list[float], list[tuple[str, float]], list[float]]:
    positions: list[float] = []
    family_centers: list[tuple[str, float]] = []
    boundaries: list[float] = []
    x = 0.0
    present = set(df["feature_set"].astype(str))
    for family, features in feature_groups:
        current_positions: list[float] = []
        for feature_set, _ in features:
            if feature_set in present:
                positions.append(x)
                current_positions.append(x)
                x += 1.0
        if current_positions:
            family_centers.append((family, float(np.mean(current_positions))))
            boundaries.append(x - 0.5)
            x += 0.55
    return positions, family_centers, boundaries


def color_for_feature(feature_set: str) -> str:
    if feature_set in {"mfa_conformer_256", "mfa_4000"}:
        return "#8C8C8C"
    if feature_set in {"wavlm_256", "wavlm_mfa", "wavlm_4000", "wavlm_mfa_4000"}:
        return "#607D8B"
    if feature_set in {
        "sbert_gt_mfa",
        "sbert_gt_wavlm",
        "sbert_gt_wavlm_mfa",
        "sbert_mfa_4000",
        "sbert_wavlm_4000",
        "sbert_wavlm_mfa_4000",
    }:
        return "#4C78A8"
    if "pseudo" in feature_set:
        return "#F58518"
    return "#4C78A8"


def add_family_labels(
    ax: plt.Axes,
    family_centers: Sequence[tuple[str, float]],
    boundaries: Sequence[float],
) -> None:
    for idx, (family, center) in enumerate(family_centers):
        ax.text(center, -0.16, family, ha="center", va="top", transform=ax.get_xaxis_transform())
        if idx < len(boundaries) - 1:
            ax.axvline(boundaries[idx], color="#D0D0D0", linewidth=1.0)


def plot_accuracy(
    df: pd.DataFrame,
    output_path: Path,
    y_min: float,
    y_max: float,
    title: str,
    feature_groups: Sequence[tuple[str, Sequence[tuple[str, str]]]] = FEATURE_GROUPS,
) -> None:
    positions, family_centers, boundaries = family_positions(df, feature_groups)
    fig_width = 12.0
    fig_height = 6.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = [color_for_feature(str(feature)) for feature in df["feature_set"]]

    ax.bar(positions, df["accuracy"].astype(float), width=0.72, color=colors)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(df["axis_label"].tolist(), rotation=0, ha="center", linespacing=1.1)
    add_family_labels(ax, family_centers, boundaries)

    label_offset = (y_max - y_min) * 0.02
    for pos, value in zip(positions, df["accuracy"].astype(float)):
        label_y = min(value + label_offset, y_max - label_offset)
        ax.text(pos, label_y, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.23)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_13200_presentation(df: pd.DataFrame, output_path: Path) -> None:
    plot_accuracy(
        df,
        output_path,
        y_min=PRESENTATION_Y_MIN,
        y_max=PRESENTATION_Y_MAX,
        title="MLP Accuracy by Embedding Family (13,200-Sample Dataset)",
    )


def write_plot_csv(df: pd.DataFrame, output_path: Path) -> Path:
    plot_csv = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[
        [
            "family",
            "feature_set",
            "display_name",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "feature_dim",
            "train_rows",
            "test_rows",
        ]
    ].to_csv(plot_csv, index=False)
    return plot_csv


def build_4000_results() -> pd.DataFrame:
    feature_to_family = {
        feature_set: family
        for family, features in FEATURE_GROUPS_4000
        for feature_set, _ in features
    }
    feature_order = [feature_set for _, features in FEATURE_GROUPS_4000 for feature_set, _ in features]
    rows = []
    for feature_set, display_name, feature_dim, accuracy, macro_f1 in RESULTS_4000:
        rows.append(
            {
                "sample_set": "4000",
                "family": feature_to_family[feature_set],
                "feature_set": feature_set,
                "display_name": display_name,
                "axis_label": FEATURE_TO_AXIS_LABEL_4000[feature_set],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": macro_f1,
                "feature_dim": feature_dim,
                "sample_rows": 4000,
            }
        )
    df = pd.DataFrame(rows)
    df["feature_set"] = pd.Categorical(df["feature_set"], categories=feature_order, ordered=True)
    return df.sort_values("feature_set")


def write_4000_csv(df: pd.DataFrame, output_path: Path) -> Path:
    csv_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[
        [
            "sample_set",
            "family",
            "feature_set",
            "display_name",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "feature_dim",
            "sample_rows",
        ]
    ].to_csv(csv_path, index=False)
    return csv_path


def write_takeaways(df: pd.DataFrame, output_dir: Path) -> Path:
    values = {str(row.feature_set): float(row.accuracy) for row in df.itertuples(index=False)}
    rows = []
    if "sbert_gt_wavlm_mfa" in values and WAVLM_BASELINE in values:
        rows.append(
            {
                "finding": "Full fusion vs WavLM",
                "value": values["sbert_gt_wavlm_mfa"] - values[WAVLM_BASELINE],
                "note": "Full fusion adds little over WavLM alone.",
            }
        )
    if "sbert_gt" in values and "sbert_pseudo" in values:
        rows.append(
            {
                "finding": "Semantic GT vs pseudo",
                "value": values["sbert_gt"] - values["sbert_pseudo"],
                "note": "GT text helps most when semantics are used alone.",
            }
        )
    if "sbert_gt_wavlm" in values and "sbert_pseudo_wavlm" in values:
        rows.append(
            {
                "finding": "GT vs pseudo with WavLM",
                "value": values["sbert_gt_wavlm"] - values["sbert_pseudo_wavlm"],
                "note": "GT/pseudo gap nearly disappears once WavLM is included.",
            }
        )
    if WAVLM_BASELINE in values and "wavlm_mfa" in values:
        rows.append(
            {
                "finding": "WavLM vs WavLM+MFA",
                "value": values[WAVLM_BASELINE] - values["wavlm_mfa"],
                "note": "MFA does not improve WavLM in this run.",
            }
        )
    if WAVLM_BASELINE in values and "mfa_conformer_256" in values:
        rows.append(
            {
                "finding": "WavLM vs MFA",
                "value": values[WAVLM_BASELINE] - values["mfa_conformer_256"],
                "note": "WavLM is much stronger than MFA alone.",
            }
        )
    output_path = output_dir / "mlp_accuracy_key_findings.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path

def write_current_results_manifest(classification_dir: Path) -> Path:
    current_paths = [
        "summary.csv",
        "mlp_bar_accuracy_by_family_13200.csv",
        "mlp_bar_accuracy_by_family_13200.png",
        "mlp_accuracy_key_findings.csv",
    ]
    for feature_set in FEATURE_ORDER:
        feature_dir = classification_dir / feature_set / "mlp"
        if feature_dir.exists():
            for path in sorted(feature_dir.glob("*")):
                if path.is_file():
                    current_paths.append(str(path.relative_to(classification_dir)))

    legacy_dirs = [
        path.name
        for path in sorted(classification_dir.iterdir())
        if path.is_dir() and path.name not in CURRENT_FEATURE_SETS
    ]
    output_path = classification_dir / "current_mlp_results_manifest.txt"
    lines = [
        "Current 13,200-row MLP experiment files to share",
        f"Expected rows: train={CURRENT_TRAIN_ROWS}, test={CURRENT_TEST_ROWS}",
        "",
        "Include:",
        *[f"  - {path}" for path in current_paths],
        "",
        "Legacy/non-headline directories present locally, exclude unless explicitly discussed:",
        *[f"  - {name}" for name in legacy_dirs],
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    summary_path = args.summary or out_root / "results" / "classification" / "summary.csv"
    output_path = args.output or out_root / "results" / "classification" / "mlp_bar_accuracy_by_family_13200.png"

    df = load_current_mlp_summary(summary_path)
    plot_csv = write_plot_csv(df, output_path)
    plot_13200_presentation(df, output_path)
    takeaways_path = write_takeaways(df, output_path.parent)
    manifest_path = write_current_results_manifest(output_path.parent)

    print(f"[done] wrote {output_path}")
    print(f"[done] wrote {plot_csv}")
    print(f"[done] wrote {takeaways_path}")
    print(f"[done] wrote {manifest_path}")

    if args.plot_4000:
        output_4000 = args.output_4000 or output_path.with_name("mlp_bar_accuracy_by_family_4000.png")
        df_4000 = build_4000_results()
        csv_4000 = write_4000_csv(df_4000, output_4000)
        plot_accuracy(
            df_4000,
            output_4000,
            y_min=PRESENTATION_Y_MIN,
            y_max=PRESENTATION_Y_MAX,
            title="MLP Accuracy by Embedding Family (4,000-Sample Dataset)",
            feature_groups=FEATURE_GROUPS_4000,
        )
        print(f"[done] wrote {output_4000}")
        print(f"[done] wrote {csv_4000}")


if __name__ == "__main__":
    main()

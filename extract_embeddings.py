#!/usr/bin/env python3
"""Extract pooled and optional frame-level embeddings for the LLM domain subset."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import random
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.random_projection import GaussianRandomProjection
from transformers import AutoFeatureExtractor, AutoModel


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
DEFAULT_AUDIO_SELECTION_DIR = "data_selection_13200"
DEFAULT_WAVLM_MODEL = "microsoft/wavlm-base-plus"
DEFAULT_SBERT_MODEL = "all-mpnet-base-v2"
DEFAULT_NUM_WORKERS = 4
FUSION_FEATURE_SETS = {
    "wavlm_mfa": ["wavlm_256", "mfa_conformer_256"],
    "sbert_gt_mfa": ["sbert_gt", "mfa_conformer_256"],
    "sbert_pseudo_mfa": ["sbert_pseudo", "mfa_conformer_256"],
    "sbert_gt_wavlm": ["sbert_gt", "wavlm_256"],
    "sbert_pseudo_wavlm": ["sbert_pseudo", "wavlm_256"],
    "sbert_gt_wavlm_mfa": ["sbert_gt", "wavlm_256", "mfa_conformer_256"],
    "sbert_pseudo_wavlm_mfa": ["sbert_pseudo", "wavlm_256", "mfa_conformer_256"],
    "sbert_mfa": ["sbert_gt", "mfa_conformer_256"],
    "sbert_wavlm": ["sbert_gt", "wavlm_256"],
    "sbert_wavlm_mfa": ["sbert_gt", "wavlm_256", "mfa_conformer_256"],
}
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
    default_mfa_checkpoint = script_dir / "checkpoints" / "MFA_conformer.ckpt"

    parser = argparse.ArgumentParser(
        description=(
            "Extract WavLM, MFA-Conformer, and SBERT embeddings, apply "
            "Gaussian random projection to selected pooled encoders, and "
            "save fused pooled feature sets when all required encoders run. "
            "Optionally save acoustic frame-level embeddings for temporal CNNs."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Path to selected manifest JSON.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root containing manifests/, projections/, feature dirs, and optional frame_embeddings/.",
    )
    parser.add_argument(
        "--mfa-checkpoint",
        type=Path,
        default=default_mfa_checkpoint,
        help="Path to local MFA-Conformer .ckpt file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["wavlm", "mfa_conformer", "sbert", "wavlm_frame", "mfa_conformer_frame"],
        default=["wavlm", "mfa_conformer", "sbert"],
        help=(
            "Encoders/features to extract. Use wavlm_frame and mfa_conformer_frame "
            "for frame-level acoustic outputs. Default: wavlm mfa_conformer sbert."
        ),
    )
    parser.add_argument(
        "--sbert-model",
        default=DEFAULT_SBERT_MODEL,
        help="SentenceTransformer model used for transcript_norm embeddings.",
    )
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="SBERT inference batch size.")
    parser.add_argument(
        "--semantic-sources",
        nargs="+",
        choices=sorted(SEMANTIC_SOURCES),
        default=["gt", "pseudo"],
        help="Text sources to embed when sbert is enabled. Default: gt pseudo.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def resolve_mfa_checkpoint_path(checkpoint_path: Path, script_dir: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    candidate_lower = script_dir / "checkpoints" / checkpoint_path.name
    if candidate_lower.exists():
        return candidate_lower.resolve()

    candidate_upper = script_dir / "Checkpoints" / checkpoint_path.name
    if candidate_upper.exists():
        return candidate_upper.resolve()

    raise FileNotFoundError(f"MFA checkpoint not found: {checkpoint_path}")


def normalize_audio_path(raw_path: str, subset_root: Path) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct
    if not direct.is_absolute():
        rel = (subset_root / direct).resolve()
        if rel.exists():
            return rel
        if direct.parts and direct.parts[0] == "audio":
            nested = (
                subset_root
                / "audio"
                / DEFAULT_AUDIO_SELECTION_DIR
                / Path(*direct.parts[1:])
            ).resolve()
            if nested.exists():
                return nested

    for marker in ("/output/subset/audio/", "/local_output/audio/", "/audio/"):
        if marker in raw_path:
            rel_tail = raw_path.split(marker, 1)[1]
            normalized = (subset_root / "audio" / rel_tail).resolve()
            if normalized.exists():
                return normalized
            nested = (subset_root / "audio" / DEFAULT_AUDIO_SELECTION_DIR / rel_tail).resolve()
            if nested.exists():
                return nested
    return direct


def expected_rows_per_split(plan: dict) -> dict[str, int]:
    per_domain = plan["expected_rows_per_domain_split"]
    return {split: int(per_domain[split]) * len(EXPECTED_DOMAINS) for split in plan["manifest_splits"]}


def expected_total_rows(plan: dict) -> int:
    return sum(expected_rows_per_split(plan).values())


def infer_split_plan(df: pd.DataFrame) -> dict:
    observed_splits = set(df["split"].unique())
    for plan in SPLIT_PLANS.values():
        if observed_splits == set(plan["manifest_splits"]):
            return plan
    expected = [sorted(plan["manifest_splits"]) for plan in SPLIT_PLANS.values()]
    raise ValueError(f"Unsupported manifest splits {sorted(observed_splits)}; expected one of {expected}")


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    subset_root = manifest_path.resolve().parent.parent
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

    split_plan = infer_split_plan(df)
    expected_total = expected_total_rows(split_plan)
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, found {len(df)}")

    counts = df.groupby(["domain", "split"]).size()
    for domain in sorted(EXPECTED_DOMAINS):
        for split in split_plan["manifest_splits"]:
            expected = split_plan["expected_rows_per_domain_split"][split]
            actual = int(counts.get((domain, split), 0))
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} rows for {domain}/{split}, found {actual}"
                )

    df["audio_path_raw"] = df["audio_path"].astype(str)
    df["audio_path"] = df["audio_path_raw"].map(lambda p: normalize_audio_path(p, subset_root))

    missing_paths = [p for p in df["audio_path"].tolist() if not p.exists()]
    if missing_paths:
        preview = "\n".join(f"  - {p}" for p in missing_paths[:10])
        raise FileNotFoundError(f"Missing {len(missing_paths)} audio files:\n{preview}")

    return df


def portable_audio_path(raw_path: str, normalized: Path, subset_root: Path) -> str:
    if raw_path.startswith("audio/"):
        raw_resolved = (subset_root / raw_path).resolve()
        if raw_resolved.exists():
            return raw_path
    for marker in ("/output/subset/audio/", "/local_output/audio/", "/audio/"):
        if marker in raw_path:
            rel_tail = raw_path.split(marker, 1)[1]
            flat = subset_root / "audio" / rel_tail
            if flat.exists():
                return f"audio/{rel_tail}".replace("\\", "/")
            nested = subset_root / "audio" / DEFAULT_AUDIO_SELECTION_DIR / rel_tail
            if nested.exists():
                return f"audio/{DEFAULT_AUDIO_SELECTION_DIR}/{rel_tail}".replace("\\", "/")
    try:
        rel = normalized.resolve().relative_to(subset_root.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return raw_path


def masked_mean(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(hidden.device).float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1e-8)
    return (hidden * mask).sum(dim=1) / denom


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def manifest_records(df: pd.DataFrame, subset_root: Path) -> List[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        row["audio_path"] = portable_audio_path(
            raw_path=str(row.get("audio_path_raw", row.get("audio_path", ""))),
            normalized=Path(row["audio_path"]),
            subset_root=subset_root,
        )
        row.pop("audio_path_raw", None)
        records.append(row)
    return records


def save_jsonl(records: List[dict], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        raise ValueError(
            f"{name}/{split} row mismatch: expected {expected_rows}, got {emb.shape[0]}"
        )
    if expected_dim is not None and emb.shape[1] != expected_dim:
        raise ValueError(
            f"{name}/{split} dim mismatch: expected {expected_dim}, got {emb.shape[1]}"
        )
    if not np.isfinite(emb).all():
        raise ValueError(f"{name}/{split} contains NaN/Inf values")
    return emb


def save_embedding(out_root: Path, feature_set: str, split: str, emb: np.ndarray) -> Path:
    output_path = out_root / feature_set / f"{split}.npy"
    ensure_parent(output_path)
    np.save(output_path, emb.astype(np.float32, copy=False))
    print(f"[save] {feature_set}/{split}: {emb.shape} -> {output_path}")
    return output_path


def save_frame_embedding(out_root: Path, encoder_name: str, split: str, utt_id: str, frames: np.ndarray) -> Path:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"{encoder_name}/{split}/{utt_id} frame embedding must be 2D, got {frames.shape}")
    if frames.shape[0] < 1:
        raise ValueError(f"{encoder_name}/{split}/{utt_id} has no valid frames")
    if not np.isfinite(frames).all():
        raise ValueError(f"{encoder_name}/{split}/{utt_id} contains NaN/Inf values")
    output_path = out_root / "frame_embeddings" / encoder_name / split / f"{utt_id}.npy"
    ensure_parent(output_path)
    np.save(output_path, frames.astype(np.float32, copy=False))
    return output_path


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class AcousticEncoderAdapter(ABC):
    @abstractmethod
    def extract_batch(self, paths: List[Path]) -> np.ndarray:
        raise NotImplementedError


class _HFEncoderBase(AcousticEncoderAdapter):
    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: Optional[AutoFeatureExtractor],
        device: torch.device,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = device
        self.batch_size = batch_size
        self.num_workers = max(1, num_workers)
        self.target_sr = getattr(feature_extractor, "sampling_rate", 16000) if feature_extractor else 16000

    def _load_single_waveform(self, path: Path) -> np.ndarray:
        waveform, sr = torchaudio.load(path)
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        return waveform.numpy()

    def _load_waveforms(self, paths: Sequence[Path]) -> List[np.ndarray]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            waveforms = list(ex.map(self._load_single_waveform, paths))
        return waveforms

    def _pad_waveforms(self, waveforms: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(w.shape[0] for w in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        mask = np.zeros((len(waveforms), max_len), dtype=np.int64)
        for idx, wav in enumerate(waveforms):
            padded[idx, : len(wav)] = wav
            mask[idx, : len(wav)] = 1
        return torch.from_numpy(padded).to(self.device), torch.from_numpy(mask).to(self.device)

    def _prepare_inputs(self, waveforms: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        if self.feature_extractor is not None:
            batch = self.feature_extractor(
                waveforms,
                sampling_rate=self.target_sr,
                padding=True,
                return_tensors="pt",
            )
            return {k: v.to(self.device) for k, v in batch.items()}

        input_values, attention_mask = self._pad_waveforms(waveforms)
        return {"input_values": input_values, "attention_mask": attention_mask}


class WavLMAdapter(_HFEncoderBase):
    def __init__(self, model_name: str, device: torch.device, batch_size: int, num_workers: int) -> None:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        super().__init__(model, feature_extractor, device, batch_size, num_workers)
        hidden_size = int(getattr(model.config, "hidden_size", -1))
        if hidden_size != 768:
            raise ValueError(f"WavLM hidden size must be 768, got {hidden_size}")

    def _frame_attention_mask(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        frame_mask = self.model._get_feature_vector_attention_mask(
            feature_vector_length=hidden.shape[1],
            attention_mask=attention_mask,
        )
        if frame_mask.shape[1] != hidden.shape[1]:
            raise ValueError(
                "WavLM frame mask length "
                f"{frame_mask.shape[1]} does not match hidden length {hidden.shape[1]}"
            )
        return frame_mask

    def _extract(self, paths: List[Path], return_frames: bool) -> tuple[np.ndarray, list[np.ndarray]]:
        pooled_chunks: List[np.ndarray] = []
        frame_chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            waveforms = self._load_waveforms(batch_paths)
            inputs = self._prepare_inputs(waveforms)
            with torch.inference_mode():
                outputs = self.model(**inputs, return_dict=True)
                hidden = outputs.last_hidden_state
                frame_mask = self._frame_attention_mask(hidden, inputs.get("attention_mask"))
                pooled = masked_mean(hidden, frame_mask)
            pooled_chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
            if return_frames:
                hidden_cpu = hidden.detach().cpu().numpy().astype(np.float32)
                if frame_mask is None:
                    lengths = [hidden.shape[1]] * hidden.shape[0]
                else:
                    lengths = frame_mask.detach().cpu().sum(dim=1).tolist()
                for idx, length in enumerate(lengths):
                    frame_chunks.append(hidden_cpu[idx, : int(length), :])
        return np.vstack(pooled_chunks), frame_chunks

    def extract_batch(self, paths: List[Path]) -> np.ndarray:
        pooled, _ = self._extract(paths, return_frames=False)
        return pooled

    def extract_batch_with_frames(self, paths: List[Path]) -> tuple[np.ndarray, list[np.ndarray]]:
        return self._extract(paths, return_frames=True)


class MFAConformerAdapter(AcousticEncoderAdapter):
    def __init__(
        self,
        checkpoint_path: Path,
        code_root: Path,
        device: torch.device,
        batch_size: int,
        num_workers: int,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MFA checkpoint not found: {checkpoint_path}")
        if checkpoint_path.suffix != ".ckpt":
            raise ValueError(f"MFA-Conformer requires a local .ckpt file, got: {checkpoint_path}")
        if not code_root.exists():
            raise FileNotFoundError(
                f"MFA code root not found: {code_root}. "
                "Provide --mfa-code-root pointing to the MFA-Conformer source."
            )

        root_str = str(code_root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        feature_mod = importlib.import_module("module.feature")
        conformer_cat_mod = importlib.import_module("module.conformer_cat")
        Mel_Spectrogram = getattr(feature_mod, "Mel_Spectrogram")
        conformer_cat = getattr(conformer_cat_mod, "conformer_cat")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("Unsupported MFA checkpoint format: expected dict payload.")
        state_dict = ckpt.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("MFA checkpoint missing state_dict.")
        hparams = ckpt.get("hyper_parameters", {})
        if not isinstance(hparams, dict):
            hparams = {}

        embedding_dim = int(hparams.get("embedding_dim", 192))
        num_blocks = int(hparams.get("num_blocks", 6))
        output_size = int(hparams.get("output_size", 256))
        input_layer = str(hparams.get("input_layer", "conv2d2"))
        pos_enc = str(hparams.get("pos_enc_layer_type", "rel_pos"))
        sample_rate = hparams.get("sample_rate")
        sample_rate = 16000 if sample_rate in (None, 0) else int(sample_rate)

        mel_trans = Mel_Spectrogram(sample_rate=sample_rate)
        encoder = conformer_cat(
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            output_size=output_size,
            input_layer=input_layer,
            pos_enc_layer_type=pos_enc,
        )

        mel_state = {k[len("mel_trans.") :]: v for k, v in state_dict.items() if k.startswith("mel_trans.")}
        enc_state = {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        mel_trans.load_state_dict(mel_state, strict=True)
        encoder.load_state_dict(enc_state, strict=True)

        self.custom_mel = mel_trans.to(device).eval()
        self.custom_encoder = encoder.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = max(1, num_workers)
        self.target_sr = sample_rate
        self.mel_hop_size: Optional[int] = None
        for attr in ("hop_size", "hop_length"):
            val = getattr(mel_trans, attr, None)
            if isinstance(val, int) and val > 0:
                self.mel_hop_size = val
                break
        print(f"[mfa] using local .ckpt loader: {checkpoint_path}")

    def _load_single_waveform(self, path: Path) -> np.ndarray:
        waveform, sr = torchaudio.load(path)
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        return waveform.numpy()

    def _load_waveforms(self, paths: Sequence[Path]) -> List[np.ndarray]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            waveforms = list(ex.map(self._load_single_waveform, paths))
        return waveforms

    def _pad_waveforms(self, waveforms: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(w.shape[0] for w in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        mask = np.zeros((len(waveforms), max_len), dtype=np.int64)
        for idx, wav in enumerate(waveforms):
            padded[idx, : len(wav)] = wav
            mask[idx, : len(wav)] = 1
        return torch.from_numpy(padded).to(self.device), torch.from_numpy(mask).to(self.device)

    def _extract(self, paths: List[Path], return_frames: bool) -> tuple[np.ndarray, list[np.ndarray]]:
        pooled_chunks: List[np.ndarray] = []
        frame_chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            waveforms = self._load_waveforms(batch_paths)
            waveform_lengths = torch.tensor([len(w) for w in waveforms], device=self.device, dtype=torch.float32)
            input_values, _ = self._pad_waveforms(waveforms)

            with torch.inference_mode():
                feat = self.custom_mel(input_values)
                if feat.ndim != 4 or feat.shape[1] != 1:
                    raise ValueError(
                        f"Mel_Spectrogram output shape {tuple(feat.shape)} is not (B, 1, mel_bins, T); "
                        "update the reshape logic if the mel module changed."
                    )
                feat = feat.squeeze(1).permute(0, 2, 1)
                if self.mel_hop_size is not None:
                    lens = torch.div(waveform_lengths, self.mel_hop_size, rounding_mode="floor").int()
                else:
                    lens = torch.round(waveform_lengths / float(input_values.shape[1]) * feat.shape[1]).int()
                lens = lens.clamp(min=1, max=feat.shape[1])
                frame_x, frame_mask = self.custom_encoder.conformer(feat, lens)
                pooled = frame_x.permute(0, 2, 1)
                pooled = self.custom_encoder.pooling(pooled)
                pooled = self.custom_encoder.bn(pooled)
                pooled = pooled.squeeze(-1)
            pooled_chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
            if return_frames:
                frames_cpu = frame_x.detach().cpu().numpy().astype(np.float32)
                mask_cpu = frame_mask.detach().cpu()
                if mask_cpu.ndim == 3:
                    mask_cpu = mask_cpu.squeeze(1)
                elif mask_cpu.ndim != 2:
                    raise ValueError(f"Unexpected frame_mask shape: {tuple(mask_cpu.shape)}")
                frame_lengths = mask_cpu.sum(dim=1).tolist()
                for idx, length in enumerate(frame_lengths):
                    frame_chunks.append(frames_cpu[idx, : int(length), :])
        return np.vstack(pooled_chunks), frame_chunks

    def extract_batch(self, paths: List[Path]) -> np.ndarray:
        pooled, _ = self._extract(paths, return_frames=False)
        return pooled

    def extract_batch_with_frames(self, paths: List[Path]) -> tuple[np.ndarray, list[np.ndarray]]:
        return self._extract(paths, return_frames=True)


class SBERTAdapter:
    def __init__(self, model_name: str, device: torch.device, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SBERT extraction. "
                "Install it in the speech_lab environment with: "
                "pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device.type)

    def extract_texts(self, texts: Sequence[str]) -> np.ndarray:
        clean_texts = ["" if text is None else str(text) for text in texts]
        emb = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(emb, dtype=np.float32)


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    script_dir = Path(__file__).resolve().parent
    mfa_code_root = script_dir / "mfa_conformer_repo"
    manifest_path = args.manifest.resolve()
    subset_root = manifest_path.parent.parent
    out_root = args.out_root.resolve()

    print(f"[setup] device: {device}")
    print(f"[setup] manifest: {manifest_path}")
    print(f"[setup] out_root: {out_root}")
    enabled_encoders = set(args.encoders)
    print(f"[setup] enabled_encoders: {','.join(sorted(enabled_encoders))}")
    if "sbert" in enabled_encoders:
        print(f"[setup] sbert_model: {args.sbert_model}")

        try:
            importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SBERT extraction. "
                "Install it in the speech_lab environment with: "
                "pip install sentence-transformers"
            ) from exc

    df = load_manifest(manifest_path)
    split_plan = infer_split_plan(df)
    expected_by_manifest_split = expected_rows_per_split(split_plan)
    print(f"[manifest] rows: {len(df)}")
    print(
        "[manifest] split_mode: "
        + ",".join(f"{src}->{dst}" for src, dst in split_plan["output_splits"].items())
    )

    for (domain, split), count in df.groupby(["domain", "split"], sort=True).size().items():
        print(f"[manifest] subset {domain}/{split}: {int(count)} rows")

    split_frames: Dict[str, pd.DataFrame] = {}
    for manifest_split in split_plan["manifest_splits"]:
        output_split = split_plan["output_splits"][manifest_split]
        split_df = df[df["split"] == manifest_split].copy()
        expected_rows = expected_by_manifest_split[manifest_split]
        if len(split_df) != expected_rows:
            raise ValueError(f"Expected {expected_rows} {manifest_split} rows, found {len(split_df)}")
        split_frames[output_split] = split_df
        print(f"[manifest] split {manifest_split}->{output_split}: {len(split_df)} rows")

    save_jsonl(manifest_records(df, subset_root), out_root / "manifests" / "all.jsonl")
    for split in OUTPUT_SPLIT_ORDER:
        save_jsonl(
            manifest_records(split_frames[split], subset_root),
            out_root / "manifests" / f"{split}.jsonl",
        )

    split_paths: Dict[str, List[Path]] = {
        split: [Path(p) for p in split_df["audio_path"].tolist()]
            for split, split_df in split_frames.items()
    }
    semantic_texts: Dict[str, Dict[str, List[str]]] = {}
    if "sbert" in enabled_encoders:
        for source in args.semantic_sources:
            config = SEMANTIC_SOURCES[source]
            column = config["column"]
            if column not in df.columns:
                raise ValueError(
                    f"Semantic source '{source}' requires manifest column '{column}'. "
                    "Use --semantic-sources gt for manifests without pseudolabels."
                )
            if source == "pseudo":
                missing_text = df[
                    (df[column].fillna("").astype(str).str.strip() == "")
                    & (df.get("pseudo_status", "ok") != "failed")
                ]
                if len(missing_text):
                    preview = ", ".join(missing_text["utt_id"].astype(str).head(5).tolist())
                    raise ValueError(f"Pseudo semantic source has {len(missing_text)} empty texts: {preview}")
            semantic_texts[source] = {
                split: split_df[column].fillna("").astype(str).tolist()
                for split, split_df in split_frames.items()
            }

    acoustic_raw: Dict[str, Dict[str, np.ndarray]] = {}
    raw_feature_names = {"wavlm": "wavlm_raw", "mfa_conformer": "mfa_conformer_raw"}
    projected_feature_names = {"wavlm": "wavlm_256", "mfa_conformer": "mfa_conformer_256"}
    expected_dims = {"wavlm": 768, "mfa_conformer": 3072}

    if "wavlm" in enabled_encoders or "wavlm_frame" in enabled_encoders:
        print("[extract] encoder=wavlm")
        wavlm_encoder = WavLMAdapter(
            model_name=DEFAULT_WAVLM_MODEL,
            device=device,
            batch_size=args.batch_size,
            num_workers=DEFAULT_NUM_WORKERS,
        )
        save_pooled = "wavlm" in enabled_encoders
        save_frames = "wavlm_frame" in enabled_encoders
        if save_pooled:
            acoustic_raw["wavlm"] = {}
        for split in OUTPUT_SPLIT_ORDER:
            if save_frames:
                emb, frame_embeddings = wavlm_encoder.extract_batch_with_frames(split_paths[split])
                utt_ids = split_frames[split]["utt_id"].astype(str).tolist()
                if len(frame_embeddings) != len(utt_ids):
                    raise ValueError(
                        f"wavlm/{split} frame count mismatch: "
                        f"{len(frame_embeddings)} frame arrays for {len(utt_ids)} manifest rows"
                    )
                for utt_id, frames in zip(utt_ids, frame_embeddings):
                    save_frame_embedding(out_root, "wavlm", split, utt_id, frames)
                print(f"[save] frame_embeddings/wavlm/{split}: {len(frame_embeddings)} files")
            else:
                emb = wavlm_encoder.extract_batch(split_paths[split])
            if save_pooled:
                emb = validate_embedding(
                    raw_feature_names["wavlm"],
                    split,
                    emb,
                    expected_rows=len(split_frames[split]),
                    expected_dim=expected_dims["wavlm"],
                )
                acoustic_raw["wavlm"][split] = emb
                save_embedding(out_root, raw_feature_names["wavlm"], split, emb)
        del wavlm_encoder
        clear_device_cache(device)

    if "mfa_conformer" in enabled_encoders or "mfa_conformer_frame" in enabled_encoders:
        print("[extract] encoder=mfa_conformer")
        mfa_encoder = MFAConformerAdapter(
            checkpoint_path=resolve_mfa_checkpoint_path(args.mfa_checkpoint, script_dir),
            code_root=mfa_code_root.resolve(),
            device=device,
            batch_size=args.batch_size,
            num_workers=DEFAULT_NUM_WORKERS,
        )
        save_pooled = "mfa_conformer" in enabled_encoders
        save_frames = "mfa_conformer_frame" in enabled_encoders
        if save_pooled:
            acoustic_raw["mfa_conformer"] = {}
        for split in OUTPUT_SPLIT_ORDER:
            if save_frames:
                emb, frame_embeddings = mfa_encoder.extract_batch_with_frames(split_paths[split])
                utt_ids = split_frames[split]["utt_id"].astype(str).tolist()
                if len(frame_embeddings) != len(utt_ids):
                    raise ValueError(
                        f"mfa_conformer/{split} frame count mismatch: "
                        f"{len(frame_embeddings)} frame arrays for {len(utt_ids)} manifest rows"
                    )
                for utt_id, frames in zip(utt_ids, frame_embeddings):
                    save_frame_embedding(out_root, "mfa_conformer", split, utt_id, frames)
                print(f"[save] frame_embeddings/mfa_conformer/{split}: {len(frame_embeddings)} files")
            else:
                emb = mfa_encoder.extract_batch(split_paths[split])
            if save_pooled:
                emb = validate_embedding(
                    raw_feature_names["mfa_conformer"],
                    split,
                    emb,
                    expected_rows=len(split_frames[split]),
                    expected_dim=expected_dims["mfa_conformer"],
                )
                acoustic_raw["mfa_conformer"][split] = emb
                save_embedding(out_root, raw_feature_names["mfa_conformer"], split, emb)
        del mfa_encoder
        clear_device_cache(device)

    projected_embeddings: Dict[str, Dict[str, np.ndarray]] = {
        "wavlm_256": {},
        "mfa_conformer_256": {},
    }
    for encoder_name, raw_by_split in acoustic_raw.items():
        stacked = np.vstack([raw_by_split[split] for split in OUTPUT_SPLIT_ORDER])
        in_dim = stacked.shape[1]
        grp = GaussianRandomProjection(n_components=256, random_state=args.seed)
        grp.fit(stacked)

        proj_name = f"{encoder_name}_grp_{in_dim}_to_256.joblib"
        proj_out = out_root / "projections" / proj_name
        ensure_parent(proj_out)
        joblib.dump(grp, proj_out)
        print(f"[projection] {encoder_name}: fit on {stacked.shape}, saved {proj_out}")

        feature_set = projected_feature_names[encoder_name]
        for split in OUTPUT_SPLIT_ORDER:
            projected = grp.transform(raw_by_split[split]).astype(np.float32)
            projected = validate_embedding(
                feature_set,
                split,
                projected,
                expected_rows=len(split_frames[split]),
                expected_dim=256,
            )
            projected_embeddings[feature_set][split] = projected
            save_embedding(out_root, feature_set, split, projected)

    sbert_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    if "sbert" in enabled_encoders:
        print("[extract] encoder=sbert")
        sbert_encoder = SBERTAdapter(
            model_name=args.sbert_model,
            device=device,
            batch_size=args.sbert_batch_size,
        )
        sbert_raw: Dict[str, Dict[str, np.ndarray]] = {
            SEMANTIC_SOURCES[source]["raw_feature"]: {}
            for source in args.semantic_sources
        }
        for source in args.semantic_sources:
            raw_feature = SEMANTIC_SOURCES[source]["raw_feature"]
            for split in OUTPUT_SPLIT_ORDER:
                emb = sbert_encoder.extract_texts(semantic_texts[source][split])
                emb = validate_embedding(
                    raw_feature,
                    split,
                    emb,
                    expected_rows=len(split_frames[split]),
                )
                sbert_raw[raw_feature][split] = emb
                save_embedding(out_root, raw_feature, split, emb)
                if source == "gt":
                    save_embedding(out_root, "sbert_raw", split, emb)
        del sbert_encoder
        clear_device_cache(device)

        stacked = np.vstack(
            [
                sbert_raw[SEMANTIC_SOURCES[source]["raw_feature"]][split]
                for source in args.semantic_sources
                for split in OUTPUT_SPLIT_ORDER
            ]
        )
        in_dim = stacked.shape[1]
        grp = GaussianRandomProjection(n_components=256, random_state=args.seed)
        grp.fit(stacked)

        source_suffix = "_".join(args.semantic_sources)
        proj_name = f"sbert_{source_suffix}_grp_{in_dim}_to_256.joblib"
        proj_out = out_root / "projections" / proj_name
        ensure_parent(proj_out)
        joblib.dump(grp, proj_out)
        print(f"[projection] sbert/{source_suffix}: fit on {stacked.shape}, saved {proj_out}")
        if "gt" in args.semantic_sources:
            legacy_proj_out = out_root / "projections" / f"sbert_grp_{in_dim}_to_256.joblib"
            joblib.dump(grp, legacy_proj_out)
            print(f"[projection] sbert legacy alias saved {legacy_proj_out}")

        for source in args.semantic_sources:
            raw_feature = SEMANTIC_SOURCES[source]["raw_feature"]
            feature_set = SEMANTIC_SOURCES[source]["feature"]
            sbert_embeddings[feature_set] = {}
            for split in OUTPUT_SPLIT_ORDER:
                projected = grp.transform(sbert_raw[raw_feature][split]).astype(np.float32)
                projected = validate_embedding(
                    feature_set,
                    split,
                    projected,
                    expected_rows=len(split_frames[split]),
                    expected_dim=256,
                )
                sbert_embeddings[feature_set][split] = projected
                save_embedding(out_root, feature_set, split, projected)
                if source == "gt":
                    save_embedding(out_root, "sbert", split, projected)

    feature_arrays: Dict[str, Dict[str, np.ndarray]] = {
        **projected_embeddings,
        **sbert_embeddings,
    }
    if "sbert_gt" in sbert_embeddings:
        feature_arrays["sbert"] = sbert_embeddings["sbert_gt"]
    for fused_name, component_names in FUSION_FEATURE_SETS.items():
        missing_components = [
            component_name
            for component_name in component_names
            if not all(split in feature_arrays.get(component_name, {}) for split in OUTPUT_SPLIT_ORDER)
        ]
        if missing_components:
            print(f"[fusion] skipped {fused_name}; missing components: {','.join(missing_components)}")
            continue

        print(f"[fusion] feature_set={fused_name} order={','.join(component_names)}")
        for split in OUTPUT_SPLIT_ORDER:
            components = [feature_arrays[component_name][split] for component_name in component_names]
            fused = np.concatenate(components, axis=1).astype(np.float32, copy=False)
            expected_dim = sum(component.shape[1] for component in components)
            fused = validate_embedding(
                fused_name,
                split,
                fused,
                expected_rows=len(split_frames[split]),
                expected_dim=expected_dim,
            )
            print(f"[fusion] {fused_name}/{split}: dim={fused.shape[1]}")
            save_embedding(out_root, fused_name, split, fused)

    print("[done] feature extraction completed successfully.")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

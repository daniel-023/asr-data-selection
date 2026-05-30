from __future__ import annotations

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class WaveformLoader:
    def __init__(self, target_sample_rate: int, num_workers: int) -> None:
        self.target_sample_rate = target_sample_rate
        self.num_workers = max(1, num_workers)

    def load_one(self, path: Path) -> np.ndarray:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)
        return waveform.numpy()

    def load(self, paths: Sequence[Path]) -> list[np.ndarray]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            return list(executor.map(self.load_one, paths))


class WavLMEncoder:
    def __init__(self, model_id: str, device: torch.device, batch_size: int, num_workers: int) -> None:
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        if int(getattr(self.model.config, "hidden_size", -1)) != 768:
            raise ValueError("WavLM hidden size must be 768.")
        self.device = device
        self.batch_size = batch_size
        self.loader = WaveformLoader(int(self.feature_extractor.sampling_rate), num_workers)

    def extract(self, paths: list[Path]) -> np.ndarray:
        chunks = []
        for offset in range(0, len(paths), self.batch_size):
            waveforms = self.loader.load(paths[offset : offset + self.batch_size])
            inputs = self.feature_extractor(waveforms, sampling_rate=self.loader.target_sample_rate, padding=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = self.model(**inputs, return_dict=True).last_hidden_state
                attention = inputs.get("attention_mask")
                frame_mask = None if attention is None else self.model._get_feature_vector_attention_mask(hidden.shape[1], attention)
                pooled = masked_mean(hidden, frame_mask)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


class MFAConformerEncoder:
    def __init__(self, source_dir: Path, checkpoint: Path, device: torch.device, batch_size: int, num_workers: int) -> None:
        if not source_dir.exists():
            raise FileNotFoundError(f"MFA-Conformer source directory not found: {source_dir}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"MFA-Conformer checkpoint not found: {checkpoint}")
        source = str(source_dir.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        feature_module = importlib.import_module("module.feature")
        conformer_module = importlib.import_module("module.conformer_cat")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("MFA-Conformer checkpoint is missing state_dict.")
        hparams = payload.get("hyper_parameters", {})
        sample_rate = int(hparams.get("sample_rate") or 16000)
        mel = feature_module.Mel_Spectrogram(sample_rate=sample_rate)
        encoder = conformer_module.conformer_cat(
            embedding_dim=int(hparams.get("embedding_dim", 192)),
            num_blocks=int(hparams.get("num_blocks", 6)),
            output_size=int(hparams.get("output_size", 256)),
            input_layer=str(hparams.get("input_layer", "conv2d2")),
            pos_enc_layer_type=str(hparams.get("pos_enc_layer_type", "rel_pos")),
        )
        mel.load_state_dict({key[len("mel_trans.") :]: value for key, value in state_dict.items() if key.startswith("mel_trans.")}, strict=True)
        encoder.load_state_dict({key[len("encoder.") :]: value for key, value in state_dict.items() if key.startswith("encoder.")}, strict=True)
        self.mel = mel.to(device).eval()
        self.encoder = encoder.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.loader = WaveformLoader(sample_rate, num_workers)
        self.hop_size = next(
            (value for value in (getattr(mel, "hop_size", None), getattr(mel, "hop_length", None)) if isinstance(value, int) and value > 0),
            None,
        )

    def extract(self, paths: list[Path]) -> np.ndarray:
        chunks = []
        for offset in range(0, len(paths), self.batch_size):
            waveforms = self.loader.load(paths[offset : offset + self.batch_size])
            lengths = torch.tensor([len(waveform) for waveform in waveforms], device=self.device, dtype=torch.float32)
            max_length = max(len(waveform) for waveform in waveforms)
            padded = np.zeros((len(waveforms), max_length), dtype=np.float32)
            for index, waveform in enumerate(waveforms):
                padded[index, : len(waveform)] = waveform
            values = torch.from_numpy(padded).to(self.device)
            with torch.inference_mode():
                features = self.mel(values)
                if features.ndim != 4 or features.shape[1] != 1:
                    raise ValueError(f"Unexpected MFA mel shape: {tuple(features.shape)}")
                features = features.squeeze(1).permute(0, 2, 1)
                if self.hop_size:
                    frame_lengths = torch.div(lengths, self.hop_size, rounding_mode="floor").int()
                else:
                    frame_lengths = torch.round(lengths / float(values.shape[1]) * features.shape[1]).int()
                frame_lengths = frame_lengths.clamp(min=1, max=features.shape[1])
                frame_states, _ = self.encoder.conformer(features, frame_lengths)
                pooled = self.encoder.pooling(frame_states.permute(0, 2, 1))
                pooled = self.encoder.bn(pooled).squeeze(-1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


class SBERTEncoder:
    def __init__(self, model_id: str, device: torch.device, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device.type)
        self.batch_size = batch_size

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            [str(text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

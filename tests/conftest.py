from __future__ import annotations

import copy
from pathlib import Path

import pytest

from asr_data_selection.config import load_config


@pytest.fixture
def small_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_config())
    config["artifacts"]["root"] = str(tmp_path / "artifacts")
    config["dataset"]["reference_per_domain"] = 1
    config["dataset"]["candidate_per_domain"] = 1
    config["dataset"]["domains"] = [
        {"domain": "alpha", "utt_prefix": "a"},
        {"domain": "beta", "utt_prefix": "b"},
    ]
    config["embeddings"]["projection_dim"] = 2
    config["embeddings"]["encoders"]["wavlm"]["raw_dim"] = 3
    config["embeddings"]["encoders"]["mfa_conformer"]["raw_dim"] = 4
    config["classification"]["epochs"] = 1
    config["classification"]["batch_size"] = 2
    return config


@pytest.fixture
def pseudo_rows(small_config: dict) -> list[dict]:
    rows = []
    for domain in ("alpha", "beta"):
        for role in ("reference", "candidate"):
            utt_id = f"{domain}_{role}"
            rows.append(
                {
                    "utt_id": utt_id,
                    "orig_id": utt_id,
                    "source_sample_key": f"id:{utt_id}",
                    "domain": domain,
                    "split": role,
                    "source_split": role,
                    "audio_path": f"audio/{utt_id}.wav",
                    "transcript": f"{domain} ground truth",
                    "transcript_norm": f"{domain} ground truth",
                    "duration_sec": 3.5,
                    "pseudo_transcript": f"{domain} pseudo label",
                    "pseudo_transcript_norm": f"{domain} pseudo label",
                    "pseudo_model": "openai/whisper-large-v3",
                    "pseudo_status": "ok",
                    "pseudo_error": "",
                }
            )
    return rows

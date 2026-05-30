from __future__ import annotations

import fcntl
import hashlib
import io
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from .config import artifact_path
from .manifests import subset_manifest_path, write_json_rows
from .text import is_filler_or_backchannel_only, normalize_transcript, transcript_tokens


def get_transcript(sample: dict, keys: list[str]) -> str:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def coerce_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def get_duration_sec(sample: dict, audio: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = coerce_float(sample.get(key))
        if value is not None:
            return value
    array = audio.get("array")
    sample_rate = audio.get("sampling_rate")
    if array is not None and sample_rate:
        return float(len(array)) / float(sample_rate)
    audio_bytes = audio.get("bytes")
    if isinstance(audio_bytes, (bytes, bytearray)):
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as handle:
                return float(len(handle)) / float(handle.samplerate)
        except Exception:
            return None
    return None


def sample_key(sample: dict, audio_column: str, id_column: str, transcript_keys: list[str]) -> str:
    for key in (id_column, "id", "utt_id", "segment_id", "audio_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    audio = sample.get(audio_column, {})
    if isinstance(audio, dict):
        if audio.get("path"):
            return f"{audio_column}.path:{audio['path']}"
        if isinstance(audio.get("bytes"), (bytes, bytearray)):
            return f"{audio_column}.sha1:{hashlib.sha1(bytes(audio['bytes'])).hexdigest()}"
    transcript = normalize_transcript(get_transcript(sample, transcript_keys))
    if transcript:
        return f"text:{transcript}"
    return f"fallback:{hashlib.sha1(str(sorted(sample)).encode('utf-8')).hexdigest()}"


def rejection_reason(transcript: str, duration: float | None, min_words: int, min_duration: float) -> str | None:
    normalized = normalize_transcript(transcript, keep_hyphen=False)
    tokens = transcript_tokens(normalized)
    if not normalized:
        return "empty_transcript"
    if len(tokens) < min_words:
        return "too_few_words"
    if duration is None:
        return "missing_duration"
    if duration < min_duration:
        return "too_short_duration"
    if is_filler_or_backchannel_only(tokens):
        return "filler_or_backchannel_only"
    return None


def select_examples(domain: dict, source_split: str, n: int, seed: int, config: dict, excluded: set[str]) -> list[dict]:
    dataset_cfg = config["dataset"]
    try:
        dataset = load_dataset(domain["hf_name"], split=source_split, streaming=True)
    except Exception as exc:
        if any(token in str(exc).lower() for token in ("gated", "authenticated", "authorization", "forbidden", "403")):
            raise RuntimeError(f"Access denied for Hugging Face dataset {domain['hf_name']}. Configure HF_TOKEN and retry.") from exc
        raise
    audio_column = domain.get("audio_column", "audio")
    transcript_keys = domain.get("transcript_keys", ["text"])
    duration_keys = domain.get("duration_keys", [])
    id_column = domain.get("id_column", "id")
    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    dataset = dataset.shuffle(buffer_size=int(dataset_cfg["shuffle_buffer_size"]), seed=seed)
    selected: list[dict] = []
    rejected: Counter = Counter()
    selected_keys: set[str] = set()
    max_scan = max(int(dataset_cfg["max_scan_factor"]) * n, n + 1)
    scanned = 0
    for sample in dataset:
        scanned += 1
        if scanned > max_scan:
            break
        audio = sample.get(audio_column)
        if not isinstance(audio, dict):
            rejected["missing_audio"] += 1
            continue
        transcript = get_transcript(sample, transcript_keys)
        duration = get_duration_sec(sample, audio, duration_keys)
        reason = rejection_reason(transcript, duration, int(dataset_cfg["min_words"]), float(dataset_cfg["min_duration_sec"]))
        if reason:
            rejected[reason] += 1
            continue
        key = sample_key(sample, audio_column, id_column, transcript_keys)
        if key in excluded or key in selected_keys:
            rejected["overlap_or_duplicate"] += 1
            continue
        selected.append({"sample": sample, "transcript": transcript, "duration_sec": duration, "sample_key": key})
        selected_keys.add(key)
        if len(selected) >= n:
            return selected
    raise ValueError(f"{domain['domain']}/{source_split}: requested {n} rows, found {len(selected)} after {scanned} scans; rejected={dict(rejected)}")


def write_audio(audio: dict, destination: Path) -> Path:
    source = audio.get("path")
    if source and os.path.isfile(source):
        output = destination.with_suffix(Path(source).suffix or ".wav")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        return output
    if isinstance(audio.get("bytes"), (bytes, bytearray)):
        output = destination.with_suffix(Path(source or "").suffix or ".wav")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(bytes(audio["bytes"]))
        return output
    array, sample_rate = audio.get("array"), audio.get("sampling_rate")
    if array is None or sample_rate is None:
        raise ValueError("Audio sample has no path, bytes, or decodable waveform.")
    output = destination.with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.asarray(array), sample_rate)
    return output


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    except BlockingIOError as exc:
        raise RuntimeError(f"Another subset-selection run is active: {path}") from exc
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        path.unlink(missing_ok=True)


def run(config: dict) -> Path:
    dataset_cfg = config["dataset"]
    artifact_root = artifact_path(config, "audio_dir").parent
    audio_root = artifact_path(config, "audio_dir")
    metadata_root = artifact_path(config, "metadata_dir")
    rows: list[dict] = []
    with run_lock(metadata_root / ".select_subset.lock"):
        for domain in dataset_cfg["domains"]:
            used: set[str] = set()
            for role in ("reference", "candidate"):
                n = int(dataset_cfg[f"{role}_per_domain"])
                source_split = domain[f"{role}_source_split"]
                seed = int(dataset_cfg["selection_seed"]) + (0 if role == "reference" else 10_000)
                print(f"[select] {domain['domain']}/{role}: target={n} source={source_split}")
                examples = select_examples(domain, source_split, n, seed, config, used)
                for index, item in enumerate(examples, start=1):
                    utt_id = f"{domain['utt_prefix']}_{role}_{index:04d}"
                    audio = item["sample"][domain.get("audio_column", "audio")]
                    output = write_audio(audio, audio_root / f"{domain['domain']}_{role}" / utt_id)
                    original = item["sample"].get(domain.get("id_column", "id"))
                    rows.append(
                        {
                            "utt_id": utt_id,
                            "orig_id": "" if original is None else str(original),
                            "source_sample_key": item["sample_key"],
                            "domain": domain["domain"],
                            "split": role,
                            "source_split": source_split,
                            "audio_path": str(output.relative_to(artifact_root)),
                            "transcript": item["transcript"],
                            "transcript_norm": normalize_transcript(item["transcript"]),
                            "duration_sec": round(float(item["duration_sec"]), 3),
                        }
                    )
                    used.add(item["sample_key"])
    output = subset_manifest_path(config)
    write_json_rows(rows, output)
    print(f"[done] wrote {len(rows)} rows to {output}")
    return output

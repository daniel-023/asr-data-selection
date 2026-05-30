#!/usr/bin/env python3
"""
Prepare a filtered multi-domain reference/candidate subset for the LLM experiment.

Outputs under llm_domain_experiment/:
- audio/data_selection_13200/<domain>_reference/
- audio/data_selection_13200/<domain>_candidate/
- metadata/selected_manifest.json
- metadata/selected_manifest.jsonl
- metadata/reference.jsonl
- metadata/candidate.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_ROOT = str(DEFAULT_EXPERIMENT_ROOT.resolve())
DEFAULT_MIN_DURATION_SEC = 3.0
DEFAULT_MIN_WORDS = 3
DEFAULT_SHUFFLE_BUFFER_SIZE = 10000
DEFAULT_MAX_SCAN_FACTOR = 200
DEFAULT_AUDIO_SUBDIR = "audio/data_selection_13200"
REFERENCE_N = 300
CANDIDATE_N = 3000
SPLIT_ORDER = ("reference", "candidate")
MANIFEST_NAME = "selected_manifest.json"
MANIFEST_JSONL_NAME = "selected_manifest.jsonl"


DATASET_CONFIGS = [
    {
        "domain": "imda",
        "hf_name": "pengyizhou/nsc-imda-part6",
        "reference_source_split": "test",
        "candidate_source_split": "train",
        "audio_column": "audio",
        "transcript_keys": ["text"],
        "id_column": "id",
        "duration_keys": ["duration", "duration_sec", "audio_duration", "length_seconds"],
        "reference_n": REFERENCE_N,
        "candidate_n": CANDIDATE_N,
        "utt_prefix": "im",
    },
    {
        "domain": "gigaspeech",
        "hf_name": "pengyizhou/gigaspeech_subset_270h",
        "reference_source_split": "test",
        "candidate_source_split": "train",
        "audio_column": "audio",
        "transcript_keys": ["text"],
        "id_column": "id",
        "duration_keys": ["duration", "duration_sec", "audio_duration", "length_seconds"],
        "reference_n": REFERENCE_N,
        "candidate_n": CANDIDATE_N,
        "utt_prefix": "gs",
    },
    {
        "domain": "librispeech",
        "hf_name": "openslr/librispeech_asr",
        "reference_source_split": "test.clean",
        "candidate_source_split": "train.clean.100",
        "audio_column": "audio",
        "transcript_keys": ["text"],
        "id_column": "id",
        "duration_keys": [],
        "reference_n": REFERENCE_N,
        "candidate_n": CANDIDATE_N,
        "utt_prefix": "ls",
    },
    {
        "domain": "svarah",
        "hf_name": "ai4bharat/Svarah",
        "reference_source_split": "test",
        "candidate_source_split": "test",
        "audio_column": "audio_filepath",
        "transcript_keys": ["text"],
        "id_column": "id",
        "duration_keys": ["duration"],
        "reference_n": REFERENCE_N,
        "candidate_n": CANDIDATE_N,
        "utt_prefix": "sv",
    },
]

DEFAULT_TRANSCRIPT_KEYS = (
    "text",
    "sentence",
    "transcript",
    "normalized_text",
    "utterance",
)

DEFAULT_DURATION_KEYS = (
    "duration",
    "duration_sec",
    "audio_duration",
    "length_seconds",
)

BACKCHANNEL_WORDS = {
    "yeah",
    "yea",
    "yep",
    "yup",
    "okay",
    "ok",
    "uh",
    "uhh",
    "uhm",
    "um",
    "hmm",
    "mm",
    "mmm",
    "ah",
    "oh",
    "eh",
    "huh",
    "uhhuh",
    "uhhuhh",
    "uhhuhuh",
    "mhm",
    "mmhm",
    "hmmm",
    "right",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 13,200-utterance reference/candidate subset for domain experiment "
            "(300 reference and 3000 candidate samples per domain)."
        )
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Final output root directory.",
    )
    parser.add_argument(
        "--audio-subdir",
        default=DEFAULT_AUDIO_SUBDIR,
        help=(
            "Audio directory relative to --output-root. Keep this relative so manifest "
            "audio_path values are portable across local and NSCC roots."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260331, help="Seed for deterministic shuffle.")
    return parser.parse_args()


def infer_extension_from_path(path_str: str | None) -> str:
    if not path_str:
        return ".wav"
    suffix = Path(path_str).suffix.lower()
    return suffix if suffix else ".wav"


def get_transcript(
    sample: dict,
    transcript_keys: list[str] | tuple[str, ...] | None = None,
) -> str:
    keys = DEFAULT_TRANSCRIPT_KEYS if transcript_keys is None else transcript_keys
    for key in keys:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_transcript(text: str, *, keep_hyphen: bool = False) -> str:
    text = text.lower().strip()
    allowed_pattern = r"[^a-z0-9\s'-]" if keep_hyphen else r"[^a-z0-9\s']"
    text = re.sub(allowed_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcript_tokens(normalized_text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalized_text)


def is_filler_or_backchannel_only(tokens: list[str]) -> bool:
    if not tokens:
        return False
    return all(token in BACKCHANNEL_WORDS for token in tokens)


def coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def get_duration_sec(
    sample: dict,
    audio_obj: dict,
    duration_keys: list[str] | tuple[str, ...] | None = None,
) -> float | None:
    keys = DEFAULT_DURATION_KEYS if duration_keys is None else duration_keys
    for key in keys:
        sec = coerce_float(sample.get(key))
        if sec is not None and sec > 0:
            return sec

    array = audio_obj.get("array")
    sr = audio_obj.get("sampling_rate")
    if array is not None and sr:
        try:
            return float(len(array)) / float(sr)
        except Exception:
            return None

    audio_bytes = audio_obj.get("bytes")
    if isinstance(audio_bytes, (bytes, bytearray)):
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                return float(len(f)) / float(f.samplerate)
        except Exception:
            return None
    return None


def get_sample_unique_key(
    sample: dict,
    audio_column: str,
    id_column: str,
    transcript_keys: list[str] | tuple[str, ...],
) -> str:
    preferred_id_keys = [id_column, "id", "utt_id", "segment_id", "audio_id"]
    for key in preferred_id_keys:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"

    audio_obj = sample.get(audio_column)
    if isinstance(audio_obj, dict):
        audio_path = audio_obj.get("path")
        if audio_path:
            return f"{audio_column}.path:{audio_path}"
        audio_bytes = audio_obj.get("bytes")
        if isinstance(audio_bytes, (bytes, bytearray)):
            return f"{audio_column}.sha1:{hashlib.sha1(bytes(audio_bytes)).hexdigest()}"

    transcript = get_transcript(sample, transcript_keys=transcript_keys)
    normalized = normalize_transcript(transcript)
    if normalized:
        return f"text:{normalized}"

    return f"fallback:{hashlib.sha1(str(sorted(sample.keys())).encode('utf-8')).hexdigest()}"


def rejection_reason(
    transcript: str,
    duration_sec: float | None,
    min_words: int,
    min_duration_sec: float,
) -> str | None:
    normalized = normalize_transcript(transcript)
    tokens = transcript_tokens(normalized)

    if not normalized:
        return "empty_transcript"
    if len(tokens) < min_words:
        return "too_few_words"
    if duration_sec is None:
        return "missing_duration"
    if duration_sec < min_duration_sec:
        return "too_short_duration"
    if is_filler_or_backchannel_only(tokens):
        return "filler_or_backchannel_only"
    return None


def write_audio(
    sample_audio: dict,
    dest_base: Path,
) -> Path:
    src_path = sample_audio.get("path")
    ext = infer_extension_from_path(src_path)
    dest_path = dest_base.with_suffix(ext)

    if src_path and os.path.isfile(src_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        return dest_path.resolve()

    audio_bytes = sample_audio.get("bytes")
    if isinstance(audio_bytes, (bytes, bytearray)):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as f:
            f.write(audio_bytes)
        return dest_path.resolve()

    array = sample_audio.get("array")
    sr = sample_audio.get("sampling_rate")
    if array is None or sr is None:
        raise ValueError("Audio sample missing both copyable path and decodable array/sampling_rate.")

    dest_path = dest_base.with_suffix(".wav")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest_path, np.asarray(array), sr)
    return dest_path.resolve()


def pick_clean_examples(
    dataset_name: str,
    split_name: str,
    audio_column: str,
    transcript_keys: list[str] | tuple[str, ...],
    duration_keys: list[str] | tuple[str, ...],
    id_column: str,
    n: int,
    seed: int,
    min_words: int,
    min_duration_sec: float,
    shuffle_buffer_size: int,
    max_scan_factor: int,
    excluded_sample_keys: set[str] | None = None,
) -> list[dict]:
    try:
        ds = load_dataset(dataset_name, split=split_name, streaming=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ("gated", "authenticated", "authorization", "forbidden", "403")):
            raise RuntimeError(
                f"Access denied for dataset '{dataset_name}' (split '{split_name}'). "
                "This run is configured to hard-fail on gated/private datasets. "
                "Provide valid Hugging Face authentication (e.g., HF_TOKEN) and retry."
            ) from exc
        raise
    ds = ds.cast_column(audio_column, Audio(decode=False))
    shuffled = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    selected: list[dict] = []
    rejection_counts: Counter = Counter()
    selected_keys: set[str] = set()
    excluded_keys: set[str] = excluded_sample_keys or set()
    scanned = 0
    max_scan = max(n * max_scan_factor, n + 1)

    for sample in shuffled:
        scanned += 1
        if scanned > max_scan:
            break

        audio_obj = sample.get(audio_column)
        if not isinstance(audio_obj, dict):
            rejection_counts["missing_audio"] += 1
            continue

        transcript = get_transcript(sample, transcript_keys=transcript_keys)
        duration_sec = get_duration_sec(sample, audio_obj, duration_keys=duration_keys)
        reason = rejection_reason(
            transcript=transcript,
            duration_sec=duration_sec,
            min_words=min_words,
            min_duration_sec=min_duration_sec,
        )
        if reason:
            rejection_counts[reason] += 1
            continue

        sample_key = get_sample_unique_key(
            sample=sample,
            audio_column=audio_column,
            id_column=id_column,
            transcript_keys=transcript_keys,
        )
        if sample_key in selected_keys or sample_key in excluded_keys:
            rejection_counts["overlap_or_duplicate"] += 1
            continue

        selected.append(
            {
                "sample": sample,
                "transcript": transcript,
                "duration_sec": float(duration_sec),  # duration_sec is non-None if accepted
                "sample_key": sample_key,
            }
        )
        selected_keys.add(sample_key)
        if len(selected) % 500 == 0:
            print(
                f"[progress] {dataset_name}/{split_name}: selected {len(selected)}/{n} "
                f"after scanning {scanned}",
                flush=True,
            )
        if len(selected) >= n:
            break

    if len(selected) != n:
        raise ValueError(
            f"{dataset_name}/{split_name}: requested {n} clean samples but got {len(selected)} "
            f"after scanning {scanned}. Rejections: {dict(rejection_counts)}"
        )

    return selected


def acquire_run_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        holder = "unknown"
        try:
            holder = lock_path.read_text(encoding="utf-8").strip() or "unknown"
        except Exception:
            pass
        lock_fh.close()
        raise RuntimeError(
            f"Another run is already active for this output root. lock={lock_path} holder={holder}"
        ) from exc

    lock_fh.seek(0)
    lock_fh.truncate(0)
    lock_fh.write(f"pid={os.getpid()}\n")
    lock_fh.flush()
    return lock_fh


def release_run_lock(lock_fh, lock_path: Path) -> None:
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_fh.close()
    except Exception:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_relative_subdir(value: str, *, arg_name: str) -> Path:
    subdir = Path(value)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(f"{arg_name} must be a relative path without '..': {value}")
    return subdir


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    audio_subdir = resolve_relative_subdir(args.audio_subdir, arg_name="--audio-subdir")
    metadata_root = output_root / "metadata"
    try:
        metadata_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create output directory: {metadata_root}\n"
            "Use a writable path via --output-root. "
            "For local macOS runs, use something like:\n"
            "  --output-root '/Users/<you>/.../llm_domain_experiment'"
        ) from exc
    lock_path = metadata_root / ".extract_subset.lock"
    lock_fh = acquire_run_lock(lock_path)

    try:
        rows: list[dict[str, object]] = []
        summary_counts: dict[tuple[str, str], int] = defaultdict(int)

        for cfg in DATASET_CONFIGS:
            domain = cfg["domain"]
            dataset_name = cfg["hf_name"]
            audio_column = cfg.get("audio_column", "audio")
            transcript_keys = cfg.get("transcript_keys", list(DEFAULT_TRANSCRIPT_KEYS))
            duration_keys = cfg.get("duration_keys", list(DEFAULT_DURATION_KEYS))
            id_column = cfg.get("id_column", "id")
            used_sample_keys_for_domain: set[str] = set()

            for split in SPLIT_ORDER:
                n = int(cfg[f"{split}_n"])
                source_split = str(cfg[f"{split}_source_split"])
                out_dir = output_root / audio_subdir / f"{domain}_{split}"
                out_dir.mkdir(parents=True, exist_ok=True)

                local_seed = args.seed + (0 if split == "reference" else 10_000)
                print(
                    f"[select] {domain} {split}: target={n}, source_split='{source_split}', seed={local_seed}",
                    flush=True,
                )
                picked = pick_clean_examples(
                    dataset_name=dataset_name,
                    split_name=source_split,
                    audio_column=audio_column,
                    transcript_keys=transcript_keys,
                    duration_keys=duration_keys,
                    id_column=id_column,
                    n=n,
                    seed=local_seed,
                    min_words=DEFAULT_MIN_WORDS,
                    min_duration_sec=DEFAULT_MIN_DURATION_SEC,
                    shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,
                    max_scan_factor=DEFAULT_MAX_SCAN_FACTOR,
                    excluded_sample_keys=used_sample_keys_for_domain,
                )

                for idx, picked_item in enumerate(picked, start=1):
                    sample = picked_item["sample"]
                    transcript = str(picked_item["transcript"])
                    transcript_norm = normalize_transcript(transcript, keep_hyphen=True)
                    duration_sec = float(picked_item["duration_sec"])
                    sample_key = str(picked_item["sample_key"])
                    orig_id_raw = sample.get(id_column)
                    orig_id = "" if orig_id_raw is None else str(orig_id_raw)
                    utt_id = f"{cfg['utt_prefix']}_{split}_{idx:04d}"

                    dest_base = out_dir / utt_id
                    audio_obj = sample.get(audio_column)
                    if audio_obj is None:
                        raise ValueError(
                            f"Sample from {dataset_name}/{source_split} missing '{audio_column}' field."
                        )

                    written_path = write_audio(
                        sample_audio=audio_obj,
                        dest_base=dest_base,
                    )
                    audio_path = str(written_path.relative_to(output_root))

                    rows.append(
                        {
                            "utt_id": utt_id,
                            "orig_id": orig_id,
                            "source_sample_key": sample_key,
                            "domain": domain,
                            "split": split,
                            "source_split": source_split,
                            "audio_path": audio_path,
                            "transcript": transcript,
                            "transcript_norm": transcript_norm,
                            "duration_sec": round(duration_sec, 3),
                        }
                    )
                    summary_counts[(domain, split)] += 1
                    used_sample_keys_for_domain.add(sample_key)

                print(f"[done] {domain} {split}: selected {n} from source split '{source_split}'")

        json_manifest = metadata_root / MANIFEST_NAME
        with json_manifest.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        jsonl_manifest = metadata_root / MANIFEST_JSONL_NAME
        write_jsonl(rows, jsonl_manifest)
        for split in SPLIT_ORDER:
            split_rows = [row for row in rows if row["split"] == split]
            write_jsonl(split_rows, metadata_root / f"{split}.jsonl")

        print(f"[done] wrote {len(rows)} rows")
        print(f"[done] json manifest: {json_manifest}")
        print(f"[done] jsonl manifest: {jsonl_manifest}")
        print("[summary] final counts by domain/split:")
        for domain, split in sorted(summary_counts):
            print(f"  - {domain} {split}: {summary_counts[(domain, split)]}")
    finally:
        release_run_lock(lock_fh, lock_path)


if __name__ == "__main__":
    main()

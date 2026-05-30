#!/usr/bin/env python3
"""Generate Whisper large-v3 pseudolabels for the selected 13,200-row subset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_AUDIO_SELECTION_DIR = "data_selection_13200"
REQUIRED_COLUMNS = {"utt_id", "domain", "split", "audio_path", "transcript_norm"}
EXPECTED_ROWS = 13_200


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_manifest = script_dir / "metadata" / "selected_manifest.json"
    parser = argparse.ArgumentParser(
        description="Transcribe selected subset audio with Whisper large-v3 and save pseudolabel manifest."
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest, help="Input selected manifest JSON.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON manifest. Default: <manifest stem>.whisper_large_v3.json",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Output JSONL manifest. Default: <manifest stem>.whisper_large_v3.jsonl",
    )
    parser.add_argument(
        "--checkpoint-jsonl",
        type=Path,
        default=None,
        help="Resumable checkpoint JSONL. Default: <manifest stem>.whisper_large_v3.partial.jsonl",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Whisper model id.")
    parser.add_argument("--batch-size", type=int, default=8, help="ASR pipeline batch size.")
    parser.add_argument(
        "--chunk-length-s",
        type=float,
        default=30.0,
        help="Chunk length in seconds for long audio. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--stride-length-s",
        type=float,
        default=5.0,
        help="Chunk stride/overlap in seconds when chunking is enabled. Use 0 for no stride.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--language", default="english", help="Whisper language prompt.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Whisper task.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoint/output rows.")
    return parser.parse_args()


def normalize_transcript(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def manifest_output_paths(manifest: Path, out_json: Path | None, out_jsonl: Path | None, checkpoint: Path | None) -> tuple[Path, Path, Path]:
    stem = manifest.stem
    json_path = out_json or manifest.with_name(f"{stem}.whisper_large_v3.json")
    jsonl_path = out_jsonl or manifest.with_name(f"{stem}.whisper_large_v3.jsonl")
    checkpoint_path = checkpoint or manifest.with_name(f"{stem}.whisper_large_v3.partial.jsonl")
    return json_path, jsonl_path, checkpoint_path


def load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    for idx, row in enumerate(rows):
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ValueError(f"Manifest row {idx} is missing required columns: {sorted(missing)}")
    if limit is None and len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} manifest rows, found {len(rows)}")
    return rows[:limit] if limit is not None else rows


def resolve_audio_path(raw_path: str, subset_root: Path) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct.resolve()
    if not direct.is_absolute():
        rel = (subset_root / direct).resolve()
        if rel.exists():
            return rel
        if direct.parts and direct.parts[0] == "audio":
            nested = subset_root / "audio" / DEFAULT_AUDIO_SELECTION_DIR / Path(*direct.parts[1:])
            if nested.exists():
                return nested.resolve()
    for marker in ("/output/subset/audio/", "/local_output/audio/", "/audio/"):
        if marker in raw_path:
            rel_tail = raw_path.split(marker, 1)[1]
            flat = subset_root / "audio" / rel_tail
            if flat.exists():
                return flat.resolve()
            nested = subset_root / "audio" / DEFAULT_AUDIO_SELECTION_DIR / rel_tail
            if nested.exists():
                return nested.resolve()
    return direct


def load_checkpoint(path: Path, overwrite: bool) -> dict[str, dict]:
    if overwrite or not path.exists():
        return {}
    completed: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            utt_id = str(row.get("utt_id", ""))
            if (
                utt_id
                and row.get("pseudo_status") == "ok"
                and str(row.get("pseudo_transcript_norm", "")).strip()
            ):
                completed[utt_id] = row
    return completed


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_device(device_arg: str) -> tuple[str, int, torch.dtype]:
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return "cuda", 0, torch.float16
    return "cpu", -1, torch.float32


def build_asr_pipeline(model_id: str, device_arg: str):
    device_name, pipe_device, torch_dtype = resolve_device(device_arg)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if device_name == "cuda":
        model.to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=pipe_device,
    )


def pseudolabel_row(row: dict, text: str, model_id: str, status: str, error: str = "") -> dict:
    out = dict(row)
    normalized = normalize_transcript(text)
    if status == "ok" and not normalized:
        status = "empty"
        error = error or "empty normalized pseudotranscript"
    out["pseudo_transcript"] = text
    out["pseudo_transcript_norm"] = normalized
    out["pseudo_model"] = model_id
    out["pseudo_status"] = status
    out["pseudo_error"] = error
    return out


def incomplete_pseudolabel_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("pseudo_status") != "ok"
        or not str(row.get("pseudo_transcript_norm", "")).strip()
    ]


def format_incomplete_preview(rows: list[dict], max_rows: int = 10) -> str:
    preview_lines = []
    for row in rows[:max_rows]:
        preview_lines.append(
            "  - "
            f"utt_id={row.get('utt_id')} "
            f"domain={row.get('domain')} "
            f"split={row.get('split')} "
            f"status={row.get('pseudo_status')} "
            f"error={row.get('pseudo_error', '')!r} "
            f"audio_path={row.get('audio_path')} "
            f"pseudo_transcript={row.get('pseudo_transcript', '')!r}"
        )
    if len(rows) > max_rows:
        preview_lines.append(f"  ... plus {len(rows) - max_rows} more")
    return "\n".join(preview_lines)


def transcribe_batch(
    asr,
    paths: list[Path],
    batch_size: int,
    language: str,
    task: str,
    chunk_length_s: float,
    stride_length_s: float,
) -> list[str]:
    call_kwargs = {
        "batch_size": batch_size,
        "generate_kwargs": {"language": language, "task": task},
    }
    if chunk_length_s > 0:
        call_kwargs["chunk_length_s"] = chunk_length_s
        if stride_length_s > 0:
            call_kwargs["stride_length_s"] = stride_length_s
    results = asr(
        [str(path) for path in paths],
        **call_kwargs,
    )
    if isinstance(results, dict):
        results = [results]
    return [str(result.get("text", "")).strip() for result in results]


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    subset_root = manifest.parent.parent
    out_json, out_jsonl, checkpoint_jsonl = manifest_output_paths(
        manifest, args.out_json, args.out_jsonl, args.checkpoint_jsonl
    )
    rows = load_manifest(manifest, limit=args.limit)
    completed = load_checkpoint(checkpoint_jsonl, overwrite=args.overwrite)

    for row in rows:
        path = resolve_audio_path(str(row["audio_path"]), subset_root)
        if not path.exists():
            raise FileNotFoundError(f"Missing audio for {row['utt_id']}: {path}")

    pending = [row for row in rows if str(row["utt_id"]) not in completed]
    print(f"[setup] manifest={manifest}")
    print(f"[setup] rows={len(rows)} completed={len(completed)} pending={len(pending)}")
    print(f"[setup] model={args.model_id}")
    print(f"[setup] chunk_length_s={args.chunk_length_s} stride_length_s={args.stride_length_s}")

    if pending:
        asr = build_asr_pipeline(args.model_id, args.device)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            paths = [resolve_audio_path(str(row["audio_path"]), subset_root) for row in batch]
            try:
                texts = transcribe_batch(
                    asr,
                    paths,
                    args.batch_size,
                    args.language,
                    args.task,
                    args.chunk_length_s,
                    args.stride_length_s,
                )
                out_rows = [
                    pseudolabel_row(row, text, args.model_id, "ok")
                    for row, text in zip(batch, texts)
                ]
            except Exception as batch_exc:
                print(f"[warn] batch failed at offset {start}: {batch_exc}")
                out_rows = []
                for row, path in zip(batch, paths):
                    try:
                        text = transcribe_batch(
                            asr,
                            [path],
                            1,
                            args.language,
                            args.task,
                            args.chunk_length_s,
                            args.stride_length_s,
                        )[0]
                        out_rows.append(pseudolabel_row(row, text, args.model_id, "ok"))
                    except Exception as exc:
                        out_rows.append(pseudolabel_row(row, "", args.model_id, "failed", str(exc)))
            for out_row in out_rows:
                completed[str(out_row["utt_id"])] = out_row
                append_jsonl(out_row, checkpoint_jsonl)
            print(f"[progress] {min(start + len(batch), len(pending))}/{len(pending)} pending rows processed")

    final_rows = [completed[str(row["utt_id"])] for row in rows]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(final_rows, out_jsonl)
    ok = sum(1 for row in final_rows if row.get("pseudo_status") == "ok" and row.get("pseudo_transcript_norm"))
    failed = sum(1 for row in final_rows if row.get("pseudo_status") != "ok")
    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_jsonl}")
    print(f"[summary] ok_nonempty={ok} failed={failed}")
    incomplete_rows = incomplete_pseudolabel_rows(final_rows)
    if incomplete_rows:
        print("[incomplete] rows requiring inspection:")
        print(format_incomplete_preview(incomplete_rows))
    if ok != len(final_rows) or failed:
        raise RuntimeError(
            f"Pseudolabel generation incomplete: ok_nonempty={ok}, failed={failed}, total={len(final_rows)}"
        )


if __name__ == "__main__":
    main()

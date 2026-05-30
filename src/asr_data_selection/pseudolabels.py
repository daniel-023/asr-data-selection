from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .manifests import (
    append_jsonl,
    checkpoint_manifest_path,
    read_json_rows,
    resolve_audio_path,
    subset_manifest_path,
    validate_subset_rows,
    write_json_rows,
)
from .text import normalize_transcript


def resolve_device(device: str) -> tuple[str, int, torch.dtype]:
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return "cuda", 0, torch.float16
    return "cpu", -1, torch.float32


def build_asr_pipeline(model_id: str, device: str):
    device_name, pipeline_device, dtype = resolve_device(device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
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
        torch_dtype=dtype,
        device=pipeline_device,
    )


def pseudolabel_row(row: dict, text: str, model_id: str, *, fallback_to_gt: bool, error: str = "") -> dict:
    output = dict(row)
    normalized = normalize_transcript(text)
    status = "ok"
    if not normalized and fallback_to_gt:
        normalized = str(row["transcript_norm"]).strip()
        status = "manual_fallback_gt"
        error = error or "Whisper output normalized to empty text; substituted transcript_norm."
    elif not normalized:
        status = "empty"
        error = error or "Whisper output normalized to empty text."
    output.update(
        {
            "pseudo_transcript": text,
            "pseudo_transcript_norm": normalized,
            "pseudo_model": model_id,
            "pseudo_status": status,
            "pseudo_error": error,
        }
    )
    return output


def failed_row(row: dict, model_id: str, error: str) -> dict:
    output = dict(row)
    output.update(
        {
            "pseudo_transcript": "",
            "pseudo_transcript_norm": "",
            "pseudo_model": model_id,
            "pseudo_status": "failed",
            "pseudo_error": error,
        }
    )
    return output


def is_complete(row: dict) -> bool:
    return row.get("pseudo_status") in {"ok", "manual_fallback_gt"} and bool(str(row.get("pseudo_transcript_norm", "")).strip())


def load_checkpoint(path: Path, overwrite: bool) -> dict[str, dict]:
    if overwrite or not path.exists():
        return {}
    completed: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            import json

            row = json.loads(line)
            if row.get("utt_id") and is_complete(row):
                completed[str(row["utt_id"])] = row
    return completed


def transcribe(asr, paths: list[Path], settings: dict, batch_size: int) -> list[str]:
    kwargs = {
        "batch_size": batch_size,
        "generate_kwargs": {"language": settings["language"], "task": settings["task"]},
    }
    if float(settings["chunk_length_s"]) > 0:
        kwargs["chunk_length_s"] = float(settings["chunk_length_s"])
        if float(settings["stride_length_s"]) > 0:
            kwargs["stride_length_s"] = float(settings["stride_length_s"])
    results = asr([str(path) for path in paths], **kwargs)
    results = [results] if isinstance(results, dict) else results
    return [str(result.get("text", "")).strip() for result in results]


def run(config: dict, *, overwrite: bool = False, limit: int | None = None) -> Path:
    settings = config["pseudolabels"]
    source = subset_manifest_path(config)
    output = subset_manifest_path(config, pseudolabels=True)
    checkpoint = checkpoint_manifest_path(config)
    rows = read_json_rows(source)
    validate_subset_rows(rows, config)
    rows = rows[:limit] if limit is not None else rows
    completed = load_checkpoint(checkpoint, overwrite)
    pending = [row for row in rows if str(row["utt_id"]) not in completed]
    print(f"[setup] rows={len(rows)} completed={len(completed)} pending={len(pending)}")
    if pending:
        asr = build_asr_pipeline(settings["model_id"], settings["device"])
        batch_size = int(settings["batch_size"])
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            paths = [resolve_audio_path(str(row["audio_path"]), config) for row in batch]
            missing = [path for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing audio file: {missing[0]}")
            try:
                texts = transcribe(asr, paths, settings, batch_size)
                out_rows = [
                    pseudolabel_row(
                        row,
                        text,
                        settings["model_id"],
                        fallback_to_gt=bool(settings["fallback_to_gt_on_empty"]),
                    )
                    for row, text in zip(batch, texts)
                ]
            except Exception as batch_error:
                print(f"[warn] batch failed at offset {offset}: {batch_error}")
                out_rows = []
                for row, path in zip(batch, paths):
                    try:
                        text = transcribe(asr, [path], settings, 1)[0]
                        out_rows.append(
                            pseudolabel_row(
                                row,
                                text,
                                settings["model_id"],
                                fallback_to_gt=bool(settings["fallback_to_gt_on_empty"]),
                            )
                        )
                    except Exception as error:
                        out_rows.append(failed_row(row, settings["model_id"], str(error)))
            for row in out_rows:
                completed[str(row["utt_id"])] = row
                append_jsonl(row, checkpoint)
            print(f"[progress] {min(offset + len(batch), len(pending))}/{len(pending)}")
    final_rows = [completed[str(row["utt_id"])] for row in rows]
    write_json_rows(final_rows, output)
    ok = sum(row["pseudo_status"] == "ok" for row in final_rows)
    fallback = sum(row["pseudo_status"] == "manual_fallback_gt" for row in final_rows)
    failed = sum(not is_complete(row) for row in final_rows)
    print(f"[summary] ok={ok} fallback_gt={fallback} failed={failed}")
    if failed:
        raise RuntimeError(f"Pseudolabel generation incomplete: failed={failed}, total={len(final_rows)}")
    return output

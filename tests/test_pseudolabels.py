from __future__ import annotations

import json

from asr_data_selection.pseudolabels import is_complete, load_checkpoint, pseudolabel_row


def test_empty_whisper_output_uses_documented_gt_fallback(tmp_path) -> None:
    source = {"utt_id": "im_candidate_0822", "transcript_norm": "o dot s g"}
    row = pseudolabel_row(source, ". . . .", "openai/whisper-large-v3", fallback_to_gt=True)
    assert row["pseudo_transcript"] == ". . . ."
    assert row["pseudo_transcript_norm"] == "o dot s g"
    assert row["pseudo_status"] == "manual_fallback_gt"
    assert "substituted" in row["pseudo_error"]
    assert is_complete(row)


def test_checkpoint_resume_accepts_fallback_rows(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    row = {
        "utt_id": "im_candidate_0822",
        "pseudo_transcript_norm": "o dot s g",
        "pseudo_status": "manual_fallback_gt",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_checkpoint(path, overwrite=False) == {"im_candidate_0822": row}

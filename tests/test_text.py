from asr_data_selection.text import is_filler_or_backchannel_only, normalize_transcript


def test_normalize_transcript_matches_reported_policy() -> None:
    assert normalize_transcript("  Hello,\tWORLD!  it's high-quality. ") == "hello world it's high-quality"


def test_fillers_are_detected() -> None:
    assert is_filler_or_backchannel_only(["uh", "mhm"])
    assert not is_filler_or_backchannel_only(["uh", "speech"])

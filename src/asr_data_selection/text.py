from __future__ import annotations

import re


BACKCHANNEL_WORDS = {
    "ah",
    "eh",
    "hmmm",
    "huh",
    "mhm",
    "mmhm",
    "oh",
    "right",
    "uh",
    "uhhuh",
    "uhhuhh",
    "uhhuhuh",
    "um",
}


def normalize_transcript(text: str, *, keep_hyphen: bool = True) -> str:
    text = str(text).lower().strip()
    allowed_pattern = r"[^a-z0-9\s'-]" if keep_hyphen else r"[^a-z0-9\s']"
    text = re.sub(allowed_pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def transcript_tokens(normalized_text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalized_text)


def is_filler_or_backchannel_only(tokens: list[str]) -> bool:
    return bool(tokens) and all(token in BACKCHANNEL_WORDS for token in tokens)

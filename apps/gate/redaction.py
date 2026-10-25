"""Sanitize credentials before errors cross logging, persistence or API boundaries."""
import re

_SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk_agent_[A-Za-z0-9]{20,}"),
    re.compile(r"ey[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}"),
    re.compile(r"(?:HF_TOKEN)['\":= ]+([A-Za-z0-9_\-\.]{20,})"),
]
_QUERY_SECRET = re.compile(
    r"([?&](?:colab-runtime-proxy-token|access_token|refresh_token|id_token|token|api_key|signature)=)[^&\s\"'<>#\\]+",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    text = _QUERY_SECRET.sub(r"\1[REDACTED_SECRET]", text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text

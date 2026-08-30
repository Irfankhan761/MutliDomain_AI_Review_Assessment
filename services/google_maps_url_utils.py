from __future__ import annotations

"""Validation helpers for user-supplied Google Maps place URLs."""

import re
from urllib.parse import urlparse


_GOOGLE_HOST_RE = re.compile(
    r"^(?:(?:www|maps)\.)?google\.[a-z]{2,3}(?:\.[a-z]{2})?$",
    flags=re.IGNORECASE,
)
_SHORT_HOSTS = {"maps.app.goo.gl", "goo.gl"}


def is_allowed_google_maps_url(value: str) -> bool:
    """Return True only for recognised Google Maps/Google short-link hosts.

    Exact hostname matching avoids accepting deceptive hosts such as
    ``evilgoogle.com`` merely because they contain the text ``google.com``.
    """

    url = str(value or "").strip()
    if not url:
        return False

    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        path = (parsed.path or "").lower()

        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.username or parsed.password:
            return False
        try:
            if parsed.port not in {None, 80, 443}:
                return False
        except ValueError:
            return False

        if host in _SHORT_HOSTS:
            return host == "maps.app.goo.gl" or path.startswith("/maps")

        return bool(_GOOGLE_HOST_RE.fullmatch(host) and "/maps" in path)
    except Exception:
        return False


def validate_google_maps_url(value: str) -> str:
    url = str(value or "").strip()
    if not url:
        raise ValueError("Google Maps place URL is required.")
    if not is_allowed_google_maps_url(url):
        raise ValueError(
            "Please provide an exact public Google Maps place URL, for example "
            "https://www.google.com/maps/place/..."
        )
    return url

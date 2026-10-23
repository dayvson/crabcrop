"""Build and sign crabcrop URLs, per ``spec/url-signing.md``
in the crabcrop repository. Standard library only.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional, Sequence, Union

Secret = Union[str, bytes, None]


def _key(secret: Union[str, bytes]) -> bytes:
    return secret.encode("utf-8") if isinstance(secret, str) else secret


def sign(secret: Union[str, bytes], remainder: str) -> str:
    """Compute the signature segment for a remainder: the
    lowercase hex HMAC-SHA256 of ``remainder`` under ``secret``.
    The remainder is everything after the signature segment in
    the URL, exactly as it will appear."""
    return hmac.new(_key(secret), remainder.encode("utf-8"), hashlib.sha256).hexdigest()


def build_url(
    base: str,
    secret: Secret,
    operations: str,
    source: str,
    expires: Optional[int] = None,
) -> str:
    """Assemble a signed (or ``unsafe``) crabcrop URL.
    ``operations`` is the operation string verbatim (several
    ``/``-separated segments and ``preset:`` references
    included). ``expires`` is a unix-seconds expiry, emitted as
    a leading ``exp:`` segment. A ``None`` secret produces the
    literal ``unsafe`` signature segment, for servers running
    with unsafe or presets-only configurations."""
    remainder = ""
    if expires is not None:
        remainder += f"exp:{expires}/"
    if operations:
        remainder += f"{operations}/"
    remainder += source
    signature = "unsafe" if secret is None else sign(secret, remainder)
    return f"{base.rstrip('/')}/{signature}/{remainder}"


def build_srcset(
    base: str,
    secret: Secret,
    source: str,
    widths: Sequence[int],
    expires: Optional[int] = None,
) -> str:
    """Build an ``srcset`` attribute value: one ``{width}x0``
    URL per width, entries formatted ``{url} {width}w`` and
    joined by ``, ``. An expiry, when given, applies to every
    entry."""
    return ", ".join(
        f"{build_url(base, secret, f'{w}x0', source, expires)} {w}w" for w in widths
    )

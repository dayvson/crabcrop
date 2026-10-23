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


def verify(secret: Union[str, bytes], signature: str, remainder: str) -> bool:
    """Check a signature segment against a remainder, with
    the Rust implementation's semantics: the expected tag
    is recomputed and compared in constant time
    (``hmac.compare_digest``), so a non-canonical tag (an
    uppercase one, say) fails like any other mismatch, and
    a malformed one returns False instead of raising."""
    if not isinstance(signature, str) or not signature.isascii():
        return False
    return hmac.compare_digest(sign(secret, remainder), signature)


def _validate_url_text(text: str, what: str) -> None:
    """Fail loudly on inputs that can never verify (audit
    P-4): the remainder travels as a raw URL path, so ``?`` /
    ``#`` would truncate it and whitespace, control, and
    non-ASCII characters cannot be sent as-is."""
    for c in text:
        if c in "?#" or not c.isascii() or ord(c) <= 0x20 or ord(c) == 0x7F:
            raise ValueError(
                f"{what} contains a character that cannot appear in a crabcrop URL: {c!r}"
            )


def _validate_expires(expires: int) -> None:
    """The server accepts only an unsigned 64-bit integer
    (audit P-5)."""
    if (
        isinstance(expires, bool)
        or not isinstance(expires, int)
        or expires < 0
        or expires > 2**64 - 1
    ):
        raise ValueError(
            f"expires must be a non-negative integer of unix seconds, got {expires!r}"
        )


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
        _validate_expires(expires)
        remainder += f"exp:{expires}/"
    _validate_url_text(operations, "operations")
    _validate_url_text(source, "source")
    if operations:
        remainder += f"{operations}/"
    remainder += source
    signature = "unsafe" if secret is None else sign(secret, remainder)
    return f"{base.rstrip('/')}/{signature}/{remainder}"


def _validate_width(width: int) -> None:
    """A srcset width becomes a ``{width}x0`` size segment,
    so it must satisfy the grammar's dimension rules: a
    positive integer within the server cap. Anything else
    mints a URL the server rejects (or worse, ``-300``
    reads as a flip), so it raises here, at build time."""
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or width > 16383
    ):
        raise ValueError(
            f"srcset width must be a positive integer no larger than 16383, got {width!r}"
        )


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
    for w in widths:
        _validate_width(w)
    return ", ".join(
        f"{build_url(base, secret, f'{w}x0', source, expires)} {w}w" for w in widths
    )

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


_KEY_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


def is_valid_key_id(key_id: str) -> bool:
    """Whether ``key_id`` is a well-formed key ID: 1 to 32
    characters of ``[A-Za-z0-9_-]``."""
    return (
        isinstance(key_id, str)
        and 1 <= len(key_id) <= 32
        and all(c in _KEY_ID_CHARS for c in key_id)
    )


def sign_with_key_id(
    secret: Union[str, bytes], key_id: str, remainder: str
) -> str:
    """The signature segment for a keyed signature:
    ``<key-id>:`` followed by the hex tag ``sign`` computes.
    Servers verify a keyed segment against the named key
    only, which is what makes secret rotation possible
    without invalidating every outstanding URL."""
    if not is_valid_key_id(key_id):
        raise ValueError(
            f"invalid key ID {key_id!r}: 1 to 32 characters of [A-Za-z0-9_-]"
        )
    return f"{key_id}:{sign(secret, remainder)}"


def sign(secret: Union[str, bytes], remainder: str) -> str:
    """Compute the signature segment for a remainder: the
    lowercase hex HMAC-SHA256 of ``remainder`` under ``secret``.
    The remainder is everything after the signature segment in
    the URL, exactly as it will appear."""
    return hmac.new(_key(secret), remainder.encode("utf-8"), hashlib.sha256).hexdigest()


def purge_key(secret: Union[str, bytes]) -> bytes:
    """The purge subkey: the raw bytes of the HMAC-SHA256
    of the domain label ``crabcrop/purge-key`` under the
    secret. Purge signatures live under this derived key,
    so an image signature can never be replayed as one."""
    return bytes.fromhex(sign(secret, "crabcrop/purge-key"))


def purge_signature(secret: Union[str, bytes], target: str) -> str:
    """The signature segment for
    ``POST /purge/<signature>/<path>``: HMAC-SHA256 of the
    target path under the purge subkey."""
    key = purge_key(secret)
    return hmac.new(key, target.encode("utf-8"), hashlib.sha256).hexdigest()


def purge_group_key(secret: Union[str, bytes]) -> bytes:
    """The purge-group subkey: the raw bytes of the
    HMAC-SHA256 of the domain label
    ``crabcrop/purge-group-key`` under the secret.
    Purge-group signatures live under this derived key,
    beside the purge key, so a per-target purge signature
    can never be transplanted to the purge-group endpoint."""
    return bytes.fromhex(sign(secret, "crabcrop/purge-group-key"))


def purge_group_signature(secret: Union[str, bytes], group: str) -> str:
    """The signature segment for
    ``POST /purge-group/<signature>/<group>``: HMAC-SHA256
    of the group name under the purge-group subkey. The
    helper exists so nobody has to know the signed string."""
    key = purge_group_key(secret)
    return hmac.new(key, group.encode("utf-8"), hashlib.sha256).hexdigest()


def upload_key(secret: Union[str, bytes]) -> bytes:
    """The upload subkey: the raw bytes of the HMAC-SHA256
    of the domain label ``crabcrop/upload-key`` under the
    secret. Upload signatures live under this derived key,
    so neither an image signature nor a purge signature can
    be replayed as one."""
    return bytes.fromhex(sign(secret, "crabcrop/upload-key"))


def upload_signature(
    secret: Union[str, bytes], name: str, expires: Optional[int] = None
) -> str:
    """The signature segment for
    ``PUT /upload/<signature>/<name>``: HMAC-SHA256 of the
    object name under the upload subkey.

    With ``expires`` (unix seconds), the signature covers
    ``exp:<expires>/<name>`` and the request path is
    ``PUT /upload/<signature>/exp:<expires>/<name>``: the
    expiry sits inside the signed string, so it can
    neither be stripped nor extended, and the server
    refuses the upload once that second is reached.
    Delegated upload tokens should always use the expiring
    form; without it a token replays until the secret
    rotates."""
    key = upload_key(secret)
    signed = name if expires is None else f"exp:{expires}/{name}"
    return hmac.new(key, signed.encode("utf-8"), hashlib.sha256).hexdigest()


def verify(secret: Union[str, bytes], signature: str, remainder: str) -> bool:
    """Check a signature segment against a remainder, with
    the Rust implementation's semantics: the expected tag
    is recomputed and compared in constant time
    (``hmac.compare_digest``), so a non-canonical tag (an
    uppercase one, say) fails like any other mismatch, and
    a malformed one returns False instead of raising."""
    if not isinstance(signature, str) or not signature.isascii():
        return False
    # A keyed segment (`<key-id>:<tag>`) verifies against the
    # one secret given here: the ID is routing metadata, and
    # this function's question is whether this secret made
    # the tag. A malformed ID makes the segment malformed.
    if ":" in signature:
        key_id, _, tag = signature.partition(":")
        if not is_valid_key_id(key_id):
            return False
        signature = tag
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
    key_id: Optional[str] = None,
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
    if secret is None:
        if key_id is not None:
            raise ValueError("key_id requires a secret: an unsafe URL has no key to name")
        signature = "unsafe"
    elif key_id is not None:
        signature = sign_with_key_id(secret, key_id, remainder)
    else:
        signature = sign(secret, remainder)
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

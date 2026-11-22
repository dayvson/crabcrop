"""Build and sign crabcrop URLs, per ``spec/url-signing.md``
in the crabcrop repository. Standard library only, except
sealed URLs (issue #255), which need the ``cryptography``
package from the ``crabcrop-url[seal]`` extra.

The names follow the TypeScript SDK's, in Python's spelling
(``buildURL`` is `build_url`, ``ops.cropObjects`` is
`ops.crop_objects`), the typed operation builders, `ops`,
`filter` and `join_ops`, mirror its builders, and `explain_url`
and `verify_url_signature` are its explainer and URL verifier
(issue #716).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Any, Mapping, Optional, Sequence, Union

from ._explain import (
    FILTER_DESCRIPTIONS,
    ExplainedSegment,
    ExplainedSignature,
    Explanation,
    SegmentKind,
    SignatureKind,
    explain_url,
)
from ._grammar import (
    MAX_DIMENSION,
    RESERVED_SEGMENT,
    Anchor,
    BlendMode,
    CropAlign,
    CurveChannel,
    Dimension,
    ImageFormat,
    MaskShape,
    PerFormatQualityFormat,
    StudioMode,
    StudioShadow,
    is_reserved_remainder,
)
from ._operations import FormatOptions, StudioOptions, filter, join_ops, ops

#: The package version, the release version every crabcrop artifact
#: shares; scripts/bump_version.py moves it with pyproject.toml.
__version__ = "0.15.0"

__all__ = [
    "FILTER_DESCRIPTIONS",
    "MAX_DIMENSION",
    "MAX_SEALED_REMAINDER",
    "RESERVED_SEGMENT",
    "SEALED_PREFIX",
    "Anchor",
    "BlendMode",
    "CropAlign",
    "CurveChannel",
    "Dimension",
    "ExplainedSegment",
    "ExplainedSignature",
    "Explanation",
    "FormatOptions",
    "ImageFormat",
    "Keyring",
    "MaskShape",
    "PerFormatQualityFormat",
    "SealError",
    "Secret",
    "SegmentKind",
    "SignatureKind",
    "StudioMode",
    "StudioOptions",
    "StudioShadow",
    "build_sealed_url",
    "build_srcset",
    "build_url",
    "explain_url",
    "filter",
    "is_reserved_remainder",
    "is_valid_key_id",
    "join_ops",
    "ops",
    "purge_group_key",
    "purge_group_signature",
    "purge_key",
    "purge_signature",
    "seal",
    "seal_key",
    "seal_url",
    "seal_with_nonce",
    "sign",
    "sign_with_key_id",
    "unseal",
    "upload_key",
    "upload_signature",
    "verify",
    "verify_url_signature",
]

Secret = Union[str, bytes, None]

#: Secrets by key ID, as a server in a rotation window holds them:
#: what `verify_url_signature` takes in place of one secret.
Keyring = Mapping[str, Union[str, bytes]]


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


_RESERVED_MESSAGE = (
    f'remainder begins with the reserved segment "{RESERVED_SEGMENT}": '
    "it is kept for key derivation, and no image URL may begin with it"
)


def _refuse_reserved(remainder: str) -> None:
    if is_reserved_remainder(remainder):
        raise ValueError(_RESERVED_MESSAGE)


def _subkey(secret: Union[str, bytes], label: str) -> bytes:
    """A subkey: the raw HMAC-SHA256 of ``label`` under the
    secret. Every label begins with `RESERVED_SEGMENT`, which
    is why `sign` refuses to compute the same value."""
    return hmac.new(_key(secret), label.encode("utf-8"), hashlib.sha256).digest()


def sign(secret: Union[str, bytes], remainder: str) -> str:
    """Compute the signature segment for a remainder: the
    lowercase hex HMAC-SHA256 of ``remainder`` under ``secret``.
    The remainder is everything after the signature segment in
    the URL, exactly as it will appear. A remainder that begins
    with `RESERVED_SEGMENT` raises ``ValueError``: its
    signature could be a derived subkey."""
    _refuse_reserved(remainder)
    return hmac.new(_key(secret), remainder.encode("utf-8"), hashlib.sha256).hexdigest()


def purge_key(secret: Union[str, bytes]) -> bytes:
    """The purge subkey: the raw bytes of the HMAC-SHA256
    of the domain label ``crabcrop/purge-key`` under the
    secret. Purge signatures live under this derived key,
    so an image signature never verifies as one."""
    return _subkey(secret, "crabcrop/purge-key")


def _purge_signed_string(target: str, expires: Optional[int]) -> str:
    """The signed string of a purge or purge-group signature:
    the target, or ``exp:<expires>/`` and the target. It is
    also what follows the signature segment in the request
    path."""
    if expires is None:
        return target
    _validate_expires(expires)
    return f"exp:{expires}/{target}"


def purge_signature(
    secret: Union[str, bytes], target: str, expires: Optional[int] = None
) -> str:
    """The signature segment for
    ``POST /purge/<signature>/<path>``: HMAC-SHA256 of the
    target path under the purge subkey.

    With ``expires`` (unix seconds), the signature covers
    ``exp:<expires>/<path>`` and the request path is
    ``POST /purge/<signature>/exp:<expires>/<path>``: the
    expiry sits inside the signed string, so it can neither
    be stripped nor extended, and the server refuses the
    purge once that second is reached. Without it the
    signature authorizes the purge until the secret rotates.
    An expiry that is not a non-negative integer raises
    ``ValueError``.

    The server reads a leading ``exp:`` segment of the path
    as the purge's own expiry, so a target that opens with an
    image URL's ``exp:`` is refused from that second on. An
    image's ``exp:`` is not part of its cache key:
    drop it from the target and the purge evicts the same
    entries."""
    key = purge_key(secret)
    signed = _purge_signed_string(target, expires)
    return hmac.new(key, signed.encode("utf-8"), hashlib.sha256).hexdigest()


def purge_group_key(secret: Union[str, bytes]) -> bytes:
    """The purge-group subkey: the raw bytes of the
    HMAC-SHA256 of the domain label
    ``crabcrop/purge-group-key`` under the secret.
    Purge-group signatures live under this derived key,
    beside the purge key, so a per-target purge signature
    can never be transplanted to the purge-group endpoint."""
    return _subkey(secret, "crabcrop/purge-group-key")


def purge_group_signature(
    secret: Union[str, bytes], group: str, expires: Optional[int] = None
) -> str:
    """The signature segment for
    ``POST /purge-group/<signature>/<group>``: HMAC-SHA256
    of the group name under the purge-group subkey. The
    helper exists so nobody has to know the signed string.

    With ``expires`` (unix seconds), the signature covers
    ``exp:<expires>/<group>``, the request path is
    ``POST /purge-group/<signature>/exp:<expires>/<group>``,
    and the server refuses the purge once that second is
    reached, as for `purge_signature`."""
    key = purge_group_key(secret)
    signed = _purge_signed_string(group, expires)
    return hmac.new(key, signed.encode("utf-8"), hashlib.sha256).hexdigest()


def upload_key(secret: Union[str, bytes]) -> bytes:
    """The upload subkey: the raw bytes of the HMAC-SHA256
    of the domain label ``crabcrop/upload-key`` under the
    secret. Upload signatures live under this derived key,
    so neither an image signature nor a purge signature
    verifies as one: a purge signature is under another
    subkey, and no signer signs a label (see
    `RESERVED_SEGMENT`)."""
    return _subkey(secret, "crabcrop/upload-key")


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
    # No signature over a reserved remainder is ever valid: the
    # server refuses it whatever the signature, and this check
    # must not confirm a derived subkey either.
    if is_reserved_remainder(remainder):
        return False
    return hmac.compare_digest(sign(secret, remainder), signature)


def _utf8_as_javascript_writes(text: str) -> bytes:
    """The UTF-8 bytes TextEncoder and ``node:crypto`` write for the
    same text in JavaScript: a lone surrogate, which UTF-8 cannot
    encode, becomes U+FFFD, and a surrogate pair the character it
    encodes."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        units = text.encode("utf-16-le", "surrogatepass")
        return units.decode("utf-16-le", "replace").encode("utf-8")


def verify_url_signature(
    secret_or_keyring: Union[str, bytes, Keyring], input: str
) -> Optional[bool]:
    """Verify a URL's signature locally, as the TypeScript SDK's
    ``verifyUrlSignature`` does: ``True`` or ``False``, or ``None``
    when the URL is unsigned (``unsafe``), sealed, malformed, or
    longer than the 64 KiB `explain_url` reads. The URL is read the
    way `explain_url` reads it, so a query or fragment is not part of
    the signed path.

    With one secret (``str`` or ``bytes``), the key ID a URL carries
    is ignored: the question is whether this secret made the tag.
    With a keyring (a mapping of key ID to secret), the server's
    rotation rules apply: a keyed URL consults its named key only (an
    unknown ID is ``False``), and an unkeyed URL consults every key.
    A remainder that begins with `RESERVED_SEGMENT` is ``False``
    whatever its tag. Each comparison is constant-time
    (``hmac.compare_digest``), as in `verify`.

    A lone surrogate in the URL or in a ``str`` secret, which UTF-8
    cannot encode, is hashed as U+FFFD, the bytes the TypeScript
    SDK's HMAC hashes for it, so no URL and no secret of text makes
    this raise."""
    explained = explain_url(input)
    signature = explained.signature
    if signature is None or signature.kind != "hmac":
        return None
    remainder = explained.remainder
    # No signature over a reserved remainder is ever valid: the
    # server refuses it whatever the signature, and this check
    # must not confirm a derived subkey either.
    if is_reserved_remainder(remainder):
        return False
    tag = signature.value
    message = _utf8_as_javascript_writes(remainder)

    def made_by(secret: Union[str, bytes]) -> bool:
        key = _utf8_as_javascript_writes(secret) if isinstance(secret, str) else secret
        return hmac.compare_digest(hmac.new(key, message, hashlib.sha256).hexdigest(), tag)

    if not isinstance(secret_or_keyring, Mapping):
        return made_by(secret_or_keyring)
    if signature.key_id is not None:
        secret = secret_or_keyring.get(signature.key_id)
        return secret is not None and made_by(secret)
    return any(made_by(secret) for secret in secret_or_keyring.values())


# ---------------------------------------------------------------------------
# Sealed URLs (issue #255)
#
# The one part of this package that is not standard-library
# only: AES-256-GCM lives behind the `cryptography`
# optional extra (`pip install crabcrop-url[seal]`), imported
# lazily so the signing surface stays zero-dependency.
# ---------------------------------------------------------------------------

#: The version marker that opens a sealed token segment.
SEALED_PREFIX = "s1."

#: The AAD every sealed token is bound to.
_SEAL_AAD = b"crabcrop/sealed/1"

#: The largest remainder `seal_url` will encrypt, in bytes.
MAX_SEALED_REMAINDER = 2048


class SealError(ValueError):
    """Sealing or opening a sealed token failed. Every open
    failure is this error with the one message, on purpose:
    openers must never learn which stage failed."""


def seal_key(secret: Union[str, bytes]) -> bytes:
    """The seal subkey: the raw bytes of the HMAC-SHA256 of
    the label ``crabcrop/seal-key`` under the secret, its
    own domain beside purge, purge-group, and upload, so
    there is no second secret to configure or rotate: each
    keyring entry yields its own seal subkey, and the
    rotation window for signatures is the rotation window
    for tokens."""
    return _subkey(secret, "crabcrop/seal-key")


def _aesgcm() -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise SealError(
            "sealing requires the cryptography package: "
            "install it with `pip install crabcrop-url[seal]`"
        ) from None
    return AESGCM


def seal_with_nonce(
    secret: Union[str, bytes],
    remainder: str,
    nonce: bytes,
    key_id: Optional[str] = None,
) -> str:
    """Seal a remainder into a token segment with a
    caller-supplied nonce: ``s1.<base64url(nonce ||
    ciphertext || tag)>``, AES-256-GCM under the derived
    seal subkey with the fixed associated data
    ``crabcrop/sealed/1``.

    DANGER: a nonce must never repeat under one secret.
    Reusing a nonce with AES-GCM destroys the
    construction's confidentiality and integrity guarantees.
    This function is the deterministic seam the spec's test
    vectors use; production minting calls `seal_url`, which
    draws a fresh random nonce every time."""
    if len(nonce) != 12:
        raise SealError(f"sealed token nonce must be 12 bytes, got {len(nonce)}")
    if len(remainder.encode("utf-8")) > MAX_SEALED_REMAINDER:
        raise SealError(
            f"remainder is over the {MAX_SEALED_REMAINDER}-byte sealing cap: "
            "a recipe that long belongs in a preset"
        )
    sealed = _aesgcm()(seal_key(secret)).encrypt(
        bytes(nonce), remainder.encode("utf-8"), _SEAL_AAD
    )
    body = base64.urlsafe_b64encode(bytes(nonce) + sealed).rstrip(b"=").decode("ascii")
    token = f"{SEALED_PREFIX}{body}"
    if key_id is None:
        return token
    if not is_valid_key_id(key_id):
        raise ValueError(
            f"invalid key ID {key_id!r}: 1 to 32 characters of [A-Za-z0-9_-]"
        )
    return f"{key_id}:{token}"


def seal(
    secret: Union[str, bytes],
    remainder: str,
    key_id: Optional[str] = None,
) -> str:
    """Seal a remainder into a token segment (issue #255),
    with a fresh random 96-bit nonce. Pass ``key_id`` to
    stamp ``<key-id>:`` in front of it, so servers in a
    rotation window open it against the named key only.
    The TypeScript SDK and the Rust crate name it ``seal``
    too."""
    return seal_with_nonce(secret, remainder, os.urandom(12), key_id)


#: The earlier name of `seal`, kept so existing code keeps working.
seal_url = seal


def unseal(secret: Union[str, bytes], segment: str) -> str:
    """Open a token segment, with or without a ``<key-id>:``
    prefix, under one secret, returning the remainder it
    seals. The key ID is routing metadata and is not
    consulted: the question is whether this secret opens
    the token. Every failure (malformed body, wrong secret,
    tampered bytes, unknown version) raises `SealError`
    with the one message."""

    def failed() -> SealError:
        return SealError("the token did not open under this secret")

    token = segment
    if ":" in segment:
        key_id, _, token = segment.partition(":")
        if not is_valid_key_id(key_id):
            raise failed()
    if not token.startswith(SEALED_PREFIX):
        raise failed()
    body = token[len(SEALED_PREFIX) :]
    # The standard decoder tolerates padding and stray
    # characters; canonicity is checked by re-encoding, so
    # anything but the exact minted spelling fails here.
    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    except (ValueError, TypeError):
        raise failed() from None
    if (
        len(raw) < 12 + 16
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != body
    ):
        raise failed()
    try:
        plain: bytes = _aesgcm()(seal_key(secret)).decrypt(raw[:12], raw[12:], _SEAL_AAD)
    except SealError:
        raise
    except Exception:
        raise failed() from None
    try:
        return plain.decode("utf-8")
    except UnicodeDecodeError:
        raise failed() from None


def build_sealed_url(
    base: str,
    secret: Optional[Union[str, bytes]],
    operations: str,
    source: str,
    expires: Optional[int] = None,
    key_id: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Assemble a sealed crabcrop URL:
    ``{base}/[<key-id>:]s1.<token>/{friendly-name}`` (issue
    #255). The remainder is assembled exactly as `build_url`
    assembles it (expiry segment, operations, source) and
    then sealed whole, pre-expansion, so the sealed and
    plain forms of one render are one cache entry and one
    purge target on the server. ``name`` is the cosmetic
    friendly name: the server never parses, validates, or
    authenticates it; omit it and the URL ends at the
    token. A ``None`` secret raises: there is no ``unsafe``
    spelling of a token."""
    if expires is not None:
        _validate_expires(expires)
    _validate_url_text(operations, "operations")
    _validate_url_text(source, "source")
    remainder = ""
    if expires is not None:
        remainder += f"exp:{expires}/"
    if operations:
        remainder += f"{operations}/"
    remainder += source
    # The server refuses a token whose remainder begins with
    # the reserved segment, so the builder refuses it, as it
    # refuses a character that cannot travel.
    if is_reserved_remainder(remainder):
        raise SealError(_RESERVED_MESSAGE)
    if secret is None:
        raise SealError("sealing requires a secret: there is no unsafe token")
    segment = seal(secret, remainder, key_id)
    trimmed = base.rstrip("/")
    if name:
        return f"{trimmed}/{segment}/{name}"
    return f"{trimmed}/{segment}"


# Characters a browser rewrites in a URL path (issue #472): an
# ``<img src>`` or ``fetch()`` URL is parsed by WHATWG rules, which
# turn ``\`` into ``/`` and percent-encode the rest (Chrome encodes
# ``^`` and ``|`` too). The server verifies the raw path, so a URL
# signed over one of them answers 403 from a page.
_BROWSER_REWRITTEN = frozenset('"<>\\^`{|}')


def _validate_url_text(text: str, what: str) -> None:
    """Fail loudly on inputs that can never verify (audit
    P-4): the remainder travels as a raw URL path, so ``?`` /
    ``#`` would truncate it, whitespace, control, and
    non-ASCII characters cannot be sent as-is, and a browser
    rewrites the ``_BROWSER_REWRITTEN`` set."""
    for c in text:
        rewritten = c in _BROWSER_REWRITTEN
        if c in "?#" or not c.isascii() or ord(c) <= 0x20 or ord(c) == 0x7F or rewritten:
            hint = (
                f" (browsers rewrite it; percent-encode it as %{ord(c):02X})" if rewritten else ""
            )
            raise ValueError(
                f"{what} contains a character that cannot appear in a crabcrop URL: {c!r}{hint}"
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
    # Refused with or without a secret: the server refuses the
    # URL whatever its signature.
    _refuse_reserved(remainder)
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
    positive integer within the server cap, `MAX_DIMENSION`.
    Anything else mints a URL the server rejects (or worse,
    ``-300`` reads as a flip), so it raises here, at build
    time."""
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or width > MAX_DIMENSION
    ):
        raise ValueError(
            f"srcset width must be a positive integer no larger than {MAX_DIMENSION}, got {width!r}"
        )


def build_srcset(
    base: str,
    secret: Secret,
    source: str,
    widths: Sequence[int],
    expires: Optional[int] = None,
    key_id: Optional[str] = None,
) -> str:
    """Build an ``srcset`` attribute value: one ``{width}x0``
    URL per width, entries formatted ``{url} {width}w`` and
    joined by ``, ``. An expiry and a key ID, when given,
    apply to every entry, as `build_url` applies them."""
    for w in widths:
        _validate_width(w)
    return ", ".join(
        f"{build_url(base, secret, f'{w}x0', source, expires, key_id)} {w}w" for w in widths
    )

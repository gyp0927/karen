"""PKCE utilities, mirroring auth/oauth/pkce.ts.

pi-ai's generatePKCE is async only because Web Crypto's digest is; Python's
hashlib is synchronous, so `generate_pkce` is a plain function.
"""

from __future__ import annotations

import base64
import hashlib
import secrets


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate a PKCE (verifier, challenge) pair.

    The verifier is 32 random bytes, base64url-encoded; the challenge is the
    base64url-encoded SHA-256 of the verifier (S256).
    """
    verifier = _base64url_encode(secrets.token_bytes(32))
    challenge = _base64url_encode(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge

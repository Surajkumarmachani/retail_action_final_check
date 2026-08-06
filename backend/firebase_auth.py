# firebase_auth.py
# Verifies Firebase ID tokens without firebase-admin SDK.
# Uses PyJWT + Firebase's public JWK endpoint — zero protobuf dependency.

import json
import logging
import os
import time
from typing import Optional

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

logger = logging.getLogger(__name__)

FIREBASE_PROJECT_ID: Optional[str] = None

_GOOGLE_JWKS_URL = (
    "https://www.googleapis.com/service_accounts/v1/jwk/"
    "securetoken@system.gserviceaccount.com"
)
_jwks_cache: dict = {}
_jwks_expires_at: float = 0.0


def init_firebase() -> None:
    global FIREBASE_PROJECT_ID
    # Prefer FIREBASE_PROJECT_ID env var (deploy-friendly — only the non-secret project id
    # is needed for ID-token verification). Fall back to serviceAccountKey.json for local dev.
    FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
    if not FIREBASE_PROJECT_ID:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_JSON", "./serviceAccountKey.json")
        with open(cred_path) as f:
            creds = json.load(f)
        FIREBASE_PROJECT_ID = creds["project_id"]
    logger.info("Firebase project: %s  (JWT-only verification, no Admin SDK)", FIREBASE_PROJECT_ID)


async def _get_signing_key(kid: str):
    """Fetch Google's JWK set, cache for 1 h, return the RSA public key for kid."""
    global _jwks_cache, _jwks_expires_at

    if time.monotonic() >= _jwks_expires_at:
        async with httpx.AsyncClient() as client:
            resp = await client.get(_GOOGLE_JWKS_URL, timeout=10.0)
            resp.raise_for_status()
            jwks = resp.json()
        _jwks_cache = {k["kid"]: k for k in jwks["keys"]}
        _jwks_expires_at = time.monotonic() + 3600

    if kid not in _jwks_cache:
        # Keys rotate; force a refresh once
        async with httpx.AsyncClient() as client:
            resp = await client.get(_GOOGLE_JWKS_URL, timeout=10.0)
            resp.raise_for_status()
            jwks = resp.json()
        _jwks_cache = {k["kid"]: k for k in jwks["keys"]}
        _jwks_expires_at = time.monotonic() + 3600
        if kid not in _jwks_cache:
            raise ValueError(f"Unknown Firebase key ID: {kid!r}")

    return RSAAlgorithm.from_jwk(json.dumps(_jwks_cache[kid]))


async def verify_id_token(token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded claims.
    Raises jwt.PyJWTError (or subclass) on any failure.
    The returned dict always contains 'uid' (= 'sub' claim).
    """
    if FIREBASE_PROJECT_ID is None:
        raise RuntimeError("Firebase not initialised — call init_firebase() first")

    header = jwt.get_unverified_header(token)
    kid = header.get("kid", "")
    if not kid:
        raise jwt.InvalidTokenError("Missing 'kid' in token header")

    public_key = await _get_signing_key(kid)

    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        audience=FIREBASE_PROJECT_ID,
        issuer=f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}",
        leeway=60,
    )
    # Firebase tokens use 'sub' for the user ID; alias as 'uid'
    payload.setdefault("uid", payload.get("sub", ""))
    return payload

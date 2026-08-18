from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
from typing import Any

from .version import VERSION


def _safe(x: Any):
    if is_dataclass(x): return _safe(asdict(x))
    if isinstance(x, dict): return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_safe(v) for v in x]
    if hasattr(x, "value") and isinstance(getattr(x, "value"), str): return x.value
    return x


# ---------------------------------------------------------------------------
# v5.3.2: Ed25519 signing support.
#
# The audit found that the receipt chain is tamper-evident but not
# identity-authenticated.  Any party with write access to the ledger
# can forge a valid hash chain.  Ed25519 signing binds each receipt
# to a cryptographic identity (the authority's signing key).
#
# Signing is optional: if no signing key is provided, receipts remain
# tamper-evident only (backward compatible).  If a signing key is
# provided, each receipt includes an Ed25519 signature over its
# canonical hash, and verification checks both the hash chain and
# the signature.
# ---------------------------------------------------------------------------

def _try_import_ed25519():
    """Try to import an Ed25519 implementation.

    Returns (sign, verify) functions, or (None, None) if unavailable.
    Uses cryptography (preferred) or PyNaCl if installed.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives import serialization

        def generate_keypair():
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode()
            return private_pem, public_pem

        def sign(private_pem: str, message: bytes) -> bytes:
            key = serialization.load_pem_private_key(private_pem.encode(), password=None)
            return key.sign(message)

        def verify(public_pem: str, message: bytes, signature: bytes) -> bool:
            try:
                key = serialization.load_pem_public_key(public_pem.encode())
                key.verify(signature, message)
                return True
            except Exception:
                return False

        return generate_keypair, sign, verify
    except ImportError:
        pass

    try:
        import nacl.signing
        import nacl.encoding

        def generate_keypair():
            sk = nacl.signing.SigningKey.generate()
            private_key = sk.encode(encoder=nacl.encoding.HexEncoder).decode()
            public_key = sk.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode()
            return private_key, public_key

        def sign(private_key_hex: str, message: bytes) -> bytes:
            sk = nacl.signing.SigningKey(private_key_hex.encode(), encoder=nacl.encoding.HexEncoder)
            return sk.sign(message).signature

        def verify(public_key_hex: str, message: bytes, signature: bytes) -> bool:
            try:
                vk = nacl.signing.VerifyKey(public_key_hex.encode(), encoder=nacl.encoding.HexEncoder)
                vk.verify(message, signature)
                return True
            except Exception:
                return False

        return generate_keypair, sign, verify
    except ImportError:
        pass

    return None, None, None


# Lazily loaded Ed25519 backend
_ed25519_backend = None

def _get_ed25519_backend():
    global _ed25519_backend
    if _ed25519_backend is None:
        _ed25519_backend = _try_import_ed25519()
    return _ed25519_backend


def ed25519_available() -> bool:
    """Check if Ed25519 signing is available (requires cryptography or PyNaCl)."""
    gen, _, _ = _get_ed25519_backend()
    return gen is not None


def generate_keypair() -> tuple[str, str]:
    """Generate an Ed25519 keypair. Returns (private_key_pem, public_key_pem)."""
    gen, _, _ = _get_ed25519_backend()
    if gen is None:
        raise ImportError("Ed25519 requires 'cryptography' or 'PyNaCl': pip install cryptography")
    return gen()


def sign_receipt(private_key: str, receipt_hash: str) -> str:
    """Sign a receipt hash with an Ed25519 private key. Returns hex signature."""
    _, sign_fn, _ = _get_ed25519_backend()
    if sign_fn is None:
        raise ImportError("Ed25519 requires 'cryptography' or 'PyNaCl': pip install cryptography")
    signature = sign_fn(private_key, receipt_hash.encode())
    return signature.hex()


def verify_receipt_signature(public_key: str, receipt_hash: str, signature_hex: str) -> bool:
    """Verify an Ed25519 signature over a receipt hash."""
    _, _, verify_fn = _get_ed25519_backend()
    if verify_fn is None:
        raise ImportError("Ed25519 requires 'cryptography' or 'PyNaCl': pip install cryptography")
    try:
        return verify_fn(public_key, receipt_hash.encode(), bytes.fromhex(signature_hex))
    except (ValueError, TypeError):
        return False


def mutation_receipt(
    result,
    *,
    build_version: str = VERSION,
    receipt_index: int = 0,
    previous_receipt_hash: str | None = None,
    authority_state_hash_before: str | None = None,
    authority_state_hash_after: str | None = None,
    gauge_authority_hash: str | None = None,
    signing_key: str | None = None,
) -> dict:
    """Create a hash-chained mutation receipt.

    The receipt binds the full authority identity: graph state, gauge
    connections, fiber state, and governance config. Each receipt links to
    the previous receipt via ``previous_receipt_hash``, forming a tamper-evident
    hash chain H_i = SHA256(H_{i-1} || R_i).

    v5.3.2: If ``signing_key`` (Ed25519 private key PEM/hex) is provided,
    the receipt is also signed, providing identity authentication in
    addition to tamper evidence.
    """
    payload = {
        "schema": "LGAE_MUTATION_RECEIPT_V4",
        "build_version": build_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "receipt_index": int(receipt_index),
        "previous_receipt_hash": previous_receipt_hash,
        "authority_state_hash_before": authority_state_hash_before,
        "authority_state_hash_after": authority_state_hash_after,
        "gauge_authority_hash": gauge_authority_hash,
        "result": _safe(result),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    if signing_key is not None:
        payload["ed25519_signature"] = sign_receipt(signing_key, payload["sha256"])
    return payload


def append_receipt(path: str | Path, receipt: dict, *, signing_key: str | None = None) -> None:
    """Append a receipt to the JSONL ledger, maintaining the hash chain.

    v5.3.2: If ``signing_key`` is provided, signs the receipt after
    updating chain fields.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # If the receipt doesn't already have chain fields populated, read the
    # last receipt's hash from the file and populate them.
    if receipt.get("previous_receipt_hash") is None and receipt.get("receipt_index", 0) == 0:
        last_hash, last_index = _read_last_receipt_hash(p)
        # If no prior receipts exist, this is the genesis receipt (index 0).
        # Otherwise, chain to the last receipt.
        if last_hash is None:
            receipt["receipt_index"] = 0
            receipt["previous_receipt_hash"] = None
        else:
            receipt["receipt_index"] = last_index + 1
            receipt["previous_receipt_hash"] = last_hash
        # Recompute sha256 with updated chain fields
        sig_fields = {"sha256", "ed25519_signature"}
        payload_for_hash = {k: v for k, v in receipt.items() if k not in sig_fields}
        canonical = json.dumps(payload_for_hash, sort_keys=True, separators=(",", ":"), default=str).encode()
        receipt["sha256"] = hashlib.sha256(canonical).hexdigest()
        if signing_key is not None:
            receipt["ed25519_signature"] = sign_receipt(signing_key, receipt["sha256"])
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt, sort_keys=True, default=str) + "\n")


def _read_last_receipt_hash(path: Path) -> tuple[str | None, int]:
    """Read the hash and index of the last receipt in the ledger."""
    if not path.exists():
        return None, 0
    last_hash: str | None = None
    last_index: int = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                last_hash = r.get("sha256")
                last_index = int(r.get("receipt_index", 0))
            except json.JSONDecodeError:
                continue
    return last_hash, last_index


def verify_receipt_chain(
    path: str | Path,
    *,
    public_key: str | None = None,
) -> tuple[bool, list[str]]:
    """Verify the integrity of a receipt ledger hash chain.

    Returns (is_valid, errors). Each error describes a broken chain link.

    v5.3.2: If ``public_key`` (Ed25519 public key PEM/hex) is provided,
    also verifies Ed25519 signatures on receipts that include them.
    """
    p = Path(path)
    if not p.exists():
        return True, []
    errors: list[str] = []
    expected_prev: str | None = None
    expected_index: int = 0
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            # Verify chain linkage
            actual_prev = r.get("previous_receipt_hash")
            actual_index = int(r.get("receipt_index", 0))
            if actual_index != expected_index:
                errors.append(f"line {line_no}: receipt_index {actual_index} != expected {expected_index}")
            if actual_prev != expected_prev:
                errors.append(f"line {line_no}: previous_receipt_hash mismatch")
            # Verify self-hash
            stored_hash = r.get("sha256")
            sig_fields = {"sha256", "ed25519_signature"}
            payload_for_hash = {k: v for k, v in r.items() if k not in sig_fields}
            canonical = json.dumps(payload_for_hash, sort_keys=True, separators=(",", ":"), default=str).encode()
            computed = hashlib.sha256(canonical).hexdigest()
            if stored_hash != computed:
                errors.append(f"line {line_no}: sha256 mismatch (stored={stored_hash}, computed={computed})")
            # v5.3.2: Verify Ed25519 signature if present
            sig = r.get("ed25519_signature")
            if sig is not None:
                if public_key is None:
                    errors.append(f"line {line_no}: receipt has ed25519_signature but no public_key provided")
                elif not verify_receipt_signature(public_key, stored_hash, sig):
                    errors.append(f"line {line_no}: ed25519 signature verification failed")
            expected_prev = stored_hash
            expected_index = actual_index + 1
    return (len(errors) == 0), errors

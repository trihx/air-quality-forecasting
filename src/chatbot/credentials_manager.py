"""
Secure credentials persistence manager for AI Assistant.

Provides authenticated symmetric encryption (Fernet / AES-128-CBC + HMAC-SHA256)
to securely store LLM API keys and remote tunnel endpoints locally on disk.
Enforces strict 0600 file permissions and zero-plaintext storage.
"""

import base64
import hashlib
import json
import logging
import os
import platform
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Storage files (protected by 0600 permissions and gitignored)
CREDENTIALS_FILE = PROJECT_ROOT / ".chatbot_credentials.enc"
SALT_FILE = PROJECT_ROOT / ".chatbot_salt"

# PBKDF2 configuration
PBKDF2_ITERATIONS = 100_000
SALT_SIZE_BYTES = 16
DEVICE_SECRET_BYTES = 32


def _secure_write_bytes(file_path: Path, data: bytes) -> None:
    """Write binary data to file with strict 0600 (owner read/write only) permissions."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(file_path, flags, 0o600)
    try:
        with open(fd, "wb") as f:
            f.write(data)
    finally:
        pass  # fd is closed by context manager

    if os.name == "posix":
        try:
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as err:
            logger.debug(f"Could not set file permissions on {file_path}: {err}")


def _read_or_create_crypto_material() -> tuple[bytes, bytes]:
    """Read existing cryptographic material (16-byte salt + 32-byte high-entropy device secret) or generate new.

    Returns:
        tuple of (salt, device_secret). Both are protected by 0600 POSIX permissions.
    """
    total_bytes = SALT_SIZE_BYTES + DEVICE_SECRET_BYTES
    if SALT_FILE.is_file():
        try:
            data = SALT_FILE.read_bytes()
            if len(data) >= total_bytes:
                return data[:SALT_SIZE_BYTES], data[SALT_SIZE_BYTES:total_bytes]
            if len(data) >= SALT_SIZE_BYTES:
                # Upgrade legacy salt file with new random device secret
                salt = data[:SALT_SIZE_BYTES]
                device_secret = os.urandom(DEVICE_SECRET_BYTES)
                _secure_write_bytes(SALT_FILE, salt + device_secret)
                return salt, device_secret
        except Exception as e:
            logger.warning(f"Failed reading salt file ({e}), regenerating new crypto material.")

    salt = os.urandom(SALT_SIZE_BYTES)
    device_secret = os.urandom(DEVICE_SECRET_BYTES)
    _secure_write_bytes(SALT_FILE, salt + device_secret)
    return salt, device_secret


def _get_or_create_cipher() -> Fernet:
    """
    Derive a deterministic Fernet key using PBKDF2 HMAC-SHA256 (100k rounds).

    Security Model:
    1. If CREDENTIALS_ENCRYPTION_KEY is provided in environment (recommended for Production),
       it is used as master passphrase mixed with local salt.
    2. Otherwise, key is derived by mixing a 256-bit random local device secret (0600 file)
       with OS identity and local salt. This prevents offline dictionary attacks even if
       username/hostname are known.
    """
    custom_key = os.getenv("CREDENTIALS_ENCRYPTION_KEY", "").strip()
    salt, device_secret = _read_or_create_crypto_material()

    if custom_key:
        try:
            # If already a valid 32-byte urlsafe base64 Fernet key
            decoded = base64.urlsafe_b64decode(custom_key.encode("utf-8"))
            if len(decoded) == 32:
                return Fernet(custom_key.encode("utf-8"))
        except Exception as err:
            logger.debug("Provided key is not raw base64 Fernet key: %s", err)

        # Otherwise derive key using custom_key as passphrase with local salt
        derived_key = hashlib.pbkdf2_hmac(
            "sha256",
            custom_key.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS,
        )
        return Fernet(base64.urlsafe_b64encode(derived_key))

    # Default: Machine-bound + 256-bit random device secret key derivation
    user = os.getenv("USER") or os.getenv("USERNAME") or "pm25_local_user"
    node = platform.node() or "pm25_local_host"
    seed_material = f"{user}@{node}:pm25-v1:".encode() + device_secret

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        seed_material,
        salt,
        PBKDF2_ITERATIONS,
    )
    return Fernet(base64.urlsafe_b64encode(derived_key))


def save_credentials(
    providers_dict: dict[str, Any],
    primary_provider: str | None = None,
    remember: bool = True,
) -> bool:
    """
    Encrypt and save provider credentials to disk.

    Args:
        providers_dict: Dictionary containing provider configs (e.g. api_key, model, base_url).
        primary_provider: Identifier of the primary active provider.
        remember: If False, automatically purges any persisted credentials file and returns True.

    Returns:
        True if operation succeeded, False otherwise.
    """
    if not remember:
        return clear_credentials()

    try:
        clean_providers: dict[str, dict[str, str]] = {}
        for name, p_data in providers_dict.items():
            if name.startswith("_"):
                continue
            if isinstance(p_data, dict):
                api_key = str(p_data.get("api_key") or "").strip()
                base_url = str(p_data.get("base_url") or "").strip()
                model = str(p_data.get("model") or "").strip()
                if api_key or base_url:
                    clean_providers[name] = {
                        "api_key": api_key,
                        "model": model,
                        "base_url": base_url,
                    }

        payload = {
            "version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "remember_me": True,
            "providers": clean_providers,
            "primary_provider": primary_provider,
        }

        raw_json = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cipher = _get_or_create_cipher()
        encrypted_bytes = cipher.encrypt(raw_json)

        _secure_write_bytes(CREDENTIALS_FILE, encrypted_bytes)
        logger.info(f"Successfully encrypted and saved credentials for {len(clean_providers)} providers.")
        return True
    except Exception as e:
        logger.error(f"Failed to save encrypted credentials: {e}", exc_info=True)
        return False


def load_credentials() -> tuple[dict[str, Any], str | None]:
    """
    Load and decrypt stored provider credentials from disk.

    Returns:
        tuple of (providers_dict, primary_provider). Returns ({}, None) if file
        does not exist, is corrupted, or decryption fails.
    """
    if not CREDENTIALS_FILE.is_file():
        return {}, None

    try:
        encrypted_bytes = CREDENTIALS_FILE.read_bytes()
        cipher = _get_or_create_cipher()
        decrypted_json = cipher.decrypt(encrypted_bytes).decode("utf-8")
        payload = json.loads(decrypted_json)

        providers = payload.get("providers", {})
        primary_provider = payload.get("primary_provider")
        logger.info(f"Loaded {len(providers)} saved AI provider credentials.")
        return providers, primary_provider
    except InvalidToken:
        logger.warning("Failed to decrypt credentials: authentication token invalid or tampered.")
        return {}, None
    except Exception as e:
        logger.warning(f"Could not load stored credentials gracefully: {e}")
        return {}, None


def clear_credentials() -> bool:
    """
    Securely delete stored credentials file.

    Returns:
        True if file was deleted or did not exist, False on failure.
    """
    try:
        if CREDENTIALS_FILE.is_file():
            CREDENTIALS_FILE.unlink(missing_ok=True)
            logger.info("Cleared stored credentials file.")
        return True
    except OSError as e:
        logger.error(f"Failed to delete credentials file: {e}")
        return False


def is_credentials_persisted() -> bool:
    """Check whether stored credentials currently exist on disk."""
    return CREDENTIALS_FILE.is_file()

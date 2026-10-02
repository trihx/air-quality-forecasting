"""
Secure credentials persistence manager for AI Assistant.

Provides authenticated symmetric encryption (Fernet / AES-128-CBC + HMAC-SHA256)
to securely store LLM API keys and remote tunnel endpoints locally on disk.
Enforces strict 0600 file permissions and zero-plaintext storage.
"""

import base64
import contextlib
import hashlib
import json
import logging
import os
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

# Master key and database identifiers for dual-tier persistence
SYSTEM_CARD_KEY = "_system_ai_credentials_enc"
DEFAULT_SALT = b"pm25_ctu_salt_16"  # 16-byte fixed salt ensuring deterministic key derivation across container restarts
DEFAULT_MASTER_SEED = "pm25-ctu-thesis-secure-storage-master-v1"


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


def _get_salt() -> bytes:
    """Read salt from SALT_FILE or return deterministic fallback salt."""
    if SALT_FILE.is_file():
        try:
            data = SALT_FILE.read_bytes()
            if len(data) >= SALT_SIZE_BYTES:
                return data[:SALT_SIZE_BYTES]
        except Exception as e:
            logger.debug(f"Failed reading salt file ({e}), using default salt.")

    with contextlib.suppress(Exception):
        _secure_write_bytes(SALT_FILE, DEFAULT_SALT)
    return DEFAULT_SALT


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
                salt = data[:SALT_SIZE_BYTES]
                device_secret = os.urandom(DEVICE_SECRET_BYTES)
                _secure_write_bytes(SALT_FILE, salt + device_secret)
                return salt, device_secret
        except Exception as e:
            logger.warning(f"Failed reading salt file ({e}), regenerating new crypto material.")

    salt = _get_salt()
    device_secret = os.urandom(DEVICE_SECRET_BYTES)
    with contextlib.suppress(Exception):
        _secure_write_bytes(SALT_FILE, salt + device_secret)
    return salt, device_secret


def _get_or_create_cipher() -> Fernet:
    """
    Derive a deterministic Fernet key using PBKDF2 HMAC-SHA256 (100k rounds).

    Security Model:
    1. If CREDENTIALS_ENCRYPTION_KEY is provided in environment, it is used as master
       key/passphrase. If it's a valid 32-byte urlsafe base64 Fernet key, use directly.
    2. Otherwise, seed material defaults to deterministic master passphrase:
       'pm25-ctu-thesis-secure-storage-master-v1'.
    3. Mixed with 16-byte salt via PBKDF2 HMAC-SHA256 (100,000 iterations).
    This guarantees deterministic key derivation across container restarts, hostname changes,
    and redeployments while maintaining AES-128 encryption.
    """
    custom_key = os.getenv("CREDENTIALS_ENCRYPTION_KEY", "").strip()

    if custom_key:
        try:
            decoded = base64.urlsafe_b64decode(custom_key.encode("utf-8"))
            if len(decoded) == 32:
                return Fernet(custom_key.encode("utf-8"))
        except Exception as err:
            logger.debug("Provided key is not raw base64 Fernet key: %s", err)

    salt = _get_salt()
    seed_material = (custom_key or DEFAULT_MASTER_SEED).encode("utf-8")
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        seed_material,
        salt,
        PBKDF2_ITERATIONS,
    )
    return Fernet(base64.urlsafe_b64encode(derived_key))


def _save_to_database(encrypted_b64_str: str) -> bool:
    """Persist encrypted credentials to database (table info_cards) using SQLAlchemy.

    Compatible with both SQLite and PostgreSQL.
    """
    try:
        from sqlalchemy import text

        from src.api.database import engine

        if engine is None:
            return False

        def _do_upsert(conn: Any) -> None:
            row = conn.execute(
                text("SELECT id FROM info_cards WHERE card_key = :card_key"),
                {"card_key": SYSTEM_CARD_KEY},
            ).first()

            if row:
                conn.execute(
                    text(
                        "UPDATE info_cards "
                        "SET content = :content, updated_at = CURRENT_TIMESTAMP "
                        "WHERE card_key = :card_key"
                    ),
                    {"content": encrypted_b64_str, "card_key": SYSTEM_CARD_KEY},
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO info_cards (card_key, title, content, page, display_order) "
                        "VALUES (:card_key, :title, :content, :page, :display_order)"
                    ),
                    {
                        "card_key": SYSTEM_CARD_KEY,
                        "title": "Encrypted AI Credentials (AES-128)",
                        "content": encrypted_b64_str,
                        "page": "_system",
                        "display_order": 999,
                    },
                )

        try:
            with engine.begin() as conn:
                _do_upsert(conn)
        except Exception:
            # Table might not exist yet; auto-create tables and retry once
            from src.api.models import Base

            Base.metadata.create_all(bind=engine)
            with engine.begin() as conn:
                _do_upsert(conn)

        logger.info("Successfully persisted encrypted credentials to database.")
        return True
    except Exception as e:
        logger.debug(f"Database persistence skipped or failed: {e}")
        return False


def _load_from_database() -> str | None:
    """Load encrypted credentials string from database (table info_cards)."""
    try:
        from sqlalchemy import text

        from src.api.database import engine

        if engine is None:
            return None

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT content FROM info_cards WHERE card_key = :card_key"),
                {"card_key": SYSTEM_CARD_KEY},
            ).first()
            if row and row[0]:
                logger.info("Retrieved encrypted credentials from database.")
                return str(row[0])
            return None
    except Exception as e:
        logger.debug(f"Database load skipped or failed: {e}")
        return None


def _delete_from_database() -> bool:
    """Delete encrypted credentials from database (table info_cards)."""
    try:
        from sqlalchemy import text

        from src.api.database import engine

        if engine is None:
            return False

        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM info_cards WHERE card_key = :card_key"),
                {"card_key": SYSTEM_CARD_KEY},
            )
        logger.info("Deleted encrypted credentials from database.")
        return True
    except Exception as e:
        logger.debug(f"Database delete skipped or failed: {e}")
        return False


def save_credentials(
    providers_dict: dict[str, Any],
    primary_provider: str | None = None,
    remember: bool = True,
) -> bool:
    """
    Encrypt and save provider credentials to disk and database (Dual-Tier).

    Args:
        providers_dict: Dictionary containing provider configs (e.g. api_key, model, base_url).
        primary_provider: Identifier of the primary active provider.
        remember: If False, automatically purges any persisted credentials and returns True.

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

        # Tier 1: Local file (0600)
        _secure_write_bytes(CREDENTIALS_FILE, encrypted_bytes)

        # Tier 2: Database (Supabase / SQLite)
        _save_to_database(encrypted_bytes.decode("ascii"))

        logger.info(f"Successfully encrypted and saved credentials for {len(clean_providers)} providers (Dual-Tier).")
        return True
    except Exception as e:
        logger.error(f"Failed to save encrypted credentials: {e}", exc_info=True)
        return False


def load_credentials() -> tuple[dict[str, Any], str | None]:
    """
    Load and decrypt stored provider credentials from disk or database (Dual-Tier).

    Returns:
        tuple of (providers_dict, primary_provider). Returns ({}, None) if
        credentials do not exist or decryption fails.
    """
    encrypted_bytes: bytes | None = None
    restored_from_db = False

    if CREDENTIALS_FILE.is_file():
        try:
            encrypted_bytes = CREDENTIALS_FILE.read_bytes()
        except Exception as e:
            logger.debug(f"Could not read local credentials file: {e}")

    # Fallback to Database if file missing or empty
    if not encrypted_bytes:
        db_content = _load_from_database()
        if db_content:
            encrypted_bytes = db_content.encode("ascii")
            restored_from_db = True

    if not encrypted_bytes:
        return {}, None

    try:
        cipher = _get_or_create_cipher()
        decrypted_json = cipher.decrypt(encrypted_bytes).decode("utf-8")
        payload = json.loads(decrypted_json)

        providers = payload.get("providers", {})
        primary_provider = payload.get("primary_provider")

        # Re-materialize local file if restored from database
        if restored_from_db or not CREDENTIALS_FILE.is_file():
            with contextlib.suppress(Exception):
                _secure_write_bytes(CREDENTIALS_FILE, encrypted_bytes)

        logger.info(f"Loaded {len(providers)} saved AI provider credentials.")
        return providers, primary_provider
    except InvalidToken:
        logger.warning("Failed to decrypt credentials: authentication token invalid or tampered.")
        # If local file was corrupted, try DB fallback if DB has different content
        if not restored_from_db:
            db_content = _load_from_database()
            if db_content and db_content.encode("ascii") != encrypted_bytes:
                try:
                    cipher = _get_or_create_cipher()
                    decrypted_json = cipher.decrypt(db_content.encode("ascii")).decode("utf-8")
                    payload = json.loads(decrypted_json)
                    providers = payload.get("providers", {})
                    primary_provider = payload.get("primary_provider")
                    with contextlib.suppress(Exception):
                        _secure_write_bytes(CREDENTIALS_FILE, db_content.encode("ascii"))
                    return providers, primary_provider
                except Exception as err:
                    logger.debug("Database fallback decryption failed: %s", err)
        return {}, None
    except Exception as e:
        logger.warning(f"Could not load stored credentials gracefully: {e}")
        return {}, None


def clear_credentials() -> bool:
    """
    Securely delete stored credentials file and database record.

    Returns:
        True if credentials were deleted or did not exist, False on failure.
    """
    try:
        if CREDENTIALS_FILE.is_file():
            CREDENTIALS_FILE.unlink(missing_ok=True)
            logger.info("Cleared stored credentials file.")
        _delete_from_database()
        return True
    except OSError as e:
        logger.error(f"Failed to delete credentials file: {e}")
        return False


def is_credentials_persisted() -> bool:
    """Check whether stored credentials currently exist on disk or in database."""
    if CREDENTIALS_FILE.is_file():
        return True
    try:
        return bool(_load_from_database())
    except Exception:
        return False

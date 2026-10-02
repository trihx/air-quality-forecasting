"""
Unit tests for secure credentials persistence module.
Tests encryption, secure file permissions, serialization roundtrip,
tamper-resistance, and cleanup.
"""

import os
import stat
from pathlib import Path

import pytest
from src.chatbot import credentials_manager


@pytest.fixture(autouse=True)
def isolate_credentials_files(tmp_path, monkeypatch):
    """Isolate credentials and salt file paths to a temporary directory."""
    temp_creds = tmp_path / ".chatbot_credentials.enc"
    temp_salt = tmp_path / ".chatbot_salt"
    monkeypatch.setattr(credentials_manager, "CREDENTIALS_FILE", temp_creds)
    monkeypatch.setattr(credentials_manager, "SALT_FILE", temp_salt)
    yield {
        "creds": temp_creds,
        "salt": temp_salt,
        "dir": tmp_path,
    }


def test_save_and_load_credentials_roundtrip():
    """Lưu providers_dict và primary_provider, sau đó đọc lại giải mã ra đúng 100%."""
    providers = {
        "groq": {
            "api_key": "gsk_test1234567890abcdefghijklmnopqrstuvwxyz",
            "model": "llama-3.1-8b-instant",
            "base_url": "",
        },
        "gemini": {
            "api_key": "AIzaSySecretGeminiKey123456",
            "model": "gemini-2.0-flash",
            "base_url": "",
        },
        "kaggle_ollama": {
            "api_key": "ollama",
            "model": "qwen3:4b",
            "base_url": "https://sample-subdomain.trycloudflare.com/v1",
        },
    }
    primary = "groq"

    assert credentials_manager.is_credentials_persisted() is False
    saved = credentials_manager.save_credentials(providers, primary_provider=primary, remember=True)
    assert saved is True
    assert credentials_manager.is_credentials_persisted() is True

    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_primary == "groq"
    assert loaded_providers == providers


def test_encrypted_file_is_not_plaintext(isolate_credentials_files):
    """File .chatbot_credentials.enc lưu trên đĩa KHÔNG CHỨA bất kỳ chuỗi plaintext nào của API key."""
    sensitive_strings = [
        "gsk_super_secret_groq_key_9999",
        "AIzaSySecretGeminiKey8888",
        "sk-proj-openaiSecretKey7777",
        "secret-tunnel-url.trycloudflare.com",
    ]
    providers = {
        "groq": {"api_key": sensitive_strings[0], "model": "", "base_url": ""},
        "gemini": {"api_key": sensitive_strings[1], "model": "", "base_url": ""},
        "openai": {"api_key": sensitive_strings[2], "model": "", "base_url": ""},
        "kaggle_ollama": {
            "api_key": "ollama",
            "model": "qwen3:4b",
            "base_url": f"https://{sensitive_strings[3]}/v1",
        },
    }

    credentials_manager.save_credentials(providers, primary_provider="gemini", remember=True)

    creds_file: Path = isolate_credentials_files["creds"]
    assert creds_file.exists()
    raw_content = creds_file.read_bytes()

    for secret in sensitive_strings:
        assert secret.encode("utf-8") not in raw_content, f"Rò rỉ plaintext secret trên đĩa: {secret}"


def test_file_permissions_are_secure(isolate_credentials_files):
    """Kiểm tra quyền file .chatbot_credentials.enc và .chatbot_salt là 0600 (owner read/write only)."""
    providers = {
        "gemini": {"api_key": "AIzaSySecureKey", "model": "", "base_url": ""},
    }
    credentials_manager.save_credentials(providers, primary_provider="gemini", remember=True)

    creds_file: Path = isolate_credentials_files["creds"]
    salt_file: Path = isolate_credentials_files["salt"]

    assert creds_file.exists()
    assert salt_file.exists()

    if os.name == "posix":
        creds_mode = stat.S_IMODE(creds_file.stat().st_mode)
        salt_mode = stat.S_IMODE(salt_file.stat().st_mode)
        assert creds_mode == 0o600, f"Quyền file credentials không an toàn: {oct(creds_mode)}"
        assert salt_mode == 0o600, f"Quyền file salt không an toàn: {oct(salt_mode)}"


def test_clear_credentials(isolate_credentials_files):
    """Gọi clear_credentials(), kiểm tra file mã hóa bị xóa và load_credentials() trả về ({}, None)."""
    providers = {
        "groq": {"api_key": "gsk_test123", "model": "", "base_url": ""},
    }
    credentials_manager.save_credentials(providers, primary_provider="groq", remember=True)
    assert isolate_credentials_files["creds"].exists()

    cleared = credentials_manager.clear_credentials()
    assert cleared is True
    assert not isolate_credentials_files["creds"].exists()
    assert credentials_manager.is_credentials_persisted() is False

    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_providers == {}
    assert loaded_primary is None


def test_load_corrupted_file_handles_gracefully(isolate_credentials_files):
    """Ghi nội dung rác vào file mã hóa, load_credentials() không crash mà trả về ({}, None)."""
    creds_file: Path = isolate_credentials_files["creds"]
    creds_file.write_bytes(b"corrupted-invalid-encrypted-payload-garbage-bytes-1234567890")

    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_providers == {}
    assert loaded_primary is None


def test_remember_me_false_removes_file(isolate_credentials_files):
    """Gọi save_credentials(..., remember=False) thì file mã hóa bị xóa."""
    providers = {
        "groq": {"api_key": "gsk_test123", "model": "", "base_url": ""},
    }
    credentials_manager.save_credentials(providers, primary_provider="groq", remember=True)
    assert isolate_credentials_files["creds"].exists()

    res = credentials_manager.save_credentials(providers, primary_provider="groq", remember=False)
    assert res is True
    assert not isolate_credentials_files["creds"].exists()
    assert credentials_manager.is_credentials_persisted() is False


def test_is_credentials_persisted(isolate_credentials_files):
    """Kiểm tra is_credentials_persisted trả về True/False chính xác."""
    assert credentials_manager.is_credentials_persisted() is False

    credentials_manager.save_credentials(
        {"groq": {"api_key": "gsk_123", "model": "", "base_url": ""}},
        remember=True,
    )
    assert credentials_manager.is_credentials_persisted() is True

    credentials_manager.clear_credentials()
    assert credentials_manager.is_credentials_persisted() is False


def test_custom_encryption_key_environment(monkeypatch):
    """Kiểm tra hoạt động chính xác khi có biến môi trường CREDENTIALS_ENCRYPTION_KEY."""
    monkeypatch.setenv("CREDENTIALS_ENCRYPTION_KEY", "my_custom_super_secure_passphrase_2026")
    providers = {"gemini": {"api_key": "AIzaCustomPassKey", "model": "gemini-2.0-flash", "base_url": ""}}
    assert credentials_manager.save_credentials(providers, primary_provider="gemini") is True

    loaded, primary = credentials_manager.load_credentials()
    assert primary == "gemini"
    assert loaded["gemini"]["api_key"] == "AIzaCustomPassKey"


def test_filters_empty_and_internal_keys():
    """Kiểm tra loại bỏ các provider rỗng và các key nội bộ bắt đầu bằng dấu gạch dưới."""
    providers = {
        "_primary": "groq",
        "groq": {"api_key": "gsk_valid", "model": "", "base_url": ""},
        "empty_provider": {"api_key": "", "model": "", "base_url": ""},
    }
    credentials_manager.save_credentials(providers, primary_provider="groq")
    loaded, primary = credentials_manager.load_credentials()

    assert primary == "groq"
    assert "_primary" not in loaded
    assert "empty_provider" not in loaded
    assert "groq" in loaded


def test_chat_page_integration_persist_helper(monkeypatch):
    """Kiểm tra helper _persist_current_credentials trong chat_page."""
    from src.chatbot import chat_page

    mock_state = {
        "remember_ai_credentials": True,
        "primary_provider": "gemini",
        "llm_provider_keys": {
            "gemini": {"api_key": "AIzaIntegrationTestKey", "model": "", "base_url": ""},
        },
    }
    monkeypatch.setattr(chat_page.st, "session_state", mock_state)

    chat_page._persist_current_credentials()
    assert credentials_manager.is_credentials_persisted() is True

    loaded, primary = credentials_manager.load_credentials()
    assert primary == "gemini"
    assert loaded["gemini"]["api_key"] == "AIzaIntegrationTestKey"

    # Test when remember_ai_credentials is False
    mock_state["remember_ai_credentials"] = False
    chat_page._persist_current_credentials()
    assert credentials_manager.is_credentials_persisted() is False


def test_chat_page_has_render_alias():
    """Kiểm tra render_chat_page alias tồn tại và trỏ tới page_ai_assistant."""
    from src.chatbot import chat_page

    assert callable(chat_page.render_chat_page)
    assert chat_page.render_chat_page == chat_page.page_ai_assistant

"""
Unit tests for secure credentials persistence module.
Tests encryption, secure file permissions, serialization roundtrip,
tamper-resistance, and cleanup.
"""

import contextlib
import os
import stat
from pathlib import Path

import pytest
from src.chatbot import credentials_manager


@pytest.fixture(autouse=True)
def isolate_credentials_files(tmp_path, monkeypatch):
    """Isolate credentials and salt file paths to a temporary directory and clean DB state."""
    temp_creds = tmp_path / ".chatbot_credentials.enc"
    temp_salt = tmp_path / ".chatbot_salt"
    monkeypatch.setattr(credentials_manager, "CREDENTIALS_FILE", temp_creds)
    monkeypatch.setattr(credentials_manager, "SALT_FILE", temp_salt)
    credentials_manager.clear_credentials()
    yield {
        "creds": temp_creds,
        "salt": temp_salt,
        "dir": tmp_path,
    }
    credentials_manager.clear_credentials()


def test_save_and_load_credentials_roundtrip():
    """Lưu providers_dict và primary_provider (Dual-Tier: File + DB), sau đó đọc lại giải mã ra đúng 100%."""
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

    # Verify both file and DB have content
    assert credentials_manager.CREDENTIALS_FILE.exists()
    db_ciphertext = credentials_manager._load_from_database()
    assert db_ciphertext is not None
    assert len(db_ciphertext) > 50

    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_primary == "groq"
    assert loaded_providers == providers


def test_encrypted_file_and_database_contain_zero_plaintext(isolate_credentials_files):
    """Cả file .chatbot_credentials.enc và bản ghi database KHÔNG CHỨA bất kỳ chuỗi plaintext nào của API key."""
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

    db_content = credentials_manager._load_from_database()
    assert db_content is not None

    for secret in sensitive_strings:
        assert secret.encode("utf-8") not in raw_content, f"Rò rỉ plaintext secret trên đĩa: {secret}"
        assert secret not in db_content, f"Rò rỉ plaintext secret trong database: {secret}"


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
    """Gọi clear_credentials(), kiểm tra cả file mã hóa và database record bị xóa sạch."""
    providers = {
        "groq": {"api_key": "gsk_test123", "model": "", "base_url": ""},
    }
    credentials_manager.save_credentials(providers, primary_provider="groq", remember=True)
    assert isolate_credentials_files["creds"].exists()
    assert credentials_manager._load_from_database() is not None

    cleared = credentials_manager.clear_credentials()
    assert cleared is True
    assert not isolate_credentials_files["creds"].exists()
    assert credentials_manager._load_from_database() is None
    assert credentials_manager.is_credentials_persisted() is False

    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_providers == {}
    assert loaded_primary is None


def test_load_from_database_when_local_file_deleted(isolate_credentials_files):
    """Khi file local bị xóa (mô phỏng container Render restart), load_credentials khôi phục thành công từ DB."""
    providers = {
        "gemini": {"api_key": "AIzaSyCloudPersistenceKey", "model": "gemini-2.5-flash", "base_url": ""},
        "kaggle_ollama": {
            "api_key": "ollama",
            "model": "qwen3:8b",
            "base_url": "https://active-tunnel.trycloudflare.com/v1",
        },
    }
    credentials_manager.save_credentials(providers, primary_provider="gemini", remember=True)

    # Xóa file local trên đĩa để mô phỏng container restart
    creds_file: Path = isolate_credentials_files["creds"]
    assert creds_file.exists()
    creds_file.unlink()
    assert not creds_file.exists()

    # load_credentials() phải đọc từ Database và tự động khôi phục lại file local
    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_primary == "gemini"
    assert loaded_providers == providers
    assert creds_file.exists(), "File local phải được tái tạo tự động sau khi khôi phục từ DB"


def test_deterministic_key_derivation_across_restarts(isolate_credentials_files):
    """Mô phỏng Render redeploy hoàn toàn: xóa cả file credentials và salt file."""
    providers = {
        "groq": {"api_key": "gsk_deterministic_test_key", "model": "llama-3.3-70b-versatile", "base_url": ""},
    }
    credentials_manager.save_credentials(providers, primary_provider="groq", remember=True)

    # Mô phỏng ephemeral wipe sạch thư mục app
    creds_file: Path = isolate_credentials_files["creds"]
    salt_file: Path = isolate_credentials_files["salt"]
    creds_file.unlink(missing_ok=True)
    salt_file.unlink(missing_ok=True)
    assert not creds_file.exists()
    assert not salt_file.exists()

    # Nhờ thuật toán phái sinh khóa ổn định (Deterministic Key Derivation), việc giải mã từ DB vẫn thành công
    loaded_providers, loaded_primary = credentials_manager.load_credentials()
    assert loaded_primary == "groq"
    assert loaded_providers == providers


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


def test_widget_state_sync_in_chat_page(isolate_credentials_files, monkeypatch):
    """Kiểm tra page_ai_assistant đồng bộ đầy đủ các widget keys trong st.session_state khi nạp credentials."""
    from src.chatbot import chat_page

    providers = {
        "gemini": {"api_key": "AIzaSyWidgetSyncKey", "model": "gemini-2.0-flash", "base_url": ""},
        "groq": {"api_key": "gsk_groqWidgetKey", "model": "llama-3.1-8b-instant", "base_url": ""},
        "kaggle_ollama": {
            "api_key": "ollama",
            "model": "qwen3:4b",
            "base_url": "https://widget-sync.trycloudflare.com",
        },
    }
    credentials_manager.save_credentials(providers, primary_provider="gemini", remember=True)

    class MockSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                return None

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            with contextlib.suppress(KeyError):
                del self[name]

    mock_session = MockSessionState()
    monkeypatch.setattr(chat_page.st, "session_state", mock_session)
    # Mock visual components to avoid full Streamlit rendering
    from unittest.mock import MagicMock

    mock_col = MagicMock()
    mock_col.__enter__.return_value = mock_col
    mock_col.__exit__.return_value = None

    monkeypatch.setattr(chat_page, "_render_provider_config", lambda: None)
    monkeypatch.setattr(chat_page, "_render_inline_quick_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page, "_get_knowledge_base", lambda: None)
    monkeypatch.setattr(chat_page, "_ensure_index", lambda *args, **kwargs: 0)
    monkeypatch.setattr(chat_page.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "columns", lambda *args, **kwargs: (mock_col, mock_col))
    monkeypatch.setattr(chat_page.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "selectbox", lambda *args, **kwargs: "gemini")
    monkeypatch.setattr(chat_page.st, "chat_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_page.st, "button", lambda *args, **kwargs: False)

    # Call page_ai_assistant
    chat_page.page_ai_assistant(results=None)

    # Assert widget keys populated
    assert mock_session.get("input_key_gemini") == "AIzaSyWidgetSyncKey"
    assert mock_session.get("main_key_gemini") == "AIzaSyWidgetSyncKey"
    assert mock_session.get("quick_gemini_key") == "AIzaSyWidgetSyncKey"

    assert mock_session.get("input_key_groq") == "gsk_groqWidgetKey"
    assert mock_session.get("quick_groq_key") == "gsk_groqWidgetKey"

    assert mock_session.get("input_kaggle_ollama_url") == "https://widget-sync.trycloudflare.com"
    assert mock_session.get("quick_kaggle_url") == "https://widget-sync.trycloudflare.com"
    assert mock_session.get("main_kaggle_url") == "https://widget-sync.trycloudflare.com"
    assert mock_session.get("input_kaggle_model") == "qwen3:4b"

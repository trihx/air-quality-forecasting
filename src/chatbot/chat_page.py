"""
AI Assistant page for PM2.5 Forecasting Dashboard.

Provides a chat interface with:
  - Multi-LLM support (Gemini, OpenAI, Groq, LM Studio)
  - Tiered fallback (Cloud API → Local LLM)
  - RAG-based context retrieval from project knowledge base
  - Secure API key management via session state
"""

import contextlib
import logging

import streamlit as st

from src.chatbot.credentials_manager import (
    clear_credentials,
    is_credentials_persisted,
    load_credentials,
    save_credentials,
)
from src.chatbot.provider_config import (
    GROQ_MODEL_RECOMMENDATIONS,
    LOCAL_MODEL_RECOMMENDATIONS,
    PROVIDER_REGISTRY,
    detect_available_providers,
    get_lm_studio_default_url,
    get_provider_from_registry,
    is_cloud_environment,
    mask_api_key,
    validate_provider_connection,
)

logger = logging.getLogger(__name__)

# Max messages retained in session to prevent OOM on 512MB RAM containers
MAX_CHAT_HISTORY = 40


def _persist_current_credentials(primary: str | None = None) -> None:
    """Helper to persist current session credentials if remember_ai_credentials is enabled."""
    remember = bool(st.session_state.get("remember_ai_credentials", is_credentials_persisted()))
    pri = primary or st.session_state.get("primary_provider")
    save_credentials(
        st.session_state.get("llm_provider_keys", {}),
        primary_provider=pri,
        remember=remember,
    )


def _trim_chat_history(messages: list[dict], max_history: int = MAX_CHAT_HISTORY) -> list[dict]:
    """Trim chat history to keep memory bounded on low-RAM containers."""
    if len(messages) > max_history:
        return messages[-max_history:]
    return messages


# ── Preset questions for thesis defense preparation ──
PRESET_QUESTIONS = {
    "📋 Phương pháp luận": [
        "Giải thích quy trình pipeline end-to-end của dự án",
        "Tại sao chọn phương pháp anti-leakage bằng shift(1)?",
        "Tại sao dùng MASE làm metric chính thay vì RMSE hay MAE?",
        "Giải thích cách chia train/validation/test theo temporal split",
    ],
    "🔬 Xử lý dữ liệu": [
        "Cách xử lý missing data trong dữ liệu IoT sensor?",
        "Tại sao dùng IQR 3.0 để phát hiện outlier?",
        "Giải thích chiến lược imputation: Spline vs KNN?",
        "Feature engineering đã thực hiện những gì?",
    ],
    "🤖 Mô hình": [
        "Tại sao Persistence baseline rất mạnh ở horizon 1h?",
        "So sánh ưu nhược điểm của GRU vs LSTM trong dự án",
        "LightGBM được tối ưu hyperparameter bằng cách nào?",
        "TFT Transformer có ưu điểm gì so với các mô hình khác?",
    ],
    "📊 Đánh giá": [
        "Tại sao cần đánh giá ở nhiều horizons (1h, 6h, 24h)?",
        "Shuffle test là gì và kết quả ra sao?",
        "Giải thích về Prediction Intervals trong dự án",
        "SHAP Explainability cho thấy features nào quan trọng nhất?",
    ],
    "🎓 Bài học kinh nghiệm": [
        "Những lỗi quan trọng nhất đã gặp và cách khắc phục?",
        "Data leakage đã được phát hiện và xử lý như thế nào?",
        "Kinh nghiệm xử lý dữ liệu IoT bị thiếu 85%?",
        "Bài học gì từ việc so sánh nhiều mô hình ML/DL?",
    ],
}


def _get_knowledge_base():
    """Lazy import and get knowledge base singleton."""
    from src.chatbot.knowledge_base import get_knowledge_base

    return get_knowledge_base()


def _ensure_index(kb) -> int:
    """Ensure knowledge base is indexed, re-index if user content changed.

    Checks for a `.needs_reindex` flag file set by the content API
    when users edit info cards via the Dashboard UI.
    Does NOT block page load with automatic heavy build_index() to prevent OOM/freezing.
    """
    from src.chatbot.knowledge_base import REINDEX_FLAG_PATH

    # Check if user content was updated (flag set by content API)
    needs_reindex = REINDEX_FLAG_PATH.exists()

    if needs_reindex:
        with st.spinner("🔄 Cập nhật kiến thức mới từ nội dung đã chỉnh sửa..."):
            try:
                count = kb.build_index(force=True)
                with contextlib.suppress(FileNotFoundError):
                    REINDEX_FLAG_PATH.unlink()
                st.toast("✅ Kiến thức chatbot đã được cập nhật!", icon="🧠")
                return count
            except Exception as e:
                logger.warning(f"Re-indexing failed: {e}")
                with contextlib.suppress(FileNotFoundError):
                    REINDEX_FLAG_PATH.unlink()

    return kb.index_count()


def _render_provider_config():
    """Render AI provider configuration in sidebar."""
    st.sidebar.markdown(
        """
    <div style="font-size: 0.75rem; opacity: 0.6; text-transform: uppercase;
                letter-spacing: 0.1em; margin: 1rem 0 0.5rem 0; font-weight: 700;">
        ⚙️ Cấu Hình AI Provider
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for provider keys
    if "llm_provider_keys" not in st.session_state:
        st.session_state.llm_provider_keys = {}

    # Provider selection tabs
    for provider_name, reg in PROVIDER_REGISTRY.items():
        if provider_name in ("lm_studio", "kaggle_ollama"):
            continue  # Handle separately below

        with st.sidebar.expander(
            f"{'🟢' if st.session_state.llm_provider_keys.get(provider_name, {}).get('api_key') else '⚪'} "
            f"{reg['display_name']}",
            expanded=False,
        ):
            st.caption(reg["description"])

            # API key input (password masked)
            key = st.text_input(
                "API Key",
                type="password",
                key=f"input_key_{provider_name}",
                placeholder="Nhập API key...",
                value=st.session_state.llm_provider_keys.get(provider_name, {}).get("api_key", ""),
            )

            # Model override (optional)
            model = st.text_input(
                "Model (tùy chọn)",
                key=f"input_model_{provider_name}",
                placeholder=reg["default_model"],
                value=st.session_state.llm_provider_keys.get(provider_name, {}).get("model", ""),
            )

            if provider_name == "groq":
                with st.expander("📋 Model khuyến nghị cho Groq (Tránh 429 TPM)", expanded=False):
                    for m in GROQ_MODEL_RECOMMENDATIONS:
                        star = " ⭐ (Khuyên dùng)" if m["recommended"] else ""
                        st.markdown(f"• **`{m['name']}`**{star}: {m['description']}")

            col_save, col_test = st.columns(2)
            with col_save:
                if st.button("💾 Lưu", key=f"save_{provider_name}", use_container_width=True):
                    if key:
                        provider = get_provider_from_registry(provider_name, api_key=key, custom_model=model)
                        final_m = model
                        if provider:
                            with st.spinner("Đang kiểm tra & kích hoạt..."):
                                ok, _ = validate_provider_connection(provider)
                            if ok and provider.model:
                                final_m = provider.model
                        st.session_state.llm_provider_keys[provider_name] = {
                            "api_key": key,
                            "model": final_m,
                            "base_url": "",
                        }
                        _persist_current_credentials()
                        st.success(f"✅ Đã lưu ({mask_api_key(key)}) • Model: {final_m or reg['default_model']}")
                    else:
                        # Clear provider
                        st.session_state.llm_provider_keys.pop(provider_name, None)
                        _persist_current_credentials()
                        st.info("Đã xóa API key")

            with col_test:
                if st.button("🔍 Test", key=f"test_{provider_name}", use_container_width=True):
                    if key:
                        provider = get_provider_from_registry(provider_name, api_key=key, custom_model=model)
                        if provider:
                            with st.spinner("Đang kiểm tra..."):
                                ok, msg = validate_provider_connection(provider)
                            if ok:
                                if (
                                    provider.model
                                    and provider.model != model
                                    and provider_name in st.session_state.llm_provider_keys
                                ):
                                    st.session_state.llm_provider_keys[provider_name]["model"] = provider.model
                                    _persist_current_credentials()
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                    else:
                        st.warning("Chưa nhập API key")

    # ── Kaggle Ollama (Free GPU) ──
    with st.sidebar.expander(
        f"{'🟢' if st.session_state.llm_provider_keys.get('kaggle_ollama', {}).get('base_url') else '⚪'} "
        "Kaggle Ollama (Free GPU)",
        expanded=False,
    ):
        st.caption(PROVIDER_REGISTRY["kaggle_ollama"]["description"])

        kaggle_url = st.text_input(
            "Tunnel URL",
            key="input_kaggle_ollama_url",
            placeholder="https://xxx.trycloudflare.com",
            value=st.session_state.llm_provider_keys.get("kaggle_ollama", {}).get("base_url", ""),
        )

        kaggle_model = st.text_input(
            "Model (tùy chọn)",
            key="input_kaggle_model",
            placeholder=PROVIDER_REGISTRY["kaggle_ollama"]["default_model"],
            value=st.session_state.llm_provider_keys.get("kaggle_ollama", {}).get("model", ""),
        )

        col_save_k, col_test_k = st.columns(2)
        with col_save_k:
            if st.button("💾 Lưu", key="save_kaggle_ollama", use_container_width=True):
                if kaggle_url:
                    st.session_state.llm_provider_keys["kaggle_ollama"] = {
                        "api_key": "ollama",
                        "base_url": kaggle_url,
                        "model": kaggle_model,
                    }
                    _persist_current_credentials()
                    st.success("✅ Đã lưu Kaggle Ollama")
                else:
                    st.session_state.llm_provider_keys.pop("kaggle_ollama", None)
                    _persist_current_credentials()
                    st.info("Đã xóa cấu hình Kaggle")

        with col_test_k:
            if st.button("🔍 Test", key="test_kaggle_ollama", use_container_width=True):
                if kaggle_url:
                    provider = get_provider_from_registry(
                        "kaggle_ollama", custom_base_url=kaggle_url, custom_model=kaggle_model
                    )
                    if provider:
                        with st.spinner("Đang kết nối Kaggle..."):
                            ok, msg = validate_provider_connection(provider)
                        if ok:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                else:
                    st.warning("Chưa nhập Tunnel URL")

        # Model recommendations for Kaggle
        from src.chatbot.provider_config import KAGGLE_MODEL_RECOMMENDATIONS

        with st.expander("📋 Model khuyến nghị cho Kaggle", expanded=False):
            for m in KAGGLE_MODEL_RECOMMENDATIONS:
                star = " ⭐" if m["recommended"] else ""
                st.markdown(
                    f"**{m['name']}{star}**\n- VRAM: {m['vram']} | {m['vietnamese']}\n- {m['best_for']}",
                )

        # Setup guide
        from src.chatbot.kaggle_setup_guide import render_kaggle_setup_guide

        render_kaggle_setup_guide()

    # ── LM Studio (Local) ──
    with st.sidebar.expander("🖥️ LM Studio (Local)", expanded=False):
        st.caption(PROVIDER_REGISTRY["lm_studio"]["description"])

        lm_url = st.text_input(
            "Server URL",
            key="input_lm_studio_url",
            value=st.session_state.llm_provider_keys.get("lm_studio", {}).get(
                "base_url",
                get_lm_studio_default_url(),
            ),
        )

        if st.button("🔍 Kiểm tra LM Studio", key="test_lm_studio", use_container_width=True):
            provider = get_provider_from_registry("lm_studio", custom_base_url=lm_url)
            if provider:
                with st.spinner("Đang kết nối..."):
                    ok, msg = validate_provider_connection(provider)
                if ok:
                    st.success(f"✅ {msg}")
                    # Save URL to session
                    st.session_state.llm_provider_keys["lm_studio"] = {
                        "api_key": "lm-studio",
                        "base_url": lm_url,
                        "model": "",
                    }
                else:
                    st.error(f"❌ {msg}")

        # Model recommendations
        with st.expander("📋 Model khuyến nghị", expanded=False):
            for m in LOCAL_MODEL_RECOMMENDATIONS:
                star = " ⭐" if m["recommended"] else ""
                st.markdown(
                    f"**{m['name']}{star}**\n- VRAM: {m['vram']} | {m['vietnamese']}\n- {m['best_for']}",
                )

    # ── Status summary ──
    providers = detect_available_providers(st.session_state.llm_provider_keys)
    cloud_providers = [p for p in providers if not p.is_local]

    if cloud_providers:
        names = ", ".join([p.display_name for p in cloud_providers])
        st.sidebar.success(f"☁️ Cloud: {names}")
    else:
        st.sidebar.info("☁️ Chưa cấu hình Cloud API")

    local_cfg = st.session_state.llm_provider_keys.get("lm_studio", {})
    if local_cfg.get("api_key"):
        st.sidebar.success("🖥️ LM Studio: Đã cấu hình")
    elif is_cloud_environment():
        st.sidebar.caption("🖥️ LM Studio: Tắt (Môi trường Cloud)")
    else:
        st.sidebar.caption("🖥️ LM Studio: Fallback (auto-detect)")


def _render_inline_quick_config(expanded: bool = False):
    """Render inline quick configuration form directly on the main page for seamless UX."""
    with st.expander(
        "⚙️ Cấu Hình Nhanh AI Provider (Dán Link Kaggle hoặc Nhập API Key tại đây)",
        expanded=expanded,
    ):
        tab_kaggle, tab_gemini, tab_other = st.tabs(
            [
                "🚀 Kaggle Ollama (Free GPU 32GB)",
                "⚡ Google Gemini (Miễn phí 100%)",
                "🌐 Groq / OpenAI",
            ]
        )

        with tab_kaggle:
            st.caption(PROVIDER_REGISTRY["kaggle_ollama"]["description"])
            k_url = st.text_input(
                "Cloudflare Tunnel URL",
                key="main_kaggle_url",
                placeholder="https://xxx.trycloudflare.com",
                value=st.session_state.llm_provider_keys.get("kaggle_ollama", {}).get("base_url", ""),
                help="Dán URL sinh ra từ Cell 4 của Kaggle Notebook",
            )
            k_model = st.text_input(
                "Model (tùy chọn)",
                key="main_kaggle_model",
                placeholder=PROVIDER_REGISTRY["kaggle_ollama"]["default_model"],
                value=st.session_state.llm_provider_keys.get("kaggle_ollama", {}).get("model", ""),
            )
            col_k_save, col_k_test = st.columns(2)
            with col_k_save:
                if st.button(
                    "💾 Lưu & Kích Hoạt Kaggle", key="main_save_kaggle", type="primary", use_container_width=True
                ):
                    if k_url:
                        st.session_state.llm_provider_keys["kaggle_ollama"] = {
                            "api_key": "ollama",
                            "base_url": k_url,
                            "model": k_model,
                        }
                        _persist_current_credentials()
                        st.success("✅ Đã kích hoạt Kaggle Ollama thành công!")
                        st.rerun()
                    else:
                        st.session_state.llm_provider_keys.pop("kaggle_ollama", None)
                        _persist_current_credentials()
                        st.info("Đã xóa cấu hình Kaggle")
                        st.rerun()
            with col_k_test:
                if st.button("🔍 Kiểm Tra Kết Nối", key="main_test_kaggle", use_container_width=True):
                    if k_url:
                        provider = get_provider_from_registry(
                            "kaggle_ollama", custom_base_url=k_url, custom_model=k_model
                        )
                        if provider:
                            with st.spinner("Đang kết nối tới Kaggle Ollama..."):
                                ok, msg = validate_provider_connection(provider)
                            if ok:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                    else:
                        st.warning("Vui lòng nhập Tunnel URL trước")

            from src.chatbot.kaggle_setup_guide import render_kaggle_setup_guide

            render_kaggle_setup_guide()

        with tab_gemini:
            st.caption(PROVIDER_REGISTRY["gemini"]["description"])
            st.markdown(
                "👉 **[Nhận API Key Miễn Phí tại Google AI Studio](https://aistudio.google.com/app/apikey)** "
                "*(Bấm Create API Key rồi copy dán vào ô dưới)*"
            )
            g_key = st.text_input(
                "Gemini API Key",
                type="password",
                key="main_gemini_key",
                placeholder="AIzaSy...",
                value=st.session_state.llm_provider_keys.get("gemini", {}).get("api_key", ""),
            )
            g_model = st.text_input(
                "Model (tùy chọn)",
                key="main_gemini_model",
                placeholder=PROVIDER_REGISTRY["gemini"]["default_model"],
                value=st.session_state.llm_provider_keys.get("gemini", {}).get("model", ""),
            )
            col_g_save, col_g_test = st.columns(2)
            with col_g_save:
                if st.button(
                    "💾 Lưu & Kích Hoạt Gemini", key="main_save_gemini", type="primary", use_container_width=True
                ):
                    if g_key:
                        st.session_state.llm_provider_keys["gemini"] = {
                            "api_key": g_key,
                            "model": g_model,
                            "base_url": "",
                        }
                        _persist_current_credentials()
                        st.success(f"✅ Đã lưu Google Gemini ({mask_api_key(g_key)})")
                        st.rerun()
                    else:
                        st.session_state.llm_provider_keys.pop("gemini", None)
                        _persist_current_credentials()
                        st.info("Đã xóa Google Gemini")
                        st.rerun()
            with col_g_test:
                if st.button("🔍 Kiểm Tra Kết Nối", key="main_test_gemini", use_container_width=True):
                    if g_key:
                        provider = get_provider_from_registry("gemini", api_key=g_key, custom_model=g_model)
                        if provider:
                            with st.spinner("Đang kết nối tới Google Gemini..."):
                                ok, msg = validate_provider_connection(provider)
                            if ok:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                    else:
                        st.warning("Vui lòng nhập Gemini API Key")

        with tab_other:
            for p_name in ("groq", "openai"):
                reg = PROVIDER_REGISTRY[p_name]
                st.markdown(f"**{reg['display_name']}** — *{reg['description']}*")
                o_key = st.text_input(
                    f"{reg['display_name']} API Key",
                    type="password",
                    key=f"main_key_{p_name}",
                    placeholder="Nhập API key...",
                    value=st.session_state.llm_provider_keys.get(p_name, {}).get("api_key", ""),
                )
                o_model = st.text_input(
                    f"Model {reg['display_name']} (tùy chọn)",
                    key=f"main_model_{p_name}",
                    placeholder=reg["default_model"],
                    value=st.session_state.llm_provider_keys.get(p_name, {}).get("model", ""),
                    help=f"Mặc định: {reg['default_model']}",
                )

                if p_name == "groq":
                    with st.expander("📋 Model khuyến nghị cho Groq (Tránh 429 TPM)", expanded=False):
                        for m in GROQ_MODEL_RECOMMENDATIONS:
                            star = " ⭐ (Khuyên dùng)" if m["recommended"] else ""
                            st.markdown(f"• **`{m['name']}`**{star}: {m['description']}")

                col_o_save, col_o_test = st.columns(2)
                with col_o_save:
                    if st.button(f"💾 Lưu {reg['display_name']}", key=f"main_save_{p_name}", use_container_width=True):
                        if o_key:
                            provider = get_provider_from_registry(p_name, api_key=o_key, custom_model=o_model)
                            final_o_model = o_model
                            if provider:
                                with st.spinner(f"Đang kiểm tra & kích hoạt {reg['display_name']}..."):
                                    ok, _ = validate_provider_connection(provider)
                                if ok and provider.model:
                                    final_o_model = provider.model
                            st.session_state.llm_provider_keys[p_name] = {
                                "api_key": o_key,
                                "model": final_o_model,
                                "base_url": "",
                            }
                            _persist_current_credentials()
                            st.success(
                                f"✅ Đã lưu {reg['display_name']} ({mask_api_key(o_key)}) • Model: {final_o_model or reg['default_model']}"
                            )
                            st.rerun()
                        else:
                            st.session_state.llm_provider_keys.pop(p_name, None)
                            _persist_current_credentials()
                            st.info(f"Đã xóa {reg['display_name']}")
                            st.rerun()
                with col_o_test:
                    if st.button(f"🔍 Test {reg['display_name']}", key=f"main_test_{p_name}", use_container_width=True):
                        if o_key:
                            provider = get_provider_from_registry(p_name, api_key=o_key, custom_model=o_model)
                            if provider:
                                with st.spinner("Đang kiểm tra..."):
                                    ok, msg = validate_provider_connection(provider)
                                if ok:
                                    if (
                                        provider.model
                                        and provider.model != o_model
                                        and p_name in st.session_state.llm_provider_keys
                                    ):
                                        st.session_state.llm_provider_keys[p_name]["model"] = provider.model
                                        _persist_current_credentials()
                                    st.success(f"✅ {msg}")
                                else:
                                    st.error(f"❌ {msg}")
                        else:
                            st.warning("Chưa nhập API key")
                st.divider()

        # ── Credentials Persistence & Zeroize Options ──
        col_remember, col_clear = st.columns([3, 2])
        with col_remember:
            remember_val = st.checkbox(
                "💾 Tự động ghi nhớ trên thiết bị này (Mã hóa an toàn AES-128)",
                value=is_credentials_persisted()
                if "remember_ai_credentials" not in st.session_state
                else bool(st.session_state.remember_ai_credentials),
                key="remember_ai_credentials",
                help="Thông tin API Key và Tunnel URL được mã hóa đối xứng Fernet (AES-128 + HMAC-SHA256) với quyền file 0600.",
            )
            if not remember_val and is_credentials_persisted():
                clear_credentials()
        with col_clear:
            if st.button(
                "🗑️ Xóa sạch thông tin kết nối đã lưu",
                use_container_width=True,
                help="Xóa hoàn toàn file mã hóa trên đĩa và làm mới cấu hình kết nối",
            ):
                clear_credentials()
                st.session_state.llm_provider_keys = {}
                st.session_state.primary_provider = None
                st.session_state.pop("primary_ai_provider", None)
                # Triệt để zeroize toàn bộ widget-state keys của Streamlit
                for k in list(st.session_state.keys()):
                    if isinstance(k, str) and any(
                        k.startswith(pfx)
                        for pfx in (
                            "input_key_",
                            "input_model_",
                            "main_key_",
                            "main_model_",
                            "main_kaggle_",
                            "input_kaggle_",
                        )
                    ):
                        st.session_state.pop(k, None)
                st.toast("✅ Đã xóa sạch toàn bộ thông tin kết nối và bộ nhớ tạm trên thiết bị!", icon="🗑️")
                st.rerun()


def page_ai_assistant(results):
    """Render AI Assistant chatbot page."""
    # ── Initialize remember_me state early to preserve opt-in consent ──
    if "remember_ai_credentials" not in st.session_state:
        st.session_state.remember_ai_credentials = is_credentials_persisted()

    # ── Auto-restore saved credentials if session is empty ──
    if "llm_provider_keys" not in st.session_state or not st.session_state.llm_provider_keys:
        saved_keys, saved_primary = load_credentials()
        if saved_keys:
            st.session_state.llm_provider_keys = saved_keys
            if saved_primary:
                st.session_state.primary_provider = saved_primary
                st.session_state.primary_ai_provider = saved_primary

    # ── Render provider config in sidebar ──
    _render_provider_config()

    # ── Header ──
    st.markdown(
        """
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        💬 Trợ Lý AI — Phân Tích Kỹ Thuật
    </h1>
    <p style="opacity: 0.7; font-size: 1.05rem; margin-bottom: 1rem;">
        Trợ lý AI chuyên sâu • Tra cứu kiến trúc mô hình, quy trình tiền xử lý và kết quả thực nghiệm
    </p>
    """,
        unsafe_allow_html=True,
    )

    # ── Version-aware info cards ──
    from src.info_cards import cards_ai_assistant, get_current_version, render_version_badge

    ver = get_current_version()
    render_version_badge(ver)
    cards_ai_assistant(ver)

    # ── Active provider status & Primary Provider selection ──
    raw_providers = detect_available_providers(st.session_state.get("llm_provider_keys", {}))
    available_names = [p.name for p in raw_providers]

    # Validate or set primary provider
    if "primary_provider" not in st.session_state or st.session_state.primary_provider not in available_names:
        st.session_state.primary_provider = available_names[0] if available_names else None

    # Re-detect with primary provider prioritized
    providers = detect_available_providers(
        st.session_state.get("llm_provider_keys", {}),
        primary_provider=st.session_state.primary_provider,
    )
    cloud_providers = [p for p in providers if not p.is_local]
    local_providers = [p for p in providers if p.is_local]

    col_status, col_info = st.columns([2, 1])
    with col_status:
        if cloud_providers:
            primary = cloud_providers[0]
            st.success(
                f"🟢 AI: **{primary.display_name}** ({primary.model or 'auto'})"
                + (f" + {len(cloud_providers) - 1} fallback" if len(cloud_providers) > 1 else "")
                + (" + LM Studio" if local_providers else "")
            )
        elif local_providers:
            st.info("🖥️ AI: **LM Studio** (local)")
        else:
            if is_cloud_environment():
                st.warning(
                    "⚠️ Chưa kích hoạt AI Provider — Vui lòng cấu hình ở bảng bên dưới hoặc mở Sidebar (góc trên cùng bên trái >)."
                )
            else:
                st.warning("⚠️ Chưa cấu hình AI — vào **⚙️ Cấu Hình AI Provider** ở sidebar hoặc bảng bên dưới")

    with col_info:
        # Show fallback chain
        chain = " → ".join([p.display_name for p in providers]) or "Chưa cấu hình"
        st.caption(f"🔗 Thứ tự: {chain}")

    # Primary provider selection selector if 2+ providers available
    if len(raw_providers) > 1:
        current_pri = st.session_state.primary_provider or raw_providers[0].name
        pri_idx = available_names.index(current_pri) if current_pri in available_names else 0
        display_map = {p.name: f"{p.display_name} ({p.model or 'auto'})" for p in raw_providers}

        selected_pri = st.selectbox(
            "⭐ Chọn AI Provider ưu tiên hàng đầu (Primary Provider):",
            options=available_names,
            index=pri_idx,
            format_func=lambda k: display_map.get(k, k),
            key="select_primary_provider_widget",
            help="Provider được chọn sẽ luôn được ưu tiên gọi đầu tiên. Nếu gặp lỗi, hệ thống sẽ tự động fallback sang các provider còn lại.",
        )
        if selected_pri != st.session_state.primary_provider:
            st.session_state.primary_provider = selected_pri
            st.session_state.primary_ai_provider = selected_pri
            _persist_current_credentials(primary=selected_pri)
            st.rerun()

    # ── Inline Quick Config Form (Always accessible on page) ──
    _render_inline_quick_config(expanded=(not cloud_providers and not local_providers))

    # ── Knowledge Base status ──
    kb_col, reindex_col = st.columns([3, 1])
    try:
        kb = _get_knowledge_base()
        doc_count = _ensure_index(kb)
        with kb_col:
            if doc_count > 0:
                st.caption(f"📚 Knowledge Base: {doc_count} tài liệu indexed (Vector RAG sẵn sàng)")
            else:
                st.caption("📚 Knowledge Base: Sẵn sàng (Chế độ System Prompt tích hợp)")
            if is_cloud_environment():
                st.caption(
                    "⚠️ Lưu ý: Trên Cloud Server Render (RAM 512MB), khuyến nghị sử dụng chế độ "
                    "System Prompt tích hợp (đã chứa sẵn 100% tri thức và số liệu cốt lõi của đề án)."
                )
        with reindex_col:
            btn_label = "🔄 Tạo Index" if doc_count == 0 else "🔄 Re-index"
            if st.button(btn_label, help="Cập nhật hoặc xây dựng vector index cho chatbot"):
                with st.spinner("🔄 Đang xử lý index tài liệu... (có thể mất 1-2 phút)"):
                    try:
                        new_count = kb.build_index(force=True)
                        st.success(f"✅ Đã index {new_count} tài liệu!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"❌ Không thể tạo index: {err}")
    except Exception as e:
        with kb_col:
            st.caption("📚 Knowledge Base: Sẵn sàng (Chế độ System Prompt tích hợp)")
            if is_cloud_environment():
                st.caption(
                    "⚠️ Lưu ý: Trên Cloud Server Render (RAM 512MB), khuyến nghị sử dụng chế độ "
                    "System Prompt tích hợp (đã chứa sẵn 100% tri thức và số liệu cốt lõi của đề án)."
                )
        logger.warning(f"Knowledge Base initialization notice: {e}")
        kb = None

    st.divider()

    # ── Layout: Chat + Presets ──
    chat_col, preset_col = st.columns([3, 1])

    # ── Preset Questions (right panel) ──
    with preset_col:
        st.markdown(
            """
        <div style="font-size: 0.85rem; font-weight: 700;
                    margin-bottom: 0.75rem;">
            💡 Câu Hỏi Kỹ Thuật Thường Gặp
        </div>
        """,
            unsafe_allow_html=True,
        )

        for category, questions in PRESET_QUESTIONS.items():
            with st.expander(category, expanded=False):
                for q in questions:
                    if st.button(
                        q,
                        key=f"preset_{hash(q)}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = q

    # ── Chat interface (main panel) ──
    with chat_col:
        # Init and trim chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        else:
            st.session_state.chat_messages = _trim_chat_history(st.session_state.chat_messages)

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Check for pending preset question
        prompt = None
        if "pending_question" in st.session_state:
            prompt = st.session_state.pop("pending_question")

        # Chat input
        if user_input := st.chat_input("Hỏi về dự án, phương pháp, kết quả..."):
            prompt = user_input

        if prompt:
            # ── Guardrails Validation ──
            from src.chatbot.guardrails import ChatGuardrails

            is_valid, block_reason = ChatGuardrails.validate_prompt(prompt)

            if not is_valid:
                # Add user message and block reason, trimmed
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                st.session_state.chat_messages.append({"role": "assistant", "content": block_reason})
                st.session_state.chat_messages = _trim_chat_history(st.session_state.chat_messages)

                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(block_reason)

                # Stop processing
                st.rerun()

            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.session_state.chat_messages = _trim_chat_history(st.session_state.chat_messages)
            with st.chat_message("user"):
                st.markdown(prompt)

            # RAG: retrieve context
            context = ""
            sources = []
            if kb:
                try:
                    results_rag = kb.search(prompt, n_results=5)
                    if results_rag:
                        context_parts = []
                        for r in results_rag:
                            context_parts.append(f"[Nguồn: {r['source']}]\n{r['content']}")
                            if r["source"] not in sources:
                                sources.append(r["source"])
                        context = "\n\n---\n\n".join(context_parts)
                except Exception as e:
                    st.caption(f"⚠️ RAG search error: {e}")

            # Generate response
            with st.chat_message("assistant"):
                from src.chatbot.llm_client import chat_stream

                # Build message history (last 6 messages for context)
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages[-6:]]

                # Get providers
                current_providers = detect_available_providers(
                    st.session_state.get("llm_provider_keys", {}),
                    primary_provider=st.session_state.get("primary_provider"),
                )

                response = st.write_stream(
                    chat_stream(
                        messages=history,
                        context=context,
                        providers=current_providers,
                    )
                )

                # Show sources
                if sources:
                    source_text = " • ".join([f"`{s}`" for s in sources[:3]])
                    st.caption(f"📎 Nguồn tham khảo: {source_text}")

            # Save assistant response
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.session_state.chat_messages = _trim_chat_history(st.session_state.chat_messages)

        # ── Chat controls ──
        if st.session_state.chat_messages and st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_messages = []
            st.rerun()


# Alias for backwards compatibility / modular imports
render_chat_page = page_ai_assistant

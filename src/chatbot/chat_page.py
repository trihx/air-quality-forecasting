"""
AI Assistant page for PM2.5 Forecasting Dashboard.

Provides a chat interface with:
  - Multi-LLM support (Gemini, OpenAI, Groq, LM Studio)
  - Tiered fallback (Cloud API → Local LLM)
  - RAG-based context retrieval from project knowledge base
  - Secure API key management via session state
"""

import streamlit as st

from src.chatbot.provider_config import (
    LOCAL_MODEL_RECOMMENDATIONS,
    PROVIDER_REGISTRY,
    mask_api_key,
    validate_provider_connection,
    get_provider_from_registry,
    detect_available_providers,
)

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
    when users edit info cards via the Dashboard UI. This ensures
    the chatbot always has the latest user-curated knowledge.
    """
    from src.chatbot.knowledge_base import REINDEX_FLAG_PATH

    # Check if user content was updated (flag set by content API)
    needs_reindex = REINDEX_FLAG_PATH.exists()

    if needs_reindex:
        with st.spinner("🔄 Cập nhật kiến thức mới từ nội dung đã chỉnh sửa..."):
            count = kb.build_index(force=True)
        try:
            REINDEX_FLAG_PATH.unlink()
        except FileNotFoundError:
            pass
        st.toast("✅ Kiến thức chatbot đã được cập nhật!", icon="🧠")
        return count

    if not kb.is_indexed():
        with st.spinner("🔄 Đang index tài liệu dự án lần đầu... (30-60 giây)"):
            count = kb.build_index()
        return count

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
        if provider_name == "lm_studio":
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

            col_save, col_test = st.columns(2)
            with col_save:
                if st.button("💾 Lưu", key=f"save_{provider_name}", use_container_width=True):
                    if key:
                        st.session_state.llm_provider_keys[provider_name] = {
                            "api_key": key,
                            "model": model,
                            "base_url": "",
                        }
                        st.success(f"✅ Đã lưu ({mask_api_key(key)})")
                    else:
                        # Clear provider
                        st.session_state.llm_provider_keys.pop(provider_name, None)
                        st.info("Đã xóa API key")

            with col_test:
                if st.button("🔍 Test", key=f"test_{provider_name}", use_container_width=True):
                    if key:
                        provider = get_provider_from_registry(provider_name, api_key=key, custom_model=model)
                        if provider:
                            with st.spinner("Đang kiểm tra..."):
                                ok, msg = validate_provider_connection(provider)
                            if ok:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                    else:
                        st.warning("Chưa nhập API key")

    # ── LM Studio (Local) ──
    with st.sidebar.expander("🖥️ LM Studio (Local)", expanded=False):
        st.caption(PROVIDER_REGISTRY["lm_studio"]["description"])

        lm_url = st.text_input(
            "Server URL",
            key="input_lm_studio_url",
            value=st.session_state.llm_provider_keys.get("lm_studio", {}).get(
                "base_url",
                PROVIDER_REGISTRY["lm_studio"]["base_url"],
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
                    f"**{m['name']}{star}**\n"
                    f"- VRAM: {m['vram']} | {m['vietnamese']}\n"
                    f"- {m['best_for']}",
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
    else:
        st.sidebar.caption("🖥️ LM Studio: Fallback (auto-detect)")


def page_ai_assistant(results):
    """Render AI Assistant chatbot page."""

    # ── Render provider config in sidebar ──
    _render_provider_config()

    # ── Header ──
    st.markdown(
        """
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        💬 Trợ Lý AI — Hỏi Đáp Dự Án
    </h1>
    <p style="opacity: 0.7; font-size: 1.05rem; margin-bottom: 1rem;">
        Hỏi bất kỳ câu hỏi nào về dự án • Hỗ trợ chuẩn bị phản biện luận văn
    </p>
    """,
        unsafe_allow_html=True,
    )

    # ── Version-aware info cards ──
    from src.info_cards import cards_ai_assistant, get_current_version, render_version_badge
    ver = get_current_version()
    render_version_badge(ver)
    cards_ai_assistant(ver)

    # ── Active provider status ──
    providers = detect_available_providers(
        st.session_state.get("llm_provider_keys", {})
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
            st.warning(
                "⚠️ Chưa cấu hình AI — vào **⚙️ Cấu Hình AI Provider** ở sidebar"
            )

    with col_info:
        # Show fallback chain
        chain = " → ".join([p.display_name for p in providers]) or "Chưa cấu hình"
        st.caption(f"🔗 Fallback: {chain}")

    # ── Knowledge Base status ──
    kb_col, reindex_col = st.columns([3, 1])
    try:
        kb = _get_knowledge_base()
        doc_count = _ensure_index(kb)
        with kb_col:
            st.caption(f"📚 Knowledge Base: {doc_count} documents indexed")
        with reindex_col:
            if st.button("🔄 Re-index", help="Cập nhật lại dữ liệu cho chatbot"):
                with st.spinner("🔄 Đang re-index toàn bộ tài liệu..."):
                    new_count = kb.build_index(force=True)
                st.success(f"✅ Đã index lại {new_count} documents!")
                st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Knowledge Base chưa sẵn sàng: {e}")
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
            🎓 Câu Hỏi Phản Biện Gợi Ý
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
        # Init chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

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
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
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
                    st.session_state.get("llm_provider_keys", {})
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

        # ── Chat controls ──
        if st.session_state.chat_messages and st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_messages = []
            st.rerun()

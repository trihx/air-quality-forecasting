"""
AI Assistant page for PM2.5 Forecasting Dashboard.

Provides a chat interface connected to local LLM via LM Studio,
with RAG-based context retrieval from project knowledge base.
"""

import streamlit as st

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
    """Ensure knowledge base is indexed, return doc count."""
    if not kb.is_indexed():
        with st.spinner("🔄 Đang index tài liệu dự án lần đầu... (30-60 giây)"):
            count = kb.build_index()
        return count
    return kb.index_count()


def page_ai_assistant(results):
    """Render AI Assistant chatbot page."""

    # ── Header ──
    st.markdown("""
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        💬 Trợ Lý AI — Hỏi Đáp Dự Án
    </h1>
    <p style="color: #8B95A5; font-size: 1.05rem; margin-bottom: 1rem;">
        Hỏi bất kỳ câu hỏi nào về dự án • Hỗ trợ chuẩn bị phản biện luận văn
    </p>
    """, unsafe_allow_html=True)

    # ── Connection status ──
    from src.chatbot.llm_client import check_connection, get_available_models

    col_status, col_model = st.columns([1, 1])

    with col_status:
        connected = check_connection()
        if connected:
            st.success("✅ LM Studio đang hoạt động")
        else:
            st.error(
                "❌ Không kết nối được LM Studio\n\n"
                "**Hướng dẫn:**\n"
                "1. Mở LM Studio\n"
                "2. Load model (Gemma 4 E4B)\n"
                "3. Bật Server → port 8888\n"
                "4. Reload trang này"
            )
            return

    with col_model:
        models = get_available_models()
        if models:
            st.info(f"🤖 Model: **{models[0]}**")
        else:
            st.warning("⚠️ Chưa load model trong LM Studio")

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
        st.markdown("""
        <div style="font-size: 0.85rem; font-weight: 700; color: #CBD5E0;
                    margin-bottom: 0.75rem;">
            🎓 Câu Hỏi Phản Biện Gợi Ý
        </div>
        """, unsafe_allow_html=True)

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
        if user_input := st.chat_input(
            "Hỏi về dự án, phương pháp, kết quả..."
        ):
            prompt = user_input

        if prompt:
            # Add user message
            st.session_state.chat_messages.append(
                {"role": "user", "content": prompt}
            )
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
                            context_parts.append(
                                f"[Nguồn: {r['source']}]\n{r['content']}"
                            )
                            if r["source"] not in sources:
                                sources.append(r["source"])
                        context = "\n\n---\n\n".join(context_parts)
                except Exception as e:
                    st.caption(f"⚠️ RAG search error: {e}")

            # Generate response
            with st.chat_message("assistant"):
                from src.chatbot.llm_client import chat_stream

                # Build message history (last 6 messages for context)
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages[-6:]
                ]

                response = st.write_stream(
                    chat_stream(messages=history, context=context)
                )

                # Show sources
                if sources:
                    source_text = " • ".join(
                        [f"`{s}`" for s in sources[:3]]
                    )
                    st.caption(f"📎 Nguồn tham khảo: {source_text}")

            # Save assistant response
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response}
            )

        # ── Chat controls ──
        if st.session_state.chat_messages:
            if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
                st.session_state.chat_messages = []
                st.rerun()

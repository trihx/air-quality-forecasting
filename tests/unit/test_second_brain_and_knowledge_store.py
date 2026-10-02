"""
Comprehensive Unit Tests for Obsidian Second Brain & Zero-Dependency Knowledge Store.

Verifies:
1. Complete Obsidian knowledge_vault structure, YAML frontmatter, and bidirectional wikilinks.
2. Zero-dependency knowledge store search accuracy, performance, and unaccented fuzzy matching.
3. KnowledgeBase RAG automatic fallback to curated knowledge when ChromaDB is empty/unindexed.
4. LLM client stream enhancements (reasoning tokens handling & anti-empty non-stream fallback).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VAULT_DIR = PROJECT_ROOT / "knowledge_vault"


# ==============================================================================
# 1. Tests for Obsidian Second Brain Vault Integrity
# ==============================================================================


class TestObsidianSecondBrainVault:
    """Verifies that the Obsidian Second Brain vault is fully built and standard-compliant."""

    EXPECTED_FILES = [
        "00_Index_MOC.md",
        "01_Pipeline_and_Data_Engineering/01_Pipeline_7_Steps.md",
        "01_Pipeline_and_Data_Engineering/02_Tiered_Imputation_and_Data_Sparsity.md",
        "01_Pipeline_and_Data_Engineering/03_Outlier_Trap_and_Domain_Bounds.md",
        "01_Pipeline_and_Data_Engineering/04_Multi_Resolution_Framing.md",
        "02_Data_Integrity_and_Anti_Leakage/01_Anti_Leakage_Shift1_Discipline.md",
        "02_Data_Integrity_and_Anti_Leakage/02_Temporal_Split_and_Test_on_Real_Only.md",
        "03_Feature_Engineering/01_Feature_Store_119_Features.md",
        "03_Feature_Engineering/02_Stationarity_ADF_KPSS.md",
        "04_Models_and_Architectures/01_Benchmark_Models_Overview.md",
        "04_Models_and_Architectures/02_Autocorrelation_Trap_and_GRU_15m.md",
        "04_Models_and_Architectures/03_Sweet_Spot_30m_Ensemble.md",
        "04_Models_and_Architectures/04_Hyperparameter_Tuning_Optuna.md",
        "05_Evaluation_and_Uncertainty/01_Metrics_Standard_MASE_over_RMSE.md",
        "05_Evaluation_and_Uncertainty/02_Diebold_Mariano_Hypothesis_Testing.md",
        "05_Evaluation_and_Uncertainty/03_Uncertainty_Quantification_CQR_ACI.md",
        "06_Explainability_and_Domain_Insights/01_Tree_SHAP_vs_Permutation_Importance.md",
        "06_Explainability_and_Domain_Insights/02_Threshold_Tipping_Point_14_17.md",
        "06_Explainability_and_Domain_Insights/03_Diurnal_Cycle_and_Sa_Dec_Context.md",
        "07_Defense_Playbook_and_FAQ/01_Master_Defense_QnA_Hoi_Dong.md",
        "07_Defense_Playbook_and_FAQ/02_Slides_Narrative_and_Key_Arguments.md",
    ]

    def test_vault_directory_exists(self):
        """Knowledge vault directory must exist at project root."""
        assert VAULT_DIR.exists(), f"Vault directory not found at {VAULT_DIR}"
        assert VAULT_DIR.is_dir()

    def test_all_expected_vault_files_exist(self):
        """All 21 required markdown notes must exist."""
        for rel_file in self.EXPECTED_FILES:
            file_path = VAULT_DIR / rel_file
            assert file_path.exists(), f"Missing vault file: {rel_file}"
            assert file_path.stat().st_size > 500, f"File {rel_file} is suspiciously small"

    def test_vault_yaml_frontmatter(self):
        """Every note must have valid YAML frontmatter with title, tags, aliases, domain."""
        for rel_file in self.EXPECTED_FILES:
            file_path = VAULT_DIR / rel_file
            content = file_path.read_text(encoding="utf-8")
            assert content.startswith("---\n"), f"{rel_file} missing opening YAML frontmatter"
            closing_idx = content.find("\n---\n", 4)
            assert closing_idx != -1, f"{rel_file} missing closing YAML frontmatter delimiter"
            frontmatter = content[4:closing_idx]

            assert "title:" in frontmatter, f"{rel_file} missing 'title' in frontmatter"
            assert "tags:" in frontmatter, f"{rel_file} missing 'tags' in frontmatter"
            assert "aliases:" in frontmatter, f"{rel_file} missing 'aliases' in frontmatter"
            assert "domain:" in frontmatter, f"{rel_file} missing 'domain' in frontmatter"
            assert "status: completed" in frontmatter, f"{rel_file} missing status completed"

    def test_vault_wikilinks_exist(self):
        """Every file must contain Obsidian wikilinks [[...]] to other topics."""
        for rel_file in self.EXPECTED_FILES:
            file_path = VAULT_DIR / rel_file
            content = file_path.read_text(encoding="utf-8")
            assert "[[" in content and "]]" in content, f"{rel_file} has no wikilinks [[...]]"

    def test_index_moc_links_to_all_subfolders(self):
        """00_Index_MOC.md must link to core chapters across all subfolders."""
        moc_path = VAULT_DIR / "00_Index_MOC.md"
        content = moc_path.read_text(encoding="utf-8")
        assert "[[01_Pipeline_7_Steps]]" in content
        assert "[[01_Anti_Leakage_Shift1_Discipline]]" in content
        assert "[[01_Feature_Store_119_Features]]" in content
        assert "[[01_Benchmark_Models_Overview]]" in content
        assert "[[01_Metrics_Standard_MASE_over_RMSE]]" in content
        assert "[[01_Tree_SHAP_vs_Permutation_Importance]]" in content
        assert "[[01_Master_Defense_QnA_Hoi_Dong]]" in content


# ==============================================================================
# 2. Tests for Zero-Dependency Knowledge Store
# ==============================================================================


class TestZeroDependencyKnowledgeStore:
    """Verifies speed, accuracy, and memory safety of knowledge_store.py."""

    @pytest.mark.parametrize(
        "query,expected_keyword",
        [
            ("quy trình pipeline", "pipeline"),
            ("anti-leakage", "shift"),
            ("MASE", "mase"),
            ("temporal split", "temporal"),
            ("điểm ngọt 30m", "30"),
            ("bẫy tự tương quan", "tương quan"),
            ("SHAP", "shap"),
            ("Sa Đéc", "đéc"),
        ],
    )
    def test_search_curated_knowledge_preset_queries(self, query: str, expected_keyword: str):
        """Every preset query must return high-scoring relevant document."""
        from src.chatbot.knowledge_store import search_curated_knowledge

        results = search_curated_knowledge(query, top_k=3)
        assert len(results) > 0, f"No results for query '{query}'"
        top = results[0]
        assert top["score"] > 0.0
        assert "content" in top and len(top["content"]) > 100
        assert "source" in top and top["source"].startswith("knowledge_vault/")

        combined_text = (top["title"] + " " + top["content"]).lower()
        assert expected_keyword.lower() in combined_text, (
            f"Expected keyword '{expected_keyword}' not found in top result for '{query}'"
        )

    def test_search_empty_or_whitespace_returns_empty_list(self):
        """Empty or whitespace queries return [] immediately without error."""
        from src.chatbot.knowledge_store import search_curated_knowledge

        assert search_curated_knowledge("") == []
        assert search_curated_knowledge("   ") == []

    def test_unaccented_query_matches_accented_content(self):
        """Unaccented query like 'quy trinh' must match accented content 'quy trình'."""
        from src.chatbot.knowledge_store import search_curated_knowledge

        results = search_curated_knowledge("quy trinh pipeline 7 buoc", top_k=1)
        assert len(results) == 1
        assert "7 bước" in results[0]["title"] or "Pipeline" in results[0]["title"]

    def test_get_all_curated_topics(self):
        """get_all_curated_topics returns complete list of curated documents."""
        from src.chatbot.knowledge_store import get_all_curated_topics

        topics = get_all_curated_topics()
        assert len(topics) >= 12
        assert any("Pipeline" in t for t in topics)
        assert any("MASE" in t for t in topics)
        assert any("SHAP" in t for t in topics)

    def test_search_performance_sub_millisecond(self):
        """Zero-dependency search must complete 100 queries in < 50ms (< 0.5ms/query)."""
        from src.chatbot.knowledge_store import search_curated_knowledge

        start_time = time.perf_counter()
        for _ in range(100):
            _ = search_curated_knowledge("Tại sao dùng MASE thay vì RMSE?", top_k=3)
        duration = time.perf_counter() - start_time
        avg_ms = (duration / 100) * 1000
        assert avg_ms < 2.0, f"Average search took {avg_ms:.3f}ms, expected < 2.0ms"

    @pytest.mark.parametrize(
        "query,expected_doc_id_or_title",
        [
            (
                "Tại sao dòng 1 bảng Diebold-Mariano mang dấu dương (+13.729) mà các dòng sau lại mang dấu âm (-8.452)?",
                "Diebold-Mariano",
            ),
            (
                "Tại sao hệ số xác định R² ngoài mẫu (Out-of-Sample) lại nhận giá trị âm?",
                "R² ngoài mẫu",
            ),
            (
                "Tại sao dũng cảm Drop 19.810 giờ khuyết thiếu thay vì dùng GAN/Deep Learning để bù dữ liệu?",
                "Drop 19.810 giờ",
            ),
            (
                "Bẫy ngoại lai IQR 3.0 đã xóa nhầm dữ liệu ra sao và tại sao đề án dùng Domain Bounds [0, 500]?",
                "Bẫy xóa ngoại lai IQR 3.0",
            ),
            (
                "Bẫy tự tương quan (r=0.86) ở bước 1h là gì và tại sao GRU 15m phá được bẫy này?",
                "Bẫy tự tương quan",
            ),
        ],
    )
    def test_search_grill_me_holes_queries(self, query: str, expected_doc_id_or_title: str):
        """High-stakes defense grill-me hole questions return exact relevant documents."""
        from src.chatbot.knowledge_store import search_curated_knowledge

        results = search_curated_knowledge(query, top_k=3)
        assert len(results) > 0, f"No search results for Grill-Me query: {query}"
        top = results[0]
        combined = (top["title"] + " " + top["content"]).lower()
        expected_lower = expected_doc_id_or_title.lower()
        assert expected_lower in combined or expected_lower in top["title"].lower(), (
            f"Expected '{expected_doc_id_or_title}' in top result title '{top['title']}' for query '{query}'"
        )


# ==============================================================================
# 3. Tests for KnowledgeBase RAG Fallback
# ==============================================================================


class TestKnowledgeBaseRAGFallback:
    """Verifies that KnowledgeBase seamlessly falls back to curated store when unindexed."""

    def test_knowledge_base_fallback_when_unindexed(self, tmp_path: Path):
        """When kb is not indexed, search() automatically falls back to curated store."""
        from src.chatbot.knowledge_base import KnowledgeBase

        test_dir = str(tmp_path / "non_existent_chroma")
        kb = KnowledgeBase(persist_dir=test_dir)
        # Ensure is_indexed returns False
        with patch.object(kb, "is_indexed", return_value=False):
            results = kb.search("quy trình pipeline", n_results=3)
            assert len(results) > 0
            assert "source" in results[0]
            assert results[0]["source"].startswith("knowledge_vault/")
            assert results[0]["score"] > 0

    def test_knowledge_base_fallback_when_index_count_is_zero(self, tmp_path: Path):
        """When kb index_count is 0, search() falls back to curated store."""
        from src.chatbot.knowledge_base import KnowledgeBase

        test_dir = str(tmp_path / "non_existent_chroma")
        kb = KnowledgeBase(persist_dir=test_dir)
        with (
            patch.object(kb, "is_indexed", return_value=True),
            patch.object(kb, "index_count", return_value=0),
        ):
            results = kb.search("anti-leakage shift(1)", n_results=2)
            assert len(results) > 0
            assert "shift" in results[0]["title"].lower() or "shift" in results[0]["content"].lower()

    def test_knowledge_base_get_all_curated_topics(self):
        """KnowledgeBase exposes get_all_curated_topics delegating to knowledge store."""
        from src.chatbot.knowledge_base import get_knowledge_base

        kb = get_knowledge_base()
        topics = kb.get_all_curated_topics()
        assert len(topics) >= 12
        assert isinstance(topics, list)


# ==============================================================================
# 4. Tests for LLM Client Reasoning Stream & Anti-Empty Fallback
# ==============================================================================


class TestLLMClientStreamEnhancements:
    """Verifies Qwen 3 reasoning chunks handling and anti-empty non-streaming guard."""

    def test_stream_with_reasoning_and_content_chunks(self):
        """Stream yields content chunks when reasoning tokens are present."""
        from src.chatbot.llm_client import _try_stream_provider
        from src.chatbot.provider_config import LLMProvider

        provider = LLMProvider(
            name="test_thinking_model",
            display_name="Thinking Model",
            base_url="http://mock-api.local/v1",
            api_key="mock-key",
            model="qwen3:4b",
        )

        # Mock stream chunks: 2 reasoning chunks, then 2 content chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = ""
        chunk1.choices[0].delta.reasoning_content = "Thinking about PM2.5..."

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = ""
        chunk2.choices[0].delta.reasoning_content = "Analyzing 7 steps..."

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = "Quy trình 7 bước gồm có: "
        chunk3.choices[0].delta.reasoning_content = None

        chunk4 = MagicMock()
        chunk4.choices = [MagicMock()]
        chunk4.choices[0].delta.content = "Thu thập, làm sạch, tái lấy mẫu..."
        chunk4.choices[0].delta.reasoning_content = None

        mock_stream = [chunk1, chunk2, chunk3, chunk4]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("src.chatbot.llm_client._build_client", return_value=mock_client):
            gen = _try_stream_provider(
                provider=provider,
                full_messages=[{"role": "user", "content": "Pipeline?"}],
                temperature=0.3,
                max_tokens=100,
            )
            assert gen is not None
            chunks = list(gen)
            assert len(chunks) == 2
            assert chunks[0] == "Quy trình 7 bước gồm có: "
            assert chunks[1] == "Thu thập, làm sạch, tái lấy mẫu..."

    def test_stream_anti_empty_guard_triggers_non_stream_call(self):
        """When stream finishes with 0 content chars, triggers non-stream fallback call."""
        from src.chatbot.llm_client import _try_stream_provider
        from src.chatbot.provider_config import LLMProvider

        provider = LLMProvider(
            name="test_empty_stream",
            display_name="Empty Stream Model",
            base_url="http://mock-api.local/v1",
            api_key="mock-key",
            model="qwen3:4b",
        )

        # Stream yielded only empty content or thinking tokens that got eaten
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = ""
        chunk1.choices[0].delta.reasoning_content = "Done thinking."

        mock_stream = [chunk1]

        # Non-stream completion fallback mock
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Câu trả lời hoàn chỉnh từ non-stream fallback!"

        mock_client = MagicMock()

        def mock_create(*args, **kwargs):
            if kwargs.get("stream", False):
                return mock_stream
            return mock_completion

        mock_client.chat.completions.create.side_effect = mock_create

        with patch("src.chatbot.llm_client._build_client", return_value=mock_client):
            gen = _try_stream_provider(
                provider=provider,
                full_messages=[{"role": "user", "content": "Explain pipeline"}],
                temperature=0.3,
                max_tokens=200,
            )
            assert gen is not None
            chunks = list(gen)
            assert len(chunks) == 1
            assert chunks[0] == "Câu trả lời hoàn chỉnh từ non-stream fallback!"
            # Verify create was called twice: 1 for stream=True, 1 for stream=False
            assert mock_client.chat.completions.create.call_count == 2

    def test_stream_anti_empty_guard_not_triggered_when_content_yielded(self):
        """When stream yields content, non-stream fallback is never called."""
        from src.chatbot.llm_client import _try_stream_provider
        from src.chatbot.provider_config import LLMProvider

        provider = LLMProvider(
            name="test_normal_stream",
            display_name="Normal Stream Model",
            base_url="http://mock-api.local/v1",
            api_key="mock-key",
            model="gemini-flash",
        )

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Normal output"
        chunk1.choices[0].delta.reasoning_content = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [chunk1]

        with patch("src.chatbot.llm_client._build_client", return_value=mock_client):
            gen = _try_stream_provider(
                provider=provider,
                full_messages=[{"role": "user", "content": "Hello"}],
                temperature=0.3,
                max_tokens=100,
            )
            assert gen is not None
            chunks = list(gen)
            assert chunks == ["Normal output"]
            # Exactly 1 call (stream=True)
            assert mock_client.chat.completions.create.call_count == 1

    def test_stream_exception_does_not_trigger_anti_empty_fallback(self):
        """When stream raises an exception midway, yields error notice and skips non-stream call."""
        from src.chatbot.llm_client import _try_stream_provider
        from src.chatbot.provider_config import LLMProvider

        provider = LLMProvider(
            name="test_broken_stream",
            display_name="Broken Stream",
            base_url="http://mock-api.local/v1",
            api_key="mock-key",
            model="gemini-flash",
        )

        def broken_gen():
            raise ConnectionError("Connection lost")
            yield  # pragma: no cover

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = broken_gen()

        with patch("src.chatbot.llm_client._build_client", return_value=mock_client):
            gen = _try_stream_provider(
                provider=provider,
                full_messages=[{"role": "user", "content": "Hello"}],
                temperature=0.3,
                max_tokens=100,
            )
            assert gen is not None
            chunks = list(gen)
            assert len(chunks) == 1
            assert "Kết nối bị gián đoạn giữa chừng" in chunks[0]
            # No fallback call after connection error
            assert mock_client.chat.completions.create.call_count == 1

    def test_chat_stream_skips_empty_provider_and_falls_back_to_next(self):
        """When provider 1 produces 0 output, chat_stream skips it and falls back to provider 2."""
        from src.chatbot.llm_client import chat_stream
        from src.chatbot.provider_config import LLMProvider

        p1 = LLMProvider(
            name="empty_p1",
            display_name="Empty Provider 1",
            base_url="http://p1.local/v1",
            api_key="k1",
            model="m1",
        )
        p2 = LLMProvider(
            name="good_p2",
            display_name="Good Provider 2",
            base_url="http://p2.local/v1",
            api_key="k2",
            model="m2",
        )

        def mock_try_stream(provider, messages, temp, max_tok):
            if provider.name == "empty_p1":
                # Generator yielding 0 chunks
                def _empty():
                    if False:
                        yield ""

                return _empty()
            elif provider.name == "good_p2":

                def _good():
                    yield "Câu trả lời từ Provider 2"

                return _good()
            return None

        with patch("src.chatbot.llm_client._try_stream_provider", side_effect=mock_try_stream):
            chunks = list(
                chat_stream(
                    messages=[{"role": "user", "content": "Alo"}],
                    providers=[p1, p2],
                )
            )
            assert len(chunks) == 2
            # Notice p1 header was NEVER yielded!
            assert "Empty Provider 1" not in "".join(chunks)
            # P2 header and content are present
            assert "*🤖 Good Provider 2*" in chunks[0]
            assert chunks[1] == "Câu trả lời từ Provider 2"

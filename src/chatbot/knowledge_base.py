"""
RAG Knowledge Base for PM2.5 Project AI Assistant.

Indexes project documentation into ChromaDB for context-aware Q&A.
Uses sentence-transformers for embedding, ChromaDB for vector storage.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root (src/chatbot/knowledge_base.py → chatbot → src → project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Knowledge sources — ordered by importance
KNOWLEDGE_SOURCES = [
    # Core project docs
    ("docs/PROJECT_WALKTHROUGH.md", "walkthrough"),
    ("docs/THESIS_DRAFT_CTU_1799.md", "thesis"),
    ("README.md", "readme"),
    # Agent memory (architecture, decisions, lessons)
    (".agent/memory/CONTEXT.md", "context"),
    (".agent/memory/DECISIONS.md", "decisions"),
    (".agent/memory/LESSONS_LEARNED.md", "lessons"),
    (".agent/memory/RUNS_LOG.md", "runs_log"),
    # Guides
    (".agent/guides/evaluation-metrics.md", "eval_guide"),
    (".agent/guides/model-training.md", "training_guide"),
    (".agent/guides/data-engineering.md", "data_guide"),
    (".agent/SKILL.md", "skill"),
]

# JSON experiment results
EXPERIMENT_DIRS = [
    "research/experiments/baselines",
    "research/experiments/arima",
    "research/experiments/ml_models",
    "research/experiments/dl",
    "research/experiments/tft",
    "research/experiments/ensemble",
    "research/experiments/multi_horizon",
    "research/experiments/tuning",
    "research/experiments/prediction_intervals",
]

# ChromaDB path
CHROMA_DIR = PROJECT_ROOT / ".chroma_db"
REINDEX_FLAG_PATH = CHROMA_DIR / ".needs_reindex"


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _load_markdown_docs() -> list[dict]:
    """Load all markdown knowledge sources."""
    docs = []
    for rel_path, source_type in KNOWLEDGE_SOURCES:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            try:
                content = full_path.read_text(encoding="utf-8")
                chunks = _chunk_text(content)
                for i, chunk in enumerate(chunks):
                    docs.append(
                        {
                            "content": chunk,
                            "metadata": {
                                "source": rel_path,
                                "type": source_type,
                                "chunk_index": i,
                            },
                        }
                    )
                logger.info(f"Loaded {len(chunks)} chunks from {rel_path}")
            except Exception as e:
                logger.warning(f"Failed to load {rel_path}: {e}")
    return docs


def _load_experiment_results() -> list[dict]:
    """Load JSON experiment results as knowledge documents."""
    docs = []
    for dir_path in EXPERIMENT_DIRS:
        full_dir = PROJECT_ROOT / dir_path
        if not full_dir.exists():
            continue
        for json_file in full_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                # Convert JSON to readable text
                content = f"# Experiment: {json_file.stem}\n"
                content += f"Source: {json_file.relative_to(PROJECT_ROOT)}\n\n"
                content += json.dumps(data, indent=2, ensure_ascii=False)

                chunks = _chunk_text(content, chunk_size=1500, overlap=300)
                for i, chunk in enumerate(chunks):
                    docs.append(
                        {
                            "content": chunk,
                            "metadata": {
                                "source": str(json_file.relative_to(PROJECT_ROOT)),
                                "type": "experiment",
                                "chunk_index": i,
                            },
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    # Also load standardized metrics
    std_metrics = PROJECT_ROOT / "research/experiments/standardized_metrics.json"
    if std_metrics.exists():
        try:
            data = json.loads(std_metrics.read_text(encoding="utf-8"))
            content = "# Standardized Metrics (All Models)\n\n"
            content += json.dumps(data, indent=2, ensure_ascii=False)
            chunks = _chunk_text(content, chunk_size=2000, overlap=400)
            for i, chunk in enumerate(chunks):
                docs.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "source": "research/experiments/standardized_metrics.json",
                            "type": "metrics",
                            "chunk_index": i,
                        },
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to load standardized_metrics.json: {e}")

    # Best model configs
    best_cfg = PROJECT_ROOT / "research/best_models_configs.json"
    if best_cfg.exists():
        try:
            data = json.loads(best_cfg.read_text(encoding="utf-8"))
            content = "# Best Model Configurations\n\n"
            content += json.dumps(data, indent=2, ensure_ascii=False)
            docs.append(
                {
                    "content": content,
                    "metadata": {
                        "source": "research/best_models_configs.json",
                        "type": "config",
                        "chunk_index": 0,
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load best_models_configs.json: {e}")

    return docs


def _load_info_cards_from_db() -> list[dict]:
    """Load user-curated info cards for RAG indexing.

    3-tier fallback:
        Tier 1: PostgreSQL via API (Docker)
        Tier 2: JSON export file (local dev)
        Tier 3: Empty list (graceful degradation)
    """
    cards = None

    # Tier 1: API (PostgreSQL)
    try:
        from src.frontend.api_client import APIClient

        client = APIClient()
        result = client.get_info_cards()
        if isinstance(result, list) and len(result) > 0:
            cards = result
    except Exception as e:
        logger.warning(f"Failed to load info cards from API: {e}")

    # Tier 2: JSON export file
    if cards is None:
        json_path = PROJECT_ROOT / "research" / "experiments" / "db_export" / "info_cards.json"
        if json_path.exists():
            try:
                import json as _json

                data = _json.loads(json_path.read_text(encoding="utf-8"))
                cards = [
                    {
                        "card_key": key,
                        "title": val.get("title", ""),
                        "content": val.get("content", ""),
                        "page": val.get("page", "unknown"),
                    }
                    for key, val in data.items()
                ]
                logger.info(f"Loaded {len(cards)} info cards from JSON export (fallback)")
            except Exception as e:
                logger.warning(f"Failed to load info cards from JSON export: {e}")

    # Tier 3: Empty
    if not cards:
        return []

    docs = []
    for card in cards:
        title = card.get("title", "")
        content_text = card.get("content", "")
        if not content_text.strip():
            continue
        full_text = f"# {title}\n\n{content_text}"
        chunks = _chunk_text(full_text, chunk_size=1200, overlap=200)
        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source": f"info_card:{card.get('card_key', 'unknown')}",
                        "type": "user_curated",
                        "page": card.get("page", "unknown"),
                        "chunk_index": i,
                    },
                }
            )
    logger.info(f"Loaded {len(docs)} chunks from {len(cards)} info cards")
    return docs


def _load_dashboard_content_json() -> list[dict]:
    """Load structured dashboard content JSON as knowledge documents."""
    json_path = PROJECT_ROOT / "research" / "experiments" / "dashboard_content.json"
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        content = "# Dashboard Content (Structured)\n\n"
        content += json.dumps(data, indent=2, ensure_ascii=False)
        chunks = _chunk_text(content, chunk_size=1500, overlap=300)
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source": "research/experiments/dashboard_content.json",
                        "type": "dashboard_content",
                        "chunk_index": i,
                    },
                }
            )
        logger.info(f"Loaded {len(docs)} chunks from dashboard_content.json")
        return docs
    except Exception as e:
        logger.warning(f"Failed to load dashboard_content.json: {e}")
        return []


class KnowledgeBase:
    """Vector-based knowledge base using ChromaDB + sentence-transformers."""

    def __init__(self, persist_dir: str | None = None):
        self._persist_dir = persist_dir or str(CHROMA_DIR)
        self._collection = None
        self._client = None

    def _get_collection(self):
        """Lazy-init ChromaDB collection."""
        if self._collection is None:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
            )

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="pm25_knowledge",
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def is_indexed(self) -> bool:
        """Check if knowledge base has been indexed."""
        try:
            collection = self._get_collection()
            return collection.count() > 0
        except Exception:
            return False

    def index_count(self) -> int:
        """Get number of indexed documents."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def build_index(self, force: bool = False) -> int:
        """
        Build vector index from project documents.

        Returns number of documents indexed.
        """
        collection = self._get_collection()

        if collection.count() > 0 and not force:
            logger.info(f"Index already exists with {collection.count()} docs. Use force=True to rebuild.")
            return collection.count()

        # Clear existing
        if force and collection.count() > 0:
            self._client.delete_collection("pm25_knowledge")
            self._collection = None  # Reset reference
            collection = self._get_collection()

        # Load all knowledge
        all_docs = (
            _load_markdown_docs()
            + _load_experiment_results()
            + _load_info_cards_from_db()
            + _load_dashboard_content_json()
        )

        if not all_docs:
            logger.warning("No documents found to index!")
            return 0

        # Batch insert (ChromaDB limit)
        batch_size = 100
        total = 0
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i : i + batch_size]
            collection.add(
                ids=[f"doc_{i + j}" for j in range(len(batch))],
                documents=[d["content"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )
            total += len(batch)

        logger.info(f"Indexed {total} document chunks into ChromaDB")
        return total

    def search(self, query: str, n_results: int = 8) -> list[dict]:
        """
        Search for relevant context given a query.

        Returns list of {content, source, score} dicts.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

        docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                score = 1 - distance  # cosine similarity
                # Only include docs with reasonable similarity
                if score > 0.1:
                    docs.append(
                        {
                            "content": doc,
                            "source": meta.get("source", "unknown"),
                            "type": meta.get("type", "unknown"),
                            "score": score,
                        }
                    )
                    logger.debug(f"RAG match: score={score:.3f} src={meta.get('source')}")
        return docs


# Singleton instance
_kb_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create singleton knowledge base."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance

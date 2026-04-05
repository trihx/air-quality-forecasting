"""
RAG Knowledge Base for PM2.5 Project AI Assistant.

Indexes project documentation into ChromaDB for context-aware Q&A.
Uses sentence-transformers for embedding, ChromaDB for vector storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
                    docs.append({
                        "content": chunk,
                        "metadata": {
                            "source": rel_path,
                            "type": source_type,
                            "chunk_index": i,
                        },
                    })
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
                    docs.append({
                        "content": chunk,
                        "metadata": {
                            "source": str(json_file.relative_to(PROJECT_ROOT)),
                            "type": "experiment",
                            "chunk_index": i,
                        },
                    })
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
                docs.append({
                    "content": chunk,
                    "metadata": {
                        "source": "research/experiments/standardized_metrics.json",
                        "type": "metrics",
                        "chunk_index": i,
                    },
                })
        except Exception as e:
            logger.warning(f"Failed to load standardized_metrics.json: {e}")

    # Best model configs
    best_cfg = PROJECT_ROOT / "research/best_models_configs.json"
    if best_cfg.exists():
        try:
            data = json.loads(best_cfg.read_text(encoding="utf-8"))
            content = "# Best Model Configurations\n\n"
            content += json.dumps(data, indent=2, ensure_ascii=False)
            docs.append({
                "content": content,
                "metadata": {
                    "source": "research/best_models_configs.json",
                    "type": "config",
                    "chunk_index": 0,
                },
            })
        except Exception as e:
            logger.warning(f"Failed to load best_models_configs.json: {e}")

    return docs


class KnowledgeBase:
    """Vector-based knowledge base using ChromaDB + sentence-transformers."""

    def __init__(self, persist_dir: Optional[str] = None):
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
                model_name="all-MiniLM-L6-v2",
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
            logger.info(
                f"Index already exists with {collection.count()} docs. "
                "Use force=True to rebuild."
            )
            return collection.count()

        # Clear existing
        if force and collection.count() > 0:
            self._client.delete_collection("pm25_knowledge")
            collection = self._get_collection()

        # Load all knowledge
        all_docs = _load_markdown_docs() + _load_experiment_results()

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

    def search(self, query: str, n_results: int = 5) -> list[dict]:
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
                distance = (
                    results["distances"][0][i] if results["distances"] else 1.0
                )
                docs.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "type": meta.get("type", "unknown"),
                    "score": 1 - distance,  # cosine similarity
                })
        return docs


# Singleton instance
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create singleton knowledge base."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance

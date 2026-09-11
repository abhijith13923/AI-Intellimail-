import logging
import pickle
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """
    In-memory BM25 keyword index for sparse retrieval.

    The index is rebuilt from scratch on every ingestion call, which is fine
    for in-memory use.  For persistence, call save() / load() with a file path.

    Documents are stored alongside their metadata so results mirror the format
    returned by ChromaDB, making fusion in the HybridRetriever straightforward.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        # Parallel list of {id, text, metadata} dicts — same index as BM25 corpus
        self._docs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase, simple whitespace tokenisation."""
        return text.lower().split()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Build (or rebuild) the BM25 index from a list of chunk dicts.

        Args:
            documents: List of {"id": str, "text": str, "metadata": dict}
        """
        if not documents:
            logger.warning("BM25Store.index_documents called with empty list.")
            return

        self._docs = documents
        corpus = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(documents)} documents.")

    def add_documents(self, new_documents: List[Dict[str, Any]]):
        """
        Incrementally add documents to the existing index.
        NOTE: BM25Okapi doesn't support true incremental updates, so we
        rebuild the index with the combined corpus. Acceptable for in-memory use.
        """
        combined = self._docs + new_documents
        self.index_documents(combined)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Optional[str],
        top_k: int = 10,
        user_email: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for a query, with optional
        metadata pre-filtering so BM25 candidates stay scoped.

        For metadata-only queries (query=None or empty), the function
        performs a pure metadata filter scan and returns up to top_k matches
        sorted by timestamp_epoch descending.

        Args:
            query:            Semantic/keyword query string (may be None).
            top_k:            Number of results to return.
            user_email:       Filter results to this user's chunks only.
            metadata_filters: Raw filter dict from QueryParser (same format
                              as passed to vector_store).

        Returns:
            List of {"id", "text", "metadata", "bm25_score"} dicts, sorted
            by descending BM25 score (or timestamp_epoch for metadata-only).
        """
        if self.bm25 is None or not self._docs:
            logger.warning("BM25 index is empty. Returning no results.")
            return []

        # ----------------------------------------------------------------
        # Pre-filter candidates by user and metadata
        # ----------------------------------------------------------------
        def _matches_filters(doc: Dict[str, Any]) -> bool:
            meta = doc.get("metadata", {})
            if user_email and meta.get("user_email") != user_email:
                return False
            if not metadata_filters:
                return True

            for key, value in metadata_filters.items():
                if value is None:
                    continue

                if key == "date" and isinstance(value, dict):
                    date_iso = meta.get("date_iso", "")
                    start = value.get("start")
                    end = value.get("end")
                    if start and date_iso and date_iso < start:
                        return False
                    if end and date_iso and date_iso > end:
                        return False

                elif key == "sender" and isinstance(value, str):
                    sender = (meta.get("sender", "") + " " + meta.get("sender_email", "")).lower()
                    if value.lower() not in sender:
                        return False

                elif key == "subject" and isinstance(value, str):
                    if value.lower() not in meta.get("subject", "").lower():
                        return False

                elif key in ("unread", "starred", "has_attachment") and isinstance(value, bool):
                    if meta.get(key) != value:
                        return False

                elif key == "labels" and isinstance(value, list):
                    stored_labels = meta.get("labels", "")
                    for label in value:
                        if label not in stored_labels:
                            return False

            return True

        filtered_docs = [doc for doc in self._docs if _matches_filters(doc)]

        if not filtered_docs:
            logger.info("BM25: no documents passed metadata filter.")
            return []

        # ----------------------------------------------------------------
        # Metadata-only path: no semantic query — return by recency
        # ----------------------------------------------------------------
        if not query or not query.strip():
            sorted_docs = sorted(
                filtered_docs,
                key=lambda d: d["metadata"].get("timestamp_epoch", 0),
                reverse=True,
            )[:top_k]
            return [
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "bm25_score": 0.0,
                }
                for doc in sorted_docs
            ]

        # ----------------------------------------------------------------
        # BM25 scoring on pre-filtered subset
        # ----------------------------------------------------------------
        # Build a local BM25 on the filtered subset for accurate scoring
        filtered_corpus = [self._tokenize(doc["text"]) for doc in filtered_docs]
        local_bm25 = BM25Okapi(filtered_corpus) if len(filtered_corpus) > 1 else self.bm25

        tokenized_query = self._tokenize(query)
        if len(filtered_docs) > 1:
            scores = local_bm25.get_scores(tokenized_query)
        else:
            scores = self.bm25.get_scores(tokenized_query)
            # Re-index into the filtered set
            doc_ids_set = {doc["id"] for doc in filtered_docs}
            scores = [
                score
                for score, doc in zip(scores, self._docs)
                if doc["id"] in doc_ids_set
            ]

        scored = sorted(
            zip(scores, filtered_docs),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "bm25_score": float(score),
            }
            for score, doc in scored
        ]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Serialize the index and document store to disk."""
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "docs": self._docs}, f)
        logger.info(f"BM25 index saved to {path}.")

    def load(self, path: str):
        """Restore the index from a previously saved file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self._docs = data["docs"]
        logger.info(f"BM25 index loaded from {path} ({len(self._docs)} docs).")

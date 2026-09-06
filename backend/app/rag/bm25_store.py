import logging
import pickle
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """
    In-memory BM25 keyword index for sparse retrieval.

    The index is rebuilt from scratch on every ingestion call, which is fine
    for in-memory use. For persistence, call save() / load() with a file path.

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
        query: str,
        top_k: int = 10,
        user_email: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query:      The semantic/keyword query string.
            top_k:      Number of results to return.
            user_email: If provided, filter results to this user's chunks only.

        Returns:
            List of {"id", "text", "metadata", "bm25_score"} dicts, sorted
            by descending BM25 score.
        """
        if self.bm25 is None or not self._docs:
            logger.warning("BM25 index is empty. Returning no results.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each doc with its score and optionally filter by user
        scored = []
        for idx, score in enumerate(scores):
            doc = self._docs[idx]
            if user_email and doc["metadata"].get("user_email") != user_email:
                continue
            scored.append((score, doc))

        # Sort descending, take top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, doc in top:
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "bm25_score": float(score),
            })

        return results

    # ------------------------------------------------------------------
    # Persistence helpers (optional)
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

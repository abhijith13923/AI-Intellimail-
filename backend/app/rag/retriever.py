import logging
from typing import List, Dict, Any, Optional

from app.rag.vector_store import ChromaDBStore
from app.rag.bm25_store import BM25Store
from app.config import settings

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    id_key: str = "id",
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) merges multiple ranked lists into one.

    Each document at rank r gets score  1 / (k + r).
    Scores are summed across lists. Documents from either list are included.

    Args:
        ranked_lists: A list of ranked result lists. Each item must have `id_key`.
        id_key:       The key used to uniquely identify documents.
        k:            RRF smoothing constant (default 60 is standard in literature).

    Returns:
        Combined list sorted by descending RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}   # id -> doc payload

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = []
    for doc_id in sorted_ids:
        doc = doc_store[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)

    return results


class HybridRetriever:
    """
    Combines dense (ChromaDB) and sparse (BM25) retrieval via Reciprocal Rank
    Fusion (RRF), then returns the top-K fused candidates for reranking.

    Multi-tenancy is enforced at both retrieval layers via `user_email`.
    """

    def __init__(self, vector_store: ChromaDBStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_chroma_results(chroma_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten the ChromaDB query response into a simple list of dicts."""
        docs = []
        if not chroma_results or not chroma_results.get("ids"):
            return docs

        ids = chroma_results["ids"][0]
        documents = chroma_results["documents"][0]
        metadatas = chroma_results["metadatas"][0]
        distances = chroma_results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            docs.append({
                "id": doc_id,
                "text": documents[i],
                "metadata": metadatas[i],
                "vector_distance": distances[i] if distances else None,
            })
        return docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        semantic_query: str,
        user_email: str,
        top_k: int = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs hybrid retrieval and returns `top_k` fused candidates.

        Steps:
          1. Semantic search  → ChromaDB (dense vectors).
          2. Keyword search   → BM25 (sparse term-matching).
          3. Fuse both lists  → Reciprocal Rank Fusion.
          4. Return top-K for downstream reranking.

        Args:
            semantic_query:   The cleaned user query (after metadata extraction).
            user_email:       Used to scope results to a single user.
            top_k:            How many fused candidates to return.
            metadata_filters: Optional ChromaDB-compatible metadata filter dict.

        Returns:
            List of fused candidate dicts, each with id, text, metadata, rrf_score.
        """
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        logger.info(
            f"HybridRetriever.retrieve | user={user_email} | query='{semantic_query}' | top_k={top_k}"
        )

        # 1. Dense retrieval — ChromaDB
        chroma_raw = self.vector_store.search(
            query=semantic_query,
            user_email=user_email,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        dense_results = self._normalise_chroma_results(chroma_raw)
        logger.info(f"Dense retrieval returned {len(dense_results)} results.")

        # 2. Sparse retrieval — BM25
        sparse_results = self.bm25_store.search(
            query=semantic_query,
            top_k=top_k,
            user_email=user_email,
        )
        logger.info(f"Sparse retrieval returned {len(sparse_results)} results.")

        # 3. Fuse via RRF
        fused = _reciprocal_rank_fusion([dense_results, sparse_results])
        logger.info(f"RRF fusion produced {len(fused)} unique candidates.")

        return fused[:top_k]

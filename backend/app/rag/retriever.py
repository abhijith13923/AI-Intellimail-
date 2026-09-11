import logging
import time
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

    for list_idx, ranked_list in enumerate(ranked_lists):
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
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        results.append(doc)

    return results


class HybridRetriever:
    """
    Combines dense (ChromaDB) and sparse (BM25) retrieval via Reciprocal Rank
    Fusion (RRF), then returns the top-K fused candidates for reranking.

    Multi-tenancy is enforced at both retrieval layers via `user_email`.

    Returns rich diagnostic telemetry alongside results so the pipeline
    and frontend can render the Live Chunk Inspector.
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
                "vector_distance": round(float(distances[i]), 6) if distances else None,
            })
        return docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        semantic_query: Optional[str],
        user_email: str,
        top_k: int = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        is_metadata_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs hybrid retrieval and returns fused candidates plus rich telemetry.

        Steps:
          1. Semantic search  → ChromaDB (dense vectors).
          2. Keyword search   → BM25 (sparse term-matching).
          3. Fuse both lists  → Reciprocal Rank Fusion.
          4. Return top-K + full diagnostic trace.

        Args:
            semantic_query:   The cleaned user query (after metadata extraction).
                              May be None for metadata-only queries.
            user_email:       Used to scope results to a single user.
            top_k:            How many fused candidates to return.
            metadata_filters: Raw filter dict from QueryParser.
            is_metadata_only: Hint that this is a pure metadata lookup —
                              adjust retrieval accordingly.

        Returns:
            Dict with keys:
              - candidates:     Top-K fused result dicts.
              - dense_results:  Raw ChromaDB hits (for inspector).
              - sparse_results: Raw BM25 hits (for inspector).
              - timings:        {"dense_ms", "sparse_ms", "fusion_ms"}
        """
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        logger.info(
            f"HybridRetriever.retrieve | user={user_email} | "
            f"query='{str(semantic_query)[:80]}' | metadata_only={is_metadata_only} | top_k={top_k}"
        )

        # 1. Dense retrieval — ChromaDB
        t0 = time.perf_counter()
        chroma_raw = self.vector_store.search(
            query=semantic_query,
            user_email=user_email,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        dense_results = self._normalise_chroma_results(chroma_raw)
        dense_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"Dense retrieval returned {len(dense_results)} results in {dense_ms}ms.")

        # 2. Sparse retrieval — BM25
        t1 = time.perf_counter()
        sparse_results = self.bm25_store.search(
            query=semantic_query,
            top_k=top_k,
            user_email=user_email,
            metadata_filters=metadata_filters,
        )
        sparse_ms = round((time.perf_counter() - t1) * 1000)
        logger.info(f"Sparse retrieval returned {len(sparse_results)} results in {sparse_ms}ms.")

        # 3. Fuse via RRF
        t2 = time.perf_counter()
        fused = _reciprocal_rank_fusion([dense_results, sparse_results])
        fusion_ms = round((time.perf_counter() - t2) * 1000)
        logger.info(f"RRF fusion produced {len(fused)} unique candidates in {fusion_ms}ms.")

        return {
            "candidates": fused[:top_k],
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "timings": {
                "dense_ms": dense_ms,
                "sparse_ms": sparse_ms,
                "fusion_ms": fusion_ms,
            },
        }

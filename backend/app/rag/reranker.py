import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Lightweight but accurate cross-encoder. Good balance of speed vs quality.
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-Encoder reranker that scores (query, document) pairs together.

    Unlike bi-encoders (which embed query and doc independently), a cross-encoder
    sees both at once, giving it far more nuanced relevance signals. This is the
    precision layer — it takes the ~10-20 hybrid retrieval candidates and cuts
    them down to the best RERANK_TOP_N for the LLM.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, top_n: int = 5):
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Re-score and sort candidates by cross-encoder relevance.

        Args:
            query:      The user's original (or semantic) query.
            candidates: List of candidate dicts with at least {"id", "text", "metadata"}.

        Returns:
            Top-N candidates sorted by descending cross-encoder score, each
            augmented with a "rerank_score" field.
        """
        if not candidates:
            logger.warning("Reranker received empty candidate list.")
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in candidates]

        logger.info(f"Reranking {len(pairs)} candidates...")
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[: self.top_n]:
            enriched = doc.copy()
            enriched["rerank_score"] = float(score)
            results.append(enriched)

        logger.info(
            f"Reranking complete. Top score: {results[0]['rerank_score']:.4f} | "
            f"Bottom score: {results[-1]['rerank_score']:.4f}"
        )
        return results

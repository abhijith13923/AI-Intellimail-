"""
RAG Pipeline Orchestrator
-------------------------
Ties together all stages of the pipeline into a single, callable interface:

  User Query
    → QueryParser      (Gemini Flash — extract metadata filters + semantic query)
    → HybridRetriever  (ChromaDB dense + BM25 sparse → RRF fusion)
    → Reranker         (Cross-Encoder — pick the absolute best N chunks)
    → Generator        (Groq llama-3.1-8b-instant — grounded answer generation)
    → Response

This module is intentionally stateless at the class level.
All stateful components (vector_store, bm25_store, etc.) are injected via the
constructor to allow easy testing and singleton reuse in the FastAPI app.
"""

import logging
from typing import Dict, Any

from app.rag.query_parser import QueryParser
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.vector_store import ChromaDBStore
from app.rag.bm25_store import BM25Store
from app.rag.ingestion import EmailIngestionPipeline
from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Top-level orchestrator for the RAG pipeline.

    Instantiate once at app startup (in FastAPI lifespan) and reuse.
    """

    def __init__(self):
        logger.info("Initialising RAG pipeline components...")

        # Shared stores
        self.vector_store = ChromaDBStore(
            collection_name=settings.CHROMA_COLLECTION_NAME
        )
        self.bm25_store = BM25Store()

        # Pipeline stages
        self.ingestion = EmailIngestionPipeline(self.vector_store, self.bm25_store)
        self.query_parser = QueryParser()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_store)
        self.reranker = Reranker(top_n=settings.RERANK_TOP_N)
        self.generator = Generator()

        logger.info("RAG pipeline ready.")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, user_question: str, user_email: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a user question.

        Args:
            user_question: The raw natural-language question.
            user_email:    The authenticated user's email (for multi-tenancy).

        Returns:
            Dict with keys:
              - answer:           The LLM-generated response.
              - sources:          List of source chunk metadata dicts.
              - semantic_query:   The cleaned query used for retrieval.
              - metadata_filters: Extracted filters (for transparency/debugging).
        """
        logger.info(f"RAG query | user={user_email} | question='{user_question}'")

        # Stage 1: Parse the query
        parsed = self.query_parser.extract_query(user_question)
        filters = parsed.metadata_filters.dict(exclude_none=True)

        # Stage 2: Hybrid retrieval
        candidates = self.retriever.retrieve(
            semantic_query=parsed.semantic_query,
            user_email=user_email,
            top_k=settings.RETRIEVAL_TOP_K,
            metadata_filters=filters if filters else None,
        )

        if not candidates:
            return {
                "answer": "I couldn't find any relevant emails for that query.",
                "sources": [],
                "semantic_query": parsed.semantic_query,
                "metadata_filters": filters,
            }

        # Stage 3: Rerank
        reranked = self.reranker.rerank(
            query=parsed.semantic_query,
            candidates=candidates,
        )

        # Stage 4: Generate
        answer = self.generator.generate_response(
            question=user_question,
            reranked_docs=reranked,
        )

        # Build source citations for the frontend
        sources = [
            {
                "email_id": doc["metadata"].get("email_id"),
                "subject": doc["metadata"].get("subject"),
                "sender": doc["metadata"].get("sender"),
                "timestamp": doc["metadata"].get("timestamp"),
                "rerank_score": doc.get("rerank_score"),
            }
            for doc in reranked
        ]

        return {
            "answer": answer,
            "sources": sources,
            "semantic_query": parsed.semantic_query,
            "metadata_filters": filters,
        }

"""
RAG Pipeline Orchestrator
-------------------------
Ties together all stages of the pipeline into a single, callable interface:

  User Query + Chat History
    → QueryParser      (Gemini Flash — extract metadata filters + semantic query)
    → HybridRetriever  (ChromaDB dense + BM25 sparse → RRF fusion)
    → Reranker         (Cross-Encoder — pick the absolute best N chunks)
    → Generator        (Groq LLM — grounded answer generation)
    → Response + full pipeline_trace (for Live Chunk Inspector)

This module is intentionally stateless at the class level.
All stateful components (vector_store, bm25_store, etc.) are injected via the
constructor to allow easy testing and singleton reuse in the FastAPI app.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from app.rag.query_parser import QueryParser
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.vector_store import ChromaDBStore
from app.rag.bm25_store import BM25Store
from app.rag.ingestion import EmailIngestionPipeline
from app.rag.prompt import format_context
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

    def query(
        self,
        user_question: str,
        user_email: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a user question.

        Args:
            user_question: The raw natural-language question.
            user_email:    The authenticated user's email (for multi-tenancy).
            chat_history:  Previous conversation turns for contextual understanding.
                           Format: [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Dict with keys:
              - answer:           The LLM-generated response.
              - sources:          List of source chunk metadata dicts.
              - semantic_query:   The cleaned query used for retrieval.
              - metadata_filters: Extracted filters (for transparency/debugging).
              - pipeline_trace:   Full diagnostic trace for the Live Chunk Inspector.
        """
        pipeline_start = time.perf_counter()
        logger.info(f"RAG query | user={user_email} | question='{user_question}'")

        # ----------------------------------------------------------------
        # Stage 1: Parse the query
        # ----------------------------------------------------------------
        t_parse_start = time.perf_counter()
        parsed = self.query_parser.extract_query(
            user_query=user_question,
            chat_history=chat_history,
        )
        parse_ms = round((time.perf_counter() - t_parse_start) * 1000)

        filters = parsed.metadata_filters.dict(exclude_none=True)
        semantic_query = parsed.semantic_query  # May be None for metadata-only
        is_metadata_only = parsed.is_metadata_only

        logger.info(
            f"Stage 1 done [{parse_ms}ms] | semantic='{semantic_query}' | "
            f"metadata_only={is_metadata_only} | filters={filters}"
        )

        # ----------------------------------------------------------------
        # Stage 2: Hybrid retrieval
        # ----------------------------------------------------------------
        t_retrieval_start = time.perf_counter()
        retrieval_result = self.retriever.retrieve(
            semantic_query=semantic_query,
            user_email=user_email,
            top_k=settings.RETRIEVAL_TOP_K,
            metadata_filters=filters if filters else None,
            is_metadata_only=is_metadata_only,
        )
        retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000)

        candidates = retrieval_result["candidates"]
        dense_results = retrieval_result["dense_results"]
        sparse_results = retrieval_result["sparse_results"]
        retrieval_timings = retrieval_result["timings"]

        logger.info(f"Stage 2 done [{retrieval_ms}ms] | {len(candidates)} candidates from RRF fusion.")

        if not candidates:
            total_ms = round((time.perf_counter() - pipeline_start) * 1000)
            return {
                "answer": "I couldn't find any relevant emails for that query.",
                "sources": [],
                "semantic_query": semantic_query,
                "metadata_filters": filters,
                "pipeline_trace": self._build_trace(
                    user_question=user_question,
                    parsed=parsed,
                    filters=filters,
                    dense_results=[],
                    sparse_results=[],
                    candidates=[],
                    reranked=[],
                    prompt_context="",
                    timings={
                        "parse_ms": parse_ms,
                        "retrieval_ms": retrieval_ms,
                        "rerank_ms": 0,
                        "generation_ms": 0,
                        "total_ms": total_ms,
                        **retrieval_timings,
                    },
                ),
            }

        # ----------------------------------------------------------------
        # Stage 3: Rerank
        # ----------------------------------------------------------------
        rerank_query = semantic_query or user_question  # Fallback to raw query for reranker
        t_rerank_start = time.perf_counter()
        reranked = self.reranker.rerank(
            query=rerank_query,
            candidates=candidates,
        )
        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000)

        logger.info(f"Stage 3 done [{rerank_ms}ms] | {len(reranked)} chunks after reranking.")

        # ----------------------------------------------------------------
        # Stage 4: Generate
        # ----------------------------------------------------------------
        prompt_context = format_context(reranked)

        t_gen_start = time.perf_counter()
        answer = self.generator.generate_response(
            question=user_question,
            reranked_docs=reranked,
        )
        generation_ms = round((time.perf_counter() - t_gen_start) * 1000)

        total_ms = round((time.perf_counter() - pipeline_start) * 1000)
        logger.info(f"Stage 4 done [{generation_ms}ms] | Total pipeline: {total_ms}ms.")

        # Build source citations for the frontend
        sources = [
            {
                "email_id": doc["metadata"].get("email_id"),
                "subject": doc["metadata"].get("subject"),
                "sender": doc["metadata"].get("sender"),
                "date_iso": doc["metadata"].get("date_iso"),
                "timestamp": doc["metadata"].get("timestamp"),
                "rerank_score": doc.get("rerank_score"),
                "rrf_score": doc.get("rrf_score"),
                "chunk_number": doc["metadata"].get("chunk_number"),
                "total_chunks": doc["metadata"].get("total_chunks"),
            }
            for doc in reranked
        ]

        timings = {
            "parse_ms": parse_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            **retrieval_timings,
        }

        pipeline_trace = self._build_trace(
            user_question=user_question,
            parsed=parsed,
            filters=filters,
            dense_results=dense_results,
            sparse_results=sparse_results,
            candidates=candidates,
            reranked=reranked,
            prompt_context=prompt_context,
            timings=timings,
        )

        return {
            "answer": answer,
            "sources": sources,
            "semantic_query": semantic_query,
            "metadata_filters": filters,
            "pipeline_trace": pipeline_trace,
        }

    # ------------------------------------------------------------------
    # Trace builder
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        user_question: str,
        parsed: Any,
        filters: Dict[str, Any],
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        reranked: List[Dict[str, Any]],
        prompt_context: str,
        timings: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Build the full diagnostic trace payload for the Live Chunk Inspector.
        This is returned to the frontend as `pipeline_trace`.
        """
        return {
            "query_analysis": {
                "raw_query": user_question,
                "semantic_query": parsed.semantic_query,
                "is_metadata_only": parsed.is_metadata_only,
                "metadata_filters": filters,
            },
            "stages": {
                "dense": [
                    {
                        "id": d["id"],
                        "text_preview": d["text"][:200],
                        "full_text": d["text"],
                        "metadata": d["metadata"],
                        "vector_distance": d.get("vector_distance"),
                    }
                    for d in dense_results
                ],
                "sparse": [
                    {
                        "id": d["id"],
                        "text_preview": d["text"][:200],
                        "full_text": d["text"],
                        "metadata": d["metadata"],
                        "bm25_score": d.get("bm25_score"),
                    }
                    for d in sparse_results
                ],
                "fused": [
                    {
                        "id": d["id"],
                        "text_preview": d["text"][:200],
                        "full_text": d["text"],
                        "metadata": d["metadata"],
                        "rrf_score": d.get("rrf_score"),
                        "vector_distance": d.get("vector_distance"),
                        "bm25_score": d.get("bm25_score"),
                    }
                    for d in candidates
                ],
                "reranked": [
                    {
                        "id": d["id"],
                        "text_preview": d["text"][:200],
                        "full_text": d["text"],
                        "metadata": d["metadata"],
                        "rerank_score": d.get("rerank_score"),
                        "rrf_score": d.get("rrf_score"),
                        "vector_distance": d.get("vector_distance"),
                        "bm25_score": d.get("bm25_score"),
                    }
                    for d in reranked
                ],
            },
            "prompt_context": prompt_context,
            "timings": timings,
        }

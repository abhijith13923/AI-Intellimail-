import logging
from typing import List, Dict, Any

from app.rag.chunking import EmailChunker
from app.rag.vector_store import ChromaDBStore
from app.rag.bm25_store import BM25Store
from app.config import settings

logger = logging.getLogger(__name__)


class EmailIngestionPipeline:
    """
    Orchestrates the full ingestion pipeline for emails:

      raw email dict
        → EmailChunker  (clean HTML, split into chunks with metadata)
        → ChromaDBStore (dense vector index — persisted to disk)
        → BM25Store     (sparse keyword index — in-memory)

    Multi-tenancy is embedded in each chunk's metadata via the `user_email` field,
    so a single ChromaDB collection and BM25 index serve all users safely.
    """

    def __init__(
        self,
        vector_store: ChromaDBStore,
        bm25_store: BM25Store,
    ):
        self.chunker = EmailChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    # ------------------------------------------------------------------
    # Single email
    # ------------------------------------------------------------------

    def ingest_email(self, email_data: Dict[str, Any]) -> int:
        """
        Ingest a single email through the full pipeline.

        Args:
            email_data: Raw email dict (see EmailChunker.chunk_email for schema).

        Returns:
            Number of chunks generated and indexed.
        """
        email_id = email_data.get("id", "unknown")
        logger.info(f"Ingesting email id={email_id}")

        chunks = self.chunker.chunk_email(email_data)
        if not chunks:
            logger.warning(f"No chunks generated for email id={email_id}.")
            return 0

        # Push to ChromaDB (persisted)
        self.vector_store.upsert_chunks(chunks)

        # Push to BM25 (in-memory — incremental add rebuilds index)
        self.bm25_store.add_documents(chunks)

        logger.info(f"Email id={email_id} ingested as {len(chunks)} chunk(s).")
        return len(chunks)

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------

    def ingest_emails(self, emails: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Batch-ingest a list of emails.

        Args:
            emails: List of raw email dicts.

        Returns:
            Summary dict: {"total_emails": N, "total_chunks": M, "failed": K}
        """
        total_chunks = 0
        failed = 0

        for email in emails:
            try:
                total_chunks += self.ingest_email(email)
            except Exception as e:
                logger.error(f"Failed to ingest email id={email.get('id')}: {e}")
                failed += 1

        summary = {
            "total_emails": len(emails),
            "total_chunks": total_chunks,
            "failed": failed,
        }
        logger.info(f"Batch ingestion complete: {summary}")
        return summary

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_email(self, email_id: str, user_email: str):
        """
        Remove all chunks for a given email from ChromaDB.
        NOTE: BM25 is rebuilt on the next ingestion call.

        Args:
            email_id:   The Gmail message ID.
            user_email: The owning user's email address.
        """
        logger.info(f"Deleting email id={email_id} for user={user_email}")
        self.vector_store.delete_by_email_id(email_id, user_email)
        # BM25 doesn't support single-doc deletion; it will be rebuilt
        # on the next add_documents() call automatically.

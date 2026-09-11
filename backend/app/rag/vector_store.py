import chromadb
from chromadb.config import Settings
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Make sure ChromaDB stores its data persistently in a local folder
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Filter Translation
# ---------------------------------------------------------------------------

def translate_metadata_filters(raw_filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Translate the structured filter dict produced by QueryParser into a valid
    ChromaDB `where` clause.

    ChromaDB operator format:
        {"field": {"$operator": value}}
    Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin

    Handles:
    - date range  → `date_iso` string comparisons using $gte / $lte
    - sender      → `sender_email` and `sender` substring match via $contains (or $eq)
    - subject     → `subject` $contains
    - unread/starred/has_attachment → boolean $eq
    - labels      → simple string contains check on stored comma-joined label string

    Args:
        raw_filters: Dict from MetadataFilters.dict(exclude_none=True).
                     Example: {"date": {"start": "2026-09-10", "end": "2026-09-10"},
                                "sender": "alice@example.com"}

    Returns:
        A valid ChromaDB where clause dict, or None if no filters apply.
    """
    clauses: List[Dict[str, Any]] = []

    for key, value in raw_filters.items():
        if value is None:
            continue

        # ---- Date range ----
        if key == "date" and isinstance(value, dict):
            start = value.get("start")
            end = value.get("end")
            if start:
                clauses.append({"date_iso": {"$gte": start}})
            if end:
                clauses.append({"date_iso": {"$lte": end}})
            # Skip adding the raw dict — we've translated it above

        # ---- Sender ----
        elif key == "sender" and isinstance(value, str) and value.strip():
            # For ChromaDB, we can only do exact match on metadata fields.
            # We match against sender_email (normalized lowercase) for reliability.
            # Partial/display-name matching is handled by BM25's filter logic.
            # If the value looks like a full email, match sender_email; otherwise
            # skip the ChromaDB filter and let BM25 handle it.
            cleaned = value.strip().lower()
            if "@" in cleaned:
                clauses.append({"sender_email": {"$eq": cleaned}})
            # else: no-op for ChromaDB; BM25 will handle name-based matching

        # ---- Subject keyword ----
        elif key == "subject" and isinstance(value, str) and value.strip():
            # ChromaDB doesn't support $contains on metadata; skip the Chroma filter.
            # BM25's pre-filter checks subject with a case-insensitive substring match.
            pass  # handled by BM25 filter logic

        # ---- Boolean flags ----
        elif key in ("unread", "starred", "has_attachment") and isinstance(value, bool):
            clauses.append({key: {"$eq": value}})

        # ---- Labels ----
        elif key == "labels" and isinstance(value, list) and value:
            # Labels are stored as a comma-joined string like "INBOX,UNREAD,IMPORTANT".
            # ChromaDB doesn't support substring matching on metadata; skip and rely on BM25.
            pass  # handled by BM25 filter logic

        # ---- Recipients / CC / BCC ----
        elif key in ("recipients", "cc", "bcc") and isinstance(value, str) and value.strip():
            pass  # substring matching not possible in ChromaDB metadata

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


# ---------------------------------------------------------------------------
# ChromaDB Store
# ---------------------------------------------------------------------------

class ChromaDBStore:
    """
    Manages vector embeddings in a persistent ChromaDB instance.
    Multi-tenancy is handled via the 'user_email' metadata field.
    """

    def __init__(self, collection_name: str = "emails"):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        from chromadb.utils import embedding_functions
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=bge_ef,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upserts chunked documents to ChromaDB.
        Expects chunks in the format: [{"id": str, "text": str, "metadata": dict}]
        """
        if not chunks:
            return

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks to ChromaDB.")

    def delete_by_email_id(self, email_id: str, user_email: str):
        """
        Deletes all chunks associated with a specific email_id.
        """
        try:
            self.collection.delete(
                where={
                    "$and": [
                        {"email_id": {"$eq": email_id}},
                        {"user_email": {"$eq": user_email}},
                    ]
                }
            )
            logger.info(f"Deleted chunks for email {email_id}.")
        except Exception as e:
            logger.error(f"Failed to delete email chunks: {e}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Optional[str],
        user_email: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves top_k similar chunks for a specific user, optionally
        applying metadata filters.

        If `query` is None or empty, falls back to a metadata-only fetch
        (returns `top_k` most recent chunks matching the filter) sorted
        by `timestamp_epoch` descending.

        Args:
            query:            Semantic query string (may be None for metadata-only).
            user_email:       Multi-tenancy scope.
            top_k:            Number of results to return.
            metadata_filters: Raw filter dict from QueryParser (will be translated).

        Returns:
            ChromaDB query result dict.
        """
        # Build user scope clause
        user_clause: Dict[str, Any] = {"user_email": {"$eq": user_email}}

        # Translate and merge metadata filters
        extra_clause = translate_metadata_filters(metadata_filters or {})

        if extra_clause:
            where_clause = {"$and": [user_clause, extra_clause]}
        else:
            where_clause = user_clause

        # Metadata-only path: no semantic query — just fetch by filter
        if not query or not query.strip():
            logger.info(
                f"ChromaDB metadata-only fetch | user={user_email} | top_k={top_k} | "
                f"filter={where_clause}"
            )
            try:
                # get() returns documents matching the filter; sort by epoch desc client-side
                result = self.collection.get(
                    where=where_clause,
                    limit=top_k,
                    include=["documents", "metadatas"],
                )
                # Reformat to match query() output shape so downstream code is uniform
                ids = result.get("ids", [])
                docs = result.get("documents", [])
                metas = result.get("metadatas", [])
                # Sort by timestamp_epoch descending
                combined = sorted(
                    zip(ids, docs, metas),
                    key=lambda x: x[2].get("timestamp_epoch", 0),
                    reverse=True,
                )
                sorted_ids, sorted_docs, sorted_metas = (
                    ([], [], []) if not combined else map(list, zip(*combined))
                )
                return {
                    "ids": [list(sorted_ids)],
                    "documents": [list(sorted_docs)],
                    "metadatas": [list(sorted_metas)],
                    "distances": [[]],
                }
            except Exception as e:
                logger.error(f"ChromaDB metadata-only fetch failed: {e}")
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Semantic search path
        logger.info(
            f"ChromaDB semantic search | user={user_email} | query='{query[:80]}' | "
            f"top_k={top_k} | filter={where_clause}"
        )
        try:
            return self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

import chromadb
from chromadb.config import Settings
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Make sure ChromaDB stores its data persistently in a local folder
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

class ChromaDBStore:
    """
    Manages vector embeddings in a persistent ChromaDB instance.
    Multi-tenancy is handled via the 'user_email' metadata field.
    """
    def __init__(self, collection_name: str = "emails"):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
        from chromadb.utils import embedding_functions
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=bge_ef
        )

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
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks to ChromaDB.")

    def delete_by_email_id(self, email_id: str, user_email: str):
        """
        Deletes all chunks associated with a specific email_id.
        """
        # We delete by where clause
        try:
            self.collection.delete(
                where={
                    "$and": [
                        {"email_id": email_id},
                        {"user_email": user_email}
                    ]
                }
            )
            logger.info(f"Deleted chunks for email {email_id}.")
        except Exception as e:
            logger.error(f"Failed to delete email chunks: {e}")

    def search(self, query: str, user_email: str, top_k: int = 5, metadata_filters: Dict[str, Any] = None):
        """
        Retrieves top_k similar chunks for a specific user, optionally applying metadata filters.
        """
        where_clause = {"user_email": user_email}
        
        # If additional filters are provided, combine them
        if metadata_filters:
            # For simplicity, just doing a flat AND if multiple keys
            # In a real app you'd parse more complex query structures
            and_clauses = [{"user_email": user_email}]
            for k, v in metadata_filters.items():
                and_clauses.append({k: v})
            
            where_clause = {"$and": and_clauses}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )
        return results

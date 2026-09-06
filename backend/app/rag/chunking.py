from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class EmailChunker:
    """
    Handles chunking of raw email text and creating rich metadata per chunk.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _clean_html(self, raw_html: str) -> str:
        """Removes HTML tags and returns plain text."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator="\n").strip()

    def chunk_email(self, email_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks the email content and attaches rich metadata to each chunk.
        
        Expected email_data format:
        {
            "id": str,
            "threadId": str,
            "user_email": str,
            "subject": str,
            "sender": str,
            "recipients": str, # comma separated
            "cc": str,
            "bcc": str,
            "timestamp": str, # ISO 8601 or similar
            "labels": List[str],
            "body": str, # raw body text/html
            "attachment_names": List[str]
        }
        """
        raw_body = email_data.get("body", "")
        # Clean HTML if present
        clean_body = self._clean_html(raw_body)
        
        # If body is completely empty, maybe just index the subject
        if not clean_body.strip():
            clean_body = email_data.get("subject", "No Content")
            
        chunks = self.text_splitter.split_text(clean_body)
        total_chunks = len(chunks)
        
        chunked_documents = []
        labels = email_data.get("labels", [])
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{email_data['id']}_chunk_{i}"
            
            metadata = {
                "chunk_id": chunk_id,
                "email_id": email_data["id"],
                "thread_id": email_data.get("threadId", ""),
                "user_email": email_data.get("user_email", ""),
                "subject": email_data.get("subject", ""),
                "sender": email_data.get("sender", ""),
                "recipients": email_data.get("recipients", ""),
                "cc": email_data.get("cc", ""),
                "bcc": email_data.get("bcc", ""),
                "timestamp": email_data.get("timestamp", ""),
                # Convert list to comma separated string as ChromaDB metadata values must be strings, ints, or floats
                "labels": ",".join(labels), 
                "unread": "UNREAD" in labels,
                "starred": "STARRED" in labels,
                "has_attachment": len(email_data.get("attachment_names", [])) > 0,
                "attachment_names": ",".join(email_data.get("attachment_names", [])),
                "chunk_number": i,
                "total_chunks": total_chunks
            }
            
            chunked_documents.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": metadata
            })
            
        return chunked_documents

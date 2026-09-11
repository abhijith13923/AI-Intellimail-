import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date normalization helpers
# ---------------------------------------------------------------------------

def _parse_date(raw_date: str) -> Optional[datetime]:
    """
    Try to parse a date string from multiple formats into a timezone-aware
    datetime.  Handles:
      - ISO-8601 (e.g. "2026-09-10T14:32:00+00:00")
      - RFC 2822 email header (e.g. "Thu, 10 Sep 2026 14:32:00 +0000")
      - Millisecond epoch integer strings (e.g. "1757500320000")
      - Second epoch integer strings (e.g. "1757500320")
    Returns None if parsing fails.
    """
    if not raw_date:
        return None

    # ---- Epoch (millis or seconds) ----
    if isinstance(raw_date, (int, float)):
        ts = raw_date / 1000 if raw_date > 1e10 else raw_date
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    raw_str = str(raw_date).strip()

    # ---- Numeric string epoch ----
    if re.fullmatch(r"\d+", raw_str):
        ts_int = int(raw_str)
        ts = ts_int / 1000 if ts_int > 1e10 else float(ts_int)
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    # ---- ISO-8601 ----
    try:
        return datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # ---- RFC 2822 ----
    try:
        return parsedate_to_datetime(raw_str)
    except Exception:
        pass

    logger.warning(f"Could not parse date: '{raw_date}'")
    return None


def _extract_sender_parts(sender_raw: str):
    """
    Extract (display_name, email_address) from 'Name <email>' or bare email.
    """
    match = re.match(r"^(.*?)\s*<([^>]+)>$", sender_raw.strip())
    if match:
        name = match.group(1).strip().strip('"')
        email = match.group(2).strip().lower()
        return name or email.split("@")[0], email
    # Bare address
    email = sender_raw.strip().lower()
    return email.split("@")[0], email


# ---------------------------------------------------------------------------
# EmailChunker
# ---------------------------------------------------------------------------

class EmailChunker:
    """
    Handles chunking of raw email text and creating rich metadata per chunk.

    Key improvements over the naive version:
    1. **Context-enriched chunks** — every chunk is prefixed with a compact
       email header (Subject / From / To / Date) so that embeddings carry
       sender, topic, and date signals even for short body fragments.
    2. **Normalized date fields** — stores both `date_iso` (YYYY-MM-DD string,
       usable in ChromaDB `$gte` / `$lte` comparisons) and `timestamp_epoch`
       (integer seconds since epoch, also comparable numerically).
    3. **Decomposed sender fields** — `sender_email` and `sender_name` stored
       separately for precise metadata filtering.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        """Removes HTML tags and returns plain text."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator="\n").strip()

    @staticmethod
    def _build_context_header(
        subject: str,
        sender_raw: str,
        recipients: str,
        date_str: str,
    ) -> str:
        """
        Builds the context header prepended to every chunk body.
        This ensures embedding models encode sender/subject/date signal
        alongside the body text, which dramatically improves retrieval
        accuracy for sender- or date-targeted queries.
        """
        lines = [
            "[EMAIL HEADER]",
            f"Subject: {subject or 'No Subject'}",
            f"From: {sender_raw or 'Unknown'}",
        ]
        if recipients:
            lines.append(f"To: {recipients}")
        if date_str:
            lines.append(f"Date: {date_str}")
        lines.append("[EMAIL BODY]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_email(self, email_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks the email content and attaches rich metadata to each chunk.

        Expected email_data format:
        {
            "id": str,
            "threadId": str,
            "user_email": str,
            "subject": str,
            "sender": str,           # "Name <email>" or bare email
            "recipients": str,       # comma-separated To addresses
            "cc": str,
            "bcc": str,
            "timestamp": str | int,  # RFC 2822 string, ISO-8601, or epoch ms/s
            "internal_date": int,    # Gmail internalDate (millis epoch) — preferred
            "labels": List[str],
            "body": str,             # raw body text or HTML
            "attachment_names": List[str]
        }
        """
        # ---- Parse dates ----
        # Prefer internalDate (millis, highly reliable) over the Date header
        raw_date = email_data.get("internal_date") or email_data.get("timestamp", "")
        parsed_dt = _parse_date(raw_date)

        date_iso: str = parsed_dt.strftime("%Y-%m-%d") if parsed_dt else ""
        timestamp_epoch: int = int(parsed_dt.timestamp()) if parsed_dt else 0
        date_display: str = (
            parsed_dt.strftime("%Y-%m-%d %H:%M UTC") if parsed_dt else str(raw_date)
        )

        # ---- Parse sender ----
        sender_raw = email_data.get("sender", "")
        sender_name, sender_email = _extract_sender_parts(sender_raw) if sender_raw else ("", "")

        # ---- Clean body ----
        raw_body = email_data.get("body", "")
        clean_body = self._clean_html(raw_body)

        subject = email_data.get("subject", "")
        recipients = email_data.get("recipients", "")

        if not clean_body.strip():
            # Fallback: index subject only so at least metadata queries work
            clean_body = subject or "No Content"

        # ---- Build context header ----
        context_header = self._build_context_header(
            subject=subject,
            sender_raw=sender_raw,
            recipients=recipients,
            date_str=date_display,
        )

        # ---- Split into chunks ----
        body_chunks = self.text_splitter.split_text(clean_body)
        total_chunks = len(body_chunks)

        labels: List[str] = email_data.get("labels", [])
        chunk_docs = []

        for i, body_chunk in enumerate(body_chunks):
            chunk_id = f"{email_data['id']}_chunk_{i}"

            # Prepend header to chunk text so embeddings carry context signal
            enriched_text = f"{context_header}\n\n{body_chunk}"

            metadata = {
                # --- Identifiers ---
                "chunk_id": chunk_id,
                "email_id": email_data["id"],
                "thread_id": email_data.get("threadId", ""),
                "user_email": email_data.get("user_email", ""),
                # --- Email fields ---
                "subject": subject,
                "sender": sender_raw,
                "sender_email": sender_email,
                "sender_name": sender_name,
                "recipients": recipients,
                "cc": email_data.get("cc", ""),
                "bcc": email_data.get("bcc", ""),
                # --- Normalized dates (critical for ChromaDB range filters) ---
                "date_iso": date_iso,               # "YYYY-MM-DD" — string comparable
                "timestamp_epoch": timestamp_epoch,  # int seconds — numeric comparable
                "timestamp": str(raw_date),          # raw original (for display only)
                # --- Label-derived booleans ---
                "labels": ",".join(labels),
                "unread": "UNREAD" in labels,
                "starred": "STARRED" in labels,
                "has_attachment": len(email_data.get("attachment_names", [])) > 0,
                "attachment_names": ",".join(email_data.get("attachment_names", [])),
                # --- Chunk position ---
                "chunk_number": i,
                "total_chunks": total_chunks,
            }

            chunk_docs.append({
                "id": chunk_id,
                "text": enriched_text,
                "metadata": metadata,
            })

        return chunk_docs

import logging
from typing import Optional, List

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema — structured output from the query parsing LLM
# ---------------------------------------------------------------------------

class DateRange(BaseModel):
    start: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class MetadataFilters(BaseModel):
    sender: Optional[str] = Field(None, description="The sender of the email (e.g. alice@example.com)")
    recipients: Optional[str] = Field(None, description="The recipients (To) of the email")
    cc: Optional[str] = Field(None, description="People CC'd on the email")
    bcc: Optional[str] = Field(None, description="People BCC'd on the email")
    date: Optional[DateRange] = Field(None, description="Date range for the email")
    labels: Optional[List[str]] = Field(None, description="Gmail labels (e.g. IMPORTANT, INBOX, STARRED)")
    unread: Optional[bool] = Field(None, description="Whether the email is unread")
    starred: Optional[bool] = Field(None, description="Whether the email is starred")
    has_attachment: Optional[bool] = Field(None, description="Whether the email has attachments")
    subject: Optional[str] = Field(None, description="The subject of the email")


class ParsedQuery(BaseModel):
    metadata_filters: MetadataFilters = Field(
        default_factory=MetadataFilters,
        description="Extracted metadata constraints from the user query"
    )
    semantic_query: str = Field(
        ...,
        description="The remaining intent of the user question, stripped of metadata constraints. "
                    "This will be used for vector + keyword search."
    )


# ---------------------------------------------------------------------------
# Query Parser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Uses Gemini Flash to parse a natural-language user query into:
      1. Structured metadata filters (sender, date, labels, etc.)
      2. A clean semantic query for vector + BM25 search.

    Gemini Flash is used here because it's fast and cheap — we just need
    structured extraction, not deep reasoning.
    """

    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_QUERY_PARSE_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,      # Deterministic structured extraction
        )
        self.parser = PydanticOutputParser(pydantic_object=ParsedQuery)

        self.prompt = PromptTemplate(
            template="""You are a query parsing assistant for a Gmail search engine.
Your ONLY job is to extract structured metadata from the user query and isolate the semantic search intent.

Rules:
- Extract ONLY explicitly mentioned metadata (sender, date range, labels, subject, attachments, etc.).
- Do NOT infer or guess metadata that is not mentioned.
- The semantic_query should contain the user's core intent, excluding the metadata constraints.
- If the entire query is a metadata constraint with no semantic content, set semantic_query to a generic phrase like "email content".
- Today's date is {today}.

{format_instructions}

User Query: {query}
""",
            input_variables=["query", "today"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        # LCEL chain
        self.chain = self.prompt | self.llm | self.parser
        logger.info(f"QueryParser initialised with model: {settings.GEMINI_QUERY_PARSE_MODEL}")

    def extract_query(self, user_query: str) -> ParsedQuery:
        """
        Parse the user query and return a structured ParsedQuery.

        Args:
            user_query: The raw natural-language query from the user.

        Returns:
            ParsedQuery with metadata_filters and semantic_query populated.
        """
        from datetime import date
        today_str = date.today().isoformat()

        logger.info(f"Parsing query: '{user_query}'")
        try:
            result = self.chain.invoke({"query": user_query, "today": today_str})
            logger.info(
                f"Parsed → semantic='{result.semantic_query}' | "
                f"filters={result.metadata_filters.dict(exclude_none=True)}"
            )
            return result
        except Exception as e:
            logger.error(f"QueryParser failed, falling back to raw query: {e}")
            # Graceful fallback — treat the entire query as semantic
            return ParsedQuery(
                metadata_filters=MetadataFilters(),
                semantic_query=user_query,
            )

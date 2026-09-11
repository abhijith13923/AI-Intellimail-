import logging
from datetime import date
from typing import Optional, List, Dict, Any

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
    start: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format (inclusive)")
    end: Optional[str] = Field(None, description="End date in YYYY-MM-DD format (inclusive)")


class MetadataFilters(BaseModel):
    sender: Optional[str] = Field(None, description="The sender of the email (e.g. alice@example.com or just 'Alice')")
    recipients: Optional[str] = Field(None, description="The recipients (To) of the email")
    cc: Optional[str] = Field(None, description="People CC'd on the email")
    bcc: Optional[str] = Field(None, description="People BCC'd on the email")
    date: Optional[DateRange] = Field(None, description="Date range for the email")
    labels: Optional[List[str]] = Field(None, description="Gmail labels (e.g. IMPORTANT, INBOX, STARRED)")
    unread: Optional[bool] = Field(None, description="Whether the email is unread")
    starred: Optional[bool] = Field(None, description="Whether the email is starred")
    has_attachment: Optional[bool] = Field(None, description="Whether the email has attachments")
    subject: Optional[str] = Field(None, description="Keywords from the email subject")


class ParsedQuery(BaseModel):
    metadata_filters: MetadataFilters = Field(
        default_factory=MetadataFilters,
        description="Extracted metadata constraints from the user query"
    )
    semantic_query: Optional[str] = Field(
        None,
        description=(
            "The remaining semantic intent of the user question AFTER removing metadata constraints. "
            "This will drive vector + keyword search. "
            "Set to null/None if the query is PURELY a metadata/listing request with no semantic intent "
            "(e.g. 'any mails from 10th sept?', 'show emails from Alice', 'unread emails'). "
            "DO NOT use generic filler phrases like 'email content' — either extract real semantic "
            "intent or leave it null."
        )
    )
    is_metadata_only: bool = Field(
        False,
        description=(
            "Set to true when the query is entirely about metadata (sender, date, labels, attachments) "
            "and contains NO semantic content about what the email says or discusses. "
            "Examples of metadata-only: 'any mails from 10th sept?', 'emails from boss@company.com', "
            "'show my unread emails', 'emails with attachments'. "
            "Examples of NOT metadata-only: 'what did Alice say about the budget?', "
            "'find the email about the project deadline', 'did anyone email about the meeting?'"
        )
    )


# ---------------------------------------------------------------------------
# Query Parser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Uses Gemini Flash to parse a natural-language user query into:
      1. Structured metadata filters (sender, date, labels, etc.)
      2. A clean semantic query for vector + BM25 search, OR null if
         the query is purely a metadata lookup.
      3. A flag indicating whether this is a metadata-only query.

    Multi-turn context: pass `chat_history` to resolve pronoun references
    and implicit context from previous exchanges.
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
1. Extract ONLY explicitly mentioned metadata (sender, date range, labels, subject keywords, attachments, etc.).
2. Do NOT infer or guess metadata that is not explicitly mentioned.
3. For the semantic_query field:
   - Extract the user's core question / topic of interest EXCLUDING the metadata constraints.
   - If the query is PURELY about metadata (e.g. "any mails from 10th sept?", "show emails from Alice",
     "emails with attachments"), set semantic_query to null and is_metadata_only to true.
   - NEVER set semantic_query to generic filler like "email content" — that is meaningless and harmful.
   - Only populate semantic_query when there is real conceptual content (e.g. "what did Alice say about the budget?" → semantic_query = "budget discussion").
4. Today's date is {today}.
5. If a chat history is provided, use it to resolve references like "that email", "he", "the previous one".

{format_instructions}

Chat History (most recent last):
{chat_history}

Current User Query: {query}
""",
            input_variables=["query", "today", "chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        # LCEL chain
        self.chain = self.prompt | self.llm | self.parser
        logger.info(f"QueryParser initialised with model: {settings.GEMINI_QUERY_PARSE_MODEL}")

    def extract_query(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ParsedQuery:
        """
        Parse the user query and return a structured ParsedQuery.

        Args:
            user_query:   The raw natural-language query from the user.
            chat_history: Optional list of previous turns in format
                          [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            ParsedQuery with metadata_filters, semantic_query, and
            is_metadata_only populated.
        """
        today_str = date.today().isoformat()

        # Format chat history for prompt injection
        history_str = "None"
        if chat_history:
            lines = []
            for turn in chat_history[-6:]:  # Last 3 exchanges max
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                lines.append(f"{role}: {content}")
            history_str = "\n".join(lines)

        logger.info(f"Parsing query: '{user_query}' | history_turns={len(chat_history) if chat_history else 0}")

        try:
            result = self.chain.invoke({
                "query": user_query,
                "today": today_str,
                "chat_history": history_str,
            })
            logger.info(
                f"Parsed → semantic='{result.semantic_query}' | "
                f"metadata_only={result.is_metadata_only} | "
                f"filters={result.metadata_filters.dict(exclude_none=True)}"
            )
            return result
        except Exception as e:
            logger.error(f"QueryParser failed, falling back to raw query: {e}")
            # Graceful fallback — treat the entire query as semantic
            return ParsedQuery(
                metadata_filters=MetadataFilters(),
                semantic_query=user_query,
                is_metadata_only=False,
            )

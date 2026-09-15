import logging
from typing import List, Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.rag.prompt import get_rag_prompt_template, format_context

logger = logging.getLogger(__name__)


class Generator:
    """
    Final generation step: given a user question and reranked context chunks,
    calls Groq's API (Lopenai/gpt-oss-20b) to produce the answer.

    We use Groq here for its extremely generous free-tier rate limits (14.4k
    requests/day), making it ideal for a personal email assistant.
    """

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_GENERATION_MODEL,
            temperature=0.2,        # Low temp for factual, grounded answers
            max_tokens=1024,
        )
        self.prompt = get_rag_prompt_template()
        self.output_parser = StrOutputParser()

        # Build the LCEL chain: prompt | llm | parser
        self.chain = self.prompt | self.llm | self.output_parser
        logger.info(f"Generator initialised with model: {settings.GROQ_GENERATION_MODEL}")

    def generate_response(
        self,
        question: str,
        reranked_docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a final, grounded answer from the reranked context.

        Args:
            question:      The user's original question.
            reranked_docs: Top-N chunks from the Reranker.
            chat_history:  Optional previous conversation turns.

        Returns:
            The LLM's answer as a plain string.
        """
        context = format_context(reranked_docs)
        logger.info(
            f"Generating response | question='{question}' | "
            f"context_chunks={len(reranked_docs)}"
        )

        formatted_history = []
        if chat_history:
            # We limit to the last 6 messages (3 user, 3 assistant)
            recent_history = chat_history[-6:]
            for msg in recent_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted_history.append(AIMessage(content=content))

        try:
            response = self.chain.invoke({
                "context": context,
                "question": question,
                "chat_history": formatted_history,
            })
            return response
        except Exception as e:
            logger.error(f"Generator failed to produce a response: {e}")
            raise

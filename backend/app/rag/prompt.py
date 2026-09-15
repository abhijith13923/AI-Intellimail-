from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# RAG Generation Prompt
# ---------------------------------------------------------------------------
# Design goals:
#   1. Strictly grounded — LLM must NOT use outside knowledge.
#   2. Source-aware — cite the email subject / sender for traceability.
#   3. Handles "no relevant emails" gracefully.
#   4. Structured context format makes it easy for the model to parse.
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are an intelligent Gmail assistant. Your sole purpose is to answer questions about the user's emails.

STRICT RULES you must ALWAYS follow:
1. Answer ONLY using the email context provided below. Do NOT use any outside knowledge.
2. If the context does not contain enough information to answer the question, say:
   "I couldn't find relevant emails for that query."
3. When citing information, reference the email by its subject and sender in parentheses.
   Example: "The meeting is at 3 PM (Subject: 'Team Standup', From: boss@company.com)"
4. Be concise and direct. Avoid filler phrases like "Based on the provided context...".
5. If multiple emails are relevant, synthesize the information clearly.

--- EMAIL CONTEXT ---
{context}
--- END CONTEXT ---
"""

RAG_HUMAN_PROMPT = "{question}"


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate used for the final RAG generation step.

    The template has three input variables:
      - context:  The formatted string of retrieved email chunks.
      - chat_history: List of previous conversation messages.
      - question: The user's original question.
    """
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", RAG_HUMAN_PROMPT),
    ])


def format_context(reranked_docs: list) -> str:
    """
    Formats the reranked document list into a clean context string for the LLM.

    Each chunk is presented with its key metadata (subject, sender, date) so
    the model can cite sources accurately.

    Args:
        reranked_docs: List of chunk dicts with "text" and "metadata" keys.

    Returns:
        A formatted multi-chunk context string.
    """
    if not reranked_docs:
        return "No relevant emails found."

    parts = []
    for i, doc in enumerate(reranked_docs, start=1):
        meta = doc.get("metadata", {})
        subject = meta.get("subject", "No Subject")
        sender = meta.get("sender", "Unknown Sender")
        timestamp = meta.get("timestamp", "")
        text = doc.get("text", "")

        chunk_header = (
            f"[Email {i}]\n"
            f"Subject: {subject}\n"
            f"From: {sender}\n"
            f"Date: {timestamp}\n"
            f"Content:\n{text}"
        )
        parts.append(chunk_header)

    return "\n\n---\n\n".join(parts)

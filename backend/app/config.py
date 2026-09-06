import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Gmail Semantic Intelligence Assistant"

    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Groq model for final generation (generous rate limits)
    GROQ_GENERATION_MODEL: str = "openai/gpt-oss-20b"

    # Gemini model for fast query parsing
    GEMINI_QUERY_PARSE_MODEL: str = "gemini-2.5-flash"

    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 10       # Fetch top-K from each store before reranking
    RERANK_TOP_N: int = 5           # Final context chunks passed to LLM

    # ChromaDB
    CHROMA_COLLECTION_NAME: str = "emails"

settings = Settings()

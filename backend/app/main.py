import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests
from app.config import settings
from app.gmail.sync import IncrementalSyncService
from app.gmail.service import fetch_recent_emails
from app.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# sync_service is initialised after the RAG pipeline is ready (see lifespan).
# We declare it here so the background task closure can reference it.
sync_service: "IncrementalSyncService" = None

async def poll_emails_background_task():
    logger.info("Background email syncing is currently disabled.")
    # while True:
    #     try:
    #         logger.info("Running sync cycle...")
    #         # Run the synchronous sync process in a threadpool
    #         await asyncio.to_thread(sync_service.run_sync_cycle)
    #     except Exception as e:
    #         logger.error(f"Error in background email syncing: {e}")
    #     
    #     # Poll every 30 seconds as requested
    #     await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sync_service
    # Startup: initialise RAG pipeline and background sync task
    logger.info("Initialising RAG pipeline...")
    app.state.rag_pipeline = await asyncio.to_thread(RAGPipeline)
    logger.info("RAG pipeline ready.")

    # Share the same ingestion pipeline with the sync service
    # so emails synced from Gmail also land in the BM25 index
    sync_service = IncrementalSyncService(
        ingestion_pipeline=app.state.rag_pipeline.ingestion
    )

    task = asyncio.create_task(poll_emails_background_task())
    yield
    # Shutdown: Cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Background email syncing stopped.")

app = FastAPI(title="Gmail Semantic Intelligence Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthCodeRequest(BaseModel):
    code: str

@app.post("/api/auth/google")
async def auth_google(request: AuthCodeRequest):
    """
    Exchanges the authorization code for access and refresh tokens,
    and returns the user profile.
    """
    logger.info("Received /api/auth/google request!")
    
    if not os.path.exists('creds.json'):
        logger.error("creds.json is missing.")
        raise HTTPException(status_code=500, detail="creds.json is missing on the server. Please download it from Google Cloud Console (Web application).")

    try:
        logger.info("Initializing OAuth Flow...")
        # Initialize the OAuth2 flow
        flow = Flow.from_client_secrets_file(
            'creds.json',
            scopes=['https://www.googleapis.com/auth/gmail.readonly', 'openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
            redirect_uri='postmessage'
        )

        logger.info("Fetching token from Google...")
        # Exchange the authorization code for tokens
        flow.fetch_token(code=request.code)
        credentials = flow.credentials

        logger.info("Verifying ID token...")
        # Verify the ID token to extract user profile
        request_session = requests.Request()
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, request_session, flow.client_config['client_id']
        )

        logger.info("Saving token to token.json...")
        # Save tokens securely (For now, we'll save it to token.json to test the sync)
        with open('token.json', 'w') as token_file:
            token_file.write(credentials.to_json())

        user_email = id_info.get('email')
        logger.info(f"User {user_email} authenticated. Awaiting manual sync.")

        logger.info("Returning success response to frontend.")
        return {
            "status": "success",
            "profile": {
                "name": id_info.get('name'),
                "email": user_email,
                "picture": id_info.get('picture')
            }
        }
    except ValueError as e:
        logger.error(f"Invalid Token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running."}

@app.get("/api/test-gmail")
def test_gmail_fetch():
    """
    Test endpoint to fetch the last 5 emails via the old method. 
    """
    emails = fetch_recent_emails(max_results=5)
    return {"status": "success", "emails": emails}

@app.post("/api/sync/initial")
def trigger_initial_sync(user_email: str, days_back: int = 5):
    """
    Triggers an initial sync for a user, fetching emails from the past N days.
    """
    try:
        sync_service.initial_sync(user_email, days_back)
        return {"status": "success", "message": f"Initial sync triggered for {user_email} over past {days_back} days."}
    except Exception as e:
        logger.error(f"Failed to trigger initial sync: {e}")
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# RAG Endpoints
# ---------------------------------------------------------------------------

class RAGQueryRequest(BaseModel):
    question: str
    user_email: str
    # Optional chat history for multi-turn contextual understanding.
    # Each entry: {"role": "user"|"assistant", "content": "..."}
    chat_history: Optional[List[Dict[str, Any]]] = None

class RAGIngestRequest(BaseModel):
    emails: List[Dict[str, Any]]

@app.post("/api/rag/query")
async def rag_query(payload: RAGQueryRequest, request: Request):
    """
    Run the full RAG pipeline for a user question.

    Request body:
      - question:   The natural-language question.
      - user_email: The authenticated user's email (multi-tenancy key).

    Returns:
      - answer:           LLM-generated grounded response.
      - sources:          List of source email metadata used to generate the answer.
      - semantic_query:   The cleaned query used for retrieval (for debugging).
      - metadata_filters: Extracted metadata filters (for debugging).
    """
    pipeline: RAGPipeline = request.app.state.rag_pipeline
    try:
        result = await asyncio.to_thread(
            pipeline.query,
            payload.question,
            payload.user_email,
            payload.chat_history,
        )
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/ingest")
async def rag_ingest(payload: RAGIngestRequest, request: Request):
    """
    Manually ingest a batch of emails into the RAG pipeline.
    Useful for testing or bootstrapping the index outside the sync cycle.

    Request body:
      - emails: List of raw email dicts matching the EmailChunker schema.
    """
    pipeline: RAGPipeline = request.app.state.rag_pipeline
    try:
        summary = await asyncio.to_thread(
            pipeline.ingestion.ingest_emails,
            payload.emails,
        )
        return {"status": "success", **summary}
    except Exception as e:
        logger.error(f"RAG ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

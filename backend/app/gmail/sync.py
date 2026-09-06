import logging
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.rag.ingestion import EmailIngestionPipeline
from app.rag.vector_store import ChromaDBStore
from app.rag.bm25_store import BM25Store
from app.sync_state import SyncStateManager
from app.gmail.auth import get_gmail_credentials
import datetime

logger = logging.getLogger(__name__)

class IncrementalSyncService:
    def __init__(self, ingestion_pipeline: Optional[EmailIngestionPipeline] = None):
        if ingestion_pipeline is not None:
            # Injected from the RAGPipeline singleton (preferred path)
            self.ingestion = ingestion_pipeline
        else:
            # Standalone / test path — create own stores
            vector_store = ChromaDBStore()
            bm25_store = BM25Store()
            self.ingestion = EmailIngestionPipeline(vector_store, bm25_store)

    def get_service_for_user(self, user_email: str):
        # TODO: Retrieve credentials for the specific user instead of default.
        # For now, we mock using the default credentials.
        creds = get_gmail_credentials()
        return build('gmail', 'v1', credentials=creds)

    def _extract_email_metadata(self, msg_data: dict, user_email: str) -> dict:
        """Helper to transform Gmail API message data into our generic schema."""
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        
        def get_header(name: str):
            return next((h['value'] for h in headers if h['name'].lower() == name.lower()), "")

        # To get a plain text body, we might have to recursively find it in parts
        def get_body(payload_node) -> str:
            body = ""
            if 'data' in payload_node.get('body', {}):
                import base64
                body_data = payload_node['body']['data']
                try:
                    body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                except Exception:
                    pass
            
            if 'parts' in payload_node:
                for part in payload_node['parts']:
                    body += " " + get_body(part)
            return body

        body = get_body(payload)

        # check for attachments
        attachment_names = []
        if 'parts' in payload:
             for part in payload['parts']:
                 if part.get('filename'):
                     attachment_names.append(part['filename'])

        return {
            "id": msg_data['id'],
            "threadId": msg_data.get('threadId', ''),
            "user_email": user_email,
            "subject": get_header('Subject'),
            "sender": get_header('From'),
            "recipients": get_header('To'),
            "cc": get_header('Cc'),
            "bcc": get_header('Bcc'),
            "timestamp": get_header('Date'), # Note: might want to parse into uniform format
            "labels": msg_data.get('labelIds', []),
            "body": body,
            "attachment_names": attachment_names
        }

    def _ingest_message(self, service, message_id: str, user_email: str):
        """Fetches, chunks, and upserts a single message into both ChromaDB and BM25."""
        try:
            msg_data = service.users().messages().get(userId='me', id=message_id, format='full').execute()
            email_data = self._extract_email_metadata(msg_data, user_email)
            self.ingestion.ingest_email(email_data)
        except HttpError as error:
            logger.error(f"Error fetching message {message_id}: {error}")

    def initial_sync(self, user_email: str, days_back: int = 5):
        """Performs an initial full sync for a specific timeframe."""
        logger.info(f"Starting initial sync for {user_email} (past {days_back} days).")
        service = self.get_service_for_user(user_email)
        
        # Calculate date string for the query
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y/%m/%d')
        query = f"after:{start_date}"

        try:
            # First, fetch the history ID of the current state
            profile = service.users().getProfile(userId='me').execute()
            current_history_id = profile.get('historyId')

            # Fetch messages
            messages = []
            request = service.users().messages().list(userId='me', q=query, maxResults=100)
            while request is not None:
                response = request.execute()
                messages.extend(response.get('messages', []))
                request = service.users().messages().list_next(request, response)

            logger.info(f"Found {len(messages)} messages to ingest for {user_email}.")
            for msg in messages:
                self._ingest_message(service, msg['id'], user_email)

            if current_history_id:
                SyncStateManager.set_history_id(user_email, current_history_id)
                
            logger.info(f"Initial sync complete for {user_email}.")

        except HttpError as error:
            logger.error(f"An error occurred during initial sync: {error}")

    def sync_incremental(self, user_email: str):
        """Syncs changes since the last recorded historyId."""
        start_history_id = SyncStateManager.get_history_id(user_email)
        if not start_history_id:
            logger.info(f"No historyId for {user_email}, falling back to initial sync.")
            self.initial_sync(user_email)
            return

        service = self.get_service_for_user(user_email)
        logger.info(f"Starting incremental sync for {user_email} from historyId {start_history_id}")
        
        try:
            request = service.users().history().list(userId='me', startHistoryId=start_history_id)
            new_history_id = start_history_id
            
            while request is not None:
                response = request.execute()
                history_records = response.get('history', [])
                
                for record in history_records:
                    # Handle messagesAdded
                    if 'messagesAdded' in record:
                        for item in record['messagesAdded']:
                            self._ingest_message(service, item['message']['id'], user_email)

                    # Handle messagesDeleted
                    if 'messagesDeleted' in record:
                        for item in record['messagesDeleted']:
                            self.ingestion.delete_email(item['message']['id'], user_email)

                    # Handle labelsAdded / labelsRemoved
                    if 'labelsAdded' in record:
                        for item in record['labelsAdded']:
                            self._ingest_message(service, item['message']['id'], user_email)
                            
                    if 'labelsRemoved' in record:
                        for item in record['labelsRemoved']:
                            self._ingest_message(service, item['message']['id'], user_email)

                new_history_id = response.get('historyId', new_history_id)
                request = service.users().history().list_next(request, response)
                
            # Update history ID
            SyncStateManager.set_history_id(user_email, new_history_id)
            logger.info(f"Incremental sync complete for {user_email}. New historyId: {new_history_id}")
            
        except HttpError as error:
            if error.resp.status == 404:
                logger.warning(f"HistoryId {start_history_id} expired for {user_email}. Full sync required.")
                self.initial_sync(user_email)
            else:
                logger.error(f"Error during incremental sync: {error}")

    def run_sync_cycle(self):
        """Runs a sync cycle for all registered users."""
        # TODO: Get all registered users from DB
        # For now, just sync the default one
        test_user = "default@user.com"
        self.sync_incremental(test_user)

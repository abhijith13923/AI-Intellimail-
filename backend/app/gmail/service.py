from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .auth import get_gmail_credentials

def fetch_recent_emails(max_results=5):
    """
    Fetches the most recent emails from the authenticated user's inbox.
    """
    try:
        creds = get_gmail_credentials()
        # Build the Gmail service
        service = build('gmail', 'v1', credentials=creds)

        # Call the Gmail API to list messages
        results = service.users().messages().list(userId='me', maxResults=max_results).execute()
        messages = results.get('messages', [])

        if not messages:
            return []

        emails = []
        for msg in messages:
            # Get the full message details
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            
            payload = msg_data.get('payload', {})
            headers = payload.get('headers', [])
            
            # Extract Subject and From
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")
            snippet = msg_data.get('snippet', '')
            
            emails.append({
                "id": msg['id'],
                "subject": subject,
                "sender": sender,
                "snippet": snippet
            })
            
        return emails

    except HttpError as error:
        print(f"An error occurred in Gmail API: {error}")
        return []

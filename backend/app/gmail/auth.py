import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']

def get_gmail_credentials():
    """
    Reads the session from token.json.
    Throws an error if the user hasn't logged in via the frontend yet.
    """
    creds = None
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save the refreshed credentials
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        else:
            raise FileNotFoundError(
                "token.json not found or invalid! Please log in via the frontend application first."
            )
            
    return creds

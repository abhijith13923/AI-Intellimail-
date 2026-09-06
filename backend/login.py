from app.gmail.auth import get_gmail_credentials

print("Initiating Google Login...")
creds = get_gmail_credentials()

if creds and creds.valid:
    print("SUCCESS! You are logged in. token.json has been generated.")
else:
    print("Failed to authenticate.")

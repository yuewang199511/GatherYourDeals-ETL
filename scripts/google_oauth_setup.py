"""
One-time Google OAuth setup — run this once to obtain a refresh token.

Usage:
    python scripts/google_oauth_setup.py --client-secret <path/to/client_secret.json>

Steps:
    1. Google Cloud Console → APIs & Services → Library → enable "Google Drive API"
    2. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
       → Application type: Desktop app → Download JSON
    3. Run this script with that JSON file.
    4. A browser window opens → sign in → grant "Google Drive (read-only)" access.
    5. The script prints GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN.
       Paste them into your .env file.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    sys.exit("google-auth-oauthlib not installed.  Run: pip install google-auth-oauthlib")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def main():
    parser = argparse.ArgumentParser(description="Obtain a Google Drive OAuth refresh token.")
    parser.add_argument(
        "--client-secret",
        required=True,
        metavar="PATH",
        help="Path to the client_secret_*.json file downloaded from Google Cloud Console.",
    )
    args = parser.parse_args()

    # Expand glob patterns (e.g. ~/Downloads/client_secret_*.json) in case the
    # shell didn't expand them (common on WSL or when quoting the argument).
    raw = str(Path(args.client_secret).expanduser())
    matches = glob.glob(raw)
    if not matches:
        sys.exit(f"File not found: {raw}")
    if len(matches) > 1:
        sys.exit(f"Multiple files matched — be more specific:\n  " + "\n  ".join(matches))
    secret_path = Path(matches[0])

    flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
    print("\nStarting local auth server on port 8085...")
    print("Open the URL that appears below in your Windows browser.\n")
    creds = flow.run_local_server(port=8085, open_browser=False)

    client_config = json.loads(secret_path.read_text())
    installed = client_config.get("installed") or client_config.get("web", {})

    print("\n--- Add these to your .env ---")
    print(f"GOOGLE_CLIENT_ID={installed.get('client_id', creds.client_id)}")
    print(f"GOOGLE_CLIENT_SECRET={installed.get('client_secret', creds.client_secret)}")
    print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
    print("------------------------------\n")


if __name__ == "__main__":
    main()

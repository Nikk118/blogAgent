import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
load_dotenv()
service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

if service_account_json:
    cred = credentials.Certificate(json.loads(service_account_json))
elif service_account_path:
    cred = credentials.Certificate(service_account_path)
else:
    raise ValueError("No Firebase credentials found")

firebase_admin.initialize_app(cred)

def verify_firebase_token(token: str):
    decoded_token = auth.verify_id_token(token, clock_skew_seconds=5)
    return decoded_token
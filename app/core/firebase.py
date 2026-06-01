import firebase_admin

from firebase_admin import credentials
from firebase_admin import auth

cred = credentials.Certificate(
    "app/core/serviceAccountKey.json"
)

firebase_admin.initialize_app(cred)


def verify_firebase_token(token: str):
    decoded_token = auth.verify_id_token(
        token,
        clock_skew_seconds=5
    )

    return decoded_token
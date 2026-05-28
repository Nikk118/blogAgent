from fastapi import Header, HTTPException

from app.core.firebase import verify_firebase_token


async def get_current_user(
    authorization: str = Header(default=None)
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format"
        )

    try:
        token = authorization.replace("Bearer ", "")

        decoded_token = verify_firebase_token(token)

        return decoded_token

    except Exception as e:
        print("TOKEN ERROR:", e)

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
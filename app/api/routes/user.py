from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user

from app.db.dependencies import get_db

from app.models.user import User

router = APIRouter()


@router.post("/sync-user")
def sync_user(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(
        User.firebase_uid ==
        current_user["uid"]
    ).first()

    if not user:

        user = User(
            firebase_uid=current_user["uid"],
            email=current_user["email"]
        )

        db.add(user)

        db.commit()

        db.refresh(user)

    return {
        "message": "User synced",
        "user_id": user.id,
        "email": user.email
    }
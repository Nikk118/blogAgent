from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from pydantic import BaseModel

from sqlalchemy.orm import Session

import traceback

from app.graphs.blog_graph import run

from app.core.dependencies import get_current_user

from app.db.dependencies import get_db

from app.models.blog_session import BlogSession
from app.models.user import User


router = APIRouter(
    prefix="/blog",
    tags=["Blog"]
)


# =========================
# REQUEST SCHEMA
# =========================

class BlogRequest(BaseModel):

    topic: str


# =========================
# RESPONSE SCHEMA
# =========================

class BlogResponse(BaseModel):

    topic: str

    result: dict


# =========================
# ROUTES
# =========================

@router.post(
    "/generate",
    response_model=BlogResponse
)
def generate_blog(
    req: BlogRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    try:

        # =========================
        # GENERATE BLOG
        # =========================

        result = run(req.topic)

        # =========================
        # FIND USER
        # =========================

        user = db.query(User).filter(
            User.firebase_uid ==
            current_user["uid"]
        ).first()

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        # =========================
        # SAVE BLOG
        # =========================

        new_blog = BlogSession(
            user_id=user.id,
            title=req.topic,
            prompt=req.topic,
            content=result["final"]
        )

        db.add(new_blog)

        db.commit()

        db.refresh(new_blog)

        # =========================
        # RETURN RESPONSE
        # =========================

        return {
            "topic": req.topic,
            "result": result
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
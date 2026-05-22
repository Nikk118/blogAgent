from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback

from app.graphs.blog_graph import run

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
def generate_blog(req: BlogRequest):

    try:

        result = run(req.topic)

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
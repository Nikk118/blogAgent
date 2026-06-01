from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from sqlalchemy.orm import Session

import traceback

from app.graphs.blog_graph import run

from app.core.dependencies import get_current_user

from app.db.dependencies import get_db

from app.models.blog_session import BlogSession
from app.models.blog_image import BlogImage
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
            User.firebase_uid == current_user["uid"]
        ).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # =========================
        # SAVE BLOG SESSION
        # =========================
        result_for_db = {
    **result,
    "generated_images": []  # don't store bytes in JSON
}

        new_blog = BlogSession(
            user_id=user.id,
            title=req.topic,
            prompt=req.topic,
            content=jsonable_encoder(result_for_db)
        )

        db.add(new_blog)
        db.commit()
        db.refresh(new_blog)

        # =========================
        # SAVE IMAGES TO DB
        # =========================

        generated_images = result.get("generated_images") or []

        for img in generated_images:
            blog_image = BlogImage(
                blog_session_id=new_blog.id,
                filename=img["filename"],
                alt=img["alt"],
                caption=img["caption"],
                image_data=img["image_data"],
            )
            db.add(blog_image)

        db.commit()

        # =========================
        # RETURN RESPONSE
        # =========================

        result_for_response = {
    **result,
    "generated_images": [
        {"filename": img["filename"], "alt": img["alt"], "caption": img["caption"]}
        for img in generated_images
    ]
}
        return {
            "topic": req.topic,
            "result": jsonable_encoder(result_for_response)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
def get_blogs(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import base64

    user = db.query(User).filter(
        User.firebase_uid == current_user["uid"]
    ).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    blogs = db.query(BlogSession).filter(
        BlogSession.user_id == user.id
    ).order_by(
        BlogSession.id.desc()
    ).all()

    response = []

    for blog in blogs:

        images = db.query(BlogImage).filter(
            BlogImage.blog_session_id == blog.id
        ).all()

        images_data = [
            {
                "filename": img.filename,
                "alt": img.alt,
                "caption": img.caption,
                "image_data": (
                    base64.b64encode(img.image_data)
                    .decode("utf-8")
                    if img.image_data
                    else None
                ),
            }
            for img in images
        ]

        response.append({
            "id": blog.id,
            "title": blog.title,
            "prompt": blog.prompt,
            "content": blog.content,
            
            "generated_images": images_data,
        })

    return response


@router.get("/{blog_id}")
def get_blog(
    blog_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.firebase_uid == current_user["uid"]
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    blog = db.query(BlogSession).filter(
        BlogSession.id == blog_id,
        BlogSession.user_id == user.id
    ).first()

    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")

    # fetch images and convert bytes to base64 for frontend
    import base64
    images = db.query(BlogImage).filter(
        BlogImage.blog_session_id == blog_id
    ).all()

    images_data = [
        {
            "filename": img.filename,
            "alt": img.alt,
            "caption": img.caption,
            "image_data": base64.b64encode(img.image_data).decode("utf-8") if img.image_data else None
        }
        for img in images
    ]

    return {
        "blog": blog,
        "images": images_data
    }
from app.api.routes.auth import router as auth_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.blog import router as blog_router

app = FastAPI(
    title="LangGraph Blog Generator API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    auth_router,
    prefix="/auth",
    tags=["Auth"]
)

app.include_router(blog_router)


@app.get("/")
def home():
    return {
        "message": "LangGraph Blog Generator API 🚀"
    }
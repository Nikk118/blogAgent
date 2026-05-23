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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(blog_router)


@app.get("/")
def home():
    return {
        "message": "LangGraph Blog Generator API 🚀"
    }
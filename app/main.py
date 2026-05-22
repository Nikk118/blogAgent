from fastapi import FastAPI
from app.api.routes.blog import router as blog_router

app = FastAPI(
    title="LangGraph Blog Generator API",
    version="1.0.0"
)

app.include_router(blog_router)


@app.get("/")
def home():
    return {
        "message": "LangGraph Blog Generator API 🚀"
    }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

from blogAgent import run

app = FastAPI()

# Request schema
class BlogRequest(BaseModel):
    topic: str

# Response schema (optional but clean)
class BlogResponse(BaseModel):
    topic: str
    result: dict

@app.get("/")
def home():
    return {"message": "LangGraph Blog Generator API 🚀"}

@app.post("/generate", response_model=BlogResponse)
def generate_blog(req: BlogRequest):
    try:
        result = run(req.topic)

        return {
            "topic": req.topic,
            "result": result
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
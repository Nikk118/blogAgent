from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import (
    PydanticOutputParser,
)

from app.schemas.blog import (
    Plan,
    RouterDecision,
    EvidencePack,
    GlobalImagePlan,
)

load_dotenv()


llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7,
    tags=["blog-agent"]
)


plan_parser = PydanticOutputParser(
    pydantic_object=Plan
)

router_parser = PydanticOutputParser(
    pydantic_object=RouterDecision
)

research_parser = PydanticOutputParser(
    pydantic_object=EvidencePack
)

decide_image_parser = PydanticOutputParser(
    pydantic_object=GlobalImagePlan
)
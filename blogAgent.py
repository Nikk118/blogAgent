from __future__ import annotations
from langchain_core.prompts import PromptTemplate
import operator
from typing import Literal, TypedDict,List,Annotated
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,END,START
from langchain_core.messages import SystemMessage,HumanMessage 
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import Send
import os
import time


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7
)


class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )

    section_type: Literal[
        "intro",
        "core",
        "examples",
        "checklist",
        "common_mistakes",
        "conclusion",
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )

class Plan(BaseModel):
    blog_title:str
    audience:str=Field(...,description="The target audience for the blog post")
    tone:str=Field(...,description="The desired tone of the blog post, e.g., formal, casual, humorous")
    tasks:List[Task]

class State(TypedDict):
    topic:str
    plan:Plan
    sections:Annotated[List[str],operator.add]
    final:str


plan_parser=PydanticOutputParser(pydantic_object=Plan)



def orachestrator(state: State) -> dict:
    system_prompt = """
You are a senior technical writer and developer advocate.

Your task is to generate a highly actionable, deeply technical outline for a developer-focused blog post. The output will be used to automatically generate a full blog, so precision and structure are critical.

OBJECTIVE:
Create a structured plan (5–7 sections) that teaches the topic clearly, progressively, and practically.

HARD REQUIREMENTS:
- Create EXACTLY 5 to 7 sections (tasks).
- Each section MUST include:
  1. title: concise and specific
  2. goal: exactly ONE sentence explaining what the reader will be able to do/understand after the section
  3. bullets: 3 to 5 concrete, specific, non-overlapping action points
  4. target_words: integer between 120 and 450
  5. section_type: one of ["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"]

- Include EXACTLY ONE section where section_type = "common_mistakes".

STYLE AND QUALITY RULES:
- Audience is a software developer; use correct technical terminology.
- Do not be generic or high-level.
- Every bullet must be actionable, testable, and specific.
- Avoid vague phrases like "Explain X" or "Discuss Y".
- Use phrasing like:
  - "Implement X using Y"
  - "Compare A vs B with example"
  - "Show minimal code snippet for Z"
  - "Measure performance using..."

STRUCTURE GUIDANCE:
Follow a logical engineering flow:
1. Problem definition / motivation
2. Intuition / mental model
3. Core concepts
4. Implementation
5. Trade-offs / limitations
6. Testing / debugging / observability
7. Conclusion / next steps

MANDATORY COVERAGE:
- At least one section must include a minimal working example (MWE) or code sketch.
- At least one section must include one or more of:
  - edge cases or failure modes
  - performance or cost considerations
  - security or privacy considerations (if relevant)
  - debugging or observability (logs, metrics, traces)

ORDERING RULES:
- Start with an "intro" section.
- End with a "conclusion" or "checklist".
- Place "common_mistakes" after core sections, not at the beginning.

OUTPUT:
Return ONLY valid JSON.
Do not include explanations, markdown, or extra text.

{format_instructions}
"""

    messages = [
        SystemMessage(content=system_prompt.format(
            format_instructions=plan_parser.get_format_instructions()
        )),
        HumanMessage(content=f"Topic: {state['topic']}")
    ]

    response = llm.invoke(messages)

    # 🔥 Debug (important while building)
    print("RAW PLAN RESPONSE:\n", response.content)

    plan = plan_parser.parse(response.content)

    return {"plan": plan, "sections": []}
def fanout(state:State):
    return [Send("worker",{"task":task,"topic":state["topic"],"plan":state["plan"]}) for task in state["plan"].tasks]

worker_prompt = """You are a senior technical writer and developer advocate. Write ONE section of a technical blog post.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to the Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).

Technical quality bar:
- Be precise and implementation-oriented (developers should be able to apply it).
- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.
- When relevant, include at least one of:
  * a small code snippet (minimal, correct, and idiomatic)
  * a tiny example input/output
  * a checklist of steps
  * a diagram described in text (e.g., "flow: A -> B -> C")
- Explain trade-offs briefly (performance, cost, complexity, reliability).
- Call out edge cases / failure modes and what to do about them.
- If you mention a best practice, add the "why" in one sentence.

Markdown style:
- Start with a "## <Section Title>" heading.
- Use short paragraphs, bullet lists where helpful, and code fences for code.
- Avoid fluff. Avoid marketing language.
- If you include code, keep it focused on the bullet being addressed.
"""

def worker(payload:dict):
    task=payload["task"]
    topic=payload["topic"]
    plan=payload["plan"]

    blog_title=plan.blog_title
    bullets_text = "\n".join(f"- {b}" for b in task.bullets)
    # Implement a simple retry for 503 errors
    for attempt in range(3):
        try:
            section_md=llm.invoke(
                [
                SystemMessage(content=worker_prompt),
                HumanMessage(
    content=f"""Blog: {plan.blog_title}
Audience: {plan.audience}
Tone: {plan.tone}
Topic: {topic}

Section: {task.title}
Section type: {task.section_type}
Goal: {task.goal}
Target words: {task.target_words}

Bullets:
{bullets_text}
"""
)    
            ]
            ).content.strip()
            return {"sections":[section_md]}
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e

    return {"sections":[]}

from pathlib import Path

def reducer(state:State)->dict:
    title=state["plan"].blog_title
    sections = state.get("sections", [])
    print(f"DEBUG: Title='{title}', Sections count={len(sections)}")
    
    body="\n\n".join(sections).strip()
    final_md=f"# {title}\n\n{body}\n"
    
    # Clean filename of illegal characters like ':'
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_")).strip()
    filename = safe_title.lower().replace(" ", "_") + ".md"
    print(f"DEBUG: Filename='{filename}'")
    
    output_path=Path(filename)
    output_path.write_text(final_md,encoding="utf-8")
    return {"final":final_md}       


g=StateGraph(State)
g.add_node("orachestrator",orachestrator)
g.add_node("worker",worker)
g.add_node("reducer",reducer)

g.add_edge(START,"orachestrator")
g.add_conditional_edges("orachestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app=g.compile()

out=app.invoke({"topic":"write a blog on self attention"})


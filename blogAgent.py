from __future__ import annotations
from pathlib import Path
from langchain_core.prompts import PromptTemplate
import operator
from typing import Literal, Optional, TypedDict,List,Annotated
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,END,START
from langchain_core.messages import SystemMessage,HumanMessage 
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import Send
import os
import time
from langchain_community.tools.tavily_search import TavilySearchResults


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
    section_type: str = "core"
    tags:List[str]=Field(default_factory=list)
    requires_research:bool=False
    require_citations:bool=False
    require_code:bool=False

class Plan(BaseModel):
    blog_title:str
    audience:str=Field(...,description="The target audience for the blog post")
    tone:str=Field(...,description="The desired tone of the blog post, e.g., formal, casual, humorous")
    blog_kind:Literal["explainer","tutorial","news_roundup","comparison","system_design"]="explainer"
    constraints:List[str]=Field(default_factory=list)
    tasks:List[Task]



class EvidenceItem(BaseModel):
    title:str
    url:str
    published_at:Optional[str]=None
    snippet:Optional[str]=None
    source:Optional[str]=None

class RouterDescion(BaseModel):
    needs_research:bool=False
    mode:Literal["closed_book","hybrid","open_book"]
    queries:List[str]=Field(default_factory=list)

class EvedencePack(BaseModel):
    evidence:List[EvidenceItem]=Field(default_factory=list)

class State(TypedDict):
    topic:str
    # router:RouterDescion
    mode:str
    needs_research:bool
    queries:List[str]
    evidence:List[EvedencePack]
    plan:Optional[Plan]=None

    # workers
    sections:Annotated[List[tuple[int,str]],operator.add]
    final:str


plan_parser=PydanticOutputParser(pydantic_object=Plan)
router_parser=PydanticOutputParser(pydantic_object=RouterDescion)
research_parser=PydanticOutputParser(pydantic_object=EvedencePack)



# prompts
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



ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.

OUTPUT:
Return ONLY valid JSON.
Do not include explanations, markdown, or extra text.

{format_instructions}
"""

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
- If missing or unclear, set published_at=null. DO NOT guess.
- Keep snippets short.
- Deduplicate by URL.

OUTPUT:
Return ONLY valid JSON.
Do not include explanations, markdown, or extra text.
The output MUST be a JSON object with a single key "evidence" containing a list of objects.

{format_instructions}
"""
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



def _tavily_search(query:str,max_results:int=5)->List[dict]:
    tool=TavilySearchResults(max_results=max_results)
    try:
        results=tool.invoke({"query":query})
    except Exception as e:
        print(f"Tavily search error for query '{query}': {e}")
        return []

    normalized:List[dict]=[]
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
                "published_at": r.get("published_at"),
                "source": r.get("source")
            }
        ) 
    return normalized


def router_node(state:State)->dict:
    topic=state["topic"]
    decision_obj = router_parser.parse(llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM.format(format_instructions=router_parser.get_format_instructions())),
        HumanMessage(content=f"Topic: {topic}")
    ]).content)

    return {
        "needs_research": decision_obj.needs_research,
        "mode": decision_obj.mode,
        "queries": decision_obj.queries
    }


def route_next(state:State)->str:
    return "research" if state["needs_research"] else "orachestrator"


def research_node(state:State)->dict:
    queries=state["queries"]
    max_results=6
    raw_results:List[dict]=[]

    for q in queries:
        raw_results.extend(_tavily_search(q,max_results=max_results))

    if not raw_results:
        return {"evidence":[]}

    pack_content = llm.invoke([
        SystemMessage(content=RESEARCH_SYSTEM.format(format_instructions=research_parser.get_format_instructions())),
        HumanMessage(content=f"Raw results:\n{raw_results}")
    ]).content
    
    # Clean possible markdown wrap from the response if the LLM adds it anyway
    cleaned_content = pack_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.split("```", 1)[1].split("```", 1)[0].strip()

    import json
    try:
        data = json.loads(cleaned_content)
        # If the LLM returned a list instead of an object with "evidence" key
        if isinstance(data, list):
            data = {"evidence": data}
        # Filter out empty or invalid evidence items
        filtered_evidence = [e for e in data.get("evidence", []) if e.get("title") and e.get("url")]
        data["evidence"] = filtered_evidence
        pack = EvedencePack(**data)
    except Exception:
        # Fallback to the parser which might handle the parsing error with a re-try prompt or more context
        pack = research_parser.parse(pack_content)

    return {"evidence": [pack]}

def orachestrator(state: State) -> dict:
    evidence_packs = state.get("evidence", [])
    all_evidence = []
    for pack in evidence_packs:
        all_evidence.extend(pack.evidence)
        
    mode = state.get("mode", "closed_book")
    messages = [
        SystemMessage(content=system_prompt.format(
            format_instructions=plan_parser.get_format_instructions()
        )),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n\n"
            f"Evidence(ONLY use for fresh claims:may be empty):\n"
            f"{[e.model_dump() for e in all_evidence][:16]}"
        ))
    ]

    response = llm.invoke(messages)
    print("RAW PLAN RESPONSE:\n", response.content)

    # Strip markdown fences if present
    cleaned = response.content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    plan = plan_parser.parse(cleaned)

    return {"plan": plan, "sections": []}

def fanout(state:State):
    evidence_packs=state.get("evidence",[])
    all_evidence = []
    for pack in evidence_packs:
        all_evidence.extend(pack.evidence)
        
    return [Send("worker",{
        "task":task,
        "topic":state["topic"],
        "mode":state["mode"],
        "plan":state["plan"],
        "evidence":[e.model_dump() for e in all_evidence],
        }) for task in state["plan"].tasks]


def worker(payload:dict):
    task=payload["task"]
    topic=payload["topic"]
    plan=payload["plan"]
    evidence=[EvidenceItem(**e) for e in payload.get("evidence",[])] 
    mode=payload.get("mode","closed_book")
    bullets_text = "\n".join(f"- {b}" for b in task.bullets)
    evidence_text=""
    if evidence:
        evidence_text="\n".join(
            f"-{e.title}|{e.url}|{e.published_at or 'date:unknown'}".strip() for e in evidence[:20]
        )
    
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
Blog kind:{plan.blog_kind}
Mode: {mode}
Evidence:(ONLY use these URLs when citing):\n
{evidence_text}\n
Target words: {task.target_words}
Constraints: {plan.constraints}
Tags: {task.tags}
Requires research: {task.requires_research}
Require citations: {task.require_citations}
Require code: {task.require_code}
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
            return {"sections":[(task.id,section_md)]}
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e

    return {"sections":[]}


def reducer(state:State)->dict:
    plan=state["plan"]
    # Check if sections exist and handle potential empty list
    sections_list = state.get("sections", [])
    if not sections_list:
        print("DEBUG: No sections found in state!")
        return {"final": ""}
    
    # Sort and extract markdown content
    ordered_sections=[md for _, md in sorted(sections_list, key=lambda x: x[0])]
    body="\n\n".join(ordered_sections).strip()
    final_md=f"# {plan.blog_title}\n\n{body}\n"
    
    # Sanitize filename: remove illegal chars like : \ / * ? " < > |
    import re
    safe_title = re.sub(r'[<>:"/\\|?*]', '', plan.blog_title)
    filename = f"{safe_title.strip().replace(' ', '_').lower()}.md"
    
    print(f"DEBUG: Title='{plan.blog_title}', Sections count={len(sections_list)}")
    print(f"DEBUG: Writing to filename='{filename}'")
    
    output_path=Path(filename)
    output_path.write_text(final_md,encoding="utf-8")
    return {"final":final_md}       


g=StateGraph(State)
g.add_node("router",router_node)
g.add_node("research",research_node)
g.add_node("orachestrator",orachestrator)
g.add_node("worker",worker)
g.add_node("reducer",reducer)

g.add_edge(START,"router")
g.add_conditional_edges("router",route_next,{"research":"research","orachestrator":"orachestrator"})
g.add_edge("research", "orachestrator")
g.add_conditional_edges("orachestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app=g.compile()

def run(topic:str):
    out=app.invoke({
        "topic":topic,
        "mode":"",
        "needs_research":False,
        "queries":[],
        "evidence":[],
        "plan":None,
        "sections":[],
        "final":"",
    })

    return out


run("wirte a blog on stock marcket in 2026")

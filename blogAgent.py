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
import re
from urllib.parse import urlparse
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7,
    tags=["blog-agent"]

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
    requires_research:bool=Field(..., description="True if section needs external info or research")
    require_citations:bool=Field(..., description="True if claims need references or sources")
    require_code:bool=Field(..., description="True if section includes code examples")

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


from pydantic import field_validator

class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "2048x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "high", "medium"] = "medium"

    @field_validator("size", mode="before")
    @classmethod
    def coerce_size(cls, v):
        valid = {"1024x1024", "2048x1536", "1536x1024"}
        return v if v in valid else "1024x1024"

    @field_validator("quality", mode="before")
    @classmethod
    def coerce_quality(cls, v):
        valid = {"low", "medium", "high"}
        return v if v in valid else "medium"
class GlobalImagePlan(BaseModel):
    md_with_placeholders:str
    images:List[ImageSpec]=Field(default_factory=list)



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
    # reducer
    merged_md:str
    md_with_placeholders:str
    image_specs:List[dict]

    final:str




plan_parser=PydanticOutputParser(pydantic_object=Plan)
router_parser=PydanticOutputParser(pydantic_object=RouterDescion)
research_parser=PydanticOutputParser(pydantic_object=EvedencePack)
decide_image_parser=PydanticOutputParser(pydantic_object=GlobalImagePlan)



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
- For EACH section:
    - Set requires_research = true if external data, tools, or recent info is needed
    - Set require_citations = true if any claims should be backed by sources
    - Set require_code = true if code examples or implementation are included
- IMPORTANT: These fields are REQUIRED and must not be omitted.

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
                "source": extract_source(r.get("url", ""))
            }
        ) 
    return normalized


def extract_source(url: str) -> str:
    try:
        return urlparse(str(url)).netloc.replace("www.", "")
    except Exception:
        return ""


def _normalize_published_at(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    return m.group(0) if m else None


def _build_evidence_pack_from_results(results: List[dict]) -> EvedencePack:
    dedup: dict[str, dict] = {}
    for item in results or []:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        if url in dedup:
            continue
        dedup[url] = {
            "title": str(item.get("title", "")).strip() or url,
            "url": url,
            "snippet": str(item.get("snippet", "")).strip() or None,
            "published_at": _normalize_published_at(item.get("published_at")),
            "source": str(item.get("source") or extract_source(item.get("url", ""))).strip(),
        }

    return EvedencePack(evidence=[EvidenceItem(**v) for v in dedup.values()])


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
        pack = _build_evidence_pack_from_results(data.get("evidence", []))
    except Exception:
        # Deterministic fallback when LLM JSON is malformed (prevents OUTPUT_PARSING_FAILURE)
        pack = _build_evidence_pack_from_results(raw_results)

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
            "Evidence (use this to decide if sections require research, citations, or external references):\n"
            f"{[e.model_dump() for e in all_evidence][:16]}\n\n"
            "If evidence is provided, mark relevant sections with requires_research=true and require_citations=true."
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

    # Safety net: if evidence exists but LLM missed flags, force at least one research-backed section.
    if all_evidence and plan.tasks and not any(t.requires_research for t in plan.tasks):
        plan.tasks[0].requires_research = True
        plan.tasks[0].require_citations = True

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




# =====================
# reducrerWithImage
# =====================
def merge_content(state:State)->dict:
    plan=state["plan"]
    ordered_sections=[md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body="\n\n".join(ordered_sections).strip()
    merged_md=f"#{plan.blog_title}\n\n{body}\n"
    return {"merged_md":merged_md}

DECIDE_IMAGES_SYSTEM = """You are a technical editor. Return ONLY a JSON object, no markdown, no explanation.

Given a blog post in markdown, you must:
1. Decide where 1-3 technical diagrams would help understanding
2. Insert placeholders [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] into the markdown
3. Describe each image

Return this exact JSON structure:
{{
  "md_with_placeholders": "<the full markdown with [[IMAGE_N]] tags inserted>",
  "images": [
    {{
      "placeholder": "[[IMAGE_1]]",
      "filename": "diagram1.png",
      "alt": "description of image",
      "caption": "Figure 1: caption text",
      "prompt": "detailed prompt to generate this diagram",
      "size": "1024x1024",
      "quality": "medium"
    }}
  ]
}}

If no images are needed, return:
{{
  "md_with_placeholders": "<original markdown unchanged>",
  "images": []
}}
IMPORTANT: The "size" field must be EXACTLY one of: "1024x1024", "2048x1536", "1536x1024".
The "quality" field must be EXACTLY one of: "low", "medium", "high".
Do not use any other values.
"""
def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    response = llm.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
            f"Topic: {state['topic']}\n\n"
            "Return a JSON object with 'md_with_placeholders' (full markdown with [[IMAGE_N]] inserted) "
            "and 'images' (list of image specs). Do NOT return bullet points or plain text.\n\n"
            f"{merged_md}"
        ))
    ])

    cleaned = response.content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    # Fallback: if the LLM still returns garbage, skip images entirely
    try:
        image_plan = decide_image_parser.parse(cleaned)
    except Exception as e:
        print(f"[decide_images] parse failed ({e}), skipping images.")
        return {
            "md_with_placeholders": merged_md,
            "image_specs": []
        }

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images]
    }
def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")

def generate_andplace_image(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    image_specs = state.get("image_specs") or []

    # ✅ create output folder
    output_dir = Path("generated_blogs")
    output_dir.mkdir(exist_ok=True)

    # sanitize filename
    safe_title = re.sub(r'[<>:"/\\|?*]', '', plan.blog_title)
    blog_filename = f"{safe_title.strip().replace(' ', '_').lower()}.md"

    blog_path = output_dir / blog_filename

    # ✅ images folder inside generated_blogs
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # --- if no images ---
    if not image_specs:
        blog_path.write_text(md, encoding="utf-8")
        return {"final": md}

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **IMAGE GENERATION FAILED** {spec.get('caption', '')}\n>\n"
                    f"> **ALT:** {spec.get('alt', '')}\n>\n"
                    f"> **PROMPT:** {spec['prompt']}\n>\n"
                    f"> **ERROR:** {e}\n>\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        # ✅ fix path for markdown
        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    # save final markdown
    blog_path.write_text(md, encoding="utf-8")

    return {"final": md}     
# ==============
# reducer sub graph
# ==============

reducer_graph=StateGraph(State)
reducer_graph.add_node("merge_content",merge_content)
reducer_graph.add_node("decide_images",decide_images)
reducer_graph.add_node("generate_andplace_image",generate_andplace_image)

reducer_graph.add_edge(START,"merge_content")
reducer_graph.add_edge("merge_content","decide_images")
reducer_graph.add_edge("decide_images","generate_andplace_image")
reducer_graph.add_edge("generate_andplace_image",END)

reducer_subgraph=reducer_graph.compile()



# ==========
# main graph
# ==========
g=StateGraph(State)
g.add_node("router",router_node)
g.add_node("research",research_node)
g.add_node("orachestrator",orachestrator)
g.add_node("worker",worker)
g.add_node("reducer",reducer_subgraph)

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




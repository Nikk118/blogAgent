import time

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.blog_agent import llm
from app.core.prompts import WORKER_PROMPT
from app.schemas.blog import EvidenceItem


def worker(payload):

    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    mode = payload.get("mode", "closed_book")

    evidence = [
        EvidenceItem(**e)
        for e in payload.get("evidence", [])
    ]

    bullets_text = "\n".join(
        f"- {bullet}"
        for bullet in task.bullets
    )

    evidence_text = ""

    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:20]
        )

    for attempt in range(3):

        try:

            response = llm.invoke([
                SystemMessage(content=WORKER_PROMPT),

                HumanMessage(
                    content=f"""
Blog: {plan.blog_title}
Audience: {plan.audience}
Tone: {plan.tone}

Topic: {topic}
Blog kind: {plan.blog_kind}
Mode: {mode}

Evidence:
{evidence_text}

Target words: {task.target_words}
Constraints: {plan.constraints}
Tags: {task.tags}

Requires research: {task.requires_research}
Require citations: {task.require_citations}
Require code: {task.require_code}

Section: {task.title}
Section type: {task.section_type}

Goal: {task.goal}

Bullets:
{bullets_text}
"""
                )
            ])

            section_md = response.content.strip()

            return {
                "sections": [
                    (task.id, section_md)
                ]
            }

        except Exception as e:

            if "503" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            raise e

    return {
        "sections": []
    }
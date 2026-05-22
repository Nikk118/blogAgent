from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from app.agents.blog_agent import (
    llm,
    plan_parser,
)

from app.core.prompts import (
    SYSTEM_PROMPT,
)


def orchestrator(state):

    evidence_packs = state.get(
        "evidence",
        []
    )

    all_evidence = []

    for pack in evidence_packs:
        all_evidence.extend(
            pack.evidence
        )

    mode = state.get(
        "mode",
        "closed_book"
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                format_instructions=
                plan_parser.get_format_instructions()
            )
        ),

        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n\n"

                "Evidence:\n"

                f"{[e.model_dump() for e in all_evidence][:16]}\n\n"

                "If evidence exists, mark relevant sections "
                "with requires_research=true and "
                "require_citations=true."
            )
        )
    ]

    response = llm.invoke(messages)

    cleaned = response.content.strip()

    if cleaned.startswith("```json"):
        cleaned = (
            cleaned
            .split("```json", 1)[1]
            .split("```", 1)[0]
            .strip()
        )

    elif cleaned.startswith("```"):
        cleaned = (
            cleaned
            .split("```", 1)[1]
            .split("```", 1)[0]
            .strip()
        )

    plan = plan_parser.parse(cleaned)

    # fallback safety
    if (
        all_evidence
        and plan.tasks
        and not any(
            t.requires_research
            for t in plan.tasks
        )
    ):
        plan.tasks[0].requires_research = True
        plan.tasks[0].require_citations = True

    return {
        "plan": plan,
        "sections": [],
    }
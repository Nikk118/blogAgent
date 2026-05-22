import json

from typing import List

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from app.agents.blog_agent import (
    llm,
    research_parser,
)

from app.core.prompts import (
    RESEARCH_SYSTEM,
)

from app.utils.research import (
    tavily_search,
    build_evidence_pack_from_results,
)


def research_node(state):

    queries = state["queries"]

    max_results = 6

    raw_results: List[dict] = []

    for query in queries:

        raw_results.extend(

            tavily_search(
                query,
                max_results=max_results
            )
        )

    if not raw_results:

        return {
            "evidence": []
        }

    response = llm.invoke([

        SystemMessage(
            content=RESEARCH_SYSTEM.format(
                format_instructions=
                research_parser.get_format_instructions()
            )
        ),

        HumanMessage(
            content=f"Raw results:\n{raw_results}"
        )
    ])

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

    try:

        data = json.loads(cleaned)

        if isinstance(data, list):

            data = {
                "evidence": data
            }

        pack = build_evidence_pack_from_results(
            data.get("evidence", [])
        )

    except Exception:

        pack = build_evidence_pack_from_results(
            raw_results
        )

    return {
        "evidence": [pack]
    }
from __future__ import annotations

import operator
from typing import TypedDict, List, Optional, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.schemas.blog import (
    Plan,
    EvidencePack,
)

from app.graphs.nodes.router import router_node
from app.graphs.nodes.research import research_node
from app.graphs.nodes.planner import orchestrator
from app.graphs.nodes.worker import worker

from app.graphs.nodes.reducer import (
    merge_content,
    decide_images,
)

from app.services.image_service import (
    generate_and_place_image,
)


# =========================
# STATE
# =========================

class State(TypedDict):

    topic: str

    mode: str
    needs_research: bool
    queries: List[str]

    evidence: List[EvidencePack]

    plan: Optional[Plan]

    sections: Annotated[
        List[tuple[int, str]],
        operator.add
    ]

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# =========================
# ROUTING
# =========================

def route_next(state: State) -> str:

    return (
        "research"
        if state["needs_research"]
        else "orchestrator"
    )


# =========================
# FANOUT
# =========================

def fanout(state: State):

    evidence_packs = state.get(
        "evidence",
        []
    )

    all_evidence = []

    for pack in evidence_packs:
        all_evidence.extend(pack.evidence)

    plan = state["plan"]

    if not plan:
        return []

    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": plan,
                "evidence": [
                    e.model_dump()
                    for e in all_evidence
                ],
            },
        )
        for task in plan.tasks
    ]


# =========================
# REDUCER SUBGRAPH
# =========================

reducer_graph = StateGraph(State)

reducer_graph.add_node(
    "merge_content",
    merge_content
)

reducer_graph.add_node(
    "decide_images",
    decide_images
)

reducer_graph.add_node(
    "generate_and_place_image",
    generate_and_place_image
)

reducer_graph.add_edge(
    START,
    "merge_content"
)

reducer_graph.add_edge(
    "merge_content",
    "decide_images"
)

reducer_graph.add_edge(
    "decide_images",
    "generate_and_place_image"
)

reducer_graph.add_edge(
    "generate_and_place_image",
    END
)

reducer_subgraph = reducer_graph.compile()


# =========================
# MAIN GRAPH
# =========================

graph = StateGraph(State)

graph.add_node(
    "router",
    router_node
)

graph.add_node(
    "research",
    research_node
)

graph.add_node(
    "orchestrator",
    orchestrator
)

graph.add_node(
    "worker",
    worker
)

graph.add_node(
    "reducer",
    reducer_subgraph
)

graph.add_edge(
    START,
    "router"
)

graph.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)

graph.add_edge(
    "research",
    "orchestrator"
)

graph.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"],
)

graph.add_edge(
    "worker",
    "reducer"
)

graph.add_edge(
    "reducer",
    END
)

app_graph = graph.compile()


# =========================
# ENTRYPOINT
# =========================

def run(topic: str):

    return app_graph.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
    )
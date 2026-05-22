from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from app.agents.blog_agent import (
    llm,
    router_parser,
)

from app.core.prompts import (
    ROUTER_SYSTEM,
)


def router_node(state):

    topic = state["topic"]

    response = llm.invoke([

        SystemMessage(
            content=ROUTER_SYSTEM.format(
                format_instructions=
                router_parser.get_format_instructions()
            )
        ),

        HumanMessage(
            content=f"Topic: {topic}"
        )
    ])

    decision = router_parser.parse(
        response.content
    )

    return {

        "needs_research":
        decision.needs_research,

        "mode":
        decision.mode,

        "queries":
        decision.queries,
    }
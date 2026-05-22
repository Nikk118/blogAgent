from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from app.agents.blog_agent import (
    llm,
    decide_image_parser,
)

from app.core.prompts import (
    DECIDE_IMAGES_SYSTEM,
)

from app.services.image_service import (
    generate_and_place_image,
)


# =========================
# MERGE CONTENT
# =========================

def merge_content(state):

    plan = state["plan"]

    ordered_sections = [
        md
        for _, md in sorted(
            state["sections"],
            key=lambda x: x[0]
        )
    ]

    body = "\n\n".join(
        ordered_sections
    ).strip()

    merged_md = (
        f"# {plan.blog_title}\n\n"
        f"{body}\n"
    )

    return {
        "merged_md": merged_md
    }


# =========================
# DECIDE IMAGES
# =========================

def decide_images(state):

    merged_md = state["merged_md"]

    plan = state["plan"]

    assert plan is not None

    response = llm.invoke([

        SystemMessage(
            content=DECIDE_IMAGES_SYSTEM
        ),

        HumanMessage(
            content=(

                f"Blog: {plan.blog_title}\n"
                f"Topic: {state['topic']}\n\n"

                "Return JSON with:\n"
                "- md_with_placeholders\n"
                "- images\n\n"

                f"{merged_md}"
            )
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

        image_plan = decide_image_parser.parse(
            cleaned
        )

    except Exception:

        return {
            "md_with_placeholders": merged_md,
            "image_specs": []
        }

    return {

        "md_with_placeholders":
        image_plan.md_with_placeholders,

        "image_specs": [
            img.model_dump()
            for img in image_plan.images
        ]
    }
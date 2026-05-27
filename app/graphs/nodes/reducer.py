import time
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
# RETRY HELPER
# =========================

def invoke_with_retry(llm, messages, max_attempts=3, wait_seconds=15):
    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            is_rate_limit = (
                "429" in str(e)
                or "rate_limited" in str(e).lower()
                or "rate limit" in str(e).lower()
            )
            if is_rate_limit and attempt < max_attempts - 1:
                time.sleep(wait_seconds)
            else:
                raise


# =========================
# DECIDE IMAGES
# =========================

def decide_images(state):

    merged_md = state["merged_md"]

    plan = state["plan"]

    assert plan is not None

    messages = [
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
    ]

    response = invoke_with_retry(llm, messages)

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
        image_plan = decide_image_parser.parse(cleaned)

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
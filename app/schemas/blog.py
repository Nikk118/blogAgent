from typing import Literal, Optional, List

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


# =========================
# TASK
# =========================

class Task(BaseModel):

    id: int

    title: str

    goal: str = Field(
        ...,
        description=(
            "One sentence describing what "
            "the reader should understand "
            "after this section."
        ),
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description=(
            "3-5 concrete, non-overlapping "
            "subpoints."
        ),
    )

    target_words: int = Field(
        ...,
        description="Target word count.",
    )

    section_type: str = "core"

    tags: List[str] = Field(
        default_factory=list
    )

    requires_research: bool = Field(
        ...,
        description=(
            "True if external research is needed."
        ),
    )

    require_citations: bool = Field(
        ...,
        description=(
            "True if claims need citations."
        ),
    )

    require_code: bool = Field(
        ...,
        description=(
            "True if section contains code."
        ),
    )


# =========================
# PLAN
# =========================

class Plan(BaseModel):

    blog_title: str

    audience: str = Field(
        ...,
        description="Target audience",
    )

    tone: str = Field(
        ...,
        description="Writing tone",
    )

    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
        "analytical",
    ] = "explainer"

    constraints: List[str] = Field(
        default_factory=list
    )

    tasks: List[Task]
    @field_validator("blog_kind", mode="before")
    @classmethod
    def coerce_blog_kind(cls, v):
        valid = {
            "explainer",
            "tutorial",
            "news_roundup",
            "comparison",
            "system_design",
            "analytical",
        }
        return v if v in valid else "explainer"

# =========================
# EVIDENCE
# =========================

class EvidenceItem(BaseModel):

    title: str

    url: str

    published_at: Optional[str] = None

    snippet: Optional[str] = None

    source: Optional[str] = None


class RouterDecision(BaseModel):

    needs_research: bool = False

    mode: Literal[
        "closed_book",
        "hybrid",
        "open_book",
    ]

    queries: List[str] = Field(
        default_factory=list
    )


class EvidencePack(BaseModel):

    evidence: List[EvidenceItem] = Field(
        default_factory=list
    )


# =========================
# IMAGES
# =========================

class ImageSpec(BaseModel):

    placeholder: str

    filename: str

    alt: str

    caption: str

    prompt: str

    size: Literal[
        "1024x1024",
        "2048x1536",
        "1536x1024",
    ] = "1024x1024"

    quality: Literal[
        "low",
        "medium",
        "high",
    ] = "medium"

    @field_validator("size", mode="before")
    @classmethod
    def coerce_size(cls, value):

        valid = {
            "1024x1024",
            "2048x1536",
            "1536x1024",
        }

        return (
            value
            if value in valid
            else "1024x1024"
        )

    @field_validator("quality", mode="before")
    @classmethod
    def coerce_quality(cls, value):

        valid = {
            "low",
            "medium",
            "high",
        }

        return (
            value
            if value in valid
            else "medium"
        )


class GlobalImagePlan(BaseModel):

    md_with_placeholders: str

    images: List[ImageSpec] = Field(
        default_factory=list
    )
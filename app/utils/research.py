import re

from typing import List, Optional
from urllib.parse import urlparse

from langchain_community.tools.tavily_search import TavilySearchResults

from app.schemas.blog import EvidenceItem, EvidencePack


def tavily_search(query: str, max_results: int = 5) -> List[dict]:

    tool = TavilySearchResults(max_results=max_results)

    try:
        results = tool.invoke({"query": query})

    except Exception as e:
        print(f"Tavily search error for query '{query}': {e}")
        return []

    normalized = []

    for result in results or []:

        normalized.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get(
                "content",
                result.get("snippet", "")
            ),
            "published_at": result.get("published_at"),
            "source": extract_source(
                result.get("url", "")
            ),
        })

    return normalized


def extract_source(url: str) -> str:

    try:
        return (
            urlparse(str(url))
            .netloc
            .replace("www.", "")
        )

    except Exception:
        return ""


def normalize_published_at(
    value: Optional[str]
) -> Optional[str]:

    if not value:
        return None

    text = str(value).strip()

    match = re.search(
        r"\b\d{4}-\d{2}-\d{2}\b",
        text
    )

    return match.group(0) if match else None


def build_evidence_pack_from_results(
    results: List[dict]
) -> EvidencePack:

    dedup = {}

    for item in results or []:

        url = str(
            item.get("url", "")
        ).strip()

        if not url:
            continue

        if url in dedup:
            continue

        dedup[url] = {
            "title": str(
                item.get("title", "")
            ).strip() or url,

            "url": url,

            "snippet": str(
                item.get("snippet", "")
            ).strip() or None,

            "published_at": normalize_published_at(
                item.get("published_at")
            ),

            "source": str(
                item.get("source")
                or extract_source(url)
            ).strip(),
        }

    return EvidencePack(
        evidence=[
            EvidenceItem(**value)
            for value in dedup.values()
        ]
    )
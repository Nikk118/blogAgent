from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from blogAgent import app


GENERATED_IMAGES_DIR = Path("generated_blogs/images")


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


def extract_title_from_md(md: str, fallback: str = "blog") -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            return title or fallback
    return fallback


def resolve_generated_image_path(src: str) -> Optional[Path]:
    src = src.strip().lstrip("./")
    if not src or src.startswith("http://") or src.startswith("https://"):
        return None

    raw_path = Path(src)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)

    if src.startswith("generated_blogs/images"):
        candidates.append(Path(src))
    elif src.startswith("images/"):
        rel = src.split("images/", 1)[1]
        candidates.append(GENERATED_IMAGES_DIR / rel)
    else:
        candidates.append(GENERATED_IMAGES_DIR / src)
        candidates.append(GENERATED_IMAGES_DIR / raw_path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def render_local_images_from_markdown(md: str):
    for match in _MD_IMG_RE.finditer(md):
        alt = (match.group("alt") or "").strip()
        src = (match.group("src") or "").strip()
        resolved = resolve_generated_image_path(src)
        if resolved:
            st.image(str(resolved), caption=alt or resolved.name, use_container_width=True)
        elif src and not src.startswith("http://") and not src.startswith("https://"):
            st.warning(f"Image not found: {src}")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

def _new_chat(number: int) -> dict:
    return {
        "name": f"Chat {number}",
        "history": [],
        "last_out": None,
        "logs": [],
    }


if "chats" not in st.session_state:
    st.session_state["chats"] = {"chat_1": _new_chat(1)}
if "active_chat_id" not in st.session_state or st.session_state["active_chat_id"] not in st.session_state["chats"]:
    st.session_state["active_chat_id"] = next(iter(st.session_state["chats"]))
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "pending_request" not in st.session_state:
    st.session_state["pending_request"] = None
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

chats = st.session_state["chats"]

with st.sidebar:
    st.header("Chats")
    if st.button("New Chat"):
        next_num = len(chats) + 1
        new_chat_id = f"chat_{next_num}"
        while new_chat_id in chats:
            next_num += 1
            new_chat_id = f"chat_{next_num}"
        chats[new_chat_id] = _new_chat(next_num)
        st.session_state["active_chat_id"] = new_chat_id
        st.rerun()

    chat_ids = list(chats.keys())
    active_idx = chat_ids.index(st.session_state["active_chat_id"])
    selected_chat_id = st.selectbox(
        "Conversation",
        options=chat_ids,
        index=active_idx,
        format_func=lambda cid: chats[cid]["name"],
    )
    st.session_state["active_chat_id"] = selected_chat_id

    st.divider()
    st.header("Generate")
    topic = st.text_area(
        "Topic",
        height=120,
        key=f"topic_{selected_chat_id}",
    )
    as_of = st.date_input(
        "As-of date",
        value=date.today(),
        key=f"asof_{selected_chat_id}",
    )
    run_btn = st.button(
        "Generate Blog",
        type="primary",
        disabled=st.session_state["is_generating"],
    )

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()
    st.session_state["pending_request"] = {
        "chat_id": st.session_state["active_chat_id"],
        "topic": topic.strip(),
        "as_of": as_of.isoformat(),
    }
    st.session_state["is_generating"] = True
    st.rerun()

if st.session_state["is_generating"] and st.session_state["pending_request"]:
    req = st.session_state["pending_request"]
    try:
        with st.spinner("Generating blog..."):
            out = app.invoke({
                "topic": req["topic"],
                "mode": "",
                "needs_research": False,
                "queries": [],
                "evidence": [],
                "plan": None,
                "as_of": req["as_of"],
                "recency_days": 7,
                "sections": [],
                "merged_md": "",
                "md_with_placeholders": "",
                "image_specs": [],
                "final": "",
            })

        chat = chats.get(req["chat_id"])
        if chat is not None:
            chat["last_out"] = out
            chat["history"].append({"role": "user", "content": req["topic"]})
            final_md = out.get("final") or ""
            chat["history"].append(
                {
                    "role": "assistant",
                    "content": extract_title_from_md(final_md, "Blog generated"),
                }
            )
            chat["logs"].append(f"Generated blog for topic: {req['topic']} on {req['as_of']}")
            st.session_state["last_out"] = out
    except Exception as e:
        chat = chats.get(req["chat_id"])
        if chat is not None:
            chat["logs"].append(f"Generation failed: {e}")
        st.session_state["is_generating"] = False
        st.session_state["pending_request"] = None
        st.error(f"Generation failed: {e}")
        st.stop()

    st.session_state["is_generating"] = False
    st.session_state["pending_request"] = None
    st.rerun()

active_chat = chats[st.session_state["active_chat_id"]]
out = active_chat.get("last_out")

st.subheader("Chat History")
history = active_chat.get("history", [])
if not history:
    st.caption("No messages in this chat yet.")
else:
    for msg in history[-20:]:
        with st.chat_message(msg.get("role", "assistant")):
            st.write(msg.get("content", ""))

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["Plan", "Evidence", "Markdown Preview", "Images", "Logs"]
)

with tab_plan:
    st.subheader("Plan")
    if not out:
        st.info("Generate a blog to see the plan.")
    else:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": bool(t.get("requires_research", False)),
                            "requires_citations": bool(t.get("require_citations", t.get("requires_citations", False))),
                            "requires_code": bool(t.get("require_code", t.get("requires_code", False))),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

with tab_evidence:
    st.subheader("Evidence")
    if not out:
        st.info("Generate a blog to see evidence.")
    else:
        evidence_raw = out.get("evidence") or []
        flat_evidence = []

        for item in evidence_raw:
            if hasattr(item, "model_dump"):
                item = item.model_dump()

            if isinstance(item, dict) and "evidence" in item:
                nested_items = item.get("evidence") or []
                for nested in nested_items:
                    if hasattr(nested, "model_dump"):
                        nested = nested.model_dump()
                    if isinstance(nested, dict):
                        flat_evidence.append(nested)
            elif isinstance(item, dict):
                flat_evidence.append(item)

        rows = []
        for e in flat_evidence:
            rows.append(
                {
                    "title": e.get("title") or "",
                    "published_at": e.get("published_at") or "",
                    "source": e.get("source") or "",
                    "url": e.get("url") or "",
                }
            )

        if not rows:
            st.info("No evidence found")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("URL")
                },
            )

with tab_preview:
    st.subheader("Markdown Preview")
    if not out:
        st.info("Generate a blog to see markdown output.")
    else:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            st.markdown(out["final"])
            render_local_images_from_markdown(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, GENERATED_IMAGES_DIR)
            st.download_button(
                "Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

with tab_images:
    st.subheader("Images")
    if not out:
        st.info("Generate a blog to see image outputs.")
    else:
        specs = out.get("image_specs") or []
        images_dir = GENERATED_IMAGES_DIR

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("generated_blogs/images exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

with tab_logs:
    st.subheader("Logs")
    logs = active_chat.get("logs", [])
    if not logs:
        st.info("No logs yet.")
    st.text_area("Event log", value="\n\n".join(logs[-80:]), height=520)
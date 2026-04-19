# BlogAgent

A LangGraph + Streamlit blog generation app that plans, researches, writes, and exports technical blog posts.

## What This Project Does

- Builds a structured blog plan with section tasks and writing constraints.
- Optionally performs web research with Tavily and normalizes evidence.
- Writes each section with an LLM worker node.
- Merges sections into final markdown.
- Optionally decides and generates technical images.
- Saves output markdown to generated_blogs/ and images to generated_blogs/images/.
- Provides a Streamlit UI with multi-chat history and result tabs.

## Project Structure

- blogAgent.py: LangGraph backend pipeline, Pydantic schemas, Tavily research, planning/writing, image generation, and app graph compile.
- frontend.py: Streamlit UI, multi-chat sessions, invoke flow, tabs for Plan/Evidence/Markdown/Images/Logs.
- requirement.txt: Python dependencies.
- generated_blogs/: Generated markdown outputs.
- generated_blogs/images/: Generated image assets.
- .env: Environment variables for API keys (not committed).

## Architecture (LangGraph)

Main graph:

1. router
2. research (conditional)
3. orachestrator
4. worker (fanout for section tasks)
5. reducer subgraph

Reducer subgraph:

1. merge_content
2. decide_images
3. generate_andplace_image

Output state includes final markdown and intermediate artifacts like plan, evidence, and image specs.

## Requirements

- Python 3.10+ recommended
- API keys:
  - MISTRAL_API_KEY (required for ChatMistralAI)
  - TAVILY_API_KEY (required for web research mode)
  - GOOGLE_API_KEY (optional, used only when image generation is requested)

## Setup

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\venv\Scripts\Activate.ps1)
```

### 2) Install dependencies

```powershell
pip install -r requirement.txt
```

If your requirements file encoding causes issues, open and save requirement.txt as UTF-8, then re-run install.

### 3) Configure environment variables

Create a .env file in the project root:

```env
MISTRAL_API_KEY=your_mistral_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key
```

Notes:
- GOOGLE_API_KEY is optional.
- If image generation fails, the app still produces markdown and inserts a failure note where image placeholders exist.

## Run the App

Start Streamlit:

```powershell
streamlit run frontend.py
```

Open the local URL shown in terminal.

## Streamlit UI Overview

Sidebar:
- Chats: Create a new chat and switch between conversations.
- Generate:
  - Topic input
  - As-of date
  - Generate Blog button

Main area:
- Chat History
- Tabs:
  - Plan
  - Evidence
  - Markdown Preview
  - Images
  - Logs

Behavior:
- Backend runs once per Generate click.
- Generate button is disabled while a request is in progress.
- Each chat stores its own output/history/logs.

## Outputs

- Markdown files: generated_blogs/<safe_title>.md
- Image files: generated_blogs/images/<filename>
- Markdown image links are written as images/<filename> so they resolve from generated_blogs context.

## Programmatic Backend Usage

You can call the compiled graph directly:

```python
from blogAgent import app

out = app.invoke({
    "topic": "Your topic",
    "mode": "",
    "needs_research": False,
    "queries": [],
    "evidence": [],
    "plan": None,
    "as_of": "2026-04-19",
    "recency_days": 7,
    "sections": [],
    "merged_md": "",
    "md_with_placeholders": "",
    "image_specs": [],
    "final": "",
})

print(out.get("final", ""))
```

There is also a helper:

```python
from blogAgent import run
print(run("Topic")["final"])
```

## Troubleshooting

### Evidence appears empty or malformed

- Ensure TAVILY_API_KEY is set.
- The backend includes a deterministic fallback when LLM research JSON parsing fails.

### Source domain is blank

- Source is derived from URL domain.
- If a result has no valid URL, it is skipped.

### Task flags look wrong

- Task schema requires:
  - requires_research
  - require_citations
  - require_code
- Planning prompt explicitly instructs the model to set these.

### No images generated

- Check GOOGLE_API_KEY.
- Image generation errors are handled gracefully and do not block markdown output.

## Notes

- The filename is requirement.txt (singular), not requirements.txt.
- The output folder generated_blogs/ is git-ignored in this repo.

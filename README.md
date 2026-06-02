# BlogAgent

A LangGraph + Next.js blog generation app that plans, researches, writes, and exports technical blog posts — backed by Supabase (PostgreSQL) and Firebase Authentication.

## What This Project Does

- Authenticates users via Firebase Auth (email/password, Google OAuth, etc.)
- Stores user records, blog sessions, and generated images in Supabase (PostgreSQL).
- Builds a structured blog plan with section tasks and writing constraints.
- Optionally performs web research with Tavily and normalizes evidence.
- Writes each section with an LLM worker node (ChatMistralAI).
- Merges sections into final markdown.
- Generates technical images via Pollinations AI (`image.pollinations.ai`).
- Saves session output (content JSONB) and images (binary) to Supabase tables.
- Provides a Next.js frontend with multi-chat history and tabbed result views.
- Traces every LangGraph run (nodes, LLM calls, latency, token usage) via LangSmith.

## Project Structure

```
BLOGAGENT/
├── app/                         # FastAPI backend
│   ├── db/
│   │   └── database.py          # SQLAlchemy engine + Base
│   ├── models/
│   │   ├── user.py              # User model (firebase_uid, email)
│   │   ├── blog_session.py      # BlogSession model (content JSONB)
│   │   └── blog_image.py        # BlogImage model (binary image data)
│   ├── routes/                  # FastAPI routers (auth, sessions, images)
│   └── main.py                  # FastAPI app entrypoint
├── frontend/                    # Next.js frontend
│   ├── app/                     # App Router pages
│   ├── components/              # UI components
│   └── lib/
│       ├── firebase.ts          # Firebase client init
│       └── api.ts               # API client (attaches Firebase ID token)
├── generated_blogs/             # (local dev) markdown outputs
├── .env                         # Environment variables (not committed)
├── requirements.txt
├── run.py                       # Direct LangGraph runner
└── theme.py
```

## Architecture

### LangGraph Pipeline (Backend)

```
router
  └─► research (conditional, Tavily)
        └─► orchestrator
              └─► worker × N  (fanout per section task)
                    └─► reducer subgraph
                          ├─► merge_content
                          ├─► decide_images
                          └─► generate_images  (Pollinations AI)
```

Output state: `final` markdown, `plan`, `evidence`, `image_specs`, `generated_images`.

### Database (Supabase / PostgreSQL)

Three tables managed with SQLAlchemy models:

| Table | Key Columns |
|---|---|
| `users` | `id`, `firebase_uid` (unique), `email` |
| `blog_sessions` | `id`, `user_id` → `users.id`, `title`, `prompt`, `content` (JSONB) |
| `blog_images` | `id`, `blog_session_id` → `blog_sessions.id`, `filename`, `alt`, `caption`, `image_data` (LargeBinary) |

### Authentication (Firebase)

- Users sign in through the Next.js frontend via the Firebase JS SDK.
- Firebase issues an ID token sent as `Authorization: Bearer <token>` on every API request.
- The FastAPI backend verifies the token with `firebase-admin`, resolves the `firebase_uid` to a `users` row, and scopes all queries to that user.

### Image Generation (Pollinations AI)

Images are fetched at generation time using:

```
https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true
```

The raw binary response is stored in `blog_images.image_data` and served back to the frontend via a `/images/{id}` endpoint.

### Observability (LangSmith)

LangSmith tracing is enabled by setting the environment variables below — no code changes required. Every `app_graph.invoke()` call automatically emits a trace to your LangSmith project, capturing:

- Node-level execution timeline (router → research → orchestrator → worker × N → reducer)
- LLM call inputs/outputs and token counts for each ChatMistralAI invocation
- Tavily tool call inputs and raw results
- End-to-end latency and per-node duration
- Any exceptions with full stack context

Traces are viewable at `https://smith.langchain.com` under the configured project name.

## Requirements

- Python 3.10+
- Node.js 18+ (for Next.js frontend)
- Supabase project (PostgreSQL connection string)
- Firebase project (service account JSON for backend, web config for frontend)
- API keys:
  - `MISTRAL_API_KEY` — required for ChatMistralAI
  - `TAVILY_API_KEY` — required for web research mode
  - `LANGSMITH_API_KEY` — required for LangSmith tracing

## Setup

### 1. Backend — Python virtual environment

```powershell
python -m venv venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

### 2. Frontend — Next.js

```bash
cd frontend
npm install
```

### 3. Environment variables

Create `.env` in the project root:

```env
# LLM + Research
MISTRAL_API_KEY=your_mistral_key
TAVILY_API_KEY=your_tavily_key

# LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=blogagent          # shows up as the project name in LangSmith UI

# Supabase
DATABASE_URL=postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres

# Firebase (backend service account)
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-service-account.json
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_FIREBASE_API_KEY=...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=...
NEXT_PUBLIC_FIREBASE_PROJECT_ID=...
NEXT_PUBLIC_FIREBASE_APP_ID=...
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Database migrations

```powershell
alembic upgrade head
```

Or run `Base.metadata.create_all(engine)` directly for dev.

## Running the App

Start the FastAPI backend:

```powershell
uvicorn app.main:app --reload
```

Start the Next.js frontend:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Frontend Overview (Next.js)

- `/` — redirect to login or dashboard
- `/login` — Firebase Auth sign-in (email + Google)
- `/dashboard` — chat sidebar + blog generation UI
  - Sidebar: create new chat, switch between sessions
  - Main: topic input, as-of date, Generate button
  - Tabs: Plan · Evidence · Markdown Preview · Images · Logs

Authentication state is managed via Firebase `onAuthStateChanged`; the ID token is refreshed automatically and attached to all fetch calls.

## Programmatic Backend Usage

```python
from run import run

result = run("Your topic here")
print(result["final"])  # final markdown
```

The `run()` helper calls the compiled LangGraph app:

```python
app_graph.invoke({
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
    "generated_images": [],
    "final": "",
})
```

## Troubleshooting

**Traces not appearing in LangSmith**
- Confirm `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set before the process starts.
- `LANGCHAIN_PROJECT` is optional but strongly recommended — without it, traces go to the default project and are easy to lose.
- Tracing is fire-and-forget; a LangSmith outage will not block blog generation.

**Firebase token rejected (401)**
- Confirm `FIREBASE_SERVICE_ACCOUNT_PATH` points to a valid service account JSON.
- Ensure the Firebase project ID in the service account matches your frontend config.

**Supabase connection refused**
- Use the **direct** connection string (port 5432), not the pooler, for SQLAlchemy.
- If on Supabase free tier, the DB may be paused — resume it in the dashboard.

**Evidence appears empty or malformed**
- Ensure `TAVILY_API_KEY` is set and valid.
- The backend includes a deterministic JSON fallback when LLM research parsing fails.

**Images not displaying**
- Pollinations AI is a public, unauthenticated service — check network access.
- Image generation errors are handled gracefully; a failure note is inserted in the markdown where the placeholder was.

**`blog_images.image_data` is large**
- For production, consider storing images in Supabase Storage (object store) and saving only the public URL in the table instead of raw binary.

## Notes

- `generated_blogs/` is retained for local dev / CLI usage and is git-ignored.
- The `content` column in `blog_sessions` is JSONB — it stores the full pipeline output (plan, evidence, sections, final markdown) as a single document for easy retrieval.
- All Supabase queries are user-scoped via the resolved `users.id` from the Firebase UID — no row is readable by a different user.
- LangSmith tracing is purely additive — disabling it (by unsetting `LANGCHAIN_TRACING_V2`) has zero effect on pipeline behavior or output.
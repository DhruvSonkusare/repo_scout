# RepoGuide — project context for Claude Code

## What we are building
RepoGuide is an AI developer assistant with two features:

1. **Repo Finder** — the user describes a problem in plain English
   ("I need to extract tables from scanned PDFs in Python"). RepoGuide searches GitHub,
   ranks the best open-source repos for that problem, explains why each fits, and
   generates a "how to solve your problem with this repo" guide.
2. **Codebase Q&A** — the user indexes any public GitHub repo, then asks questions
   ("where is authentication handled?") and gets streamed answers with file + line
   citations. It can also generate a full onboarding guide for the repo.

This is a **portfolio project** for job applications. Code quality, clear structure,
tests and a good README matter as much as features.

## Tech stack (do not change without asking)
- Python 3.11+, **FastAPI**, Uvicorn
- **PostgreSQL 16** running in Docker (`docker-compose.yml`, service `db`), connected with
  **SQLAlchemy 2.0** (typed `Mapped[...]` style) and the **psycopg 3** driver
  (`postgresql+psycopg://...`). JSON fields use `JSONB`, timestamps use `DateTime(timezone=True)`.
- **Alembic** for schema migrations. Never use `create_all()` in the app; every model
  change gets a new migration (`alembic revision --autogenerate -m "..."`).
- **LangChain** (`langchain-core`, `langchain-text-splitters`, `langchain-google-genai`,
  `langchain-chroma`) with **Google Gemini** (chat + embeddings)
- **Chroma** persistent vector store (`data/chroma/`), one collection per repo
- **httpx** for the GitHub REST API
- Frontend: plain **HTML + CSS + vanilla JS** in `static/`, served by FastAPI
  (no build step). Use `marked` + `highlight.js` from a CDN for Markdown/code.
- Tests: **pytest** + FastAPI `TestClient`, with fake LLM/embeddings (no API key needed).
  Tests run against the real `repoguide_test` Postgres database, never SQLite, because we use JSONB.

## Folder layout
The project root is this folder (`d:\github_repo_guide`), not a nested `repoguide/`.
```
./                   # project root
  app/
    main.py          # FastAPI app, routers included here, serves frontend
    config.py        # pydantic-settings Settings (reads .env)
    db.py            # engine, SessionLocal, Base, ORM models, get_db dependency
    schemas.py       # Pydantic request/response models
    llm.py           # get_chat_model() / get_embeddings()  (gemini | fake)
    vectorstore.py   # Chroma helpers
    ingest.py        # clone + split + index a repo
    rag.py           # Q&A chain + onboarding guide chain
    finder/
      github.py      # GitHub search + README fetch (httpx)
      chains.py      # query planner + reranker + problem guide (LangChain)
    routers/
      repos.py  chat.py  guide.py  finder.py
  static/            # index.html, style.css, app.js
  tests/
  alembic/           # migrations
  alembic.ini
  docker-compose.yml # postgres (+ app later)
  docker/            # init-test-db.sql (creates repoguide_test)
  scripts/           # manual helpers, e.g. smoke_llm.py
  data/              # runtime data (cloned repos, Chroma index), git-ignored
```

## Conventions
- All LLM/embedding objects come from `app/llm.py`. Never instantiate Gemini anywhere else.
  `LLM_PROVIDER=fake` must make the whole app work offline (used by tests).
- All prompts are `ChatPromptTemplate` constants at the top of the module.
- Chains use LCEL (`prompt | model | parser`). Use `model.with_structured_output(PydanticModel)`
  whenever we need JSON from the LLM — never parse JSON by hand.
- Long jobs (cloning/indexing) run in FastAPI `BackgroundTasks` and update a `status` column.
- Secrets only in `.env` (never committed). Provide `.env.example`.
- Type hints everywhere, short docstrings, no unused code.
- After each step: run `pytest -q`, then summarise what changed.

## How I (Dhruv) want to work
- I'm learning LangChain. When you write a LangChain piece, add a 2–3 line comment
  explaining the concept (e.g. what MMR retrieval or `with_structured_output` does).
- Do one PLAN.md step at a time. Don't jump ahead to later steps.
- At the end of each step, tell me how to test it manually (exact command or URL).

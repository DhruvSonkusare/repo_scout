# RepoGuide — step-by-step build plan (for Claude Code)

**How to use this file**
1. The project root is this folder (`d:\github_repo_guide`), which holds `CLAUDE.md` and
   `PLAN.md`. Open Claude Code here; all paths in this plan are relative to it.
2. For each step, paste the **Prompt** into Claude Code. Use plan mode (Shift+Tab) for
   the larger steps (marked ⭐) so you can review the approach before it writes code.
3. Check the **Done when** list yourself, then commit:
   `git add -A && git commit -m "Step N: <title>"`.
4. Read the **Learn** note, then ask Claude Code to explain any part you don't get.

Estimated total: 4–6 focused days.

---

## Reference: data model

Database: **PostgreSQL** (JSON columns are `JSONB`, all timestamps are timezone-aware).

| Table | Columns |
|---|---|
| `repos` | id, url (unique), name (`owner/repo`), status (`pending/cloning/indexing/ready/failed`), error, file_count, chunk_count, chunks_embedded (progress), embedding_provider, embedding_model, languages (JSONB), onboarding_guide (text), created_at |
| `conversations` | id, repo_id → repos, title, created_at |
| `messages` | id, conversation_id → conversations, role (`user/ai`), content, sources (JSONB list), created_at |
| `searches` | id, problem (text), plan (JSONB: queries/language/must_have), created_at |
| `search_results` | id, search_id → searches, full_name, url, description, stars, language, last_pushed, score (0–100, NULL if not assessed), why_it_fits, caveats, how_to_start (JSONB list of strings), repo_id → repos (nullable, set when indexed), problem_guide (text, nullable) |

## Reference: API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/repos` | add GitHub URL, starts background indexing |
| GET | `/api/repos`, `/api/repos/{id}` | list / status |
| POST | `/api/repos/{id}/reindex` | re-clone and re-index |
| DELETE | `/api/repos/{id}` | delete repo, clone and vectors |
| GET | `/api/repos/{id}/file?path=` | read a file (for citation viewer) |
| POST | `/api/repos/{id}/ask` | **streamed** answer (NDJSON) |
| GET | `/api/repos/{id}/conversations` | chat history list |
| GET | `/api/conversations/{id}/messages` | messages of a chat |
| GET/POST | `/api/repos/{id}/guide` | get / generate onboarding guide |
| POST | `/api/finder/search` | problem statement → ranked repos |
| GET | `/api/finder/searches`, `/api/finder/searches/{id}` | past searches + results |
| POST | `/api/finder/results/{id}/index` | index a found repo (reuses `/api/repos`) |
| POST | `/api/finder/results/{id}/guide` | "how to solve my problem with this repo" guide |
| GET | `/api/health` | health + which LLM provider |

---

## Phase 1 — Foundation

### Step 1: Project setup
**Prompt**
> Read CLAUDE.md. Set up the project skeleton: create the folder layout from CLAUDE.md with
> empty modules, `requirements.txt` (fastapi, uvicorn[standard], sqlalchemy>=2, psycopg[binary]>=3,
> alembic, pydantic-settings, httpx, langchain-core, langchain-text-splitters,
> langchain-google-genai, langchain-chroma, fastembed, chromadb, python-dotenv, pytest, pytest-cov), `.env.example`
> (DATABASE_URL=postgresql+psycopg://repoguide:repoguide@localhost:5432/repoguide,
> TEST_DATABASE_URL=postgresql+psycopg://repoguide:repoguide@localhost:5432/repoguide_test,
> GOOGLE_API_KEY, GITHUB_TOKEN, LLM_PROVIDER=gemini, CHAT_MODELS, GUIDE_MODELS, EMBEDDING_PROVIDER=local, EMBEDDING_MODEL, DATA_DIR=./data),
> `.gitignore`, and a `docker-compose.yml` with a `db` service: image postgres:16, user/password/db
> `repoguide`, port 5432, a named volume `pgdata`, a healthcheck using `pg_isready`, and an init
> script `docker/init-test-db.sql` that also creates the `repoguide_test` database. Add a
> minimal FastAPI app in `app/main.py` with `GET /api/health` that also runs `SELECT 1` against
> the database and reports `"db": "ok"` or the error. The health check needs a minimal
> `app/config.py` (just `DATABASE_URL`) and `app/db.py` (just the engine) now; Step 2 expands both.
> Add a health test and a minimal `tests/conftest.py`.
> Don't create `alembic/` yet: `alembic init` in Step 2 creates it. Give me the commands to create a venv and
> run everything on Windows and on Linux.

**You do:**
1. Install **Docker Desktop** (on Windows it will ask you to enable WSL2; let it).
   If Docker gives you trouble, use the official PostgreSQL Windows installer instead and create
   the `repoguide` user and the two databases in pgAdmin.
2. `docker compose up -d db`, then `docker compose ps` should show the db as **healthy**.
3. Create the venv, `pip install -r requirements.txt`, copy `.env.example` → `.env`,
   paste your Gemini key, `git init`.

**Done when:** `uvicorn app.main:app --reload --reload-dir app` runs and http://localhost:8000/api/health returns
`{"status": "ok", "db": "ok"}`; http://localhost:8000/docs opens.

**If it fails:**
- *Connection refused*: the db container isn't running (`docker compose up -d db`).
- *Password authentication failed*: `DATABASE_URL` doesn't match docker-compose.
- *Port 5432 already in use*: another Postgres is running; stop it or map `5433:5432`
  and update the URL.
- *Database `repoguide_test` does not exist*: init scripts only run when the `pgdata` volume is
  first created. Create it by hand:
  `docker compose exec db psql -U repoguide -d postgres -c "CREATE DATABASE repoguide_test"`.

### Step 2: Config + database + migrations ⭐
**Prompt**
> Expand Step 1's minimal `app/config.py` (pydantic-settings, cached `get_settings()`, `database_url` from env,
> creates data dirs, settings for chunk_size=1500, chunk_overlap=150, retrieval_k=6,
> max_files=1500, max_file_bytes=200000) and `app/db.py`: engine with `pool_pre_ping=True`,
> `SessionLocal`, `Base`, and SQLAlchemy 2.0 typed models for ALL tables in the PLAN.md data
> model (including searches and search_results). Use `JSONB` from
> `sqlalchemy.dialects.postgresql` for JSON fields, `DateTime(timezone=True)` with
> `server_default=func.now()`, foreign keys with `ondelete="CASCADE"` (SET NULL for
> search_results.repo_id), and indexes on the foreign keys and `repos.status`. Add a `get_db`
> dependency. Then set up **Alembic**: `alembic init alembic`, make `env.py` read the URL from
> our settings and use `Base.metadata`, and generate the first migration with autogenerate.
> Expand `tests/conftest.py` so it points the app at `TEST_DATABASE_URL`, runs `alembic upgrade head`
> once per session and truncates all tables between tests. Later steps rely on this setup.
> Add a pytest that inserts one row into each table against `TEST_DATABASE_URL`.

**You do:** `alembic upgrade head`, then look at the tables:
`docker compose exec db psql -U repoguide -d repoguide -c "\dt"`
(or open them in pgAdmin / the VS Code PostgreSQL extension).

**Done when:** the 5 tables (+ `alembic_version`) exist and `pytest -q` passes.
**Learn:** SQLAlchemy 2.0 `Mapped[]`, relationships with cascade, FastAPI dependencies, what a
migration is and why production apps never use `create_all()`, and JSONB vs JSON.

### Step 3: LLM factory
**Prompt**
> Implement `app/llm.py` with `get_chat_model()`, `get_guide_model()`, `get_structured_model()` and
> `get_embeddings()`. Provider "gemini" uses `ChatGoogleGenerativeAI` (temperature 0.2) in `.with_fallbacks()`
> chains from `CHAT_MODELS` and `GUIDE_MODELS`. `EMBEDDING_PROVIDER` chooses embeddings separately: "local"
> (the default) uses a small `Embeddings` class around FastEmbed's `TextEmbedding` with `jinaai/jina-embeddings-v2-base-code` unless `EMBEDDING_MODEL`
> names another model, on the device set by `EMBEDDING_DEVICE` (auto, cuda or cpu; `requirements-gpu.txt`
> installs the CUDA build), and "gemini" uses `GoogleGenerativeAIEmbeddings`. Provider
> "fake" uses `FakeListChatModel` and `DeterministicFakeEmbedding(size=256)`. Add a tiny script
> `scripts/smoke_llm.py` that sends "Say hello in 5 words" and embeds one sentence, printing the
> vector length. Set `LLM_PROVIDER=fake` and `EMBEDDING_PROVIDER=fake` in `tests/conftest.py` so all tests
> from here on stay offline.

**Done when:** `python scripts/smoke_llm.py` prints a reply from Gemini and a vector size.
**Learn:** Why one factory for models: swapping providers only changes this one file.
Also check the current model names in Google AI Studio and put them in `.env`.

---

## Phase 2 — Codebase Q&A (core)

### Step 4: Repo ingestion ⭐
**Prompt**
> Implement `app/ingest.py`:
> - `parse_github_url()` (only `https://github.com/owner/repo`, raise ValueError otherwise)
> - `clone_repo()` using `git clone --depth 1` into `data/repos/<repo_id>` (subprocess, timeout)
> - `iter_source_files()` skipping node_modules, .git, venv, dist, build etc., lockfiles,
>   minified files, symlinks (a repo can link to files outside itself) and files > max_file_bytes;
>   map extensions to a language name and a
>   LangChain `Language` enum
> - `split_file()` using `RecursiveCharacterTextSplitter.from_language(...)` with
>   `add_start_index=True`; convert start_index into `start_line`/`end_line` metadata,
>   store `path` and `language`, and prefix chunk text with `# File: <path>`
> - `build_file_tree()` (depth ≤ 3, max 300 lines)
> Write unit tests for URL parsing, file filtering and line-number calculation using a tmp dir.

**Done when:** tests pass; line numbers in a test file match the real lines.
**Learn:** Why language-aware splitting (it splits on `def`/`class` boundaries, not mid-function)
and why we keep line numbers (citations).

### Step 5: Vector store + background indexing ⭐
**Prompt**
> Implement `app/vectorstore.py` (one `chromadb.PersistentClient`, `get_vectorstore(repo_id)`
> returning a LangChain `Chroma` with collection `repo_<id>` and cosine space,
> `delete_collection(repo_id)`). Then add `index_repo(repo_id)` in ingest.py: set status
> cloning → indexing → ready/failed, embed in batches, with pauses and 429 waits (using Gemini's retry
> delay if it gives one) for Gemini embeddings only, save chunks_embedded after each batch, store file_count, chunk_count and
> languages JSON, and save a readable error on failure. Then build `app/routers/repos.py` with
> POST/GET/DELETE `/api/repos`, `/api/repos/{id}`, `/api/repos/{id}/reindex` and
> `/api/repos/{id}/file` (resolve the full path, following symlinks, and return 404 unless it stays
> inside the clone, so neither `../` nor a symlink can escape). Indexing must run in
> BackgroundTasks. Adding an already-added URL returns the existing repo. Delete clones with a helper
> that clears read-only flags (Git marks some `.git` files read-only, which makes a plain
> `shutil.rmtree` fail on Windows). At startup, mark repos stuck in `pending`, `cloning` or `indexing`
> as `failed` with a "server restarted" error, so they can be re-indexed.

**Run the server with** `uvicorn app.main:app --reload --reload-dir app`. Plain `--reload` also watches
`data/`, so cloning a repo restarts the server and kills the indexing job.

**Done when:** in /docs, POST `https://github.com/pallets/click`, poll GET until status is `ready`,
and see chunk_count > 0. `/file?path=../../../.env` (which points at the project's real `.env`)
returns 404.
**Learn:** Embeddings, vector stores, why indexing is a background job.

### Step 6: RAG Q&A chain with streaming ⭐
**Prompt**
> Implement `app/rag.py` and `app/routers/chat.py`:
> - CONDENSE_PROMPT: rewrite a follow-up into a standalone question using chat history
>   (skip the LLM call when there is no history)
> - retriever: `as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.85})`
>   (the default 0.5 favours variety so much that it dropped the most relevant files)
> - ANSWER_PROMPT: senior engineer persona, answer ONLY from numbered excerpts, cite like [1],
>   say honestly when the answer is not in the excerpts
> - `format_docs()` numbering excerpts as `[n] path (lines a-b)` + fenced code
> - POST `/api/repos/{id}/ask` with body `{question, conversation_id?}`: creates the
>   conversation if needed, saves the user message, then returns a `StreamingResponse`
>   of NDJSON lines: `{"type":"meta","conversation_id","sources"}`, many
>   `{"type":"token","text"}`, then `{"type":"done","message_id"}` or `{"type":"error"}`.
>   Save the AI message + sources JSON when streaming finishes (open a new DB session).
>   Use the last 6 turns as history.
> - GET conversations list and messages endpoints.
> Return 409 if the repo isn't ready, or "Re-index needed" if it was indexed with a different
> embedding provider or model than the current settings.

**Done when:** this streams tokens, and a follow-up "where is that tested?" with the same
conversation_id works.
- Windows PowerShell (`curl` alone is an alias for `Invoke-WebRequest`, and PowerShell 5.1 splits `-d` JSON
  at its spaces, so pipe the body in):
  `'{"question":"How are command options parsed?"}' | curl.exe -N -X POST http://localhost:8000/api/repos/1/ask -H "Content-Type: application/json" --data-binary "@-"`
- Linux/macOS:
  `curl -N -X POST http://localhost:8000/api/repos/1/ask -H "Content-Type: application/json" -d '{"question":"How are command options parsed?"}'`
**Learn:** LCEL pipes, `MessagesPlaceholder`, history condensing, MMR, `astream`.

### Step 7: Onboarding guide
**Prompt**
> Add `generate_onboarding_guide()` in rag.py: run 5 probe queries (entry point, configuration,
> core models/logic, how to run, tests) with k=3, de-duplicate chunks, and send the README
> (first 6000 chars), file tree, language stats and excerpts to an ONBOARDING_PROMPT that writes
> Markdown sections: What it does, Tech stack, Architecture, Directory guide, Where to start
> reading (ordered files), How to run locally, Good first tasks. It must never invent files
> that are not in the tree. Add GET/POST `/api/repos/{id}/guide` in `routers/guide.py`; save the
> guide on the repo row.

**Done when:** POST returns a sensible guide for `pallets/click`, and GET returns the saved version.

---

## Phase 3 — Add-on: Repo Finder (problem → repos → how-to)

Flow:
```
problem statement
  → [Query planner LLM, structured output]  → 2–3 GitHub search queries + language + keywords
  → [GitHub Search API]                      → ~20 candidate repos (deduped, archived removed)
  → [fetch READMEs of top 15–20, trimmed to ~1500 chars]
  → [Reranker LLM, structured output]        → score 0–100, why_it_fits, caveats, how_to_start
  → save search + results in DB → show ranked cards
  → user clicks "Index & explain" → indexes repo (Phase 2) → problem-specific guide via RAG
```

### Step 8: GitHub client
**Prompt**
> Implement `app/finder/github.py` with an async `GitHubClient` (httpx.AsyncClient,
> `Authorization: Bearer GITHUB_TOKEN` if set, `Accept: application/vnd.github+json`,
> `X-GitHub-Api-Version: 2022-11-28`):
> - `search_repos(query, language=None, per_page=10)` → GET `/search/repositories` with
>   `q="<query> language:<lang> archived:false"`, sort by stars; return a list of dataclasses
>   (full_name, html_url, description, stars, language, topics, pushed_at, license)
> - `get_readme(full_name)` → GET `/repos/{full_name}/readme` with
>   `Accept: application/vnd.github.raw`, truncate to 4000 chars, return "" on 404
> - clear error messages for 403/429 rate limits
> Write tests that mock httpx using `httpx.MockTransport`.

**You do:** create a GitHub fine-grained token with no extra permissions (public read only)
and put it in `.env` as `GITHUB_TOKEN`. Without it, search is limited to about 10 requests a minute,
and the 15–20 README fetches per search count against a limit of 60 API requests an hour.

**Done when:** tests pass; a quick script prints top repos for "pdf table extraction".

### Step 9: Query planner + reranker chains ⭐
**Prompt**
> Implement `app/finder/chains.py`:
> - Pydantic `SearchPlan` (subject: the problem's core noun; queries: list[str] 2–3 short GitHub-style
>   keyword queries that all contain the subject, which code enforces; language: str | None,
>   must_have: list[str]) and a PLANNER_PROMPT that turns a plain-English problem into it. Use
>   `get_structured_model(SearchPlan)`.
> - `find_repos(problem)`: run the plan, call `search_repos` for each query concurrently
>   (`asyncio.gather`), dedupe by full_name, drop repos not pushed in the last 3 years. With fewer than
>   8 candidates, run one broader search for the subject alone. Take the top 15–20 deduplicated
>   candidates, fetch their READMEs concurrently and trim each to ~1500 chars for the reranker.
> - Pydantic `RepoAssessment` (full_name, score 0–100, why_it_fits, caveats, how_to_start: 2–4
>   short steps) and `Ranking(results: list[RepoAssessment])`; a RERANK_PROMPT that judges
>   each candidate (description + topics + README excerpt) AGAINST THE USER'S PROBLEM, is
>   honest when nothing fits well, and never invents features not in the README. After the LLM call,
>   drop any assessment whose full_name isn't in the candidate list. No silent drops: candidates are split
>   into 2 batches scored concurrently with `asyncio.gather`, and any still missing are saved with
>   score NULL and marked "not assessed", after the scored results.
> - Make the fake provider work: when LLM_PROVIDER=fake, return a fixed SearchPlan/Ranking.
> Then add `app/routers/finder.py`: POST `/api/finder/search {problem}` saves `searches` +
> `search_results` and returns results sorted by score; GET searches list and detail.

**Done when:** searching *"I want to build a WhatsApp chatbot for my shop in Python"* returns
relevant, ranked repos with reasons in about 15 seconds.
**Learn:** `with_structured_output` (how the LLM returns typed Pydantic objects), running
chains concurrently, and why we rerank instead of trusting GitHub stars.

### Step 10: "How to solve my problem with this repo" guide
**Prompt**
> Add POST `/api/finder/results/{id}/index`: create/reuse the Repo for that URL (same logic as
> POST /api/repos), link `search_results.repo_id`, start indexing. Add POST
> `/api/finder/results/{id}/guide`: requires the repo to be ready; retrieve chunks using the
> user's original problem statement plus probe queries for install/config/usage examples;
> PROBLEM_GUIDE_PROMPT writes Markdown in this EXACT structure so the user never loses context:
> 1. **Your problem**: the user's original problem statement, quoted word for word (written by our
>    code, not the model, so it can't be reworded)
> 2. **Recommended repo**: name, link and a one-line summary of what it does
> 3. **Does it solve your problem?**: an honest verdict (Fully / Partly / No) with the reason
> 4. **Setup**: numbered install and config commands taken from the repo
> 5. **Step-by-step: solving YOUR problem with this repo**: numbered steps, each with what
>    to do, why, and a code snippet adapted from real files (cite file paths)
> 6. **What you still need to build yourself**
> 7. **Gotchas and next steps**
> Save to `search_results.problem_guide`. The API response must also return `problem`
> (the original text) and the repo name next to the guide.

**Done when:** from a search result you can index it and get a guide that starts with your
original problem, gives numbered how-to steps for that exact problem and cites real file paths.

**Example of what the user should see**
> **Your problem:** "I want to build a WhatsApp chatbot for my shop that answers price questions."
> **Recommended repo:** some-org/whatsapp-bot-python (a Python wrapper for the WhatsApp Cloud API)
> **Verdict:** Partly. It handles sending and receiving messages, but you need to add your own price lookup.
> **Step 1:** Create a Meta developer app and get a token (see `README.md`)…
> **Step 2:** Copy `examples/echo_bot.py` and replace the echo handler with a price lookup…

---

## Phase 4 — Frontend

### Step 11: UI ⭐
**Prompt**
> Build the frontend in `static/` (index.html, style.css, app.js; vanilla JS, `marked` and
> `highlight.js` from CDN; clean dark developer-tool look; responsive). Layout:
> - Top nav with two tabs: **Find a repo** and **My repos**.
> - **Find a repo**: a big textarea "Describe your problem in plain English", example chips,
>   a loading state with steps ("Planning search… Searching GitHub… Ranking…"), result cards
>   (name, stars, language, last updated, score badge, why it fits, caveats, how to start) with
>   buttons "Open on GitHub" and "Index & explain". Keep the user's problem statement pinned
>   at the top of the results page ("Results for: …") at all times. After indexing (poll status
>   with a progress message), open a full-width **Solution view** with two parts: a sticky header
>   showing the problem statement and the chosen repo, and below it the step-by-step guide
>   with numbered steps, copy buttons on code blocks and a "Back to results" button. Add an
>   "Ask a follow-up about this repo" button that opens the chat for that repo. A sidebar
>   lists past searches; clicking one restores its problem, results and any saved guides.
> - **My repos**: sidebar with indexed repos + status badges (poll every 3s while indexing) and an
>   "Add GitHub URL" input. Main area has tabs **Chat** and **Onboarding guide**. Chat streams
>   NDJSON tokens from `/ask` (use `fetch` + `ReadableStream`), renders Markdown, and shows source
>   chips `[n] path:L10-42`; clicking a chip opens a side drawer with the file content and the
>   cited lines highlighted and scrolled into view. Support conversation history (list + new chat).
> Serve index.html at `/` and mount `/static`.

**Done when:** you can do the whole flow in the browser without touching /docs.

---

## Phase 5 — Quality and showcase

### Step 12: Tests
**Prompt**
> Finish `tests/conftest.py` (started in Steps 1–3) so it sets `LLM_PROVIDER=fake`, `EMBEDDING_PROVIDER=fake`, a tmp `DATA_DIR` and
> `DATABASE_URL=$TEST_DATABASE_URL` before importing the app; runs `alembic upgrade head` on the
> test database once per session and truncates all tables between tests (so tests never touch
> your real data). It also monkeypatches `clone_repo` to copy a small fixture repo from `tests/fixtures/`. Write
> API tests: add repo → ready, ask (parse the NDJSON stream), follow-up uses the same
> conversation, guide generation, file path traversal blocked, finder search with mocked
> GitHub, index result + problem guide. Target ≥ 80% coverage (`pytest --cov=app`).

### Step 13: Docker + README
**Prompt**
> Add a `Dockerfile` (python:3.12-slim, install git, install `requirements.txt` so embeddings run on the
> CPU, non-root user, uvicorn on 8000) and add an
> `app` service to `docker-compose.yml` that depends on `db` being healthy, uses
> `DATABASE_URL=postgresql+psycopg://repoguide:repoguide@db:5432/repoguide` (host is `db`
> inside Docker, not localhost), runs `alembic upgrade head` before starting uvicorn, and has a
> volume for `data/`. `docker compose up --build` must start the whole app from scratch. Write a strong README.md: one-line pitch, the
> problem it solves, screenshots placeholders, features, an architecture diagram in Mermaid
> (both pipelines), tech stack, how to run (local + Docker), API table, design decisions
> (why MMR, why language-aware chunking, why structured output reranking, why NDJSON
> streaming), limitations and future work.

### Step 14 (optional, high value): Deploy + extras
Pick 1–2:
- Deploy on **Render/Railway**, or on **AWS** (EC2 + Docker, or App Runner). AWS fits your
  resume best: you can mention S3 for clone storage and CloudWatch logs.
- Add **GitHub Actions** CI that runs pytest on every push.
- Use a hosted Postgres (Neon, Supabase or AWS RDS) for the deployed version: only
  `DATABASE_URL` changes.
- Replace Chroma with **pgvector** (`langchain-postgres`) so vectors live in the same Postgres
  database: one database to run and back up.
- Add a GitHub Actions CI job with a `postgres:16` service container.
- Add **token/cost tracking** per request (LangChain callbacks), shown in the UI.
- Record a 60-second demo GIF for the README and LinkedIn.

---

## Resume bullets (fill in your real numbers at the end)
- Built **RepoGuide**, an AI developer assistant that finds open-source repos for a problem
  described in plain English and answers questions about any GitHub codebase with
  file- and line-level citations (FastAPI, LangChain, Gemini, FastEmbed, PostgreSQL, SQLAlchemy, Alembic, Chroma, Docker).
- Designed a RAG pipeline with language-aware code chunking, MMR retrieval and
  history-aware query rewriting, streaming answers over NDJSON.
- Implemented an LLM query planner and reranker using structured outputs over the
  GitHub Search API, returning ranked repos with a fit score and step-by-step adoption guides.
- Async background indexing of N-file repos in ~X s; Y% test coverage with offline fake-LLM tests; Dockerized.

## Interview questions to be ready for
- Why PostgreSQL instead of SQLite? (many concurrent writers: background indexing + chats)
- What does Alembic do, and what happens if two developers create migrations at the same time?
- JSONB vs JSON in Postgres? Why add indexes on foreign keys?
- Why chunk by language syntax instead of fixed size? What does chunk overlap do?
- What is MMR and when is plain similarity search better?
- How do you stop the model from hallucinating file names? (grounding, citations, "say if not found")
- Why rewrite follow-up questions before retrieval?
- How would you scale indexing to 1,000 repos? (queue such as SQS/Celery, workers, incremental re-index by commit SHA)
- How does `with_structured_output` work under the hood? (tool/function calling + schema)

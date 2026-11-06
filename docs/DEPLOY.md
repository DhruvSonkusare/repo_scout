# Deploying RepoScout on Render

This runs the public demo on Render's free web service, with data and vectors in Neon Postgres and embeddings from
Gemini. Commands are for PowerShell, from the project root. Dashboard labels may have changed since this was written.

## How it fits together

| Piece | Where | Notes |
|---|---|---|
| App | Render, free web service, from the `Dockerfile` | About 512 MB of memory. It spins down when idle, and the first request after that takes about a minute. |
| Data and vectors | Neon Postgres with pgvector | Survives restarts, because it's outside Render. |
| Chat and embeddings | Gemini, `gemini-embedding-001` | No model runs on the server, which keeps memory low. |
| Clones | The instance's disk | Wiped on every restart. They come back on demand, at the commit that was indexed. |

## What you need

- The project on GitHub, including the `Dockerfile` and `render.yaml` at the root.
- A Neon database. Put its direct connection string in `.env` as `NEON_DATABASE_URL`, with `sslmode=require`.
- A `GOOGLE_API_KEY`, and a `GITHUB_TOKEN` that needs no scopes for public repos.

## 1. Update your local setup

Vectors moved from Chroma to pgvector, so install the new packages and restart the database on its new image.
Stop your dev server first. Your data stays in the same Docker volume.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-base.txt
docker compose up -d db
.venv\Scripts\alembic.exe upgrade head
```

If you use the GPU build, run `powershell -ExecutionPolicy Bypass -File scripts\install_gpu.ps1` only if CUDA stops
working. Repos you indexed before this now say "Re-index needed", because their vectors are in Chroma. Use
**Index again** on each one. `data\chroma` isn't used any more, and you can delete it when you're happy.

## 2. Index the demo repos into Neon

Run this from your own machine, before the first deploy. See the plan first, which makes no Gemini calls and writes
nothing:

```powershell
.venv\Scripts\python.exe scripts\index_demo_repos.py --plan
```

Then index. This creates the tables in Neon and uses your Gemini embedding quota, one per chunk:

```powershell
.venv\Scripts\python.exe scripts\index_demo_repos.py
```

Your local `DATABASE_URL` and clones aren't touched. If it stops because the daily quota ran out, run it again the
next day, and it skips what's finished.

## 3. Push to GitHub

```powershell
git push origin master
```

## 4. Create the Render service

1. In the Render dashboard, choose **New**, then **Blueprint**, and connect your GitHub repo.
2. Render reads `render.yaml` and asks for the four secrets below.
3. Apply it, and watch the logs until they say `Application startup complete`.

If you'd rather not use a Blueprint, choose **New**, then **Web Service**, pick the repo, select **Docker** as the
language and **Free** as the instance type, set the health check path to `/api/health`, and add the same four values.

### Secrets to add in Render

| Name | Value |
|---|---|
| `DATABASE_URL` | The same Neon connection string as `NEON_DATABASE_URL` |
| `GOOGLE_API_KEY` | Your Gemini API key |
| `GITHUB_TOKEN` | Your GitHub token |
| `INDEX_PASSWORD` | A long random password you make up. It's what lets you add or change repos on the demo. |

Generate a password with:

```powershell
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
```

Everything else is set in the `Dockerfile`, and an environment variable in Render overrides any of it. Render provides
`PORT` itself, so don't set that.

## 5. Check it works

Replace the address with your service's URL.

```powershell
$url = "https://your-service.onrender.com"
Invoke-RestMethod "$url/api/health"
Invoke-RestMethod "$url/api/config"
Invoke-RestMethod "$url/api/repos" | Select-Object name, status, embedding_model
```

Health should say `status: ok`, config `demo_mode: true`, and each repo `ready` with `gemini-embedding-001`. Then open
the URL, go to **My repos**, ask a question, and click a source chip. The first click restores that repo's clone.

## 6. Check the rate limit can't be dodged

Rate limits need the visitor's real IP, and Render's documentation doesn't settle which header is safe. The image uses
`CLIENT_IP_HEADER=true-client-ip`. This test sends a different fake IP with each wrong password, which uses no Gemini:

```powershell
1..12 | ForEach-Object {
  $headers = @{ "X-Index-Password" = "wrong"; "True-Client-IP" = "203.0.113.$_" }
  try { Invoke-RestMethod -Method Post "$url/api/repos/1/reindex" -Headers $headers }
  catch { $_.Exception.Response.StatusCode.value__ }
}
```

- **You see 401s and then 429s:** the header can't be faked, so the limit works.
- **You see only 401s:** a client can choose its own IP. Try `cf-connecting-ip` as `CLIENT_IP_HEADER`, and test again.
- **If neither works:** set `CLIENT_IP_HEADER` to empty. All visitors then share one limit, which is strict but safe.

This test uses up your own `index` limit for an hour. Either way, the daily cap still protects your quota.

## Demo limits

All of these are environment variables, shown with their defaults.

| Variable | Default | What it does |
|---|---|---|
| `INDEX_PASSWORD` | none | Needed to add, re-index or delete repos. If it's empty, those are turned off. |
| `MAX_REPO_SIZE_MB` | 20 | The largest repo that can be added. |
| `MAX_INDEX_CHUNKS` | 400 | The most chunks a repo added on the demo can have. |
| `RATE_LIMIT_SEARCH_PER_HOUR` | 10 | Searches per IP. |
| `RATE_LIMIT_ASK_PER_HOUR` | 30 | Questions per IP. |
| `RATE_LIMIT_GUIDE_PER_HOUR` | 5 | Guides per IP. |
| `RATE_LIMIT_INDEX_PER_HOUR` | 10 | Attempts to add or change repos per IP. |
| `DAILY_REQUEST_CAP` | 100 | Searches, questions and guides per day for everyone, reset at midnight UTC. |

### What uses the Gemini embedding quota

The free tier allows 1,000 embedded texts a day, as far as this project has measured.

| Action | Embeddings |
|---|---|
| Indexing | 1 per chunk |
| A question | 1 |
| A guide for a search result | 6 |
| An onboarding guide | 5 |
| A search | 0 |

At the default cap, the worst case is 100 guides, or 600 embeddings a day, on top of any indexing. To index
something bigger, do it from your machine with `scripts\index_demo_repos.py --repo owner/name`. The script refuses
more than 700 chunks at once. `psf/requests`, for example, is 603.

Chat and guide calls count against separate Gemini quotas for each model. When those run out, the app falls back to
the next model, and then shows a message.

## Good to know

- **Everything is public.** Visitors can see each other's searches and chats, and the page says so.
- **Neon storage.** `gemini-embedding-001` vectors have 3,072 dimensions, about 12 KB per chunk, plus the text.
- **Don't index while deploying.** Startup marks any repo that's mid-indexing as failed, because it assumes a single
  server. Running the script during a restart can trigger this, and running it again fixes it.
- **Searching scans each repo's vectors.** There's no vector index, which is fine at this size.

## Troubleshooting

| What you see | What to do |
|---|---|
| Deploy fails with an out-of-memory error | Lower `--limit-concurrency` in the `Dockerfile`, or check that it installs `requirements-base.txt`. |
| `/api/health` says `db: unavailable` | Check `DATABASE_URL` in Render. The real error is in Render's logs. |
| "Re-index needed" on a demo repo | It was indexed with another model. Run the script with `EMBEDDING_MODEL` unchanged, or re-index with the password. |
| "Gemini's daily embedding quota is used up" | Wait for the reset, then run the script again. |
| A model name is dropped at startup | Check the names with `scripts\list_models.py`, and override `CHAT_MODELS` or `GUIDE_MODELS` in Render. |

## Try the image locally

```powershell
docker compose --profile app up --build
```

This serves http://localhost:8000 with demo mode off, using your `.env` and the local database. Its indexing and
questions use real Gemini calls.

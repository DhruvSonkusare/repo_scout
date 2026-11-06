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

## 7. Turn on Sign in with GitHub (optional)

Guests keep working without this. Until all four settings below are set, the button doesn't appear.

### In GitHub

1. Open your profile picture, then **Settings**, **Developer settings**, **OAuth apps**, **New OAuth App**.
2. Fill in:

   | Field | Value |
   |---|---|
   | Application name | `RepoScout` |
   | Homepage URL | `https://your-service.onrender.com` |
   | Authorization callback URL | `https://your-service.onrender.com/auth/github/callback` |

   Use **Add callback URL** to also add `http://localhost:8000/auth/github/callback` if you want to test locally.
   Leave **Enable Device Flow** off.
3. Click **Register application**, copy the **Client ID**, then click **Generate a new client secret** and copy it.

### In Render

Add these under the service's **Environment**, then save. A Blueprint only asks for new variables when it's created,
so an existing service needs them added by hand.

| Name | Value |
|---|---|
| `GITHUB_CLIENT_ID` | The Client ID from GitHub |
| `GITHUB_CLIENT_SECRET` | The client secret from GitHub |
| `SESSION_SECRET` | A random string of at least 32 characters. The app won't start with a shorter one. |
| `PUBLIC_BASE_URL` | `https://your-service.onrender.com`, exactly as the callback URL starts |

Generate the session secret with:

```powershell
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

Changing `SESSION_SECRET` signs everyone out, which is the way to end all sessions at once. Signing out only clears
the cookie in that browser, so a copied cookie keeps working until it expires after 14 days or the secret changes.

### Check it

Open the site, click **Sign in with GitHub**, and approve. The label at the top should change from **Public** to
**My workspace**. Run a search, then open the site in a private window: the search shouldn't be listed there.

## The two spaces

| | Guest | Signed in |
|---|---|---|
| Searches, chats, guides for search results | Shared with all guests | Private to the user |
| Public demo repos | Read and ask | Read and ask |
| Adding a new repo | Index password | No password, within the limits below |
| Repos the user added | | Visible to users who added them. Indexed once and shared. |
| Re-indexing or deleting a public repo | Index password | Index password |
| Rate limits | Per IP | Per user |
| Daily cap | Shared by all guests | Per user, plus a total for all signed-in users |

A repo's onboarding guide is written from its code alone, with nothing from the user, so everyone who can see the repo
shares it. A signed-in user can't see public searches and chats until they sign out.

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
| `DAILY_REQUEST_CAP` | 100 | Searches, questions and guides per day for all guests, reset at midnight UTC. |
| `USER_DAILY_REQUEST_CAP` | 30 | Searches, questions and guides per day for each signed-in user. |
| `USER_REPOS_PER_DAY` | 3 | Indexing jobs per day for each signed-in user. Adding a repo that's already here doesn't count. |
| `SIGNED_IN_DAILY_REQUEST_CAP` | 100 | The total for all signed-in users, so that many accounts can't use up the quota. |
| `SIGNED_IN_CHUNKS_PER_DAY` | 400 | Chunks all signed-in users together can index per day. |

The hourly limits count per user for someone signed in. `MAX_REPO_SIZE_MB` and `MAX_INDEX_CHUNKS` apply to everyone.

### What uses the Gemini embedding quota

The free tier allows 1,000 embedded texts a day, as far as this project has measured.

| Action | Embeddings |
|---|---|
| Indexing | 1 per chunk |
| A question | 1 |
| A guide for a search result | 6 |
| An onboarding guide | 5 |
| A search | 0 |

At the default caps, the worst case is 100 guest guides and 100 signed-in guides, or 1,200 embeddings, plus 400 chunks
of signed-in indexing. That's over the quota, but it needs every request to be a guide, and each person can only
generate 5 an hour. If it does happen, Gemini's quota error shows until the reset, and you can lower the caps. To index
something bigger, do it from your machine with `scripts\index_demo_repos.py --repo owner/name`. The script refuses
more than 700 chunks at once. `psf/requests`, for example, is 603.

Chat and guide calls count against separate Gemini quotas for each model. When those run out, the app falls back to
the next model, and then shows a message.

## Good to know

- **The guest space is public.** Guests can see each other's searches and chats, and the page says so.
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
| GitHub says the redirect URI isn't associated with the app | Make `PUBLIC_BASE_URL` and the callback URL in GitHub match exactly. |
| "Signing in with GitHub didn't finish" | Check Render's logs. It's usually a wrong client secret, or an old tab. Try again from a fresh page. |
| The app won't start, and mentions `SESSION_SECRET` | Use a value of at least 32 characters. |
| A model name is dropped at startup | Check the names with `scripts\list_models.py`, and override `CHAT_MODELS` or `GUIDE_MODELS` in Render. |

## Try the image locally

```powershell
docker compose --profile app up --build
```

This serves http://localhost:8000 with demo mode off, using your `.env` and the local database. Its indexing and
questions use real Gemini calls.

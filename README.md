# RepoScout

An AI developer assistant that finds open-source repos for a problem described in plain English, and answers
questions about any public GitHub codebase with file and line citations.

This project is under construction. The full README comes in a later step. To run the public demo on Render
with Neon Postgres, and to turn on the optional Sign in with GitHub, see [docs/DEPLOY.md](docs/DEPLOY.md).

## Notes

- **Dev server:** run `uvicorn app.main:app --reload --reload-dir app`. Plain `--reload` also watches `data/`, so
  cloning a repo restarts the server and kills the indexing job.
- **GPU embeddings (NVIDIA, Windows):** `requirements.txt` runs local embeddings on the CPU. To use CUDA, stop the
  server and run `powershell -ExecutionPolicy Bypass -File scripts\install_gpu.ps1`, because a leftover CPU
  `onnxruntime` would otherwise replace the GPU build, and embeddings would silently stay on the CPU. The server log
  says which device is in use.
- **Vectors** live in Postgres with pgvector, next to the rest of the data. Local development uses the
  `pgvector/pgvector` image from `docker-compose.yml`.
- **Gemini free tier:** daily limits are low and differ by model, and every question uses one or two requests. So
  `CHAT_MODELS` and `GUIDE_MODELS` are ordered lists: on a 404, 429, 500, 503 or 504 the next model answers. A model
  out of daily quota is skipped until midnight Pacific time, a rate-limited one for about a minute, and one that
  returns a 404 until the server restarts. The server log and each answer's `model` field say which model answered.

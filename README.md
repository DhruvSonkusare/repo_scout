# RepoGuide

An AI developer assistant that finds open-source repos for a problem described in plain English, and answers
questions about any public GitHub codebase with file and line citations.

This project is under construction. The full README comes in a later step.

## Notes

- **Dev server:** run `uvicorn app.main:app --reload --reload-dir app`. Plain `--reload` also watches `data/`, so
  cloning a repo restarts the server and kills the indexing job.
- **GPU embeddings (NVIDIA, Windows):** `requirements.txt` runs local embeddings on the CPU. To use CUDA, stop the
  server and run `powershell -ExecutionPolicy Bypass -File scripts\install_gpu.ps1`. It needs a separate script
  because `chromadb` otherwise pulls in the CPU `onnxruntime` over the GPU build, and embeddings silently stay on the
  CPU. The server log says which device is in use.
- **Gemini free tier:** daily limits are low and differ by model, and every question uses one or two requests. If
  chat returns a daily quota error, set a different `CHAT_MODEL` or try again the next day.

Intelligent Document Retrieval & Question Answering (RAG)

Upload PDF/TXT files, ask questions, get **grounded answers with citations** from **Claude (Anthropic)**.
The system refuses to guess: if the documents don't contain the answer, it says so.

| Layer | Choice |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-5-5` for answers, `claude-haiku-5-5` for follow-up rewriting) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers (local; Anthropic has no embeddings API) |
| Vector DB | FAISS `IndexFlatIP` (exact cosine) + on-disk registry |
| Retrieval | Dense + BM25 → Reciprocal-Rank-Fusion → relevance gate → duplicate collapse |
| API / UI | FastAPI (OpenAPI at `/docs`) + single-page web UI at `/` |

## Quick start

```bash
cd problem1-rag
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env && export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=...
uvicorn rag.api:app --reload                              # first start downloads the embedding model (~130 MB)
```
Open <http://localhost:8000>, drag in the files from `data/sample_docs/`, and ask questions from `data/eval/queries.json`.
Swagger UI: <http://localhost:8000/docs>.

No internet for the embedding model? `RAG_EMBEDDING_BACKEND=hashing` uses a dependency-free **lexical** fallback
(offline tests/CI only — not semantic, see `docs/evaluation_report.md`).

CLI alternative: `python -m rag.cli ingest data/sample_docs` then `python -m rag.cli ask "How long is the Orbit warranty?"`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload one or many PDF/TXT files (multipart `files`). Returns a per-file result: `ingested`, `duplicate` or `failed` (+ `error_code`) |
| GET / DELETE | `/documents`, `/documents/{id}` | List / remove documents |
| POST | `/ask` | `{"question": "...", "session_id": "optional"}` → structured answer |
| DELETE | `/sessions/{id}` | Forget a conversation |
| GET | `/health` | Index + model status |

```bash
curl -F "files=@data/sample_docs/security_handbook.pdf" localhost:8000/documents
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"What is the minimum password length?","session_id":"demo"}'
```
Response (abridged; values illustrative):
```json
{"answerable": true, "answer": "Passwords must be at least 14 characters long [S1].", "confidence": "high",
 "citations": [{"label":"S1","source":"security_handbook.pdf","page":1,"score":0.78,"snippet":"..."}],
 "conflict_detected": false, "warnings": [], "rewritten_query": "What is the minimum password length?",
 "retrieval": {"mode":"hybrid","top_score":0.78,"threshold":0.5,"passed_gate":true,"chunks_used":4}}
```
Errors are always `{"error": {"code": "...", "message": "..."}}` with meaningful HTTP codes
(413 too large, 415 unsupported type, 422 corrupt/no text, 404, 503 LLM unavailable, 502 bad model output).

## Project layout

```
rag/
  config.py          all tunables (env-overridable)       ingestion/loader.py      validate + extract PDF/TXT
  schemas.py         Pydantic models / API contract       ingestion/preprocess.py  clean text, strip headers
  exceptions.py      typed errors -> HTTP codes           chunking.py              recursive chunker
  embeddings.py      bge (local) / hashing (offline)      vectorstore.py           FAISS + persistence
  retrieval.py       hybrid, gate, dedup, conflicts       llm.py                   Anthropic wrapper
  generation.py      prompt + structured output           grounding.py             post-LLM verification
  memory.py          sessions + query rewriting           pipeline.py              orchestration
  api.py             FastAPI                              cli.py                   small CLI
static/index.html    web UI          tests/   86 tests          evaluation/run_eval.py   metrics harness
data/sample_docs/    6 docs incl. a version conflict, an exact duplicate, a prompt-injection line
data/eval/queries.json   20 labelled test queries          data/edge_cases/   scanned-style PDF (no text layer)
docs/                architecture, decisions, reasoning, evaluation report, limitations, AI disclosure, demo script
```

## Tests and evaluation

```bash
python -m pytest tests -q                                   # 86 tests, no API key / network needed (Claude is stubbed)
python -m evaluation.run_eval --calibrate --chunk-sweep     # retrieval + gate metrics, offline lexical embedder
# full run on your machine (real embeddings + real Claude):
python -m evaluation.run_eval --backend sentence-transformers --calibrate --chunk-sweep --llm
python -m evaluation.run_eval --backend sentence-transformers --embedding-model sentence-transformers/all-MiniLM-L6-v2 --calibrate   # compare embedders
```


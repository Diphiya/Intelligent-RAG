Intelligent Document Retrieval & Question Answering (RAG)
1. Project overview
A web application and API for asking questions about your own PDF and TXT documents. Upload files, ask questions in plain language, and get answers that are grounded in the documents and cited. If the documents do not contain the answer, the system says so instead of guessing.
What it does
Ingests PDF and TXT files: validation, text extraction, cleaning (repeated headers/footers removed), duplicate detection, chunking, embedding and indexing. Unsupported, corrupt, oversized or scanned (no text layer) files are rejected with a clear error.
Retrieves relevant passages with hybrid search (semantic + keyword search, merged with Reciprocal Rank Fusion), a relevance threshold, and duplicate removal.
Answers with Anthropic Claude, using only the retrieved passages. Every statement carries a citation such as `[S1]`, shown with the source file, page and passage.
Checks the answer after generation: citations must point to real passages, and numbers in the answer must appear in the cited text.
Handles hard cases: conflicting documents (for example two policy versions) are reported as a conflict, follow-up questions are rewritten using the conversation, unanswerable questions are refused, and instructions hidden inside documents (prompt injection) are ignored.
Technology
Layer	Choice
LLM	Anthropic Claude: `claude-sonnet-5-5` for answers, `claude-haiku-5-5` for rewriting follow-up questions
Embeddings	`BAAI/bge-small-en-v1.5` through sentence-transformers, run locally (Anthropic offers no embeddings API)
Vector index	FAISS `IndexFlatIP` (exact cosine similarity), saved to disk
Keyword search	BM25 (implemented in `rag/retrieval.py`)
API and UI	FastAPI, with a single-page web UI at `/` and Swagger UI at `/docs`
How a question flows
```
question -> (rewrite if follow-up) -> dense search + BM25 -> RRF merge -> relevance gate
         -> duplicate / conflict handling -> Claude (answer + citations) -> verification -> answer
```
Project layout
```
rag/                 application code
  ingestion/         file validation, text extraction, cleaning
  chunking.py        recursive chunker              embeddings.py   local embedding model
  vectorstore.py     FAISS index + persistence      retrieval.py    hybrid search, gate, dedup
  generation.py      prompt + structured output     grounding.py    post-answer verification
  memory.py          sessions + follow-up rewrite   pipeline.py     orchestration
  api.py             FastAPI app                    cli.py          command-line interface
static/index.html    web UI
data/sample_docs/    6 sample documents (a version conflict, an exact duplicate, a prompt-injection line)
data/eval/           labelled test queries          data/edge_cases/   scanned-style PDF
evaluation/          evaluation script              tests/             automated tests
```
2. Dependencies
Python 3.10 or newer and `pip`.
An Anthropic API key (create one in the Anthropic Console). It is used for answering and for rewriting follow-up questions.
Internet access on the first run, to download the embedding model (about 130 MB, cached afterwards).
About 1 GB of free disk space for the Python packages and the model.
Python packages (installed from `requirements.txt`):
Package	Used for
`anthropic`	Claude API client
`fastapi`, `uvicorn`, `python-multipart`	web API, server, file uploads
`pydantic`	data validation and the structured-answer schema
`pypdf`	PDF text extraction
`faiss-cpu`, `numpy`	vector index
`sentence-transformers`	local embedding model
`scikit-learn`	only the offline `hashing` embedder (tests)
`pytest`, `httpx`, `reportlab`	tests and generating sample PDFs
3. Setup instructions
Step 1. Get the code and open a terminal in the project folder
```bash
git clone https://github.com/Diphiya/Intelligent-RAG.git
cd Intelligent-RAG
```
Step 2. Create and activate a virtual environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```
If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` once and try again. The prompt shows `(.venv)` when the environment is active.
Step 3. Install the dependencies
```bash
pip install -r requirements.txt
```
Step 4. Add your API key
Copy the template to a file named `.env`, then open it and paste your key after `ANTHROPIC_API_KEY=`:
```powershell
# Windows (PowerShell)
copy .env.example .env
notepad .env
```
```bash
# macOS / Linux
cp .env.example .env
nano .env
```
The line should look like `ANTHROPIC_API_KEY=sk-ant-...` (no quotes, no spaces). Save the file. `.env` is listed in `.gitignore`, so it is never committed. Do not share the key.
Optional settings (add them to `.env`; the defaults work well):
Variable	Default	Meaning
`RAG_LLM_MODEL`	`claude-sonnet-5-5`	model that writes the answers
`RAG_REWRITE_MODEL`	`claude-haiku-5-5`	model that rewrites follow-up questions
`RAG_EMBEDDING_BACKEND`	`sentence-transformers`	`hashing` = offline lexical fallback (tests only, not semantic)
`RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`	`400` / `66`	chunk length and overlap in characters
`RAG_TOP_K`	`5`	passages sent to the model
`RAG_MIN_RELEVANCE`	`0.60`	similarity below which a question is treated as not answerable
`RAG_MAX_UPLOAD_MB`	`20`	maximum upload size in MB
Step 5. Verify the installation (needs no API key and no internet)
```bash
python -m pytest tests -q
```
All tests should pass (94 tests; Claude is replaced by a scripted stand-in in the tests).
4. Execution steps
Step 1. Start the server
```bash
uvicorn rag.api:app --port 8000
```
Wait for `Application startup complete`. The first start downloads the embedding model, so it takes a little longer. If you see the warning `No ANTHROPIC_API_KEY`, the `.env` file is missing or empty; see setup step 4.
Step 2. Open the app
Go to http://localhost:8000.
Step 3. Upload documents
Drag the files from `data/sample_docs/` into the upload box (or click to choose them). Each file is reported as `ingested`, `duplicate` (identical content already indexed) or an error. The sample set contains:
File	Why it is included
`hr_leave_policy_2023.txt`, `hr_leave_policy_2024.txt`	two versions that disagree (20 vs 25 leave days)
`hr_leave_policy_2024_copy.txt`	exact duplicate, detected and skipped
`orbit_home_hub_faq.txt`	contains a prompt-injection line in a review
`engineering_onboarding.txt`	distractor that also mentions "incidents"
`security_handbook.pdf`, `quarterly_report_q3.pdf`	PDFs with page footers and a table
To test error handling, also upload `data/edge_cases/scanned_like.pdf` (no text layer, so it is rejected with an OCR message).
Step 4. Ask questions
Question	What to expect
What is the minimum password length?	cited answer: 14 characters, security handbook p.1
And how often does it have to be changed? (same conversation, right after the previous one)	rewritten to a full question; answer: every 180 days
How many days of paid annual leave do employees get?	both versions reported (20 and 25 days) with a conflict badge
Who won the 2018 FIFA World Cup?	refused, not covered by the documents
What is the parental leave policy?	refused, related topic but not stated
How much does the Orbit Home Hub cost?	refused; the injected "free of charge" line is ignored
Click New conversation to start a fresh session. Open a citation to see the exact passage, file and page. More test questions with expected evidence are in `data/eval/queries.json`.
Step 5. Use the API instead (optional)
Interactive documentation is at http://localhost:8000/docs. From a terminal (on Windows PowerShell use `curl.exe`):
```bash
curl -F "files=@data/sample_docs/security_handbook.pdf" localhost:8000/documents
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question":"What is the minimum password length?","session_id":"demo"}'
```
Method	Path	Purpose
POST	`/documents`	upload one or many PDF/TXT files
GET / DELETE	`/documents`, `/documents/{id}`	list / remove documents
POST	`/ask`	ask a question; returns answer, citations, confidence, conflict info, warnings
DELETE	`/sessions/{id}`	forget a conversation
GET	`/health`	index and model status
Errors always look like `{"error": {"code": "...", "message": "..."}}` with meaningful HTTP codes (413 too large, 415 unsupported type, 422 corrupt or no text, 404 not found, 503 model unavailable, 502 bad model output).
Step 6. Command-line alternative (optional)
```bash
python -m rag.cli ingest data/sample_docs
python -m rag.cli ask "How long is the Orbit warranty?"
```
Step 7. Run the evaluation (optional)
```bash
python -m evaluation.run_eval --backend sentence-transformers --calibrate --chunk-sweep --llm
```
This runs the 25 labelled questions through retrieval and Claude and prints retrieval metrics (Hit@k, MRR, evidence recall), the relevance-threshold sweep, a chunk-size sweep, and end-to-end answer correctness. It makes a few dozen API calls (a few cents). Without `--llm` it measures retrieval only and needs no API key.
Stopping and resetting
Press `Ctrl+C` in the terminal to stop the server. The index is saved in the `storage/` folder; delete that folder to start with an empty index.

# Demo video script (target 4–5 minutes)

1. **(20 s) Intro** — problem + stack (Claude, bge-small, FAISS+BM25, FastAPI). Show `docs/architecture.png`.
2. **(40 s) Upload** — start server, drag all files from `data/sample_docs/`. Point out: the *copy* of the 2024 policy is reported as `duplicate`; upload `data/edge_cases/scanned_like.pdf` → clear "needs OCR" error; upload a `.docx` → 415.
3. **(60 s) Normal Q&A with citations** — "What is the minimum password length?" → cited answer, open the citation snippet and page. "How soon must a security incident be reported?" (note: the engineering guide also mentions incidents; the right source is chosen).
4. **(45 s) Follow-up** — in the same conversation: "And how often does it have to be changed?" → show *searched for: "How often must passwords be rotated?"*.
5. **(45 s) Conflicting documents** — "How many days of paid annual leave do employees get?" → both versions, conflict badge, 2024 supersedes 2023.
6. **(60 s) Not answerable** — "Who won the 2018 FIFA World Cup?" (gate rejects, no LLM call, show score vs threshold); "What is the parental leave policy?" (retrieval passes, Claude abstains); "How much does the Orbit Home Hub cost?" (injected "free of charge" line is ignored).
7. **(30 s) Evaluation** — show `docs/evaluation_report.md` tables and run `python -m pytest -q`.
8. **(20 s) Limitations + next steps** — OCR, reranker, larger eval set.

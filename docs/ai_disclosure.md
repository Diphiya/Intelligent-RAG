# AI Tool Disclosure

- **Claude (Anthropic), used through claude.ai** was used as an AI coding assistant to help plan this submission and to draft the code, tests, sample documents, evaluation harness and documentation in this folder. It also ran the unit tests and the offline retrieval evaluation in its own sandbox and the numbers reported in `evaluation_report.md` come from those runs.
- **Runtime use of AI:** the application itself calls Claude (Anthropic API) for answer generation and follow-up rewriting. Embeddings come from an open-source local model.
- **What the assistant could not run:** live calls to the Anthropic API and the real embedding model (no key / model-hub access in its sandbox). The candidate ran them (Run 1) and reported the output; the assistant analysed it, found two bugs in its own checking code, and proposed the tuning changes described in `evaluation_report.md`.
- **Candidate's responsibility:** I reviewed and understand the design decisions, will run the full evaluation myself, and can explain and modify every component in the technical interview.

> ✏️ Candidate: edit this file so it states accurately what *you* reviewed, changed and ran before submitting.

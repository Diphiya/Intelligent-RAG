# Known limitations and possible improvements

| # | Limitation | Consequence | Improvement |
|---|---|---|---|
| 1 | **No OCR.** Image-only PDFs have no text layer. | Upload is rejected with `no_extractable_text` (never silently indexed as empty). | Detect low text density and route pages to Tesseract or a vision-capable Claude call. |
| 2 | **Tables/multi-column PDFs** are flattened to text by `pypdf`. | Row/column relationships can be lost; numbers stay but headers may detach. | `pdfplumber`/layout-aware parser, or table extraction into Markdown. |
| 3 | **Relevance threshold is embedder-specific** and calibrated on 25 queries — the same ones used to report results, so the gate figures are optimistic. | A wrong threshold either rejects answerable questions or admits junk. | Hold-out calibration set; re-calibrate whenever the embedder or corpus changes (`--calibrate`). |
| 4 | **Topically related but unanswered questions pass the gate** (e.g. parental leave, price of the Orbit hub). | Protection falls to the LLM's abstention + post-verification. | Reranker score as a second gate; an LLM/NLI faithfulness check on the final answer. |
| 5 | **Numbers spelled out in words** ("twenty-five") are not caught by the numeric grounding check, and years/versions are accepted if they appear in the cited passage *or its file name*. | A hallucinated number written in words would not be flagged. | Number normalisation or an NLI check. |
| 6 | **Conflict detection is heuristic** (similar chunk + different numbers) plus the LLM's judgement. | Conflicts in non-numeric facts (e.g. "Monday" vs "Tuesday") are only found if the LLM notices. | Claim-level extraction and comparison. |
| 7 | **No document dates/versions metadata.** | The system can say that 2024 supersedes 2023 only when the text says so. | Capture effective dates at upload and let the user choose "latest wins". |
| 8 | **Chunks never span pages.** | A sentence that crosses a page break is split into two chunks. | Page-spanning chunks with multi-page citations. |
| 9 | **Conversation memory is in-process.** | History is lost on restart and not shared across workers. | Redis/DB-backed sessions. |
| 10 | **Single-process, no auth.** | Not suitable for multi-user production as-is; uploaded files are visible to everyone using the instance. | Per-user collections, authentication, background ingestion queue. |
| 11 | **Prompt injection** inside documents is mitigated (delimited passages, explicit rule, escaping) but not eliminated. The sample FAQ contains an injected line, and it *is* retrieved for the price question. | A weak model could obey it. | Run the `q18` case in `--llm` mode on every model change; add an output filter. |
| 12 | **Evaluation scale.** 25 queries over 6 small documents, written by the same person who wrote the documents. | Metrics have a wide margin of error (one query = 6.7 points of Hit@k). | Larger, independently written query set; judge-assisted answer scoring. |
| 13 | **Not verified in my build environment:** the real embedding model and real Claude calls (no model-hub or API access there). | Those code paths are covered by unit tests with stubs/fakes, not by a live run. | Run the commands in README "Tests and evaluation" and paste results into the report. |

- **Multi-turn behaviour is only manually tested.** The evaluation asks each question in a fresh session. Manual testing of a 4-question chat found cross-turn "corrections" by the model, false number warnings and a missed document-version conflict (see `evaluation_report.md` §6.10); these were fixed and unit-tested, but there is no automated multi-turn evaluation set yet.
- **Version-sibling retrieval is a heuristic** (chunk similarity ≥ 0.75 with different numbers); it can add one or two loosely related passages and it assumes versions of a document keep a similar layout.

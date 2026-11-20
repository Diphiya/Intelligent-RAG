# Reasoning and Technical Challenges (assessment items 1–7)

## 1. Why this chunking strategy?
See decision D1. In short: recursive splitting on paragraph/sentence boundaries keeps each rule with its numbers; 400 characters (~100 tokens) is roughly one paragraph of the policy/FAQ-style documents and is far below the embedder's 512-token limit; ~66-character overlap protects facts that straddle a boundary; chunks never cross pages so citations carry exact page numbers; the document title is prepended to the embedded text so "2023" and "2024" chunks stay distinguishable. I swept 250–1600 characters with the real embedder: Hit@1 was 1.00 at 250 and 400 characters but 0.80 at 1000+, so I moved from my initial 600 to 400 (D1).

## 2. How was the embedding model selected?
Constraints: Anthropic has no embedding API → a local open-source model; must run on a laptop CPU; must fit passages (512 tokens); should be retrieval-trained. Candidates: all-MiniLM-L6-v2 (fast, 256-token limit), **bge-small-en-v1.5** (chosen: small, retrieval-tuned, 512 tokens, query-instruction support), larger bge/e5/gte models (better but slower/heavier), hosted APIs (extra vendor and key). The selection method is built in: `run_eval.py --embedding-model X` scores each candidate on the same labelled queries (Hit@k, MRR, gate separation). bge-small ran on the candidate's machine and separates topics well (relevant 0.67–0.91 vs off-topic 0.48–0.53); a head-to-head against other models is not yet run, so it remains a reasoned default.

## 3. How does the system decide whether retrieved chunks are relevant?
Three signals, cheapest first: (a) the **absolute cosine similarity** of the best chunk must reach a calibrated threshold (0.60 default; relevant questions scored 0.67–0.91, an off-topic one 0.48) (ranking scores such as RRF are relative and cannot say "nothing is relevant", so the gate uses raw cosine); (b) only chunks above the threshold are passed to the LLM; (c) the LLM must still decide the passages *answer* the question (`answerable`), cite them, and the post-check verifies citations and numbers. The threshold is calibrated per embedder with `--calibrate`. Limitation: similarity measures topic closeness, not answerability (failure case 3).

## 4. Potential causes of incorrect retrieval
- **Ingestion:** garbled PDF text, lost table structure, headers/footers polluting chunks, scanned pages with no text.
- **Chunking:** chunks too big (several topics, diluted embedding) or too small (fact separated from its context); a fact split across a boundary.
- **Embedding:** vocabulary gap/out-of-domain terms; numbers and identifiers poorly represented; the wrong query/passage prefix for the model.
- **Query:** follow-ups with pronouns, vague questions, multi-part questions; mismatch between user wording and document wording.
- **Ranking:** term-frequency bias (observed: failure case 2), duplicates crowding out diverse evidence, conflicting versions where only one is retrieved.
- **Threshold/scale:** a gate calibrated on the wrong embedder or corpus.
Each cause maps to a mitigation in the pipeline: preprocessing, recursive chunking + overlap, contextual title header, hybrid retrieval, query rewriting, duplicate collapse, calibrated gate, and evaluation by chunk-level evidence labels.

## 5. How are hallucinations reduced?
1. **Retrieval gate**: if nothing relevant exists, the LLM is never called.
2. **Constrained prompt**: answer only from numbered passages, never from world knowledge; abstain when passages are merely related; temperature 0.
3. **Mandatory citations** in a strict schema; every claim carries `[S#]`.
4. **Deterministic verification** after generation: citations must exist; an uncited "answer" is converted to an abstention; numbers must occur in the cited text (otherwise confidence is capped and a warning is returned).
5. **Conflict handling** instead of silently picking one value.
6. **Prompt-injection hygiene**: delimited, escaped passages marked as untrusted.
7. **Evaluation** with unanswerable and adversarial queries, tracked as metrics.
Residual risk is listed in `limitations.md` (e.g. spelled-out numbers, paraphrase errors not involving numbers).

## 6. Comparison with an alternative retrieval approach
Dense-only vs BM25-only vs hybrid (+ optional reranker) is compared in `evaluation_report.md` §3. Rationale for hybrid: BM25 wins on exact tokens (names, numbers, "2024"), dense wins on paraphrase, RRF merges them without score calibration. Measured (Run 1): Hit@1 dense 0.87 / BM25 0.93 / hybrid 0.93 — a one-query difference, with hybrid tying BM25. Run 2 added paraphrased queries: hybrid 0.95 vs dense 0.90 and BM25 0.90 (BM25 missed one paraphrase entirely) — again one query apart, so the pattern is the evidence, not the numbers.

## 7. Duplicate or conflicting information
- **Exact duplicate documents** (even renamed, CRLF, extra spaces): detected by hashing normalised cleaned text, skipped, and reported with `duplicate_of` (sample: `hr_leave_policy_2024_copy.txt`).
- **Near-duplicate chunks** (cosine ≥ 0.95) across documents: merged and shown as `also_in`, so the user sees all sources but the LLM context is not wasted — **only if numbers are identical**.
- **Conflicting facts** (20 vs 25 days): never merged. Both are retrieved; a conflict hint is added; Claude must report each version with its source, state whether one explicitly supersedes the other, and set `conflict_detected`. The UI shows a "conflicting sources" badge. The system never invents a winner from file dates it does not have.

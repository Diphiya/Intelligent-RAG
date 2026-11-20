# Evaluation results (embedder: hashing)

Corpus: 6 documents, 46 chunks (chunk_size=400, overlap=66). Queries: 25 (20 answerable, 5 unanswerable).

## 1. Retrieval quality by method (answerable queries, no gate)

| mode   | Hit@1 | Hit@3 | Hit@5 | MRR | EvidenceRecall@5 | Precision@3 | queries with no relevant chunk in top-5 |
|--------|-------|-------|-------|-----|------------------|-------------|------|
| dense  | 0.85 | 0.90 | 0.95 | 0.89 | 0.95 | 0.38 | q22 |
| bm25   | 0.90 | 0.95 | 0.95 | 0.93 | 0.95 | 0.40 | q22 |
| hybrid | 0.85 | 0.95 | 0.95 | 0.89 | 0.95 | 0.40 | q22 |

Follow-up query WITHOUT rewriting (hybrid): Hit@5=0.90, misses=['q13', 'q22'] -> compare with the table above, where the follow-up is evaluated in its rewritten standalone form.

## 2. Relevance gate: best dense cosine score per query

| id | answerable | top cosine | passes gate @0.08 |
|---|---|---|---|
| q01 | yes | 0.408 | yes |
| q02 | yes | 0.227 | yes |
| q03 | yes | 0.284 | yes |
| q04 | yes | 0.113 | yes |
| q05 | yes | 0.366 | yes |
| q06 | yes | 0.468 | yes |
| q07 | yes | 0.392 | yes |
| q08 | yes | 0.421 | yes |
| q09 | yes | 0.197 | yes |
| q10 | yes | 0.272 | yes |
| q11 | yes | 0.256 | yes |
| q12 | yes | 0.406 | yes |
| q13 | yes | 0.219 | yes |
| q14 | yes | 0.104 | yes |
| q15 | yes | 0.186 | yes |
| q16 | NO | 0.036 | no |
| q17 | NO | 0.265 | yes |
| q18 | NO | 0.361 | yes |
| q19 | NO | 0.000 | no |
| q20 | NO | 0.229 | yes |
| q21 | yes | 0.098 | yes |
| q22 | yes | 0.132 | yes |
| q23 | yes | 0.095 | yes |
| q24 | yes | 0.090 | yes |
| q25 | yes | 0.056 | no |

At threshold 0.08: answerable queries kept = 95%, unanswerable queries rejected before the LLM = 40%.

### Threshold sweep (selected points)

| threshold | answerable kept | unanswerable rejected |
|---|---|---|
| 0.00 | 100% | 0% |
| 0.05 | 100% | 40% |
| 0.10 | 80% | 40% |
| 0.15 | 65% | 40% |
| 0.20 | 55% | 40% |
| 0.25 | 45% | 60% |
| 0.30 | 30% | 80% |
| 0.35 | 30% | 80% |
| 0.40 | 20% | 100% |
| 0.45 | 5% | 100% |
| 0.50 | 0% | 100% |
| 0.55 | 0% | 100% |
| 0.60 | 0% | 100% |
| 0.65 | 0% | 100% |
| 0.70 | 0% | 100% |
| 0.75 | 0% | 100% |
| 0.80 | 0% | 100% |
| 0.85 | 0% | 100% |
| 0.90 | 0% | 100% |
| 0.95 | 0% | 100% |
| 1.00 | 0% | 100% |

**Recommended threshold: 0.00** = weakest answerable top-score (0.056) minus a 0.05 safety margin -> answerable kept 100%, unanswerable rejected 0%. Set RAG_MIN_RELEVANCE=0.00. (Calibrated on the same queries it is reported on, so treat the rejection rate as optimistic.)

## 3. Chunk-size sweep (hybrid)

| chunk_size | chunks | Hit@1 | Hit@3 | MRR | EvidenceRecall@5 | Precision@3 |
|---|---|---|---|---|---|---|
| 250 | 69 | 0.75 | 0.95 | 0.86 | 1.00 | 0.43 |
| 400 | 46 | 0.85 | 0.95 | 0.89 | 0.95 | 0.40 |
| 600 | 31 | 0.85 | 1.00 | 0.93 | 1.00 | 0.43 |
| 1000 | 18 | 0.95 | 1.00 | 0.97 | 1.00 | 0.42 |
| 1600 | 13 | 0.90 | 1.00 | 0.94 | 1.00 | 0.38 |

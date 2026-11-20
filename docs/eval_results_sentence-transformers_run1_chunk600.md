<!-- Run 1: executed by the candidate on their own machine (bge-small-en-v1.5 + claude-sonnet-5-5), chunk_size=600, threshold 0.50, 20 queries.
     Archived verbatim. Later changes (chunk 400, threshold 0.60, grounding + metric fixes, 5 paraphrase queries) are evaluated in Run 2. -->
# Evaluation results (embedder: sentence-transformers)

Corpus: 6 documents, 31 chunks (chunk_size=600, overlap=100). Queries: 20 (15 answerable, 5 unanswerable).

## 1. Retrieval quality by method (answerable queries, no gate)

| mode   | Hit@1 | Hit@3 | Hit@5 | MRR | EvidenceRecall@5 | Precision@3 | queries with no relevant chunk in top-5 |
|--------|-------|-------|-------|-----|------------------|-------------|------|
| dense  | 0.87 | 1.00 | 1.00 | 0.91 | 1.00 | 0.40 | - |
| bm25   | 0.93 | 1.00 | 1.00 | 0.97 | 1.00 | 0.44 | - |
| hybrid | 0.93 | 1.00 | 1.00 | 0.97 | 1.00 | 0.42 | - |

Follow-up query WITHOUT rewriting (hybrid): Hit@5=1.00, misses=[]

## 2. Relevance gate: best dense cosine score per query (threshold 0.50)

| id | answerable | top cosine | passes gate |
|---|---|---|---|
| q01 | yes | 0.800 | yes |
| q02 | yes | 0.691 | yes |
| q03 | yes | 0.725 | yes |
| q04 | yes | 0.751 | yes |
| q05 | yes | 0.838 | yes |
| q06 | yes | 0.841 | yes |
| q07 | yes | 0.865 | yes |
| q08 | yes | 0.731 | yes |
| q09 | yes | 0.772 | yes |
| q10 | yes | 0.673 | yes |
| q11 | yes | 0.911 | yes |
| q12 | yes | 0.741 | yes |
| q13 | yes | 0.771 | yes |
| q14 | yes | 0.747 | yes |
| q15 | yes | 0.731 | yes |
| q16 | NO | 0.475 | no |
| q17 | NO | 0.686 | yes |
| q18 | NO | 0.746 | yes |
| q19 | NO | 0.531 | yes |
| q20 | NO | 0.573 | yes |

At threshold 0.50: answerable queries kept = 100%, unanswerable queries rejected before the LLM = 20%.

### Threshold sweep (selected points)

| threshold | answerable kept | unanswerable rejected |
|---|---|---|
| 0.45 | 100% | 0% |
| 0.50 | 100% | 20% |
| 0.55 | 100% | 40% |
| 0.60 | 100% | 60% |
| 0.65 | 100% | 60% |
| 0.70 | 87% | 80% |
| 0.75 | 53% | 100% |
| 0.80 | 27% | 100% |

Tool output at the time: "Recommended threshold (highest rejection with 100% answerable kept): 0.67" (this rule was later replaced; see evaluation_report.md).

## 3. Chunk-size sweep (hybrid)

| chunk_size | chunks | Hit@1 | Hit@3 | MRR | EvidenceRecall@5 | Precision@3 |
|---|---|---|---|---|---|---|
| 250 | 69 | 1.00 | 1.00 | 1.00 | 1.00 | 0.49 |
| 400 | 46 | 1.00 | 1.00 | 1.00 | 1.00 | 0.44 |
| 600 | 31 | 0.93 | 1.00 | 0.97 | 1.00 | 0.42 |
| 1000 | 18 | 0.80 | 1.00 | 0.90 | 1.00 | 0.40 |
| 1600 | 13 | 0.80 | 1.00 | 0.90 | 1.00 | 0.40 |

## 4. End-to-end generation (Claude: claude-sonnet-5-5)

| metric | value |
|---|---|
| answer_correct_rate | 0.933 |
| citation_supported_rate | 1.0 |
| correct_abstention_rate | 1.0 |
| false_abstention_rate | 0.0 |
| conflict_flag_rate | 1.0 |
| injection_followed | True  (metric artifact, see evaluation_report.md §5) |
| avg_latency_ms | 2904.7 |

| id | category | expected answerable | got | correct | cited evidence | conflict flagged | warnings |
|---|---|---|---|---|---|---|---|
| q01 | conflict | True | True | True | True | True | - |
| q02 | conflict | True | True | False | True | True | - |
| q03 | conflict | True | True | True | True | True | unsupported_number:2024 |
| q04–q15 | factual / distractor / table / follow_up | True | True | True (all 12) | True | - | - |
| q16 | unanswerable_off_topic | False | False | - | - | - | - |
| q17 | unanswerable_near_miss | False | False | - | - | - | - |
| q18 | unanswerable_injection | False | False | - | - | - | - |
| q19 | unanswerable_off_topic | False | False | - | - | - | - |
| q20 | ambiguous | False | False | - | - | - | - |

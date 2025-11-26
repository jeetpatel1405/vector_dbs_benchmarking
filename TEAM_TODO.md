# Team TODO List: Vector Database Benchmarking - Publication Readiness

**Created:** 2025-11-26
**Target:** Journal Publication Quality
**Reference:** BENCHMARKING_VECTOR_DATABASES_PERFORMANCE_FOR_LLM_KNOWLEDGE_BASES.pdf

---

## Executive Summary

The current implementation has solid infrastructure but **critical gaps** between what the journal article claims and what the code actually produces. The metrics in the paper (Recall@K, Precision@K, MRR) are not being calculated in the benchmark scripts - only similarity scores are output.

---

## Priority Legend

- 🔴 **P0 - Critical**: Publication blockers - must fix before submission
- 🟠 **P1 - High**: Research quality issues - should fix for credibility
- 🟡 **P2 - Medium**: Nice to have for stronger paper
- 🟢 **P3 - Low**: Future enhancements

---

## 🔴 P0 - CRITICAL (Week 1)

### Task 1.1: Implement Document-Level Recall@K
**Assignee:** _______________
**Estimated Time:** 4-6 hours
**Files to Modify:**
- `src/vector_dbs/rag_benchmark.py`
- All `Scripts/run_*_benchmark.py` files

**Problem:**
- Ground truth in `test_cases.json` uses `relevant_doc_ids` (e.g., "doc_001", "doc_002")
- But benchmarks retrieve **chunks**, not documents
- No mapping from retrieved chunks back to source documents

**Solution:**
```python
def calculate_recall_at_k(retrieved_chunks, ground_truth_doc_ids, k):
    """
    Calculate Recall@K using document-level ground truth.

    Args:
        retrieved_chunks: List of Chunk objects with metadata['doc_id']
        ground_truth_doc_ids: List of relevant document IDs from test_cases.json
        k: Number of top results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    # Extract doc_ids from retrieved chunks
    retrieved_doc_ids = set()
    for chunk in retrieved_chunks[:k]:
        doc_id = chunk.metadata.get('doc_id')
        if doc_id:
            retrieved_doc_ids.add(doc_id)

    # Calculate recall
    relevant_set = set(ground_truth_doc_ids)
    if len(relevant_set) == 0:
        return 0.0

    hits = len(retrieved_doc_ids & relevant_set)
    return hits / len(relevant_set)
```

**Acceptance Criteria:**
- [ ] Recall@1, Recall@3, Recall@5, Recall@10, Recall@20 calculated for each query
- [ ] Results JSON includes `recall_at_k` metrics
- [ ] Values are between 0.0 and 1.0
- [ ] Unit tests pass

---

### Task 1.2: Implement Document-Level Precision@K
**Assignee:** _______________
**Estimated Time:** 3-4 hours
**Files to Modify:** Same as Task 1.1

**Solution:**
```python
def calculate_precision_at_k(retrieved_chunks, ground_truth_doc_ids, k):
    """
    Calculate Precision@K using document-level ground truth.

    Args:
        retrieved_chunks: List of Chunk objects with metadata['doc_id']
        ground_truth_doc_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    # Extract unique doc_ids from top-k retrieved chunks
    retrieved_doc_ids = []
    seen = set()
    for chunk in retrieved_chunks[:k]:
        doc_id = chunk.metadata.get('doc_id')
        if doc_id and doc_id not in seen:
            retrieved_doc_ids.append(doc_id)
            seen.add(doc_id)

    if len(retrieved_doc_ids) == 0:
        return 0.0

    relevant_set = set(ground_truth_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_set)
    return hits / len(retrieved_doc_ids)
```

**Acceptance Criteria:**
- [ ] Precision@1, Precision@5, Precision@10 calculated
- [ ] Results JSON includes `precision_at_k` metrics
- [ ] Unit tests pass

---

### Task 1.3: Implement MRR (Mean Reciprocal Rank)
**Assignee:** _______________
**Estimated Time:** 3-4 hours
**Files to Modify:** Same as Task 1.1

**Definition:** MRR = (1/|Q|) × Σ(1/rank_i) where rank_i is the position of the first relevant result

**Solution:**
```python
def calculate_mrr(retrieved_chunks, ground_truth_doc_ids):
    """
    Calculate Mean Reciprocal Rank.

    Args:
        retrieved_chunks: List of Chunk objects with metadata['doc_id']
        ground_truth_doc_ids: List of relevant document IDs

    Returns:
        Reciprocal rank (1/rank of first relevant result, or 0 if none found)
    """
    relevant_set = set(ground_truth_doc_ids)

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        doc_id = chunk.metadata.get('doc_id')
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0  # No relevant document found

def calculate_mean_mrr(all_query_results, all_ground_truths):
    """Calculate MRR averaged across all queries."""
    reciprocal_ranks = []
    for retrieved, ground_truth in zip(all_query_results, all_ground_truths):
        rr = calculate_mrr(retrieved, ground_truth)
        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
```

**Acceptance Criteria:**
- [ ] MRR calculated for each benchmark run
- [ ] Results JSON includes `mrr` metric
- [ ] Value is between 0.0 and 1.0
- [ ] Unit tests pass

---

### Task 1.4: Fix Ingestion Phase Timing Breakdown
**Assignee:** _______________
**Estimated Time:** 2-3 hours
**Files to Modify:**
- `src/vector_dbs/rag_benchmark.py` (line 176-256)
- All `Scripts/run_*_benchmark.py` files

**Problem:**
Current results show:
```json
"parsing_time_sec": 0,
"embedding_time_sec": 0,
"insertion_time_sec": 0
```

The `IngestionMetrics` dataclass has these fields but they're not being read correctly.

**Solution:**
1. Check `IngestionMetrics` return values in `ingest_documents()`
2. Fix attribute access in benchmark scripts (lines 163-165 use wrong attribute names)
3. Ensure timing is captured in `ingest_documents()` method

**Current Code (Broken):**
```python
parsing_time = ingest_result.parsing_time if hasattr(ingest_result, 'parsing_time') else 0
```

**Should Be:**
```python
parsing_time = ingest_result.total_parsing_time if hasattr(ingest_result, 'total_parsing_time') else 0
```

**Acceptance Criteria:**
- [ ] `parsing_time_sec` shows actual parsing time
- [ ] `embedding_time_sec` shows actual embedding time
- [ ] `insertion_time_sec` shows actual insertion time
- [ ] Sum approximately equals `total_time_sec`

---

### Task 1.5: Update All Benchmark Scripts with New Metrics
**Assignee:** _______________
**Estimated Time:** 4-6 hours
**Files to Modify:**
- `Scripts/run_faiss_benchmark.py`
- `Scripts/run_chroma_benchmark.py`
- `Scripts/run_qdrant_benchmark.py`
- `Scripts/run_pgvector_benchmark.py`
- `Scripts/run_weaviate_benchmark.py`
- `Scripts/run_milvus_benchmark.py`
- `Scripts/run_opensearch_benchmark.py`

**Changes Required:**
1. Load ground truth from test_cases.json
2. After each query, retrieve chunk objects (not just IDs)
3. Calculate Recall@K, Precision@K for each top_k value
4. Calculate MRR for each query
5. Add new metrics to results JSON
6. Update visualization to show IR metrics

**New Results JSON Structure:**
```json
{
  "query_results": [
    {
      "top_k": 5,
      "avg_latency_ms": 7.07,
      "queries_per_second": 141.49,
      "avg_similarity": 0.666,
      "recall_at_k": 0.73,
      "precision_at_k": 0.45,
      "mrr": 0.82
    }
  ]
}
```

**Acceptance Criteria:**
- [ ] All 7 benchmark scripts updated
- [ ] All scripts produce consistent metrics
- [ ] Results JSON schema is consistent across databases

---

### Task 1.6: Create Metrics Utility Module
**Assignee:** _______________
**Estimated Time:** 3-4 hours
**Files to Create:**
- `src/metrics/retrieval_metrics.py`
- `src/metrics/__init__.py`

**Purpose:** Centralize all IR metric calculations to avoid code duplication

**Module Contents:**
```python
# src/metrics/retrieval_metrics.py

def recall_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """Calculate Recall@K."""
    pass

def precision_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """Calculate Precision@K."""
    pass

def mrr(retrieved_doc_ids, relevant_doc_ids):
    """Calculate Mean Reciprocal Rank for single query."""
    pass

def ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """Calculate NDCG@K (for future use)."""
    pass

def calculate_all_metrics(retrieved_chunks, ground_truth, k_values=[1, 3, 5, 10, 20]):
    """Calculate all metrics for a single query."""
    return {
        'recall': {k: recall_at_k(...) for k in k_values},
        'precision': {k: precision_at_k(...) for k in k_values},
        'mrr': mrr(...)
    }
```

**Acceptance Criteria:**
- [ ] Module created with all metric functions
- [ ] Unit tests for each metric function
- [ ] Functions match standard IR metric definitions
- [ ] Imported and used by all benchmark scripts

---

## 🟠 P1 - HIGH PRIORITY (Week 2)

### Task 2.1: Add Resource Metrics to Results
**Assignee:** _______________
**Estimated Time:** 3-4 hours

**Problem:** `ResourceMonitor` exists but results JSONs contain no CPU/memory data.

**Solution:**
1. Wire `ResourceMonitor` into benchmark scripts
2. Add resource data to results JSON
3. Create resource usage visualization

**New Results Structure:**
```json
{
  "ingestion": {
    "resources": {
      "avg_cpu_percent": 45.2,
      "peak_cpu_percent": 89.1,
      "avg_memory_mb": 512.3,
      "peak_memory_mb": 1024.5
    }
  },
  "query_results": [
    {
      "resources": {
        "avg_cpu_percent": 12.3,
        "avg_memory_mb": 256.1
      }
    }
  ]
}
```

---

### Task 2.2: Implement Multi-Run Benchmarks
**Assignee:** _______________
**Estimated Time:** 4-6 hours

**Problem:** Single runs only - no statistical reliability.

**Solution:**
1. Add `--runs N` parameter to benchmark scripts (default N=3)
2. Run each configuration N times
3. Report median, IQR, standard deviation
4. Add error bars to visualizations

**New Results Structure:**
```json
{
  "query_results": [
    {
      "top_k": 5,
      "runs": 3,
      "avg_latency_ms": {"median": 7.07, "std": 0.45, "iqr": [6.8, 7.3]},
      "recall_at_k": {"median": 0.73, "std": 0.05}
    }
  ]
}
```

---

### Task 2.3: Integrate Curated Dataset from Google Drive
**Assignee:** _______________
**Estimated Time:** 4-6 hours

**Problem:** Current corpus is only 20 documents - too small for publication.

**Steps:**
1. Download curated dataset from Google Drive
2. Analyze dataset format and structure
3. Update data loading in `DocumentParser`
4. Create enhanced ground truth with more queries
5. Re-run all benchmarks

---

### Task 2.4: Update Comparison Visualizations
**Assignee:** _______________
**Estimated Time:** 3-4 hours
**Files to Modify:**
- `Scripts/generate_comparison_plots.py`
- `Scripts/recreate_query_comparison.py`

**Changes:**
1. Add Recall@K comparison across databases
2. Add Precision@K comparison
3. Add MRR comparison
4. Add error bars for multi-run data

---

## 🟡 P2 - MEDIUM PRIORITY (Week 3)

### Task 3.1: Implement NDCG (Normalized Discounted Cumulative Gain)
**Assignee:** _______________
**Estimated Time:** 3-4 hours

Standard ranking metric that accounts for position of relevant results.

---

### Task 3.2: Add Statistical Significance Testing
**Assignee:** _______________
**Estimated Time:** 4-6 hours

Implement Welch's t-test for pairwise database comparisons.

---

### Task 3.3: Scale Testing (100-1000 documents)
**Assignee:** _______________
**Estimated Time:** 6-8 hours

Test with larger corpora to generate scaling curves.

---

### Task 3.4: Concurrent Query Testing
**Assignee:** _______________
**Estimated Time:** 6-8 hours

Multi-threaded query benchmarks for true throughput measurement.

---

## 🟢 P3 - LOW PRIORITY (Future)

### Task 4.1: Add More Vector Databases
- Pinecone (cloud)
- Vespa
- Elasticsearch with vector plugin

### Task 4.2: Hybrid Search Benchmarks
- Combine vector + keyword search
- Test re-ranking strategies

### Task 4.3: Interactive Dashboard
- Real-time benchmark monitoring
- Web-based results viewer

---

## Testing Checklist

Before marking any task complete:

- [ ] Unit tests added for new functions
- [ ] Integration test runs successfully
- [ ] Results JSON validates against schema
- [ ] Visualizations generate without errors
- [ ] Documentation updated
- [ ] Code reviewed by team member

---

## File Change Summary

### New Files to Create
```
src/metrics/
├── __init__.py
└── retrieval_metrics.py
tests/
└── test_retrieval_metrics.py
```

### Files to Modify
```
src/vector_dbs/rag_benchmark.py          # Add document-level metrics
Scripts/run_faiss_benchmark.py           # Add IR metrics
Scripts/run_chroma_benchmark.py          # Add IR metrics
Scripts/run_qdrant_benchmark.py          # Add IR metrics
Scripts/run_pgvector_benchmark.py        # Add IR metrics
Scripts/run_weaviate_benchmark.py        # Add IR metrics
Scripts/run_milvus_benchmark.py          # Add IR metrics
Scripts/run_opensearch_benchmark.py      # Add IR metrics
Scripts/generate_comparison_plots.py     # Add IR metric visualizations
Scripts/recreate_query_comparison.py     # Add IR metric visualizations
```

---

## Weekly Standup Questions

1. What P0 tasks are complete?
2. What blockers exist?
3. Are benchmark results matching expected patterns?
4. Any issues with database connectivity?

---

## Definition of Done

A task is complete when:
1. Code is implemented and tested
2. All 7 databases produce the new metrics
3. Results JSON includes new fields
4. Visualizations updated
5. Documentation updated
6. PR reviewed and merged

---

## Contact

Questions? Reach out to:
- Project Lead: _______________
- Technical Lead: _______________

---

*This document should be updated weekly as tasks are completed.*

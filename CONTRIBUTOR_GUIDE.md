# Contributor Guide: Adding Database Benchmarks

This guide shows how to create a complete benchmark for a new vector database, following the **Qdrant example** (`scripts/run_qdrant_benchmark.py`).

## Overview

We've built a complete end-to-end benchmark for Qdrant that serves as a template for all other databases. The pattern is:

1. ✅ Load test documents and test cases
2. ✅ Initialize database adapter
3. ✅ Ingest documents and measure time
4. ✅ Run queries at multiple top-k values
5. ✅ Export results to JSON
6. ✅ Generate publication-ready plots

## What We Have

### Qdrant Benchmark (Reference Implementation)
- **Script**: `scripts/run_qdrant_benchmark.py`
- **Results**: `results/qdrant_experiment_001/`
  - `results.json` - Complete experimental data
  - `latency_vs_topk.png` - Publication-ready plot (300 DPI)
  - `config.json` - Configuration used

### Test Data (Shared)
- **Documents**: `Data/test_corpus/documents/` (20 climate science documents)
- **Test Cases**: `Data/test_corpus/test_cases.json` (10 queries with ground truth)
- **Metadata**: `Data/test_corpus/corpus_info.json`

## How to Add a New Database

### Step 1: Verify Your Database Adapter Works

First, test that the adapter for your database is working:

```bash
python test_adapters.py
```

Look for ✅ next to your database name. If it fails, fix the adapter first before proceeding.

### Step 2: Copy the Qdrant Template

```bash
cp scripts/run_qdrant_benchmark.py scripts/run_YOUR_DB_benchmark.py
```

Replace `YOUR_DB` with your database name (e.g., `chroma`, `faiss`, `pgvector`, etc.)

### Step 3: Modify the Configuration

Open your new script and update the `CONFIG` dictionary:

```python
CONFIG = {
    'corpus_path': 'Data/test_corpus/documents',  # SAME for all DBs
    'test_cases_path': 'Data/test_corpus/test_cases.json',  # SAME for all DBs
    'output_dir': 'results/YOUR_DB_experiment_001',  # CHANGE THIS
    'YOUR_DB_config': {  # CHANGE THIS - database-specific settings
        # Update with your database connection settings
        # Examples:
        # Chroma: {'persist_directory': './vector_stores/chroma_db', 'collection_name': 'benchmark_test'}
        # FAISS: {'index_path': './vector_stores/faiss_index', 'index_type': 'Flat'}
        # pgvector: {'host': 'localhost', 'port': 5432, 'database': 'vectordb', 'user': 'postgres', 'password': 'postgres', 'table_name': 'benchmark_test'}
    },
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',  # SAME for all DBs
    'embedding_type': 'sentence-transformers',  # SAME for all DBs
    'chunk_size': 512,  # SAME for all DBs
    'chunk_overlap': 50,  # SAME for all DBs
    'chunk_strategy': 'fixed',  # SAME for all DBs
    'top_k_values': [1, 3, 5, 10, 20],  # SAME for all DBs
    'batch_size': 100  # SAME for all DBs
}
```

### Step 4: Update the Import Statement

Change line 20 from:

```python
from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark
```

To:

```python
from src.vector_dbs.YOUR_DB_adapter import YOUR_DBRAGBenchmark
```

For example:
- Chroma: `from src.vector_dbs.chroma_adapter import ChromaRAGBenchmark`
- FAISS: `from src.vector_dbs.faiss_adapter import FAISSRAGBenchmark`
- pgvector: `from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark`

### Step 5: Update the Benchmark Initialization

Change the benchmark class instantiation (around line 117):

```python
benchmark = YOUR_DBRAGBenchmark(  # CHANGE THIS
    db_config=CONFIG['YOUR_DB_config'],  # CHANGE THIS
    embedding_generator=embedding_gen,
    chunk_size=CONFIG['chunk_size'],
    chunk_overlap=CONFIG['chunk_overlap'],
    chunk_strategy=CONFIG['chunk_strategy']
)
```

### Step 6: Update Result Export

Change the database name in results export (around line 236):

```python
results_data = {
    'database': 'YOUR_DB',  # CHANGE THIS
    'config': CONFIG,
    # ... rest stays the same
}
```

Also update the plot title (around line 264):

```python
ax1.set_title('YOUR_DB Query Latency vs Top-K', fontsize=14, fontweight='bold')  # CHANGE THIS
```

### Step 7: Run Your Benchmark

Make sure your database is running (via Docker Compose or standalone), then:

```bash
source venv/bin/activate
python scripts/run_YOUR_DB_benchmark.py
```

**Expected time**: 1-2 minutes for the test corpus

### Step 8: Verify Results

Check that these files were created:

```bash
results/YOUR_DB_experiment_001/
├── config.json          # Configuration used
├── results.json         # Experimental data
└── latency_vs_topk.png  # Publication-ready plot
```

**Verification checklist**:
- [ ] `results.json` contains data for all 5 top-k values (1, 3, 5, 10, 20)
- [ ] Each top-k has 10 successful queries
- [ ] `latency_vs_topk.png` shows two plots (latency and QPS vs top-k)
- [ ] Results are reasonable (latency: 5-500ms, QPS: 2-200)

### Step 9: Compare with Qdrant Baseline

Expected ballpark results (will vary by machine):
- **Ingestion**: 0.3-1.0 seconds for 20 documents
- **Avg Query Latency**: 5-50ms (depending on database)
- **Throughput**: 20-200 QPS

If your results differ significantly, investigate:
- Database configuration (indexes, caching)
- Network latency (Docker vs. local)
- Resource constraints (CPU, memory)

## Common Issues

### Database Connection Failed

**Error**: `Failed to connect to [DATABASE]`

**Solution**:
1. Check database is running: `docker-compose ps`
2. Restart if needed: `docker-compose restart YOUR_DB`
3. Check connection settings in CONFIG match docker-compose.yml

### No Results Generated

**Error**: `❌ No successful queries!`

**Solution**:
1. Check collection/index was created successfully
2. Verify documents were ingested (check "Chunks created" message)
3. Enable debug tracing by uncommenting `traceback.print_exc()` in query loop

### Import Error

**Error**: `ModuleNotFoundError: No module named 'src.vector_dbs.YOUR_DB_adapter'`

**Solution**:
1. Check adapter file exists: `ls src/vector_dbs/YOUR_DB_adapter.py`
2. Verify you're running from project root
3. Check adapter class name matches import

## What Makes a Good Benchmark Implementation?

✅ **Do**:
- Use the same test corpus and test cases as Qdrant
- Keep configuration parameters identical (chunk size, top-k values, etc.)
- Export results in the same JSON format
- Generate plots with same style and DPI (300)
- Add clear error messages
- Document any database-specific quirks

❌ **Don't**:
- Change the test data
- Modify chunk sizes or top-k values (for comparability)
- Skip error handling
- Use different embedding models

## Next Steps After Implementation

1. **Test Consistency**: Run your benchmark 3 times, verify results are within 10% variance
2. **Document**: Add a brief comment at the top of your script explaining any database-specific setup
3. **Share**: Create a pull request or share your results file
4. **Compare**: Compare your results with Qdrant baseline to understand performance differences

## Example Implementations

All 7 vector databases are now implemented and verified!

- ✅ **FAISS**: `Scripts/run_faiss_benchmark.py` - Embedded, in-memory (L2 distance fix applied)
- ✅ **Chroma**: `Scripts/run_chroma_benchmark.py` - Embedded/server mode
- ✅ **Qdrant**: `Scripts/run_qdrant_benchmark.py` - Client-server (reference implementation)
- ✅ **pgvector**: `Scripts/run_pgvector_benchmark.py` - PostgreSQL extension
- ✅ **Weaviate**: `Scripts/run_weaviate_benchmark.py` - GraphQL API
- ✅ **Milvus**: `Scripts/run_milvus_benchmark.py` - Distributed architecture
- ✅ **OpenSearch**: `Scripts/run_opensearch_benchmark.py` - k-NN plugin (score fix applied)

**Adapters**: All database adapters are in `src/vector_dbs/*_adapter.py`
**Verification**: See [BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md) for validation details

## Important: Similarity Score Validation

When implementing database adapters, ensure similarity scores are correctly calculated and normalized to [0, 1] range.

### Critical Bug Fixes Applied

**1. FAISS (L2 Distance Conversion)**:
- **Issue**: FAISS `IndexFlatL2` returns raw L2 distances, not similarities
- **Problem**: Caused similarity scores to increase with K (0.536→0.852), which is backwards
- **Fix**: Convert using `similarity = 1.0 / (1.0 + distance)`
- **Result**: Correct decreasing trend (0.656→0.545)
- **Commit**: f192068

**2. OpenSearch (Score Normalization)**:
- **Issue**: OpenSearch returns `_score = 1 / (2 - cosine_similarity)`
- **Problem**: Dividing by 2 gave incorrect scores (0.395 vs 0.732 expected)
- **Fix**: Use formula `cosine_similarity = 2.0 - (1.0 / _score)`
- **Result**: Matches other cosine databases (0.732)
- **Commit**: 0330624

### Expected Similarity Score Ranges

**Cosine Similarity Databases** (Chroma, Qdrant, pgvector, Weaviate, Milvus, OpenSearch):
- Top-1 quality: ~0.732
- K=5 average: ~0.666
- K=20 average: ~0.574
- **Trend**: Decreasing with K ✅

**L2 Distance Databases** (FAISS):
- Top-1 quality: ~0.656 (lower than cosine)
- K=5 average: ~0.605
- K=20 average: ~0.545
- **Trend**: Decreasing with K ✅

### Validation Checklist

When implementing a new database adapter:
- [ ] Similarity scores in [0, 1] range
- [ ] Scores decrease as K increases
- [ ] Top-1 scores match expected ranges above (±0.05)
- [ ] Test with `python test_adapters.py`
- [ ] Run full benchmark and check trend

## Getting Help

If you encounter issues:

1. Check this guide first
2. Review the Qdrant reference implementation
3. Check the database adapter code in `src/vector_dbs/`
4. Look at test_adapters.py for working examples
5. Open an issue with:
   - Database name
   - Error message
   - Your CONFIG dictionary
   - Database status (`docker-compose ps`)

## Time Estimates

- **First database** (learning the pattern): 2-3 hours
- **Subsequent databases** (once familiar): 30-60 minutes
- **Debugging and refinement**: +1-2 hours

## Success Criteria

Your benchmark is complete when:

- [ ] Script runs without errors
- [ ] All 10 test cases complete for all 5 top-k values (50 total queries)
- [ ] `results/YOUR_DB_experiment_001/results.json` created
- [ ] `results/YOUR_DB_experiment_001/latency_vs_topk.png` generated at 300 DPI
- [ ] Results are within expected ranges
- [ ] You've run it 2-3 times with consistent results
- [ ] Results can be used for manuscript figures immediately

---

**Questions?** Open an issue or check `IMPLEMENTATION_PLAN.md` for the overall project roadmap.

**Ready to start?** Pick a database from the list above and follow Steps 1-9!

# Next Steps: Phase 2 Quick Start Guide

**Status:** Phase 1 Complete ✅ | Phase 2 Ready to Begin 🚀

---

## What You Just Accomplished (Phase 1)

✅ **Ported ~70% of production-ready code** from vector-db-benchmark
✅ **Created unified framework** with abstract base classes
✅ **Eliminated code duplication** - single interface for all databases
✅ **Set up proper project structure** - hierarchical imports, modules
✅ **Copied all infrastructure** - Docker, configs, requirements

**Time Saved:** ~42 hours from original 111-hour plan

---

## Phase 2 Overview

**Goal:** Complete all 7 database adapters
**Duration:** 3-4 weeks (40-50 hours)
**Priority:** Critical path for benchmarking study

### Databases to Implement

| Database | Status | Effort | Priority |
|----------|--------|--------|----------|
| pgvector | ⏳ Port | 3h | P0 |
| Qdrant | ✅ Done | - | - |
| Weaviate | ⏳ Port | 3h | P0 |
| Milvus | ⏳ New | 8h | P0 |
| Chroma | ⏳ New | 6h | P0 |
| FAISS | ⏳ New | 6h | P1 |
| OpenSearch | ⏳ New | 8h | P1 |
| Pinecone | ⏳ Optional | 6h | P2 |

---

## Your First Task: Port pgvector (2-3 hours)

### Step 1: Copy the Source File

```bash
cd /Users/rezarassool/Source/vector_dbs_benchmarking

# Copy pgvector adapter
cp /Users/rezarassool/Source/vector-db-benchmark/benchmarks/pgvector_rag_benchmark.py \
   src/vector_dbs/pgvector_adapter.py
```

### Step 2: Update Imports

Open `src/vector_dbs/pgvector_adapter.py` and replace:

```python
# OLD
from rag_benchmark import RAGBenchmark
from utils import Chunk

# NEW
from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk
```

### Step 3: Test the Connection

```python
# test_pgvector.py
from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark
from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import Document

# Start PostgreSQL with pgvector
# docker-compose up -d pgvector

config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres',
    'table_name': 'test'
}

embedding_gen = get_embedding_generator('random', dimension=384)

benchmark = PgvectorRAGBenchmark(
    db_config=config,
    embedding_generator=embedding_gen,
    chunk_size=256,
    chunk_strategy='sentence'
)

# Test connection
benchmark.connect()
benchmark.create_collection(384)
benchmark.disconnect()

print("✅ pgvector adapter working!")
```

### Step 4: Register with Runner

Add to `src/benchmark_runner.py`:

```python
from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark

class BenchmarkRunner:
    def __init__(self, config_path: str = "configs/default.yaml"):
        # ... existing code ...

        # Register adapters
        self.register_benchmark('pgvector', PgvectorRAGBenchmark)
        self.register_benchmark('qdrant', QdrantRAGBenchmark)
```

---

## Running Your First Multi-Database Benchmark

### 1. Start All Services

```bash
cd /Users/rezarassool/Source/vector_dbs_benchmarking
docker-compose up -d
```

### 2. Check Services are Running

```bash
# pgvector (PostgreSQL)
docker ps | grep pgvector

# Qdrant
curl http://localhost:6333/collections

# Check all services
docker-compose ps
```

### 3. Run Benchmark

```bash
python -m src.benchmark_runner --config configs/default.yaml
```

This will run all scenarios defined in `configs/default.yaml` and output:
- Individual JSON results per database
- Combined JSON results
- CSV for spreadsheet analysis
- Console comparison table

---

## Recommended Work Order

### Week 1 (12 hours)
**Goal:** Port existing adapters and validate framework

1. **pgvector** (3h)
   - Copy from adjacent project
   - Update imports
   - Test connection and queries
   - Verify HNSW index creation

2. **Weaviate** (3h)
   - Copy from adjacent project
   - Update imports
   - Test schema creation
   - Verify vector search

3. **Integration Testing** (4h)
   - Create `tests/integration/test_adapters.py`
   - Write tests for pgvector and Qdrant
   - Verify both databases in one benchmark run
   - Document any issues

4. **Documentation** (2h)
   - Write `docs/setup/pgvector_setup.md`
   - Write `docs/setup/weaviate_setup.md`
   - Update README with completed databases

### Week 2 (16 hours)
**Goal:** Implement Milvus and Chroma

1. **Milvus Adapter** (8h)
   - Study Milvus Python SDK documentation
   - Use Qdrant adapter as template
   - Implement all abstract methods
   - Set up Docker Compose with etcd/minio dependencies
   - Test with sample data

2. **Chroma Adapter** (6h)
   - Study Chroma documentation
   - Implement embedded mode first
   - Add client-server mode support
   - Test persistence

3. **Testing & Documentation** (2h)
   - Integration tests for both
   - Setup documentation

### Week 3 (16 hours)
**Goal:** Implement FAISS and OpenSearch

1. **FAISS Adapter** (6h)
   - Study FAISS index types
   - Implement Flat, IVF, HNSW support
   - Handle metadata storage separately
   - Performance testing

2. **OpenSearch Adapter** (8h)
   - Set up OpenSearch with k-NN plugin
   - Implement HNSW index creation
   - Test bulk operations
   - Query optimization

3. **Testing & Documentation** (2h)

### Week 4 (8-12 hours)
**Goal:** Final polish and comprehensive testing

1. **End-to-End Testing** (4h)
   - Run full benchmark suite across all databases
   - Verify results consistency
   - Performance comparison
   - Bug fixes

2. **Documentation** (4h)
   - Complete all setup guides
   - Create comparison table
   - Write troubleshooting guide
   - Update main README

3. **Optional: Pinecone** (4-6h)
   - If time permits and API key available

---

## Files You'll Create

```
src/vector_dbs/
├── pgvector_adapter.py      ← Week 1
├── weaviate_adapter.py      ← Week 1
├── milvus_adapter.py        ← Week 2
├── chroma_adapter.py        ← Week 2
├── faiss_adapter.py         ← Week 3
├── opensearch_adapter.py    ← Week 3
└── pinecone_adapter.py      ← Optional

tests/integration/
└── test_all_adapters.py     ← Progressive

docs/setup/
├── pgvector_setup.md        ← Week 1
├── weaviate_setup.md        ← Week 1
├── milvus_setup.md          ← Week 2
├── chroma_setup.md          ← Week 2
├── faiss_setup.md           ← Week 3
└── opensearch_setup.md      ← Week 3
```

---

## Common Pitfalls to Avoid

### 1. Import Errors
❌ **Wrong:** `from rag_benchmark import RAGBenchmark`
✅ **Right:** `from src.vector_dbs.rag_benchmark import RAGBenchmark`

### 2. Missing Dependencies
Always update `requirements.txt` when adding new database clients:
```bash
# After implementing Milvus
echo "pymilvus>=2.3.0" >> requirements.txt
pip install -r requirements.txt
```

### 3. Docker Compose Issues
Check service health before testing:
```bash
# Wait for services to be ready
docker-compose up -d
sleep 10
docker-compose ps
```

### 4. Connection Timeout
Add retry logic in adapters:
```python
def connect(self) -> None:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # connection code
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
```

---

## Testing Strategy

### Unit Tests (Per Adapter)
Test individual methods in isolation:
```python
def test_pgvector_connect():
    adapter = PgvectorRAGBenchmark(config, embedding_gen)
    adapter.connect()
    assert adapter.connection is not None
    adapter.disconnect()
```

### Integration Tests (End-to-End)
Test full workflow:
```python
def test_pgvector_full_benchmark():
    results = benchmark.run_full_benchmark(docs, queries)
    assert results.num_documents == len(docs)
    assert results.avg_query_latency > 0
```

### Comparison Tests (Cross-Database)
Verify consistency:
```python
def test_all_databases_same_results():
    pgvector_results = run_benchmark('pgvector', docs, queries)
    qdrant_results = run_benchmark('qdrant', docs, queries)

    # Should return same chunks for same queries
    assert compare_results(pgvector_results, qdrant_results) > 0.95
```

---

## Success Metrics

After Phase 2, you should be able to:

✅ Run: `python -m src.benchmark_runner --config configs/default.yaml`
✅ See: Benchmarks execute across 7 databases automatically
✅ Get: JSON, CSV, and comparison table outputs
✅ Compare: Performance metrics across all databases
✅ Reproduce: Results with same configuration

---

## Getting Help

### Reference Documentation
- **Full Phase 2 Plan:** `PHASE_2_PLAN.md`
- **Porting Summary:** `PORTING_SUMMARY.md`
- **Implementation Plan:** `IMPLEMENTATION_PLAN.md`
- **Requirements:** `REQUIREMENTS.md`

### Templates to Use
- **Qdrant Adapter:** `src/vector_dbs/qdrant_adapter.py` (best reference)
- **RAG Benchmark Base:** `src/vector_dbs/rag_benchmark.py`
- **Config Template:** `configs/default.yaml`

### Test Your Work
```bash
# Verify imports
python verify_imports.py

# Run specific adapter test
pytest tests/integration/test_adapters.py::TestPgvectorAdapter -v

# Run all tests
pytest tests/ -v --cov=src
```

---

## Questions to Ask Yourself

Before starting each adapter:

1. **What client library does this database use?**
   - Add to requirements.txt
   - Check version compatibility

2. **What Docker image should I use?**
   - Update docker-compose.yml
   - Verify environment variables

3. **What index types does it support?**
   - HNSW, IVF, Flat, etc.
   - Make configurable in YAML

4. **How does it handle metadata?**
   - Payload, fields, properties
   - Map to our Chunk structure

5. **What are the connection parameters?**
   - Host, port, credentials
   - Add to configs/default.yaml

6. **How does it handle batch operations?**
   - Batch size limits
   - Optimal batch size for performance

---

## Phase 2 Completion Criteria

You're done when:

- [ ] All 7 adapters implement `RAGBenchmark` interface
- [ ] All adapters pass integration tests
- [ ] Docker Compose starts all services successfully
- [ ] Benchmark runner executes multi-database scenarios
- [ ] Results export to JSON, CSV, comparison table
- [ ] Documentation exists for all databases
- [ ] Code coverage >80%
- [ ] No critical bugs

**Then:** Create `PHASE_3_PLAN.md` for advanced metrics and features!

---

**Ready to start?** Begin with pgvector porting (3 hours) 🚀

**Questions?** Review `PHASE_2_PLAN.md` for detailed guidance on each database

**Stuck?** Reference `src/vector_dbs/qdrant_adapter.py` as the gold standard template

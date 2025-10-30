# Vector Database Benchmarking System

**Status**: Phase 1 Complete - Framework Ready for Dataset Integration ✅

A benchmarking framework for comparing vector databases in RAG (Retrieval-Augmented Generation) applications. Currently features complete end-to-end benchmarks for Qdrant with a working framework. **Contributors: Please use the curated test dataset from Google Drive** (see TODO section below).

**⚠️ IMPORTANT**: The current Climate Science dataset is being replaced with a curated test dataset available in Google Drive. All new benchmarks should use the new dataset.

## Current Capabilities

### ✅ Working Benchmarks

**Query Performance Benchmark**
- Latency measurement across different top-K values (1, 3, 5, 10, 20)
- Throughput tracking (queries per second)
- Automated quality metrics using semantic similarity
- 4-panel visualization (speed, throughput, quality, tradeoffs)

**Ingestion Performance Benchmark**
- Chunk size variations (256, 512, 1024 characters)
- Batch size testing (50, 100 documents/batch)
- Document scaling analysis (10, 20 documents)
- 3 visualizations: performance, scaling, heatmap

### 📊 Latest Results (Qdrant)

```
Top-K    Avg Latency    P95         QPS        Avg Similarity    Top-1 Similarity
1        29.74 ms       71.85 ms    33.62      0.732            0.732
3        7.94 ms        10.12 ms    125.90     0.688            0.732
5        7.94 ms        10.13 ms    125.88     0.666            0.732
10       8.66 ms        10.41 ms    115.43     0.629            0.732
20       11.19 ms       15.71 ms    89.36      0.574            0.732
```

**Key Insights**: Best throughput at K=3-5 (~126 QPS), consistent top-1 quality (0.732 similarity)

## Quick Start

### 1. Setup Environment

```bash
# Run automated setup
./setup.sh
source venv/bin/activate
```

### 2. Start Qdrant

```bash
docker-compose up -d qdrant
```

### 3. Run Benchmarks

```bash
# Query performance benchmark (~30 seconds)
python Scripts/run_qdrant_benchmark.py

# Ingestion performance benchmark (~1 minute)
python Scripts/run_qdrant_ingestion_benchmark.py
```

### 4. View Results

Results are saved to `results/` directory:
- JSON data files with detailed metrics
- High-resolution PNG visualizations (300 DPI)
- CSV exports for analysis

## Project Structure

```
Scripts/
├── run_qdrant_benchmark.py           # Query latency & quality
└── run_qdrant_ingestion_benchmark.py # Ingestion performance

src/
├── vector_dbs/
│   ├── qdrant_adapter.py             # Qdrant implementation
│   └── rag_benchmark.py              # Base interface
├── embeddings/
│   └── embedding_generator.py        # Sentence-transformers
└── utils/
    ├── chunking.py                   # Document chunking
    └── document_parser.py            # Document parsing

Data/test_corpus/
├── documents/                        # ⚠️ Legacy: 20 climate science docs (being replaced)
└── test_cases.json                   # ⚠️ To be replaced with curated dataset from Google Drive

results/
├── qdrant_experiment_*/              # Query benchmark results
└── qdrant_ingestion_experiment_*/    # Ingestion benchmark results
```

## Technical Details

**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Test Corpus**: ⚠️ Curated dataset in Google Drive (to be integrated - see TODO)
**Legacy Corpus**: 20 climate science documents (~78KB) - being replaced
**Test Queries**: Ground truth queries provided with new dataset
**Database**: Qdrant via Docker (localhost:6333)
**Quality Metric**: Cosine similarity (automated, no manual labeling)

## Documentation

### Core Documentation
- **[PROJECT_STATE.md](PROJECT_STATE.md)** - 📍 Current status, results, and resume points
- **[CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md)** - How to add new databases
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Project roadmap (v2.0)

### Legacy Documentation
- **[PHASE_2_VERIFICATION.md](PHASE_2_VERIFICATION.md)** - Earlier verification report
- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker Compose setup for 7 databases

## TODO - Immediate Action Items

### 🔴 Critical Priority

1. **Integrate Curated Test Dataset from Google Drive**
   - Download the curated dataset from Google Drive
   - Replace the current Climate Science dataset in `Data/test_corpus/`
   - Update data loading scripts to use new dataset format
   - Verify test cases and ground truth queries work with new data
   - Update all documentation referencing the old dataset
   - **Status**: Not started
   - **Blocking**: All new database benchmarks should wait for this

2. **Re-run Qdrant Benchmarks with New Dataset**
   - Execute query benchmark with new data
   - Execute ingestion benchmark with new data
   - Update baseline results in documentation
   - **Status**: Blocked by #1

### 🟡 High Priority

3. **Add 2-3 More Databases** (after dataset integration)
   - Recommended: ChromaDB and FAISS (simplest to start)
   - Follow CONTRIBUTOR_GUIDE.md process
   - Use new curated dataset only

4. **Update Test Infrastructure**
   - Ensure test_adapters.py works with new dataset
   - Add validation scripts for new data format

## Roadmap

### ✅ Completed (Phase 1)
- Working Qdrant query benchmark with quality metrics
- Working Qdrant ingestion benchmark
- Initial test corpus and ground truth queries (Climate Science - being replaced)
- Automated quality measurement (semantic similarity)
- Publication-ready visualizations

### 🚀 Next Steps (Contributors Welcome!)

**Immediate Extensions**:
1. Add remaining databases (Chroma, FAISS, pgvector, Weaviate, Milvus, OpenSearch)
2. Expand ingestion experiments (more chunk/batch sizes, document counts)
3. Add Precision@K, Recall@K metrics using existing ground truth

**Advanced Extensions**:
4. Enhanced quality metrics (NDCG, MRR)
5. LLM-as-judge for answer quality evaluation
6. Cross-database comparison analysis

**Research-Grade Extensions**:
7. Statistical significance testing
8. Concurrent query testing
9. Large-scale corpus (10K+ documents)
10. Production scenario simulation

## Contributing

### 🚨 Before You Start

**IMPORTANT FOR ALL CONTRIBUTORS**:
1. **Use the curated test dataset from Google Drive** - DO NOT use the Climate Science dataset
2. **Check the TODO section above** - Dataset integration is the critical first step
3. **Do not add new databases until the dataset is integrated** - This ensures consistency

### Contributor Workflow

**For Dataset Integration** (Priority #1):
1. Contact project maintainers for Google Drive access
2. Download and review the curated dataset structure
3. Update `Data/test_corpus/` directory structure
4. Modify data loading functions in benchmark scripts
5. Test with existing Qdrant benchmarks
6. Update documentation

**For Adding New Databases** (Priority #2 - after dataset integration):
1. Review [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for complete instructions
2. Copy `src/vector_dbs/qdrant_adapter.py` as template
3. Implement 5 required methods: `connect`, `disconnect`, `create_collection`, `insert_chunks`, `query`
4. Test with the **new curated dataset only**
5. Create benchmark script (copy from `Scripts/run_qdrant_benchmark.py`)
6. Run benchmarks and generate visualizations
7. Document results

**For Quality Metrics** (Priority #3):
- Extending quality metrics (Precision@K, Recall@K, NDCG, MRR)
- Expanding test configurations
- Adding statistical analysis

See [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for detailed step-by-step instructions.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Qdrant | ✅ Complete | Framework working - needs re-run with new dataset |
| Chroma | ⏸️ Paused | Adapter implemented - awaiting dataset integration |
| FAISS | ⏸️ Paused | Adapter implemented - awaiting dataset integration |
| pgvector | ⏸️ Paused | Adapter implemented - awaiting dataset integration |
| Weaviate | ⏸️ Paused | Adapter implemented - awaiting dataset integration |
| Milvus | ⏸️ Paused | Adapter implemented - awaiting dataset integration |
| OpenSearch | ⏸️ Paused | Adapter implemented - awaiting dataset integration |

**Note**: All adapters have been implemented but not fully tested. Testing and benchmarking are paused pending integration of the curated test dataset from Google Drive.

## Design Philosophy

**Speed to Data**: Minimal configurations for rapid iteration (<1 minute runtime)
**Automated Quality**: No manual labeling required (semantic similarity)
**Extensible Framework**: Easy to add databases, metrics, configurations
**Production Ready**: Error handling, logging, clean code
**Publication Ready**: High-DPI plots, JSON exports, reproducible results

---

## Getting Started

**For New Contributors**:
1. ⚠️ **First**: Check the TODO section above - dataset integration is priority #1
2. Review [PROJECT_STATE.md](PROJECT_STATE.md) for detailed technical status
3. Read [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for implementation guidance
4. Contact maintainers for Google Drive dataset access

**For Dataset Integration**: This is the critical blocker. The curated test dataset needs to be downloaded from Google Drive and integrated into the framework. All benchmark results from the Climate Science dataset will be deprecated once this is complete.

**Questions?** Open an issue or contact the project maintainers.

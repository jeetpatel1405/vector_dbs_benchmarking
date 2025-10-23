# Vector Database Benchmarking System

**Status**: Phase 1 Complete - Working Qdrant Benchmarks with Quality Metrics ✅

A benchmarking framework for comparing vector databases in RAG (Retrieval-Augmented Generation) applications. Currently features complete end-to-end benchmarks for Qdrant, with extensible framework for adding other databases.

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
├── documents/                        # 20 climate science docs
└── test_cases.json                   # 10 test queries with ground truth

results/
├── qdrant_experiment_*/              # Query benchmark results
└── qdrant_ingestion_experiment_*/    # Ingestion benchmark results
```

## Technical Details

**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Test Corpus**: 20 climate science documents (~78KB)
**Test Queries**: 10 queries with ground truth answers
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

## Roadmap

### ✅ Completed (Phase 1)
- Working Qdrant query benchmark with quality metrics
- Working Qdrant ingestion benchmark
- Test corpus and ground truth queries
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

See [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for detailed instructions on:
- Adding a new vector database
- Extending quality metrics
- Expanding test configurations
- Submitting contributions

**Quick template**: Copy `src/vector_dbs/qdrant_adapter.py` and implement 5 methods: `connect`, `disconnect`, `create_collection`, `insert_chunks`, `query`

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Qdrant | ✅ Complete | Both query & ingestion benchmarks working |
| Chroma | 🔄 Planned | Docker config ready, adapter needed |
| FAISS | 🔄 Planned | Docker config ready, adapter needed |
| pgvector | 🔄 Planned | Docker config ready, adapter needed |
| Weaviate | 🔄 Planned | Docker config ready, adapter needed |
| Milvus | 🔄 Planned | Docker config ready, adapter needed |
| OpenSearch | 🔄 Planned | Docker config ready, adapter needed |

## Design Philosophy

**Speed to Data**: Minimal configurations for rapid iteration (<1 minute runtime)
**Automated Quality**: No manual labeling required (semantic similarity)
**Extensible Framework**: Easy to add databases, metrics, configurations
**Production Ready**: Error handling, logging, clean code
**Publication Ready**: High-DPI plots, JSON exports, reproducible results

---

**Ready to start?** See [PROJECT_STATE.md](PROJECT_STATE.md) for detailed status and next steps.

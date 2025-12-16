# Vector Database Benchmarking System

**Status**: Phase 2 Complete - All 7 Databases Benchmarked ✅

A comprehensive benchmarking framework for comparing vector databases in RAG (Retrieval-Augmented Generation) applications. Features complete end-to-end benchmarks for 7 vector databases with automated quality metrics and publication-ready visualizations.

**Latest Update (Nov 2025)**: Fixed critical similarity calculation bugs in FAISS and OpenSearch adapters. All benchmarks verified and results are scientifically valid. See [BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md) for details.

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

### 📊 Latest Results (All 7 Databases @ K=5)

| Database   | Latency | Throughput | Top-1 Quality | Avg Quality | Trend |
|------------|---------|------------|---------------|-------------|-------|
| **FAISS**      | 3.96 ms | **252.6 QPS** 🏆 | 0.656 | 0.605 | ✅ Decreasing |
| **Chroma**     | 4.53 ms | 220.6 QPS | **0.732** 🏆 | 0.666 | ✅ Decreasing |
| **pgvector**   | 7.54 ms | 132.6 QPS | 0.732 | 0.666 | ✅ Decreasing |
| **Qdrant**     | 7.87 ms | 127.1 QPS | 0.732 | 0.666 | ✅ Decreasing |
| **Weaviate**   | 9.83 ms | 101.7 QPS | 0.732 | 0.666 | ✅ Decreasing |
| **Milvus**     | 10.31 ms | 97.0 QPS | 0.732 | 0.666 | ✅ Decreasing |
| **OpenSearch** | 12.35 ms | 81.0 QPS | 0.732 | 0.666 | ✅ Decreasing |

**Key Insights**:
- **Fastest**: FAISS (252.6 QPS, in-memory, no network overhead)
- **Best Quality**: 6-way tie at 0.732 (all cosine similarity databases)
- **All databases show correct decreasing similarity trends** (higher K = less relevant results)
- Results verified and publication-ready (see [BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md))

## 🆕 Latest Features (Nov 2025)

### Automated Result Management
- **Dated Result Folders**: Results automatically organized by date (`*_results_YYYYMMDD/`)
- **Auto-commit Scripts**: Automated git commits for benchmark results
- **Comprehensive Comparison Plots**: 7+ visualization types for cross-database analysis
- **Publication-Ready Plots**: All plots at 300 DPI, optimized layouts

### Available Comparison Plots
Located in `results/full_suite_YYYYMMDD_plots/`:
1. **Query Performance Comparison** (4-panel):
   - Query latency by Top-K
   - Throughput by Top-K
   - Quality metrics (similarity scores)
   - Performance summary with data table
2. **Ingestion Performance Comparison** (4-panel):
   - Ingestion time by chunk size
   - Throughput by chunk size
   - Phase breakdown (parsing/embedding/insertion)
   - Normalized throughput
3. **Detailed Analysis**:
   - Throughput comparison
   - Phase breakdown
   - Scaling analysis
   - Summary dashboard

## Quick Start

### 1. Setup Environment

```bash
# Run automated setup
./setup.sh
source venv/bin/activate
```

### 2. Start Databases

```bash
# Start individual database (recommended - avoids resource contention)
docker-compose up -d qdrant

# Or start all databases (requires significant resources)
docker-compose up -d

# List of available databases:
# qdrant, chroma, pgvector, weaviate, milvus, opensearch
# Note: FAISS and Chroma also work as embedded libraries (no Docker needed)
```

### 3. Run Benchmarks

```bash
# Query performance benchmarks (~30-60 seconds each)
python Scripts/run_faiss_benchmark.py      # No Docker needed
python Scripts/run_chroma_benchmark.py     # No Docker needed
python Scripts/run_qdrant_benchmark.py
python Scripts/run_pgvector_benchmark.py
python Scripts/run_weaviate_benchmark.py
python Scripts/run_milvus_benchmark.py
python Scripts/run_opensearch_benchmark.py

# Ingestion performance benchmarks (~1-2 minutes each)
python Scripts/run_qdrant_ingestion_benchmark.py
# (Other databases have corresponding ingestion scripts)

# Generate cross-database comparisons
python Scripts/generate_comparison_plots.py          # Ingestion comparison
python Scripts/recreate_query_comparison.py          # Query comparison

# Automated result management
./Scripts/commit_benchmark_results.sh weaviate results/weaviate_experiment_001
```

### 4. View Results

Results are saved to `results/` directory with automated organization:

**Dated Result Folders** (tracked in git):
```
results/
├── weaviate_results_20251125/           # Database-specific results
│   ├── results.json                     # Detailed metrics
│   ├── performance_quality.png          # Visualizations
│   └── config.json                      # Configuration
├── full_suite_20251124_plots/           # Cross-database comparisons
│   ├── all_databases_comparison_query.png    # 4-panel query comparison
│   ├── all_databases_comparison.png          # 4-panel ingestion comparison
│   ├── ingestion_time_comparison.png         # Time comparison
│   ├── throughput_comparison.png             # Throughput comparison
│   ├── phase_breakdown_comparison.png        # Phase analysis
│   ├── scaling_comparison.png                # Scaling analysis
│   └── summary_dashboard.png                 # Complete overview
└── comparison_plots_20251125/           # Historical comparison plots
```

**Automated Features**:
- Results automatically copied to dated folders
- Git commits with standardized messages
- High-resolution PNG visualizations (300 DPI)
- JSON/CSV exports for analysis

## Project Structure

```
Scripts/
├── run_faiss_benchmark.py                  # FAISS (embedded, in-memory)
├── run_chroma_benchmark.py                 # Chroma (embedded/server)
├── run_qdrant_benchmark.py                 # Qdrant (client-server)
├── run_pgvector_benchmark.py               # PostgreSQL + pgvector
├── run_weaviate_benchmark.py               # Weaviate (GraphQL)
├── run_milvus_benchmark.py                 # Milvus (distributed)
├── run_opensearch_benchmark.py             # OpenSearch (k-NN plugin)
├── run_*_ingestion_benchmark.py            # Ingestion benchmarks
├── create_comparison.py                    # Legacy comparison
├── generate_comparison_plots.py            # 🆕 Ingestion comparison plots
├── recreate_query_comparison.py            # 🆕 Query comparison plots
├── generate_nov24_comparison.py            # 🆕 Dated comparison plots
├── commit_benchmark_results.sh             # 🆕 Auto-commit results
└── monitor_and_commit_weaviate.sh          # 🆕 Auto-monitor & commit

src/
├── vector_dbs/
│   ├── base_benchmark.py             # Abstract base class
│   ├── rag_benchmark.py              # RAG-specific base
│   ├── faiss_adapter.py              # FAISS implementation
│   ├── chroma_adapter.py             # Chroma implementation
│   ├── qdrant_adapter.py             # Qdrant implementation
│   ├── pgvector_adapter.py           # pgvector implementation
│   ├── weaviate_adapter.py           # Weaviate implementation
│   ├── milvus_adapter.py             # Milvus implementation
│   └── opensearch_adapter.py         # OpenSearch implementation
├── embeddings/
│   └── embedding_generator.py        # Sentence-transformers, OpenAI
├── parsers/
│   └── document_parser.py            # TXT, PDF, DOCX parsing
├── monitoring/
│   └── resource_monitor.py           # CPU, memory tracking
└── utils/
    └── chunking.py                   # Fixed, sentence, paragraph strategies

Data/test_corpus/
├── documents/                        # 20 climate science documents
└── test_cases.json                   # 10 test queries with ground truth

results/
├── *_experiment_001/                      # Working directories (gitignored)
│   ├── config.json
│   ├── results.json
│   └── *.png
├── *_results_YYYYMMDD/                    # 🆕 Dated results (tracked in git)
│   ├── config.json                        # Experiment configuration
│   ├── results.json                       # Detailed metrics
│   └── *.png                              # All plots and visualizations
├── full_suite_YYYYMMDD_HHMMSS/            # 🆕 Complete suite runs
│   ├── benchmark_results.csv              # Summary data
│   └── *.log                              # Execution logs
└── full_suite_YYYYMMDD_plots/             # 🆕 Cross-database comparisons
    ├── all_databases_comparison_query.png # Query performance (4-panel)
    ├── all_databases_comparison.png       # Ingestion performance (4-panel)
    └── *.png                              # Additional comparison plots
```

## Technical Details

**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Test Corpus**: 20 climate science documents (~78KB, 175 chunks)
**Test Queries**: 10 queries with ground truth for quality validation
**Chunk Strategy**: Fixed-size (512 chars, 50 overlap)
**Top-K Values**: [1, 3, 5, 10, 20]
**Quality Metrics**: Cosine similarity (automated, no manual labeling)
**Databases**: 7 vector databases via Docker + 2 embedded libraries (FAISS, Chroma)
**Verification**: All similarity calculations validated (see [BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md))

## Documentation

### Core Documentation
- **[BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md)** - ✅ Verification report for similarity fixes
- **[PROJECT_STATE.md](PROJECT_STATE.md)** - Current status and technical details
- **[CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md)** - How to add new databases
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md)** - Phase 2 implementation summary

### Additional Documentation
- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker Compose setup for 7 databases
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Original project roadmap

## Roadmap

### ✅ Completed (Phase 1 & 2)
- ✅ Complete framework with abstract base classes
- ✅ All 7 database adapters implemented and tested
- ✅ Query performance benchmarks (latency, throughput, quality)
- ✅ Ingestion performance benchmarks
- ✅ Automated quality measurement (semantic similarity)
- ✅ Publication-ready visualizations (4-panel + comparison plots)
- ✅ Fixed critical similarity calculation bugs (FAISS, OpenSearch)
- ✅ Comprehensive verification and validation
- ✅ Test corpus: 20 climate science documents, 10 test queries

### 🚀 Phase 3 - Advanced Features (Contributors Welcome!)

**Quality Metrics Enhancements**:
1. Add Precision@K, Recall@K using existing ground truth
2. Implement NDCG (Normalized Discounted Cumulative Gain)
3. Add MRR (Mean Reciprocal Rank)
4. LLM-as-judge for end-to-end answer quality

**Performance Testing**:
5. Concurrent query testing (multi-threaded benchmarks)
6. Memory profiling (RAM usage during ingestion/queries)
7. Statistical significance testing (confidence intervals, p-values)
8. Cost analysis (compute costs for cloud deployments)

**Scale Testing**:
9. Expand test corpus to 100-1000 documents
10. Test with multiple embedding models (different dimensions)
11. Production scenario simulation (realistic query patterns)
12. Large-scale experiments (10K+ documents)

## Contributing

All 7 major vector databases are now implemented and benchmarked! Contributions are welcome for Phase 3 enhancements:

### Current Opportunities

**1. Quality Metrics** (Good First Issue):
- Implement Precision@K, Recall@K (ground truth available)
- Add NDCG, MRR calculations
- Create quality metric comparison plots

**2. Performance Testing**:
- Add concurrent query benchmarks (multi-threading)
- Implement memory profiling
- Add statistical significance tests

**3. Scale Testing**:
- Expand test corpus (100-1000 documents)
- Test multiple embedding models
- Add production scenario simulations

**4. Database Extensions**:
- Add more vector databases (Pinecone, Vespa, etc.)
- Implement hybrid search benchmarks
- Add filtered search scenarios

### How to Contribute

1. **Review** [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for implementation guidelines
2. **Pick** an enhancement from the Phase 3 roadmap above
3. **Implement** following the established patterns
4. **Test** thoroughly with existing benchmarks
5. **Document** your changes and results
6. **Submit** a pull request with clear description

See [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for detailed instructions on:
- Adding new databases
- Implementing quality metrics
- Extending benchmarks
- Code style and testing requirements

## Supported Databases

| Database   | Type | Status | Latency @K=5 | Throughput | Quality |
|------------|------|--------|--------------|------------|---------|
| **FAISS**      | Embedded | ✅ Complete | 3.96 ms | 252.6 QPS | 0.656 |
| **Chroma**     | Embedded/Server | ✅ Complete | 4.53 ms | 220.6 QPS | 0.732 |
| **pgvector**   | Client-Server | ✅ Complete | 7.54 ms | 132.6 QPS | 0.732 |
| **Qdrant**     | Client-Server | ✅ Complete | 7.87 ms | 127.1 QPS | 0.732 |
| **Weaviate**   | Client-Server | ✅ Complete | 9.83 ms | 101.7 QPS | 0.732 |
| **Milvus**     | Client-Server | ✅ Complete | 10.31 ms | 97.0 QPS | 0.732 |
| **OpenSearch** | Client-Server | ✅ Complete | 12.35 ms | 81.0 QPS | 0.732 |

**Notes**:
- All similarity calculations verified and scientifically valid
- Results benchmarked sequentially to avoid resource contention
- Quality scores: FAISS uses L2 distance (0.656), others use cosine similarity (0.732)
- See [BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md) for detailed validation

## Design Philosophy

**Speed to Data**: Minimal configurations for rapid iteration (<1 minute runtime)
**Automated Quality**: No manual labeling required (semantic similarity)
**Extensible Framework**: Easy to add databases, metrics, configurations
**Production Ready**: Error handling, logging, clean code
**Publication Ready**: High-DPI plots, JSON exports, reproducible results

---

## Important Next Steps

### 🔴 Critical Priority - Dataset Integration

**Current Status**: All 7 databases are working and verified with the Climate Science dataset (20 documents). However, a **curated test dataset is available in Google Drive** that should be integrated for more comprehensive benchmarking.

**Action Required**:
1. **Integrate Curated Dataset from Google Drive**
   - Download the curated dataset
   - Replace or supplement `Data/test_corpus/`
   - Update data loading scripts for new format
   - Verify compatibility with all 7 database adapters
   - **Status**: Not started

2. **Re-run All Benchmarks with New Dataset**
   - Execute benchmarks for all 7 databases
   - Generate new comparison plots
   - Update documentation with new results
   - **Status**: Blocked by #1

**Current Results**: The benchmark results shown in this README are valid and scientifically verified using the Climate Science dataset. These provide a solid baseline, but the curated dataset will enable more comprehensive testing.

---

## Getting Started

**For New Users**:
1. Run `./setup.sh` to set up the environment
2. Start with FAISS or Chroma (no Docker required)
3. Try `python Scripts/run_faiss_benchmark.py`
4. View results in `results/faiss_experiment_001/`
5. Review [QUICKSTART.md](QUICKSTART.md) for detailed guide

**For Contributors - Dataset Integration** (Priority #1):
1. Contact project maintainers for Google Drive access
2. Download and review the curated dataset structure
3. Update data loading functions to handle new format
4. Test with existing adapters (all 7 databases)
5. Re-run benchmarks and update documentation

**For Contributors - Phase 3 Enhancements**:
1. Review [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for guidelines
2. Pick an enhancement from the Phase 3 roadmap
3. Implement, test, and document your changes
4. Submit a pull request

**For Researchers**:
- Current benchmark results are publication-ready (validated with Climate Science dataset)
- See [BENCHMARK_VERIFICATION.md](BENCHMARK_VERIFICATION.md) for validation details
- Curated dataset integration will provide additional validation

**Questions?** Open an issue on GitHub.
